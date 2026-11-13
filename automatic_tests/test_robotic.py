"""Acceptance tests for robot mode (docs/robotic-gripper_spec.md section 10, R1-R14).

Standalone runner (pytest is not installed). From repo root, through the env:

    conda run -n environmentAAP python automatic_tests/test_robotic.py                    # core
    conda run -n environmentAAP python automatic_tests/test_robotic.py --groups core freecad engine
    conda run -n environmentAAP python automatic_tests/test_robotic.py --tests R5 R11

Groups:
  core     no FreeCAD, no grasp engine: rule, quarter, choice, loop (real optimizer on synthetic AoGs,
           fake engine through the real GripperAnalysis), rotation, Optimization hook, mode switch, output.
  freecad  point clouds (determinism, mm, counts) and the tool check on a synthetic scene with the real
           tools/gripper/screwing_effector.step.
  engine   the real JMS engine in the `parallelgrippercode` env: worker vs the unchanged script
           (equivalence, same seed), crash -> EngineError, empty obstacles, alone scores on
           industrial_fastner against spec 5.2 (a few minutes).

Results go to automatic_tests/output/robotic_results.json, scratch to automatic_tests/output/robotic_test/.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import optimization as op  # noqa: E402
from gripper_analysis_robotic import (GripperAnalysis, Rotation, feasibility, intrinsically_infeasible,  # noqa: E402
                                      relative_score, surviving_grasps)
from gripper_engine_robotic import EngineResult, Grasp  # noqa: E402
from robot_config_robotic import RobotConfig  # noqa: E402
from robot_loop_robotic import (StepEval, choose, edges_to_prune, quarter_count, rotation_for,  # noqa: E402
                                run_robot_loop)
from test_optimization import all_free_aog, make_tables, only  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
SCRATCH = OUT_DIR / "robotic_test"
FASTENER_STEP = REPO_ROOT / "data" / "industrial_fastner" / "industrial_fastner.stp"
SCREW_TOOL = REPO_ROOT / "tools" / "gripper" / "screwing_effector.step"
SPEC_ALONE = {"industrial_fastner_part1": 0.488, "industrial_fastner_part2": 0.754, "industrial_fastner_part3": 0.937,
              "industrial_fastner_part5": 0.551, "industrial_fastner_part6": 0.498}


class Recorder:
    def __init__(self):
        self.results: dict[str, dict] = {}
        self._current = None

    def start(self, tid):
        self._current = tid
        self.results[tid] = {"passed": True, "checks": [], "notes": []}

    def check(self, ok, label):
        e = self.results[self._current]
        e["checks"].append({"label": label, "passed": bool(ok)})
        if not ok:
            e["passed"] = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}", flush=True)

    def note(self, text):
        self.results[self._current]["notes"].append(text)
        print(f"  note: {text}", flush=True)

    def error(self, exc):
        e = self.results[self._current]
        e["passed"] = False
        e["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"  [ERROR] {e['error']}", flush=True)
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def toy_cloud(c: int, n: int = 10):
    """A 10-point cloud per component, offset so each target is distinguishable; extent > 1 mm."""
    rng = np.random.default_rng(c)
    P = rng.random((n, 3)) * 5 + np.array([100.0 * c, 0, 0])
    return P, np.tile([0.0, 0.0, 1.0], (n, 1))


class FakeEngine:
    """score = base[target] * (1 - 0.15 * number of obstacle components); counts calls."""

    def __init__(self, base, n_per_cloud=10):
        self.base, self.n_per = base, n_per_cloud
        self.calls = 0

    def evaluate(self, target, obstacles):
        self.calls += 1
        c = int(round(target[0][:, 0].mean() / 100.0))
        k = 0 if obstacles is None else len(obstacles[0]) // self.n_per
        s = max(0.0, self.base[c] * (1 - 0.15 * k))
        g = Grasp(tcp=np.zeros(3), axes=np.eye(3), contact_area=1.0, grasp_width=10.0, score=s)
        return EngineResult(grasps=[g] if s > 0 else [])


class RotEngine:
    """Rotation fake: obstacle centroid decides which of three grasps (A .9, B .8, C .7) survive."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, target, obstacles):
        self.calls += 1
        mk = lambda x, s: Grasp(tcp=np.array([x, 0.0, 0.0]), axes=np.eye(3), contact_area=1, grasp_width=5, score=s)
        A, B, C = mk(1, 0.9), mk(2, 0.8), mk(3, 0.7)
        if obstacles is None:
            return EngineResult(grasps=[A, B, C])
        cx, cy = obstacles[0][:, 0].mean(), obstacles[0][:, 1].mean()
        out = [B]
        if cx > -5:
            out.append(A)
        if cy > -5:
            out.append(C)
        return EngineResult(grasps=out)


class NoTools:
    def step(self, state, c, senses, preferred=None):
        return SimpleNamespace(feasible=True, tool=None, checked=False, senses={}, contact_with={}, sense=preferred,
                               to_json=lambda directions=None: {"sense_direction": None, "senses": []})


def fake_step(step, state, comp, kind="gripper", feasible=True, rel=1.0, intrinsic=False):
    g = None
    if kind == "gripper":
        g = SimpleNamespace(score_relative=None if intrinsic else rel, intrinsic=intrinsic, reason=None if feasible else "x",
                            score_context=None if intrinsic else rel, score_alone=1.0, feasible=feasible)
    return StepEval(step=step, edge=step, state=state, component=comp, label=f"c{comp}", kind=kind,
                    feasible=feasible, grasp=g,
                    tool=None if kind == "gripper" else SimpleNamespace(contact_with={}, feasible=feasible))


def real_optimizer(aog, tables, cfg):
    return lambda blocked: op.optimize(aog, tables, cfg, blocked_edges=blocked or None)


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------


def test_r1(rec, args):
    """Relative score and feasibility rule (spec 5.2, decided thresholds 0.5 / 0.5)."""
    cfg = RobotConfig()
    rec.check(cfg.grasp_rel_threshold == 0.5 and cfg.grasp_abs_floor == 0.5 and cfg.grasp_rotation_step_deg == 90,
              "defaults: rel 0.5, floor 0.5, rotation step 90 deg")
    rec.check(relative_score(0.9, 0.6) == 1.0, "S_rel clips at 1")
    rec.check(abs(relative_score(0.3, 0.6) - 0.5) < 1e-12, "S_rel = S_context / S_alone")
    rec.check(relative_score(0.3, 0.0) == 0.0 and not feasibility(0.3, 0.0, cfg)[0], "S_alone = 0 -> infeasible")
    rec.check(feasibility(0.5, 1.0, cfg)[0], "boundary inclusive: S_rel = 0.5 and S = 0.5 is feasible")
    rec.check(not feasibility(0.5, 1.0, RobotConfig(grasp_boundary_inclusive=False))[0], "exclusive boundary rejects it")
    rec.check(not feasibility(0.45, 0.5, cfg)[0], "S_rel 0.9 but S 0.45 below the floor -> infeasible")
    rec.check(not feasibility(0.6, 1.0, RobotConfig(grasp_rel_threshold=0.7))[0], "S_rel 0.6 below 0.7 -> infeasible")
    rec.check(intrinsically_infeasible(0.488, cfg) and not intrinsically_infeasible(0.551, cfg),
              "part1-like S_alone 0.488 fails alone at floor 0.5, 0.551 does not")


def test_r2(rec, args):
    """Quarter pruning (spec 6): worked example, N < 4, infeasible always pruned, tool steps excluded."""
    cfg = RobotConfig()
    steps = [fake_step(i + 1, 100 + i, i, rel=r) for i, r in enumerate((0.9, 0.8, 0.7, 0.6))]
    pr = [k for k, _ in edges_to_prune(steps, cfg)]
    rec.check(pr == [(103, 3)], f"0.9/0.8/0.7/0.6 -> only the 0.6 edge pruned ({pr})")
    rec.check(quarter_count(3) == 1 and quarter_count(1) == 1 and quarter_count(8) == 2 and quarter_count(0) == 0,
              "max(1, floor(N/4)): N=3 -> 1, N=1 -> 1, N=8 -> 2, N=0 -> 0")
    steps = [fake_step(1, 10, 0, rel=0.9), fake_step(2, 11, 1, rel=0.2, feasible=False),
             fake_step(3, 12, 2, rel=0.3, feasible=False), fake_step(4, 13, 3, kind="tool", feasible=False),
             fake_step(5, 14, 4, kind="tool", feasible=True)]
    pr = {k for k, _ in edges_to_prune(steps, cfg)}
    rec.check(pr == {(11, 1), (12, 2), (13, 3)}, f"every infeasible edge (gripper and tool) pruned ({sorted(pr)})")
    steps = [fake_step(1, 10, 0, rel=0.9), fake_step(2, 11, 1, feasible=False, intrinsic=True),
             fake_step(3, 12, 2, rel=0.8)]
    pr = [k for k, _ in edges_to_prune(steps, cfg)]
    rec.check(pr == [(12, 2)], f"intrinsically infeasible steps are not pruned nor counted ({pr})")
    steps = [fake_step(1, 10, 0, rel=0.95, feasible=True), fake_step(2, 11, 1, rel=0.99, feasible=False)]
    pr = [k for k, _ in edges_to_prune(steps, RobotConfig(grasp_quarter_fraction=0.25))]
    rec.check(pr[0] == (11, 1), "infeasible ranks below any feasible step regardless of S_rel")


def test_r3(rec, args):
    """Final choice ranking (spec 6): (1) infeasible count, (2) weakest S_rel, (3) sum, (4) cost."""
    base = {"n_infeasible": 0, "min_relative": 0.5, "sum_relative": 2.0, "total_cost": 0.5}
    r = lambda **kw: {**base, **kw}
    rec.check(choose([r(n_infeasible=1, min_relative=0.9), r(n_infeasible=0, min_relative=0.1)]) == 1, "(1) fewer infeasible wins")
    rec.check(choose([r(min_relative=0.4, sum_relative=9), r(min_relative=0.6, sum_relative=1)]) == 1, "(2) bottleneck before sum")
    rec.check(choose([r(sum_relative=2.0, total_cost=0.1), r(sum_relative=2.5, total_cost=0.9)]) == 1, "(3) sum before cost")
    rec.check(choose([r(total_cost=0.3), r(total_cost=0.2)]) == 1, "(4) lower cost last")


def test_r4(rec, args):
    """Loop (spec 6): blocked edges only grow, never reused, records = iterations, cache counts, choice."""
    aog = all_free_aog(4)
    tables = make_tables(aog)
    cfg_opt = only("use_travel", "use_direction_change", n_iterations=40)
    eng = FakeEngine([0.9, 0.8, 0.7, 0.95])
    clouds = [toy_cloud(c) for c in range(4)]
    an = GripperAnalysis(eng, clouds, [f"k{c}" for c in range(4)], [f"c{c}" for c in range(4)], RobotConfig())
    prep = SimpleNamespace(stander_records={})
    res = run_robot_loop(aog, None, prep, an, NoTools(), RobotConfig(), optimize_fn=real_optimizer(aog, tables, cfg_opt))
    rec.note(f"{res.iterations} iteration(s), stop: {res.stop_reason}, {len(res.blocked)} blocked, engine calls {eng.calls}")
    rec.check(res.iterations == len(res.records) == len(res.sequences), "record count equals iterations run")
    rec.check(res.iterations == 10 or "no path" in res.stop_reason or "no new pruning" in res.stop_reason,
              "runs 10 iterations unless no path remains / nothing new to prune")
    seen, ok_grow, ok_unused = set(), True, True
    for r in res.records:
        edges = {(int(aog.node_mask[aog.edge_src[e]]), int(aog.edge_comp[e])) for e in r["edge_path"]}
        if edges & seen:
            ok_unused = False
        before = len(seen)
        seen |= set(r["pruned"])
        ok_grow &= len(seen) >= before
    rec.check(ok_grow and len(seen) == len(res.blocked), "blocked edges only grow")
    rec.check(ok_unused, "no sequence uses an edge blocked in an earlier iteration")
    # A last step (only the removed part present) has no obstacles: same call as its alone score.
    distinct = {(s.state, s.component) for r in res.records for s in r["steps"] if s.state != (1 << s.component)}
    rec.check(eng.calls == len(distinct) + 4,
              f"engine calls = distinct (state, component) with obstacles + 4 alone ({eng.calls})")
    rec.check(res.chosen == choose(res.records), "chosen record follows the ranking")

    # Repeated candidate: an optimizer that ignores the blocked set; only alone calls + first sequence.
    eng2 = FakeEngine([0.9, 0.8, 0.7, 0.95])
    an2 = GripperAnalysis(eng2, clouds, [f"k{c}" for c in range(4)], [f"c{c}" for c in range(4)], RobotConfig())
    first = op.optimize(aog, tables, cfg_opt)
    calls = []
    res2 = run_robot_loop(aog, None, prep, an2, NoTools(), RobotConfig(),
                          optimize_fn=lambda b: (calls.append(1), first)[1])
    rec.check(eng2.calls == 4 + 3, f"a repeated sequence is not re-evaluated (engine calls {eng2.calls} = 4 alone "
                                   "+ 3 steps with obstacles)")
    rec.check(res2.iterations == 2 and "no new pruning" in res2.stop_reason,
              f"stops once nothing new can be pruned ({res2.iterations}, {res2.stop_reason})")

    # Early exit: 2 components -> 2 paths; one edge pruned per iteration.
    aog2 = all_free_aog(2)
    t2 = make_tables(aog2)
    an3 = GripperAnalysis(FakeEngine([0.9, 0.9]), [toy_cloud(0), toy_cloud(1)], ["a", "b"], ["c0", "c1"], RobotConfig())
    res3 = run_robot_loop(aog2, None, prep, an3, NoTools(), RobotConfig(), optimize_fn=real_optimizer(aog2, t2, cfg_opt))
    rec.check("no path" in res3.stop_reason and res3.iterations < 10, f"early exit when no path ({res3.iterations}, {res3.stop_reason})")


def test_r5(rec, args):
    """Rotation (spec 5.3): engine at every angle, only grasps feasible at all angles kept; static = 1 call."""
    ga = [EngineResult(grasps=[Grasp(np.array([0., 0, 0]), np.eye(3), 1, 5, 0.9), Grasp(np.array([5., 0, 0]), np.eye(3), 1, 5, 0.8)]),
          EngineResult(grasps=[Grasp(np.array([0.1, 0, 0]), np.eye(3), 1, 5, 0.6)])]
    sv = surviving_grasps(ga, tol_mm=0.5, angle_deg=5)
    rec.check(len(sv) == 1 and abs(sv[0].score - 0.6) < 1e-12, "matching keeps grasps present at every angle, lowest score")
    flipped = [ga[0], EngineResult(grasps=[Grasp(np.array([0., 0, 0]), np.diag([-1., 1, -1]), 1, 5, 0.9)])]
    rec.check(not surviving_grasps(flipped, 0.5, 5), "same TCP but flipped axes does not match")

    eng = RotEngine()
    P = np.array([[0.0, 0, 0], [3, 0, 0], [0, 3, 0]])
    clouds = [(P, np.tile([0, 0, 1.0], (3, 1))), (np.array([[10.0, 0, 0]] * 3), np.tile([0, 0, 1.0], (3, 1)))]
    an = GripperAnalysis(eng, clouds, ["t", "o"], ["part", "obst"], RobotConfig(grasp_abs_floor=0.0))
    s = an.step(0b11, 0)
    rec.check(eng.calls == 2 and abs(s.score_context - 0.9) < 1e-12 and not s.rotation_checked,
              f"non-rotating step: alone + one call, best 0.9 ({eng.calls} calls)")
    rot = Rotation(axis_point=(0, 0, 0), axis_dir=(0, 0, 1), nut_id="n", nut_label="nut")
    s = an.step(0b11, 0, rot)
    rec.check(eng.calls == 5, f"rotating step: 3 more calls at 90/180/270 (angle 0 shared) ({eng.calls})")
    rec.check(s.rotation_checked and s.angles == [0.0, 90.0, 180.0, 270.0], f"angles tested {s.angles}")
    rec.check(abs(s.score_context - 0.8) < 1e-12 and abs(s.score_relative - 0.8 / 0.9) < 1e-12,
              f"only B (0.8) survives every angle; S_rel = 0.8/0.9 ({s.score_context})")
    an.step(0b11, 0, rot)
    rec.check(eng.calls == 5, "repeated rotating step is served from the cache")

    # rotation_for: partner of a present nut rotates; nut gone or partner a screw -> no rotation.
    aog = SimpleNamespace(component_ids=["P", "N", "S"], component_labels=["part", "nut", "screw"])
    recs = {"N": SimpleNamespace(category=2, partner="P", options=[SimpleNamespace(direction=(1.0, 0, 0))],
                                 bbox=((0, 0, 0), (2, 2, 2))),
            "S": SimpleNamespace(category=1, partner=None, options=[], bbox=None)}
    r = rotation_for(0b011, 0, aog, recs)
    rec.check(r is not None and r.nut_label == "nut" and r.axis_point == (1.0, 1.0, 1.0), "partner of a present nut rotates about its axis")
    rec.check(rotation_for(0b001, 0, aog, recs) is None, "nut already removed -> no rotation")


def test_r6(rec, args):
    """Optimization hook (spec 6.1): None = old behaviour and key, blocked edge never used, key differs."""
    aog = all_free_aog(4)
    tables = make_tables(aog)
    cfg = only("use_travel", "use_direction_change", n_iterations=40)
    r0 = op.optimize(aog, tables, cfg)
    r1 = op.optimize(aog, tables, cfg, blocked_edges=None)
    r2 = op.optimize(aog, tables, cfg, blocked_edges=set())
    rec.check(r0.edge_path == r1.edge_path == r2.edge_path and r0.total_cost == r1.total_cost == r2.total_cost,
              "blocked_edges None/empty reproduces the result")
    rec.check(r0.config_hash == r1.config_hash == r2.config_hash == op.compute_config_hash(aog, tables, cfg),
              "blocked_edges None/empty keeps the cache key")
    e = r0.edge_path[0]
    blk = {(int(aog.node_mask[aog.edge_src[e]]), int(aog.edge_comp[e]))}
    r3 = op.optimize(aog, tables, cfg, blocked_edges=blk)
    used = {(int(aog.node_mask[aog.edge_src[x]]), int(aog.edge_comp[x])) for x in r3.edge_path}
    rec.check(not (used & blk), "a blocked edge never appears in the chosen sequence")
    rec.check(r3.config_hash != r0.config_hash, "the cache key differs with blocked edges")
    root_edges = {(int(aog.node_mask[aog.root]), int(aog.edge_comp[x])) for x in aog.out_edges(aog.root)}
    try:
        op.optimize(aog, tables, cfg, blocked_edges=root_edges)
        rec.check(False, "blocking every root edge raises NoPathError")
    except op.NoPathError:
        rec.check(True, "blocking every root edge raises NoPathError")


def test_r7(rec, args):
    """Mode switch (spec 10): flag wins, no flag asks, invalid errors, no terminal -> manual."""
    import main
    rec.check(main.resolve_mode("robot", ask=lambda p: "manual", interactive=True) == "robot", "flag wins over the prompt")
    rec.check(main.resolve_mode(None, ask=lambda p: "robot", interactive=True) == "robot", "no flag: asks (robot)")
    rec.check(main.resolve_mode(None, ask=lambda p: "", interactive=True) == "manual", "no flag: Enter = manual")
    answers = iter(["banana", "r"])
    rec.check(main.resolve_mode(None, ask=lambda p: next(answers), interactive=True) == "robot", "invalid answer asks again")
    rec.check(main.resolve_mode(None, interactive=False) == "manual", "no flag and no terminal: manual")
    try:
        main.resolve_mode("auto")
        rec.check(False, "invalid flag value raises")
    except ValueError:
        rec.check(True, "invalid flag value raises")
    import subprocess
    p = subprocess.run([sys.executable, str(REPO_ROOT / "src" / "main.py"), "a.stp", "b.csv", "out", "--mode", "auto"],
                       capture_output=True, text=True)
    rec.check(p.returncode == 2 and "invalid choice" in p.stderr, "CLI rejects --mode auto")


def test_r8(rec, args):
    """Output (spec 8): plan.json keeps every old key and gains the robot block."""
    from hierarchy import Container, Hierarchy
    from levels import LevelPlan
    from output_robotic import write_robot_plan
    import main

    aog = all_free_aog(3)
    tables = make_tables(aog)
    cfg = only("use_travel", n_iterations=20)
    an = GripperAnalysis(FakeEngine([0.9, 0.8, 0.6]), [toy_cloud(c) for c in range(3)], ["a", "b", "c"],
                         list(aog.component_labels), RobotConfig())
    loop = run_robot_loop(aog, None, SimpleNamespace(stander_records={}), an, NoTools(), RobotConfig(),
                          optimize_fn=real_optimizer(aog, tables, cfg))
    root = Container(id="root", label="Toy", parent=None, leaf_ids=list(aog.component_ids), child_ids=[],
                     processed=True, all_leaf_ids=list(aog.component_ids))
    hier = Hierarchy(root="root", containers={"root": root}, leaf_label={c: c for c in aog.component_ids},
                     leaf_parent={c: "root" for c in aog.component_ids})
    plan = LevelPlan(container_id="root", label="Toy", level_dir=SCRATCH, prep=None, interference=None, aog=aog,
                     sequence=loop.sequences[loop.chosen])
    out = SCRATCH / "r8"
    out.mkdir(parents=True, exist_ok=True)
    d = write_robot_plan(hier, plan, loop, aog, an, RobotConfig(), out, "toy")
    manual = main.build_plan(hier, plan, "toy")
    rec.check(all(k in d and d[k] == manual[k] for k in manual), "every manual key present and unchanged")
    r = d["robot"]
    need = {"mode", "feasible", "failing_steps", "steps", "loop", "alone_scores", "thresholds", "config"}
    rec.check(need <= set(r), f"robot block has {sorted(need)}")
    st = r["steps"][0]
    rec.check({"end_effector", "tcp", "axes", "grasp_width", "contact_area", "score", "score_relative",
               "rotation_checked", "angles"} <= set(st), "gripper step fields present")
    rec.check([s["step"] for s in r["steps"]] == [s["step"] for s in d["level"]["steps"]], "robot steps align with level steps")
    rec.check(len(r["loop"]["sequences"]) == loop.iterations and sum(s["chosen"] for s in r["loop"]["sequences"]) == 1,
              "every recorded sequence listed, exactly one chosen")
    rec.check(json.loads((out / "plan.json").read_text())["robot"]["mode"] == "robot", "plan.json written with the robot block")


class NodeEngine:
    """Sub-assembly fake: a one-part target scores 0.9 * (1 - 0.05 * obstacles); a multi-part target (a node)
    scores node_table[number of parts]. Counts part calls and node calls separately."""

    def __init__(self, node_table, n_per_cloud=10):
        self.table, self.n_per = node_table, n_per_cloud
        self.part_calls = self.node_calls = 0

    def evaluate(self, target, obstacles):
        n_t = len(target[0]) // self.n_per
        k = 0 if obstacles is None else len(obstacles[0]) // self.n_per
        if n_t == 1:
            self.part_calls += 1
            s = 0.9 * (1 - 0.05 * k)
        else:
            self.node_calls += 1
            s = self.table[n_t]
        return EngineResult(grasps=[Grasp(tcp=np.zeros(3), axes=np.eye(3), contact_area=1.0, grasp_width=10.0, score=s)])

    @property
    def calls(self):
        return self.part_calls + self.node_calls


def test_r13(rec, args):
    """Optional sub-assembly check (spec 13.6): an edge needs both nodes graspable; nodes cached; root never pruned."""
    from output_robotic import robot_block
    from robot_loop_robotic import evaluate_sequence
    aog = all_free_aog(4)
    tables = make_tables(aog)
    cfg_opt = only("use_travel", "use_direction_change", n_iterations=40)
    clouds = [toy_cloud(c) for c in range(4)]
    keys, labels = [f"k{c}" for c in range(4)], [f"c{c}" for c in range(4)]
    prep = SimpleNamespace(stander_records={})
    seq = op.optimize(aog, tables, cfg_opt)

    # off (default): no node calls, no node fields, identical feasibility
    eng = NodeEngine({2: 0.9, 3: 0.2, 4: 0.9})
    an = GripperAnalysis(eng, clouds, keys, labels, RobotConfig(grasp_abs_floor=0.5))
    steps = evaluate_sequence(seq, aog, {}, an, NoTools())
    rec.check(RobotConfig().check_subassembly_grasp is False, "off by default")
    rec.check(eng.node_calls == 0 and all(s.node_after is None for s in steps) and all(s.feasible for s in steps),
              "off: no sub-assembly calls, no node fields, all part steps feasible")

    # on: 3-part sub-assemblies are not graspable -> the two edges touching them are infeasible
    cfg = RobotConfig(check_subassembly_grasp=True)
    eng = NodeEngine({2: 0.9, 3: 0.2, 4: 0.9})
    an = GripperAnalysis(eng, clouds, keys, labels, cfg)
    steps = evaluate_sequence(seq, aog, {}, an, NoTools())
    bad = [s.step for s in steps if not s.feasible]
    pc = lambda m: bin(m).count("1")
    expect = [s.step for s in steps if pc(s.state) == 3 or pc(s.state & ~(1 << s.component)) == 3]
    rec.check(bad == expect and len(bad) == 2, f"edges into/out of the 3-part node infeasible ({bad})")
    rec.check(all(s.part_feasible for s in steps), "the part checks themselves still pass")
    pr = edges_to_prune(steps, cfg)
    rec.check(any("sub-assembly not graspable" in why for _, why in pr), "node failures are pruned with their reason")
    n_nodes = len({s.state for s in steps if pc(s.state) > 1} | {s.state & ~(1 << s.component) for s in steps
                                                                  if pc(s.state & ~(1 << s.component)) > 1})
    rec.check(eng.node_calls == n_nodes, f"one engine call per distinct multi-part node ({eng.node_calls} = {n_nodes})")
    an.alone(0)  # already evaluated by the steps above
    before = eng.calls
    rec.check(an.node(1 << 0).score == an.alone(0) and eng.calls == before,
              "a one-part node reuses that part's alone call (no new engine call)")
    evaluate_sequence(seq, aog, {}, an, NoTools())
    rec.check(eng.calls == before, "re-evaluating a sequence makes no new calls (nodes cached)")

    # the full assembly not graspable: reported, never pruned
    eng = NodeEngine({2: 0.9, 3: 0.9, 4: 0.2})
    an = GripperAnalysis(eng, clouds, keys, labels, cfg)
    steps = evaluate_sequence(seq, aog, {}, an, NoTools())
    root_steps = [s for s in steps if s.state == s.root_state]
    rec.check(len(root_steps) == 1 and not root_steps[0].feasible, "the step producing the full assembly is infeasible")
    why = dict(edges_to_prune(steps, cfg)).get(root_steps[0].edge_key, "")
    rec.check("sub-assembly" not in why,
              f"a failure of the full-assembly node is never a pruning reason (reason given: {why or 'not pruned'})")

    # loop + output
    eng = NodeEngine({2: 0.9, 3: 0.2, 4: 0.9})
    an = GripperAnalysis(eng, clouds, keys, labels, cfg)
    res = run_robot_loop(aog, None, prep, an, NoTools(), cfg, optimize_fn=real_optimizer(aog, tables, cfg_opt))
    rec.note(f"loop: {res.iterations} iteration(s), stop: {res.stop_reason}")
    rec.check(all(r["n_infeasible"] >= 2 for r in res.records), "every sequence crosses a 3-part node: >= 2 infeasible")
    blk = robot_block(res, aog, an, cfg)
    st0 = blk["steps"][0]
    rec.check(blk["subassembly_scores"] and "subassembly_after" in st0 and "part_feasible" in st0,
              "plan robot block has subassembly_scores and per-step node results")
    rec.check(blk["thresholds"]["check_subassembly_grasp"] is True and blk["thresholds"]["subassembly_grasp_floor"] == 0.5,
              "thresholds report the option and the node floor (default = absolute floor)")


# ---------------------------------------------------------------------------
# freecad
# ---------------------------------------------------------------------------


def test_r9(rec, args):
    """Point clouds (spec 3): deterministic, mm, counts above the engine minimum, cache round trip."""
    import freecad  # noqa: F401
    import Part
    from FreeCAD import Vector
    from pointclouds_robotic import component_cloud, component_clouds

    box = Part.makeBox(40, 20, 10, Vector(5, 6, 7))
    a = component_cloud(box, 0.3, 60000, 1)
    b = component_cloud(box, 0.3, 60000, 1)
    rec.check(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), "same input twice -> identical cloud")
    c = component_cloud(box, 0.3, 60000, 2)
    rec.check(not np.array_equal(a[0], c[0]), "another seed -> another cloud")
    lo, hi = a[0].min(0), a[0].max(0)
    rec.check(np.allclose(lo, [5, 6, 7], atol=0.05) and np.allclose(hi, [45, 26, 17], atol=0.05), f"in mm, global placement ({lo}, {hi})")
    rec.check(len(a[0]) == min(60000, max(1000, int(box.Area / 0.09))), f"count from area/spacing^2 ({len(a[0])})")
    small = Part.makeBox(2, 2, 2)
    rec.check(len(component_cloud(small, 0.3, 60000, 1)[0]) >= 1000, "small part still gets >= 1000 points (> 500 engine minimum)")
    n = np.linalg.norm(a[1], axis=1)
    rec.check(np.allclose(n, 1.0), "unit normals")
    cache = SCRATCH / "r9_clouds"
    shutil.rmtree(cache, ignore_errors=True)
    c1, k1 = component_clouds(["box"], [box], cache_dir=cache)
    c2, k2 = component_clouds(["box"], [box], cache_dir=cache)
    rec.check(k1 == k2 and np.array_equal(c1[0][0], c2[0][0]) and np.array_equal(c1[0][0], a[0]), "disk cache round trip")


def test_r10(rec, args):
    """Tool check (spec 7) with the real screw tool on a synthetic scene."""
    import freecad  # noqa: F401
    import Part
    from FreeCAD import Vector
    from tool_check import ToolChecker, load_tool

    tool = load_tool(SCREW_TOOL)
    rec.check(abs(tool.tip.z - 241.0) < 1.0, f"bit tip found on the +Z axis (z = {tool.tip.z:.2f})")
    plate = Part.makeBox(60, 60, 10, Vector(-30, -30, 0))
    screw = Part.makeCylinder(3, 14, Vector(0, 0, 0), Vector(0, 0, 1)).fuse(
        Part.makeCylinder(5, 4, Vector(0, 0, 10), Vector(0, 0, 1)))
    lid = Part.makeBox(30, 30, 5, Vector(-15, -15, 20))  # right above the head, in the tool's way
    base = Part.makeBox(80, 80, 20, Vector(-40, -40, -20))  # under the plate: blocks a tool from -z
    shapes = [plate, screw, lid, base]
    dirs = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    names = ["plate", "screw", "lid", "base"]
    tc = ToolChecker(shapes, names, [False, True, False, False], {1: 1}, dirs, RobotConfig())
    rec.check(tc.tool_contact(0b1111, 1, 0), "lid above the head: contact from +z")
    rec.check(not tc.tool_contact(0b1011, 1, 0), "lid removed: no contact from +z")
    rec.check(tc.tool_contact(0b1011, 1, 1), "from -z the tool runs into the base: contact")
    rec.check(not tc.tool_contact(0b0011, 1, 1), "base removed: the tool only meets the screw's own end from -z")
    s = tc.step(0b1111, 1, [0, 1], preferred=0)
    rec.check(not s.feasible and s.contact_with.get(0) == ["lid"], f"both senses blocked -> infeasible ({s.contact_with})")
    s = tc.step(0b1011, 1, [0, 1], preferred=1)
    rec.check(s.feasible and s.sense == 0, "one free sense -> feasible, the free sense is reported")
    tc2 = ToolChecker(shapes, names, [False, True, False, False], {1: 3}, dirs,
                      RobotConfig(tools={1: "screwing_effector.step", 2: "nut_effector.stp", 3: None}))
    s = tc2.step(0b1111, 1, [0, 1], preferred=0)
    rec.check(s.feasible and not s.checked, "category mapped to no tool: unchecked, feasible (stub)")

    # Interface planes of the three tools (2026-11-12 rule: bbox max Z of the solid the axis meets highest).
    cfg = RobotConfig()
    tips = {cat: load_tool(Path(cfg.tool_dir) / cfg.tools[cat]).tip.z for cat in (1, 2, 3)}
    rec.check(abs(tips[1] - 241.0) < 0.1 and abs(tips[2] - 239.0) < 0.1 and abs(tips[3] - 235.0) < 0.1,
              f"interface planes: screw 241, nut 239 (socket rim, not cavity floor), bushing 235 ({tips})")

    # Overlap: screw 0; nut / bushing half their extent along the axis (12 mm nut -> 6 mm inside the tool).
    nut = Part.makeCylinder(8, 12, Vector(0, 0, 10), Vector(0, 0, 1)).cut(
        Part.makeCylinder(4, 12, Vector(0, 0, 10), Vector(0, 0, 1)))  # z 10..22
    plate2 = Part.makeBox(80, 80, 10, Vector(-40, -40, 0))
    for cat, expect, what in ((1, 22.0, "screw: no overlap, tool face on the head (z 22)"),
                              (2, 16.0, "nut: tool face 6 mm into the 12 mm nut (z 16)"),
                              (3, 16.0, "bushing: tool face half its extent into it (z 16)")):
        tcx = ToolChecker([plate2, nut], ["plate", "stander"], [False, True], {1: cat}, dirs, cfg)
        low = min(s.optimalBoundingBox().ZMin for s in tcx._tool_solids(1, 0))  # BoundBox is loose here
        rec.check(abs(low - expect) < 0.05, f"{what}; placed tool lowest z = {low:.2f}")
    tcn = ToolChecker([plate2, nut, lid], ["plate", "nut", "lid"], [False, True, False], {1: 2}, dirs, cfg)
    rec.check(not tcn.tool_contact(0b011, 1, 0), "nut socket from +z on a nut sitting on a plate: no contact")
    rec.check(tcn.tool_contact(0b111, 1, 0), "with a lid above the nut: contact")


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


_FASTENER_CLOUDS = None


def fastener_clouds():
    """industrial_fastner leaf clouds at the spec's settings: {label: (P, N)}."""
    global _FASTENER_CLOUDS
    if _FASTENER_CLOUDS is None:
        import freecad  # noqa: F401
        import FreeCAD
        import Import
        from pointclouds_robotic import component_clouds
        doc = FreeCAD.newDocument("robot_test_fastener")
        Import.insert(str(FASTENER_STEP), doc.Name)
        doc.recompute()
        labels, shapes = [], []
        for o in doc.Objects:  # leaves, as gripper_test/make_pcds_freecad.py
            if o.isDerivedFrom("Part::Feature") and not o.Shape.isNull() and o.Shape.Faces:
                s = o.Shape.copy()
                s.Placement = o.getGlobalPlacement()
                labels.append(o.Label)
                shapes.append(s)
        clouds, _ = component_clouds(labels, shapes, cache_dir=SCRATCH / "fastener_clouds", labels=labels)
        _FASTENER_CLOUDS = dict(zip(labels, clouds))
    return _FASTENER_CLOUDS


def test_r11(rec, args):
    """Engine wrapper (spec 4/10): runs (UTF-8), deterministic, empty obstacles, crash raises, alone scores."""
    from gripper_engine_robotic import EngineError, GraspEngine
    clouds = fastener_clouds()
    work = SCRATCH / "r11_engine"
    shutil.rmtree(work, ignore_errors=True)
    with GraspEngine(seed=1, work_dir=work) as eng:
        t0 = time.time()
        scores = {}
        for label, ref in SPEC_ALONE.items():
            scores[label] = eng.evaluate(clouds[label], None).score
        rec.note("alone scores " + ", ".join(f"{k[-5:]} {v:.3f} (spec {SPEC_ALONE[k]})" for k, v in scores.items())
                 + f"; {time.time() - t0:.0f} s")
        rec.check(all(abs(scores[k] - SPEC_ALONE[k]) <= 0.06 for k in scores), "alone scores within 0.06 of spec 5.2")
        p3 = clouds["industrial_fastner_part3"]
        a = eng.evaluate(p3, None)
        b = eng.evaluate(p3, (np.zeros((0, 3)), np.zeros((0, 3))))
        rec.check(len(a.grasps) == len(b.grasps) and a.score == b.score, "None and an empty obstacle cloud agree")
        rec.check(a.score == scores["industrial_fastner_part3"], "same input twice -> identical score (seeded)")
    bad = SCRATCH / "bad_gripper.yaml"
    bad.write_text("a_pg: 25.0\n", encoding="utf-8")
    with GraspEngine(seed=1, gripper_yaml=bad, work_dir=SCRATCH / "r11_bad") as eng:
        try:
            eng.evaluate(clouds["industrial_fastner_part3"], None)
            rec.check(False, "an engine crash raises EngineError")
        except EngineError as exc:
            rec.check("KeyError" in str(exc), "an engine crash raises EngineError (not a score of 0)")


def test_r12(rec, args):
    """Function copy == unchanged script (same seed): alone and with obstacles (spec 4)."""
    from gripper_engine_robotic import GraspEngine
    from pointclouds_robotic import merge_clouds
    clouds = fastener_clouds()
    p = "industrial_fastner_part3"
    obst = merge_clouds([c for k, c in clouds.items() if k != p])
    with GraspEngine(seed=1, work_dir=SCRATCH / "r12_w") as w, GraspEngine(seed=1, mode="script", work_dir=SCRATCH / "r12_s") as s:
        for name, ob in (("alone", None), ("assembly", obst)):
            t0 = time.time()
            rw = w.evaluate(clouds[p], ob)
            t1 = time.time()
            rs = s.evaluate(clouds[p], ob)
            t2 = time.time()
            rec.note(f"{name}: worker {len(rw.grasps)} grasps best {rw.score:.4f} ({t1 - t0:.0f} s), "
                     f"script {len(rs.grasps)} grasps best {rs.score:.4f} ({t2 - t1:.0f} s)")
            same = len(rw.grasps) == len(rs.grasps) and all(
                np.allclose(g.tcp, h.tcp, atol=1e-6) and abs(g.score - h.score) < 1e-9
                for g, h in zip(rw.grasps, rs.grasps))
            rec.check(same, f"{name}: identical grasps from the function copy and the unchanged script")


TESTS = {
    "R1": ("core", test_r1), "R2": ("core", test_r2), "R3": ("core", test_r3), "R4": ("core", test_r4),
    "R5": ("core", test_r5), "R6": ("core", test_r6), "R7": ("core", test_r7), "R8": ("core", test_r8),
    "R13": ("core", test_r13),
    "R9": ("freecad", test_r9), "R10": ("freecad", test_r10),
    "R11": ("engine", test_r11), "R12": ("engine", test_r12),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="+")
    ap.add_argument("--groups", nargs="+", default=["core"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rec = Recorder()
    todo = [t for t, (g, _) in TESTS.items() if (args.tests and t in args.tests) or (not args.tests and g in args.groups)]
    times = {}
    for tid in todo:
        print(f"\n=== {tid}: {TESTS[tid][1].__doc__.splitlines()[0]}", flush=True)
        rec.start(tid)
        t0 = time.time()
        try:
            TESTS[tid][1](rec, args)
        except Exception as exc:  # noqa: BLE001
            rec.error(exc)
        times[tid] = time.time() - t0
    print("\n| test | result | time | notes |\n|---|---|---|---|")
    for tid in todo:
        r = rec.results[tid]
        status = "ERROR" if "error" in r else ("PASS" if r["passed"] else "FAIL")
        print(f"| {tid} | {status} | {times[tid]:.1f} s | {r.get('error', '; '.join(r['notes']))} |")
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "robotic_results.json").write_text(json.dumps(rec.results, indent=1), encoding="utf-8")
    return 0 if all(r["passed"] for r in rec.results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
