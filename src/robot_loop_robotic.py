"""Robot mode: verify-prune-reoptimize loop.

Inputs
    - STEP file, BOM CSV and output folder (`run_robot_pipeline`), a `RobotConfig` and the usual stage configs.
    - Internally: the AoG, interference result and preprocessing result of the (flat) assembly, the
      per-component point clouds, a `GripperAnalysis` (grasp engine) and a `ToolChecker`.
Outputs
    - `plan.json` with a `robot` block (written through `output_robotic.write_robot_plan`), for the
      sequence the loop chose, and the returned plan dict.
    - `<output_dir>/robot/`: clouds, engine cache, engine work folder.

What this module does
    Same pipeline as manual mode up to and including Optimization (unchanged). Then, repeatedly:

        blocked = {}                                    # (state, component) -> iteration that pruned it
        for it in 1..robot_max_iterations:
            seq = optimize_sequence(..., blocked_edges=blocked)   # NoPathError -> stop
            steps = evaluate(seq)                       # every step; cached; a repeated sequence is not re-evaluated
            prune = infeasible edges + worst quarter of the gripper steps
            blocked |= prune
        choice = choose(records)

    A step is checked with the gripper (grasp engine, `gripper_analysis_robotic`) for normal parts, or with
    the tool check (`tool_check`) for standers.

Implementation decisions:
  - Intrinsically infeasible gripper components (fail even alone, see gripper_analysis_robotic) are
    never pruned and do not count in the quarter.
  - If an iteration adds no new blocked edge, the next optimization would return the same sequence
    (the ACO is seeded), so the loop stops there instead of repeating it ("no new pruning").
  - Rotating partner steps are found structurally (the removed part is a nut's `partner` and that nut
    is still present), the same rule `optimization._apply_turns` uses for its "partner rotates" row,
    but without requiring a BOM pitch (the pitch only matters for the displayed turn count).

Robot mode is for flat assemblies only: a STEP with nested sub-assemblies is rejected.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gripper_analysis_robotic import GripperAnalysis, Rotation
from optimization import NoPathError, OptimizationConfig, optimize_sequence
from robot_config_robotic import RobotConfig

log = logging.getLogger("aap.robot")


# ---------------------------------------------------------------------------
# Step evaluation
# ---------------------------------------------------------------------------


def rotation_for(state: int, c: int, aog, stander_records: dict) -> Rotation | None:
    """[MAIN] The nut axis component c turns about when removed from `state`, or None.

    Input: the AoG state bitmask, the component index, the AoG and the stander records. Output: a
    `Rotation` if c is the partner of a nut that is still present (so c turns while it is removed, and the
    grasp must survive that rotation), else None. A screw never rotates here (its own thread makes it a
    tool step anyway). With several present nuts the first one's axis is used, with a warning.
    """
    cid = aog.component_ids[c]
    own = stander_records.get(cid)
    if own is not None and own.category == 1:
        return None  # a screw turns on its own thread; it is a tool step anyway
    found = []
    for n, nid in enumerate(aog.component_ids):
        rec = stander_records.get(nid)
        if rec is None or rec.category != 2 or rec.partner != cid or not (state >> n) & 1:
            continue
        d = rec.options[0].direction
        lo, hi = rec.bbox if rec.bbox is not None else ((0, 0, 0), (0, 0, 0))
        point = tuple((float(_xyz(lo)[i]) + float(_xyz(hi)[i])) / 2 for i in range(3))
        found.append(Rotation(axis_point=point, axis_dir=tuple(float(v) for v in _xyz(d)), nut_id=nid,
                              nut_label=aog.component_labels[n]))
    if len(found) > 1:
        log.warning("%s is the partner of %d present nuts; using %s's axis for the rotation check",
                    aog.component_labels[c], len(found), found[0].nut_label)
    return found[0] if found else None


def _xyz(v) -> tuple[float, float, float]:
    """[AUX] A FreeCAD Vector or a 3-sequence as a tuple of floats."""
    if hasattr(v, "x"):
        return (float(v.x), float(v.y), float(v.z))
    return tuple(float(a) for a in v)


@dataclass
class StepEval:
    """Check result of one step of a sequence (output of `evaluate_sequence`, input of pruning, choice and plan.json)."""

    step: int  # int. Assembly step number of the sequence.
    edge: int  # int. AoG edge index of the step.
    state: int  # int bitmask. Components present before removal.
    component: int  # int. AoG component index of the removed part.
    label: str  # str. BOM label of the part.
    kind: str  # str. "gripper" (normal part) or "tool" (stander).
    feasible: bool  # bool. The edge: part check and (when enabled) both sub-assembly nodes.
    grasp: object | None = None  # StepGrasp for a gripper step, else None.
    tool: object | None = None  # ToolStep for a tool step, else None.
    # Optional sub-assembly check (RobotConfig.check_subassembly_grasp): the two nodes of the edge, i.e. the
    # sub-assembly after the insertion (`state`) and before it (`state` without the component). None when off.
    node_after: object | None = None  # NodeGrasp
    node_before: object | None = None  # NodeGrasp
    root_state: int = 0  # int bitmask. The full assembly's mask (its node is in every path: never a reason to prune).

    @property
    def edge_key(self) -> tuple[int, int]:
        """[AUX] (state, component): the key under which an edge is blocked in the loop."""
        return (self.state, self.component)

    @property
    def part_feasible(self) -> bool:
        """[AUX] The part's own check (tool or grasp), ignoring the optional sub-assembly nodes."""
        if self.kind == "tool":
            return bool(self.tool.feasible)
        return bool(self.grasp.feasible)

    @property
    def failing_nodes(self) -> list:
        """[AUX] The sub-assembly nodes of this edge that are not graspable (empty when the check is off)."""
        return [n for n in (self.node_after, self.node_before) if n is not None and not n.feasible]

    @property
    def score_relative(self) -> float | None:
        """[AUX] S_rel of a gripper step, None for a tool step."""
        return None if self.grasp is None else self.grasp.score_relative

    @property
    def intrinsic(self) -> bool:
        """[AUX] True if the gripper step is intrinsically infeasible (fails even alone)."""
        return bool(self.grasp is not None and self.grasp.intrinsic)


def evaluate_sequence(seq, aog, stander_records: dict, analysis: GripperAnalysis, tool_checker) -> list[StepEval]:
    """[MAIN] Evaluate every step of an OptimizationResult (assembly order), never stopping at a failure.

    Inputs: the sequence, the AoG, the stander records, the gripper analysis (grasp engine) and the tool
    checker. Output: one `StepEval` per step. Standers are checked with the tool (any free sense without
    contact is enough, the planned sense preferred); other parts with the grasp analysis, including the
    nut-partner rotation. With `check_subassembly_grasp` on, each edge also needs both of its nodes (the
    sub-assembly before and after the insertion) to be graspable; node results are cached per state in the
    analysis. Every step is evaluated so the report shows all failures, not only the first.
    """
    check_nodes = bool(getattr(analysis.config, "check_subassembly_grasp", False))
    root_state = int(aog.node_mask[aog.root])
    out = []
    for s in seq.steps:
        e, c = int(s["edge"]), int(s["component"])
        state = int(aog.node_mask[aog.edge_src[e]])
        if bool(aog.is_stander[c]):
            senses = [int(k) for k in aog.edge_direction_indices(e)]
            ts = tool_checker.step(state, c, senses, preferred=int(s["removal_direction_index"]))
            ev = StepEval(step=int(s["step"]), edge=e, state=state, component=c, label=aog.component_labels[c],
                          kind="tool", feasible=ts.feasible, tool=ts, root_state=root_state)
        else:
            g = analysis.step(state, c, rotation_for(state, c, aog, stander_records))
            ev = StepEval(step=int(s["step"]), edge=e, state=state, component=c, label=aog.component_labels[c],
                          kind="gripper", feasible=g.feasible, grasp=g, root_state=root_state)
        if check_nodes:
            ev.node_after = analysis.node(state)
            ev.node_before = analysis.node(state & ~(1 << c))
            ev.feasible = ev.part_feasible and not ev.failing_nodes
        out.append(ev)
    return out


# ---------------------------------------------------------------------------
# Pruning and choice
# ---------------------------------------------------------------------------


def quarter_count(n_gripper: int, fraction: float = 0.25) -> int:
    """[AUX] How many gripper steps make up the "worst quarter": max(1, floor(n * fraction)), 0 if there are none."""
    return 0 if n_gripper <= 0 else max(1, int(math.floor(n_gripper * fraction)))


def edges_to_prune(steps: list[StepEval], config: RobotConfig) -> list[tuple[tuple[int, int], str]]:
    """[MAIN] [(edge key, reason)]: every infeasible edge + the worst quarter of the gripper steps.

    Input: the evaluated steps of one sequence and the config. Output: the edges to block before the next
    optimization, each with the reason. Pruning the infeasible ones forces the optimizer to find another
    route; pruning the worst quarter pushes it towards better grasps even when nothing failed. Not pruned:
    intrinsically infeasible steps (infeasible in every state), and a failure of the full assembly's node
    alone (it is in every path).
    """
    out: dict[tuple[int, int], str] = {}
    # Every infeasible edge.
    for s in steps:
        if s.feasible or s.intrinsic:
            continue
        if not s.part_feasible:
            if s.kind == "tool":
                out[s.edge_key] = "tool contact in every free sense"
            else:
                out[s.edge_key] = f"infeasible grasp ({s.grasp.reason})"
            continue
        # The part is fine, a sub-assembly node is not. The full assembly is in every path, so a failure of
        # its node alone is reported but never pruned (pruning could only disconnect the AoG).
        avoidable = [n for n in s.failing_nodes if n.state != s.root_state]
        if avoidable:
            out[s.edge_key] = "sub-assembly not graspable (" + "; ".join(n.reason for n in avoidable) + ")"
    # The worst quarter of the (non-intrinsic) gripper steps, by relative score.
    grip = [s for s in steps if s.kind == "gripper" and not s.intrinsic]
    # ranked by the part's own grasp (the quarter is about grasp quality; node failures are pruned above)
    grip.sort(key=lambda s: (s.part_feasible, s.score_relative if s.score_relative is not None else -1.0))
    for s in grip[:quarter_count(len(grip), config.grasp_quarter_fraction)]:
        out.setdefault(s.edge_key, f"worst quarter (relative score {s.score_relative:.3f})")
    return list(out.items())


def sequence_metrics(steps: list[StepEval]) -> dict:
    """[AUX] Figures that compare sequences: number of infeasible steps, weakest S_rel, sum of S_rel (gripper steps only)."""
    rel = [s.score_relative for s in steps if s.kind == "gripper" and s.score_relative is not None]
    return {"n_infeasible": sum(1 for s in steps if not s.feasible),
            "min_relative": min(rel) if rel else 1.0,
            "sum_relative": float(sum(rel))}


def choose(records: list[dict]) -> int:
    """[MAIN] Index of the chosen record: fewer infeasible steps, higher weakest relative score,
    higher sum of relative scores, lower optimization cost (compared in that order)."""
    if not records:
        raise ValueError("No recorded sequences to choose from")
    return min(range(len(records)), key=lambda i: (records[i]["n_infeasible"], -records[i]["min_relative"],
                                                   -records[i]["sum_relative"], records[i]["total_cost"]))


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


@dataclass
class RobotLoopResult:
    """Result of the robot loop (output of `run_robot_loop`, input of `output_robotic`)."""

    records: list[dict]  # list of dicts, one per iteration: iteration, edge_path, total_cost, metrics, steps (StepEval), pruned.
    sequences: list  # list of OptimizationResult, one per record.
    blocked: dict  # dict (state, component) -> {"iteration", "reason"}. Every edge pruned so far.
    chosen: int  # int. Index in `records` of the chosen sequence (see `choose`).
    iterations: int  # int. Number of iterations run.
    stop_reason: str  # str. Why the loop stopped (for the report).
    engine_calls: int = 0  # int. Engine calls actually made (not cached).
    wall_time_s: float = 0.0  # float, s. Duration of the loop.
    extra: dict = field(default_factory=dict)  # dict. Free information added by the caller (engine details).


def run_robot_loop(aog, interference_result, prep, analysis: GripperAnalysis, tool_checker,
                   config: RobotConfig, optimization_config: OptimizationConfig | None = None,
                   optimization_cache_dir: str | Path | None = None, optimize_fn=None) -> RobotLoopResult:
    """[MAIN] The verify-prune-reoptimize loop.

    Inputs: the AoG, interference result and preprocessing result, the gripper analysis and tool checker,
    the config and the optimization settings. `optimize_fn(blocked_edges)` may replace the real optimizer
    (tests); default `optimize_sequence`. Output: a `RobotLoopResult` with every sequence tried and the
    chosen one.

    Each iteration optimizes with the blocked edges, evaluates every step of the new sequence (a sequence
    already tried is not re-evaluated), prunes the infeasible edges and the worst quarter, and stops when
    nothing new is pruned, no path is left (`NoPathError`) or the iteration limit is reached. The best
    recorded sequence is chosen at the end.
    """
    t0 = time.perf_counter()
    if optimize_fn is None:
        def optimize_fn(blocked):
            return optimize_sequence(aog, interference_result, prep, optimization_config,
                                     cache_dir=optimization_cache_dir, blocked_edges=blocked or None)
    blocked: dict[tuple[int, int], dict] = {}
    records, sequences = [], []
    seen: dict[tuple[int, ...], int] = {}
    stop_reason = f"reached robot_max_iterations ({config.robot_max_iterations})"
    for it in range(1, config.robot_max_iterations + 1):
        # 1. Optimize with the edges blocked so far.
        try:
            seq = optimize_fn(set(blocked))
        except NoPathError as exc:
            stop_reason = f"no path left after pruning ({exc})"
            log.info("Robot loop: iteration %d: %s", it, stop_reason)
            break
        # 2. Evaluate every step (reuse the result of an identical earlier sequence).
        key = tuple(int(e) for e in seq.edge_path)
        if key in seen:
            steps = records[seen[key]]["steps"]  # a sequence already tried is not re-evaluated
            log.info("Robot loop: iteration %d repeats the sequence of iteration %d", it, records[seen[key]]["iteration"])
        else:
            steps = evaluate_sequence(seq, aog, prep.stander_records, analysis, tool_checker)
            seen[key] = len(records)
        # 3. Record it and block the edges to prune.
        metrics = sequence_metrics(steps)
        prune = edges_to_prune(steps, config)
        new = [(k, why) for k, why in prune if k not in blocked]
        for k, why in new:
            blocked[k] = {"iteration": it, "reason": why}
        records.append({"iteration": it, "edge_path": list(key), "total_cost": float(seq.total_cost), **metrics,
                        "steps": steps, "pruned": [k for k, _ in new]})
        sequences.append(seq)
        log.info("Robot loop: iteration %d: cost %.4f, %d infeasible step(s), weakest relative %.3f, pruned %d "
                 "new edge(s) (%d blocked)", it, seq.total_cost, metrics["n_infeasible"], metrics["min_relative"],
                 len(new), len(blocked))
        for s in steps:
            if s.kind == "gripper":
                g = s.grasp
                log.info("    step %d %s: gripper %s (S=%s, S_rel=%s%s)%s", s.step, s.label,
                         "ok" if s.part_feasible else "INFEASIBLE",
                         "-" if g.score_context is None else f"{g.score_context:.3f}",
                         "-" if g.score_relative is None else f"{g.score_relative:.3f}",
                         ", rotated" if g.rotation_checked else "", "" if s.part_feasible else f" -- {g.reason}")
            else:
                t = s.tool
                log.info("    step %d %s: tool %s %s", s.step, s.label, t.tool or "(none, unchecked)",
                         "ok" if s.part_feasible else f"CONTACT {t.contact_with}")
            if s.node_after is not None:
                fmt = lambda n: "-" if n.score is None else f"{n.score:.3f}{'' if n.feasible else ' NOT graspable'}"
                log.info("        sub-assemblies: after insertion (%d parts) %s, before (%d parts) %s",
                         s.node_after.n_components, fmt(s.node_after), s.node_before.n_components, fmt(s.node_before))
        # 4. Nothing new blocked: the next optimization would return the same sequence.
        if not new:
            stop_reason = f"no new pruning after iteration {it} (the next candidate would repeat it)"
            break
    if not records:
        raise NoPathError("Robot loop produced no sequence (the first optimization already failed)")
    chosen = choose(records)
    return RobotLoopResult(records=records, sequences=sequences, blocked=blocked, chosen=chosen,
                           iterations=len(records), stop_reason=stop_reason,
                           engine_calls=getattr(analysis, "n_engine_calls", 0),
                           wall_time_s=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Pipeline entry (flat assemblies only)
# ---------------------------------------------------------------------------


def engine_tag(config: RobotConfig, gripper_yaml: Path) -> str:
    """[AUX] Identity of the engine for the grasp disk cache: YAML bytes, function copy, seed, runner.

    ENGINE SWAP POINT: reads `SOURCE_SHA256` and `COPY_VERSION` from `third_party/jms_grasp/jms_engine.py`
    with a regex. A different engine must define both (or this function must change), otherwise cached
    scores of the old engine would be reused.
    """
    from gripper_engine_robotic import ENGINE_COPY_DIR, TRANSFER_VERSION
    copy = (ENGINE_COPY_DIR / "jms_engine.py").read_text(encoding="utf-8")
    sha = re.search(r'SOURCE_SHA256 = "([0-9a-f]+)"', copy)
    ver = re.search(r'COPY_VERSION = "([^"]+)"', copy)
    h = hashlib.sha256(gripper_yaml.read_bytes())
    h.update(f"src:{sha.group(1) if sha else '?'};copy:{ver.group(1) if ver else '?'};seed:{config.engine_seed};"
             f"mode:{config.engine_mode};xfer:{TRANSFER_VERSION}".encode())
    return h.hexdigest()


def run_robot_pipeline(step_path, bom_path, output_dir, robot_config: RobotConfig | None = None,
                       preprocessing_config=None, collision_config=None,
                       optimization_config: OptimizationConfig | None = None, force_interference: bool = False,
                       max_level_components: int = 20) -> dict:
    """[MAIN] Robot-mode pipeline: Preprocessing -> Interference -> AoG -> robot loop (Optimization + grasp/tool checks) -> plan.json.

    Inputs: STEP, BOM, output folder, optional configs (None = defaults), `force_interference`, the level
    size guard. Output: the plan.json dict (also written), with the `robot` block. Flat assemblies only
    (ValueError for nested sub-assemblies, or if the AoG is incomplete).

    It builds the root level exactly like `levels.solve_level` (same caches), then the point clouds, the
    tool checker and the grasp engine (as a context manager, so the worker is always closed), computes
    S_alone for every non-stander, runs the loop and writes the plan for the chosen sequence.
    """
    from aog_generator import AoGConfig, build_aog
    from gripper_engine_robotic import GraspEngine
    from hierarchy import build_hierarchy, check_level_sizes
    from interference_identification import identify_interferences
    from levels import LevelPlan, build_level_document
    from output_robotic import write_robot_plan
    from pointclouds_robotic import component_clouds
    from preprocessing import load_assembly, load_bom, load_stander_categories, run_preprocessing_on_document
    from shapes import component_shapes
    from tool_check import ToolChecker
    import csv

    config = robot_config or RobotConfig()
    step_path, bom_path, output_dir = Path(step_path), Path(bom_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    robot_dir = output_dir / "robot"
    robot_dir.mkdir(exist_ok=True)

    # Load the inputs and refuse nested sub-assemblies.
    doc = load_assembly(step_path)
    bom_labels = load_bom(bom_path)
    bom_categories = load_stander_categories(bom_path)
    with open(bom_path, newline="", encoding="utf-8-sig") as f:
        bom_rows = list(csv.DictReader(f))
    hier = build_hierarchy(doc, bom_rows=bom_rows)
    root = hier.containers[hier.root]
    if root.child_ids:
        raise ValueError(f"Robot mode supports flat assemblies only; "
                         f"{step_path.name} has nested sub-assemblies: "
                         f"{[hier.containers[c].label for c in root.child_ids]}")
    check_level_sizes(hier, max_components=max_level_components)

    # Same calls and paths as levels.solve_level for the root, so manual-mode caches are reused.
    doc_L, compound_ids, members = build_level_document(hier, hier.root, doc, output_dir)
    prep = run_preprocessing_on_document(
        doc_L, bom_labels, preprocessing_config, collision_config=collision_config,
        bom_categories=bom_categories, step_path=step_path, bom_path=bom_path,
        level_id=hier.root, level_label=root.label, compound_ids=frozenset(compound_ids), members=members)
    interf = identify_interferences(prep, collision_config, output_dir / "interference", force=force_interference)
    aog = build_aog(interf, prep, config=AoGConfig(), cache_dir=output_dir / "aog")
    if not aog.complete:
        raise ValueError("The AoG cannot reach the fully disassembled state; robot mode needs a complete AoG")

    # Point clouds for the grasp engine and the tool checker (needs the shapes and stander categories).
    shapes = component_shapes(aog.component_ids, aog.is_stander, prep)
    clouds, keys = component_clouds(list(aog.component_ids), shapes, config.pcd_spacing_mm, config.pcd_max_points,
                                    config.pcd_seed, cache_dir=robot_dir / "clouds", labels=list(aog.component_labels))
    categories = {c: prep.stander_records[cid].category for c, cid in enumerate(aog.component_ids)
                  if cid in prep.stander_records}
    tool_checker = ToolChecker(shapes, list(aog.component_labels), list(aog.is_stander), categories,
                               np.asarray(aog.directions), config)

    # Grasp engine (worker subprocess lives inside this block), gripper analysis and the loop.
    with GraspEngine(engine_python=config.engine_python, gripper_yaml=config.gripper_yaml,
                     timeout_s=config.engine_timeout_s, seed=config.engine_seed, mode=config.engine_mode,
                     work_dir=robot_dir / "engine_work", engine_repo=config.engine_repo) as engine:
        analysis = GripperAnalysis(engine, clouds, keys, list(aog.component_labels), config,
                                   cache_dir=(robot_dir / "engine_cache") if config.disk_cache else None,
                                   engine_tag=engine_tag(config, engine.gripper_yaml))
        # Alone scores for every non-stander component up front (reported in the output).
        for c in range(aog.n_components):
            if not aog.is_stander[c]:
                analysis.alone(c)
        loop = run_robot_loop(aog, interf, prep, analysis, tool_checker, config, optimization_config,
                              optimization_cache_dir=output_dir / "optimization")
        loop.extra["engine"] = {"python": str(engine.engine_python), "gripper_yaml": str(engine.gripper_yaml),
                                "mode": engine.mode, "calls": analysis.n_engine_calls,
                                "disk_cache_hits": analysis.n_disk_hits}

    # Plan for the chosen sequence, with the robot block.
    chosen_seq = loop.sequences[loop.chosen]
    inseparable = [(interf.component_labels[i], interf.component_labels[j])
                   for i in range(len(interf.component_ids)) for j in range(i + 1, len(interf.component_ids))
                   if not interf.free[i, j, :].any()]
    plan = LevelPlan(container_id=hier.root, label=root.label, level_dir=output_dir, prep=prep,
                     interference=interf, aog=aog, sequence=chosen_seq, inseparable_pairs=inseparable)
    plan_dict = write_robot_plan(hier, plan, loop, aog, analysis, config, output_dir, assembly_name=step_path.stem)
    r = plan_dict["robot"]
    if r["feasible"]:
        log.info("Robot plan written to %s: chosen sequence (iteration %d) is fully feasible",
                 output_dir / "plan.json", r["loop"]["chosen_iteration"])
    else:
        log.warning("Robot plan written to %s: chosen sequence (iteration %d) still has %d infeasible step(s): %s",
                    output_dir / "plan.json", r["loop"]["chosen_iteration"], len(r["failing_steps"]),
                    [f"{f['step']} {f['label']}" for f in r["failing_steps"]])
    return plan_dict
