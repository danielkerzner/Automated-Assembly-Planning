"""Acceptance tests for the Optimization module (docs/optimization-spec.md, O1-O14).

Standalone runner (pytest is not installed in environmentAAP). From repo root:

    python automatic_tests/test_optimization.py                       # core group (no FreeCAD run)
    python automatic_tests/test_optimization.py --tests O1 O5         # individual tests
    python automatic_tests/test_optimization.py --groups core real    # + FreeCAD geometry and pipeline run
    python automatic_tests/test_optimization.py --groups core scale   # + O14 at N = 20

Groups:
  core   hand-built and synthetic AoGs, plus industrial_fastner from the cached AoG (tests/industrial_fastner/aog/, real
         travel table) with the real BOM (tool, group) and synthetic volumes, areas, CoGs and contact patches.
  real   O9 on FreeCAD shapes through extract_contact_patches, and O13 through optimize_sequence on industrial_fastner
         (run_preprocessing + a scratch copy of the interference cache + AoG + term tables).

The suite never writes to tests/industrial_fastner/; a final "cache" check confirms the cached files are byte-identical.
  scale  O14 at N = 20 (core runs O14 at N = 12).

Brute force (O4): every root-to-terminal edge path is enumerated; for each, the best direction per edge is found exactly
by a chain DP over the direction-dependent terms (M1 per step, M2 between consecutive steps; no other term depends on the
direction), then the path is evaluated with CostModel.evaluate. This equals enumerating all (edge, direction) paths.

Scratch output goes to automatic_tests/output/optimization_test/; results to automatic_tests/output/optimization_results.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import shutil
import sys
import time
import tracemalloc
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aog_generator as ag  # noqa: E402
import optimization as op  # noqa: E402
from optimization import OptimizationConfig, TermTables  # noqa: E402
from test_aog import AXES, fastener_inputs_from_cache, make_inputs, random_inputs  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
SCRATCH = OUT_DIR / "optimization_test"
FASTENER_DIR = REPO_ROOT / "tests" / "industrial_fastner"
FASTENER_STEP = REPO_ROOT / "data" / "industrial_fastner" / "industrial_fastner.stp"
FASTENER_BOM = REPO_ROOT / "data" / "industrial_fastner" / "generated_bom.csv"

PX, NX, PY, NY, PZ, NZ = range(6)  # indices into AXES and the default gravity_dirs
FLAGS = [op.FLAG_OF[t] for t in op.TERMS]
TOL = 1e-9


# ---------------------------------------------------------------------------
# Tiny test harness (same shape as test_aog.py)
# ---------------------------------------------------------------------------


class Recorder:
    def __init__(self):
        self.results: dict[str, dict] = {}
        self._current: str | None = None

    def start(self, test_id: str):
        self._current = test_id
        self.results[test_id] = {"passed": True, "checks": [], "notes": []}

    def check(self, ok: bool, label: str):
        entry = self.results[self._current]
        entry["checks"].append({"label": label, "passed": bool(ok)})
        if not ok:
            entry["passed"] = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}", flush=True)

    def note(self, text: str):
        self.results[self._current]["notes"].append(text)
        print(f"  note: {text}", flush=True)

    def error(self, exc: BaseException):
        entry = self.results[self._current]
        entry["passed"] = False
        entry["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"  [ERROR] {entry['error']}", flush=True)
        traceback.print_exc()


CACHED_FILES = ["interference/interference.json", "interference/interference.npz", "aog/aog.json", "aog/aog.npz"]


def cache_snapshot() -> dict[str, str | None]:
    """SHA-256 of the cached industrial_fastner files the suite reads (None if absent)."""
    import hashlib
    out = {}
    for name in CACHED_FILES:
        path = FASTENER_DIR / name
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return out


def fresh_dir(name: str) -> Path:
    path = SCRATCH / name
    shutil.rmtree(path, ignore_errors=True)
    return path


def raises(fn, needle: str) -> bool:
    try:
        fn()
    except ValueError as exc:
        return needle in str(exc)
    return False


def only(*flags, **extra) -> OptimizationConfig:
    """Config with only the given flags on."""
    return OptimizationConfig(**{f: f in flags for f in FLAGS}, **extra)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_tables(aog, volume=None, area=None, contact=None, cog=None, tool=None, group=None, pitch=None,
                patches=None, has_tool=True, has_group=True) -> TermTables:
    N = aog.n_components
    return TermTables(
        component_ids=list(aog.component_ids),
        volume=np.ones(N) if volume is None else np.asarray(volume, float),
        area=np.ones(N) if area is None else np.asarray(area, float),
        contact=np.zeros((N, N), bool) if contact is None else np.asarray(contact, bool),
        cog=np.zeros((N, 3)) if cog is None else np.asarray(cog, float),
        tool=tool if tool is not None else [("T" if s else None) for s in aog.is_stander],
        group=group if group is not None else [f"g{c}" for c in range(N)],
        pitch=pitch if pitch is not None else [None] * N,
        has_tool_column=has_tool,
        has_group_column=has_group,
        patches={} if patches is None else patches,
    )


def path_for(aog, assembly_order, removal_dirs):
    """Walk-order (edges, removal dirs) for an assembly order of components with their removal directions."""
    edges, v = [], aog.root
    for c in reversed(assembly_order):
        es = [e for e in aog.out_edges(v) if int(aog.edge_comp[e]) == c]
        if not es:
            raise ValueError(f"component {c} cannot be removed at node {v}")
        edges.append(es[0])
        v = int(aog.edge_dst[es[0]])
    return np.array(edges), np.array(list(reversed(removal_dirs)))


def all_free_aog(n, n_standers=0, stander_dir_axis=PZ):
    free = np.ones((n, n, 6), dtype=bool)
    so = {s: AXES[stander_dir_axis] for s in range(n, n + n_standers)}
    return ag.generate_aog(make_inputs(free, stander_options=so))


def add_patch(patches, i, j, tri, normal_into_i):
    """Store triangles between i and j with their normal pointing from j into i (key order i < j)."""
    tri = np.asarray(tri, float).reshape(-1, 3, 3)
    normal = np.tile(np.asarray(normal_into_i, float), (len(tri), 1)) if np.ndim(normal_into_i) == 1 \
        else np.asarray(normal_into_i, float)
    key, sign = ((i, j), 1.0) if i < j else ((j, i), -1.0)
    old_t, old_n = patches.get(key, (np.zeros((0, 3, 3)), np.zeros((0, 3))))
    patches[key] = (np.concatenate([old_t, tri]), np.concatenate([old_n, sign * normal]))


def rect(origin, u, v):
    """Two triangles of the parallelogram origin + [0,1] u + [0,1] v."""
    o, u, v = (np.asarray(x, float) for x in (origin, u, v))
    return [[o, o + u, o + u + v], [o, o + u + v, o + v]]


def cylinder_strip(x0, x1, r, n=24):
    """Triangles of a cylinder around the x axis between x0 and x1, normals pointing to the axis (from hole into pin)."""
    tris, normals = [], []
    th = np.linspace(0, 2 * np.pi, n + 1)
    for a, b in zip(th[:-1], th[1:]):
        p = [np.array([x, r * np.cos(t), r * np.sin(t)]) for x in (x0, x1) for t in (a, b)]
        m = (a + b) / 2
        inward = -np.array([0.0, np.cos(m), np.sin(m)])
        tris += [[p[0], p[2], p[3]], [p[0], p[3], p[1]]]
        normals += [inward, inward]
    return np.array(tris), np.array(normals)


# ---------------------------------------------------------------------------
# industrial_fastner (cached AoG, real travel table)
# ---------------------------------------------------------------------------

_STATE: dict = {}


def bom_columns():
    columns, rows, _ = op._read_bom(FASTENER_BOM)
    return columns, rows


def fastener():
    """(aog, tables): cached AoG with real travel; real BOM tool/group; synthetic geometry (seeded)."""
    if "fastener" not in _STATE:
        aog = ag.load_aog(FASTENER_DIR / "aog")
        if aog is None:
            raise RuntimeError("tests/industrial_fastner/aog has no cached AoG; run manual_tests/test_aog_manual.py first")
        inputs = fastener_inputs_from_cache()
        if list(inputs.component_ids) != list(aog.component_ids):
            raise RuntimeError("cached AoG and cached interference disagree on the component order")
        N = aog.n_components
        rng = np.random.default_rng(1)
        cog = rng.uniform(0, 10, size=(N, 3))
        patches = {}
        for i in range(N):
            for j in range(i + 1, N):
                if inputs.contact[i, j]:
                    c = cog[i]
                    shift = 5.0 if rng.random() < 0.5 else 0.0  # half of the patches miss the CoG
                    o = [c[0] - 3 + shift, c[1] - 3 + shift, 0.0]
                    add_patch(patches, i, j, rect(o, [6, 0, 0], [0, 6, 0]), [0, 0, 1.0])
        columns, rows = bom_columns()
        tables = make_tables(
            aog, volume=rng.uniform(1, 100, N), area=rng.uniform(1, 100, N), contact=inputs.contact, cog=cog,
            tool=[rows[label].get("tool") or None for label in aog.component_labels],
            group=[rows[label].get("group") or None for label in aog.component_labels],
            has_tool="tool" in columns, has_group="group" in columns, patches=patches)
        _STATE["fastener"] = (aog, tables)
    return _STATE["fastener"]


def edge_paths(aog, model):
    """All root-to-terminal edge paths over live edges."""
    out = []

    def rec(v, es):
        if v == aog.terminal:
            out.append(es)
            return
        for e in model.lv_edges[model.lv_ptr[v]:model.lv_ptr[v + 1]]:
            rec(int(aog.edge_dst[e]), es + [int(e)])

    rec(aog.root, [])
    return out


def best_directions(model, edges):
    """Exact best removal direction per edge for a fixed edge path (chain DP over M1 + M2)."""
    ks = [model._edge_ks(e) for e in edges]
    comp = [int(model.edge_comp[e]) for e in edges]
    D = model.c_travel * model.travel[comp[0], ks[0]]
    back = []
    for s in range(1, len(edges)):
        pair = model.c_dir * model._dir_levels(np.repeat(ks[s - 1], len(ks[s])), np.tile(ks[s], len(ks[s - 1])))
        tot = D[:, None] + pair.reshape(len(ks[s - 1]), len(ks[s]))
        back.append(np.argmin(tot, axis=0))
        D = tot.min(axis=0) + model.c_travel * model.travel[comp[s], ks[s]]
    j = int(np.argmin(D))
    choice = [j]
    for b in reversed(back):
        j = int(b[j])
        choice.append(j)
    choice.reverse()
    return np.array([int(k[c]) for k, c in zip(ks, choice)])


def brute_force(aog, tables, config):
    model = op.CostModel(aog, tables, config)
    best, n = np.inf, 0
    for es in edge_paths(aog, model):
        es = np.array(es)
        best = min(best, model.evaluate(es, best_directions(model, es))["total_cost"])
        n += 1
    return best, n


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_o1(rec, args):
    aog, tables = fastener()
    no_tool = replace(tables, has_tool_column=False)
    no_group = replace(tables, has_group_column=False)
    blank_tool = replace(tables, tool=[None] * aog.n_components)
    blank_group = replace(tables, group=[None if c == 0 else g for c, g in enumerate(tables.group)])
    rec.check(raises(lambda: op.validate_config(OptimizationConfig(**{f: False for f in FLAGS}), True, True),
                     "At least one"), "no flag True raises")
    rec.check(raises(lambda: op.validate_config(OptimizationConfig(use_stability=False), True, True),
                     "use_gravity_change"), "gravity change without stability raises")
    rec.check(raises(lambda: op.CostModel(aog, no_tool, OptimizationConfig(use_tool_change=True)), "'tool' column"),
              "tool change without a 'tool' column raises, naming the column")
    rec.check(raises(lambda: op.CostModel(aog, no_group, OptimizationConfig()), "'group' column"),
              "grouping without a 'group' column raises, naming the column")
    standers = [aog.component_labels[c] for c in np.flatnonzero(aog.is_stander)]
    rec.check(raises(lambda: op.CostModel(aog, blank_tool, OptimizationConfig(use_tool_change=True)), standers[0]),
              f"stander with empty tool raises, naming the part ({standers[0]})")
    rec.check(raises(lambda: op.CostModel(aog, blank_group, OptimizationConfig()), aog.component_labels[0]),
              f"part with empty group raises, naming the part ({aog.component_labels[0]})")
    ok = True
    for t in (no_tool, blank_tool):
        try:
            op.CostModel(aog, t, OptimizationConfig(use_tool_change=False))
        except ValueError:
            ok = False
    for t in (no_group, blank_group):
        try:
            op.CostModel(aog, t, OptimizationConfig(use_grouping=False))
        except ValueError:
            ok = False
    rec.check(ok, "the same tables are accepted when the flag is off")
    columns, _ = bom_columns()
    rec.check({"tool", "group"} <= set(columns), f"industrial_fastner bom.csv has tool and group columns: {columns}")


def test_o2(rec, args):
    w = op.resolve_weights(OptimizationConfig(use_tool_change=True))
    rec.check(abs(sum(w.values()) - 1) < 1e-12, f"all on: sum 1 ({sum(w.values()):.12f})")
    rec.check(all(abs(w[t] - op.DOMINANCE[r].sum() / 49) < 1e-12 for r, t in enumerate(op.TERMS)),
              "all on: weights = row sums / 49")
    off = op.resolve_weights(OptimizationConfig(use_tool_change=False, use_grouping=False))
    rec.check(abs(sum(off.values()) - 1) < 1e-12 and "tool_change" not in off and "grouping" not in off,
              "M3, M7 off: sum 1, terms dropped")
    rec.check(all(abs(off[a] / off[b] - w[a] / w[b]) < 1e-12 for a in off for b in off), "ratios kept")
    over = op.resolve_weights(OptimizationConfig(use_tool_change=True, weights={"travel": 10}))
    raw = {t: (10 if t == "travel" else op.DEFAULT_WEIGHTS[t]) for t in op.TERMS}
    rec.check(abs(sum(over.values()) - 1) < 1e-12 and all(abs(over[t] - raw[t] / sum(raw.values())) < 1e-12
                                                           for t in op.TERMS), "override then renormalize")
    rec.check(raises(lambda: op.resolve_weights(OptimizationConfig(weights={"nope": 1})), "Unknown"),
              "unknown term raises")


def test_o3(rec, args):
    cases = [("industrial_fastner", *fastener())]
    rng = np.random.default_rng(5)
    while len(cases) < 5:
        inputs = random_inputs(rng, int(rng.integers(8, 13)))
        aog = ag.generate_aog(inputs)
        if aog.complete:
            N = aog.n_components
            cases.append((f"synthetic N={N}", aog, make_tables(aog, volume=rng.uniform(1, 50, N),
                                                                 area=rng.uniform(1, 50, N))))
    for name, aog, tables in cases:
        cfg = only("use_travel", "use_size_contacts", use_cost_to_go=False)
        model = op.CostModel(aog, tables, cfg)
        dp = model.evaluate(*model.dp_optimum())["total_cost"]
        res = op.optimize(aog, tables, cfg)
        rec.check(abs(res.total_cost - dp) < 1e-9,
                  f"{name}: ACO {res.total_cost:.6f} = DP {dp:.6f} (greedy {res.history['greedy_cost']:.6f})")


def test_o4(rec, args):
    aog, tables = fastener()
    configs = [(f[4:], only(f)) for f in FLAGS if f not in ("use_gravity_change", "use_travel")]
    configs += [("default (all flags, tool change on)", OptimizationConfig()),
                ("default without tool change", OptimizationConfig(use_tool_change=False)),
                ("stability + gravity_change", only("use_stability", "use_gravity_change")),
                ("travel + size_contacts", only("use_travel", "use_size_contacts"))]
    seeds = range(3)
    for name, cfg in configs:
        bf, n = brute_force(aog, tables, cfg)
        costs = [op.optimize(aog, tables, replace(cfg, seed=s)).total_cost for s in seeds]
        rec.check(all(abs(c - bf) < 1e-9 for c in costs),
                  f"{name}: ACO = brute force {bf:.6f} on seeds {list(seeds)} ({n} edge paths; got "
                  f"{', '.join(f'{c:.6f}' for c in costs)})")
    # Known limitation, reported only.
    bf, _ = brute_force(aog, tables, only("use_travel"))
    for ctg in (False, True):
        costs = [op.optimize(aog, tables, only("use_travel", use_cost_to_go=ctg, seed=s)).total_cost for s in range(10)]
        hits = sum(abs(c - bf) < 1e-9 for c in costs)
        rec.note(f"travel only (not asserted), use_cost_to_go={ctg}: optimum {bf:.6f} found on {hits}/10 seeds")


def test_o5(rec, args):
    aog = all_free_aog(3)
    tables = make_tables(aog)
    for ratio in (0.25, 0.6):
        model = op.CostModel(aog, tables, only("use_direction_change", dir_change_axis_ratio=ratio))
        cases = [([PX, PX, PX], 0.0), ([PX, PX, NX], ratio), ([PX, NX, PY], ratio + 1), ([PX, PY, PZ], 2.0),
                 ([PZ, NZ, PZ], 2 * ratio)]
        for dirs, expect in cases:
            ev = model.evaluate(*path_for(aog, [0, 1, 2], dirs))
            names = [["+x", "-x", "+y", "-y", "+z", "-z"][k] for k in dirs]
            rec.check(abs(ev["raw"]["direction_change"] - expect) < TOL
                      and abs(ev["normalized"]["direction_change"] - expect / 2) < TOL,
                      f"ratio {ratio}, removal {names}: raw {ev['raw']['direction_change']:.3f} = {expect:.3f}, "
                      f"normalized / (N - 1)")


def test_o6(rec, args):
    aog = all_free_aog(4)
    tables = make_tables(aog, group=["A", "A", "B", "C"])
    model = op.CostModel(aog, tables, only("use_grouping"))
    cases = [([0, 2, 1, 3], 1, "leave A unfinished (c1 missing) -> 1; B, A finished -> 0"),
             ([0, 1, 2, 3], 0, "A finished before leaving, singletons free"),
             ([2, 0, 3, 1], 1, "leave singleton B 0; leave A unfinished 1; leave singleton C 0"),
             ([2, 3, 0, 1], 0, "singletons then A")]
    for order, expect, label in cases:
        ev = model.evaluate(*path_for(aog, order, [PX] * 4))
        rec.check(ev["raw"]["grouping"] == expect and abs(ev["normalized"]["grouping"] - expect / 3) < TOL,
                  f"order {order}: {ev['raw']['grouping']:.0f} = {expect} ({label})")


def test_o7(rec, args):
    aog = all_free_aog(2, n_standers=3)
    tables = make_tables(aog, tool=[None, None, "A", "B", "A"])
    model = op.CostModel(aog, tables, only("use_tool_change"))
    dirs = {0: PX, 1: PX, 2: PZ, 3: PZ, 4: PZ}
    cases = [([2, 0, 3, 1, 4], 2, "A, B, A with movable parts between: 2 changes"),
             ([2, 4, 0, 1, 3], 1, "A, A, B: 1 change"),
             ([0, 2, 4, 3, 1], 1, "movable first, then A, A, B: 1 change"),
             ([3, 0, 1, 2, 4], 1, "B, A, A: 1 change")]
    for order, expect, label in cases:
        ev = model.evaluate(*path_for(aog, order, [dirs[c] for c in order]))
        per_step = ev["steps"]["tool_change"]
        movable_zero = all(per_step[s] == 0 for s, c in enumerate(order) if c < 2)
        rec.check(ev["raw"]["tool_change"] == expect and abs(ev["normalized"]["tool_change"] - expect / 2) < TOL
                  and movable_zero, f"order {order}: {ev['raw']['tool_change']:.0f} = {expect} ({label})")


def test_o8(rec, args):
    aog = all_free_aog(3)
    vol, area = np.array([100.0, 1.0, 10.0]), np.array([60.0, 2.0, 20.0])
    contact = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], bool)
    tables = make_tables(aog, volume=vol, area=area, contact=contact)
    cfg = only("use_size_contacts", size_alpha=0.5, size_beta=0.3, size_gamma=0.2)
    model = op.CostModel(aog, tables, cfg)
    nb = contact.sum(1).astype(float)
    S = 0.5 * vol / vol.mean() + 0.3 * area / area.mean() + 0.2 * nb / nb.mean()
    rec.check(np.allclose(model.size, S), f"S_i = {np.round(S, 4).tolist()}")

    def raw(order):
        return model.evaluate(*path_for(aog, order, [PX] * 3))["raw"]["size_contacts"]

    big_first, big_last = raw([0, 2, 1]), raw([1, 2, 0])
    expect = sum(S[c] * t / 3 for t, c in enumerate([0, 2, 1], start=1))
    rec.check(abs(big_first - expect) < TOL, f"path value {big_first:.4f} = sum S_i t / N {expect:.4f}")
    rec.check(big_last > big_first, f"large part last costs more ({big_last:.4f} > {big_first:.4f})")
    res = op.optimize(aog, tables, cfg)
    rec.check([s["component"] for s in res.steps] == [0, 2, 1], "ACO inserts large -> small "
              f"({[s['component'] for s in res.steps]})")


def o9_cases():
    """(name, n parts, free (n, n, 6), cog (n, 3), pair -> (tri, normal from j into i) list, part i under test,
    expectations {(support_region, gravity index): class})."""
    def free_except(n, blocked):  # blocked: (i, j) -> list of removal dirs of i blocked by j
        f = np.ones((n, n, 6), bool)
        for (i, j), ks in blocked.items():
            for k in ks:
                f[i, j, k] = False
                f[j, i, k ^ 1] = False  # AXES_NEG = k ^ 1
        return f

    cases = []
    # Overhanging block (1) on a base (0): patch x in [5, 10], CoG x = 15.
    cases.append(("overhanging block", 2, free_except(2, {(1, 0): [NZ]}), [[5, 5, 5], [15, 5, 15]],
                  [(1, 0, rect([5, 0, 10], [5, 0, 0], [0, 10, 0]), [0, 0, 1])], 1,
                  {("hull", NZ): 1, ("union", NZ): 1, ("hull", PZ): 1}))
    cases.append(("block on a base (not overhanging)", 2, free_except(2, {(1, 0): [NZ]}), [[5, 5, 5], [5, 5, 15]],
                  [(1, 0, rect([0, 0, 10], [10, 0, 0], [0, 10, 0]), [0, 0, 1])], 1,
                  {("hull", NZ): 2, ("union", NZ): 2, ("hull", PZ): 1}))
    # Block (1) in a U-shaped container (0), open towards +z.
    u = [(1, 0, rect([0, 0, 0], [10, 0, 0], [0, 10, 0]), [0, 0, 1]),
         (1, 0, rect([0, 0, 0], [0, 10, 0], [0, 0, 10]), [1, 0, 0]),
         (1, 0, rect([10, 0, 0], [0, 10, 0], [0, 0, 10]), [-1, 0, 0]),
         (1, 0, rect([0, 0, 0], [10, 0, 0], [0, 0, 10]), [0, 1, 0]),
         (1, 0, rect([0, 10, 0], [10, 0, 0], [0, 0, 10]), [0, -1, 0])]
    cases.append(("block in U-shaped container", 2, free_except(2, {(1, 0): [PX, NX, PY, NY, NZ]}),
                  [[5, 5, 5], [5, 5, 5]], u, 1,
                  {("hull", NZ): 2, ("union", NZ): 2, ("hull", PX): 2, ("union", PX): 2, ("hull", PZ): 1,
                   ("union", PZ): 1}))
    # Pin (1) along x through one plate (0) spanning x in [10, 20].
    tri, nrm = cylinder_strip(10, 20, 2)
    cases.append(("pin through one plate", 2, free_except(2, {(1, 0): [PY, NY, PZ, NZ]}), [[15, 0, 0], [15, 0, 0]],
                  [(1, 0, tri, nrm)], 1,
                  {("hull", NZ): 2, ("union", NZ): 2, ("hull", PX): 1, ("union", PX): 1}))
    # Block (2) on two rails (0, 1): patches x in [0, 2] and [18, 20], CoG x = 10.
    cases.append(("block on two rails", 3, free_except(3, {(2, 0): [NZ], (2, 1): [NZ]}),
                  [[1, 5, 5], [19, 5, 5], [10, 5, 15]],
                  [(2, 0, rect([0, 0, 10], [2, 0, 0], [0, 10, 0]), [0, 0, 1]),
                   (2, 1, rect([18, 0, 10], [2, 0, 0], [0, 10, 0]), [0, 0, 1])], 2,
                  {("hull", NZ): 2, ("union", NZ): 1, ("hull", PZ): 1}))
    # Pin (2) through two separated plates (0: x in [0, 5], 1: x in [25, 30]), CoG x = 15.
    t0, n0 = cylinder_strip(0, 5, 2)
    t1, n1 = cylinder_strip(25, 30, 2)
    cases.append(("pin through two separated plates", 3,
                  free_except(3, {(2, 0): [PY, NY, PZ, NZ], (2, 1): [PY, NY, PZ, NZ]}),
                  [[2.5, 0, 0], [27.5, 0, 0], [15, 0, 0]], [(2, 0, t0, n0), (2, 1, t1, n1)], 2,
                  {("hull", NZ): 2, ("union", NZ): 1, ("hull", PX): 1, ("union", PX): 1, ("hull", PY): 2}))
    return cases


def o9_check(rec, name, n, free, cog, patches, part, expect, prefix=""):
    aog = ag.generate_aog(make_inputs(free))
    contact = np.zeros((n, n), bool)
    for i, j in patches:
        contact[i, j] = contact[j, i] = True
    edge = next(e for e in aog.out_edges(aog.root) if int(aog.edge_comp[e]) == part)
    for region in ("hull", "union"):
        tables = make_tables(aog, cog=cog, contact=contact, patches=patches)
        model = op.CostModel(aog, tables, only("use_stability", support_region=region))
        cls = np.where(model.stability_classes(edge), 1, 2)
        for (reg, g), want in expect.items():
            if reg == region:
                gname = ["+x", "-x", "+y", "-y", "+z", "-z"][g]
                rec.check(cls[g] == want, f"{prefix}{name}, {region}, g = {gname}: class {cls[g]} (expected {want})")


def test_o9(rec, args):
    for name, n, free, cog, pairs, part, expect in o9_cases():
        patches = {}
        for i, j, tri, normal in pairs:
            add_patch(patches, i, j, tri, normal)
        o9_check(rec, name, n, free, cog, patches, part, expect)
    # Degenerate hull: line contact (cylinder lying on a plane) and a CoG on / off the line.
    line = {}
    add_patch(line, 1, 0, [[[0, -0.001, 0], [10, -0.001, 0], [10, 0.001, 0]], [[0, -0.001, 0], [10, 0.001, 0],
                                                                            [0, 0.001, 0]]], [0, 0, 1])
    free = np.ones((2, 2, 6), bool)
    free[1, 0, NZ] = free[0, 1, PZ] = False
    o9_check(rec, "line contact, CoG on the line", 2, free, [[5, 0, -5], [5, 0.005, 2]], line, 1,
             {("hull", NZ): 2})
    o9_check(rec, "line contact, CoG 1 mm off the line", 2, free, [[5, 0, -5], [5, 1.0, 2]], line, 1,
             {("hull", NZ): 1})
    if "real" in args.groups:
        test_o9_geometry(rec)
    else:
        rec.note("FreeCAD geometry variant skipped (run with --groups real)")


def test_o9_geometry(rec):
    """The O9 configurations as FreeCAD shapes, patches through extract_contact_patches."""
    import freecad  # noqa: F401
    import FreeCAD
    import Part

    V = FreeCAD.Vector

    def box(x, y, z, dx, dy, dz):
        return Part.makeBox(dx, dy, dz, V(x, y, z))

    def pin(x0, x1, r=2.0):
        return Part.makeCylinder(r, x1 - x0, V(x0, 0, 0), V(1, 0, 0))

    def plate(x0, x1, r=2.0):
        return box(x0, -10, -10, x1 - x0, 20, 20).cut(pin(x0 - 1, x1 + 1, r))

    u_container = box(-2, -2, -2, 14, 14, 12).cut(box(0, 0, 0, 10, 10, 20))
    shapes_cases = {
        "overhanging block": ([box(0, 0, 0, 10, 10, 10), box(5, 0, 10, 20, 10, 10)], 1),
        "block on a base (not overhanging)": ([box(0, 0, 0, 10, 10, 10), box(0, 0, 10, 10, 10, 10)], 1),
        "block in U-shaped container": ([u_container, box(0, 0, 0, 10, 10, 10)], 1),
        "pin through one plate": ([plate(10, 20), pin(0, 30)], 1),
        "block on two rails": ([box(0, 0, 0, 2, 10, 10), box(18, 0, 0, 2, 10, 10), box(0, 0, 10, 20, 10, 10)], 2),
        "pin through two separated plates": ([plate(0, 5), plate(25, 30), pin(0, 30)], 2),
    }
    for name, n, free, _, pairs, part, expect in o9_cases():
        shapes, part2 = shapes_cases[name]
        assert part2 == part
        patches = {}
        for i, j, _, _ in pairs:
            if (min(i, j), max(i, j)) not in patches:
                tri, normal = op.extract_contact_patches(shapes[i], shapes[j], 0.1)
                add_patch(patches, i, j, tri, normal)
                rec.note(f"{name}: patches {i}-{j}: {len(tri)} triangles, {op._area(tri):.1f} mm^2")
        cog = [op._center_of_mass(s) for s in shapes]
        o9_check(rec, name, n, free, cog, patches, part, expect, prefix="FreeCAD: ")


def test_o10(rec, args):
    aog, tables = fastener()
    model = op.CostModel(aog, tables, OptimizationConfig())
    rng = np.random.default_rng(3)
    G = len(model.gravity)
    worst = 0.0
    for trial in range(30):
        n = int(rng.integers(2, 7))
        cls = rng.random((n, G)) < 0.5
        cls[0] = False
        _, cost = model._gravity_viterbi(cls)
        best = min(sum(model.c_stab * cls[s, gs[s]] for s in range(n))
                   + sum(model.c_grav * model.switch[gs[s - 1], gs[s]] for s in range(1, n))
                   for gs in itertools.product(range(G), repeat=n))
        worst = max(worst, abs(best - cost))
    rec.check(worst < 1e-12, f"Viterbi = enumeration over 6^n assignments, 30 random tables, n <= 6 (max diff {worst:.1e})")


def test_o11(rec, args):
    aog, tables = fastener()
    dead = [int(v) for v in np.flatnonzero(~aog.live)]
    labels = [sorted(aog.component_labels[c] for c in aog.state_components(v)) for v in aog.dead_ends]
    rec.check(len(aog.dead_ends) >= 1, f"industrial_fastner has dead end(s): {labels}")
    for seed in range(5):
        res = op.optimize(aog, tables, OptimizationConfig(seed=seed))
        nodes = [aog.root] + [int(aog.edge_dst[e]) for e in res.edge_path]
        rec.check(all(aog.live[nodes]) and not set(nodes) & set(dead), f"seed {seed}: best path visits only live nodes")


def test_o12(rec, args):
    aog, tables = fastener()
    res = op.optimize(aog, tables, OptimizationConfig())
    rec.check([s["edge"] for s in res.steps] == list(reversed(res.edge_path)), "steps = AoG path reversed")
    rec.check(all(s["direction_index"] == int(aog.neg[s["removal_direction_index"]]) for s in res.steps),
              "insertion direction = neg[removal direction]")
    rec.check(all(np.allclose(s["direction"], -aog.directions[s["removal_direction_index"]]) for s in res.steps),
              "insertion vector = -removal vector")
    pops = [bin(int(aog.node_mask[aog.edge_src[s["edge"]]])).count("1") for s in res.steps]
    rec.check(pops == [s["step"] for s in res.steps], f"t = popcount(S): {pops}")
    rec.check(all(np.allclose(s["start_offset"], s["travel"] * aog.directions[s["removal_direction_index"]])
                  for s in res.steps), "start offset = travel x removal direction")
    rec.check(sorted(s["component"] for s in res.steps) == list(range(aog.n_components)), "every part exactly once")


def test_o13(rec, args):
    aog, tables = fastener()
    a = op.optimize(aog, tables, OptimizationConfig(seed=4))
    b = op.optimize(aog, tables, OptimizationConfig(seed=4))
    rec.check(a.edge_path == b.edge_path and a.removal_directions == b.removal_directions
              and a.history["best_cost"] == b.history["best_cost"], "same config and seed: identical results")
    rec.check(a.config_hash != op.compute_config_hash(aog, tables, OptimizationConfig(seed=4, weights={"travel": 1})),
              "changing a weight changes the result key")
    ir = SimpleNamespace(cache_key="interference", contact_threshold=0.1)
    bom = FASTENER_BOM.read_bytes()
    base = op.tables_cache_key(aog, ir, bom, OptimizationConfig())
    rec.check(base == op.tables_cache_key(aog, ir, bom, OptimizationConfig(weights={"travel": 1}, use_grouping=False)),
              "changing weights or flags keeps the term-table key")
    rec.check(base != op.tables_cache_key(aog, ir, bom, OptimizationConfig(gravity_dirs=((0, 0, -1), (1, 0, 0)))),
              "changing the gravity directions changes the term-table key")
    if "real" in args.groups:
        test_o13_pipeline(rec)
    else:
        rec.note("pipeline caching variant skipped (run with --groups real)")


def test_o13_pipeline(rec):
    import freecad  # noqa: F401
    from collision import CollisionConfig
    from interference_identification import identify_interferences
    from preprocessing import run_preprocessing

    out = fresh_dir("pipeline")
    # Work on a copy of the interference cache: the suite must never modify tests/industrial_fastner/.
    shutil.copytree(FASTENER_DIR / "interference", out / "interference")
    prep = run_preprocessing(FASTENER_STEP, FASTENER_BOM)
    ir = identify_interferences(prep, CollisionConfig(step=2.0), out / "interference")
    aog = ag.build_aog(ir, prep, cache_dir=out / "aog")
    cfg = OptimizationConfig(use_tool_change=True)
    t0 = time.perf_counter()
    r1 = op.optimize_sequence(aog, ir, prep, cfg, cache_dir=out / "optimization")
    t1 = time.perf_counter() - t0
    tables_file = out / "optimization" / op.TABLES_NAME
    stamp = tables_file.stat().st_mtime_ns
    r2 = op.optimize_sequence(aog, ir, prep, cfg, cache_dir=out / "optimization")
    rec.check(r2.edge_path == r1.edge_path and r2.total_cost == r1.total_cost, "second run loads the cached result")
    r3 = op.optimize_sequence(aog, ir, prep, replace(cfg, weights={"travel": 0.2}), cache_dir=out / "optimization")
    rec.check(tables_file.stat().st_mtime_ns == stamp and r3.config_hash != r1.config_hash,
              "weight change: new result, term tables reused")
    op.optimize_sequence(aog, ir, prep, replace(cfg, gravity_dirs=((0, 0, -1), (0, 0, 1))), cache_dir=out / "optimization")
    rec.check(tables_file.stat().st_mtime_ns != stamp, "gravity change: term tables recomputed")
    seq = [(s["component_label"][len("industrial_fastner_"):], s["stability_class"]) for s in r1.steps]
    rec.note(f"industrial_fastner (real geometry): total {r1.total_cost:.4f} in {t1:.1f} s; sequence {seq}")


def _scale_case(rec, N, budget_s=60.0):
    rng = np.random.default_rng(N)
    while True:
        inputs = random_inputs(rng, N)
        aog = ag.generate_aog(inputs)
        if aog.complete and aog.n_components == N:
            break
    tables = make_tables(aog, volume=rng.uniform(1, 100, N), area=rng.uniform(1, 100, N),
                         contact=np.zeros((N, N), bool) if inputs.contact is None else inputs.contact,
                         cog=rng.uniform(0, 10, (N, 3)), tool=["T" if s else None for s in aog.is_stander],
                         group=[f"g{c % 4}" for c in range(N)])
    tracemalloc.start()
    t0 = time.perf_counter()
    res = op.optimize(aog, tables, OptimizationConfig(use_tool_change=True))
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    moves = op.CostModel(aog, tables, OptimizationConfig()).n_moves
    estimate = memory_estimate(aog.n_nodes, aog.n_edges, moves)
    rec.note(f"N={N}: V={aog.n_nodes}, E={aog.n_edges}, moves={moves}, {res.history['iterations']} iterations, "
             f"{elapsed:.1f} s, peak traced {peak / 1e6:.1f} MB (estimate {estimate / 1e6:.1f} MB), "
             f"cost {res.total_cost:.4f} (greedy {res.history['greedy_cost']:.4f})")
    rec.check(elapsed < budget_s and peak < estimate,
              f"N={N}: {elapsed:.1f} s < {budget_s:.0f} s and {peak / 1e6:.1f} MB < {estimate / 1e6:.1f} MB")


def memory_estimate(V, E, moves):
    """Spec section 7: 48 bytes per edge + 16 per move + 32 per node + 32 MB."""
    return 48 * E + 16 * moves + 32 * V + 32e6


def test_o14(rec, args):
    _scale_case(rec, 12)
    if "scale" in args.groups:
        _scale_case(rec, 20)
    else:
        rec.note("N = 20 skipped (run with --groups scale)")


def _stander_rec(category, partner=None, nut_rule=None):
    return SimpleNamespace(category=category, partner=partner, nut_rule=nut_rule)


def test_o15(rec, args):
    """Thread pitch and rotation counts (docs/stander_processing-spec.md section 12). Synthetic,
    no FreeCAD: _apply_turns is pure Python over step dicts + StanderRecord-like stand-ins."""
    ids = ["A", "B", "S", "N", "M", "N2", "O", "S2"]
    aog = SimpleNamespace(component_ids=ids, component_labels=list(ids))
    prep = SimpleNamespace(stander_records={
        "S": _stander_rec(1),
        "N": _stander_rec(2, partner="S", nut_rule="locked"),
        "N2": _stander_rec(2, partner="M", nut_rule="free"),
        "O": _stander_rec(3),
        "S2": _stander_rec(1),
    })
    tables = SimpleNamespace(pitch=[None, None, 1.5, 1.0, None, 2.0, None, None])

    # Screw with pitch 1.5 and travel 18 -> 12.0 turns; nut after a stander-screw partner rotates
    # itself; category 3 never gets turns; missing pitch (S2) gives None.
    steps = [
        {"component": 0, "component_label": "A", "travel": 10.0},
        {"component": 2, "component_label": "S", "travel": 18.0},
        {"component": 3, "component_label": "N", "travel": 5.0},
        {"component": 6, "component_label": "O", "travel": 3.0},
        {"component": 7, "component_label": "S2", "travel": 4.0},
    ]
    op._apply_turns(aog, prep, tables, steps)
    by_label = {s["component_label"]: s for s in steps}
    rec.check(by_label["S"]["turns"] == 12.0, f"screw S: 18/1.5 == 12.0 turns (got {by_label['S']['turns']})")
    rec.check(by_label["N"]["turns"] == 5.0, f"nut N after its stander-screw partner: rotates itself (got {by_label['N']['turns']})")
    rec.check(by_label["O"]["turns"] is None, "category-3 stander O never gets turns")
    rec.check(by_label["S2"]["turns"] is None, "screw S2 with missing pitch gives turns=None")
    rec.check(by_label["S2"]["turns_pitch"] is None and by_label["S2"]["turns_note"] is None,
              "missing-pitch step has no pitch/note either")

    # Nut assembled before its stander-screw partner (locked, screw added after): the screw keeps
    # its own row-1 turns; the nut's "partner rotates" override is skipped (partner is a screw).
    steps2 = [
        {"component": 3, "component_label": "N", "travel": 5.0},
        {"component": 2, "component_label": "S", "travel": 18.0},
    ]
    op._apply_turns(aog, prep, tables, steps2)
    rec.check(steps2[0]["turns"] is None, "nut N before screw partner S: N itself does not rotate")
    rec.check(steps2[1]["turns"] == 12.0, "screw S keeps its own turns regardless of N's position")

    # Nut before a movable partner: the turns go to the partner's own step, turns_note == "nut".
    steps3 = [
        {"component": 5, "component_label": "N2", "travel": 9.0},
        {"component": 4, "component_label": "M", "travel": 20.0},
    ]
    op._apply_turns(aog, prep, tables, steps3)
    rec.check(steps3[0]["turns"] is None, "nut N2 assembled before its movable partner M: N2 itself does not rotate")
    rec.check(steps3[1]["turns"] == 10.0, f"movable partner M rotates: 20/2.0 == 10.0 (got {steps3[1]['turns']})")
    rec.check(steps3[1]["turns_note"] == "nut", "partner-rotates case sets turns_note == 'nut'")
    rec.check(steps3[1]["turns_nut_label"] == "N2", "partner-rotates case names the responsible nut")

    # Text output (12.4): assembly_sequence.txt format (2 decimals on turns), manual format (1
    # decimal), "nut" phrasing, and no text at all when turns is None.
    plain = op._fmt_turns_text({"turns": 12.5, "turns_pitch": 1.5, "turns_note": None}, decimals=2)
    rec.check(plain == ", rotate 12.50 turns clockwise (pitch 1.5 mm)", f"plain turns text (got {plain!r})")
    manual = op._fmt_turns_text({"turns": 12.5, "turns_pitch": 1.5, "turns_note": None}, decimals=1)
    rec.check(manual == ", rotate 12.5 turns clockwise (pitch 1.5 mm)", f"manual (1 decimal) turns text (got {manual!r})")
    nut_text = op._fmt_turns_text(
        {"turns": 10.0, "turns_pitch": 2.0, "turns_note": "nut", "turns_nut_label": "N2"}, decimals=2)
    rec.check(nut_text == ", rotate 10.00 turns clockwise (thread of nut N2, pitch 2 mm)",
              f"nut-rotates-partner turns text (got {nut_text!r})")
    none_text = op._fmt_turns_text({"turns": None}, decimals=2)
    rec.check(none_text == "", f"no text when turns is None (got {none_text!r})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("O1", test_o1), ("O2", test_o2), ("O3", test_o3), ("O4", test_o4), ("O5", test_o5), ("O6", test_o6),
    ("O7", test_o7), ("O8", test_o8), ("O9", test_o9), ("O10", test_o10), ("O11", test_o11), ("O12", test_o12),
    ("O13", test_o13), ("O14", test_o14), ("O15", test_o15),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="*", help="individual test ids, e.g. O1 O5")
    ap.add_argument("--groups", nargs="*", default=["core"], choices=["core", "real", "scale"],
                    help="core (default), real (FreeCAD geometry + pipeline), scale (O14 at N = 20)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="    %(name)s %(levelname)s: %(message)s")

    rec = Recorder()
    cache_before = cache_snapshot()
    for test_id, fn in TESTS:
        if args.tests and test_id not in args.tests:
            continue
        print(f"\n=== {test_id} ===", flush=True)
        rec.start(test_id)
        t0 = time.perf_counter()
        try:
            fn(rec, args)
        except Exception as exc:  # noqa: BLE001
            rec.error(exc)
        rec.results[test_id]["seconds"] = round(time.perf_counter() - t0, 3)

    print("\n=== cache ===", flush=True)
    rec.start("cache")
    after = cache_snapshot()
    for name, digest in cache_before.items():
        rec.check(after.get(name) == digest, f"tests/industrial_fastner/{name} byte-identical after the run")
    rec.results["cache"]["seconds"] = 0.0

    print("\n| Test | Result | Time | Notes |\n|---|---|---|---|")
    for test_id, r in rec.results.items():
        status = "PASS" if r["passed"] else ("ERROR" if "error" in r else "FAIL")
        failed = [c["label"] for c in r["checks"] if not c["passed"]]
        notes = "; ".join(failed[:2] + r["notes"][:2] + ([r["error"]] if "error" in r else []))
        print(f"| {test_id} | {status} | {r['seconds']:.1f} s | {notes} |")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "optimization_results.json"
    out.write_text(json.dumps({"groups": args.groups, "results": rec.results}, indent=2))
    print(f"\nDetails written to {out}")
    return 0 if all(r["passed"] for r in rec.results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
