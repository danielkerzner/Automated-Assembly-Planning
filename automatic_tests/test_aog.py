"""Acceptance tests for the AoG generator (A1-A13).

[SUPP] Automatic test file: not needed to run the pipeline. Module under test: `src/aog_generator.py`
(`generate_aog`, `collect_inputs`, `build_aog`, the travel table, the direction extension, the cache and
the sequence read-out).

How the tests are built
    Most tests feed `generate_aog` synthetic `AoGInputs` (tiny hand-made interference tensors, built with
    `make_inputs`) so that the expected graph is known by construction. The two main reference sources are
    (1) `brute_force`, an independent naive enumeration of all 2^N states written from the standard rules, and
    (2) the cached `industrial_fastner` matrix and `report.json` committed in `output/industrial_fastner/`
    (read-only). `random_inputs` generates random tensors with random standard gates, nut links, contacts and
    BOM groups, so the brute-force comparison covers combinations nobody wrote by hand.

Standalone runner (pytest is not installed in environmentAAP). From repo root:

    python automatic_tests/test_aog.py                         # core group (seconds, no FreeCAD run)
    python automatic_tests/test_aog.py --tests A1 A6           # individual tests
    python automatic_tests/test_aog.py --groups core real      # + FreeCAD run on industrial_fastner
    python automatic_tests/test_aog.py --groups scale          # A5 at N = 20

Groups:
  core   all tests on synthetic AoGInputs, plus industrial_fastner built from the cached interference
         matrix (output/industrial_fastner/interference/) and output/industrial_fastner/report.json
         (committed reference results; read-only).
  real   industrial_fastner through run_preprocessing + identify_interferences + build_aog (A2b, A8
         assembly-box check). Interference cache in the scratch folder (h = 2.0; ~4 min the first time).
  scale  A5 at N = 20.

Scratch output goes to automatic_tests/output/aog_test/; results to automatic_tests/output/aog_results.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import tracemalloc
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import aog_generator as ag  # noqa: E402
from aog_generator import AoGConfig, AoGInputs  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
SCRATCH = OUT_DIR / "aog_test"
FASTENER_DIR = REPO_ROOT / "output" / "industrial_fastner"
FASTENER_STEP = REPO_ROOT / "data" / "industrial_fastner" / "industrial_fastner.stp"
FASTENER_BOM = REPO_ROOT / "data" / "industrial_fastner" / "generated_bom.csv"
# Standard IDs -> Labels of industrial_fastner (report.json stores IDs only; resolved by the real run).
FASTENER_STANDARD_LABELS = {"Part__Feature003": "industrial_fastner_part7",
                           "Part__Feature004": "industrial_fastner_part4",
                           "Part__Feature005": "industrial_fastner_part001"}

AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
AXES_NEG = np.array([1, 0, 3, 2, 5, 4])
AXIS_NAMES = ["+x", "-x", "+y", "-y", "+z", "-z"]


# ---------------------------------------------------------------------------
# Tiny test harness (same shape as the other test files)
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


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def __enter__(self):
        logging.getLogger("aap.aog").addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger("aap.aog").removeHandler(self)


def fresh_dir(name: str) -> Path:
    path = SCRATCH / name
    shutil.rmtree(path, ignore_errors=True)
    return path


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------


def make_inputs(free, directions=AXES, neg=AXES_NEG, standard_options=None, standard_holds=None,
                nut_links=None, bbox_lo=None, bbox_hi=None, box=((0, 0, 0), (100, 100, 100)), ids=None,
                key="synthetic", contact=None, group=None):
    """Build AoGInputs for a synthetic case. `standard_options`: {bit: [(direction, blockers), ...]}
    (one entry = one RemovalOption; most tests only need a single-option standard, so a bare
    `{bit: direction}` shorthand is also accepted and expanded to one option with no blockers)."""
    free = np.asarray(free, dtype=bool)
    n = free.shape[0]
    standard_options = standard_options or {}
    norm_options = {}
    for s, opts in standard_options.items():
        if isinstance(opts, tuple) and len(opts) == 2 and not isinstance(opts[0], (tuple, list)):
            opts = [opts]
        elif not isinstance(opts, list):
            opts = [(opts, set())]  # bare direction shorthand: one option, no blockers
        norm_options[s] = [(np.asarray(d, dtype=float), set(b)) for d, b in opts]
    N = n + len(norm_options)
    ids = ids or [f"c{i}" for i in range(N)]
    return AoGInputs(
        component_ids=list(ids), component_labels=list(ids), n_movable=n, free=free,
        directions=np.asarray(directions, dtype=float), neg=np.asarray(neg),
        standard_options=norm_options,
        standard_holds={s: set(h) for s, h in (standard_holds or {}).items()},
        nut_links={s: (int(p), r) for s, (p, r) in (nut_links or {}).items()},
        bbox_lo=np.zeros((N, 3)) if bbox_lo is None else np.asarray(bbox_lo, dtype=float),
        bbox_hi=np.ones((N, 3)) if bbox_hi is None else np.asarray(bbox_hi, dtype=float),
        assembly_lo=np.asarray(box[0], dtype=float), assembly_hi=np.asarray(box[1], dtype=float),
        cache_key=key,
        contact=None if contact is None else np.asarray(contact, dtype=bool),
        group=None if group is None else list(group),
    )


def contact_from_pairs(N, pairs):
    contact = np.zeros((N, N), dtype=bool)
    for a, b in pairs:
        contact[a, b] = contact[b, a] = True
    return contact


def random_inputs(rng: np.random.Generator, N: int) -> AoGInputs:
    m = int(rng.integers(0, min(3, N - 1) + 1))
    n = N - m
    D = len(AXES)
    p = rng.uniform(0.3, 0.97)
    free = np.ones((n, n, D), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            row = rng.random(D) < p
            free[i, j] = row
            free[j, i, AXES_NEG] = row  # antisymmetry, as the real matrix
    standard_options, standard_holds, nut_links = {}, {}, {}
    for s in range(n, N):
        n_opts = 2 if rng.random() < 0.3 else 1
        opts = []
        for _ in range(n_opts):
            d = AXES[rng.integers(D)] if rng.random() < 0.5 else rng.normal(size=3)
            blockers = {int(j) for j in range(N) if j != s and rng.random() < 0.25}
            opts.append((d, blockers))
        standard_options[s] = opts
        if rng.random() < 0.3:
            standard_holds[s] = {int(j) for j in range(n) if rng.random() < 0.3}
        if n_opts == 1 and rng.random() < 0.4:
            candidates = [c for c in range(N) if c != s]
            if candidates:
                nut_links[s] = (int(rng.choice(candidates)), "locked" if rng.random() < 0.5 else "free")
    contact = None
    if rng.random() < 0.75:
        q = rng.uniform(0.15, 0.7)
        contact = np.triu(rng.random((N, N)) < q, 1)
        contact[n:, n:] = False  # no standard-standard contacts, as in Preprocessing
        contact |= contact.T
    group = None
    if rng.random() < 0.6:  # a few random groups of 2-3 parts (movable and standards mixed)
        names = [None] * N
        for g in range(int(rng.integers(1, 3))):
            for c in rng.choice(N, size=min(N, int(rng.integers(2, 4))), replace=False):
                names[int(c)] = f"g{g}"
        group = names
    return make_inputs(free, standard_options=standard_options, standard_holds=standard_holds,
                      nut_links=nut_links, contact=contact, group=group)


def fixture(rel: str) -> Path:
    """output/industrial_fastner/<rel> (committed reference results, read-only)."""
    return FASTENER_DIR / rel


def fastener_inputs_from_cache() -> AoGInputs:
    """industrial_fastner inputs from the cached matrix + report.json (no FreeCAD, no movable AABBs).

    report.json (field names): category, options[] (direction,
    exit_point, blockers), holds, contact_neighbors, bbox, partner, nut_rule.
    """
    meta = json.loads(fixture("interference/interference.json").read_text(encoding="utf-8"))
    with np.load(fixture("interference/interference.npz")) as a:
        free, dirs, neg = a["free"], a["directions"], a["neg"]
        if "contact" not in a.files:
            raise RuntimeError("cached interference matrix has no contact matrix; run once with --groups real")
        movable_contact = a["contact"]
    report = json.loads(fixture("report.json").read_text(encoding="utf-8"))
    ids = meta["component_ids"]
    n = len(ids)
    index = {c: i for i, c in enumerate(ids)}
    standard_ids = sorted(report["standard_records"])
    index_all = {c: i for i, c in enumerate(ids + standard_ids)}
    standard_options, standard_holds, nut_links = {}, {}, {}
    lo, hi = [[0, 0, 0]] * n, [[0, 0, 0]] * n
    N = n + len(standard_ids)
    contact = np.zeros((N, N), dtype=bool)
    contact[:n, :n] = movable_contact
    for b, sid in enumerate(standard_ids, start=n):
        rec = report["standard_records"][sid]
        standard_options[b] = [(opt["direction"], {index_all[m] for m in opt["blockers"]}) for opt in rec["options"]]
        standard_holds[b] = {index[m] for m in rec["holds"]}
        if rec["nut_rule"] is not None and rec["partner"] is not None and rec["partner"] in index_all:
            nut_links[b] = (index_all[rec["partner"]], rec["nut_rule"])
        lo, hi = lo + [rec["bbox"][0]], hi + [rec["bbox"][1]]
        for g in rec["contact_neighbors"]:
            contact[b, index[g]] = contact[index[g], b] = True
    labels = meta["component_labels"] + [FASTENER_STANDARD_LABELS.get(s, s) for s in standard_ids]
    inputs = make_inputs(free, dirs, neg, standard_options, standard_holds, nut_links, lo, hi,
                         box=report["assembly_bbox"], ids=ids + standard_ids, key=meta["cache_key"],
                         contact=contact, group=ag.read_bom_groups(FASTENER_BOM, labels))
    inputs.component_labels = labels
    return inputs


# ---------------------------------------------------------------------------
# Brute force reference (A1): naive enumeration over all 2^N subsets
# ---------------------------------------------------------------------------


def _free_directions(free, directions, i, subset):
    # InterferenceResult.free_directions, called unbound on a stand-in (no InterferenceResult needed).
    try:
        import freecad  # noqa: F401  (sets up the FreeCAD lib path in the conda env)
        from interference_identification import InterferenceResult
        fn = InterferenceResult.free_directions
    except Exception:  # noqa: BLE001  (FreeCAD unavailable: same logic inline)
        def fn(self, i, subset):
            members = list(subset)
            if not members:
                return np.arange(len(self.directions))
            return np.flatnonzero(np.all(self.free[i, members, :], axis=0))
    return fn(SimpleNamespace(free=free, directions=directions), i, subset)


def brute_force(inputs: AoGInputs):
    """(node set, edge set) with edges as (src_mask, comp, dst_mask, frozenset(direction indices)).

    Independent reference for the standard model: blocker masks per
    standard option (plus nut_links folded into them / into movable `extra`), holds checked on
    the resulting state T, not derived from `generate_aog`'s own mask code.
    """
    N, n = len(inputs.component_ids), inputs.n_movable
    dirs, _, kd = ag.extend_directions(inputs.directions, inputs.neg, inputs.standard_options)

    def members(mask):
        return [c for c in range(N) if (mask >> c) & 1]

    def connected(mask):
        if inputs.contact is None:
            return True
        comps = members(mask)
        if len(comps) <= 1:
            return True
        seen, stack = {comps[0]}, [comps[0]]
        while stack:
            a = stack.pop()
            for b in comps:
                if b not in seen and (inputs.contact[a, b] or inputs.contact[b, a]):
                    seen.add(b)
                    stack.append(b)
        return len(seen) == len(comps)

    # Fold nut_links into blocker sets / movable extra, exactly as the standard model defines them
    # (independently of generate_aog's own numpy mask code).
    blockers = {s: [set(b) for _d, b in inputs.standard_options[s]] for s in range(n, N)}
    holds = {s: set(inputs.standard_holds.get(s, ())) for s in range(n, N)}
    extra = {i: set() for i in range(n)}
    for nut, (partner, rule) in inputs.nut_links.items():
        if rule == "locked":
            for bset in blockers[nut]:
                bset.add(partner)
        elif rule == "free":
            if partner < n:
                extra[partner].add(nut)
            else:
                for bset in blockers[partner]:
                    bset.add(nut)

    def valid(mask):
        if not connected(mask):
            return False
        present = set(members(mask))
        return all(not (s in present and holds[s] and not holds[s] <= present) for s in range(n, N))

    groups = {}
    if inputs.group is not None:
        for c, g in enumerate(inputs.group):
            if g:
                groups.setdefault(g, set()).add(c)
    groups = [m for m in groups.values() if len(m) >= 2]

    out = {}
    for S in range(1 << N):
        present = set(members(S))
        edges = []
        for c in present:
            T = S & ~(1 << c)
            if not valid(T):
                continue
            if c < n:
                if extra[c] & present:
                    continue
                ks = _free_directions(inputs.free, inputs.directions, c, [j for j in present if j < n])
            else:
                ks = [kd[c][o] for o, bset in enumerate(blockers[c]) if not (bset & present)]
            if len(ks):
                edges.append((S, c, T, frozenset(int(k) for k in ks)))
        if groups:
            present = set(members(S))
            partial = [g for g in groups if 0 < len(g & present) < len(g)]
            if partial:  # D12: only members of the unfinished group may move
                edges = [e for e in edges if any(e[1] in g for g in partial)]
            for g in groups:  # D13: one member per group and state, the lowest feasible one
                movers = [e[1] for e in edges if e[1] in g]
                if movers:
                    edges = [e for e in edges if e[1] not in g or e[1] == min(movers)]
        out[S] = edges
    full = (1 << N) - 1
    seen, stack = {full}, [full]
    while stack:
        S = stack.pop()
        for _, _, T, _ in out[S]:
            if T not in seen:
                seen.add(T)
                stack.append(T)
    edges = {e for S in seen for e in out[S]}
    return seen, edges


def module_sets(aog):
    nodes = {int(m) for m in aog.node_mask}
    edges = {(int(aog.node_mask[aog.edge_src[e]]), int(aog.edge_comp[e]), int(aog.node_mask[aog.edge_dst[e]]),
              frozenset(int(k) for k in aog.edge_direction_indices(e))) for e in range(aog.n_edges)}
    return nodes, edges


def structure_problems(aog) -> list[str]:
    N, V, E = aog.n_components, aog.n_nodes, aog.n_edges
    problems = []
    if int(aog.node_mask[aog.root]) != (1 << N) - 1:
        problems.append("root is not the full state")
    if aog.terminal is not None and int(aog.node_mask[aog.terminal]) != 0:
        problems.append("terminal is not the empty state")
    pop = np.array([bin(int(m)).count("1") for m in aog.node_mask])
    if not np.array_equal(pop, N - aog.node_level.astype(int)):
        problems.append("popcount != N - level")
    for L in range(N + 1):
        a, b = aog.level_offsets[L], aog.level_offsets[L + 1]
        if np.any(aog.node_level[a:b] != L):
            problems.append(f"level_offsets inconsistent at level {L}")
        if b - a > 1 and np.any(np.diff(aog.node_mask[a:b]) <= 0):
            problems.append(f"level {L} not sorted/unique")
    if aog.level_offsets[-1] != V or len(aog.level_offsets) != N + 2:
        problems.append("level_offsets shape/end wrong")
    src_m, dst_m = aog.node_mask[aog.edge_src], aog.node_mask[aog.edge_dst]
    bits = np.left_shift(np.uint64(1), aog.edge_comp.astype(np.uint64))
    if np.any((src_m & bits) == 0) or np.any(dst_m != (src_m & ~bits)):
        problems.append("an edge does not remove exactly one present bit")
    if np.any(aog.node_level[aog.edge_dst] != aog.node_level[aog.edge_src] + 1):
        problems.append("an edge skips levels (not layered / not a DAG)")
    if np.any(np.diff(aog.edge_src) < 0):
        problems.append("edges not sorted by src")
    if aog.out_ptr[0] != 0 or aog.out_ptr[-1] != E or len(aog.out_ptr) != V + 1:
        problems.append("out_ptr ends wrong")
    for v in range(V):
        if np.any(aog.edge_src[aog.out_edges(v)] != v):
            problems.append(f"CSR wrong at node {v}")
            break
    if np.any(aog.edge_dirs[:, 0] == 0):
        problems.append("edge with no direction")
    return problems


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_STATE: dict = {}


def fastener_cached():
    if "fastener" not in _STATE:
        inputs = fastener_inputs_from_cache()
        _STATE["fastener"] = (inputs, ag.generate_aog(inputs))
    return _STATE["fastener"]


def fastener_real():
    if "real" not in _STATE:
        import freecad  # noqa: F401  (sets up the FreeCAD lib path in the conda env)
        from collision import CollisionConfig
        from interference_identification import identify_interferences
        from preprocessing import run_preprocessing

        print("  (running preprocessing + interference on industrial_fastner)", flush=True)
        prep = run_preprocessing(FASTENER_STEP, FASTENER_BOM)
        ir = identify_interferences(prep, CollisionConfig(step=2.0), SCRATCH / "fastener_interference")
        aog = ag.build_aog(ir, prep, cache_dir=fresh_dir("fastener_real"))
        _STATE["real"] = (prep, ir, aog)
    return _STATE["real"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_a1(rec, args):
    """A1 - The generated graph equals a brute-force enumeration.

    Tests: `generate_aog` (states, edges and the free direction set of every edge).
    What: compares nodes and edges with `brute_force`, an independent naive implementation of the same
    rules, on the real industrial_fastner inputs and on 200 random synthetic cases (N up to 12, random
    standard gates, holds, nut links, contacts and groups).
    """
    inputs, aog = fastener_cached()
    nodes, edges = brute_force(inputs)
    m_nodes, m_edges = module_sets(aog)
    rec.check(nodes == m_nodes and edges == m_edges,
              f"industrial_fastner (N={len(inputs.component_ids)}): V={len(m_nodes)}, E={len(m_edges)}; "
              f"brute force V={len(nodes)}, E={len(edges)}; identical={nodes == m_nodes and edges == m_edges}")
    rng = np.random.default_rng(1234)
    mismatches, sizes = [], []
    for case in range(200):
        N = int(rng.integers(1, 13)) if case % 4 == 0 else int(rng.integers(1, 10))
        inputs = random_inputs(rng, max(N, 2))
        aog = ag.generate_aog(inputs)
        if brute_force(inputs) != module_sets(aog):
            mismatches.append(case)
        sizes.append(aog.n_nodes)
    rec.note(f"random cases: V from {min(sizes)} to {max(sizes)}, "
             f"{sum(1 for s in sizes if s > 1)} non-trivial")
    rec.check(not mismatches, f"200 random synthetic tensors (N <= 12, random gates): "
              f"{len(mismatches)} mismatches {mismatches[:5]}")


def test_a2(rec, args):
    """A2 - The graph is structurally a valid layered DAG.

    Tests: the arrays built by `generate_aog` (levels, CSR edge index, masks).
    What: runs `structure_problems` on industrial_fastner and 20 random graphs: root is the full state,
    terminal the empty one, popcount equals N - level, every edge removes exactly one present part and moves
    exactly one level, edges are sorted by source, the CSR pointers are consistent, no edge without a direction.
    """
    rng = np.random.default_rng(7)
    graphs = [("industrial_fastner", fastener_cached()[1])]
    graphs += [(f"random #{i}", ag.generate_aog(random_inputs(rng, 10))) for i in range(20)]
    for name, aog in graphs:
        problems = structure_problems(aog)
        if problems or name == "industrial_fastner":
            rec.check(not problems, f"{name}: DAG/layered/one-bit edges/root/terminal/CSR ({problems or 'ok'})")
    rec.check(all(not structure_problems(a) for _, a in graphs), f"all {len(graphs)} graphs structurally valid")


def _vec_key(v) -> tuple[float, ...]:
    return tuple(round(float(x), 3) + 0.0 for x in v)


def _root_out(aog):
    """{component id: [direction vectors]} of the root's out-edges (independent of direction indexing)."""
    return {aog.component_ids[int(aog.edge_comp[e])]: [_vec_key(aog.directions[k]) for k in aog.edge_direction_indices(e)]
            for e in aog.out_edges(aog.root)}


def vector_sets(aog):
    """module_sets with direction vectors instead of indices, to compare graphs whose candidate
    direction tables differ (the committed fixture has 26 directions, live Preprocessing fewer)."""
    edges = {(int(aog.node_mask[aog.edge_src[e]]), int(aog.edge_comp[e]), int(aog.node_mask[aog.edge_dst[e]]),
              frozenset(_vec_key(aog.directions[k]) for k in aog.edge_direction_indices(e)))
             for e in range(aog.n_edges)}
    return {int(m) for m in aog.node_mask}, edges


def same_graph_up_to_directions(real, cached) -> bool:
    """Same states and moves (src, component, dst), and on every edge the real run's free direction
    vectors are a subset of the cached run's (the fixture's 26-direction table includes the live 10)."""
    real_nodes, real_edges = vector_sets(real)
    cached_nodes, cached_edges = vector_sets(cached)
    cached_dirs = {e[:3]: e[3] for e in cached_edges}
    return (real_nodes == cached_nodes and {e[:3] for e in real_edges} == set(cached_dirs)
            and all(e[3] <= cached_dirs[e[:3]] for e in real_edges))


FASTENER_ROOT = {
    "Part__Feature002": [(0.0, 0.0, -1.0)],  # part1, -z
    "Part__Feature006": [(-1.0, 0.0, 0.0)],  # part5, -x
    # part6 (+x): freely removable at the root -- part7 (the nut) is "locked" on it as partner
    # (P8: partner leaves before the nut), so part6 has no extra constraint, only part7 does (and
    # part7 is therefore NOT in this dict: it only becomes removable once part6 is gone).
    "Part__Feature007": [(1.0, 0.0, 0.0)],  # part6
    # part4, part001 (category 3, no BOM group in the current generated_bom.csv): both +y/-y free.
    "Part__Feature004": [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)],
    "Part__Feature005": [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)],
}


def test_a2b(rec, args):
    """A2b - Root out-edges of industrial_fastner are the known ones.

    Tests: `generate_aog` on a real assembly (and, with `--groups real`, the whole chain Preprocessing ->
    Interference -> `build_aog`).
    What: checks which components can leave first and in which directions (`FASTENER_ROOT`: part1 -z, part5 -x,
    part6 +x, part4 and part001 +/-y; the nut part7 only after its partner part6). With `real`, the live
    run must give the same graph as the cached inputs up to the candidate-direction table.
    """
    inputs, aog = fastener_cached()
    got = _root_out(aog)
    for cid, ks in got.items():
        rec.note(f"root out-edge: {aog.component_labels[aog.component_ids.index(cid)]} ({cid}) {ks}")
    rec.check(got == FASTENER_ROOT, f"root out-edges (cached inputs) == {FASTENER_ROOT}")
    rec.note(f"V={aog.n_nodes}, E={aog.n_edges}, per level {aog.stats['nodes_per_level']}, "
             f"dead ends {len(aog.dead_ends)}, complete={aog.complete}")
    if "real" in args.groups:
        prep, ir, real = fastener_real()
        rec.check(_root_out(real) == FASTENER_ROOT, "root out-edges (real FreeCAD run) match")
        rec.check(same_graph_up_to_directions(real, aog),
                  "real run graph == cached-input graph (== brute force): same states and moves, real "
                  "directions a subset (live Preprocessing has fewer candidate directions than the fixture)")


def _a3_run():
    # p (0), q (1) movable, free everywhere, touching each other; standard s (2) is a screw with
    # hold {p} (must stay present while s leaves) and touches both p and q (contact_neighbors,
    # connectivity only, P10 -- not a gate any more: q must NOT be required present, unlike the
    # old "prerequisites+contacts" gate mode). s also touching q (not just p) keeps {p, s}
    # connected after q leaves, so D11 (no-floating-parts) cannot confound the holds check below.
    free = np.ones((2, 2, 6), dtype=bool)
    ir = SimpleNamespace(component_ids=["p", "q"], component_labels=["p", "q"], free=free,
                         directions=AXES, neg=AXES_NEG, cache_key="a3",
                         contact=np.array([[False, True], [True, False]]))
    box = SimpleNamespace(XMin=0, YMin=0, ZMin=0, XMax=1, YMax=1, ZMax=1)
    reduced = SimpleNamespace(Name="a3", getObject=lambda cid: SimpleNamespace(Shape=SimpleNamespace(BoundBox=box)))
    opt = SimpleNamespace(direction=(0, 0, 1), exit_point=(0, 0, 1), blockers=set())
    rec_s = SimpleNamespace(category=1, options=[opt], holds={"p"}, contact_neighbors={"p", "q"},
                            bbox=((0, 0, 0), (1, 1, 1)), partner=None, nut_rule=None)
    prep = SimpleNamespace(reduced_assembly=reduced, standard_records={"s": rec_s},
                           assembly_bbox=((0, 0, 0), (1, 1, 1)), step_path=None)
    config = AoGConfig()
    inputs = ag.collect_inputs(ir, prep, config)
    return inputs, ag.generate_aog(inputs, config)


def test_a3(rec, args):
    """A3 - Standard holds are gates, contact neighbours are not (rule P10).

    Tests: `collect_inputs` (turning a standard record into holds) and the hold rule in `generate_aog`.
    What: a screw s holds p and also touches p and q. Only p must appear in s's holds; p cannot leave while s
    is present, q can; no state contains s without p.
    """
    inputs, aog = _a3_run()
    rec.check(inputs.standard_holds == {2: {0}}, f"holds of s = {inputs.standard_holds[2]} (only p, not q)")
    root_comps = {int(aog.edge_comp[e]) for e in aog.out_edges(aog.root)}
    rec.check(2 in root_comps, "s removable at the root")
    rec.check(0 not in root_comps, "p (a hold) not removable while s present")
    rec.check(1 in root_comps, "q (only a contact_neighbor, not a hold -- P10) IS removable while s present")
    s_bit = 1 << 2
    bad = [int(m) for m in aog.node_mask if int(m) & s_bit and not (int(m) >> 0) & 1]
    rec.check(not bad, f"every node with s present also has its hold p present ({len(bad)} violations)")
    rec.check(aog.complete, f"complete, V={aog.n_nodes}, E={aog.n_edges}")


def test_a4(rec, args):
    """A4 - A stack with blocked sides forces the removal order.

    Tests: `generate_aog` on a hand-made tensor (A on B on C in a channel, C extractable only upward).
    What: expects 4 nodes, 3 edges, the unique order A, B, C with the right directions, no dead ends.
    """
    # Stack A (top), B, C (bottom) in a channel: side directions blocked. C sits in a blind seat and can
    # only be extracted upward (so free[C, :, -z] is False; not antisymmetric, fine for the AoG).
    A, B, C = 0, 1, 2
    free = np.zeros((3, 3, 6), dtype=bool)
    for i in range(3):
        free[i, i] = True
    up, down = AXIS_NAMES.index("+z"), AXIS_NAMES.index("-z")
    free[A, B, up] = free[A, C, up] = True
    free[B, C, up] = True
    free[B, A, down] = True
    aog = ag.generate_aog(make_inputs(free, ids=["A", "B", "C"]))
    steps = [(s["component_label"], [AXIS_NAMES[k] for k in s["removal_direction_indices"]])
             for s in aog.sequence(list(range(aog.n_edges)))] if aog.n_edges == 3 else []
    rec.check(aog.n_nodes == 4 and aog.n_edges == 3, f"4 nodes, 3 edges (got {aog.n_nodes}, {aog.n_edges})")
    rec.check(steps == [("A", ["+z"]), ("B", ["+z"]), ("C", AXIS_NAMES)],
              f"forced order A, B, C (C alone is free everywhere): {steps}")
    rec.check(aog.complete and not len(aog.dead_ends), "complete, no dead ends")


def _scale_case(rec, N):
    free = np.ones((N, N, 6), dtype=bool)
    inputs = make_inputs(free)
    tracemalloc.start()
    t0 = time.perf_counter()
    aog = ag.generate_aog(inputs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    V, E = aog.n_nodes, aog.n_edges
    rec.note(f"N={N}: {elapsed:.2f} s, arrays {aog.stats['array_bytes'] / 1e6:.1f} MB, "
             f"peak traced {peak / 1e6:.1f} MB")
    rec.check(V == 2 ** N and E == N * 2 ** (N - 1), f"N={N}: V={V} (2^N={2 ** N}), E={E} (N*2^(N-1)={N * 2 ** (N - 1)})")
    return elapsed, peak


def test_a5(rec, args):
    """A5 - Scaling and size of the full graph.

    Tests: `generate_aog` performance and node/edge counts when nothing blocks anything.
    What: N = 12 and 16 (and 20 with `--groups scale`) free parts must give exactly V = 2^N and E = N*2^(N-1);
    N = 20 must finish within 120 s and 2 GB. Time and memory are reported as notes.
    """
    for N in (12, 16):
        _scale_case(rec, N)
    if "scale" in args.groups:
        elapsed, peak = _scale_case(rec, 20)
        rec.check(elapsed < 120 and peak < 2e9, f"N=20 within budget (< 120 s, < 2 GB): {elapsed:.1f} s, "
                  f"{peak / 1e9:.2f} GB")
    else:
        rec.note("N = 20 skipped (run with --groups scale)")


def test_a6(rec, args):
    """A6 - Dead ends are kept, marked and reported, not pruned.

    Tests: `generate_aog` handling of states where no part can move (`dead_ends`, `live`, `complete`).
    What: two parts blocking each other give a single dead-end root; with a free third part the state {A, B}
    is reached, kept and marked dead, the root is not live, the graph is incomplete, and the warnings are logged.
    """
    # Two parts blocking each other in every direction.
    free = np.zeros((2, 2, 6), dtype=bool)
    free[0, 0] = free[1, 1] = True
    with LogCapture() as cap:
        aog = ag.generate_aog(make_inputs(free, ids=["A", "B"]))
    rec.check(aog.n_nodes == 1 and list(aog.dead_ends) == [0] and not aog.live[0] and not aog.complete,
              f"interlocked pair: V={aog.n_nodes}, dead ends {list(aog.dead_ends)}, complete={aog.complete}")
    rec.check(any("dead-end" in r.getMessage() for r in cap.records), "dead-end warning logged")
    # Plus a free third part: {A, B} is reached, kept, and marked dead.
    free = np.ones((3, 3, 6), dtype=bool)
    free[0, 1] = free[1, 0] = False
    with LogCapture() as cap:
        aog = ag.generate_aog(make_inputs(free, ids=["A", "B", "C"]))
    dead_masks = [int(aog.node_mask[v]) for v in aog.dead_ends]
    rec.check(dead_masks == [0b011] and aog.n_nodes == 2 and aog.n_edges == 1,
              f"with free C: nodes {aog.n_nodes}, edges {aog.n_edges}, dead-end masks {dead_masks} (nothing pruned)")
    rec.check(not aog.live.any() and not aog.complete and aog.terminal is None,
              "live False on the dead end and on the root that only reaches it; complete False")
    rec.check(any("incomplete" in r.getMessage() for r in cap.records), "incomplete warning logged")


def test_a7(rec, args):
    """A7 - Standard directions extend the direction table consistently.

    Tests: `extend_directions` (the AoG direction table with the standard removal directions added).
    What: a standard direction equal to a candidate reuses its index; a new one appends d and -d (and -d is then
    reused); `dirs[neg[k]] == -dirs[k]` always; on industrial_fastner no direction is added; movable parts
    never use the appended directions.
    """
    diag = np.array([1, 1, 0]) / np.sqrt(2)
    standard_opts = {6: [(AXES[4] + 1e-9, set())], 7: [(diag, set())], 8: [(-diag, set())]}
    dirs, neg, kd = ag.extend_directions(AXES, AXES_NEG, standard_opts)
    rec.check(kd[6] == [4], f"standard direction equal to a candidate reuses its index (got {kd[6]})")
    rec.check(len(dirs) == 8 and kd[7] == [6] and kd[8] == [7],
              f"new direction appends d and -d, and -d is then reused (D'={len(dirs)}, kd={kd})")
    rec.check(np.allclose(dirs[neg], -dirs), "dirs[neg[k]] == -dirs[k] for all k")
    _, aog = fastener_cached()
    rec.check(len(aog.directions) == len(fastener_cached()[0].directions) and np.allclose(aog.directions[aog.neg], -aog.directions),
              f"industrial_fastner: standard directions all reuse candidates (D'={len(aog.directions)})")
    # A standard with a direction that is not a candidate: movable parts never use the new index.
    free = np.ones((2, 2, 6), dtype=bool)
    aog = ag.generate_aog(make_inputs(free, standard_options={2: diag}))
    new_bits = np.uint64((1 << 6) | (1 << 7))
    movable_edges = aog.edge_comp < 2
    rec.check(not np.any(aog.edge_dirs[movable_edges, 0] & new_bits)
              and np.all(aog.edge_dirs[~movable_edges, 0] == np.uint64(1 << 6)),
              "appended directions used only by the standard")


def test_a8(rec, args):
    """A8 - Travel table (distance a part must move to leave the assembly box).

    Tests: `compute_travel_table` and `AoG.travel` / `edge_distances`.
    What: hand-computed distances for axis-aligned, oblique, touching-face and already-outside cases (plus the
    margin); travel is always >= margin and does not depend on which parts are present; with `--groups real`
    the assembly box equals the union of the component boxes.
    """
    m = 1.0
    box = ((0, 0, 0), (10, 10, 10))
    s2 = np.sqrt(0.5)
    comps = [((2, 2, 2), (4, 4, 4)), ((2, 6, 2), (4, 8, 4)), ((8, 2, 2), (10, 4, 4)), ((12, 2, 2), (14, 4, 4))]
    lo, hi = np.array([c[0] for c in comps], float), np.array([c[1] for c in comps], float)
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [s2, s2, 0], [0, 1, 0]])
    t = ag.compute_travel_table(lo, hi, *box, dirs, m)
    cases = [
        ("axis-aligned +x", t[0, 0], 8 + m), ("axis-aligned -x", t[0, 1], 4 + m),
        ("oblique (1,1,0)/sqrt2 (y limits)", t[1, 2], 4 / s2 + m),
        ("touching box face from inside, +x (own width)", t[2, 0], 2 + m), ("touching box face from inside, -x", t[2, 1], 10 + m),
        ("already outside, +x", t[3, 0], m), ("already outside, +y", t[3, 3], m),
    ]
    for label, got, want in cases:
        rec.check(abs(got - want) < 1e-9, f"{label}: {got:.4f} (expected {want:.4f})")
    rec.check(np.all(t >= m), "travel >= margin everywhere")
    # Travel does not depend on which other components are present.
    free = np.ones((4, 4, 4), dtype=bool)
    neg = np.array([1, 0, 3, 2])
    dirs4 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
    full = ag.generate_aog(make_inputs(free, dirs4, neg, bbox_lo=lo, bbox_hi=hi, box=box))
    sub = ag.generate_aog(make_inputs(free[:2, :2], dirs4, neg, bbox_lo=lo[:2], bbox_hi=hi[:2], box=box))
    rec.check(np.array_equal(full.travel[:2], sub.travel), "travel rows unchanged when other components are removed")
    rec.check(all(np.array_equal(full.edge_distances(e), full.travel[full.edge_comp[e], full.edge_direction_indices(e)])
                  for e in range(full.n_edges)), "edge_distances = travel[comp, k] on every edge")
    if "real" in args.groups:
        prep, ir, aog = fastener_real()
        inputs = ag.collect_inputs(ir, prep)
        union_lo, union_hi = inputs.bbox_lo.min(axis=0), inputs.bbox_hi.max(axis=0)
        rec.check(np.allclose(union_lo, inputs.assembly_lo, atol=1e-6) and np.allclose(union_hi, inputs.assembly_hi, atol=1e-6),
                  f"industrial_fastner: assembly box {inputs.assembly_lo.round(3)}..{inputs.assembly_hi.round(3)} == "
                  f"union of component boxes {union_lo.round(3)}..{union_hi.round(3)}")
        rec.check(np.all(aog.travel >= aog.config.margin), "industrial_fastner: travel >= margin")
    else:
        rec.note("industrial_fastner assembly-box check skipped (run with --groups real)")


def _arrays_equal(a, b):
    return all(np.array_equal(getattr(a, f), getattr(b, f)) and getattr(a, f).dtype == getattr(b, f).dtype
               for f in ag._ARRAY_FIELDS)


def test_a9(rec, args):
    """A9 - Persistence and cache key.

    Tests: `save_aog`, `load_aog`, `export_csv_dot` and `compute_cache_key`.
    What: save then load gives identical arrays and metadata, a different key returns None, the CSV/DOT
    exports exist; changing any input that affects the graph (standard option blockers or direction, holds,
    nut rule, margin, BOM groups, group_one_order) changes the key, identical inputs keep it.
    """
    inputs, aog = fastener_cached()
    cache = fresh_dir("persistence")
    ag.save_aog(aog, cache)
    ag.export_csv_dot(aog, cache)
    loaded = ag.load_aog(cache, aog.cache_key)
    rec.check(loaded is not None and _arrays_equal(aog, loaded)
              and loaded.component_ids == aog.component_ids and loaded.config == aog.config
              and loaded.complete == aog.complete and loaded.terminal == aog.terminal,
              "save/load gives identical arrays and metadata")
    rec.check(ag.load_aog(cache, "other") is None, "load with a different key returns None (recompute)")
    rec.check(all((cache / f).exists() for f in ("aog_nodes.csv", "aog_edges.csv", "aog.dot")), "exports written")
    base = ag.compute_cache_key(inputs, AoGConfig())
    s0 = inputs.n_movable  # the nut (Part__Feature003): has a non-trivial nut_links entry to mutate
    variants = {}
    v = fastener_inputs_from_cache()
    d0, blockers0 = v.standard_options[s0][0]
    v.standard_options[s0][0] = (d0, blockers0 | {0})
    variants["standard option blockers"] = ag.compute_cache_key(v, AoGConfig())
    v = fastener_inputs_from_cache()
    d0, blockers0 = v.standard_options[s0][0]
    v.standard_options[s0][0] = (-np.asarray(d0, dtype=float), blockers0)
    variants["standard option direction"] = ag.compute_cache_key(v, AoGConfig())
    v = fastener_inputs_from_cache()
    v.standard_holds[s0] = v.standard_holds[s0] | {0}
    variants["standard holds"] = ag.compute_cache_key(v, AoGConfig())
    v = fastener_inputs_from_cache()
    partner, rule = v.nut_links[s0]
    v.nut_links[s0] = (partner, "free" if rule == "locked" else "locked")
    variants["nut_rule"] = ag.compute_cache_key(v, AoGConfig())
    variants["AoGConfig.margin"] = ag.compute_cache_key(inputs, AoGConfig(margin=2.0))
    v = fastener_inputs_from_cache()
    v.group = [None] * len(v.group)
    variants["BOM groups"] = ag.compute_cache_key(v, AoGConfig())
    variants["AoGConfig.group_one_order"] = ag.compute_cache_key(inputs, AoGConfig(group_one_order=False))
    for name, key in variants.items():
        rec.check(key != base, f"changing the {name} changes cache_key")
    rec.check(ag.compute_cache_key(fastener_inputs_from_cache(), AoGConfig()) == base, "same inputs, same key")


def _sample_paths(aog, rng, count):
    paths = []
    for _ in range(count):
        v, path = aog.root, []
        while v != aog.terminal:
            choices = [e for e in aog.out_edges(v) if aog.live[aog.edge_dst[e]]]
            e = int(rng.choice(choices))
            path.append(e)
            v = int(aog.edge_dst[e])
        paths.append(path)
    return paths


def test_a10(rec, args):
    """A10 - Every path of the graph is a valid removal order and assembly order.

    Tests: `AoG.sequence` and `AoG.assembly_sequence` (the read-out used by the optimiser).
    What: 100 random root-to-terminal paths on industrial_fastner and on random graphs are replayed
    step by step against the original rules (component present, directions equal the free set, holds kept,
    free nuts leave before their partner); the assembly sequence must use `neg` of the removal directions
    and place standards after their holds.
    """
    rng = np.random.default_rng(99)
    cases = [("industrial_fastner", *fastener_cached())]
    while len(cases) < 4:
        inputs = random_inputs(rng, 9)
        aog = ag.generate_aog(inputs)
        if aog.complete and inputs.standard_options:
            cases.append((f"random N=9 ({len(inputs.standard_options)} standards)", inputs, aog))
    for name, inputs, aog in cases:
        if not aog.complete:
            rec.check(False, f"{name}: graph incomplete, no root->terminal path")
            continue
        n, N = inputs.n_movable, len(inputs.component_ids)
        # A "free" nut blocks its movable partner from leaving while the nut is still present (P8).
        free_blockers_of = {c: set() for c in range(n)}
        for nut, (partner, rule) in inputs.nut_links.items():
            if rule == "free" and partner < n:
                free_blockers_of[partner].add(nut)
        bad_replay = bad_assembly = 0
        for path in _sample_paths(aog, rng, 100):
            present = set(range(N))
            for step in aog.sequence(path):
                c = step["component"]
                ok = c in present
                if c < n:
                    ks = _free_directions(inputs.free, inputs.directions, c, [j for j in present if j < n])
                    ok &= set(step["removal_direction_indices"]) == set(int(k) for k in ks) and len(ks) > 0
                    ok &= not (free_blockers_of[c] & present)
                present.discard(c)
                ok &= all(
                    s not in present or not inputs.standard_holds.get(s) or inputs.standard_holds[s] <= present
                    for s in range(n, N))
                bad_replay += not ok
            bad_replay += bool(present)
            placed: set[int] = set()
            for step in aog.assembly_sequence(path):
                c = step["component"]
                rem = step["removal_direction_indices"]
                ok = step["direction_indices"] == [int(aog.neg[k]) for k in rem]
                ok &= np.allclose(step["directions"], -np.asarray(step["removal_directions"]))
                if c >= n:
                    ok &= inputs.standard_holds.get(c, set()) <= placed
                placed.add(c)
                bad_assembly += not ok
        rec.check(bad_replay == 0 and bad_assembly == 0,
                  f"{name}: 100 sampled paths replay as valid removal orders ({bad_replay} bad steps, holds "
                  f"and free-nut/movable-partner order checked); assembly sequence uses neg and places "
                  f"standards after their holds ({bad_assembly} bad steps)")


def test_a11(rec, args):
    """A11 - Determinism.

    Tests: `generate_aog`.
    What: two runs on the same inputs (industrial_fastner and a random N=12 case) give byte-identical arrays.
    """
    inputs = fastener_inputs_from_cache()
    a, b = ag.generate_aog(inputs), ag.generate_aog(fastener_inputs_from_cache())
    same = all(getattr(a, f).tobytes() == getattr(b, f).tobytes() for f in ag._ARRAY_FIELDS)
    rec.check(same and a.cache_key == b.cache_key, "industrial_fastner: two runs give byte-identical arrays")
    rng1, rng2 = np.random.default_rng(5), np.random.default_rng(5)
    a, b = ag.generate_aog(random_inputs(rng1, 12)), ag.generate_aog(random_inputs(rng2, 12))
    rec.check(all(getattr(a, f).tobytes() == getattr(b, f).tobytes() for f in ag._ARRAY_FIELDS),
              f"random N=12 (V={a.n_nodes}): two runs give byte-identical arrays")


def _disconnected_nodes(aog, contact):
    bad = []
    for v in range(aog.n_nodes):
        comps = aog.state_components(v)
        if len(comps) > 1:
            seen, stack = {comps[0]}, [comps[0]]
            while stack:
                a = stack.pop()
                for b in comps:
                    if b not in seen and contact[a, b]:
                        seen.add(b)
                        stack.append(b)
            if len(seen) != len(comps):
                bad.append(v)
    return bad


def test_a12(rec, args):
    """A12 - Connectivity rule: no state whose parts float apart.

    Tests: the connectivity filter of `generate_aog` (needs the contact matrix, `AoGInputs.contact`).
    What: a chain A-B-C drops {A, C} and B cannot go first; without contact data the rule is off (V = 8);
    a plate with two separate parts never shows {X, Y}; two parts joined only through a bolt that holds both
    give a dead-end root; a disconnected full assembly only warns; and every state of industrial_fastner and
    of 50 random graphs is one connected group.
    """
    # Chain A - B - C, all free: removing B first would leave A and C apart.
    A, B, C = 0, 1, 2
    free = np.ones((3, 3, 6), dtype=bool)
    chain = contact_from_pairs(3, [(A, B), (B, C)])
    aog = ag.generate_aog(make_inputs(free, ids=["A", "B", "C"], contact=chain))
    masks = {int(m) for m in aog.node_mask}
    rec.check(0b101 not in masks, "chain A-B-C: state {A, C} is not in the AoG")
    rec.check(all(int(aog.edge_comp[e]) != B for e in aog.out_edges(aog.root)), "B is not removable first")
    rec.check(aog.n_nodes == 7 and aog.stats["disconnected_dropped"] == 1 and aog.complete,
              f"V={aog.n_nodes} (7 = 8 subsets minus {{A, C}}), dropped {aog.stats['disconnected_dropped']}, "
              f"complete={aog.complete}")
    # Same parts without contact data: rule off, every subset present.
    aog_off = ag.generate_aog(make_inputs(free, ids=["A", "B", "C"]))
    rec.check(aog_off.n_nodes == 8 and not aog_off.stats["connectivity_rule"], "contact=None: rule off, V=8")
    # Base plate P with X and Y on it (X, Y do not touch): {X, Y} must never appear.
    P, X, Y = 0, 1, 2
    aog = ag.generate_aog(make_inputs(free, ids=["P", "X", "Y"], contact=contact_from_pairs(3, [(P, X), (P, Y)])))
    rec.check(0b110 not in {int(m) for m in aog.node_mask}, "plate P with X, Y: state {X, Y} dropped")
    # Two parts touching only through a standard s (bolt), s holding both A and B present: removing
    # A or B first breaks s's holds (invalid T); removing s first leaves {A, B} disconnected (D11,
    # since they only ever touched through s) -- every first move is blocked, root is a dead end.
    free2 = np.ones((2, 2, 6), dtype=bool)
    bridge = contact_from_pairs(3, [(0, 2), (1, 2)])
    with LogCapture() as cap:
        aog = ag.generate_aog(make_inputs(free2, standard_options={2: AXES[4]}, standard_holds={2: {0, 1}},
                                          contact=bridge, ids=["A", "B", "s"]))
    rec.check(aog.n_nodes == 1 and list(aog.dead_ends) == [0] and not aog.complete,
              f"A, B held only by standard s (holds {{A, B}}): root is a dead end (V={aog.n_nodes})")
    rec.check(any("dead-end" in r.getMessage() for r in cap.records), "dead end reported, not resolved")
    # Full assembly not connected: warning, no mixed state reachable.
    with LogCapture() as cap:
        aog = ag.generate_aog(make_inputs(free, ids=["A", "B", "C"], contact=contact_from_pairs(3, [(A, B)])))
    rec.check(any("not one connected group" in r.getMessage() for r in cap.records),
              "disconnected full assembly logs a warning")
    rec.check(not _disconnected_nodes(aog, contact_from_pairs(3, [(A, B)]))[1:],
              "and no other disconnected state is generated")
    # Every node of the fastener AoG and of random graphs is connected.
    inputs, aog = fastener_cached()
    bad = _disconnected_nodes(aog, inputs.contact)
    rec.check(not bad, f"industrial_fastner: all {aog.n_nodes} states connected "
              f"({aog.stats['disconnected_dropped']} dropped, {len(bad)} violations)")
    rng = np.random.default_rng(42)
    violations = 0
    for _ in range(50):
        inputs = random_inputs(rng, 9)
        if inputs.contact is None:
            continue
        aog = ag.generate_aog(inputs)
        violations += len([v for v in _disconnected_nodes(aog, inputs.contact) if v != aog.root])
    rec.check(violations == 0, f"random graphs: no disconnected non-root state ({violations} violations)")


def _path_count(aog) -> int:
    """Number of root -> terminal paths (DP over the layered DAG)."""
    if aog.terminal is None:
        return 0
    paths = [0] * aog.n_nodes
    paths[aog.terminal] = 1
    for v in range(aog.n_nodes - 1, -1, -1):
        for e in aog.out_edges(v):
            paths[v] += paths[int(aog.edge_dst[e])]
    return paths[aog.root]


def _group_violations(aog, groups: list[set[int]]) -> int:
    """Edges that move a non-member while some group is partly removed (state-level D12 check)."""
    bad = 0
    for e in range(aog.n_edges):
        present = set(aog.state_components(int(aog.edge_src[e])))
        c = int(aog.edge_comp[e])
        bad += any(0 < len(g & present) < len(g) and c not in g for g in groups)
    return bad


def _blocks_contiguous(order: list[int], g: set[int]) -> bool:
    pos = [i for i, c in enumerate(order) if c in g]
    return pos == list(range(pos[0], pos[0] + len(pos)))


def test_a13(rec, args):
    """A13 - BOM groups: members are assembled consecutively (D12) in one order (D13).

    Tests: the group rules of `generate_aog` (`AoGConfig.group_consecutive`, `group_one_order`) and
    `read_bom_groups`.
    What: 8 free parts with a group of 3 give 40320 -> 4320 -> 720 paths as the rules are switched on, with the
    exact node/edge counts; no edge moves a non-member while a group is partly removed; sampled paths remove
    the group contiguously in one order; the first feasible member is used when the lowest index is blocked; a
    group that cannot be consecutive makes the graph incomplete (no fallback) unless the rule is off;
    group_one_order without group_consecutive raises; industrial_fastner has no multi-member group so grouped
    and ungrouped graphs are identical.
    """
    N = 8
    free = np.ones((N, N, 6), dtype=bool)
    grp = [None, None, "M4", None, "M4", None, "M4", None]
    g = {2, 4, 6}
    full = ag.generate_aog(make_inputs(free, group=grp), AoGConfig(group_consecutive=False, group_one_order=False))
    cons = ag.generate_aog(make_inputs(free, group=grp), AoGConfig(group_one_order=False))
    one = ag.generate_aog(make_inputs(free, group=grp))
    rec.check(_path_count(full) == 40320 and _path_count(cons) == 4320 and _path_count(one) == 720,
              f"8 free parts, group of 3: paths {_path_count(full)} -> consecutive {_path_count(cons)} (6!*3!) "
              f"-> one order {_path_count(one)} (6!)")
    rec.check((cons.n_nodes, cons.n_edges, one.n_nodes, one.n_edges) == (256, 544, 128, 256),
              f"sizes: consecutive V={cons.n_nodes}, E={cons.n_edges}; one order V={one.n_nodes}, E={one.n_edges} "
              f"(no groups: V={full.n_nodes}, E={full.n_edges})")
    rec.check(_group_violations(cons, [g]) == 0 and _group_violations(one, [g]) == 0,
              "no edge moves another part while the group is partly removed")
    rng = np.random.default_rng(13)
    bad = 0
    for path in _sample_paths(one, rng, 200):
        order = [step["component"] for step in one.sequence(path)]
        bad += not _blocks_contiguous(order, g) or [c for c in order if c in g] != [2, 4, 6]
    rec.check(bad == 0, f"200 sampled paths: group removed consecutively, in one order 2, 4, 6 ({bad} bad)")
    sub = module_sets(one)[1] <= module_sets(cons)[1] <= module_sets(full)[1]
    rec.check(sub, "one-order graph is a subgraph of the consecutive graph, which is a subgraph of the ungrouped one")

    # First feasible member: group {A, B}, A can only leave after B. Strict index order would be stuck.
    free2 = np.ones((2, 2, 6), dtype=bool)
    free2[0, 1, :] = False
    aog = ag.generate_aog(make_inputs(free2, ids=["A", "B"], group=["G", "G"]))
    order = [s["component_label"] for s in aog.sequence(_sample_paths(aog, rng, 1)[0])] if aog.complete else []
    rec.check(aog.complete and order == ["B", "A"], f"group {{A, B}} with A behind B: complete, order {order}")

    # Infeasible: group {A, C} but B must leave between them (C before B before A).
    free3 = np.ones((3, 3, 6), dtype=bool)
    free3[0, 1, :] = False  # A blocked by B
    free3[1, 2, :] = False  # B blocked by C
    with LogCapture() as cap:
        aog = ag.generate_aog(make_inputs(free3, ids=["A", "B", "C"], group=["G", None, "G"]))
    rec.check(not aog.complete and len(aog.dead_ends) > 0,
              f"group that cannot be consecutive: incomplete, dead ends {list(aog.dead_ends)} (fails, no fallback)")
    rec.check(any("group rule" in r.getMessage() for r in cap.records), "incomplete warning names the group rule")
    aog = ag.generate_aog(make_inputs(free3, ids=["A", "B", "C"], group=["G", None, "G"]),
                          AoGConfig(group_consecutive=False, group_one_order=False))
    rec.check(aog.complete, "same case with the group rule off: complete")
    try:
        ag.generate_aog(make_inputs(free3, group=["G", None, "G"]), AoGConfig(group_consecutive=False))
        rec.check(False, "group_one_order without group_consecutive should raise")
    except ValueError:
        rec.check(True, "group_one_order without group_consecutive raises ValueError")

    # industrial_fastner: the current generated_bom.csv gives every component its own `group`
    # value (no shared group of >= 2 members any more -- unlike the older bom.csv this file once
    # referenced), so D12/D13 has no effect on this fixture. Confirm that directly rather than
    # asserting a specific grouping the committed BOM no longer defines.
    inputs, aog = fastener_cached()
    plain = ag.generate_aog(inputs, AoGConfig(group_consecutive=False, group_one_order=False))
    rec.check(aog.stats["groups"] == {}, f"no BOM group with >= 2 members in this fixture ({aog.stats['groups']})")
    rec.check(module_sets(aog) == module_sets(plain),
              f"with no multi-member group, grouped (V={aog.n_nodes}, E={aog.n_edges}) and ungrouped "
              f"(V={plain.n_nodes}, E={plain.n_edges}) graphs are identical")
    if "real" in args.groups:
        _, _, real = fastener_real()
        rec.check(same_graph_up_to_directions(real, aog) and real.stats["groups"] == aog.stats["groups"],
                  f"real run reads the same groups from the BOM and gives the same graph ({real.stats['groups']})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("A1", test_a1), ("A2", test_a2), ("A2b", test_a2b), ("A3", test_a3), ("A4", test_a4), ("A5", test_a5),
    ("A6", test_a6), ("A7", test_a7), ("A8", test_a8), ("A9", test_a9), ("A10", test_a10), ("A11", test_a11),
    ("A12", test_a12), ("A13", test_a13),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="*", help="individual test ids, e.g. A1 A6")
    ap.add_argument("--groups", nargs="*", default=["core"], choices=["core", "real", "scale"],
                    help="core (default), real (FreeCAD run on industrial_fastner), scale (A5 at N = 20)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="    %(name)s %(levelname)s: %(message)s")

    rec = Recorder()
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

    print("\n| Test | Result | Notes |\n|---|---|---|")
    for test_id, r in rec.results.items():
        status = "PASS" if r["passed"] else ("ERROR" if "error" in r else "FAIL")
        failed = [c["label"] for c in r["checks"] if not c["passed"]]
        notes = "; ".join(failed[:2] + r["notes"][:1] + ([r["error"]] if "error" in r else []))
        print(f"| {test_id} | {status} | {notes} |")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "aog_results.json"
    out.write_text(json.dumps({"groups": args.groups, "results": rec.results}, indent=2))
    print(f"\nDetails written to {out}")
    return 0 if all(r["passed"] for r in rec.results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
