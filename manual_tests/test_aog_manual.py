"""Manual smoke test and visualiser for the AoG generator (src/aog_generator.py).

Counterpart of manual_tests/test_preprocessing_manual.py and manual_tests/test_interference_manual.py. It builds the AoG,
runs a set of independent consistency checks on it, and writes the graph in forms you can look at:

  <out_dir>/aog/
    aog_graph.html        the graph, self-contained: open it in any browser (no server, no internet).
                          One row per level (components removed so far), one node per state, one edge
                          per removal move. Hover for details, click a node to highlight everything
                          connected to it, click a legend chip to hide a component's edges, wheel to
                          zoom, drag to pan, "Example sequence" to highlight one full disassembly.
    aog_report.json       readable report: components, direction table, states per level, the moves
                          available with everything assembled, dead ends, an example sequence, checks
    aog_sequence.txt      the example sequence as text (removal and the reversed assembly order)
    aog_nodes.csv, aog_edges.csv, aog.dot, aog.npz, aog.json
                          written by the module itself (small graphs only for the csv/dot files)

Checks (independent of the module's bitmask code where possible):
  S1 structure       layered DAG, dst = src minus one bit, sorted, CSR consistent
  S2 brute force     every state and move recomputed naively from the interference tensor
  S3 standers        gate rule on every state, stander move present wherever the stander is
  S4 directions      neg table consistent, stander edges carry exactly their own direction
  S5 travel          travel table recomputed cell by cell, plus two hand-computed cases
  S6 paths           random root-to-terminal walks replayed, assembly order reversed correctly
  S7 live / dead     reachability recomputed, dead ends match
  S8 persistence     save / load round trip and cache-key mismatch
  S9 determinism     a second generation is identical
  S10 cache key      changes when a gate set or the config changes

Run (from repo root, environmentAAP active) on the real assembly:

    python manual_tests/test_aog_manual.py

or on a hand-built assembly that needs no FreeCAD and no STEP file (good for seeing what the
graph looks like):

    python manual_tests/test_aog_manual.py --synthetic demo        # 4 parts + 2 screws, a stack with a side block
    python manual_tests/test_aog_manual.py --synthetic stack       # A on B on C: a single chain
    python manual_tests/test_aog_manual.py --synthetic lattice --n 5   # 5 independent parts: every subset
    python manual_tests/test_aog_manual.py --synthetic random --n 6 --m 2 --seed 3   # random blocking, dead ends

Add --open to open the HTML when it is written. The real run reuses the cached interference matrix
(--force-interference recomputes it) and the cached AoG (--force-aog regenerates it).

Expected result: every check S1-S10 prints PASS and the exit code is 0 (S2 is skipped when N is above
--brute-max-n). The console gives the number of states and moves, whether the AoG is complete (root live) and its
dead ends. For industrial_fastner the AoG is complete with no dead end (V = 61 states, E = 141 moves). The
synthetic 'stack' is a single chain, 'lattice --n 5' has all 2^5 = 32 states, 'random' may end with dead ends.
aog_graph.html opens in a browser and the example sequence (aog_sequence.txt) removes every part exactly once.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import tempfile
import time
import webbrowser
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from aog_generator import (  # noqa: E402
    AoG,
    AoGConfig,
    AoGInputs,
    build_aog,
    collect_inputs,
    compute_travel_table,
    generate_aog,
    load_aog,
    save_aog,
)

DEFAULT_STEP = REPO_ROOT / "data" / "industrial_fastner" / "industrial_fastner.stp"
DEFAULT_BOM = REPO_ROOT / "data" / "industrial_fastner" / "generated_bom.csv"
DEFAULT_OUT = REPO_ROOT / "output" / "industrial_fastner"
DEFAULT_H = 5.0

HTML_MAX_NODES = 1500  # beyond this the SVG gets too heavy to be useful
HTML_MAX_EDGES = 6000
BRUTE_MAX_N = 12  # the naive reference enumerates subsets in Python


def _vec(v) -> list[float]:
    return [round(float(c), 6) + 0.0 for c in v]


def _bits(mask: int, n: int) -> list[int]:
    return [c for c in range(n) if (mask >> c) & 1]


# ---------------------------------------------------------------------------
# Synthetic assemblies (no FreeCAD): the same AoGInputs the pipeline adapter would build
# ---------------------------------------------------------------------------

AXES = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float
)
AXNEG = np.array([1, 0, 3, 2, 5, 4])
XP, XN, YP, YN, ZP, ZN = range(6)


def _make_inputs(labels, n, blocked, standers, boxes, key) -> AoGInputs:
    """blocked: (i, j, k) triples meaning "i cannot leave past j along axis k" (the antisymmetric
    twin is added). standers: list of (label, unit direction, holds set of movable indices, box) --
    one RemovalOption each, no blockers of its own (these demos only model holds); holds = movable
    indices that must stay present while the stander leaves (rule P6)."""
    free = np.ones((n, n, len(AXES)), dtype=bool)
    for i, j, k in blocked:
        free[i, j, k] = False
        free[j, i, AXNEG[k]] = False
    for i in range(n):
        free[i, i, :] = True
    all_labels = list(labels) + [s[0] for s in standers]
    lo = np.array([b[0] for b in boxes] + [s[3][0] for s in standers], dtype=float)
    hi = np.array([b[1] for b in boxes] + [s[3][1] for s in standers], dtype=float)
    return AoGInputs(
        component_ids=[f"Part__{lab}" for lab in all_labels],
        component_labels=all_labels,
        n_movable=n,
        free=free,
        directions=AXES.copy(),
        neg=AXNEG.copy(),
        stander_options={n + q: [(np.asarray(s[1], dtype=float), set())] for q, s in enumerate(standers)},
        stander_holds={n + q: set(s[2]) for q, s in enumerate(standers)},
        nut_links={},
        bbox_lo=lo,
        bbox_hi=hi,
        assembly_lo=lo.min(axis=0),
        assembly_hi=hi.max(axis=0),
        cache_key=key,
    )


def synthetic_stack() -> AoGInputs:
    """A lies on B lies on C. Only A can leave first; then B; then C. Sideways is free."""
    blocked = [(0, 1, ZN), (0, 2, ZN), (1, 2, ZN)]
    boxes = [([20, 20, 30], [80, 80, 40]), ([10, 10, 10], [90, 90, 30]), ([0, 0, 0], [100, 100, 10])]
    return _make_inputs(["A (top)", "B", "C (base)"], 3, blocked, [], boxes, "synthetic-stack")


def synthetic_demo() -> AoGInputs:
    """Base plate P, bracket B on P, cover C on B, side block S next to B; screw1 holds C to B
    (leaves upwards), screw2 holds B to P (leaves downwards). Screws must come off first."""
    P, B, C, S = 0, 1, 2, 3
    blocked = [
        (B, P, ZN), (C, B, ZN), (C, P, ZN),  # the stack: nothing goes down through what is below
        (S, B, XN), (S, P, ZN),              # the side block cannot move into B or down into P
    ]
    boxes = [
        ([0, 0, 0], [100, 100, 10]),    # P
        ([10, 10, 10], [90, 90, 30]),   # B
        ([20, 20, 30], [80, 80, 40]),   # C
        ([90, 20, 10], [110, 80, 30]),  # S
    ]
    standers = [
        ("screw1", [0, 0, 1], {C, B}, ([48, 48, 20], [52, 52, 45])),
        ("screw2", [0, 0, -1], {B, P}, ([48, 48, -5], [52, 52, 20])),
    ]
    return _make_inputs(["P (plate)", "B (bracket)", "C (cover)", "S (side)"], 4, blocked, standers,
                        boxes, "synthetic-demo")


def synthetic_lattice(n: int) -> AoGInputs:
    """n parts that never block each other: the AoG is the full subset lattice (2^n states)."""
    boxes = [([12 * i, 0, 0], [12 * i + 10, 10, 10]) for i in range(n)]
    return _make_inputs([f"p{i}" for i in range(n)], n, [], [], boxes, f"synthetic-lattice-{n}")


def synthetic_random(n: int, m: int, seed: int, p_block: float = 0.3) -> AoGInputs:
    """Random blocking (may create dead ends) and random stander gates."""
    rng = random.Random(seed)
    blocked = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(6)
               if rng.random() < p_block]
    boxes = []
    for _ in range(n):
        lo = [rng.uniform(0, 80) for _ in range(3)]
        boxes.append((lo, [c + rng.uniform(10, 30) for c in lo]))
    standers = []
    for q in range(m):
        lo = [rng.uniform(0, 80) for _ in range(3)]
        gate = set(rng.sample(range(n), k=min(n, rng.choice([1, 2]))))
        standers.append((f"screw{q}", AXES[rng.randrange(6)], gate, (lo, [c + 5 for c in lo])))
    return _make_inputs([f"p{i}" for i in range(n)], n, blocked, standers, boxes,
                        f"synthetic-random-{n}-{m}-{seed}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _result(cid, name, problems, detail_ok="") -> dict:
    return {"id": cid, "name": name, "passed": not problems,
            "detail": detail_ok if not problems else "; ".join(problems[:8])
            + (f" (+{len(problems) - 8} more)" if len(problems) > 8 else "")}


def check_structure(aog: AoG) -> dict:
    N, V, E = aog.n_components, aog.n_nodes, aog.n_edges
    p: list[str] = []
    full = (1 << N) - 1
    mask = aog.node_mask.astype(np.uint64)
    if int(mask[aog.root]) != full:
        p.append("root is not the full assembly")
    if aog.terminal is not None and int(mask[aog.terminal]) != 0:
        p.append("terminal mask is not 0")
    if len(np.unique(mask)) != V:
        p.append("duplicate node masks")
    pc = sum(((mask >> np.uint64(b)) & np.uint64(1)).astype(np.int64) for b in range(N))
    if np.any(pc != N - aog.node_level.astype(np.int64)):
        p.append("popcount != N - level for some node")
    for L in range(N + 1):
        a, b = int(aog.level_offsets[L]), int(aog.level_offsets[L + 1])
        if b - a > 1 and not np.all(mask[a + 1:b] > mask[a:b - 1]):
            p.append(f"level {L} not strictly sorted")
    if E:
        src, dst = aog.edge_src.astype(np.int64), aog.edge_dst.astype(np.int64)
        bit = np.uint64(1) << aog.edge_comp.astype(np.uint64)
        sm, dm = mask[src], mask[dst]
        if np.any((sm & bit) == 0):
            p.append("an edge removes a component that is not present")
        if np.any(dm != (sm & ~bit)):
            p.append("dst != src minus the removed bit")
        if np.any(aog.node_level[dst] != aog.node_level[src] + 1):
            p.append("an edge does not go exactly one level down")
        if np.any(np.diff(src) < 0):
            p.append("edges not sorted by src")
        same = np.diff(src) == 0
        if np.any(same & (np.diff(aog.edge_comp.astype(np.int64)) <= 0)):
            p.append("edges of one node not strictly sorted by component (or duplicated)")
        if np.any(aog.edge_dirs == 0):
            p.append("an edge with no free direction")
        if np.any(np.diff(aog.out_ptr) != np.bincount(src, minlength=V)):
            p.append("out_ptr inconsistent with edge_src")
    return _result("S1", "structure", p, f"{V} nodes, {E} edges, layered DAG, CSR consistent")


def _stander_dir0(inputs: AoGInputs, s: int) -> np.ndarray:
    """First (and, for every demo this script builds, only) option's direction of stander `s`."""
    return np.asarray(inputs.stander_options[s][0][0], dtype=float)


def _expected_stander_dir(aog: AoG, inputs: AoGInputs, s: int) -> int:
    d = _stander_dir0(inputs, s)
    d = d / np.linalg.norm(d)
    return int(np.flatnonzero(np.linalg.norm(aog.directions - d, axis=1) < 1e-6)[0])


def check_bruteforce(aog: AoG, inputs: AoGInputs, max_n: int) -> dict:
    N, n = aog.n_components, inputs.n_movable
    if N > max_n:
        return {"id": "S2", "name": "brute force", "passed": True, "skipped": True,
                "detail": f"skipped (N={N} > {max_n})"}
    free, D = inputs.free, inputs.free.shape[2]
    gates = {s: frozenset(inputs.stander_holds.get(s, ())) for s in range(n, N)}
    sdir = {s: _expected_stander_dir(aog, inputs, s) for s in range(n, N)}

    def valid(S):
        return all(gates[s] <= S for s in S if s >= n)

    full = frozenset(range(N))
    ref_nodes, ref_edges, stack = {full}, {}, [full]
    while stack:
        S = stack.pop()
        for c in sorted(S):
            if c < n:
                rest = [j for j in S if j < n and j != c]
                dirs = frozenset(int(k) for k in np.flatnonzero(np.all(free[c, rest, :], axis=0)))
                assert D == free.shape[2]
            else:
                if not gates[c] <= S:
                    continue
                dirs = frozenset({sdir[c]})
            if not dirs:
                continue
            T = S - {c}
            if not valid(T):
                continue
            ref_edges[(S, c)] = dirs
            if T not in ref_nodes:
                ref_nodes.add(T)
                stack.append(T)

    got_nodes = {frozenset(_bits(int(m), N)) for m in aog.node_mask}
    got_edges = {}
    for e in range(aog.n_edges):
        S = frozenset(_bits(int(aog.node_mask[aog.edge_src[e]]), N))
        got_edges[(S, int(aog.edge_comp[e]))] = frozenset(int(k) for k in aog.edge_direction_indices(e))

    p = []
    if got_nodes != ref_nodes:
        p.append(f"node sets differ: {len(got_nodes - ref_nodes)} extra, {len(ref_nodes - got_nodes)} missing")
    missing = ref_edges.keys() - got_edges.keys()
    extra = got_edges.keys() - ref_edges.keys()
    if missing or extra:
        p.append(f"edge sets differ: {len(extra)} extra, {len(missing)} missing")
    wrong = [k for k in ref_edges.keys() & got_edges.keys() if ref_edges[k] != got_edges[k]]
    if wrong:
        S, c = wrong[0]
        p.append(f"{len(wrong)} edges with different direction sets, e.g. remove {aog.component_labels[c]} "
                 f"from {sorted(S)}: module {sorted(got_edges[(S, c)])} vs reference {sorted(ref_edges[(S, c)])}")
    return _result("S2", "brute force", p,
                   f"{len(ref_nodes)} states and {len(ref_edges)} moves identical to the naive reference")


def check_standers(aog: AoG, inputs: AoGInputs) -> dict:
    N, n, V = aog.n_components, inputs.n_movable, aog.n_nodes
    if N == n:
        return {"id": "S3", "name": "standers", "passed": True, "skipped": True, "detail": "no standers"}
    mask = aog.node_mask.astype(np.uint64)
    p = []
    for s in range(n, N):
        gate = np.uint64(sum(1 << g for g in inputs.stander_holds.get(s, ())))
        present = ((mask >> np.uint64(s)) & np.uint64(1)).astype(bool)
        has_move = np.bincount(aog.edge_src[aog.edge_comp == s].astype(np.int64), minlength=V) > 0
        label = aog.component_labels[s]
        if np.any(present & ((mask & gate) != gate)):
            p.append(f"{label}: present in a state without its whole gate")
        if np.any(present & ~has_move):
            p.append(f"{label}: no removal move in a state where it is present")
        if np.any(~present & has_move):
            p.append(f"{label}: removal move in a state where it is absent")
    return _result("S3", "standers", p, f"{N - n} standers: gate holds in all {V} states, "
                   "each stander removable wherever present")


def check_directions(aog: AoG, inputs: AoGInputs) -> dict:
    p = []
    d, neg = aog.directions, aog.neg
    n, D = inputs.n_movable, inputs.free.shape[2]
    if np.any(np.linalg.norm(d[neg] + d, axis=1) > 1e-6):
        p.append("directions[neg[k]] != -directions[k]")
    if np.any(neg[neg] != np.arange(len(neg))):
        p.append("neg is not an involution")
    if np.any(np.abs(np.linalg.norm(d, axis=1) - 1) > 1e-6):
        p.append("a direction is not a unit vector")
    if not np.allclose(d[:D], inputs.directions):
        p.append("candidate directions were changed or reordered")
    words = aog.edge_dirs[:, 0]
    comp = aog.edge_comp.astype(np.int64)
    candidate_bits = np.uint64((1 << D) - 1)
    movable = comp < n
    if np.any(movable & ((words & ~candidate_bits) != 0)):
        p.append("a movable part's edge carries a non-candidate direction")
    for s in range(n, aog.n_components):
        want = np.uint64(1) << np.uint64(_expected_stander_dir(aog, inputs, s))
        if np.any((comp == s) & (words != want)):
            p.append(f"stander {aog.component_labels[s]}: an edge does not carry exactly its own direction")
    return _result("S4", "directions", p, f"{len(d)} directions ({len(d) - D} appended for standers), "
                   "neg table consistent, stander edges carry their own direction")


def _travel_ref(lo, hi, blo, bhi, d, margin) -> float:
    start, end = -np.inf, np.inf
    for a in range(3):
        if abs(d[a]) > 1e-12:
            t1, t2 = (blo[a] - hi[a]) / d[a], (bhi[a] - lo[a]) / d[a]
            s, e = min(t1, t2), max(t1, t2)
        elif lo[a] < bhi[a] and hi[a] > blo[a]:
            s, e = -np.inf, np.inf
        else:
            s, e = np.inf, -np.inf
        start, end = max(start, s), min(end, e)
    return (max(0.0, end) if start < end else 0.0) + margin


def check_travel(aog: AoG, inputs: AoGInputs) -> dict:
    p = []
    N, Dp = aog.n_components, len(aog.directions)
    for c in range(N):
        for k in range(Dp):
            ref = _travel_ref(inputs.bbox_lo[c], inputs.bbox_hi[c], inputs.assembly_lo,
                              inputs.assembly_hi, aog.directions[k], aog.config.margin)
            if not np.isclose(aog.travel[c, k], ref, rtol=1e-9, atol=1e-9):
                p.append(f"travel[{c},{k}] = {aog.travel[c, k]:.6f}, reference {ref:.6f}")
    if np.any(aog.travel < aog.config.margin - 1e-12):
        p.append("a travel value is below the margin")
    # Hand-computed: a 10 mm cube at the corner of a 100 mm assembly box, margin 1.
    lo, hi = np.zeros((1, 3)), np.full((1, 3), 10.0)
    t = compute_travel_table(lo, hi, np.zeros(3), np.full(3, 100.0),
                             np.array([[1.0, 0, 0], [-1.0, 0, 0]]), 1.0)
    if not np.allclose(t[0], [101.0, 11.0]):  # +x: 100 to pass the far face; -x: 10 to clear the near face
        p.append(f"hand case: got {t[0].tolist()}, expected [101.0, 11.0]")
    return _result("S5", "travel", p, f"{N * Dp} cells match the reference; hand cases OK; all >= margin "
                   f"({aog.config.margin} mm); range {aog.travel.min():.2f}..{aog.travel.max():.2f} mm")


def check_paths(aog: AoG, inputs: AoGInputs, samples: int, seed: int = 0) -> dict:
    if not aog.complete:
        return {"id": "S6", "name": "paths", "passed": True, "skipped": True,
                "detail": "skipped (AoG is incomplete)"}
    N, n = aog.n_components, inputs.n_movable
    rng = random.Random(seed)
    gates = {s: set(inputs.stander_holds.get(s, ())) for s in range(n, N)}
    p, walked = [], 0
    for _ in range(samples):
        v, path = aog.root, []
        while v != aog.terminal:
            live_out = [e for e in aog.out_edges(v) if aog.live[aog.edge_dst[e]]]
            if not live_out:
                p.append(f"live node {v} has no live out-edge")
                break
            e = rng.choice(live_out)
            path.append(e)
            v = int(aog.edge_dst[e])
        else:
            walked += 1
            state = set(range(N))
            for e in path:
                c = int(aog.edge_comp[e])
                if c not in state:
                    p.append("path removes a component twice")
                    break
                rest = state - {c}
                if c < n:
                    for k in aog.edge_direction_indices(e):
                        if not all(inputs.free[c, j, k] for j in rest if j < n):
                            p.append(f"path step remove {aog.component_labels[c]} along {k} is blocked")
                else:
                    if not gates[c] <= state:
                        p.append(f"stander {aog.component_labels[c]} removed without its gate")
                state = rest
                if any(not gates[s] <= state for s in state if s >= n):
                    p.append("a stander is left without its gate")
            asm = aog.assembly_sequence(path)
            order = [st["component"] for st in asm]
            if sorted(order) != list(range(N)):
                p.append("assembly sequence is not a permutation of the components")
            pos = {c: i for i, c in enumerate(order)}
            for s, g in gates.items():
                if any(pos[j] > pos[s] for j in g):
                    p.append(f"assembly order places stander {aog.component_labels[s]} before its gate")
            for st, e in zip(asm, reversed(path)):
                ks = aog.edge_direction_indices(e)
                if not np.allclose(np.array(st["directions"]), -aog.directions[ks]):
                    p.append("assembly directions are not the negated removal directions")
                    break
        if len(p) > 8:
            break
    return _result("S6", "paths", p, f"{walked} random root-to-terminal walks replayed independently; "
                   "assembly order valid, standers after their gates")


def check_live(aog: AoG) -> dict:
    """Reachability of the terminal, recomputed level by level with a different reduction."""
    V = aog.n_nodes
    live = np.zeros(V, dtype=bool)
    if aog.terminal is not None:
        live[aog.terminal] = True
    for L in range(aog.n_components - 1, -1, -1):
        a, b = int(aog.level_offsets[L]), int(aog.level_offsets[L + 1])
        ptr = aog.out_ptr[a:b + 1]
        e0, e1 = int(ptr[0]), int(ptr[-1])
        if e0 == e1:
            continue
        has = ptr[1:] > ptr[:-1]
        starts = (ptr[:-1] - e0)[has]
        vals = live[aog.edge_dst[e0:e1]].astype(np.uint8)
        live[a:b][has] = np.maximum.reduceat(vals, starts).astype(bool)
    dead = np.flatnonzero((aog.node_mask != 0) & (aog.out_ptr[1:] == aog.out_ptr[:-1]))
    p = []
    if not np.array_equal(live, aog.live):
        p.append(f"live flags differ in {int((live != aog.live).sum())} nodes")
    if not np.array_equal(dead, aog.dead_ends):
        p.append("dead-end list differs")
    if aog.complete != bool(live[aog.root]):
        p.append("`complete` disagrees with the root's live flag")
    return _result("S7", "live / dead ends", p,
                   f"{int(live.sum())}/{V} nodes live, {len(dead)} dead ends, complete={aog.complete}")


def check_persistence(aog: AoG) -> dict:
    p = []
    with tempfile.TemporaryDirectory() as tmp:
        save_aog(aog, tmp)
        loaded = load_aog(tmp, aog.cache_key)
        if loaded is None:
            p.append("load_aog returned None for a matching key")
        else:
            for name in ("node_mask", "node_level", "level_offsets", "edge_src", "edge_dst", "edge_comp",
                         "edge_dirs", "out_ptr", "live", "dead_ends", "travel", "directions", "neg"):
                if not np.array_equal(getattr(aog, name), getattr(loaded, name)):
                    p.append(f"array {name} differs after a round trip")
            if (loaded.complete, loaded.root, loaded.terminal) != (aog.complete, aog.root, aog.terminal):
                p.append("scalar fields differ after a round trip")
        if load_aog(tmp, "0" * 64) is not None:
            p.append("load_aog accepted a wrong cache key")
    return _result("S8", "persistence", p, "save/load round trip identical; wrong key rejected")


def check_determinism(aog: AoG, inputs: AoGInputs) -> dict:
    again = generate_aog(inputs, aog.config)
    p = [f"array {name} differs" for name in ("node_mask", "edge_src", "edge_dst", "edge_comp", "edge_dirs",
                                             "out_ptr", "live", "travel")
         if not np.array_equal(getattr(aog, name), getattr(again, name))]
    if again.cache_key != aog.cache_key:
        p.append("cache key differs between runs")
    return _result("S9", "determinism", p, "a second generation gives identical arrays and key")


def check_cache_key(aog: AoG, inputs: AoGInputs) -> dict:
    p = []
    n, N = inputs.n_movable, len(inputs.component_ids)
    key = aog.cache_key
    from aog_generator import compute_cache_key

    if compute_cache_key(inputs, replace(aog.config, margin=aog.config.margin + 1.0)) == key:
        p.append("changing the margin did not change the key")
    if N > n:
        s = n
        gate = set(inputs.stander_holds.get(s, ()))
        other = set(range(n)) - gate
        new_gate = (gate | {next(iter(other))}) if other else (gate - {next(iter(gate))} if gate else gate)
        if new_gate != gate:
            changed = replace(inputs, stander_holds={**inputs.stander_holds, s: new_gate})
            if compute_cache_key(changed, aog.config) == key:
                p.append("changing a holds set did not change the key")
        d0, blockers0 = inputs.stander_options[s][0]
        moved_options = {**inputs.stander_options, s: [(-np.asarray(d0, dtype=float), blockers0)]}
        moved = replace(inputs, stander_options=moved_options)
        if compute_cache_key(moved, aog.config) == key:
            p.append("changing a stander direction did not change the key")
    return _result("S10", "cache key", p, "key changes with the config, holds sets and stander directions")


def run_checks(aog: AoG, inputs: AoGInputs, samples: int, brute_max_n: int) -> list[dict]:
    return [
        check_structure(aog),
        check_bruteforce(aog, inputs, brute_max_n),
        check_standers(aog, inputs),
        check_directions(aog, inputs),
        check_travel(aog, inputs),
        check_paths(aog, inputs, samples),
        check_live(aog),
        check_persistence(aog),
        check_determinism(aog, inputs),
        check_cache_key(aog, inputs),
    ]


# ---------------------------------------------------------------------------
# Example sequence, layout, report
# ---------------------------------------------------------------------------


def edge_min_travel(aog: AoG) -> tuple[np.ndarray, np.ndarray]:
    """Per edge: the smallest travel over its free directions and the direction that gives it."""
    E = aog.n_edges
    best = np.full(E, np.inf)
    arg = np.full(E, -1, dtype=np.int64)
    words = aog.edge_dirs[:, 0]
    for k in range(len(aog.directions)):
        has = ((words >> np.uint64(k)) & np.uint64(1)).astype(bool)
        t = aog.travel[aog.edge_comp.astype(np.int64), k]
        better = has & (t < best - 1e-12)
        best[better] = t[better]
        arg[better] = k
    return best, arg


def example_sequence(aog: AoG) -> list[int]:
    """Cheapest full disassembly by total travel, each step using its shortest direction.

    Only an illustration: it ignores direction continuity, which is the ACO's job."""
    if not aog.complete:
        return []
    cost, _ = edge_min_travel(aog)
    best = np.full(aog.n_nodes, np.inf)
    best[aog.terminal] = 0.0
    for L in range(aog.n_components - 1, -1, -1):
        a, b = int(aog.out_ptr[aog.level_offsets[L]]), int(aog.out_ptr[aog.level_offsets[L + 1]])
        if a < b:
            np.minimum.at(best, aog.edge_src[a:b], cost[a:b] + best[aog.edge_dst[a:b]])
    path, v = [], aog.root
    while v != aog.terminal:
        cands = [e for e in aog.out_edges(v)
                 if np.isclose(cost[e] + best[aog.edge_dst[e]], best[v], rtol=1e-9, atol=1e-9)]
        e = cands[0]
        path.append(e)
        v = int(aog.edge_dst[e])
    return path


def layout_ranks(aog: AoG, sweeps: int = 6) -> np.ndarray:
    """Position of every node inside its level (0..size-1), ordered by barycentre sweeps to keep
    edge crossings low."""
    N, V = aog.n_components, aog.n_nodes
    parents = [[] for _ in range(V)]
    children = [[] for _ in range(V)]
    for s, d in zip(aog.edge_src.tolist(), aog.edge_dst.tolist()):
        children[s].append(d)
        parents[d].append(s)
    levels = [list(range(int(aog.level_offsets[L]), int(aog.level_offsets[L + 1]))) for L in range(N + 1)]
    pos = np.zeros(V)
    for nodes in levels:
        for i, v in enumerate(nodes):
            pos[v] = (i + 0.5) / max(1, len(nodes))

    def sweep(order, nbrs):
        for nodes in order:
            if len(nodes) < 2:
                continue
            bc = {v: (np.mean([pos[u] for u in nbrs[v]]) if nbrs[v] else pos[v]) for v in nodes}
            nodes.sort(key=lambda v: (bc[v], v))
            for i, v in enumerate(nodes):
                pos[v] = (i + 0.5) / len(nodes)

    for _ in range(sweeps):
        sweep(levels[1:], parents)
        sweep(reversed(levels[:-1]), children)
    rank = np.zeros(V, dtype=np.int64)
    for nodes in levels:
        for i, v in enumerate(nodes):
            rank[v] = i
    return rank


def sequence_text(aog: AoG, path: list[int]) -> str:
    if not path:
        return "No complete sequence (the AoG is incomplete).\n"
    lines = ["Example disassembly (cheapest total travel; each step along its shortest direction).",
             "The ACO will choose directions itself; this only illustrates the graph.", ""]
    total = 0.0
    for st in aog.sequence(path):
        i = int(np.argmin(st["distances"]))
        d, dist = st["removal_directions"][i], st["distances"][i]
        total += dist
        alt = len(st["removal_direction_indices"])
        lines.append(f"  {st['step']:>2}. remove {st['component_label']:<14} along {tuple(_vec(d))}  "
                     f"travel {dist:8.2f} mm   ({alt} free direction{'s' if alt != 1 else ''})")
    lines += ["", f"Total travel: {total:.2f} mm", "", "Assembly order (reverse of the above):"]
    for st in aog.assembly_sequence(path):
        i = int(np.argmin(st["distances"]))
        lines.append(f"  {st['step']:>2}. insert {st['component_label']:<14} along "
                     f"{tuple(_vec(st['directions'][i]))}  starting {st['distances'][i]:.2f} mm outside")
    return "\n".join(lines) + "\n"


def build_report(aog: AoG, inputs: AoGInputs, checks: list[dict], path: list[int]) -> dict:
    N, n = aog.n_components, inputs.n_movable
    sizes = np.diff(aog.level_offsets)[: N + 1].tolist()
    root_moves = []
    for e in aog.out_edges(aog.root):
        c = int(aog.edge_comp[e])
        ks = aog.edge_direction_indices(e)
        root_moves.append({
            "component": aog.component_labels[c],
            "is_stander": bool(aog.is_stander[c]),
            "directions": [_vec(aog.directions[k]) for k in ks],
            "travel_mm": [round(float(aog.travel[c, k]), 3) for k in ks],
        })
    seq = aog.sequence(path) if path else []
    return {
        "components": [{"bit": c, "label": aog.component_labels[c], "id": aog.component_ids[c],
                        "kind": "stander" if aog.is_stander[c] else "movable"} for c in range(N)],
        "directions": [{"index": k, "vector": _vec(aog.directions[k]),
                        "neg": int(aog.neg[k]), "appended_for_standers": k >= inputs.free.shape[2]}
                       for k in range(len(aog.directions))],
        "stander_holds": {aog.component_labels[s]: sorted(aog.component_labels[g]
                                                          for g in inputs.stander_holds.get(s, ()))
                          for s in range(n, N)},
        "nodes": aog.n_nodes,
        "edges": aog.n_edges,
        "states_per_level": sizes,
        "complete": aog.complete,
        "dead_ends": [{"node": int(v), "present": [aog.component_labels[c] for c in aog.state_components(int(v))]}
                      for v in aog.dead_ends[:100]],
        "moves_with_everything_assembled": root_moves,
        "example_sequence": [{"step": s["step"], "component": s["component_label"],
                              "directions": [_vec(d) for d in s["removal_directions"]],
                              "travel_mm": [round(t, 3) for t in s["distances"]]} for s in seq],
        "stats": aog.stats,
        "cache_key": aog.cache_key,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# HTML visualisation
# ---------------------------------------------------------------------------


def build_payload(aog: AoG, rank: np.ndarray, path: list[int]) -> dict:
    N = aog.n_components
    dead = set(int(v) for v in aog.dead_ends)
    sizes = np.diff(aog.level_offsets)[: N + 1].tolist()
    nodes = [[int(aog.node_level[v]), int(rank[v]), int(aog.live[v]), int(v in dead),
              format(int(aog.node_mask[v]), "x")] for v in range(aog.n_nodes)]
    edges = []
    for e in range(aog.n_edges):
        c = int(aog.edge_comp[e])
        ks = aog.edge_direction_indices(e)
        edges.append([int(aog.edge_src[e]), int(aog.edge_dst[e]), c, [int(k) for k in ks],
                      [round(float(aog.travel[c, k]), 2) for k in ks]])
    return {
        "comps": [{"label": aog.component_labels[c], "stander": bool(aog.is_stander[c])} for c in range(N)],
        "dirs": [_vec(d) for d in aog.directions],
        "levelSizes": sizes,
        "nodes": nodes,
        "edges": edges,
        "sequence": path,
        "stats": {"N": N, "V": aog.n_nodes, "E": aog.n_edges, "dead": len(dead),
                  "complete": bool(aog.complete)},
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AoG graph</title>
<style>
  :root {
    --bg: #ffffff; --panel: #f6f7f9; --line: #d5d9e0; --text: #1c2330; --muted: #5d677a;
    --node: #8792a6; --node-stroke: #ffffff; --root: #1c2330; --terminal: #1f8a4c; --dead: #c8382e;
    --accent: #2a6fdb;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #12161d; --panel: #1b212b; --line: #2f3846; --text: #e6eaf1; --muted: #93a0b5;
      --node: #7f8ba0; --node-stroke: #12161d; --root: #e6eaf1; --terminal: #4cc383; --dead: #ff7166;
      --accent: #6aa3ff;
    }
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { background: var(--bg); color: var(--text); font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
         display: flex; flex-direction: column; }
  header { padding: 10px 16px; border-bottom: 1px solid var(--line); display: flex; flex-wrap: wrap;
           gap: 6px 20px; align-items: baseline; }
  header h1 { font-size: 16px; margin: 0; font-weight: 650; }
  header .stats { color: var(--muted); font-size: 13px; }
  header .bad { color: var(--dead); font-weight: 600; }
  main { flex: 1; display: flex; min-height: 0; }
  #stage { flex: 1; position: relative; min-width: 0; overflow: hidden; }
  svg { width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }
  svg.dragging { cursor: grabbing; }
  aside { width: 320px; border-left: 1px solid var(--line); background: var(--panel); overflow-y: auto;
          padding: 12px 14px; }
  aside h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);
             margin: 14px 0 6px; font-weight: 600; }
  aside h2:first-child { margin-top: 0; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 6px; }
  button { font: inherit; padding: 4px 10px; border: 1px solid var(--line); border-radius: 6px;
           background: var(--bg); color: var(--text); cursor: pointer; }
  button:hover { border-color: var(--accent); }
  button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  .legend { display: flex; flex-direction: column; gap: 3px; }
  .chip { display: flex; align-items: center; gap: 8px; padding: 2px 6px; border-radius: 5px; cursor: pointer;
          user-select: none; }
  .chip:hover { background: var(--bg); }
  .chip.off { opacity: .35; }
  .chip .sw { width: 22px; height: 0; border-top: 3px solid; flex: none; }
  .chip .sw.st { border-top-style: dashed; }
  .chip .n { margin-left: auto; color: var(--muted); font-size: 12px; }
  #detail { font-size: 13px; }
  #detail .k { color: var(--muted); }
  #detail ul, #steps { margin: 4px 0 0; padding-left: 18px; }
  #steps li { cursor: default; padding: 1px 0; }
  #steps li:hover { color: var(--accent); }
  .lvl { fill: var(--muted); font-size: 12px; }
  .node { stroke: var(--node-stroke); stroke-width: 1.5; cursor: pointer; }
  .edge { fill: none; }
  #tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--line);
         border-radius: 6px; padding: 6px 9px; font-size: 12px; max-width: 340px; display: none; z-index: 5;
         box-shadow: 0 2px 8px rgba(0,0,0,.18); }
  .hint { color: var(--muted); font-size: 12px; }
  @media (max-width: 760px) { main { flex-direction: column; } aside { width: auto; max-height: 40%;
      border-left: 0; border-top: 1px solid var(--line); } }
</style>
</head>
<body>
<header>
  <h1>AoG graph</h1>
  <span class="stats" id="stats"></span>
</header>
<main>
  <div id="stage"><svg id="svg"><g id="vp"><g id="lvl"></g><g id="eg"></g><g id="ng"></g></g></svg><div id="tip"></div></div>
  <aside>
    <h2>View</h2>
    <div class="toolbar">
      <button id="bFit">Fit</button>
      <button id="bSeq">Example sequence</button>
      <button id="bClear">Clear selection</button>
    </div>
    <p class="hint">Wheel: zoom &middot; drag: pan &middot; click a node: highlight all paths through it &middot; click a legend row: hide that component's edges.</p>
    <h2>Removed component (edge colour)</h2>
    <div class="legend" id="legend"></div>
    <h2>Selection</h2>
    <div id="detail"><span class="hint">Nothing selected.</span></div>
    <h2>Example sequence</h2>
    <ol id="steps"></ol>
    <p class="hint">Cheapest total travel, each step along its shortest free direction. Only an illustration: direction choice belongs to the ACO.</p>
  </aside>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
  const D = JSON.parse(document.getElementById('data').textContent);
  const NS = 'http://www.w3.org/2000/svg';
  const N = D.comps.length, DX = 30, DY = 130, PADX = 110, PADY = 50, R = 7;
  const svg = document.getElementById('svg'), vp = document.getElementById('vp');
  const gL = document.getElementById('lvl'), gE = document.getElementById('eg'), gN = document.getElementById('ng');
  const tip = document.getElementById('tip'), stage = document.getElementById('stage');
  const maxRow = Math.max(...D.levelSizes);
  const lastLevel = Math.max(...D.levelSizes.map((s, i) => s ? i : 0));
  const W = maxRow * DX + PADX * 2, H = lastLevel * DY + PADY * 2;
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const color = c => `hsl(${Math.round((c * 360 / N + 12) % 360)} 62% ${matchMedia('(prefers-color-scheme: dark)').matches ? 62 : 44}%)`;
  const esc = s => String(s).replace(/[&<>"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
  const r3 = x => Math.round(x * 1000) / 1000 + 0;
  function dirName(v) {
    const nz = v.map((x, i) => [x, i]).filter(p => Math.abs(p[0]) > 1e-6);
    if (nz.length === 1 && Math.abs(Math.abs(nz[0][0]) - 1) < 1e-6) return (nz[0][0] > 0 ? '+' : '−') + 'xyz'[nz[0][1]];
    return '(' + v.map(r3).join(', ') + ')';
  }
  const pos = D.nodes.map(n => ({
    x: W / 2 + (n[1] - (D.levelSizes[n[0]] - 1) / 2) * DX, y: PADY + n[0] * DY }));
  const masks = D.nodes.map(n => BigInt('0x' + n[4]));
  const present = v => { const p = []; for (let c = 0; c < N; c++) if ((masks[v] >> BigInt(c)) & 1n) p.push(c); return p; };
  const out = D.nodes.map(() => []), inn = D.nodes.map(() => []);
  D.edges.forEach((e, i) => { out[e[0]].push(i); inn[e[1]].push(i); });
  const edgeCount = new Array(N).fill(0); D.edges.forEach(e => edgeCount[e[2]]++);

  const el = (name, attrs, parent) => { const n = document.createElementNS(NS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; };

  D.levelSizes.forEach((sz, L) => {
    if (!sz) return;
    const t = el('text', {x: 10, y: PADY + L * DY + 4, class: 'lvl'}, gL);
    t.textContent = `L${L}: ${sz} state${sz > 1 ? 's' : ''}`;
    const s = el('text', {x: 10, y: PADY + L * DY + 19, class: 'lvl'}, gL);
    s.textContent = `${N - L} assembled`;
  });

  const edgeEls = D.edges.map((e, i) => {
    const a = pos[e[0]], b = pos[e[1]], st = D.comps[e[2]].stander;
    const ln = el('line', {x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: 'edge', stroke: color(e[2]),
      'stroke-width': 1.3, opacity: 0.32, 'stroke-dasharray': st ? '5 3' : ''}, gE);
    ln.addEventListener('mousemove', ev => showEdgeTip(ev, i));
    ln.addEventListener('mouseleave', hideTip);
    return ln;
  });
  const nodeEls = D.nodes.map((n, v) => {
    const c = el('circle', {cx: pos[v].x, cy: pos[v].y, r: R, class: 'node'}, gN);
    c.addEventListener('mousemove', ev => showNodeTip(ev, v));
    c.addEventListener('mouseleave', hideTip);
    c.addEventListener('click', ev => { if (!moved) select(v); });
    return c;
  });
  function paintNodes() {
    nodeEls.forEach((c, v) => {
      const n = D.nodes[v];
      let fill = css('--node');
      if (n[0] === 0) fill = css('--root');
      else if (n[0] === N) fill = css('--terminal');
      else if (n[3]) fill = css('--dead');
      else if (!n[2]) fill = css('--dead');
      c.setAttribute('fill', fill);
      c.setAttribute('r', (n[0] === 0 || n[0] === N) ? R + 2 : R);
    });
  }
  paintNodes();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    paintNodes(); edgeEls.forEach((l, i) => l.setAttribute('stroke', color(D.edges[i][2])));
    buildLegend(); });

  // ---- tooltips -----------------------------------------------------------
  function placeTip(ev, html) {
    tip.innerHTML = html; tip.style.display = 'block';
    const r = stage.getBoundingClientRect();
    let x = ev.clientX - r.left + 14, y = ev.clientY - r.top + 14;
    if (x + tip.offsetWidth > r.width - 4) x = ev.clientX - r.left - tip.offsetWidth - 14;
    if (y + tip.offsetHeight > r.height - 4) y = ev.clientY - r.top - tip.offsetHeight - 14;
    tip.style.left = Math.max(4, x) + 'px'; tip.style.top = Math.max(4, y) + 'px';
  }
  function hideTip() { tip.style.display = 'none'; }
  function edgeInfo(i) {
    const e = D.edges[i];
    return e[3].map((k, j) => `${dirName(D.dirs[k])}&nbsp;${e[4][j]}&nbsp;mm`).join(' &middot; ');
  }
  function showEdgeTip(ev, i) {
    const e = D.edges[i], c = D.comps[e[2]];
    placeTip(ev, `<b>Remove ${esc(c.label)}</b>${c.stander ? ' (stander)' : ''}<br>${e[3].length} free direction${e[3].length > 1 ? 's' : ''}: ${edgeInfo(i)}`);
  }
  function stateSummary(v) {
    const p = present(v), n = D.nodes[v];
    const flag = n[0] === 0 ? 'root' : n[0] === N ? 'fully disassembled' : (n[3] ? 'DEAD END' : (n[2] ? '' : 'cannot reach the end'));
    return `<b>State ${v}</b> &middot; level ${n[0]}${flag ? ' &middot; ' + flag : ''}<br>` +
      `<span class="k">assembled (${p.length}):</span> ${p.map(c => esc(D.comps[c].label)).join(', ') || '—'}`;
  }
  function showNodeTip(ev, v) { placeTip(ev, stateSummary(v) + `<br><span class="k">moves out:</span> ${out[v].length}`); }

  // ---- highlight / filters -----------------------------------------------
  const hidden = new Set(); let selected = null, seqOn = false, hoverStep = null;
  const seqEdges = new Set(D.sequence), seqNodes = new Set();
  { let v = 0; seqNodes.add(0); D.sequence.forEach(i => { seqNodes.add(D.edges[i][1]); }); }
  function related(v) {
    const nodes = new Set([v]), edges = new Set();
    let stack = [v];
    while (stack.length) { const u = stack.pop(); for (const i of inn[u]) { edges.add(i);
      const p = D.edges[i][0]; if (!nodes.has(p)) { nodes.add(p); stack.push(p); } } }
    stack = [v];
    const seen = new Set([v]);
    while (stack.length) { const u = stack.pop(); for (const i of out[u]) { edges.add(i);
      const d = D.edges[i][1]; nodes.add(d); if (!seen.has(d)) { seen.add(d); stack.push(d); } } }
    return {nodes, edges};
  }
  function refresh() {
    const rel = selected === null ? null : related(selected);
    D.edges.forEach((e, i) => {
      const l = edgeEls[i];
      l.style.display = hidden.has(e[2]) ? 'none' : '';
      let op = 0.32, w = 1.3;
      if (rel) { if (rel.edges.has(i)) { op = 0.85; w = 1.8; } else op = 0.04; }
      if (seqOn && seqEdges.has(i)) { op = 1; w = 3.6; }
      if (hoverStep === i) { op = 1; w = 5; }
      l.setAttribute('opacity', op); l.setAttribute('stroke-width', w);
    });
    nodeEls.forEach((c, v) => {
      let op = 1;
      if (rel && !rel.nodes.has(v)) op = 0.12;
      c.setAttribute('opacity', op);
      const ring = v === selected || (seqOn && seqNodes.has(v));
      c.setAttribute('stroke', ring ? css('--accent') : css('--node-stroke'));
      c.setAttribute('stroke-width', ring ? 3 : 1.5);
    });
  }
  function select(v) {
    selected = (selected === v) ? null : v;
    const d = document.getElementById('detail');
    if (selected === null) { d.innerHTML = '<span class="hint">Nothing selected.</span>'; refresh(); return; }
    const moves = out[v].map(i => { const e = D.edges[i], c = D.comps[e[2]];
      return `<li>${esc(c.label)}${c.stander ? ' (stander)' : ''}: ${edgeInfo(i)}</li>`; }).join('');
    d.innerHTML = stateSummary(v) + `<br><span class="k">Moves out (${out[v].length}):</span><ul>${moves || '<li>none</li>'}</ul>`;
    refresh();
  }
  document.getElementById('bClear').onclick = () => { selected = null;
    document.getElementById('detail').innerHTML = '<span class="hint">Nothing selected.</span>'; refresh(); };
  const bSeq = document.getElementById('bSeq');
  bSeq.onclick = () => { seqOn = !seqOn; bSeq.classList.toggle('on', seqOn); refresh(); };
  if (!D.sequence.length) { bSeq.disabled = true; bSeq.title = 'The AoG has no complete sequence'; }

  function buildLegend() {
    const lg = document.getElementById('legend'); lg.innerHTML = '';
    D.comps.forEach((c, i) => {
      const row = document.createElement('div'); row.className = 'chip' + (hidden.has(i) ? ' off' : '');
      row.innerHTML = `<span class="sw${c.stander ? ' st' : ''}" style="border-color:${color(i)}"></span>` +
        `<span>${esc(c.label)}${c.stander ? ' <span class="k">(stander)</span>' : ''}</span><span class="n">${edgeCount[i]}</span>`;
      row.onclick = () => { hidden.has(i) ? hidden.delete(i) : hidden.add(i); buildLegend(); refresh(); };
      lg.appendChild(row);
    });
  }
  buildLegend();

  const steps = document.getElementById('steps');
  D.sequence.forEach((i, s) => {
    const e = D.edges[i], c = D.comps[e[2]];
    let j = 0; e[4].forEach((t, q) => { if (t < e[4][j]) j = q; });
    const li = document.createElement('li');
    li.innerHTML = `${esc(c.label)} → ${dirName(D.dirs[e[3][j]])}, ${e[4][j]}&nbsp;mm`;
    li.onmouseenter = () => { hoverStep = i; refresh(); };
    li.onmouseleave = () => { hoverStep = null; refresh(); };
    steps.appendChild(li);
  });
  if (!D.sequence.length) steps.innerHTML = '<li class="hint" style="list-style:none;margin-left:-18px">no complete sequence</li>';

  // ---- header -------------------------------------------------------------
  const S = D.stats;
  document.getElementById('stats').innerHTML =
    `${S.N} components &middot; ${S.V} states &middot; ${S.E} moves` +
    (S.dead ? ` &middot; <span class="bad">${S.dead} dead end${S.dead > 1 ? 's' : ''}</span>` : '') +
    (S.complete ? '' : ' &middot; <span class="bad">incomplete: the root cannot reach the end</span>');

  // ---- pan / zoom ---------------------------------------------------------
  let view = {x: 0, y: 0, k: 1}, moved = false;
  const apply = () => vp.setAttribute('transform', `translate(${view.x} ${view.y}) scale(${view.k})`);
  function fit() {
    const r = svg.getBoundingClientRect();
    const k = Math.min(r.width / W, r.height / H, 1.6);
    view = {k, x: (r.width - W * k) / 2, y: (r.height - H * k) / 2}; apply();
  }
  document.getElementById('bFit').onclick = fit;
  addEventListener('resize', fit);
  svg.addEventListener('wheel', ev => {
    ev.preventDefault();
    const r = svg.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    const f = ev.deltaY < 0 ? 1.15 : 1 / 1.15, k = Math.min(8, Math.max(0.05, view.k * f)), g = k / view.k;
    view.x = mx - (mx - view.x) * g; view.y = my - (my - view.y) * g; view.k = k; apply();
  }, {passive: false});
  let drag = null;
  svg.addEventListener('pointerdown', ev => { drag = {x: ev.clientX, y: ev.clientY, vx: view.x, vy: view.y}; moved = false; });
  addEventListener('pointermove', ev => {
    if (!drag) return;
    const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) { moved = true; svg.classList.add('dragging'); hideTip(); }
    if (moved) { view.x = drag.vx + dx; view.y = drag.vy + dy; apply(); }
  });
  addEventListener('pointerup', () => { drag = null; svg.classList.remove('dragging'); setTimeout(() => moved = false, 0); });
  fit(); refresh();
})();
</script>
</body>
</html>
"""


def write_html(aog: AoG, rank: np.ndarray, path: list[int], out_file: Path) -> None:
    payload = build_payload(aog, rank, path)
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    out_file.write_text(HTML_TEMPLATE.replace("__DATA__", data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_summary(aog: AoG, inputs: AoGInputs, report: dict, checks: list[dict], path: list[int],
                  elapsed: float, files: dict[str, Path | None]) -> None:
    N, n = aog.n_components, inputs.n_movable
    print()
    print(f"AoG: {N} components ({n} movable, {N - n} standers), {len(aog.directions)} directions, "
          f"{aog.n_nodes} states, {aog.n_edges} moves, complete={aog.complete}, "
          f"{len(aog.dead_ends)} dead ends")
    print("States per level:  " + "  ".join(f"L{L}:{s}" for L, s in enumerate(report["states_per_level"])))
    if report["stander_holds"]:
        print("Stander holds (must all still be present when the stander is removed):")
        for s, g in report["stander_holds"].items():
            print(f"  {s}: {', '.join(g) if g else '(none)'}")
    print("Moves with everything assembled:")
    for m in report["moves_with_everything_assembled"]:
        dirs = ", ".join(f"{tuple(d)} {t} mm" for d, t in zip(m["directions"], m["travel_mm"]))
        print(f"  {m['component']}{' (stander)' if m['is_stander'] else ''}: {dirs}")
    if len(aog.dead_ends):
        print(f"Dead ends ({len(aog.dead_ends)}, first 5):")
        for v in aog.dead_ends[:5]:
            print(f"  state {int(v)}: {', '.join(aog.component_labels[c] for c in aog.state_components(int(v)))}")
    print()
    print(sequence_text(aog, path).rstrip())
    print()
    print("Checks:")
    for c in checks:
        tag = "SKIP" if c.get("skipped") else ("PASS" if c["passed"] else "FAIL")
        print(f"  {c['id']:<4} {tag}  {c['name']:<16} {c['detail']}")
    failed = [c for c in checks if not c["passed"]]
    print()
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed" + ("" if not failed else "  <-- FAILURES ABOVE")
          + f"   ({elapsed:.1f} s including generation)")
    for label, f in files.items():
        if f is not None:
            print(f"{label}: {f}")


def main(args: argparse.Namespace) -> int:
    """Pipeline: get the AoG inputs (real: run_preprocessing -> identify_interferences (cached) -> build_aog
    (cached); synthetic: a hand-built AoGInputs -> generate_aog), run the independent checks S1-S10, pick an
    example sequence, write report / sequence text / HTML graph, print the summary."""
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    cfg = AoGConfig(margin=args.margin)
    t0 = time.perf_counter()

    if args.synthetic:
        builders = {
            "stack": synthetic_stack,
            "demo": synthetic_demo,
            "lattice": lambda: synthetic_lattice(args.n),
            "random": lambda: synthetic_random(args.n, args.m, args.seed),
        }
        inputs = builders[args.synthetic]()
        out_dir = (args.out or REPO_ROOT / "tests" / "synthetic" / args.synthetic) / "aog"
        print(f"Synthetic assembly '{args.synthetic}': {len(inputs.component_ids)} components ...", flush=True)
        aog = generate_aog(inputs, cfg)
        save_aog(aog, out_dir)
        from aog_generator import EXPORT_MAX_NODES, export_csv_dot

        if aog.n_nodes <= EXPORT_MAX_NODES:
            export_csv_dot(aog, out_dir)
    else:
        import freecad  # noqa: F401  (sets up the FreeCAD lib path in the conda env; must come first)

        from collision import CollisionConfig
        from interference_identification import identify_interferences
        from preprocessing import run_preprocessing

        step, bom, out = args.paths if args.paths else (DEFAULT_STEP, DEFAULT_BOM, DEFAULT_OUT)
        out_dir = Path(out) / "aog"
        print(f"Preprocessing {Path(step).name} ...", flush=True)
        prep = run_preprocessing(step, bom)
        print(f"Interference Identification (h = {args.step}; cached unless --force-interference) ...", flush=True)
        interference = identify_interferences(prep, CollisionConfig(step=args.step), Path(out) / "interference",
                                              force=args.force_interference)
        print("AoG generation (cached unless --force-aog) ...", flush=True)
        aog = build_aog(interference, prep, cfg, cache_dir=out_dir, force=args.force_aog)
        inputs = collect_inputs(interference, prep, cfg)

    # The checks regenerate and reload the AoG on purpose; keep the module's own logging quiet.
    aog_log = logging.getLogger("aap.aog")
    previous_level = aog_log.level
    aog_log.setLevel(logging.ERROR)
    try:
        checks = run_checks(aog, inputs, args.samples, args.brute_max_n)
    finally:
        aog_log.setLevel(previous_level)
    path = example_sequence(aog)
    report = build_report(aog, inputs, checks, path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aog_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "aog_sequence.txt").write_text(sequence_text(aog, path), encoding="utf-8")

    html_path = None
    if args.no_html:
        pass
    elif aog.n_nodes > HTML_MAX_NODES or aog.n_edges > HTML_MAX_EDGES:
        print(f"HTML graph skipped: {aog.n_nodes} states / {aog.n_edges} moves is too big to draw usefully "
              f"(limits {HTML_MAX_NODES} / {HTML_MAX_EDGES}). Inspect aog_report.json or the csv files.")
    else:
        html_path = out_dir / "aog_graph.html"
        write_html(aog, layout_ranks(aog), path, html_path)

    print_summary(aog, inputs, report, checks, path, time.perf_counter() - t0,
                  {"Graph": html_path, "Report": out_dir / "aog_report.json",
                   "Sequence": out_dir / "aog_sequence.txt", "All files in": out_dir})
    if html_path is not None and args.open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the AoG, check it, and write an interactive HTML view of the graph.")
    parser.add_argument("paths", nargs="*", type=Path, help="[step_path bom_path out_dir] (default: industrial_fastner)")
    parser.add_argument("--synthetic", choices=["stack", "demo", "lattice", "random"],
                        help="use a hand-built assembly instead of the STEP pipeline (no FreeCAD needed)")
    parser.add_argument("--n", type=int, default=5, help="movable parts for lattice/random (default 5)")
    parser.add_argument("--m", type=int, default=2, help="standers for random (default 2)")
    parser.add_argument("--seed", type=int, default=1, help="seed for random (default 1)")
    parser.add_argument("--out", type=Path, help="output folder for synthetic runs (default tests/synthetic/<name>)")
    parser.add_argument("--step", type=float, default=DEFAULT_H,
                        help=f"collision step h in mm (default {DEFAULT_H}, calibrated for industrial_fastner only)")
    parser.add_argument("--force-interference", action="store_true", help="recompute the interference matrix")
    parser.add_argument("--force-aog", action="store_true", help="regenerate the AoG even if a cached one matches")
    parser.add_argument("--margin", type=float, default=AoGConfig.margin, help="travel margin in mm (default 1.0)")
    parser.add_argument("--samples", type=int, default=200, help="random paths replayed by check S6 (default 200)")
    parser.add_argument("--brute-max-n", type=int, default=BRUTE_MAX_N,
                        help=f"largest N for the brute-force check S2 (default {BRUTE_MAX_N})")
    parser.add_argument("--no-html", action="store_true", help="skip the HTML graph")
    parser.add_argument("--open", action="store_true", help="open the HTML graph in the browser when done")
    ns = parser.parse_args()
    if ns.paths and len(ns.paths) != 3:
        parser.error("give either no paths or all three: step_path bom_path out_dir")
    sys.exit(main(ns))
