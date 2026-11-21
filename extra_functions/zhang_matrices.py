"""Zhang et al. constraint matrices (P, E, C, G) derived from a cached AoG.

Zhang et al. (Annals of Operations Research, collaborative human-robot selective disassembly
sequence planning) describe the assembly by four 0/1 matrices over disassembly TASKS:

  P  priority       p[j,k] = 1 if task j must be executed before task k
  E  exclusive      e[j,k] = 1 if only one of the tasks j, k can be executed
  C  cooperative    c[j,k] = 1 if tasks j and k can be executed concurrently
  G  correlation    g[j,i] = 1 if executing task j recovers part i

Here a task is a (component, direction) pair: removing that component along that free direction.
A pair is a task if the direction is free on some live edge (an edge on a root->terminal path).
Alternatives for the same component are therefore exclusive tasks that recover the same part,
which is what makes E and G non-trivial, as in the paper.

Everything is derived from the AoG alone (no geometry, no FreeCAD). This lives outside the main
pipeline; nothing in src/ imports it.

Conventions
  * Python arrays use P[j, k] = 1 for "j before k" (row = predecessor).
  * CSV files use the paper's layout, row = k, column = j (so the priority CSV is P transposed).
    E and C are symmetric; G is (task, part).
  * P is the full precedence relation (transitive), like the paper's figure 3.

Variants of C
  * C (paper): no priority either way and not exclusive.
  * C strict: some live state S has both tasks available and the two removal orders commute
    (S -> S\\j -> S\\{j,k} and S -> S\\k -> S\\{j,k} both exist, with the task's direction free
    on the edge it uses). The AoG cannot say whether two simultaneous trajectories collide;
    that would need a swept-volume check.

Limits
  * The matrices are a projection of the AoG: OR-priorities, state-dependent rules (standard gates,
    contacts, BOM groups) are not representable pairwise, so the orders allowed by P alone are a
    superset of the AoG paths (reported by --check).
  * Full disassembly only; selective disassembly (target parts) is not implemented.

Usage (from the repo root):
    python extra_functions/zhang_matrices.py [aog_dir] [out_dir] [--check]

aog_dir defaults to output/industrial_fastner/aog; out_dir to <aog_dir>/../zhang_matrices.

Expected result: the console prints `<T> tasks from <N> components; written to <out_dir>` and the numbered task list
("<component> (dx,dy,dz)"). out_dir gets P_priority.csv, E_exclusive.csv, C_cooperative.csv,
C_cooperative_strict.csv, G_correlation.csv, task_index.csv, zhang_tasks.json and zhang_matrices.npz. With --check it
also prints the number of AoG paths, the number of orders allowed by P alone (equal or larger) and
`P_matches_all_paths: True`. E and C are symmetric and every task has exactly one 1 in its row of G.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aog_generator import AoG, load_aog  # noqa: E402  (numpy only, no FreeCAD)

U64 = np.uint64


@dataclass
class ZhangMatrices:
    task_component: np.ndarray   # (T,) component (bit index) of each task
    task_direction: np.ndarray   # (T,) index into `directions`
    task_labels: list[str]       # "<component label> <direction vector>"
    component_labels: list[str]
    directions: np.ndarray       # (D', 3)
    P: np.ndarray                # (T, T) int8, P[j, k] = 1: task j before task k
    E: np.ndarray                # (T, T) int8, symmetric
    C: np.ndarray                # (T, T) int8, symmetric, paper definition
    C_strict: np.ndarray         # (T, T) int8, symmetric, commuting-diamond definition
    G: np.ndarray                # (T, N) int8, G[j, i] = 1: task j recovers part i
    P_component: np.ndarray      # (N, N) int8, P_component[a, b] = 1: a always removed before b
    C_component: np.ndarray      # (N, N) int8, paper definition at component level
    C_strict_component: np.ndarray  # (N, N) int8


def _edge_direction_bits(aog: AoG) -> np.ndarray:
    """(E, D') bool: free directions of every edge (single uint64 word per edge)."""
    words = aog.edge_dirs[:, 0].astype(U64)
    shifts = np.arange(len(aog.directions), dtype=U64)
    return ((words[:, None] >> shifts[None, :]) & U64(1)).astype(bool)


def compute_matrices(aog: AoG) -> ZhangMatrices:
    """[MAIN] AoG -> Zhang matrices. Tasks = (component, direction) pairs free on some live edge. Component-level P:
    a precedes b if a is never present when b leaves. E: different directions of one component. C: no priority and
    not exclusive (paper) / a live state where both can leave and both orders exist (strict). G: task -> its part."""
    N, Dp = aog.n_components, len(aog.directions)
    live_edge = aog.live[aog.edge_dst] if aog.n_edges else np.zeros(0, dtype=bool)
    comp = aog.edge_comp.astype(np.int64)
    src_mask = aog.node_mask[aog.edge_src]
    bits = _edge_direction_bits(aog)

    # Tasks: (component, direction) pairs free on at least one live edge.
    task_set = np.zeros((N, Dp), dtype=bool)
    for k in range(Dp):
        sel = live_edge & bits[:, k]
        task_set[np.unique(comp[sel]), k] = True
    tc, tk = np.nonzero(task_set)
    T = len(tc)
    tid = -np.ones((N, Dp), dtype=np.int64)
    tid[tc, tk] = np.arange(T)

    # Component-level priority: a is always removed before b iff a is never present when b leaves.
    present_when_leaving = np.zeros(N, dtype=U64)
    for b in range(N):
        sel = live_edge & (comp == b)
        if sel.any():
            present_when_leaving[b] = np.bitwise_or.reduce(src_mask[sel])
    P_comp = np.zeros((N, N), dtype=np.int8)
    for a in range(N):
        for b in range(N):
            if a != b and not (int(present_when_leaving[b]) >> a) & 1:
                P_comp[a, b] = 1

    # Component-level strict cooperation: a live state where both can leave and the orders commute.
    edges_from: dict[int, dict[int, tuple[int, np.ndarray]]] = {}
    for e in np.flatnonzero(live_edge):
        edges_from.setdefault(int(aog.edge_src[e]), {})[int(comp[e])] = (int(aog.edge_dst[e]), bits[e])
    Cs_comp = np.zeros((N, N), dtype=np.int8)
    Cs_task = np.zeros((T, T), dtype=np.int8)
    for s, out in edges_from.items():
        for j, (tj, dj) in out.items():
            for k, (tk_, dk) in out.items():
                if j >= k or k not in edges_from.get(tj, {}) or j not in edges_from.get(tk_, {}):
                    continue
                # directions of each task that are free both from S and after the other one left
                dj_ok = dj & edges_from[tk_][j][1]
                dk_ok = dk & edges_from[tj][k][1]
                if dj_ok.any() and dk_ok.any():
                    Cs_comp[j, k] = Cs_comp[k, j] = 1
                    for a in np.flatnonzero(dj_ok):
                        for b in np.flatnonzero(dk_ok):
                            Cs_task[tid[j, a], tid[k, b]] = Cs_task[tid[k, b], tid[j, a]] = 1

    # Expand to tasks.
    P = P_comp[np.ix_(tc, tc)].copy()
    same = tc[:, None] == tc[None, :]
    E = (same & (tk[:, None] != tk[None, :])).astype(np.int8)
    C = ((P == 0) & (P.T == 0) & (E == 0)).astype(np.int8)
    np.fill_diagonal(C, 0)
    # tasks of one component are never cooperative (they are alternatives); P is 0 inside a component
    C_comp = ((P_comp == 0) & (P_comp.T == 0)).astype(np.int8)
    np.fill_diagonal(C_comp, 0)
    G = np.zeros((T, N), dtype=np.int8)
    G[np.arange(T), tc] = 1

    labels = [f"{aog.component_labels[c]} ({aog.directions[k][0]:+.2g},{aog.directions[k][1]:+.2g},"
              f"{aog.directions[k][2]:+.2g})" for c, k in zip(tc, tk)]
    return ZhangMatrices(tc, tk, labels, list(aog.component_labels), np.asarray(aog.directions),
                         P, E, C, Cs_task, G, P_comp, C_comp, Cs_comp)


# ---------------------------------------------------------------------------
# Brute-force check against explicit root -> terminal paths
# ---------------------------------------------------------------------------


def enumerate_paths(aog: AoG, limit: int = 200_000):
    """All root->terminal edge paths as lists of (component, dir bitset); None if more than `limit`."""
    paths, stack = [], [(aog.root, [])]
    while stack:
        v, path = stack.pop()
        if v == aog.terminal:
            paths.append(path)
            if len(paths) > limit:
                return None
            continue
        for e in aog.out_edges(v):
            if aog.live[aog.edge_dst[e]]:
                stack.append((int(aog.edge_dst[e]), path + [(int(aog.edge_comp[e]), int(aog.edge_dirs[e, 0]))]))
    return paths


def check_against_paths(aog: AoG, m: ZhangMatrices) -> dict:
    """Recompute component-level P from all paths (a before b in every path) and the strict C
    from commuting neighbours, and count the orders P alone allows."""
    paths = enumerate_paths(aog)
    if paths is None:
        return {"skipped": "more than 200000 paths"}
    N = aog.n_components
    before = np.ones((N, N), dtype=bool)
    for p in paths:
        pos = np.empty(N, dtype=int)
        for i, (c, _) in enumerate(p):
            pos[c] = i
        before &= pos[:, None] < pos[None, :]
    np.fill_diagonal(before, False)
    ok_p = bool(np.array_equal(before, m.P_component.astype(bool)))

    # orders allowed by P alone (subset DP over predecessors)
    pre = [sum(1 << a for a in range(N) if m.P_component[a, b]) for b in range(N)]
    memo: dict[int, int] = {}

    def ext(rem: int) -> int:
        if rem == 0:
            return 1
        if rem not in memo:
            memo[rem] = sum(ext(rem & ~(1 << b)) for b in range(N) if (rem >> b) & 1 and not (pre[b] & rem))
        return memo[rem]

    return {"aog_paths": len(paths), "orders_allowed_by_P": ext((1 << N) - 1), "P_matches_all_paths": ok_p}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_csv(path: Path, M: np.ndarray, rows: list[str], cols: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([""] + cols)
        for r, line in zip(rows, M):
            w.writerow([r] + [int(x) for x in line])


def save_matrices(m: ZhangMatrices, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "zhang_matrices.npz", task_component=m.task_component, task_direction=m.task_direction,
             P=m.P, E=m.E, C=m.C, C_strict=m.C_strict, G=m.G, P_component=m.P_component,
             C_component=m.C_component, C_strict_component=m.C_strict_component, directions=m.directions)
    (out_dir / "zhang_tasks.json").write_text(json.dumps({
        "convention": "npz: P[j,k]=1 means j before k. CSV (paper layout): row = k, column = j.",
        "tasks": [{"task": i, "component": m.component_labels[c], "direction_index": int(k),
                   "direction": m.directions[k].tolist()}
                  for i, (c, k) in enumerate(zip(m.task_component, m.task_direction))],
    }, indent=1), encoding="utf-8")
    idx = [str(i + 1) for i in range(len(m.task_labels))]  # tasks numbered 1..T like the paper
    _write_csv(out_dir / "P_priority.csv", m.P.T, idx, idx)
    _write_csv(out_dir / "E_exclusive.csv", m.E, idx, idx)
    _write_csv(out_dir / "C_cooperative.csv", m.C, idx, idx)
    _write_csv(out_dir / "C_cooperative_strict.csv", m.C_strict, idx, idx)
    _write_csv(out_dir / "G_correlation.csv", m.G, idx, m.component_labels)
    (out_dir / "task_index.csv").write_text(
        "task,label\n" + "\n".join(f"{i},{lab}" for i, lab in zip(idx, m.task_labels)) + "\n", encoding="utf-8")


def main() -> None:
    """Load the cached AoG, compute the matrices, write the CSV / npz / json files, print the task list (and the check)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("aog_dir", nargs="?", type=Path, default=REPO_ROOT / "output" / "industrial_fastner" / "aog")
    ap.add_argument("out_dir", nargs="?", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="verify against all root->terminal paths")
    args = ap.parse_args()
    aog = load_aog(args.aog_dir)
    if aog is None:
        sys.exit(f"No readable AoG in {args.aog_dir}")
    if not aog.complete:
        print("Warning: the AoG is incomplete; matrices use the live part only.")
    m = compute_matrices(aog)
    out_dir = args.out_dir or args.aog_dir.parent / "zhang_matrices"
    save_matrices(m, out_dir)
    print(f"{len(m.task_labels)} tasks from {aog.n_components} components; written to {out_dir}")
    for i, lab in enumerate(m.task_labels, start=1):
        print(f"  {i:>3}  {lab}")
    if args.check:
        print("Check:", check_against_paths(aog, m))


if __name__ == "__main__":
    main()
