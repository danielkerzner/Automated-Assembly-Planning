"""Optimization (ACO): the lowest-cost assembly sequence through the AoG.

Implements docs/optimization-spec.md. Ants walk the AoG from the root (full assembly) to the
terminal (empty) over `live` nodes only, choosing at each node a live out-edge and one of its free
directions. The walk is a disassembly path; every cost term is defined in assembly order (the
edge leaving state S is assembly step t = popcount(S)) and only ever compares consecutive steps,
so the walking direction does not matter.

Costs, not desirabilities: lower is better. Each enabled term is scaled to [0, 1] over a complete
path and multiplied by its weight (pairwise dominance ranking, renormalized over the enabled
terms). The ACO heuristic is eta = 1 / (eps + immediate cost + optional exact remaining static
cost); pheromone (MAX-MIN Ant System) is one float32 per move (AoG edge, free direction) and the deposit is 1 / (eps + path cost).

Layout:
  OptimizationConfig, resolve_weights, validate_config   configuration (spec section 3)
  TermTables                                             per-component data + contact patches
  CostModel                                              cost terms, stability, Viterbi, DP (pure numpy)
  run_aco / optimize                                     the ACO itself (pure numpy, no FreeCAD)
  collect_tables, extract_contact_patches                FreeCAD side (geometry), cached on disk
  optimize_sequence                                      pipeline entry with caching
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("aap.optimization")

REPO_ROOT = Path(__file__).resolve().parent.parent
U64 = np.uint64

# Cost terms in the order of the dominance matrix (A..G), with the flag that enables each.
TERMS = ("travel", "direction_change", "tool_change", "size_contacts", "grouping", "stability",
         "gravity_change")
FLAG_OF = {
    "travel": "use_travel",
    "direction_change": "use_direction_change",
    "tool_change": "use_tool_change",
    "size_contacts": "use_size_contacts",
    "grouping": "use_grouping",
    "stability": "use_stability",
    "gravity_change": "use_gravity_change",
}
# Pairwise dominance ranking (spec 3.2): 0 = row less important, 1 = equal, 2 = more important.
DOMINANCE = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 1],
    [2, 2, 1, 1, 2, 1, 2],
    [2, 2, 1, 1, 1, 0, 1],
    [2, 2, 0, 1, 1, 0, 0],
    [2, 2, 1, 2, 2, 1, 2],
    [2, 1, 0, 1, 2, 0, 1],
])
DEFAULT_WEIGHTS = {t: float(DOMINANCE[r].sum()) / DOMINANCE.sum() for r, t in enumerate(TERMS)}

AXES6 = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0),
         (0.0, 0.0, -1.0))

EPS = 1e-3  # eps in eta = 1 / (eps + cost) and in the deposit 1 / (eps + C); costs are O(1)
FALL_TOL = 1e-9  # d . g > FALL_TOL: free direction d lets the component fall under g
NORMAL_TOL = 1e-3  # a patch bears load under g if its normal n has n . g < -NORMAL_TOL
PATCH_DEFLECTION = 0.1  # mm, tessellation deflection for contact patches
AREA_EPS = 1e-9  # mm^2, faces of a boolean common smaller than this are ignored
HULL_TOL = 0.01  # mm, a projected CoG this close to the support hull counts as supported (line/point contact)
TABLES_VERSION = "2"  # bump when the term-table contents change


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OptimizationConfig:
    # Criterion flags (spec 3.1)
    use_travel: bool = True  # M1
    use_direction_change: bool = True  # M2
    use_tool_change: bool = True  # M3, needs the BOM `tool` column
    use_size_contacts: bool = True  # M6
    use_grouping: bool = True  # M7
    use_stability: bool = True  # M8 class-1 penalty
    use_gravity_change: bool = True  # M8 gravity change, needs use_stability
    # Weight overrides, term name -> value; merged into DEFAULT_WEIGHTS, then renormalized.
    weights: dict[str, float] | None = None
    # M2: same direction 0, same axis opposite sign = this ratio, other axis 1
    dir_change_axis_ratio: float = 0.25
    # M6 mix (sum 1; any may be 0)
    size_alpha: float = 1 / 3
    size_beta: float = 1 / 3
    size_gamma: float = 1 / 3
    # M8
    gravity_dirs: tuple = AXES6
    stability_fall_test: bool = True
    # "hull" (default, rigid-body assumption): CoG must lie in the convex hull of the projected
    # load-bearing patches (bridging supports such as a block on two rails are stable).
    # "union" (strict): CoG must lie in one of the patches themselves.
    support_region: str = "hull"
    # ACO (spec 6.3-6.5); None = derived from N
    # Optional: add the exact remaining static cost (M1 + M6, by DP) to eta. Off by default: the lookahead
    # covers only the static terms, so the ants chase the static optimum (spec 6.3). Needed for travel-only.
    use_cost_to_go: bool = False
    alpha: float = 1.0
    beta: float = 3.0
    rho: float = 0.2
    n_ants: int | None = None  # max(2N, 10)
    n_iterations: int = 200
    tau_min_ratio: float | None = None  # 1 / (2N)
    stagnation_limit: int = 50
    global_best_interval: int = 10  # every k-th iteration the global best deposits instead
    seed: int = 0
    force: bool = False  # recompute cached term tables and result


def resolve_weights(config: OptimizationConfig) -> dict[str, float]:
    """Weights of the enabled terms, renormalized to sum to 1 (ratios kept)."""
    raw = dict(DEFAULT_WEIGHTS)
    if config.weights:
        unknown = set(config.weights) - set(TERMS)
        if unknown:
            raise ValueError(f"Unknown weight term(s) {sorted(unknown)}; valid: {list(TERMS)}")
        raw.update({t: float(v) for t, v in config.weights.items()})
    negative = [t for t, v in raw.items() if v < 0]
    if negative:
        raise ValueError(f"Weights must be >= 0, got negative for {negative}")
    enabled = {t: raw[t] for t in TERMS if getattr(config, FLAG_OF[t])}
    total = sum(enabled.values())
    if total <= 0:
        raise ValueError("The enabled terms have total weight 0")
    return {t: v / total for t, v in enabled.items()}


def validate_config(config: OptimizationConfig, has_tool_column: bool, has_group_column: bool) -> None:
    """Spec 3.1 validation; raises ValueError, never skips a term silently."""
    if not any(getattr(config, FLAG_OF[t]) for t in TERMS):
        raise ValueError("At least one criterion flag must be True")
    if config.use_gravity_change and not config.use_stability:
        raise ValueError("use_gravity_change=True requires use_stability=True")
    if config.use_tool_change and not has_tool_column:
        raise ValueError("use_tool_change=True needs a 'tool' column in the BOM (bom.csv); add it or "
                         "set use_tool_change=False")
    if config.use_grouping and not has_group_column:
        raise ValueError("use_grouping=True needs a 'group' column in the BOM (bom.csv); add it or "
                         "set use_grouping=False")
    if config.support_region not in ("union", "hull"):
        raise ValueError(f"support_region must be 'union' or 'hull', got {config.support_region!r}")
    if config.dir_change_axis_ratio < 0:
        raise ValueError("dir_change_axis_ratio must be >= 0")
    if config.use_stability and len(config.gravity_dirs) == 0:
        raise ValueError("gravity_dirs is empty")
    if not 0 < config.rho < 1:
        raise ValueError("rho must be in (0, 1)")
    resolve_weights(config)


# ---------------------------------------------------------------------------
# Term tables (per component; FreeCAD-free once built)
# ---------------------------------------------------------------------------


@dataclass
class TermTables:
    component_ids: list[str]  # AoG order (bit index)
    volume: np.ndarray  # (N,) mm^3
    area: np.ndarray  # (N,) mm^2
    contact: np.ndarray  # (N, N) bool, symmetric; Nb_i = contact[i].sum()
    cog: np.ndarray  # (N, 3) centre of gravity, mm
    tool: list[str | None]  # BOM `tool` (standers; None if blank or absent)
    group: list[str | None]  # BOM `group` (None if blank or absent)
    pitch: list[float | None]  # BOM `pitch`, mm/turn (screw/nut rows only; None if blank/non-positive/absent)
    has_tool_column: bool
    has_group_column: bool
    # i < j -> (tri (T, 3, 3), normal (T, 3)); normal = outward normal of j's surface, i.e. pointing
    # from j into i. None = not computed (stability off).
    patches: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None
    cache_key: str = ""

    @property
    def n_contacts(self) -> np.ndarray:
        return np.asarray(self.contact, dtype=bool).sum(axis=1)

    def contact_triangles(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Contact triangles between i and j with normals pointing from j into i."""
        empty = (np.zeros((0, 3, 3)), np.zeros((0, 3)))
        if not self.patches:
            return empty
        if i < j:
            return self.patches.get((i, j), empty)
        tri, normal = self.patches.get((j, i), empty)
        return tri, -normal


# ---------------------------------------------------------------------------
# Small geometry helpers (pure numpy)
# ---------------------------------------------------------------------------


def _bits(mask: int, n: int) -> list[int]:
    return [b for b in range(n) if (mask >> b) & 1]


def _plane_basis(g: np.ndarray) -> np.ndarray:
    """(3, 2) orthonormal basis of the plane perpendicular to g."""
    a = np.array([1.0, 0.0, 0.0]) if abs(g[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = a - a.dot(g) * g
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(g, u)], axis=1)


def _point_in_triangles(P: np.ndarray, p: np.ndarray, tol: float = 1e-9) -> bool:
    """True if 2D point p lies in any of the triangles P (T, 3, 2)."""
    a = P[:, 0]
    v0, v1, w = P[:, 1] - a, P[:, 2] - a, p - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (w * v0).sum(1), (w * v1).sum(1)
    den = d00 * d11 - d01 * d01
    ok = den > 1e-12 * d00 * d11
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (d11 * d20 - d01 * d21) / den
        r = (d00 * d21 - d01 * d20) / den
    return bool(np.any(ok & (s >= -tol) & (r >= -tol) & (s + r <= 1 + tol)))


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """2D convex hull (Andrew's monotone chain), counter-clockwise, no repeated end point."""
    pts = np.unique(np.round(pts, 9), axis=0)
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _point_in_hull(pts: np.ndarray, p: np.ndarray, tol: float = HULL_TOL) -> bool:
    """True if 2D point p lies in the convex hull of pts or within `tol` of it.

    The tolerance makes degenerate hulls usable: a line contact (cylinder on a plane) or a point
    contact supports a CoG that lies on it within `tol`.
    """
    h = _convex_hull(pts)
    if len(h) >= 3:
        e = np.roll(h, -1, axis=0) - h
        w = p - h
        if np.all(e[:, 0] * w[:, 1] - e[:, 1] * w[:, 0] >= 0):
            return True
    if len(h) == 1:
        return bool(np.linalg.norm(p - h[0]) <= tol)
    a, b = h, np.roll(h, -1, axis=0)  # boundary segments (a 2-point hull gives the segment twice)
    ab = b - a
    s = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-30), 0.0, 1.0)
    return bool(np.min(np.linalg.norm(a + s[:, None] * ab - p, axis=1)) <= tol)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


class CostModel:
    """All cost terms of the spec on one AoG; pure numpy (the geometry is in `tables`).

    Internal ("ACO") path cost: sum over enabled terms of weight * term, with M1/M6 divided by
    their DP range but without subtracting the DP minimum (spec section 5). The reported cost
    subtracts that constant offset, so every reported normalized term lies in [0, 1].
    """

    def __init__(self, aog, tables: TermTables, config: OptimizationConfig):
        validate_config(config, tables.has_tool_column, tables.has_group_column)
        if not aog.complete:
            raise ValueError("The AoG is incomplete (root cannot reach the terminal); nothing to optimize")
        N = aog.n_components
        if list(tables.component_ids) != list(aog.component_ids):
            raise ValueError("TermTables.component_ids do not match the AoG component order")
        if config.use_stability and tables.patches is None:
            raise ValueError("use_stability=True but the term tables have no contact patches")
        self.aog, self.tables, self.config = aog, tables, config
        self.N = N
        self.weights = resolve_weights(config)
        w = self.weights
        self.is_stander = np.asarray(aog.is_stander, dtype=bool)
        self.n_standers = int(self.is_stander.sum())
        self.travel = np.asarray(aog.travel, dtype=float)
        self.neg = np.asarray(aog.neg, dtype=np.int64)
        self.directions = np.asarray(aog.directions, dtype=float)
        self.Dp = len(self.directions)
        self.edge_comp = aog.edge_comp.astype(np.int64)
        self.edge_dirs = aog.edge_dirs[:, 0]
        self.popcount = (N - aog.node_level.astype(np.int64))  # t of the out-edges of each node

        # Live edges in CSR form (edges are sorted by src, so the filtered list stays sorted).
        self.live_edge = aog.live[aog.edge_src] & aog.live[aog.edge_dst]
        self.lv_edges = np.flatnonzero(self.live_edge).astype(np.int64)
        counts = np.bincount(aog.edge_src[self.lv_edges], minlength=aog.n_nodes)
        self.lv_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        # Moves = (edge, free direction) pairs; move id = move_ptr[e] + rank of k among e's bits.
        self.move_ptr = np.concatenate(([0], np.cumsum(np.bitwise_count(self.edge_dirs)))).astype(np.int64)
        self.n_moves = int(self.move_ptr[-1])

        if self.n_standers and config.use_tool_change:
            missing = [aog.component_labels[c] for c in np.flatnonzero(self.is_stander)
                       if not tables.tool[c]]
            if missing:
                raise ValueError(f"use_tool_change=True but these standers have no 'tool' in the BOM: {missing}")
        if config.use_grouping:
            missing = [aog.component_labels[c] for c in range(N) if not tables.group[c]]
            if missing:
                raise ValueError(f"use_grouping=True but these parts have no 'group' in the BOM: {missing}")

        # M6 size score S_i (Eq. 2.1, Nb divided by its mean).
        vol, area = np.asarray(tables.volume, float), np.asarray(tables.area, float)
        nb = tables.n_contacts.astype(float)
        size = np.zeros(N)
        for coef, x in ((config.size_alpha, vol), (config.size_beta, area), (config.size_gamma, nb)):
            if coef and x.mean() > 0:
                size += coef * x / x.mean()
        self.size = size

        # Min-max bounds of the additive terms by exact DP over the live layers.
        tmin, tmax, _ = self._edge_travel_minmax()
        self.travel_lo, self.travel_hi = self._dp_value(tmin, "min"), self._dp_value(tmax, "max")
        edge_size = self.size[self.edge_comp] * self.popcount[aog.edge_src] / N
        self.size_lo, self.size_hi = self._dp_value(edge_size, "min"), self._dp_value(edge_size, "max")

        def per_unit(term, rng):
            return w[term] / rng if term in w and rng > 0 else 0.0

        n1 = max(N - 1, 1)
        self.c_travel = per_unit("travel", self.travel_hi - self.travel_lo)
        self.c_size = per_unit("size_contacts", self.size_hi - self.size_lo)
        self.c_dir = per_unit("direction_change", n1)
        self.c_tool = per_unit("tool_change", max(self.n_standers - 1, 1))
        self.c_group = per_unit("grouping", n1)
        self.c_stab = per_unit("stability", n1)  # per class-1 insertion
        self.c_grav = per_unit("gravity_change", n1)  # per 180 degrees of gravity change
        self.offset = self.c_travel * self.travel_lo + self.c_size * self.size_lo

        # Key ids for the generic "consecutive steps share a key" term (M3, M7).
        tools = sorted({t for c, t in enumerate(tables.tool) if t and self.is_stander[c]})
        self.tool_id = np.array([tools.index(tables.tool[c]) if self.is_stander[c] and tables.tool[c]
                                 else -1 for c in range(N)], dtype=np.int64)
        keys = [g if g else f"#{c}" for c, g in enumerate(tables.group)]  # blanks only when M7 is off
        groups = sorted(set(keys))
        self.group_id = np.array([groups.index(g) for g in keys], dtype=np.int64)
        gmask = {g: sum(1 << c for c in range(N) if keys[c] == g) for g in groups}
        self.group_mask = np.array([gmask[g] for g in keys], dtype=U64)

        # Cost-to-go (spec 6.3): exact min remaining static cost (M1 + M6) from each node.
        self.cost_to_go = None
        if config.use_cost_to_go and (self.c_travel or self.c_size):
            self.cost_to_go = self._dp_table(self._static_edge_cost(tmin), "min")

        # Gravity directions and the switch angle table (fraction of 180 degrees).
        G = np.asarray(config.gravity_dirs, dtype=float).reshape(-1, 3)
        G = G / np.linalg.norm(G, axis=1, keepdims=True)
        self.gravity = G
        self.switch = np.arccos(np.clip(G @ G.T, -1.0, 1.0)) / np.pi
        self.dir_falls = self.directions @ G.T > FALL_TOL  # (D', G)
        self.stability_cache: dict[tuple[int, int], np.ndarray] = {}  # (state S, i) -> class1 (G,)

    # -- DP helpers --------------------------------------------------------------------------

    def _edge_travel_minmax(self):
        """Per edge: min and max travel over its free directions, and the argmin direction."""
        E = self.aog.n_edges
        tmin, tmax = np.full(E, np.inf), np.full(E, -np.inf)
        kmin = np.zeros(E, dtype=np.int64)
        for k in range(self.Dp):
            has = ((self.edge_dirs >> U64(k)) & U64(1)).astype(bool)
            val = self.travel[self.edge_comp, k]
            better = has & (val < tmin)
            tmin = np.where(better, val, tmin)
            kmin = np.where(better, k, kmin)
            tmax = np.where(has & (val > tmax), val, tmax)
        return tmin, tmax, kmin

    def _dp_table(self, cost_e: np.ndarray, mode: str) -> np.ndarray:
        """best[v]: min (or max) additive cost from node v to the terminal over live edges."""
        aog = self.aog
        best = np.full(aog.n_nodes, np.inf if mode == "min" else -np.inf)
        best[aog.terminal] = 0.0
        reduce = np.minimum if mode == "min" else np.maximum
        for L in range(self.N - 1, -1, -1):
            a, b = aog.out_ptr[aog.level_offsets[L]], aog.out_ptr[aog.level_offsets[L + 1]]
            es = np.arange(a, b)[self.live_edge[a:b]]
            reduce.at(best, aog.edge_src[es], cost_e[es] + best[aog.edge_dst[es]])
        return best

    def _dp_value(self, cost_e: np.ndarray, mode: str) -> float:
        return float(self._dp_table(cost_e, mode)[self.aog.root])

    def _static_edge_cost(self, tmin: np.ndarray) -> np.ndarray:
        """Weighted static cost (M1 at the edge's best direction + M6) of every edge."""
        return (self.c_travel * tmin
                + self.c_size * self.size[self.edge_comp] * self.popcount[self.aog.edge_src] / self.N)

    def dp_optimum(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact optimum of the static terms (M1 + M6) by DP: (edge path, removal directions).

        The exact reference of spec section 8 when only use_travel / use_size_contacts are on;
        with other terms enabled it is only the optimum of the static part.
        """
        aog = self.aog
        tmin, _, kmin = self._edge_travel_minmax()
        cost_e = self._static_edge_cost(tmin)
        best = self._dp_table(cost_e, "min")
        edges, ks, v = [], [], aog.root
        while v != aog.terminal:
            es = self.lv_edges[self.lv_ptr[v]:self.lv_ptr[v + 1]]
            vals = cost_e[es] + best[aog.edge_dst[es]]
            e = int(es[np.argmin(vals)])
            edges.append(e)
            ks.append(int(kmin[e]) if self.c_travel else int(self._edge_ks(e)[0]))
            v = int(aog.edge_dst[e])
        return np.array(edges, dtype=np.int64), np.array(ks, dtype=np.int64)

    # -- terms -------------------------------------------------------------------------------

    def move_ids(self, edges, ks) -> np.ndarray:
        """Pheromone index of the moves (edge, direction)."""
        below = self.edge_dirs[edges] & ((U64(1) << np.asarray(ks).astype(U64)) - U64(1))
        return self.move_ptr[edges] + np.bitwise_count(below).astype(np.int64)

    def _edge_ks(self, e: int) -> np.ndarray:
        word = int(self.edge_dirs[e])
        return np.array([k for k in range(self.Dp) if (word >> k) & 1], dtype=np.int64)

    def _dir_levels(self, prev_k: np.ndarray, k: np.ndarray) -> np.ndarray:
        """M2 level between consecutive directions (removal or insertion, same relations):
        same 0, same axis opposite sign dir_change_axis_ratio, other axis 1."""
        prev_safe = np.where(prev_k >= 0, prev_k, 0)
        lvl = np.where(self.neg[prev_safe] == k, self.config.dir_change_axis_ratio, 1.0)
        return np.where((prev_k < 0) | (prev_k == k), 0.0, lvl)

    def stability_classes(self, e: int) -> np.ndarray:
        """class1[g] for inserting edge e's component into the state without it (cached by (S, i))."""
        S, i = int(self.aog.node_mask[self.aog.edge_src[e]]), int(self.edge_comp[e])
        key = (S, i)
        cls = self.stability_cache.get(key)
        if cls is None:
            present = S & ~(1 << i)
            if present == 0:  # t = 1: nothing to test, class 2 for every g
                cls = np.zeros(len(self.gravity), dtype=bool)
            else:
                cls = ~self._supported(i, present)
                if self.config.stability_fall_test:
                    cls |= self.dir_falls[self._edge_ks(e)].any(axis=0)
            self.stability_cache[key] = cls
        return cls

    def _supported(self, i: int, present: int) -> np.ndarray:
        """supported[g]: CoG of i, projected along g, lies in the support region of the patches
        between i and the present components whose normal opposes g."""
        tris, normals = [], []
        for j in _bits(present, self.N):
            tri, normal = self.tables.contact_triangles(i, j)
            if len(tri):
                tris.append(tri)
                normals.append(normal)
        out = np.zeros(len(self.gravity), dtype=bool)
        if not tris:
            return out
        tri, normal = np.concatenate(tris), np.concatenate(normals)
        for gi, g in enumerate(self.gravity):
            keep = normal @ g < -NORMAL_TOL
            if not keep.any():
                continue
            B = _plane_basis(g)
            P, p = tri[keep] @ B, self.tables.cog[i] @ B
            out[gi] = (_point_in_triangles(P, p) if self.config.support_region == "union"
                       else _point_in_hull(P.reshape(-1, 2), p))
        return out

    def move_costs(self, edges, ks, prev_comp, prev_dir, last_tool) -> np.ndarray:
        """Weighted immediate cost of moves (edge, direction) given the ant memory (spec 6.3).

        Walk order is disassembly: the previous move is assembly step t + 1. History terms
        are charged on the pair (t, t + 1); stability uses the min over g (no switching). With
        use_cost_to_go the exact remaining static cost from the move's destination is added
        (minus that of its source).
        """
        aog = self.aog
        comp = self.edge_comp[edges]
        src = aog.edge_src[edges]
        c = np.zeros(len(edges))
        if self.c_travel:
            c += self.c_travel * self.travel[comp, ks]
        if self.c_size:
            c += self.c_size * self.size[comp] * self.popcount[src] / self.N
        if self.c_dir:
            c += self.c_dir * self._dir_levels(prev_dir, ks)
        if self.c_tool:
            c += self.c_tool * (self.is_stander[comp] & (last_tool >= 0) & (self.tool_id[comp] != last_tool))
        if self.c_group:
            # Penalty for step t + 1 (prev): key changes while step t's group is unfinished after t.
            has = prev_comp >= 0
            pc = np.where(has, prev_comp, 0)
            unfinished = (self.group_mask[comp] & ~aog.node_mask[src]) != 0
            c += self.c_group * (has & (self.group_id[pc] != self.group_id[comp]) & unfinished)
        if self.c_stab:
            uniq, inv = np.unique(edges, return_inverse=True)
            all1 = np.array([self.stability_classes(int(e)).all() for e in uniq])
            c += self.c_stab * all1[inv]
        if self.cost_to_go is not None:
            # Relative to the node's own cost-to-go, so the static part of c becomes the move's
            # regret (0 on an optimal static path); a per-node constant would flatten eta.
            c += self.cost_to_go[aog.edge_dst[edges]] - self.cost_to_go[src]
            np.maximum(c, 0.0, out=c)  # regret >= 0 up to rounding
        return c

    def _gravity_viterbi(self, cls: np.ndarray) -> tuple[np.ndarray, float]:
        """Exact best gravity sequence for class1 table cls (N, G) in assembly order."""
        unary = self.c_stab * cls.astype(float)
        if not self.c_grav:
            g = np.argmin(unary, axis=1)
            return g, float(unary.min(axis=1).sum())
        pair = self.c_grav * self.switch
        D = unary[0].copy()
        back = np.zeros(cls.shape, dtype=np.int64)
        for s in range(1, len(cls)):
            tot = D[:, None] + pair
            back[s] = np.argmin(tot, axis=0)
            D = tot.min(axis=0) + unary[s]
        g = np.zeros(len(cls), dtype=np.int64)
        g[-1] = int(np.argmin(D))
        for s in range(len(cls) - 1, 0, -1):
            g[s - 1] = back[s, g[s]]
        return g, float(D.min())

    def check_path(self, edges: np.ndarray, ks: np.ndarray) -> None:
        aog = self.aog
        if len(edges) != self.N or len(ks) != self.N:
            raise ValueError(f"A path needs exactly N = {self.N} edges")
        if int(aog.edge_src[edges[0]]) != aog.root or int(aog.edge_dst[edges[-1]]) != aog.terminal:
            raise ValueError("Path does not run from the root to the terminal")
        if np.any(aog.edge_src[edges[1:]] != aog.edge_dst[edges[:-1]]):
            raise ValueError("Path edges are not connected")
        if np.any(((self.edge_dirs[edges] >> ks.astype(U64)) & U64(1)) == 0):
            raise ValueError("A direction is not free on its edge")

    def evaluate(self, edges, ks) -> dict:
        """Exact cost of a walk-order path (edges, removal directions), incl. the gravity Viterbi.

        Returns per-step arrays in assembly order (index s = step t - 1).
        """
        edges, ks = np.asarray(edges, dtype=np.int64), np.asarray(ks, dtype=np.int64)
        self.check_path(edges, ks)
        aog, N = self.aog, self.N
        e_a, k_a = edges[::-1], ks[::-1]  # assembly order
        comp = self.edge_comp[e_a]
        S = aog.node_mask[aog.edge_src[e_a]]  # state after this assembly step
        t = np.arange(1, N + 1)
        steps: dict[str, np.ndarray] = {}  # per-step weighted internal contributions
        raw: dict[str, float] = {}
        norm: dict[str, float] = {}
        n1 = max(N - 1, 1)

        if "travel" in self.weights:
            v = self.travel[comp, k_a]
            raw["travel"] = float(v.sum())
            rng = self.travel_hi - self.travel_lo
            norm["travel"] = (raw["travel"] - self.travel_lo) / rng if rng > 0 else 0.0
            steps["travel"] = self.c_travel * v
        if "size_contacts" in self.weights:
            v = self.size[comp] * t / N
            raw["size_contacts"] = float(v.sum())
            rng = self.size_hi - self.size_lo
            norm["size_contacts"] = (raw["size_contacts"] - self.size_lo) / rng if rng > 0 else 0.0
            steps["size_contacts"] = self.c_size * v
        if "direction_change" in self.weights:
            v = np.zeros(N)
            v[1:] = self._dir_levels(k_a[:-1], k_a[1:])
            raw["direction_change"] = float(v.sum())
            norm["direction_change"] = raw["direction_change"] / n1
            steps["direction_change"] = self.c_dir * v
        if "tool_change" in self.weights:
            v = np.zeros(N)
            idx = np.flatnonzero(self.is_stander[comp])
            tid = self.tool_id[comp[idx]]
            v[idx[1:]] = (tid[1:] != tid[:-1])
            raw["tool_change"] = float(v.sum())
            norm["tool_change"] = raw["tool_change"] / max(self.n_standers - 1, 1)
            steps["tool_change"] = self.c_tool * v
        if "grouping" in self.weights:
            v = np.zeros(N)
            gid = self.group_id[comp]
            unfinished = (self.group_mask[comp] & ~S) != 0
            v[1:] = (gid[1:] != gid[:-1]) & unfinished[:-1]
            raw["grouping"] = float(v.sum())
            norm["grouping"] = raw["grouping"] / n1
            steps["grouping"] = self.c_group * v
        gravity = cls_at = None
        if "stability" in self.weights:
            cls = np.array([self.stability_classes(int(e)) for e in e_a])  # (N, G)
            gravity, _ = self._gravity_viterbi(cls)
            cls_at = cls[np.arange(N), gravity]
            raw["stability"] = float(cls_at.sum())  # class-1 insertions
            norm["stability"] = float(cls_at.sum()) / n1
            steps["stability"] = self.c_stab * cls_at
            if "gravity_change" in self.weights:
                v = np.zeros(N)
                v[1:] = self.switch[gravity[:-1], gravity[1:]]
                raw["gravity_change"] = float(v.sum())  # sum of angle / 180 degrees
                norm["gravity_change"] = float(v.sum()) / n1
                steps["gravity_change"] = self.c_grav * v

        internal = float(sum(s.sum() for s in steps.values()))
        return {
            "internal_cost": internal,
            "total_cost": internal - self.offset,
            "raw": raw,
            "normalized": norm,
            "steps": steps,
            "gravity": gravity,
            "stability_class": None if cls_at is None else np.where(cls_at, 1, 2),
        }


# ---------------------------------------------------------------------------
# ACO
# ---------------------------------------------------------------------------


def _sample_segments(logw: np.ndarray, starts: np.ndarray, rng, ends: np.ndarray | None = None) -> np.ndarray:
    """One index per segment [starts[s], ends[s]) drawn with probability proportional to exp(logw).

    Without `ends`, the segments are contiguous and cover logw (ends = next start).
    """
    if ends is None:
        ends = np.append(starts[1:], len(logw))
    size = ends - starts  # all segments are non-empty
    s_end = np.cumsum(size)
    s_start = s_end - size
    idx = np.repeat(starts - s_start, size) + np.arange(s_end[-1])  # gathered segment members
    lw = logw[idx]
    w = np.exp(lw - np.repeat(np.maximum.reduceat(lw, s_start), size))
    cw = np.cumsum(w)
    base = np.where(s_start > 0, cw[np.maximum(s_start - 1, 0)], 0.0)
    target = base + rng.random(len(starts)) * (cw[s_end - 1] - base)
    local = np.clip(np.searchsorted(cw, target, side="right"), s_start, s_end - 1)
    return idx[local]


def _walk(model: CostModel, n_ants: int, tau=None, rng=None, alpha=1.0, beta=3.0, greedy=False):
    """All ants walk root -> terminal in lockstep (every path has exactly N edges).

    Returns (edges (A, N), removal directions (A, N)) in walk (disassembly) order.
    """
    aog, N, A = model.aog, model.N, n_ants
    node = np.full(A, aog.root, dtype=np.int64)
    prev_comp = np.full(A, -1, dtype=np.int64)
    prev_dir = np.full(A, -1, dtype=np.int64)
    last_tool = np.full(A, -1, dtype=np.int64)
    path_e = np.zeros((A, N), dtype=np.int64)
    path_k = np.zeros((A, N), dtype=np.int64)
    bitk = np.arange(model.Dp, dtype=U64)
    ants = np.arange(A)
    for step in range(N):
        # Candidate live edges of every ant, flattened (owner = ant index, sorted).
        start = model.lv_ptr[node]
        cnt = model.lv_ptr[node + 1] - start
        owner_e = np.repeat(ants, cnt)
        within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        cand = model.lv_edges[np.repeat(start, cnt) + within]
        # Expand each edge into its free directions.
        r, k = np.nonzero(((model.edge_dirs[cand][:, None] >> bitk) & U64(1)).astype(bool))
        owner, e = owner_e[r], cand[r]
        c = model.move_costs(e, k, prev_comp[owner], prev_dir[owner], last_tool[owner])
        starts = np.searchsorted(owner, ants)
        if greedy:
            order = np.lexsort((c, owner))
            pick = order[np.searchsorted(owner[order], ants)]
        else:
            # Two-stage choice: an edge (weighted by its best direction), then a direction within
            # it. Sampling moves directly would give an edge one ticket per free direction, so
            # nearly free parts (up to D' directions) would almost always be removed first.
            logw = alpha * np.log(tau[model.move_ids(e, k)].astype(float)) - beta * np.log(EPS + c)
            move_starts = np.searchsorted(r, np.arange(len(cand)))  # moves of each candidate edge
            edge_logw = np.maximum.reduceat(logw, move_starts)
            row = _sample_segments(edge_logw, np.searchsorted(owner_e, ants), rng)
            pick = _sample_segments(logw, move_starts[row], rng,
                                    np.append(move_starts, len(logw))[row + 1])
        e_pick, k_pick = e[pick], k[pick]
        comp = model.edge_comp[e_pick]
        path_e[:, step], path_k[:, step] = e_pick, k_pick
        node = aog.edge_dst[e_pick].astype(np.int64)
        prev_comp, prev_dir = comp, k_pick
        last_tool = np.where(model.is_stander[comp], model.tool_id[comp], last_tool)
    return path_e, path_k


def run_aco(model: CostModel) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """MAX-MIN Ant System over the AoG (spec 6.4). Returns (edges, dirs, evaluation, history)."""
    cfg, N = model.config, model.N
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    n_ants = cfg.n_ants if cfg.n_ants is not None else max(2 * N, 10)
    ratio = cfg.tau_min_ratio if cfg.tau_min_ratio is not None else 1.0 / (2 * N)
    evals: dict[bytes, dict] = {}

    def evaluate(pe, pk):
        key = pe.tobytes() + pk.tobytes()
        if key not in evals:
            evals[key] = model.evaluate(pe, pk)
        return evals[key]

    ge, gk = _walk(model, 1, greedy=True)
    best_e, best_k = ge[0], gk[0]
    best = evaluate(best_e, best_k)
    greedy_cost = best["total_cost"]
    tau_max = 1.0 / (cfg.rho * (EPS + best["internal_cost"]))
    tau = np.full(model.n_moves, tau_max, dtype=np.float32)
    history_cost, stagnation, stop = [], 0, "iterations"
    it = 0
    for it in range(cfg.n_iterations):
        pe, pk = _walk(model, n_ants, tau, rng, cfg.alpha, cfg.beta)
        results = [evaluate(pe[a], pk[a]) for a in range(n_ants)]
        costs = np.array([r["internal_cost"] for r in results])
        ib = int(np.argmin(costs))
        if costs[ib] < best["internal_cost"] - 1e-12:
            best, best_e, best_k, stagnation = results[ib], pe[ib].copy(), pk[ib].copy(), 0
        else:
            stagnation += 1
        history_cost.append(best["total_cost"])

        tau *= (1.0 - cfg.rho)
        if cfg.global_best_interval and (it + 1) % cfg.global_best_interval == 0:
            dep_e, dep_k, dep_c = best_e, best_k, best["internal_cost"]
        else:
            dep_e, dep_k, dep_c = pe[ib], pk[ib], costs[ib]
        tau[model.move_ids(dep_e, dep_k)] += 1.0 / (EPS + dep_c)
        tau_max = 1.0 / (cfg.rho * (EPS + best["internal_cost"]))
        np.clip(tau, tau_max * ratio, tau_max, out=tau)
        if it % 20 == 0:
            log.debug("ACO iteration %d: best %.6f, iteration best %.6f", it, best["total_cost"],
                      costs[ib] - model.offset)
        if stagnation >= cfg.stagnation_limit:
            stop = "stagnation"
            break

    history = {
        "best_cost": history_cost,
        "greedy_cost": greedy_cost,
        "iterations": it + 1 if cfg.n_iterations else 0,
        "n_ants": n_ants,
        "seed": cfg.seed,
        "stopped_by": stop,
        "distinct_paths_evaluated": len(evals),
        "aco_cost": best["internal_cost"],
        "wall_time_s": time.perf_counter() - t0,
    }
    return best_e, best_k, best, history


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    steps: list[dict]  # assembly order
    total_cost: float  # sum of weighted normalized terms, in [0, 1]
    cost_breakdown: dict[str, dict]  # term -> raw, normalized, weight, weighted
    history: dict
    config_hash: str
    edge_path: list[int] = field(default_factory=list)  # AoG edges, disassembly (walk) order
    removal_directions: list[int] = field(default_factory=list)  # walk order


def _build_result(model: CostModel, edges, ks, ev: dict, history: dict, config_hash: str) -> OptimizationResult:
    aog, N = model.aog, model.N
    e_a, k_a = edges[::-1], ks[::-1]
    steps = []
    for s in range(N):
        e, k = int(e_a[s]), int(k_a[s])
        c = int(model.edge_comp[e])
        gi = None if ev["gravity"] is None else int(ev["gravity"][s])
        travel = float(model.travel[c, k])
        steps.append({
            "step": s + 1,
            "component": c,
            "component_id": aog.component_ids[c],
            "component_label": aog.component_labels[c],
            "is_stander": bool(model.is_stander[c]),
            "edge": e,
            "removal_direction_index": k,
            "direction_index": int(model.neg[k]),  # insertion direction
            "direction": model.directions[model.neg[k]].tolist(),
            "travel": travel,
            # start point = final position + start_offset; the part then moves along `direction`
            "start_offset": (travel * model.directions[k]).tolist(),
            "gravity_index": gi,
            "gravity": None if gi is None else model.gravity[gi].tolist(),
            "stability_class": None if ev["stability_class"] is None else int(ev["stability_class"][s]),
            "costs": {term: float(v[s]) for term, v in ev["steps"].items()},
        })
    breakdown = {term: {"raw": ev["raw"][term], "normalized": ev["normalized"][term],
                        "weight": wt, "weighted": wt * ev["normalized"][term]}
                 for term, wt in model.weights.items()}
    return OptimizationResult(
        steps=steps,
        total_cost=float(ev["total_cost"]),
        cost_breakdown=breakdown,
        history=history,
        config_hash=config_hash,
        edge_path=[int(e) for e in edges],
        removal_directions=[int(k) for k in ks],
    )


def _apply_turns(aog, prep, tables: TermTables, steps: list[dict]) -> None:
    """Add `turns`/`turns_pitch`/`turns_note`/`turns_nut_label` to each step, in place (display
    only -- docs/stander_processing-spec.md section 12; the ACO cost and the AoG are unaffected).

    `turns_nut_label` is an extra field beyond the spec's literal 3 (`turns`, `turns_pitch`,
    `turns_note`): the "partner rotates" case (12.2 row 3) needs the responsible nut's label for
    the output text (12.4, "thread of nut <label>"), and no listed field carries it, so it is
    added here rather than re-derived twice downstream.
    """
    index = {cid: i for i, cid in enumerate(aog.component_ids)}
    step_of = {s["component"]: i for i, s in enumerate(steps)}
    for s in steps:
        s["turns"] = None
        s["turns_pitch"] = None
        s["turns_note"] = None
        s["turns_nut_label"] = None

    # partner bit -> [(nut bit, nut pitch)], only for nuts whose partner rotates (row 3).
    partner_groups: dict[int, list[tuple[int, float]]] = {}

    for c, cid in enumerate(aog.component_ids):
        rec = prep.stander_records.get(cid)
        if rec is None or c not in step_of:
            continue
        if rec.category == 1:  # screw: own step, own pitch (row 1)
            p = tables.pitch[c]
            if p is None:
                continue
            i = step_of[c]
            steps[i]["turns"] = steps[i]["travel"] / p
            steps[i]["turns_pitch"] = p
        elif rec.category == 2 and rec.partner is not None:  # nut
            partner_bit = index.get(rec.partner)
            nut_pitch = tables.pitch[c]
            if partner_bit is None or nut_pitch is None or partner_bit not in step_of:
                continue
            partner_before = step_of[partner_bit] < step_of[c]
            if partner_before:  # row 2: the nut rotates, on its own step
                i = step_of[c]
                steps[i]["turns"] = steps[i]["travel"] / nut_pitch
                steps[i]["turns_pitch"] = nut_pitch
            else:  # row 3: the partner rotates -- unless it's a screw, which already got its own
                   # turns above (12.2: "the nut then only matters when the nut comes after the screw").
                partner_rec = prep.stander_records.get(rec.partner)
                if partner_rec is not None and partner_rec.category == 1:
                    continue
                partner_groups.setdefault(partner_bit, []).append((c, nut_pitch))

    for partner_bit, nuts in partner_groups.items():
        i = step_of[partner_bit]
        first_nut, first_pitch = nuts[0]
        pitches = {p for _, p in nuts}
        if len(pitches) > 1:
            log.warning("Partner %r is rotated by several nuts with different pitches (%s); using %r's pitch",
                       aog.component_labels[partner_bit],
                       ", ".join(aog.component_labels[n] for n, _ in nuts), aog.component_labels[first_nut])
        steps[i]["turns"] = steps[i]["travel"] / first_pitch
        steps[i]["turns_pitch"] = first_pitch
        steps[i]["turns_note"] = "nut"
        steps[i]["turns_nut_label"] = aog.component_labels[first_nut]


def compute_config_hash(aog, tables: TermTables, config: OptimizationConfig) -> str:
    cfg = asdict(config)
    cfg.pop("force")
    h = hashlib.sha256()
    h.update(f"aog:{aog.cache_key};tables:{tables.cache_key};".encode())
    h.update(json.dumps(cfg, sort_keys=True, default=list).encode())
    h.update(json.dumps(resolve_weights(config), sort_keys=True).encode())
    return h.hexdigest()


def optimize(aog, tables: TermTables, config: OptimizationConfig | None = None,
             stability_cache: dict | None = None) -> OptimizationResult:
    """Run the ACO on an AoG with ready term tables (pure numpy, no FreeCAD, no disk cache)."""
    config = config or OptimizationConfig()
    model = CostModel(aog, tables, config)
    if stability_cache:
        model.stability_cache.update(stability_cache)
    edges, ks, ev, history = run_aco(model)
    result = _build_result(model, edges, ks, ev, history, compute_config_hash(aog, tables, config))
    result.history["stability_cache"] = model.stability_cache  # handed back for caching, not saved
    _log_summary(result)
    return result


def _log_summary(result: OptimizationResult) -> None:
    h = result.history
    log.info("Optimization: total cost %.4f (greedy %.4f), %d iterations x %d ants, stopped by %s, "
             "%d distinct paths, %.2f s", result.total_cost, h["greedy_cost"], h["iterations"], h["n_ants"],
             h["stopped_by"], h["distinct_paths_evaluated"], h["wall_time_s"])
    log.info("Cost breakdown: %s", ", ".join(f"{t}={b['normalized']:.3f}x{b['weight']:.3f}"
                                             for t, b in result.cost_breakdown.items()))
    for s in result.steps:
        log.info("  step %d: %s along %s, travel %.1f mm, g=%s, class %s", s["step"], s["component_label"],
                 _fmt_vec(s["direction"]), s["travel"],
                 "-" if s["gravity"] is None else _fmt_vec(s["gravity"]), s["stability_class"] or "-")


def _fmt_vec(v) -> str:
    return "(" + ", ".join(f"{round(float(c), 3) + 0.0:g}" for c in v) + ")"


def _fmt_turns_text(s: dict, decimals: int) -> str:
    """docs/stander_processing-spec.md 12.4: the turns clause for one step, or "" if `turns` is
    None. `decimals` controls the turns count only (2 for assembly_sequence.txt, 1 for the
    manual); the pitch is shown as given, not padded."""
    turns = s.get("turns")
    if turns is None:
        return ""
    pitch = s.get("turns_pitch")
    pitch_txt = f"{pitch:.2f}".rstrip("0").rstrip(".") if pitch is not None else "?"
    if s.get("turns_note") == "nut":
        return (f", rotate {turns:.{decimals}f} turns clockwise (thread of nut "
                f"{s.get('turns_nut_label')}, pitch {pitch_txt} mm)")
    return f", rotate {turns:.{decimals}f} turns clockwise (pitch {pitch_txt} mm)"


# ---------------------------------------------------------------------------
# FreeCAD side: term tables and contact patches
# ---------------------------------------------------------------------------


def _read_bom(bom_path) -> tuple[list[str], dict[str, dict], bytes]:
    data = Path(bom_path).read_bytes()
    with open(bom_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = [c.strip() for c in (reader.fieldnames or [])]
        rows = {}
        for row in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if row.get("label"):
                rows[row["label"]] = row
    if "label" not in columns or "is_stander" not in columns:
        raise ValueError(f"BOM {bom_path} needs the columns 'label' and 'is_stander'")
    return columns, rows, data


def _component_shapes(aog, prep) -> list:
    """Global shapes in AoG order (shared helper in shapes.py)."""
    from shapes import component_shapes

    return component_shapes(aog.component_ids, aog.is_stander, prep)


def _center_of_mass(shape) -> list[float]:
    solids = shape.Solids
    if not solids:
        raise ValueError("Component shape has no solid; cannot compute its centre of gravity")
    vols = np.array([s.Volume for s in solids])
    cogs = np.array([[s.CenterOfMass.x, s.CenterOfMass.y, s.CenterOfMass.z] for s in solids])
    return (vols @ cogs / vols.sum()).tolist()


def _contact_matrix(aog, interference_result, prep) -> np.ndarray:
    """Same contact graph as the AoG connectivity rule: interference contacts between movable
    parts plus StanderRecord.contact_neighbors."""
    if getattr(interference_result, "contact", None) is None:
        raise ValueError("InterferenceResult has no contact matrix; re-run identify_interferences")
    N = aog.n_components
    index = {cid: i for i, cid in enumerate(aog.component_ids)}
    contact = np.zeros((N, N), dtype=bool)
    mov = [index[c] for c in interference_result.component_ids]
    contact[np.ix_(mov, mov)] = np.asarray(interference_result.contact, dtype=bool)
    for sid, rec in prep.stander_records.items():
        for g in rec.contact_neighbors:
            contact[index[sid], index[g]] = contact[index[g], index[sid]] = True
    contact |= contact.T
    np.fill_diagonal(contact, False)
    return contact


def _common_faces(fa, fb) -> list:
    try:
        return [f for f in fa.common(fb).Faces if f.Area > AREA_EPS]
    except Exception:  # noqa: BLE001  (OCC boolean failure: fall back to tessellation)
        return []


def _triangles(face, deflection: float) -> np.ndarray:
    pts, facets = face.tessellate(deflection)
    if not facets:
        return np.zeros((0, 3, 3))
    P = np.array([[p.x, p.y, p.z] for p in pts])
    return P[np.array(facets)]


def _region_triangles(fi, fj, threshold: float, deflection: float) -> np.ndarray:
    """Triangles of the contact region of face fi (component i) against face fj (component j).

    1. Boolean common of the faces (exact for coincident faces).
    2. Parallel planes with a gap <= threshold: fi is moved onto fj's plane, then 1.
    3. Otherwise (curved faces with clearance): fi's tessellation, keeping triangles whose centroid
       lies within `threshold` of fj.
    """
    import FreeCAD
    import Part

    faces = _common_faces(fi, fj)
    if not faces and fi.Surface.TypeId == "Part::GeomPlane" and fj.Surface.TypeId == "Part::GeomPlane":
        ai, aj = FreeCAD.Vector(fi.Surface.Axis), FreeCAD.Vector(fj.Surface.Axis)
        if abs(ai.dot(aj)) > 1 - 1e-6:
            gap = (fj.Surface.Position - fi.Surface.Position).dot(ai)
            if 0 < abs(gap) <= threshold:
                moved = fi.copy()
                moved.translate(ai * gap)
                faces = _common_faces(moved, fj)
    if faces:
        return np.concatenate([_triangles(f, deflection) for f in faces])
    tri = _triangles(fi, deflection)
    if not len(tri):
        return tri
    keep = [fj.distToShape(Part.Vertex(FreeCAD.Vector(*c)))[0] <= threshold for c in tri.mean(axis=1)]
    return tri[np.array(keep, dtype=bool)]


def extract_contact_patches(shape_i, shape_j, threshold: float,
                            deflection: float = PATCH_DEFLECTION) -> tuple[np.ndarray, np.ndarray]:
    """Contact patches between two shapes as triangles: (tri (T, 3, 3), normal (T, 3)).

    normal = outward normal of j's surface at the contact = -(outward normal of i), i.e. it
    points from j into i. A patch carries i under gravity g when normal . g < 0.
    """
    import FreeCAD

    tris, normals = [], []
    for fi in shape_i.Faces:
        box = fi.BoundBox
        box.enlarge(threshold)
        sign = None
        for fj in shape_j.Faces:
            if not box.intersect(fj.BoundBox):
                continue
            try:
                if fi.distToShape(fj)[0] > threshold:
                    continue
            except Exception:  # noqa: BLE001
                continue
            tri = _region_triangles(fi, fj, threshold, deflection)
            for t in tri:
                c = FreeCAD.Vector(*t.mean(axis=0))
                n = fi.normalAt(*fi.Surface.parameter(c))
                if sign is None:  # make sure n is i's outward normal (once per face)
                    step = min(0.01, threshold)
                    inside_minus = shape_i.isInside(c - n * step, 1e-7, False)
                    inside_plus = shape_i.isInside(c + n * step, 1e-7, False)
                    sign = -1.0 if inside_plus and not inside_minus else 1.0
                tris.append(t)
                normals.append([-sign * n.x, -sign * n.y, -sign * n.z])
    if not tris:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    return np.array(tris, dtype=float), np.array(normals, dtype=float)


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path, key: str) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001  (corrupt cache: recompute)
        log.warning("Could not read %s (%s); recomputing", path, exc)
        return None
    return data if data.get("key") == key else None


TABLES_NAME = "term_tables.json"
PATCHES_NAME = "contact_patches.json"
STABILITY_NAME = "stability_classes.json"
RESULT_NAME = "optimization_result.json"
SEQUENCE_NAME = "assembly_sequence.txt"


def _contact_threshold(interference_result) -> float:
    thr = getattr(interference_result, "contact_threshold", None)
    if thr is None:
        from interference_identification import CONTACT_THRESHOLD
        thr = CONTACT_THRESHOLD
    return float(thr)


def tables_cache_key(aog, interference_result, bom_bytes: bytes, config: OptimizationConfig) -> str:
    """Key of the weight- and flag-independent term tables (spec 7)."""
    h = hashlib.sha256(f"v{TABLES_VERSION};aog:{aog.cache_key};"
                       f"interference:{interference_result.cache_key};".encode())
    h.update(b"bom:" + hashlib.sha256(bom_bytes).digest())
    G = np.asarray(config.gravity_dirs, dtype=float).reshape(-1, 3)
    h.update(b"gravity:" + np.round(G / np.linalg.norm(G, axis=1, keepdims=True), 9).tobytes())
    h.update(f"contact:{_contact_threshold(interference_result)};deflection:{PATCH_DEFLECTION};"
             f"normal_tol:{NORMAL_TOL}".encode())
    return h.hexdigest()


def collect_tables(aog, interference_result, prep, config: OptimizationConfig, cache_dir: Path,
                   force: bool = False) -> TermTables:
    """Term tables for the pipeline objects; geometry is cached under cache_dir."""
    if prep.bom_path is None:
        raise ValueError("PreprocessingResult.bom_path is missing; the BOM is needed for M3/M7")
    columns, rows, bom_bytes = _read_bom(prep.bom_path)
    key = tables_cache_key(aog, interference_result, bom_bytes, config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shapes = None

    data = None if force else _read_json(cache_dir / TABLES_NAME, key)
    if data is None:
        shapes = _component_shapes(aog, prep)
        data = {
            "key": key,
            "component_ids": list(aog.component_ids),
            "volume": [float(s.Volume) for s in shapes],
            "area": [float(s.Area) for s in shapes],
            "cog": [_center_of_mass(s) for s in shapes],
            "contact": _contact_matrix(aog, interference_result, prep).astype(int).tolist(),
        }
        _write_json_atomic(cache_dir / TABLES_NAME, data)
    else:
        log.info("Loaded cached term tables from %s", cache_dir)

    tool, group, pitch = [], [], []
    auto_group_override = getattr(prep, "auto_group_override", None) or {}
    for cid, label in zip(aog.component_ids, aog.component_labels):
        row = rows.get(label, {})
        tool.append(row.get("tool") or None)
        # docs/stander_processing-spec.md section 12: pitch is read for screw/nut rows only; blank,
        # unparsable or non-positive gives None and (for an actual screw/nut) a warning naming the label.
        rec = prep.stander_records.get(cid)
        p = None
        raw_pitch = row.get("pitch")
        if raw_pitch:
            try:
                p = float(raw_pitch)
            except ValueError:
                p = None
            if p is not None and p <= 0:
                p = None
        if p is None and "pitch" in columns and rec is not None and rec.category in (1, 2):
            log.warning("Stander %r has no usable 'pitch' in the BOM; no turn count will be shown for it", label)
        pitch.append(p)
        if cid in auto_group_override:
            group.append(auto_group_override[cid])
        else:
            # A component with no BOM group (no row, or a blank cell) is its own group of one --
            # same convention extra_functions/generate_bom.py uses when seeding a fresh BOM, and
            # what lets a fused/virtual container with no row ever reach Optimization at all
            # (docs/subassembly-spec.md section 10; was previously a hard ValueError, H10/R6).
            group.append(row.get("group") or label)
    tables = TermTables(
        component_ids=list(aog.component_ids),
        volume=np.array(data["volume"], dtype=float),
        area=np.array(data["area"], dtype=float),
        contact=np.array(data["contact"], dtype=bool),
        cog=np.array(data["cog"], dtype=float),
        tool=tool,
        group=group,
        pitch=pitch,
        has_tool_column="tool" in columns,
        has_group_column="group" in columns,
        cache_key=key,
    )
    if config.use_stability:  # stability off skips all contact geometry
        tables.patches = _collect_patches(aog, prep, tables, key, cache_dir, force, shapes,
                                          _contact_threshold(interference_result))
    return tables


def _collect_patches(aog, prep, tables, key, cache_dir, force, shapes, threshold):
    """Contact patches for every touching pair, checkpointed per pair (resumes after a crash)."""
    path = cache_dir / PATCHES_NAME
    data = None if force else _read_json(path, key)
    data = data or {"key": key, "pairs": {}}
    pairs = [(i, j) for i, j in zip(*np.nonzero(np.triu(tables.contact, 1)))]
    todo = [(int(i), int(j)) for i, j in pairs if f"{i}|{j}" not in data["pairs"]]
    if todo:
        shapes = shapes or _component_shapes(aog, prep)
        log.info("Extracting contact patches for %d pair(s)", len(todo))
    for i, j in todo:
        t0 = time.perf_counter()
        tri, normal = extract_contact_patches(shapes[i], shapes[j], threshold)
        data["pairs"][f"{i}|{j}"] = {"tri": tri.tolist(), "normal": normal.tolist()}
        _write_json_atomic(path, data)
        log.info("  patches %s - %s: %d triangle(s), %.1f mm^2, %.2f s", aog.component_labels[i],
                 aog.component_labels[j], len(tri), _area(tri), time.perf_counter() - t0)
    out = {}
    for name, p in data["pairs"].items():
        i, j = (int(x) for x in name.split("|"))
        out[(i, j)] = (np.array(p["tri"], dtype=float).reshape(-1, 3, 3),
                       np.array(p["normal"], dtype=float).reshape(-1, 3))
    return out


def _area(tri: np.ndarray) -> float:
    if not len(tri):
        return 0.0
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


# ---------------------------------------------------------------------------
# Persistence of the result
# ---------------------------------------------------------------------------


def save_result(result: OptimizationResult, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = asdict(result)
    data["history"] = {k: v for k, v in data["history"].items() if k != "stability_cache"}
    data["date"] = datetime.datetime.now().isoformat(timespec="seconds")
    _write_json_atomic(cache_dir / RESULT_NAME, data)
    lines = [f"Assembly sequence (total cost {result.total_cost:.4f})", ""]
    for s in result.steps:
        lines.append(f"{s['step']:>3}. {s['component_label']}{' [stander]' if s['is_stander'] else ''}: "
                     f"insert along {_fmt_vec(s['direction'])}, travel {s['travel']:.2f} mm, start offset "
                     f"{_fmt_vec(s['start_offset'])}"
                     + _fmt_turns_text(s, decimals=2)
                     + ("" if s["gravity"] is None else
                        f", gravity {_fmt_vec(s['gravity'])}, class {s['stability_class']}"))
    lines += ["", "Cost breakdown (normalized x weight):"]
    for t, b in result.cost_breakdown.items():
        lines.append(f"  {t}: raw {b['raw']:.4f}, normalized {b['normalized']:.4f}, weight {b['weight']:.4f}, "
                     f"weighted {b['weighted']:.4f}")
    (cache_dir / SEQUENCE_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_result(cache_dir: Path, config_hash: str) -> OptimizationResult | None:
    path = cache_dir / RESULT_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read cached optimization result (%s); recomputing", exc)
        return None
    if data.get("config_hash") != config_hash:
        return None
    data.pop("date", None)
    return OptimizationResult(**data)


def _stability_key(tables: TermTables, config: OptimizationConfig) -> str:
    return hashlib.sha256(f"{tables.cache_key};fall:{config.stability_fall_test};"
                          f"region:{config.support_region}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def _default_cache_dir(prep) -> Path:
    stem = Path(prep.step_path).stem if prep.step_path is not None else prep.reduced_assembly.Name
    return REPO_ROOT / "output" / stem / "optimization"


def optimize_sequence(aog, interference_result, prep, config: OptimizationConfig | None = None,
                      cache_dir: str | Path | None = None, force: bool | None = None) -> OptimizationResult:
    """Optimized assembly sequence for the pipeline objects, with on-disk caching (spec 7)."""
    config = config or OptimizationConfig()
    force = config.force if force is None else force
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir(prep)
    columns, _, _ = _read_bom(prep.bom_path) if prep.bom_path is not None else ([], {}, b"")
    validate_config(config, "tool" in columns, "group" in columns)  # fail before any geometry work

    tables = collect_tables(aog, interference_result, prep, config, cache_dir, force)
    config_hash = compute_config_hash(aog, tables, config)
    if not force:
        cached = load_result(cache_dir, config_hash)
        if cached is not None:
            log.info("Loaded cached optimization result from %s (total cost %.4f)", cache_dir, cached.total_cost)
            return cached

    stab_path, stab_key = cache_dir / STABILITY_NAME, _stability_key(tables, config)
    stability = {}
    stab_data = None if (force or not config.use_stability) else _read_json(stab_path, stab_key)
    if stab_data:
        G = len(config.gravity_dirs)
        for name, bits in stab_data["classes"].items():
            S, i = (int(x) for x in name.split("|"))
            stability[(S, i)] = np.array([(bits >> g) & 1 for g in range(G)], dtype=bool)

    result = optimize(aog, tables, config, stability_cache=stability)
    new_cache = result.history.pop("stability_cache")
    if config.use_stability:
        _write_json_atomic(stab_path, {"key": stab_key, "classes": {
            f"{S}|{i}": int(sum(1 << g for g, b in enumerate(cls) if b)) for (S, i), cls in new_cache.items()}})
    _apply_turns(aog, prep, tables, result.steps)
    save_result(result, cache_dir)
    return result
