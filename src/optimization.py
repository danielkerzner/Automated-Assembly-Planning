"""Optimization (ACO): the lowest-cost assembly sequence through the AoG.

Cost terms M1-M8 and the ACO search; also computes the screw turn counts.

Inputs
    - An `AoG` (graph of all feasible disassembly sequences, from aog_generator.py).
    - The `InterferenceResult` (contacts, cache key) and `PreprocessingResult` (stander records, reduced
      assembly, BOM path), used on the FreeCAD side to build the per-component data.
    - The BOM CSV: `tool` (M3), `group` (M7) and `pitch` (screw turns) columns.
    - An `OptimizationConfig` (which cost terms are on, weights, ACO parameters).
    - Robot mode only: a set of forbidden (state mask, component) edges (`blocked_edges`).
Outputs
    - An `OptimizationResult`: the assembly sequence step by step (component, insertion direction,
      travel, start offset, gravity, stability class, per-term costs), the total cost, the cost
      breakdown and the ACO history. Cached as `optimization_result.json` and `assembly_sequence.txt`.
      Consumed by the Illustration Generator.

What this module does
    Ants walk the AoG from the root (full assembly) to the terminal (empty) over `live` nodes only,
    choosing at each node a live out-edge and then one of its free directions. The walk is a
    disassembly path; every cost term is defined in assembly order (the edge leaving state S is assembly
    step t = popcount(S)) and only ever compares consecutive steps, so the walking direction does not
    matter.

    Costs, not desirabilities: lower is better. Each enabled term is scaled to [0, 1] over a complete path
    and multiplied by its weight (pairwise dominance ranking, renormalised over the enabled terms). The
    ACO heuristic is eta = 1 / (eps + immediate cost + optional exact remaining static cost); pheromone
    (MAX-MIN Ant System) is one float32 per move (AoG edge, free direction) and the deposit is
    1 / (eps + path cost).

Layout:
  OptimizationConfig, resolve_weights, validate_config   configuration
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
# travel = M1, direction_change = M2, tool_change = M3, size_contacts = M6, grouping = M7,
# stability and gravity_change = M8.
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
# Pairwise dominance ranking: 0 = row less important, 1 = equal, 2 = more important.
# The default weight of a term is its row sum divided by the total (DEFAULT_WEIGHTS below).
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

# The six axis directions: the default gravity candidates (the assembly may be built in any orientation).
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
    """Settings of the optimisation (config input of `optimize` and `optimize_sequence`).

    The cost-term flags select which criteria enter the sequence cost; the ACO fields tune the search.
    """

    # Criterion flags, all bool. Recommended: all True when the BOM has the needed columns.
    use_travel: bool = True  # M1: total travel distance.
    use_direction_change: bool = True  # M2: changes of insertion direction between consecutive steps.
    use_tool_change: bool = True  # M3: tool changes between consecutive standers. Needs the BOM `tool` column.
    use_size_contacts: bool = True  # M6: big/highly connected parts should go in early.
    use_grouping: bool = True  # M7: members of one BOM group should be consecutive. Needs the `group` column.
    use_stability: bool = True  # M8: class-1 penalty (part not supported when inserted).
    use_gravity_change: bool = True  # M8: change of the gravity direction between steps. Needs use_stability.
    # dict {term name: value} or None. Weight overrides, merged into DEFAULT_WEIGHTS then renormalised over
    # the enabled terms. Recommended: None (dominance-ranking defaults).
    weights: dict[str, float] | None = None
    # float >= 0. M2: same direction costs 0, same axis opposite sign costs this ratio, another axis costs 1.
    # Recommended: 0.25.
    dir_change_axis_ratio: float = 0.25
    # floats, M6 mix of volume / area / number of contacts in the size score (sum 1; any may be 0).
    # Recommended: 1/3 each.
    size_alpha: float = 1 / 3
    size_beta: float = 1 / 3
    size_gamma: float = 1 / 3
    # M8: tuple of gravity candidates (unit vectors); the best one per step is chosen by Viterbi.
    # Recommended: the six axes (AXES6).
    gravity_dirs: tuple = AXES6
    # bool. M8: also count a part as unstable if one of its free insertion directions lets it fall under g.
    # Recommended: True.
    stability_fall_test: bool = True
    # str, "hull" or "union". "hull" (default, rigid-body assumption): CoG must lie in the convex hull of
    # the projected load-bearing patches (bridging supports such as a block on two rails are stable).
    # "union" (strict): CoG must lie in one of the patches themselves.
    support_region: str = "hull"
    # ACO; None = derived from N (the number of components).
    # bool. Add the exact remaining static cost (M1 + M6, by DP) to eta. Off by default: the lookahead
    # covers only the static terms, so the ants chase the static optimum. Needed for travel-only.
    use_cost_to_go: bool = False
    alpha: float = 1.0  # float. Pheromone exponent in the move probability. Recommended: 1.
    beta: float = 3.0  # float. Heuristic (1 / cost) exponent. Recommended: 3.
    rho: float = 0.2  # float in (0, 1). Pheromone evaporation rate. Recommended: 0.2.
    n_ants: int | None = None  # int or None. Ants per iteration; None = max(2N, 10).
    n_iterations: int = 200  # int. Maximum number of iterations. Recommended: 200.
    tau_min_ratio: float | None = None  # float or None. MAX-MIN lower bound as a fraction of tau_max; None = 1 / (2N).
    stagnation_limit: int = 50  # int. Stop after this many iterations without improvement.
    global_best_interval: int = 10  # int. Every k-th iteration the global best deposits instead of the iteration best.
    seed: int = 0  # int. Random seed (results are reproducible for a fixed seed).
    force: bool = False  # bool. Recompute cached term tables and result.


def resolve_weights(config: OptimizationConfig) -> dict[str, float]:
    """[AUX] Weights of the enabled terms, renormalised to sum to 1 (ratios kept).

    Starts from DEFAULT_WEIGHTS, applies the config overrides, keeps only the enabled terms. Raises
    ValueError for unknown terms, negative weights or a total weight of 0.
    """
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
    """[AUX] Validation: raises ValueError for inconsistent settings, never skips a term silently.

    Checks, for example, that at least one term is on, that M3/M7 have their BOM columns, and that the
    ACO and support-region parameters are in range.
    """
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
    """Per-component data the cost terms need (input of `CostModel`; built by `collect_tables`).

    Everything is indexed in AoG component order (the bit index), so the cost model needs no FreeCAD.
    """

    component_ids: list[str]  # list of obj.Name in AoG order (bit index); must equal aog.component_ids.
    volume: np.ndarray  # (N,) float, mm^3. Component volume (M6 size score).
    area: np.ndarray  # (N,) float, mm^2. Component surface area (M6 size score).
    contact: np.ndarray  # (N, N) bool, symmetric. Touching pairs; Nb_i = contact[i].sum() (M6).
    cog: np.ndarray  # (N, 3) float, mm. Centre of gravity (M8 stability).
    tool: list[str | None]  # BOM `tool` per component (standers; None if blank or absent). M3.
    group: list[str | None]  # BOM `group` per component. M7. A component with no BOM group gets its own label.
    pitch: list[float | None]  # BOM `pitch`, mm/turn (screw/nut rows only; None if blank, non-positive or absent). Display of turns.
    has_tool_column: bool  # bool. The BOM has a `tool` column (validation of use_tool_change).
    has_group_column: bool  # bool. The BOM has a `group` column (validation of use_grouping).
    # Dict {(i, j) with i < j: (tri (T, 3, 3), normal (T, 3))}. Contact patches as triangles; normal = outward
    # normal of j's surface, i.e. pointing from j into i. None = not computed (stability off).
    patches: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None
    cache_key: str = ""  # str. Key of the cached tables (also part of the result's config hash).

    @property
    def n_contacts(self) -> np.ndarray:
        """[AUX] (N,) number of contacts of each component (the Nb of the M6 size score)."""
        return np.asarray(self.contact, dtype=bool).sum(axis=1)

    def contact_triangles(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """[AUX] Contact triangles between i and j with normals pointing from j into i (empty if none stored)."""
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
    """[AUX] Indices of the set bits of `mask` among the first n bits."""
    return [b for b in range(n) if (mask >> b) & 1]


def _plane_basis(g: np.ndarray) -> np.ndarray:
    """[AUX] (3, 2) orthonormal basis of the plane perpendicular to g (to project patches along gravity)."""
    a = np.array([1.0, 0.0, 0.0]) if abs(g[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = a - a.dot(g) * g
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(g, u)], axis=1)


def _point_in_triangles(P: np.ndarray, p: np.ndarray, tol: float = 1e-9) -> bool:
    """[AUX] True if the 2D point p lies in any of the triangles P (T, 3, 2) (barycentric test)."""
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
    """[AUX] 2D convex hull (Andrew's monotone chain), counter-clockwise, no repeated end point."""
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
    """[AUX] True if the 2D point p lies in the convex hull of pts or within `tol` of it.

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


class NoPathError(ValueError):
    """Robot mode: the blocked edges leave no root-to-terminal path."""


def blocked_liveness(aog, blocked_edges) -> tuple[np.ndarray, np.ndarray]:
    """[AUX] Robot mode: liveness of the AoG with the forbidden edges removed. Returns (edge_ok, live).

    `blocked_edges` are (state mask, component) pairs. edge_ok[e] is False for a blocked edge; live[v] is
    True iff node v still reaches the terminal over the remaining edges (computed level by level).
    """
    blocked = {(int(s), int(c)) for s, c in blocked_edges}
    src_mask = aog.node_mask[aog.edge_src]
    edge_ok = np.array([(int(m), int(c)) not in blocked for m, c in zip(src_mask, aog.edge_comp)], dtype=bool)
    live = np.zeros(aog.n_nodes, dtype=bool)
    if aog.terminal is not None:
        live[aog.terminal] = True
    for L in range(aog.n_components - 1, -1, -1):
        a, b = aog.out_ptr[aog.level_offsets[L]], aog.out_ptr[aog.level_offsets[L + 1]]
        es = np.arange(a, b)
        es = es[edge_ok[a:b] & live[aog.edge_dst[a:b]]]
        live[aog.edge_src[es]] = True
    return edge_ok, live & aog.live


def blocked_key(blocked_edges) -> str:
    """[SUPP] Canonical text of a blocked-edge set, for cache keys."""
    return ";".join(f"{int(s)}:{int(c)}" for s, c in sorted({(int(s), int(c)) for s, c in blocked_edges}))


class CostModel:
    """All cost terms on one AoG; pure numpy (the geometry is in `tables`).

    Internal ("ACO") path cost: sum over enabled terms of weight * term, with M1/M6 divided by
    their DP range but without subtracting the DP minimum. The reported cost
    subtracts that constant offset, so every reported normalized term lies in [0, 1].

    Used by `run_aco` (immediate move costs `move_costs`, path costs `evaluate`) and by `dp_optimum`
    (exact optimum of the static terms, a reference).
    """

    def __init__(self, aog, tables: TermTables, config: OptimizationConfig,
                 blocked_edges: set[tuple[int, int]] | None = None):
        """[MAIN] Precompute everything the cost terms need from the AoG, the term tables and the config.

        This is where each term's normalisation is fixed: the min and max of the additive terms (M1 travel,
        M6 size) over all complete paths come from an exact DP over the live layers, and each term gets a
        per-unit weight (c_travel, c_size, c_dir, c_tool, c_group, c_stab, c_grav) so that its normalised
        value lies in [0, 1]. Also builds the live-edge CSR, the move numbering (one pheromone value per
        edge and free direction), the key ids of the tool/group terms and the gravity tables.
        Raises ValueError for an incomplete AoG or inconsistent tables, NoPathError in robot mode when
        the blocked edges disconnect the root from the terminal.
        """
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
        # Robot mode: forbidden (state, component) edges are dropped
        # and node liveness recomputed without them; None/empty leaves this exactly as before.
        live, edge_ok = aog.live, None
        if blocked_edges:
            edge_ok, live = blocked_liveness(aog, blocked_edges)
            if not live[aog.root]:
                raise NoPathError(f"The {len(blocked_edges)} blocked edge(s) disconnect the root from the terminal")
        self.live_edge = live[aog.edge_src] & live[aog.edge_dst]
        if edge_ok is not None:
            self.live_edge &= edge_ok
        self.lv_edges = np.flatnonzero(self.live_edge).astype(np.int64)
        counts = np.bincount(aog.edge_src[self.lv_edges], minlength=aog.n_nodes)
        self.lv_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        # Moves = (edge, free direction) pairs; move id = move_ptr[e] + rank of k among e's bits.
        self.move_ptr = np.concatenate(([0], np.cumsum(np.bitwise_count(self.edge_dirs)))).astype(np.int64)
        self.n_moves = int(self.move_ptr[-1])

        # Input checks: M3 needs a tool on every stander, M7 a group on every part.
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

        # Per-unit weight of each term: weight divided by the range its raw value can take.
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

        # Cost-to-go: exact min remaining static cost (M1 + M6) from each node.
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
        """[AUX] Per edge: min and max travel over its free directions, and the argmin direction."""
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
        """[AUX] best[v]: min (or max) additive cost from node v to the terminal over live edges.

        Backward dynamic programming, one AoG level at a time, with the per-edge cost `cost_e`.
        """
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
        """[AUX] DP value at the root: the min (or max) total of `cost_e` over complete paths."""
        return float(self._dp_table(cost_e, mode)[self.aog.root])

    def _static_edge_cost(self, tmin: np.ndarray) -> np.ndarray:
        """[AUX] Weighted static cost (M1 at the edge's best direction + M6) of every edge."""
        return (self.c_travel * tmin
                + self.c_size * self.size[self.edge_comp] * self.popcount[self.aog.edge_src] / self.N)

    def dp_optimum(self) -> tuple[np.ndarray, np.ndarray]:
        """[AUX] Exact optimum of the static terms (M1 + M6) by DP: (edge path, removal directions).

        The exact reference when only use_travel / use_size_contacts are on;
        with other terms enabled it is only the optimum of the static part. Used to check the ACO.
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
        """[AUX] Pheromone index of the moves (edge, direction): the edge's offset plus the rank of k among its free directions."""
        below = self.edge_dirs[edges] & ((U64(1) << np.asarray(ks).astype(U64)) - U64(1))
        return self.move_ptr[edges] + np.bitwise_count(below).astype(np.int64)

    def _edge_ks(self, e: int) -> np.ndarray:
        """[AUX] Indices of the free directions of edge e."""
        word = int(self.edge_dirs[e])
        return np.array([k for k in range(self.Dp) if (word >> k) & 1], dtype=np.int64)

    def _dir_levels(self, prev_k: np.ndarray, k: np.ndarray) -> np.ndarray:
        """[AUX] M2 level between consecutive directions (removal or insertion, same relations).

        Same direction 0, same axis opposite sign `dir_change_axis_ratio`, other axis 1.
        prev_k < 0 means "no previous step" and costs 0.
        """
        prev_safe = np.where(prev_k >= 0, prev_k, 0)
        lvl = np.where(self.neg[prev_safe] == k, self.config.dir_change_axis_ratio, 1.0)
        return np.where((prev_k < 0) | (prev_k == k), 0.0, lvl)

    def stability_classes(self, e: int) -> np.ndarray:
        """[AUX] class1[g] for inserting edge e's component into the state without it, per gravity candidate.

        Class 1 (True) = not supported or able to fall (M8 penalty). Cached by (state, component).
        """
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
        """[AUX] supported[g]: is the CoG of i, projected along g, inside the support region?

        The support region is built from the contact patches between i and the present components whose
        normal opposes g (the load-bearing ones): their convex hull ("hull") or their union ("union").
        """
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
        """[MAIN] Weighted immediate cost of moves (edge, direction) given the ant's memory.

        Inputs: arrays over the candidate moves of all ants: the AoG edge, the removal direction, the
        previous component and direction of the ant that owns the move, and its last tool. Output: the
        cost of each move, which feeds the heuristic eta = 1 / (eps + cost) of the ACO.

        Walk order is disassembly: the previous move is assembly step t + 1, so history terms (M2, M3, M7)
        are charged on the pair (t, t + 1); stability uses the min over g (no switching). With
        use_cost_to_go the exact remaining static cost from the move's destination is added (minus that
        of its source).
        """
        aog = self.aog
        comp = self.edge_comp[edges]
        src = aog.edge_src[edges]
        c = np.zeros(len(edges))
        # M1 travel and M6 size: static terms of the move itself.
        if self.c_travel:
            c += self.c_travel * self.travel[comp, ks]
        if self.c_size:
            c += self.c_size * self.size[comp] * self.popcount[src] / self.N
        # M2: direction change relative to the ant's previous move.
        if self.c_dir:
            c += self.c_dir * self._dir_levels(prev_dir, ks)
        # M3: tool change between consecutive standers.
        if self.c_tool:
            c += self.c_tool * (self.is_stander[comp] & (last_tool >= 0) & (self.tool_id[comp] != last_tool))
        if self.c_group:
            # Penalty for step t + 1 (prev): key changes while step t's group is unfinished after t.
            has = prev_comp >= 0
            pc = np.where(has, prev_comp, 0)
            unfinished = (self.group_mask[comp] & ~aog.node_mask[src]) != 0
            c += self.c_group * (has & (self.group_id[pc] != self.group_id[comp]) & unfinished)
        # M8: penalty if the part is class 1 (unsupported / able to fall) for every gravity candidate.
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
        """[AUX] Exact best gravity sequence for the class-1 table cls (N, G) in assembly order (Viterbi).

        Minimises the sum of the stability penalties plus the cost of switching gravity between steps.
        Returns (gravity index per step, total cost).
        """
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
        """[AUX] Validate that (edges, ks) is a complete root-to-terminal path with free directions (ValueError if not)."""
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
        """[MAIN] Exact cost of a complete path given in walk order (edges, removal directions).

        Inputs: the AoG edge ids and removal-direction indices of a root-to-terminal path (the order an ant
        walked it). Output: a dict with `internal_cost` (what the ACO compares), `total_cost` (reported,
        = internal - offset, in [0, 1]), the raw and normalised value of each term, the per-step
        weighted contributions (`steps`), the gravity chosen at each step by Viterbi and the stability
        class (1 or 2) of each insertion. Arrays are in assembly order (index s = step t - 1).
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

        # Each enabled term: raw value over the path, its normalisation to [0, 1], per-step weighted cost.
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
            # M8: the best gravity sequence by Viterbi, then the class-1 count and the gravity changes along it.
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
    """[AUX] Roulette-wheel draw, vectorised: one index per segment [starts[s], ends[s]) with probability proportional to exp(logw).

    Without `ends`, the segments are contiguous and cover logw (ends = next start). Used to pick the edge
    of every ant, and then its direction, in one numpy pass.
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
    """[MAIN] All ants walk root -> terminal in lockstep (every path has exactly N edges).

    Inputs: the cost model, the number of ants, the pheromone vector `tau` (one float32 per move), the
    random generator, the exponents alpha (pheromone) and beta (heuristic), and `greedy` (a single ant that
    always takes the cheapest move; used to seed the best path). Output: (edges (A, N), removal directions
    (A, N)) in walk (disassembly) order.

    At each step every ant lists its live out-edges, expands each into its free directions, computes the
    immediate cost of every move and chooses one with probability proportional to tau^alpha * (1/(eps+cost))^beta.
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
        # Expand each edge into its free directions and cost every resulting move.
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
        # Advance every ant along its chosen move and update its memory (previous component, direction, tool).
        e_pick, k_pick = e[pick], k[pick]
        comp = model.edge_comp[e_pick]
        path_e[:, step], path_k[:, step] = e_pick, k_pick
        node = aog.edge_dst[e_pick].astype(np.int64)
        prev_comp, prev_dir = comp, k_pick
        last_tool = np.where(model.is_stander[comp], model.tool_id[comp], last_tool)
    return path_e, path_k


def run_aco(model: CostModel) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """[MAIN] MAX-MIN Ant System over the AoG.

    Input: a `CostModel`. Output: (edges, removal directions, evaluation dict, history dict) of the best
    path found, in walk order.

    Starts from a greedy path (which also sets tau_max), then each iteration sends `n_ants` ants, evaluates
    their exact path costs, evaporates the pheromone, lets the iteration best (or, every
    `global_best_interval` iterations, the global best) deposit 1 / (eps + cost) on its moves, and clips the
    pheromone to [tau_min, tau_max]. Stops after `n_iterations` or `stagnation_limit` iterations without
    improvement.
    """
    cfg, N = model.config, model.N
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    n_ants = cfg.n_ants if cfg.n_ants is not None else max(2 * N, 10)
    ratio = cfg.tau_min_ratio if cfg.tau_min_ratio is not None else 1.0 / (2 * N)
    evals: dict[bytes, dict] = {}

    # Memoised path evaluation: the same path is often found many times.
    def evaluate(pe, pk):
        key = pe.tobytes() + pk.tobytes()
        if key not in evals:
            evals[key] = model.evaluate(pe, pk)
        return evals[key]

    # Initial best: the greedy path; its cost defines tau_max and the initial pheromone.
    ge, gk = _walk(model, 1, greedy=True)
    best_e, best_k = ge[0], gk[0]
    best = evaluate(best_e, best_k)
    greedy_cost = best["total_cost"]
    tau_max = 1.0 / (cfg.rho * (EPS + best["internal_cost"]))
    tau = np.full(model.n_moves, tau_max, dtype=np.float32)
    history_cost, stagnation, stop = [], 0, "iterations"
    it = 0
    for it in range(cfg.n_iterations):
        # One iteration: all ants walk, then keep the iteration best and update the global best.
        pe, pk = _walk(model, n_ants, tau, rng, cfg.alpha, cfg.beta)
        results = [evaluate(pe[a], pk[a]) for a in range(n_ants)]
        costs = np.array([r["internal_cost"] for r in results])
        ib = int(np.argmin(costs))
        if costs[ib] < best["internal_cost"] - 1e-12:
            best, best_e, best_k, stagnation = results[ib], pe[ib].copy(), pk[ib].copy(), 0
        else:
            stagnation += 1
        history_cost.append(best["total_cost"])

        # Pheromone update: evaporation, one deposit (iteration best, or global best every k-th
        # iteration), then clipping to the MAX-MIN bounds.
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
    """The optimised assembly sequence (output of `optimize` / `optimize_sequence`; input of the Illustration Generator)."""

    # list of dicts, assembly order. Per step: component, ids and labels, is_stander, AoG edge, insertion
    # direction (index and vector), travel (mm), start_offset (final position + offset = start point),
    # gravity and stability class (None if stability is off), per-term costs, and, added later,
    # turns/turns_pitch/turns_note/turns_nut_label for screws and nuts.
    steps: list[dict]
    total_cost: float  # float in [0, 1]. Sum of the weighted normalised terms.
    cost_breakdown: dict[str, dict]  # {term: {raw, normalized, weight, weighted}}, for the report.
    history: dict  # ACO history: best cost per iteration, greedy cost, iterations, stop reason, wall time...
    config_hash: str  # str. Hash of the AoG, tables, config and blocked edges; identifies the cached result.
    edge_path: list[int] = field(default_factory=list)  # AoG edge ids, disassembly (walk) order.
    removal_directions: list[int] = field(default_factory=list)  # removal direction index per edge, walk order.


def _build_result(model: CostModel, edges, ks, ev: dict, history: dict, config_hash: str) -> OptimizationResult:
    """[AUX] Turn the best path and its evaluation into an `OptimizationResult` (assembly order, insertion directions)."""
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
    """[AUX] Add `turns`, `turns_pitch`, `turns_note` and `turns_nut_label` to each step, in place.

    Display only: how many turns a screw, or the nut that
    drives it, must be rotated (travel / thread pitch). The ACO cost and the AoG are unaffected. Rows of
    A screw turns on its own step; a nut turns on its own step if its partner left first;
    otherwise the partner (if not a screw) turns, and `turns_nut_label` names the responsible nut for
    the output text ("thread of nut <label>").
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


def compute_config_hash(aog, tables: TermTables, config: OptimizationConfig, blocked_edges=None) -> str:
    """[SUPP] Cache key of the result: hash of the AoG key, table key, config (without `force`), weights and blocked edges."""
    cfg = asdict(config)
    cfg.pop("force")
    h = hashlib.sha256()
    h.update(f"aog:{aog.cache_key};tables:{tables.cache_key};".encode())
    h.update(json.dumps(cfg, sort_keys=True, default=list).encode())
    h.update(json.dumps(resolve_weights(config), sort_keys=True).encode())
    if blocked_edges:  # robot mode only; manual-mode keys are unchanged
        h.update(f"blocked:{blocked_key(blocked_edges)}".encode())
    return h.hexdigest()


def optimize(aog, tables: TermTables, config: OptimizationConfig | None = None,
             stability_cache: dict | None = None, blocked_edges=None) -> OptimizationResult:
    """[MAIN] Run the ACO on an AoG with ready term tables (pure numpy, no FreeCAD, no disk cache).

    Inputs: the AoG, the `TermTables`, the config, an optional pre-filled stability cache (so repeated runs
    skip the geometry-free support tests) and `blocked_edges`: optional set of (state mask, component) AoG
    edges the sequence must not use (robot mode). Output: an
    `OptimizationResult`; its history carries the stability cache back for the caller to save. Raises
    NoPathError when the blocked edges leave no path.
    """
    config = config or OptimizationConfig()
    model = CostModel(aog, tables, config, blocked_edges=blocked_edges)
    if stability_cache:
        model.stability_cache.update(stability_cache)
    edges, ks, ev, history = run_aco(model)
    result = _build_result(model, edges, ks, ev, history,
                           compute_config_hash(aog, tables, config, blocked_edges))
    result.history["stability_cache"] = model.stability_cache  # handed back for caching, not saved
    _log_summary(result)
    return result


def _log_summary(result: OptimizationResult) -> None:
    """[SUPP] Log the cost, the ACO run summary, the breakdown and one line per step."""
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
    """[SUPP] Format a vector with up to 3 decimals for logs and the text sequence."""
    return "(" + ", ".join(f"{round(float(c), 3) + 0.0:g}" for c in v) + ")"


def _fmt_turns_text(s: dict, decimals: int) -> str:
    """[SUPP] Text of the turns clause of one step, or "" if `turns` is None.

    `decimals` controls the turns count only (2 for assembly_sequence.txt, 1 for the manual); the pitch
    is shown as given, not padded.
    """
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
    """[AUX] Read the BOM: (column names, {label: row dict}, raw bytes for the cache key). Needs `label` and `is_stander` columns."""
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
    """[AUX] Global shapes in AoG order (shared helper in shapes.py)."""
    from shapes import component_shapes

    return component_shapes(aog.component_ids, aog.is_stander, prep)


def _center_of_mass(shape) -> list[float]:
    """[AUX] Volume-weighted centre of mass of a shape (over its solids), as [x, y, z]."""
    solids = shape.Solids
    if not solids:
        raise ValueError("Component shape has no solid; cannot compute its centre of gravity")
    vols = np.array([s.Volume for s in solids])
    cogs = np.array([[s.CenterOfMass.x, s.CenterOfMass.y, s.CenterOfMass.z] for s in solids])
    return (vols @ cogs / vols.sum()).tolist()


def _contact_matrix(aog, interference_result, prep) -> np.ndarray:
    """[AUX] (N, N) contact matrix in AoG order: the same contact graph as the AoG connectivity rule.

    Interference contacts between movable parts plus `StanderRecord.contact_neighbors`.
    """
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
    """[AUX] Faces of the boolean common of two faces larger than AREA_EPS ([] if OCC fails)."""
    try:
        return [f for f in fa.common(fb).Faces if f.Area > AREA_EPS]
    except Exception:  # noqa: BLE001  (OCC boolean failure: fall back to tessellation)
        return []


def _triangles(face, deflection: float) -> np.ndarray:
    """[AUX] Tessellate a face into triangles, (T, 3, 3) array of vertex coordinates."""
    pts, facets = face.tessellate(deflection)
    if not facets:
        return np.zeros((0, 3, 3))
    P = np.array([[p.x, p.y, p.z] for p in pts])
    return P[np.array(facets)]


def _region_triangles(fi, fj, threshold: float, deflection: float) -> np.ndarray:
    """[AUX] Triangles of the contact region of face fi (component i) against face fj (component j).

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
    """[MAIN] Contact patches between two shapes as triangles: (tri (T, 3, 3), normal (T, 3)).

    Inputs: the two global shapes, the contact `threshold` (mm, same as the interference contact
    threshold) and the tessellation deflection. Output: the triangles of every region where a face of i
    lies within `threshold` of a face of j, with normal = outward normal of j's surface at the contact
    = -(outward normal of i), i.e. it points from j into i. A patch carries i under gravity g when
    normal . g < 0; these patches are what the M8 stability term projects along g to test support.
    """
    import FreeCAD

    tris, normals = [], []
    for fi in shape_i.Faces:
        # Candidate face pairs: bounding boxes (enlarged by the threshold) intersect and the faces are close.
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
    """[SUPP] Write JSON to a temporary file, then rename it over the target (no half-written cache files)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path, key: str) -> dict | None:
    """[SUPP] Cache function: read a JSON cache file and return it only if its stored `key` matches."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001  (corrupt cache: recompute)
        log.warning("Could not read %s (%s); recomputing", path, exc)
        return None
    return data if data.get("key") == key else None


# File names of the cache and report files.
TABLES_NAME = "term_tables.json"  # volumes, areas, CoG, contact matrix
PATCHES_NAME = "contact_patches.json"  # contact triangles per touching pair
STABILITY_NAME = "stability_classes.json"  # cached stability classes per (state, component)
RESULT_NAME = "optimization_result.json"  # the result
SEQUENCE_NAME = "assembly_sequence.txt"  # the result as readable text


def _contact_threshold(interference_result) -> float:
    """[SUPP] Contact threshold (mm) of the interference result, or the module default if it has none."""
    thr = getattr(interference_result, "contact_threshold", None)
    if thr is None:
        from interference_identification import CONTACT_THRESHOLD
        thr = CONTACT_THRESHOLD
    return float(thr)


def tables_cache_key(aog, interference_result, bom_bytes: bytes, config: OptimizationConfig) -> str:
    """[SUPP] Key of the weight- and flag-independent term tables: AoG, interference, BOM, gravity, patch settings."""
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
    """[MAIN] Build the `TermTables` for the pipeline objects (the FreeCAD side of the optimisation).

    Inputs: the AoG, the InterferenceResult, the PreprocessingResult (stander records, BOM path,
    `auto_group_override`), the config and the cache folder. Output: a `TermTables` with volume, area,
    centre of gravity and contacts per component (geometry, cached as JSON), the BOM `tool`, `group` and
    `pitch` columns, and, when stability is on, the contact patches of every touching pair.

    A component with no BOM group (no row, or a blank cell) is its own group of one; an automatic
    sub-assembly takes `prep.auto_group_override`. This lets a fused or virtual container with no BOM row
    reach Optimization.
    """
    if prep.bom_path is None:
        raise ValueError("PreprocessingResult.bom_path is missing; the BOM is needed for M3/M7")
    columns, rows, bom_bytes = _read_bom(prep.bom_path)
    key = tables_cache_key(aog, interference_result, bom_bytes, config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shapes = None

    # Geometry tables: load from cache, or compute from the shapes and save.
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

    # BOM columns per component (matched on label): tool (M3), pitch (turns), group (M7).
    tool, group, pitch = [], [], []
    auto_group_override = getattr(prep, "auto_group_override", None) or {}
    for cid, label in zip(aog.component_ids, aog.component_labels):
        row = rows.get(label, {})
        tool.append(row.get("tool") or None)
        # Pitch is read for screw/nut rows only; blank,
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
            # (was previously a hard ValueError, H10/R6).
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
    """[AUX] Contact patches for every touching pair, checkpointed per pair (resumes after a crash).

    A pair where *both* components are fused sub-assemblies (`prep.compound_ids`) is skipped entirely:
    `extract_contact_patches` is O(faces_i * faces_j), and two
    large fused shapes merely grazing at one near-point (the contact test is a single whole-shape distance
    check, not an area check) can cost minutes for zero real contact area, as measured on
    `lunar_motor_tire_complete` (tire vs. motor: 895.87 s, 0 triangles). A pair with only *one* side a
    fused sub-assembly (a leaf resting on it) is NOT skipped: that contact can carry real support area the
    stability and gravity_change terms need. A skipped pair never gets a `tables.patches` entry;
    `TermTables.contact_triangles` already returns the empty fallback for it.
    """
    path = cache_dir / PATCHES_NAME
    data = None if force else _read_json(path, key)
    data = data or {"key": key, "pairs": {}}
    compound_ids = getattr(prep, "compound_ids", None) or frozenset()
    pairs = [(i, j) for i, j in zip(*np.nonzero(np.triu(tables.contact, 1)))]
    skipped = [(int(i), int(j)) for i, j in pairs
              if aog.component_ids[int(i)] in compound_ids and aog.component_ids[int(j)] in compound_ids]
    if skipped:
        log.info("Skipping contact-patch extraction for %d sub-assembly/sub-assembly pair(s): "
                 "%s", len(skipped),
                 [(aog.component_labels[i], aog.component_labels[j]) for i, j in skipped])
    skip_set = set(skipped)
    todo = [(int(i), int(j)) for i, j in pairs if (int(i), int(j)) not in skip_set and f"{i}|{j}" not in data["pairs"]]
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
    """[SUPP] Total area (mm^2) of a (T, 3, 3) triangle array (for the log)."""
    if not len(tri):
        return 0.0
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


# ---------------------------------------------------------------------------
# Persistence of the result
# ---------------------------------------------------------------------------


def save_result(result: OptimizationResult, cache_dir: Path) -> None:
    """[SUPP] Cache function: write the result as JSON and as the readable text `assembly_sequence.txt`."""
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
    """[SUPP] Cache function: load the saved result if its `config_hash` matches, else None."""
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
    """[SUPP] Key of the cached stability classes: table key plus the fall-test and support-region settings."""
    return hashlib.sha256(f"{tables.cache_key};fall:{config.stability_fall_test};"
                          f"region:{config.support_region}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def _default_cache_dir(prep) -> Path:
    """[SUPP] Default cache folder: output/<step file name>/optimization under the repo root."""
    stem = Path(prep.step_path).stem if prep.step_path is not None else prep.reduced_assembly.Name
    return REPO_ROOT / "output" / stem / "optimization"


def optimize_sequence(aog, interference_result, prep, config: OptimizationConfig | None = None,
                      cache_dir: str | Path | None = None, force: bool | None = None,
                      blocked_edges=None) -> OptimizationResult:
    """[MAIN] Pipeline entry point: the optimised assembly sequence for the pipeline objects, with on-disk caching.

    Inputs: the AoG, InterferenceResult and PreprocessingResult, the config, the cache folder (default
    output/<assembly>/optimization), `force` (recompute everything) and `blocked_edges` (robot mode): optional set of (state mask, component) edges to forbid. None or
    empty is the unchanged manual path. Otherwise the term tables and stability cache are still shared from
    `cache_dir`, but the result is cached in its own folder `cache_dir/blocked/<hash prefix>/`, so the
    manual result file is never overwritten.
    Output: an `OptimizationResult` (with the screw/nut turns added to its steps).

    Order of work: validate the config before any geometry, build the tables, return the cached result if
    its hash matches, otherwise run the ACO (reusing the saved stability classes), save and return.
    """
    config = config or OptimizationConfig()
    force = config.force if force is None else force
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir(prep)
    columns, _, _ = _read_bom(prep.bom_path) if prep.bom_path is not None else ([], {}, b"")
    validate_config(config, "tool" in columns, "group" in columns)  # fail before any geometry work

    tables = collect_tables(aog, interference_result, prep, config, cache_dir, force)
    config_hash = compute_config_hash(aog, tables, config, blocked_edges)
    result_dir = cache_dir / "blocked" / config_hash[:16] if blocked_edges else cache_dir
    if not force:
        cached = load_result(result_dir, config_hash)
        if cached is not None:
            log.info("Loaded cached optimization result from %s (total cost %.4f)", result_dir, cached.total_cost)
            return cached

    # Restore the stability classes computed by earlier runs (bits per gravity candidate).
    stab_path, stab_key = cache_dir / STABILITY_NAME, _stability_key(tables, config)
    stability = {}
    stab_data = None if (force or not config.use_stability) else _read_json(stab_path, stab_key)
    if stab_data:
        G = len(config.gravity_dirs)
        for name, bits in stab_data["classes"].items():
            S, i = (int(x) for x in name.split("|"))
            stability[(S, i)] = np.array([(bits >> g) & 1 for g in range(G)], dtype=bool)

    result = optimize(aog, tables, config, stability_cache=stability, blocked_edges=blocked_edges)
    new_cache = result.history.pop("stability_cache")
    if config.use_stability:
        _write_json_atomic(stab_path, {"key": stab_key, "classes": {
            f"{S}|{i}": int(sum(1 << g for g, b in enumerate(cls) if b)) for (S, i), cls in new_cache.items()}})
    _apply_turns(aog, prep, tables, result.steps)
    save_result(result, result_dir)
    return result
