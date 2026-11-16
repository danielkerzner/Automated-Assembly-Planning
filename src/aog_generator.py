"""AoG generator: the graph of every feasible single-component disassembly sequence.

Uses the three-class stander model (screw / nut / other). Phase 2 helpers:
classify_dead_ends, find_candidate_groups.

Inputs
    - `InterferenceResult` (Interference Identification): the tensor free[i, j, k] over the movable parts,
      the candidate directions, the contact graph.
    - `PreprocessingResult`: the stander records (removal options, holds, nut partners, contacts), the
      whole-assembly bounding box, the reduced assembly (for component bounding boxes) and the BOM path
      (for the `group` column). `collect_inputs` packs these into an `AoGInputs`.
    - An `AoGConfig` (margin and the two BOM-group switches).
Outputs
    - An `AoG`: nodes are sub-assembly states, edges are single-component removal moves carrying every free
      direction, plus the travel distance of each component along each direction, the live/dead-end
      analysis and statistics. Cached on disk (`aog.npz`, `aog.json`), with CSV and DOT exports for small
      graphs. Consumed by Optimization (ACO) and, through it, by the Illustration Generator.
    - Phase 2: `classify_dead_ends` and `find_candidate_groups` tell the per-level orchestration
      (`levels.py`) which dead ends can be repaired by merging parts into an automatic sub-assembly.

What this module does
    Nodes are states (a bitmask over all N components, standers included); an edge removes one component
    and stores the set of directions along which it can leave as a bitset. The graph is generated level by
    level (level L = L components removed), vectorised over all states of a level, and each state is
    expanded exactly once. There are no geometry calls: removal freedom comes from InterferenceResult.free
    through the composition property (a part is free along k in a state iff it is free past every part still
    present). A stander may have several removal options (direction, blocker set); a screw's holds must stay
    present while it leaves; a nut's `nut_rule` orders it against its `partner` (stander or movable). All of
    this is encoded as blocker masks, not as a single gate. Travel distances are measured to
    the constant whole-assembly box. States whose components do not form one group through physical
    contacts (parts floating apart) are dropped, with the edges leading to them. Members of one BOM group are
    removed consecutively, in one order per group block (D12, D13). There is no node limit; the only size
    limit is MAX_COMPONENTS (one uint64 bitmask per state).

`generate_aog` is pure numpy on `AoGInputs`, so tests need no FreeCAD. `collect_inputs` and `build_aog`
connect it to the pipeline objects.
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
from typing import Iterator

import numpy as np

log = logging.getLogger("aap.aog")

REPO_ROOT = Path(__file__).resolve().parent.parent
DIR_TOL = 1e-6  # tolerance for treating two directions as equal
AXIS_ZERO = 1e-12  # direction component treated as 0 in the travel model
EXPORT_MAX_NODES = 500  # CSV/DOT export is written only for graphs up to this many nodes
MAX_COMPONENTS = 64  # one uint64 bitmask per state
MAX_DIRECTIONS = 64  # one uint64 word per edge (W = 1)

U64 = np.uint64


@dataclass
class AoGConfig:
    """Settings of the AoG generation (config input of `generate_aog` and `build_aog`)."""

    # float, mm. Clearance added to every travel distance. Same default as CollisionConfig.margin.
    # Recommended: 1.0.
    margin: float = 1.0
    # bool. D12: members of one BOM group are removed (so also assembled) consecutively: while a group is
    # partly removed, only its remaining members may move. A blank group is a group of one.
    # Recommended: True.
    group_consecutive: bool = True
    # bool. D13: one order per group block: of the feasible moves removing members of the same group from a
    # state, only the lowest-index member is kept (removes the k! orderings). Needs group_consecutive.
    # Recommended: True.
    group_one_order: bool = True


@dataclass
class AoGInputs:
    """Everything `generate_aog` needs, as plain arrays (input). Built by `collect_inputs`, or by hand in tests.

    Component order: the n movable components first, then the standers. A component's position is its bit
    index in the state masks.
    """

    component_ids: list[str]  # list of obj.Name, movable then standers.
    component_labels: list[str]  # list of obj.Label (stander label looked up in the original document); display only.
    n_movable: int  # int. Number of movable components; the remaining N - n are standers.
    free: np.ndarray  # (n, n, D) bool, from InterferenceResult.free (movable parts only).
    directions: np.ndarray  # (D, 3) float, the interference candidate directions (stander directions are appended later).
    neg: np.ndarray  # (D,) int64, neg[k] = index of -directions[k] in directions.
    # Bit index -> one entry per RemovalOption: (unit direction,
    # set of blocker bit indices, which may include stander bits). 1 entry for a screw or nut, 2 for a
    # category-3 "other" stander. The blockers must already be gone before the stander can leave that way.
    stander_options: dict[int, list[tuple[np.ndarray, set[int]]]]
    stander_holds: dict[int, set[int]]  # bit index -> movable bit indices that must stay present (screws only).
    # Nut bit index -> (partner bit index, "locked" | "free"); the partner may be movable or a stander.
    nut_links: dict[int, tuple[int, str]]
    bbox_lo: np.ndarray  # (N, 3) float, mm. Global AABB minimum of every component (travel model).
    bbox_hi: np.ndarray  # (N, 3) float, mm. Global AABB maximum of every component.
    assembly_lo: np.ndarray  # (3,) float, mm. Whole-assembly box minimum, all components (from Preprocessing).
    assembly_hi: np.ndarray  # (3,) float, mm. Whole-assembly box maximum.
    cache_key: str  # str. Interference cache key + hash of the stander data; identifies these inputs.
    # (N, N) bool, symmetric: components physically touching (movable-movable from the interference contact
    # matrix, stander-movable from StanderRecord.contact_neighbors). States whose parts do not form one
    # connected group are dropped. None = no contact data, rule off (synthetic tests).
    contact: np.ndarray | None = None
    # (N,) list of the BOM `group` per component (None entry = blank or no BOM row: a group of one).
    # None = no group data.
    group: list[str | None] | None = None
    # Stander bit -> one entry per option, parallel to stander_options[s] (P5 screw-travel extension): a cap
    # on that option's reported travel distance (D10), or None to keep the bbox-only distance. Screws only;
    # empty = no overrides anywhere.
    stander_max_travel: dict[int, list[float | None]] = field(default_factory=dict)


@dataclass
class AoG:
    """The generated graph (output of `generate_aog`, input of Optimization and of the Phase 2 helpers).

    Nodes are numbered level by level (level L = L components removed), sorted by mask within a level;
    the root (all components present) is node 0 and the terminal (nothing left) the last node, if reached.
    Edges are stored in arrays sorted by source node, then by component (CSR layout via `out_ptr`).
    """

    component_ids: list[str]  # movable then standers (same order as AoGInputs).
    component_labels: list[str]  # display labels.
    is_stander: np.ndarray  # (N,) bool, True for standers.
    directions: np.ndarray  # (D', 3) float; interference candidates plus appended stander directions.
    neg: np.ndarray  # (D',) int; neg[k] = index of -directions[k] in directions.
    node_mask: np.ndarray  # (V,) uint64; bit c set = component c still present. Node id = index.
    node_level: np.ndarray  # (V,) int8; number of components removed so far.
    level_offsets: np.ndarray  # (N+2,) int64; first node id of each level, [N+1] = V.
    edge_src: np.ndarray  # (E,) int32; source node id of each edge.
    edge_dst: np.ndarray  # (E,) int32; destination node id.
    edge_comp: np.ndarray  # (E,) int8; the component removed by the edge (bit index).
    edge_dirs: np.ndarray  # (E, W) uint64; bitset over `directions`: every free direction of the edge.
    out_ptr: np.ndarray  # (V+1,) int64; CSR offsets, out-edges of v are out_ptr[v]..out_ptr[v+1].
    live: np.ndarray  # (V,) bool; the node can reach the terminal (a full disassembly exists from it).
    dead_ends: np.ndarray  # (k,) int64; node ids that are non-terminal and have no out-edge.
    complete: bool  # bool. The root is live: at least one full sequence exists.
    root: int  # int. Node id of the root (all components present), always 0.
    terminal: int | None  # int or None. Node id of the fully disassembled state, None if never reached.
    travel: np.ndarray  # (N, D') float, mm; travel[c, k] = distance c must move along direction k to clear the assembly.
    config: AoGConfig  # the settings used.
    stats: dict  # sizes, timings and rule counters (logged and saved to JSON).
    cache_key: str  # key identifying the inputs and config, for the cache.
    component_groups: list[str | None] | None = None  # BOM group per component (as in AoGInputs.group).
    _in_order: np.ndarray | None = field(default=None, repr=False, compare=False)  # lazy in-edge index (see in_edges)
    _in_ptr: np.ndarray | None = field(default=None, repr=False, compare=False)  # lazy in-edge offsets

    @property
    def n_components(self) -> int:
        """[AUX] Number of components N (movable plus standers)."""
        return len(self.component_ids)

    @property
    def n_nodes(self) -> int:
        """[AUX] Number of nodes V."""
        return len(self.node_mask)

    @property
    def n_edges(self) -> int:
        """[AUX] Number of edges E."""
        return len(self.edge_src)

    # Queries on the graph structure.
    def out_edges(self, v: int) -> range:
        """[AUX] Edge ids of all out-edges from node v, in the order of edge_src then edge_comp."""
        return range(int(self.out_ptr[v]), int(self.out_ptr[v + 1]))

    def in_edges(self, v: int) -> np.ndarray:
        """[AUX] Edge ids of all in-edges to node v. The in-edge index is built on first use and kept."""
        if self._in_order is None:
            self._in_order = np.argsort(self.edge_dst, kind="stable").astype(np.int64)
            counts = np.bincount(self.edge_dst, minlength=self.n_nodes)
            self._in_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        return self._in_order[self._in_ptr[v]:self._in_ptr[v + 1]]

    def edge_direction_indices(self, e: int) -> np.ndarray:
        """[AUX] Indices k into `directions` of every free direction of edge e (decoded from the bitset)."""
        return _bitset_indices(self.edge_dirs[e], len(self.directions))

    def state_components(self, v: int) -> list[int]:
        """[AUX] Indices of the components still present in the state of node v."""
        mask = int(self.node_mask[v])
        return [c for c in range(self.n_components) if (mask >> c) & 1]

    def edge_distances(self, e: int) -> np.ndarray:
        """[AUX] Travel distances along the free directions of edge e, in the order of edge_direction_indices(e)."""
        return self.travel[int(self.edge_comp[e]), self.edge_direction_indices(e)]

    def edge_displacements(self, e: int) -> np.ndarray:
        """[AUX] Displacement vectors (distance x direction) of edge e, in the order of edge_direction_indices(e)."""
        ks = self.edge_direction_indices(e)
        return self.travel[int(self.edge_comp[e]), ks][:, None] * self.directions[ks]

    def hyperarcs(self, v: int) -> Iterator[tuple[int, frozenset[int]]]:
        """[AUX] AND/OR view: (edge id, {child sub-assembly masks}) for every AND-arc out of v.

        Each arc splits the sub-assembly node_mask[v] into the remaining state and the trivial
        single-part child (the removed component), both given as component bitmasks.
        """
        for e in self.out_edges(v):
            c = int(self.edge_comp[e])
            yield e, frozenset({int(self.node_mask[self.edge_dst[e]]), 1 << c})

    # Queries that turn an edge path into disassembly / assembly sequences.

    def _check_path(self, edge_path) -> list[int]:
        """[AUX] Check that `edge_path` is a connected path starting at the root; return it as a list of ints."""
        path = [int(e) for e in edge_path]
        node = self.root
        for e in path:
            if int(self.edge_src[e]) != node:
                raise ValueError(f"Edge {e} does not start at node {node}; not a connected path")
            node = int(self.edge_dst[e])
        return path

    def _step(self, e: int, c: int, ks: np.ndarray) -> dict:
        """[AUX] Dict describing the removal of component c along directions[ks] via edge e (ids, directions, distances)."""
        return {
            "edge": e,
            "component": c,
            "component_id": self.component_ids[c],
            "component_label": self.component_labels[c],
            "is_stander": bool(self.is_stander[c]),
            "src": int(self.edge_src[e]),
            "dst": int(self.edge_dst[e]),
            "removal_direction_indices": [int(k) for k in ks],
            "removal_directions": self.directions[ks].tolist(),
            "distances": self.travel[c, ks].tolist(),
            "displacements": (self.travel[c, ks][:, None] * self.directions[ks]).tolist(),
        }

    def sequence(self, edge_path) -> list[dict]:
        """[AUX] Removal steps in disassembly order for an edge path, each with all free directions of its edge."""
        steps = []
        for i, e in enumerate(self._check_path(edge_path)):
            step = self._step(e, int(self.edge_comp[e]), self.edge_direction_indices(e))
            step["step"] = i + 1
            steps.append(step)
        return steps

    def assembly_sequence(self, edge_path) -> list[dict]:
        """[AUX] The path reversed as assembly steps; insertion directions are neg[removal direction].

        `start_offsets` = travel * removal direction: the component starts at final position +
        offset (outside the whole-assembly box) and moves along the insertion direction.
        """
        steps = []
        for i, e in enumerate(reversed(self._check_path(edge_path))):
            c = int(self.edge_comp[e])
            ks = self.edge_direction_indices(e)
            step = self._step(e, c, ks)
            step["step"] = i + 1
            step["direction_indices"] = [int(self.neg[k]) for k in ks]
            step["directions"] = self.directions[self.neg[ks]].tolist()
            step["start_offsets"] = step.pop("displacements")
            steps.append(step)
        return steps

    def to_networkx(self):
        """[SUPP] Convert the graph to a networkx DiGraph for visualisation (optional dependency)."""
        import networkx as nx  # optional dependency

        g = nx.DiGraph()
        for v in range(self.n_nodes):
            g.add_node(v, mask=int(self.node_mask[v]), level=int(self.node_level[v]),
                       live=bool(self.live[v]),
                       components=[self.component_labels[c] for c in self.state_components(v)])
        for e in range(self.n_edges):
            c = int(self.edge_comp[e])
            g.add_edge(int(self.edge_src[e]), int(self.edge_dst[e]), edge=e, component=c,
                       label=self.component_labels[c],
                       directions=[int(k) for k in self.edge_direction_indices(e)])
        return g


# ---------------------------------------------------------------------------------------------------
# Auxiliary functions for AoG generation
# ---------------------------------------------------------------------------------------------------


def _bitset_indices(words: np.ndarray, n_bits: int) -> np.ndarray:
    """[AUX] Indices of the bits set in a bitset stored as uint64 words (bits >= n_bits are ignored)."""
    ks = []
    for w, word in enumerate(np.atleast_1d(words)):
        word = int(word)
        ks.extend(64 * w + b for b in range(64) if (word >> b) & 1)
    return np.array([k for k in ks if k < n_bits], dtype=np.int64)


def _unit(v) -> np.ndarray:
    """[AUX] Unit vector in the direction of v (ValueError for a zero vector)."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Zero-length direction")
    return v / norm


def extend_directions(
    directions: np.ndarray, neg: np.ndarray,
    stander_options: dict[int, list[tuple[np.ndarray, set[int]]]],
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    """[AUX] Extend the candidate directions with every stander option's direction.

    A direction equal to an existing entry (|d - e| < DIR_TOL) reuses its index; otherwise d and -d are
    appended (-d only if absent) and `neg` is extended. Returns (dirs', neg', {s: [k_opt0, ...]}): one
    column index per option, in the order of `stander_options[s]` (a category-3 stander with two options
    gets two columns). This puts stander moves in the same direction table as the movable parts.
    """
    dirs = [np.asarray(d, dtype=float) for d in directions]
    neg_list = [int(x) for x in neg]
    kd: dict[int, list[int]] = {}

    def find(d):
        for k, e in enumerate(dirs):
            if np.linalg.norm(d - e) < DIR_TOL:
                return k
        return None

    for s in sorted(stander_options):
        cols = []
        for direction, _blockers in stander_options[s]:
            d = _unit(direction)
            k = find(d)
            if k is None:
                k = len(dirs)
                dirs.append(d)
                neg_list.append(-1)
                m = find(-d)
                if m is None:
                    m = len(dirs)
                    dirs.append(-d)
                    neg_list.append(k)
                neg_list[k] = m
                neg_list[m] = k
            cols.append(k)
        kd[s] = cols
    out = np.array(dirs, dtype=float).reshape(-1, 3)
    return out, np.array(neg_list, dtype=np.int64), kd


def compute_travel_table(
    bbox_lo: np.ndarray,
    bbox_hi: np.ndarray,
    assembly_lo: np.ndarray,
    assembly_hi: np.ndarray,
    directions: np.ndarray,
    margin: float,
) -> np.ndarray:
    """[AUX] travel[c, k]: distance component c must move along directions[k] for its AABB to leave the box.

    The box is the fixed whole-assembly box B; `margin` is added, so every value is
    >= margin. Vectorised slab method over all components and directions at once: for each axis the
    interval of travel during which the moving box overlaps B is found, and the exit is the end of the
    intersection of the three intervals.
    """
    lo = np.asarray(bbox_lo, dtype=float)[:, None, :]  # (N, 1, 3)
    hi = np.asarray(bbox_hi, dtype=float)[:, None, :]
    b_lo = np.asarray(assembly_lo, dtype=float)[None, None, :]
    b_hi = np.asarray(assembly_hi, dtype=float)[None, None, :]
    d = np.asarray(directions, dtype=float)[None, :, :]  # (1, D, 3)

    moving = np.abs(d) > AXIS_ZERO
    safe_d = np.where(moving, d, 1.0)
    t1 = (b_lo - hi) / safe_d
    t2 = (b_hi - lo) / safe_d
    overlap = (lo < b_hi) & (hi > b_lo)  # static overlap on that axis (touching = outside)
    t_start = np.where(moving, np.minimum(t1, t2), np.where(overlap, -np.inf, np.inf))
    t_end = np.where(moving, np.maximum(t1, t2), np.where(overlap, np.inf, -np.inf))
    start = t_start.max(axis=2)
    end = t_end.min(axis=2)
    nonempty = start < end
    return np.where(nonempty, np.maximum(0.0, end), 0.0) + margin


def _mask_of(bits) -> int:
    """[AUX] Python int bitmask with the given bit indices set."""
    m = 0
    for b in bits:
        m |= 1 << int(b)
    return m


def connected_states(masks: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """[AUX] For each state mask, True if its components form one group through contacts.

    Vectorised flood fill: start from the lowest present bit, repeatedly add the present contact
    neighbours of everything reached. `adj[c]` = bitmask of the components touching c. The empty
    state and single components count as connected.
    """
    masks = np.asarray(masks, dtype=U64)
    reach = masks & (~masks + U64(1))  # lowest set bit (0 for the empty state)
    for _ in range(len(adj)):
        grown = reach.copy()
        for c in range(len(adj)):
            grown |= ((reach >> U64(c)) & U64(1)) * adj[c]
        grown &= masks
        if np.array_equal(grown, reach):
            break
        reach = grown
    return reach == masks


def compute_cache_key(inputs: AoGInputs, config: AoGConfig) -> str:
    """[SUPP] Cache key of the AoG: SHA-256 over the inputs (free tensor, directions, stander options, holds,
    nut links, boxes, contacts, groups) and the config."""
    h = hashlib.sha256()
    h.update(b"inputs:" + inputs.cache_key.encode())
    h.update(("ids:" + "|".join(inputs.component_ids)).encode())
    h.update(f"n:{inputs.n_movable};".encode())
    h.update(b"free:" + np.ascontiguousarray(inputs.free, dtype=bool).tobytes())
    dirs, neg, kd = extend_directions(inputs.directions, inputs.neg, inputs.stander_options)
    h.update(b"dirs:" + np.round(dirs, 9).tobytes() + b"neg:" + neg.tobytes())
    for s in sorted(inputs.stander_options):
        max_travel = inputs.stander_max_travel.get(s, [])
        opts_repr = [
            {"k": kd[s][o], "dir": np.round(_unit(d), 9).tolist(), "blockers": sorted(int(b) for b in blockers),
             "max_travel": (None if o >= len(max_travel) or max_travel[o] is None
                           else round(float(max_travel[o]), 9))}
            for o, (d, blockers) in enumerate(inputs.stander_options[s])
        ]
        h.update(f"stander:{s}:{json.dumps(opts_repr, sort_keys=True)}:"
                 f"holds{sorted(int(x) for x in inputs.stander_holds.get(s, ()))};".encode())
    nut_repr = {str(k): [int(v[0]), v[1]] for k, v in sorted(inputs.nut_links.items())}
    h.update(("nut_links:" + json.dumps(nut_repr, sort_keys=True)).encode())
    for arr in (inputs.bbox_lo, inputs.bbox_hi, inputs.assembly_lo, inputs.assembly_hi):
        h.update(b"box:" + np.round(np.asarray(arr, dtype=float), 9).tobytes())
    if inputs.contact is None:
        h.update(b"contact:none")
    else:
        h.update(b"contact:" + np.ascontiguousarray(inputs.contact, dtype=bool).tobytes())
    h.update(("groups:" + json.dumps(inputs.group)).encode())
    h.update(("config:" + json.dumps(asdict(config), sort_keys=True)).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------
# Main function: AoG generation
# ---------------------------------------------------------------------------------------------------

def generate_aog(inputs: AoGInputs, config: AoGConfig | None = None) -> AoG:
    """[MAIN] Build the AoG from `inputs` (pure numpy, no geometry calls).

    Input: an `AoGInputs` (movable parts first, then standers) and an `AoGConfig`.
    Output: an `AoG` with all reachable states, the removal edges with their free directions, the travel
    table, the live nodes (those that can reach the fully disassembled state) and the dead ends.

    Method: a state is a bitmask of the components still present. For a movable part c in state S, the
    bit k of its direction set is 1 iff no part present in S blocks it along k, i.e. (S & blocker_mask[c, k])
    == 0; a stander has one blocker mask per removal option. Every move c -> S without c is kept only if the
    resulting state is valid (screw holds present, parts still connected, BOM group order respected). Each
    level of the graph is the set of unique resulting states of the previous level. Finally the live nodes
    are found by propagating reachability of the terminal backwards, and the dead ends are the non-terminal
    nodes without out-edge.
    """
    config = config or AoGConfig()
    t0 = time.perf_counter()

    # Input checks: size limits, array shapes and one options entry per stander.
    N, n = len(inputs.component_ids), inputs.n_movable
    if N > MAX_COMPONENTS:  # Component limit check
        raise ValueError(f"AoG supports at most {MAX_COMPONENTS} components, got {N}")
    if N == 0:  # No components check
        raise ValueError("AoG needs at least one component")
    free = np.asarray(inputs.free, dtype=bool)
    D = len(inputs.directions)
    if free.shape != (n, n, D):  # Check that the free array has the correct shape
        raise ValueError(f"free has shape {free.shape}, expected {(n, n, D)}")
    standers = list(range(n, N))
    if set(inputs.stander_options) != set(standers):  # one options entry per stander bit
        raise ValueError(f"stander_options must have one entry per stander bit {standers}")

    # Direction table: interference directions plus the directions of the stander options.
    dirs, neg, kd = extend_directions(inputs.directions, inputs.neg, inputs.stander_options)
    Dp = len(dirs)
    if Dp > MAX_DIRECTIONS:  # Check that the number of directions does not exceed the maximum allowed
        raise ValueError(f"D' = {Dp} directions exceeds {MAX_DIRECTIONS} (one uint64 word per edge)")

    # Blocker mask of each movable part and direction: the parts that block it along that direction.
    bm = np.zeros((n, D), dtype=U64)
    for i in range(n):
        for k in range(D):
            bm[i, k] = U64(_mask_of(j for j in range(n) if j != i and not free[i, j, k]))

    # Blocker masks replace the old single gate.
    # blockmask[s][o] = components that must be gone before stander s may leave via option o
    # (may include stander bits, category 3/P11). holdmask[s] = movable components that must stay
    # present while s leaves (screws only, P6).
    blockmask: dict[int, list] = {
        s: [U64(_mask_of(blockers)) for _direction, blockers in inputs.stander_options[s]]
        for s in standers
    }
    holdmask = {s: U64(_mask_of(inputs.stander_holds.get(s, ()))) for s in standers}
    for s in standers:
        if any(b >= n for b in inputs.stander_holds.get(s, ())):
            raise ValueError(f"Holds of stander bit {s} contain non-movable bits")

    # Nut/partner order (P8): applied once, as extra blocker bits -- "locked" blocks the nut on
    # its partner, "free" blocks the partner on the nut (on the partner's own blockmask if it's a
    # stander, or on `extra[partner]` -- applied to every one of its directions -- if movable).
    extra = np.zeros(n, dtype=U64)
    for nut, (partner, rule) in inputs.nut_links.items():
        if nut not in blockmask:
            raise ValueError(f"nut_links key {nut} is not a stander bit")
        if not (0 <= partner < N) or partner == nut:
            raise ValueError(f"nut_links partner {partner} for stander bit {nut} is invalid")
        if rule == "locked":
            partner_bit = U64(1) << U64(partner)
            blockmask[nut] = [m | partner_bit for m in blockmask[nut]]
        elif rule == "free":
            nut_bit = U64(1) << U64(nut)
            if partner < n:
                extra[partner] |= nut_bit
            else:
                blockmask[partner] = [m | nut_bit for m in blockmask[partner]]
        else:
            raise ValueError(f"Unknown nut_rule {rule!r} for stander bit {nut}")
    bm = bm | extra[:, None]  # fold the extra blockers into every direction of their movable owner

    # Contact adjacency as bitmasks: adj[c] = components touching c (None = connectivity rule off).
    adj = None
    if inputs.contact is not None:
        contact = np.asarray(inputs.contact, dtype=bool)
        if contact.shape != (N, N):
            raise ValueError(f"contact has shape {contact.shape}, expected {(N, N)}")
        contact = contact | contact.T
        np.fill_diagonal(contact, False)
        adj = np.array([_mask_of(np.flatnonzero(contact[c])) for c in range(N)], dtype=U64)
        if not connected_states(np.array([(1 << N) - 1], dtype=U64), adj)[0]:
            log.warning("The full assembly is not one connected group through contacts (check the contact "
                        "threshold); no state mixing the separate groups can be reached")

    # Groups (D12, D13): only groups with at least two members constrain anything.
    if config.group_one_order and not config.group_consecutive:
        raise ValueError("group_one_order needs group_consecutive (one order per group block)")
    use_groups = config.group_consecutive and inputs.group is not None
    comp_gid = np.full(N, -1, dtype=np.int64)  # group index per component, -1 = group of one
    gmasks: list = []  # bitmask of each group with >= 2 members
    group_names: list[str] = []
    if use_groups:
        if len(inputs.group) != N:
            raise ValueError(f"group has {len(inputs.group)} entries, expected {N}")
        members: dict[str, list[int]] = {}
        for c, g in enumerate(inputs.group):
            if g:
                members.setdefault(g, []).append(c)
        for g in sorted(members):
            if len(members[g]) >= 2:
                comp_gid[members[g]] = len(gmasks)
                gmasks.append(U64(_mask_of(members[g])))
                group_names.append(g)
        use_groups = bool(gmasks)
    one_order = use_groups and config.group_one_order

    # A state T is valid only if every stander still present in it has all of its holds present
    # too: a screw's hold cannot have legally left while the
    # screw is still there, on any path, so such a T is unreachable and must be filtered out
    # regardless of which edge led to it.
    def valid(T: np.ndarray) -> np.ndarray:
        ok = np.ones(len(T), dtype=bool)
        for s in standers:
            hm = holdmask[s]
            if hm == 0:
                continue
            present = ((T >> U64(s)) & U64(1)).astype(bool)
            ok &= ~present | ((T & hm) == hm)
        return ok

    # Iterative generation: `level` holds the unique states of the current level (starting with the root,
    # all components present); each loop builds the next level.
    full = (1 << N) - 1
    level = np.array([full], dtype=U64)
    node_chunks = [level]
    level_sizes = [1]
    edge_chunks: list[tuple[np.ndarray, ...]] = []
    offset = 0
    dropped_per_level = [0]
    order_pruned = 0

    # For each level (L components removed), expand every state once by removing one present component.
    for L in range(N):
        S = level
        srcs, comps, dirsets, dsts = [], [], [], []
        # D12: the group block of each state = the group that is partly removed (at most one,
        # because the rule below never lets a second group start while one is unfinished).
        if use_groups:
            block = np.zeros(len(S), dtype=U64)
            for g in gmasks:
                part = S & g
                block |= np.where((part != 0) & (part != g), g, U64(0))
        # Try to remove each component c from every state of the level, vectorised over the states.
        for c in range(N):
            bit = U64(1 << c)
            present = (S & bit) != 0
            if c < n:
                # Movable part: direction k is free in S iff no blocker of (c, k) is still present.
                ds = np.zeros(len(S), dtype=U64)
                for k in range(D):
                    ds |= ((S & bm[c, k]) == 0).astype(U64) << U64(k)  # bm already folds in extra[c]
                ok = present & (ds != 0)
            else:
                # Every option that is free (its blockers all gone from S) contributes its own
                # direction column -- a category-3 stander can have up to two at once (D5).
                ds = np.zeros(len(S), dtype=U64)
                for o, col in enumerate(kd[c]):
                    ds |= ((S & blockmask[c][o]) == 0).astype(U64) << U64(col)
                ok = present & (ds != 0)
                hm = holdmask[c]
                if hm:  # a screw may leave only while all its holds are still present (P6)
                    ok &= (S & hm) == hm
            T = S & ~bit
            ok &= valid(T)
            if use_groups:  # inside a group block only that group's members may move
                ok &= (block == 0) | ((block & bit) != 0)
            idx = np.flatnonzero(ok)
            if len(idx):
                srcs.append(idx)
                comps.append(np.full(len(idx), c, dtype=np.int8))
                dirsets.append(ds[idx])
                dsts.append(T[idx])
        # No valid move from any state: the graph ends here.
        if not srcs:
            break

        # Merge the moves of all components and deduplicate the resulting states: they form the next level.
        src_local = np.concatenate(srcs)
        comp = np.concatenate(comps)
        dset = np.concatenate(dirsets)
        dst_mask = np.concatenate(dsts)
        nxt = np.unique(dst_mask)  # node dedup: the "already analyzed" check

        # Drop states whose parts would float apart (not one group through contacts), together
        # with every edge leading to them. Checked once per unique state.
        dropped = 0
        if adj is not None:
            conn = connected_states(nxt, adj)
            dropped = int((~conn).sum())
            if dropped:
                keep = conn[np.searchsorted(nxt, dst_mask)]
                src_local, comp, dset, dst_mask = src_local[keep], comp[keep], dset[keep], dst_mask[keep]
                nxt = nxt[conn]
        dropped_per_level.append(dropped)

        # D13: of the feasible moves that remove members of the same group from the same state, keep
        # only the lowest-index member, so each group block is walked in one order. Runs after every
        # other filter, so the kept member is always one that can actually leave.
        if one_order:
            gid = comp_gid[comp.astype(np.int64)]
            sel = np.lexsort((comp, gid, src_local))
            sel = sel[gid[sel] >= 0]
            if len(sel):
                first = np.ones(len(sel), dtype=bool)
                first[1:] = (src_local[sel][1:] != src_local[sel][:-1]) | (gid[sel][1:] != gid[sel][:-1])
                drop = sel[~first]
                if len(drop):
                    keep = np.ones(len(comp), dtype=bool)
                    keep[drop] = False
                    src_local, comp, dset, dst_mask = src_local[keep], comp[keep], dset[keep], dst_mask[keep]
                    nxt = np.unique(dst_mask)
                    order_pruned += len(drop)
        if not len(nxt):
            break

        # Store this level's edges (with global node ids) and the new level's nodes.
        next_offset = offset + len(S)
        dst = np.searchsorted(nxt, dst_mask).astype(np.int64) + next_offset
        order = np.lexsort((comp, src_local))
        edge_chunks.append(((src_local[order] + offset).astype(np.int32), dst[order].astype(np.int32),
                            comp[order], dset[order]))
        node_chunks.append(nxt)
        level_sizes.append(len(nxt))
        offset = next_offset
        level = nxt

    # Assemble the node arrays: masks, level offsets and the level of every node.
    node_mask = np.concatenate(node_chunks).astype(U64)
    V = len(node_mask)
    sizes = level_sizes + [0] * (N + 1 - len(level_sizes))
    level_offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)  # (N+2,)
    node_level = np.repeat(np.arange(N + 1, dtype=np.int8), sizes)

    # Assemble the edge arrays (empty arrays if no edge exists) and the CSR offsets.
    if edge_chunks:
        edge_src = np.concatenate([ch[0] for ch in edge_chunks])
        edge_dst = np.concatenate([ch[1] for ch in edge_chunks])
        edge_comp = np.concatenate([ch[2] for ch in edge_chunks])
        edge_dirs = np.concatenate([ch[3] for ch in edge_chunks]).reshape(-1, 1)
    else:
        edge_src = np.zeros(0, dtype=np.int32)
        edge_dst = np.zeros(0, dtype=np.int32)
        edge_comp = np.zeros(0, dtype=np.int8)
        edge_dirs = np.zeros((0, 1), dtype=U64)
    out_ptr = np.concatenate(([0], np.cumsum(np.bincount(edge_src, minlength=V)))).astype(np.int64)

    root = 0
    terminal = V - 1 if int(node_mask[-1]) == 0 else None

    # Live nodes: propagate "can reach the terminal" backwards, level by level. Dead ends are non-terminal
    # nodes with no out-edge.
    live = np.zeros(V, dtype=bool)
    if terminal is not None:
        live[terminal] = True
    for L in range(N - 1, -1, -1):
        a, b = out_ptr[level_offsets[L]], out_ptr[level_offsets[L + 1]]
        if a < b:
            np.logical_or.at(live, edge_src[a:b], live[edge_dst[a:b]])
    has_out = out_ptr[1:] > out_ptr[:-1]
    dead_ends = np.flatnonzero((node_mask != 0) & ~has_out).astype(np.int64)

    # Travel table: the distance each component must move along each direction to leave the assembly box,
    # plus the margin.
    travel = compute_travel_table(inputs.bbox_lo, inputs.bbox_hi, inputs.assembly_lo,
                                  inputs.assembly_hi, dirs, config.margin)
    # P5 screw-travel extension: a stander option with a max_travel override is reported as whichever
    # is shorter, that cap or the bbox-based distance above (min(), not a replacement -- the override was
    # computed against Preprocessing's own bbox-exit formula, which can differ slightly from this one).
    for s, caps in inputs.stander_max_travel.items():
        for o, cap in enumerate(caps):
            if cap is not None:
                col = kd[s][o]
                travel[s, col] = min(travel[s, col], cap)

    # Pack everything into the AoG dataclass and record the statistics.
    aog = AoG(
        component_ids=list(inputs.component_ids),
        component_labels=list(inputs.component_labels),
        is_stander=np.arange(N) >= n,
        directions=dirs,
        neg=neg,
        node_mask=node_mask,
        node_level=node_level,
        level_offsets=level_offsets,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_comp=edge_comp,
        edge_dirs=edge_dirs,
        out_ptr=out_ptr,
        live=live,
        dead_ends=dead_ends,
        complete=bool(live[root]),
        root=root,
        terminal=terminal,
        travel=travel,
        config=config,
        stats={},
        cache_key=compute_cache_key(inputs, config),
        component_groups=None if inputs.group is None else list(inputs.group),
    )
    aog.stats = {
        "group_rule": use_groups, "group_one_order": one_order,
        "groups": {name: [inputs.component_labels[c] for c in np.flatnonzero(comp_gid == i)]
                   for i, name in enumerate(group_names)},
        "group_order_pruned_edges": int(order_pruned),
        "N": N, "n_movable": n, "n_standers": N - n, "D": D, "D_prime": Dp,
        "V": V, "E": len(edge_src), "nodes_per_level": sizes,
        "connectivity_rule": adj is not None, "disconnected_dropped": int(sum(dropped_per_level)),
        "dead_ends": len(dead_ends), "live_nodes": int(live.sum()), "complete": aog.complete,
        "time_s": time.perf_counter() - t0, "array_bytes": _array_bytes(aog),
    }
    _log_summary(aog)
    return aog


# ---------------------------------------------------------------------------
# Phase 2: dead-end classification and candidate groups.
# Pure numpy/Python, no geometry calls -- built on InterferenceResult's own query methods
# (removable, blockers), which already implement exactly "interf.removable(S's movable members)"
# and "i is blocked by j along k" from the usual wording.
# ---------------------------------------------------------------------------


def classify_dead_ends(aog: AoG, interf) -> dict[int, str]:
    """[MAIN] Classify each dead-end node of `aog` as "blocking" or "rule_induced".

    Input: the AoG and the `InterferenceResult` it was built from. Output: {node id: "blocking" |
    "rule_induced"}. Only blocking dead ends can be repaired by a Phase 2 merge, so `levels.py` uses this to
    decide whether merging is worth trying.

    "blocking": `interf.removable(movable members present)` is empty, i.e. no movable component present in
    the state has any free direction, judged purely on the raw interference tensor (no stander blockers,
    no connectivity rule, no BOM group ordering).
    "rule_induced": otherwise. Some present part had a geometrically available move (a movable part with a
    free direction, or any present stander), but every resulting state was filtered out by a stander
    blocker/hold rule, the connectivity rule or a D12/D13 group-ordering rule. A stander never makes a dead
    end "blocking" by itself: standers are not in the interference tensor, so a stander stuck on its
    blockers is rule-induced even if every movable part present is also fully blocked (a merge cannot free
    it). A state with no movable members at all (an all-stander dead end) is also "rule_induced": there is
    no movable group for Phase 2 to merge.
    """
    n = len(interf.component_ids)
    kinds: dict[int, str] = {}
    for v in aog.dead_ends:
        S = int(aog.node_mask[v])
        movable_present = [c for c in range(n) if (S >> c) & 1]
        blocking = bool(movable_present) and not interf.removable(movable_present)
        kinds[int(v)] = "blocking" if blocking else "rule_induced"
    return kinds


def find_candidate_groups(aog: AoG, interf, node_id: int) -> list[dict]:
    """[MAIN] Candidate rigid-body groups for one blocking dead end.

    Input: the AoG, the `InterferenceResult` and a dead-end node id. Output: one entry per unique
    (direction, group), as {"direction": k, "members": [indices]}, with indices into
    `interf.component_ids` (the same space as `aog.state_components`).

    Method: for direction k, closure_k(i) is everything reachable from movable component i through "i is
    blocked by j" edges (`interf.blockers(i, k, movable_present)`). By construction nothing outside a
    closure blocks anything inside it, so every closure with 2 <= |A| < (movable members present) is
    automatically a valid group that could leave together along k, with no separate check. The caller
    (`levels.py`) then picks the winner among these candidates.
    """
    n = len(interf.component_ids)
    D = len(interf.directions)
    S = int(aog.node_mask[node_id])
    movable_present = [c for c in range(n) if (S >> c) & 1]
    n_present = len(movable_present)
    seen: set[tuple[int, frozenset]] = set()
    out: list[dict] = []
    for k in range(D):
        for c in movable_present:
            # Grow the closure of c along k: add every part that blocks a member, until nothing new appears.
            A = {c}
            frontier = set(interf.blockers(c, k, movable_present))
            while frontier - A:
                new = frontier - A
                A |= new
                frontier = set()
                for j in new:
                    frontier |= set(interf.blockers(j, k, movable_present))
            if 2 <= len(A) < n_present and (k, frozenset(A)) not in seen:
                seen.add((k, frozenset(A)))
                out.append({"direction": k, "members": sorted(A)})
    return out


def _array_bytes(aog: AoG) -> int:
    """[SUPP] Total bytes of the AoG's numpy arrays (for the statistics)."""
    return int(sum(getattr(aog, name).nbytes for name in _ARRAY_FIELDS))


# ---------------------------------------------------------------------------------------------------
# Adapters from the pipeline objects (PreprocessingResult, InterferenceResult) and FreeCAD
# ---------------------------------------------------------------------------------------------------

def _vec3(v) -> list[float]:
    """[AUX] Convert a 3-vector (sequence or FreeCAD.Vector) to a list of three floats."""
    return [float(v[0]), float(v[1]), float(v[2])] if not hasattr(v, "x") else [v.x, v.y, v.z]


def _stander_label(prep, stander_id: str) -> str:
    """[AUX] obj.Label of a stander (display only; falls back to the id).

    The original document is not on PreprocessingResult, so it is looked up by name
    (run_preprocessing names the reduced doc '<doc>_reduced').
    """
    try:
        import FreeCAD

        name = prep.reduced_assembly.Name
        if name.endswith("_reduced"):
            obj = FreeCAD.getDocument(name[: -len("_reduced")]).getObject(stander_id)
            if obj is not None:
                return obj.Label
    except Exception:  # noqa: BLE001  (label is display only)
        pass
    return stander_id


def read_bom_groups(bom_path, labels: list[str]) -> list[str | None] | None:
    """[AUX] BOM `group` value for each label (rows matched on Label, as in Optimization).

    A blank group or a missing row gives None (a group of one). Returns None when there is no BOM file or
    it has no `group` column.
    """
    if bom_path is None or not Path(bom_path).exists():
        return None
    with open(bom_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = [(c or "").strip() for c in (reader.fieldnames or [])]
        if "group" not in columns:
            return None
        groups = {}
        for row in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if row.get("label"):
                groups[row["label"]] = row.get("group") or None
    return [groups.get(label) for label in labels]


def collect_inputs(interference_result, prep, config: AoGConfig | None = None) -> AoGInputs:
    """[MAIN] Build an `AoGInputs` from an InterferenceResult and a PreprocessingResult.

    This is where the stander model meets the graph: stander names are
    mapped to bit indices over movable-then-stander space (a blocker may itself be a stander, P5/P11);
    `options`/`blockers` come from `rec.options`, `holds` from `rec.holds` (movable only, P6), `nut_links`
    from `rec.partner`/`rec.nut_rule`; the contact matrix is extended with the stander contacts
    (`rec.contact_neighbors`, P10); component boxes come from the reduced assembly and the stander
    records. A hash of all stander data is folded into the cache key. Raises ValueError if the inputs are
    inconsistent (unknown blockers, missing bbox or contact matrix).
    """
    config = config or AoGConfig()
    if prep.assembly_bbox is None:
        raise ValueError("PreprocessingResult.assembly_bbox is missing; re-run Preprocessing")

    # Component order and labels: movable parts (interference order), then standers (sorted by id).
    movable = list(interference_result.component_ids)
    index = {cid: i for i, cid in enumerate(movable)}
    stander_ids = sorted(prep.stander_records)
    n = len(movable)
    ids = movable + stander_ids
    index_all = {cid: i for i, cid in enumerate(ids)}
    labels = list(interference_result.component_labels) + [_stander_label(prep, s) for s in stander_ids]

    # Bounding boxes of the movable parts (the reduced document stores global shapes).
    bbox_lo, bbox_hi = [], []
    for cid in movable:
        obj = prep.reduced_assembly.getObject(cid)
        if obj is None:
            raise ValueError(f"Movable component {cid} not found in the reduced assembly")
        bb = obj.Shape.BoundBox  # the reduced doc stores shapes at their global placement
        bbox_lo.append([bb.XMin, bb.YMin, bb.ZMin])
        bbox_hi.append([bb.XMax, bb.YMax, bb.ZMax])

    # Contact matrix: movable-movable block from the interference result; stander rows are added below.
    if getattr(interference_result, "contact", None) is None:
        raise ValueError("InterferenceResult has no contact matrix; re-run identify_interferences")
    N = len(ids)
    contact = np.zeros((N, N), dtype=bool)
    contact[:n, :n] = np.asarray(interference_result.contact, dtype=bool)

    stander_options: dict[int, list[tuple[np.ndarray, set[int]]]] = {}
    stander_holds: dict[int, set[int]] = {}
    nut_links: dict[int, tuple[int, str]] = {}
    stander_max_travel: dict[int, list[float | None]] = {}
    h = hashlib.sha256(b"interference:" + interference_result.cache_key.encode())
    for b, sid in enumerate(stander_ids, start=n):
        rec = prep.stander_records[sid]
        if rec.bbox is None:
            raise ValueError(f"StanderRecord {sid} has no bbox; re-run Preprocessing")

        # Removal options: direction and blocker bits (names mapped to indices).
        options = []
        for opt in rec.options:
            unknown = set(opt.blockers) - set(index_all)
            if unknown:
                raise ValueError(f"Blockers of stander {sid} name components not in this level: "
                                 f"{sorted(unknown)}")
            options.append((_unit(_vec3(opt.direction)), {index_all[m] for m in opt.blockers}))
        stander_options[b] = options
        stander_max_travel[b] = [getattr(opt, "max_travel", None) for opt in rec.options]

        # Holds (movable only).
        unknown_holds = set(rec.holds) - set(index)
        if unknown_holds:
            raise ValueError(f"Holds of stander {sid} name components not in the interference "
                             f"matrix (holds are movable-only, P6): {sorted(unknown_holds)}")
        stander_holds[b] = {index[m] for m in rec.holds}

        # Nut link: the nut orders itself against its partner (needs a partner present in this level).
        if rec.nut_rule is not None:
            if rec.partner is None or rec.partner not in index_all:
                log.warning("Nut stander %r has nut_rule=%r but no usable partner in this level "
                           "(partner=%r); treating it as a plain stander with no nut order",
                           sid, rec.nut_rule, rec.partner)
            else:
                nut_links[b] = (index_all[rec.partner], rec.nut_rule)

        # A stander touches its contact neighbours regardless (connectivity rule, P10).
        for g in rec.contact_neighbors:
            if g not in index:
                raise ValueError(f"Contact neighbour {g} of stander {sid} is not in the interference matrix")
            contact[b, index[g]] = contact[index[g], b] = True
        bbox_lo.append(_vec3(rec.bbox[0]))
        bbox_hi.append(_vec3(rec.bbox[1]))
        # Fold the whole record into the cache key so any change in stander data invalidates the cache.
        h.update(json.dumps({
            "id": sid,
            "category": rec.category,
            "options": [
                {"direction": [round(x, 9) for x in _vec3(opt.direction)], "blockers": sorted(opt.blockers),
                 "max_travel": (None if getattr(opt, "max_travel", None) is None
                               else round(opt.max_travel, 9))}
                for opt in rec.options
            ],
            "holds": sorted(rec.holds),
            "contact_neighbors": sorted(rec.contact_neighbors),
            "partner": rec.partner,
            "nut_rule": rec.nut_rule,
            "bbox": [[round(x, 9) for x in _vec3(p)] for p in rec.bbox],
        }, sort_keys=True).encode())
    assembly_lo, assembly_hi = (np.array(_vec3(p), dtype=float) for p in prep.assembly_bbox)
    h.update(b"assembly_bbox:" + np.round(np.concatenate([assembly_lo, assembly_hi]), 9).tobytes())

    return AoGInputs(
        component_ids=ids,
        component_labels=labels,
        n_movable=n,
        free=np.asarray(interference_result.free, dtype=bool),
        directions=np.asarray(interference_result.directions, dtype=float),
        neg=np.asarray(interference_result.neg, dtype=np.int64),
        stander_options=stander_options,
        stander_holds=stander_holds,
        nut_links=nut_links,
        bbox_lo=np.array(bbox_lo, dtype=float).reshape(-1, 3),
        bbox_hi=np.array(bbox_hi, dtype=float).reshape(-1, 3),
        assembly_lo=assembly_lo,
        assembly_hi=assembly_hi,
        cache_key=h.hexdigest(),
        contact=contact,
        group=read_bom_groups(getattr(prep, "bom_path", None), labels),
        stander_max_travel=stander_max_travel,
    )


# ---------------------------------------------------------------------------
# Persistence and exports (supplementary)
# ---------------------------------------------------------------------------

NPZ_NAME = "aog.npz"  # numpy arrays of the AoG
JSON_NAME = "aog.json"  # metadata, statistics and the cache key
_ARRAY_FIELDS = ("is_stander", "directions", "neg", "node_mask", "node_level", "level_offsets",
                 "edge_src", "edge_dst", "edge_comp", "edge_dirs", "out_ptr", "live", "dead_ends",
                 "travel")  # AoG fields saved in the .npz
# Bump whenever a cached AoG's on-disk meaning could change
# without its cache_key necessarily changing (e.g. a loader/field rename) -- rejects old caches
# outright rather than risking a silent misread. 2 = the stander_options/holds/nut_links model
# (was 1, implicitly, under stander_dir/stander_gate).
AOG_SCHEMA_VERSION = 2


def save_aog(aog: AoG, cache_dir: str | Path) -> None:
    """[SUPP] Cache function: write the AoG arrays (.npz) and its metadata (.json) to `cache_dir`."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / NPZ_NAME, **{name: getattr(aog, name) for name in _ARRAY_FIELDS})
    meta = {
        "schema_version": AOG_SCHEMA_VERSION,
        "cache_key": aog.cache_key,
        "component_ids": aog.component_ids,
        "component_labels": aog.component_labels,
        "config": asdict(aog.config),
        "stats": aog.stats,
        "complete": aog.complete,
        "root": aog.root,
        "terminal": aog.terminal,
        "dead_ends": [int(v) for v in aog.dead_ends],
        "component_groups": aog.component_groups,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    tmp = cache_dir / (JSON_NAME + ".tmp")
    tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(tmp, cache_dir / JSON_NAME)


def load_aog(cache_dir: str | Path, cache_key: str | None = None) -> AoG | None:
    """[SUPP] Cache function: load a cached AoG.

    Returns None if it is absent, unreadable, of an old schema or (when `cache_key` is given) the key
    does not match.
    """
    cache_dir = Path(cache_dir)
    json_path, npz_path = cache_dir / JSON_NAME, cache_dir / NPZ_NAME
    if not (json_path.exists() and npz_path.exists()):
        return None
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        if meta.get("schema_version") != AOG_SCHEMA_VERSION:
            log.info("Cached AoG is an old on-disk schema (%r != %r); recomputing",
                     meta.get("schema_version"), AOG_SCHEMA_VERSION)
            return None
        if cache_key is not None and meta.get("cache_key") != cache_key:
            log.info("Cached AoG is stale (key mismatch); recomputing")
            return None
        with np.load(npz_path) as arrays:
            data = {name: arrays[name] for name in _ARRAY_FIELDS}
        return AoG(
            component_ids=meta["component_ids"],
            component_labels=meta["component_labels"],
            complete=meta["complete"],
            root=meta["root"],
            terminal=meta["terminal"],
            config=AoGConfig(**meta["config"]),
            stats=meta["stats"],
            cache_key=meta["cache_key"],
            component_groups=meta.get("component_groups"),
            **data,
        )
    except Exception as exc:  # noqa: BLE001  (corrupt cache: recompute rather than fail)
        log.warning("Could not load cached AoG (%s); recomputing", exc)
        return None


def _fmt_vec(v) -> str:
    """[SUPP] Format a vector with 6 decimals for the exports (-0.0 prints as 0.0)."""
    return "(" + ", ".join(f"{round(float(c), 6) + 0.0:.6f}" for c in v) + ")"


def _state_label(aog: AoG, v: int) -> str:
    """[SUPP] Readable label of a node: the labels of the components present, joined by ';'."""
    return "; ".join(aog.component_labels[c] for c in aog.state_components(v)) or "(empty)"


def export_csv_dot(aog: AoG, out_dir: str | Path) -> None:
    """[SUPP] Export aog_nodes.csv, aog_edges.csv and aog.dot (Graphviz) for the thesis (small graphs only)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dead = set(int(v) for v in aog.dead_ends)
    with open(out_dir / "aog_nodes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["node", "level", "mask", "live", "dead_end", "components"])
        for v in range(aog.n_nodes):
            w.writerow([v, int(aog.node_level[v]), int(aog.node_mask[v]), bool(aog.live[v]),
                        v in dead, _state_label(aog, v)])
    with open(out_dir / "aog_edges.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["edge", "src", "dst", "component", "is_stander", "direction_indices",
                    "directions", "distances_mm"])
        for e in range(aog.n_edges):
            c = int(aog.edge_comp[e])
            ks = aog.edge_direction_indices(e)
            w.writerow([e, int(aog.edge_src[e]), int(aog.edge_dst[e]), aog.component_labels[c],
                        bool(aog.is_stander[c]), " ".join(str(int(k)) for k in ks),
                        "; ".join(_fmt_vec(aog.directions[k]) for k in ks),
                        " ".join(f"{aog.travel[c, k]:.3f}" for k in ks)])
    lines = ["digraph AoG {", "  rankdir=TB;", '  node [shape=box, fontsize=9];']
    for L in range(aog.n_components + 1):
        a, b = int(aog.level_offsets[L]), int(aog.level_offsets[L + 1])
        if a == b:
            continue
        lines.append("  { rank=same; " + " ".join(f"n{v};" for v in range(a, b)) + " }")
        for v in range(a, b):
            style = "" if aog.live[v] else ', style=dashed, color=red'
            label = _state_label(aog, v).replace('"', "'").replace("; ", "\\n")
            lines.append(f'  n{v} [label="{v}\\n{label}"{style}];')
    for e in range(aog.n_edges):
        c = int(aog.edge_comp[e])
        ks = " ".join(str(int(k)) for k in aog.edge_direction_indices(e))
        lines.append(f'  n{int(aog.edge_src[e])} -> n{int(aog.edge_dst[e])} '
                     f'[label="{aog.component_labels[c]}\\n[{ks}]", fontsize=8];')
    lines.append("}")
    (out_dir / "aog.dot").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_summary(aog: AoG) -> None:
    """[SUPP] Log the AoG size, active rules, dead ends and a warning if the graph is incomplete."""
    s = aog.stats
    log.info("AoG: N=%d (n=%d movable, m=%d standers), D'=%d, V=%d, E=%d, complete=%s, %.2f s, %.1f MB",
             s["N"], s["n_movable"], s["n_standers"], s["D_prime"], s["V"], s["E"], aog.complete,
             s["time_s"], s["array_bytes"] / 1e6)
    log.info("AoG nodes per level: %s", s["nodes_per_level"])
    if s.get("group_rule"):
        log.info("AoG group rule on: groups %s assembled consecutively; one order per block %s "
                 "(%d edges removed)", s["groups"], s["group_one_order"], s["group_order_pruned_edges"])
    if s.get("connectivity_rule"):
        log.info("AoG dropped %d disconnected state(s) (parts floating apart)", s["disconnected_dropped"])
    else:
        log.info("AoG connectivity rule off (no contact data in the inputs)")
    if len(aog.dead_ends):
        shown = aog.dead_ends[:20]
        log.warning("AoG has %d dead-end state(s) (non-linear moves needed, out of scope): %s%s",
                    len(aog.dead_ends), "; ".join(f"{int(v)}: [{_state_label(aog, int(v))}]" for v in shown),
                    " ..." if len(aog.dead_ends) > len(shown) else "")
    if not aog.complete:
        if s.get("group_rule"):
            log.warning("AoG is incomplete: the root cannot reach the fully disassembled state. The BOM group "
                        "rule is on (D12/D13, groups %s): a group that cannot be assembled consecutively, or "
                        "whose first feasible member leads into a dead end, also causes this", s.get("groups"))
        else:
            log.warning("AoG is incomplete: the root cannot reach the fully disassembled state")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _default_cache_dir(prep) -> Path:
    """[SUPP] Default cache folder: output/<step file name>/aog under the repo root."""
    stem = Path(prep.step_path).stem if prep.step_path is not None else prep.reduced_assembly.Name
    return REPO_ROOT / "output" / stem / "aog"


def build_aog(
    interference_result,
    prep,
    config: AoGConfig | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> AoG:
    """[MAIN] Pipeline entry point: the AoG for an InterferenceResult and PreprocessingResult, cached on disk.

    Packs the pipeline objects with `collect_inputs`, returns the cached AoG if its key matches (unless
    `force`), otherwise runs `generate_aog`, saves it and, for graphs of at most EXPORT_MAX_NODES nodes,
    writes the CSV/DOT exports next to the cache. Default cache folder: output/<assembly>/aog.
    """
    config = config or AoGConfig()
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir(prep)
    inputs = collect_inputs(interference_result, prep, config)
    key = compute_cache_key(inputs, config)
    if not force:
        cached = load_aog(cache_dir, key)
        if cached is not None:
            log.info("Loaded cached AoG from %s (V=%d, E=%d)", cache_dir, cached.n_nodes, cached.n_edges)
            return cached
    aog = generate_aog(inputs, config)
    save_aog(aog, cache_dir)
    if aog.n_nodes <= EXPORT_MAX_NODES:
        export_csv_dot(aog, cache_dir)
    return aog
