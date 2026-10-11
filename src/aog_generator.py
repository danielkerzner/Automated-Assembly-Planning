"""AoG generator: the graph of every feasible single-component disassembly sequence.

Implements docs/aog-spec.md. Nodes are sub-assembly states (bitmask over all N components,
standers included), edges are removal moves carrying every free direction as a bitset. The
graph is generated iteratively, level by level (level L = L components removed), vectorised
over all states of a level; each state is expanded exactly once. No geometry calls: removal
freedom comes from InterferenceResult.free via the composition property, standers follow the
precedence (gate) rule, and travel distances are measured to the constant whole-assembly box.
States whose components do not form one group through physical contacts (parts floating apart)
are dropped, with the edges leading to them.

`generate_aog` is pure numpy on `AoGInputs`, so tests need no FreeCAD. `collect_inputs` and
`build_aog` connect it to the pipeline objects.
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

# Create a logger for this module
log = logging.getLogger("aap.aog")

# Define hyperparameters for maximum size of the AoG when Exporting
REPO_ROOT = Path(__file__).resolve().parent.parent
DIR_TOL = 1e-6  # direction equality (spec "Direction table")
AXIS_ZERO = 1e-12  # direction component treated as 0 in the travel model
EXPORT_MAX_NODES = 500 # Maximum number of nodes
MAX_COMPONENTS = 64 # Maximum number of components
MAX_DIRECTIONS = 64  # one uint64 word per edge (W = 1)

U64 = np.uint64


class AoGSizeError(RuntimeError):
    """Generation aborted because the graph exceeds AoGConfig.max_nodes (never a pruning)."""

# Dataclass for the AoG configuration
@dataclass
class AoGConfig:
    # Define if in the stander components, if the neighbouring comonents will also 
    # be dealt as prerequisitesor
    stander_gate: str = "prerequisites+contacts"  
    margin: float = 1.0  # mm, travel-model clearance (same default as CollisionConfig.margin)
    max_nodes: int = 4_000_000  # maximum size of the AoG when Computing

# Dataclass for the AoG inputs 
@dataclass
class AoGInputs:
    component_ids: list[str]  # movable then standers; obj.Label or stander_id
    component_labels: list[str] # 
    n_movable: int # number of movable components (the rest are standers)
    free: np.ndarray  # (n, n, D) bool from InterferenceResult
    directions: np.ndarray  # (D, 3) candidates + appended stander directions
    neg: np.ndarray  # (D,) int64, neg[k] = index of -directions[k] in directions
    stander_dir: dict[int, np.ndarray]  # bit index -> unit dismantle direction
    stander_gate: dict[int, set[int]]  # bit index -> movable bit indices (per config)
    bbox_lo: np.ndarray  # (N, 3) global AABBs of every component, mm (travel model)
    bbox_hi: np.ndarray  # (N, 3)
    assembly_lo: np.ndarray  # (3,) Preprocessing's whole-assembly box, all components
    assembly_hi: np.ndarray  # (3,)
    cache_key: str  # interference cache_key + stander data hash
    # (N, N) bool, symmetric: components physically touching (movable-movable from the interference
    # contact matrix, stander-movable from StanderRecord.contact_neighbors). States whose parts do
    # not form one connected group are dropped. None = no contact data, rule off (synthetic tests).
    contact: np.ndarray | None = None

# Dataclass for the AoG itself
@dataclass
class AoG:
    component_ids: list[str] # movable then standers; obj.Label or stander_id
    component_labels: list[str]
    is_stander: np.ndarray  # (N,) boolean array, True for standers
    directions: np.ndarray  # (D', 3) candidates + appended stander directions
    neg: np.ndarray  # neg[k] = index of -directions[k] in directions
    node_mask: np.ndarray  # (V,) uint64, sorted within each level; node id = index
    node_level: np.ndarray  # (V,) int8, components removed so far
    level_offsets: np.ndarray  # (N+2,) first node id of each level; [N+1] = V
    edge_src: np.ndarray  # (E,) int32, source node id
    edge_dst: np.ndarray  # (E,) int32, destination node id
    edge_comp: np.ndarray  # (E,) int8, the component removed (bit index)
    edge_dirs: np.ndarray  # (E, W) uint64, bitset over `directions`: all free directions
    out_ptr: np.ndarray  # (V+1,) CSR offsets (edges sorted by src, then comp)
    live: np.ndarray  # (V,) bool, node can reach the terminal
    dead_ends: np.ndarray  # node ids: non-terminal, no out-edge
    complete: bool  # root is live
    root: int # number of the root node (all components present)
    terminal: int | None # number of the terminal node (all components removed), or None if not present
    travel: np.ndarray  # (N, D') float, mm: travel[c, k], constant per component and direction
    config: AoGConfig
    stats: dict
    cache_key: str # Cache variables
    _in_order: np.ndarray | None = field(default=None, repr=False, compare=False)
    _in_ptr: np.ndarray | None = field(default=None, repr=False, compare=False)

    # Properties of the AoG
    @property
    def n_components(self) -> int:
        return len(self.component_ids)

    @property
    def n_nodes(self) -> int:
        return len(self.node_mask)

    @property
    def n_edges(self) -> int:
        return len(self.edge_src)

    # Queries for the AoG structure
    def out_edges(self, v: int) -> range:
        """Edge ids of all out-edges from node v, in the order of edge_src then edge_comp."""
        return range(int(self.out_ptr[v]), int(self.out_ptr[v + 1]))

    def in_edges(self, v: int) -> np.ndarray:
        """Edge ids of all in-edges to node v, in the order of edge_dst then edge_comp."""
        if self._in_order is None:
            self._in_order = np.argsort(self.edge_dst, kind="stable").astype(np.int64)
            counts = np.bincount(self.edge_dst, minlength=self.n_nodes)
            self._in_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        return self._in_order[self._in_ptr[v]:self._in_ptr[v + 1]]

    def edge_direction_indices(self, e: int) -> np.ndarray:
        """Indices of directions for the given edge."""
        return _bitset_indices(self.edge_dirs[e], len(self.directions))

    def state_components(self, v: int) -> list[int]:
        """Component ids of all components in the state of node v."""
        mask = int(self.node_mask[v])
        return [c for c in range(self.n_components) if (mask >> c) & 1]

    def edge_distances(self, e: int) -> np.ndarray:
        """Distances along the free directions of edge e, in the same order as edge_direction_indices(e)."""
        return self.travel[int(self.edge_comp[e]), self.edge_direction_indices(e)]

    def edge_displacements(self, e: int) -> np.ndarray:
        """Displacements along the free directions of edge e, in the same order as edge_direction_indices(e)."""
        ks = self.edge_direction_indices(e)
        return self.travel[int(self.edge_comp[e]), ks][:, None] * self.directions[ks]

    def hyperarcs(self, v: int) -> Iterator[tuple[int, frozenset[int]]]:
        """AND/OR view: (edge id, {child sub-assembly masks}) for every AND-arc out of v.

        Each arc splits the sub-assembly node_mask[v] into the remaining state and the trivial
        single-part child (the removed component), both given as component bitmasks.
        """
        for e in self.out_edges(v):
            c = int(self.edge_comp[e])
            yield e, frozenset({int(self.node_mask[self.edge_dst[e]]), 1 << c})

    # Queries for disassembly/assembly sequences.

    def _check_path(self, edge_path) -> list[int]:
        """Check that the given edge path is a valid connected path from the root, and return it as a list of ints."""
        path = [int(e) for e in edge_path]
        node = self.root
        for e in path:
            if int(self.edge_src[e]) != node:
                raise ValueError(f"Edge {e} does not start at node {node}; not a connected path")
            node = int(self.edge_dst[e])
        return path

    def _step(self, e: int, c: int, ks: np.ndarray) -> dict:
        """Return a dict describing the removal of component c along directions[ks] via edge e."""
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
        """Removal steps in disassembly order, each with all free directions of its edge."""
        steps = []
        for i, e in enumerate(self._check_path(edge_path)):
            step = self._step(e, int(self.edge_comp[e]), self.edge_direction_indices(e))
            step["step"] = i + 1
            steps.append(step)
        return steps

    def assembly_sequence(self, edge_path) -> list[dict]:
        """The path reversed as assembly steps; insertion directions are neg[removal direction].

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

    # Conversion to NetworkX for vizualisation (optional dependency).
    def to_networkx(self):
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


# Auxiliary functions for AoG generation and pipeline integration.


def _bitset_indices(words: np.ndarray, n_bits: int) -> np.ndarray:
    """Return the indices of the bits set in a bitset represented as an array of uint64 words."""
    ks = []
    for w, word in enumerate(np.atleast_1d(words)):
        word = int(word)
        ks.extend(64 * w + b for b in range(64) if (word >> b) & 1)
    return np.array([k for k in ks if k < n_bits], dtype=np.int64)


def _unit(v) -> np.ndarray:
    """Return a unit vector in the direction of v."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Zero-length direction")
    return v / norm


def extend_directions(
    directions: np.ndarray, neg: np.ndarray, stander_dir: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Candidate directions extended with the stander dismantle directions (spec "Direction table").

    A stander direction equal to an existing entry (|d - e| < 1e-6) reuses its index; otherwise d
    and -d are appended (-d only if absent) and `neg` extended. Returns (dirs', neg', {s: k}).
    """
    dirs = [np.asarray(d, dtype=float) for d in directions]
    neg_list = [int(x) for x in neg]
    kd: dict[int, int] = {}

    def find(d):
        for k, e in enumerate(dirs):
            if np.linalg.norm(d - e) < DIR_TOL:
                return k
        return None

    for s in sorted(stander_dir):
        d = _unit(stander_dir[s])
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
        kd[s] = k
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
    """travel[c, k]: distance c must move along directions[k] for its AABB to leave the fixed
    whole-assembly box B, plus `margin` (spec "Travel model"). Always >= margin."""
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
    """Return a bitmask with the given bit indices set."""
    m = 0
    for b in bits:
        m |= 1 << int(b)
    return m


def connected_states(masks: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """For each state mask, True if its components form one group through contacts.

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
    """Cache key for AoG generation, including the interference cache_key and the stander data."""
    h = hashlib.sha256()
    h.update(b"inputs:" + inputs.cache_key.encode())
    h.update(("ids:" + "|".join(inputs.component_ids)).encode())
    h.update(f"n:{inputs.n_movable};".encode())
    h.update(b"free:" + np.ascontiguousarray(inputs.free, dtype=bool).tobytes())
    dirs, neg, kd = extend_directions(inputs.directions, inputs.neg, inputs.stander_dir)
    h.update(b"dirs:" + np.round(dirs, 9).tobytes() + b"neg:" + neg.tobytes())
    for s in sorted(inputs.stander_dir):
        h.update(f"stander:{s}:{np.round(_unit(inputs.stander_dir[s]), 9).tolist()}:k{kd[s]}:"
                 f"gate{sorted(int(g) for g in inputs.stander_gate.get(s, ()))};".encode())
    for arr in (inputs.bbox_lo, inputs.bbox_hi, inputs.assembly_lo, inputs.assembly_hi):
        h.update(b"box:" + np.round(np.asarray(arr, dtype=float), 9).tobytes())
    if inputs.contact is None:
        h.update(b"contact:none")
    else:
        h.update(b"contact:" + np.ascontiguousarray(inputs.contact, dtype=bool).tobytes())
    h.update(("config:" + json.dumps(asdict(config), sort_keys=True)).encode())
    return h.hexdigest()


# Main AoG generation function, pure numpy, no geometry calls. Returns an AoG dataclass.
def generate_aog(inputs: AoGInputs, config: AoGConfig | None = None) -> AoG:
    """Build the AoG from `inputs` (pure numpy, no geometry calls)."""
    config = config or AoGConfig()
    t0 = time.perf_counter()

    # Check inputs to make sure it does not exceed the maximum number of components 
    # and that the shapes of the arrays are correct.
    N, n = len(inputs.component_ids), inputs.n_movable
    if N > MAX_COMPONENTS: # Component limit check
        raise ValueError(f"AoG supports at most {MAX_COMPONENTS} components, got {N}")
    if N == 0: # No components check
        raise ValueError("AoG needs at least one component")
    free = np.asarray(inputs.free, dtype=bool)
    D = len(inputs.directions)
    if free.shape != (n, n, D): # Check that the free array has the correct shape
        raise ValueError(f"free has shape {free.shape}, expected {(n, n, D)}")
    standers = list(range(n, N))
    if set(inputs.stander_dir) != set(standers): # Check that the stander_dir has one entry per stander bit
        raise ValueError(f"stander_dir must have one entry per stander bit {standers}")

    dirs, neg, kd = extend_directions(inputs.directions, inputs.neg, inputs.stander_dir)
    Dp = len(dirs)
    if Dp > MAX_DIRECTIONS: # Check that the number of directions does not exceed the maximum allowed
        raise ValueError(f"D' = {Dp} directions exceeds {MAX_DIRECTIONS} (one uint64 word per edge)")

    bm = np.zeros((n, D), dtype=U64)
    # Build a bitmask for each movable component and direction, indicating which other 
    # components block its movement in that direction.
    for i in range(n):
        for k in range(D):
            bm[i, k] = U64(_mask_of(j for j in range(n) if j != i and not free[i, j, k]))
    gate = {s: U64(_mask_of(inputs.stander_gate.get(s, ()))) for s in standers}
    for s in standers:
        if int(gate[s]) >> n: # Check that the gate of stander bit s does not contain non-movable bits
            raise ValueError(f"Gate of stander bit {s} contains non-movable bits")
        
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

    # Define a function to check if a state is valid
    def valid(T: np.ndarray) -> np.ndarray:
        ok = np.ones(len(T), dtype=bool)
        for s in standers:
            present = ((T >> U64(s)) & U64(1)).astype(bool)
            ok &= ~present | ((T & gate[s]) == gate[s])
        return ok

    # Variables for the iterative generation of the AoG
    full = (1 << N) - 1
    level = np.array([full], dtype=U64)
    node_chunks = [level]
    level_sizes = [1]
    edge_chunks: list[tuple[np.ndarray, ...]] = []
    offset = 0
    total = 1
    dropped_per_level = [0]

    # For each level of the AoG (number of compoents), generate the next level by removing one component 
    # at a time and checking if the resulting state is valid.
    for L in range(N):
        S = level
        srcs, comps, dirsets, dsts = [], [], [], []
        # For each component, check if it is present in the current state and 
        # if it can be removed (i.e., if the resulting state is valid). 
        # If so, add the resulting state to the next level.
        for c in range(N):
            bit = U64(1 << c)
            present = (S & bit) != 0
            if c < n:
                ds = np.zeros(len(S), dtype=U64)
                for k in range(D):
                    ds |= ((S & bm[c, k]) == 0).astype(U64) << U64(k)
                ok = present & (ds != 0)
            else:
                ds = np.full(len(S), U64(1) << U64(kd[c]), dtype=U64)
                ok = present & ((S & gate[c]) == gate[c])
            T = S & ~bit
            ok &= valid(T)
            idx = np.flatnonzero(ok)
            if len(idx):
                srcs.append(idx)
                comps.append(np.full(len(idx), c, dtype=np.int8))
                dirsets.append(ds[idx])
                dsts.append(T[idx])
        # If no valid next states were found, break the loop.
        if not srcs:
            break

        # Concatenate the results from all components and deduplicate the resulting states to form the next level.
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
        if not len(nxt):
            break
        total += len(nxt)

        # Check if the total number of nodes exceeds the maximum allowed, and raise an error if it does.
        if total > config.max_nodes:
            raise AoGSizeError(
                f"AoG exceeds max_nodes={config.max_nodes} at level {L + 1} ({total} nodes so far)")

        # Compute the offsets for the next level and store the edges and nodes for the current level.
        next_offset = offset + len(S)
        dst = np.searchsorted(nxt, dst_mask).astype(np.int64) + next_offset
        order = np.lexsort((comp, src_local))
        edge_chunks.append(((src_local[order] + offset).astype(np.int32), dst[order].astype(np.int32),
                            comp[order], dset[order]))
        node_chunks.append(nxt)
        level_sizes.append(len(nxt))
        offset = next_offset
        level = nxt
    
    # Concatenate the node chunks to form the final node mask and compute the level offsets and node levels.
    node_mask = np.concatenate(node_chunks).astype(U64)
    V = len(node_mask)
    sizes = level_sizes + [0] * (N + 1 - len(level_sizes))
    level_offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)  # (N+2,)
    node_level = np.repeat(np.arange(N + 1, dtype=np.int8), sizes)

    # Concatenate the edge chunks to form the final edge arrays, or create empty arrays if there are no edges.
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

    # Compute the live nodes by propagating the reachability from the terminal node backwards through the graph.
    live = np.zeros(V, dtype=bool)
    if terminal is not None:
        live[terminal] = True
    for L in range(N - 1, -1, -1):
        a, b = out_ptr[level_offsets[L]], out_ptr[level_offsets[L + 1]]
        if a < b:
            np.logical_or.at(live, edge_src[a:b], live[edge_dst[a:b]])
    has_out = out_ptr[1:] > out_ptr[:-1]
    dead_ends = np.flatnonzero((node_mask != 0) & ~has_out).astype(np.int64)

    # Compute the travel table for all components and directions, which gives the distance 
    # each component must move along each direction to leave the assembly bounding box, plus a margin.
    travel = compute_travel_table(inputs.bbox_lo, inputs.bbox_hi, inputs.assembly_lo,
                                  inputs.assembly_hi, dirs, config.margin)

    # Create the AoG dataclass with all the computed information and return it.
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
    )
    aog.stats = {
        "N": N, "n_movable": n, "n_standers": N - n, "D": D, "D_prime": Dp,
        "V": V, "E": len(edge_src), "nodes_per_level": sizes,
        "connectivity_rule": adj is not None, "disconnected_dropped": int(sum(dropped_per_level)),
        "dead_ends": len(dead_ends), "live_nodes": int(live.sum()), "complete": aog.complete,
        "time_s": time.perf_counter() - t0, "array_bytes": _array_bytes(aog),
    }
    _log_summary(aog)
    return aog


def _array_bytes(aog: AoG) -> int:
    """Return the total number of bytes used by the numpy arrays in the AoG dataclass."""
    return int(sum(getattr(aog, name).nbytes for name in _ARRAY_FIELDS))


# Functions for adapting to pipeline objects (PreprocessingResult, InterferenceResult) and FreeCAD.

def _vec3(v) -> list[float]:
    """Convert a 3D vector to a list of floats, handling FreeCAD.Vector if present."""
    return [float(v[0]), float(v[1]), float(v[2])] if not hasattr(v, "x") else [v.x, v.y, v.z]


def _stander_label(prep, stander_id: str) -> str:
    """obj.Label of a stander. The original document is not on PreprocessingResult, so look it up
    by name (run_preprocessing names the reduced doc '<doc>_reduced'); fall back to the ID."""
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


def collect_inputs(interference_result, prep, config: AoGConfig | None = None) -> AoGInputs:
    """Build AoGInputs from InterferenceResult + PreprocessingResult (bounding boxes only)."""
    config = config or AoGConfig()
    if config.stander_gate not in ("prerequisites+contacts", "prerequisites"):
        raise ValueError(f"Unknown stander_gate mode {config.stander_gate!r}")
    if prep.assembly_bbox is None:
        raise ValueError("PreprocessingResult.assembly_bbox is missing; re-run Preprocessing")

    movable = list(interference_result.component_ids)
    index = {cid: i for i, cid in enumerate(movable)}
    stander_ids = sorted(prep.stander_records)
    n = len(movable)
    ids = movable + stander_ids
    labels = list(interference_result.component_labels) + [_stander_label(prep, s) for s in stander_ids]

    bbox_lo, bbox_hi = [], []
    for cid in movable:
        obj = prep.reduced_assembly.getObject(cid)
        if obj is None:
            raise ValueError(f"Movable component {cid} not found in the reduced assembly")
        bb = obj.Shape.BoundBox  # the reduced doc stores shapes at their global placement
        bbox_lo.append([bb.XMin, bb.YMin, bb.ZMin])
        bbox_hi.append([bb.XMax, bb.YMax, bb.ZMax])

    if getattr(interference_result, "contact", None) is None:
        raise ValueError("InterferenceResult has no contact matrix; re-run identify_interferences")
    N = len(ids)
    contact = np.zeros((N, N), dtype=bool)
    contact[:n, :n] = np.asarray(interference_result.contact, dtype=bool)

    stander_dir, stander_gate = {}, {}
    h = hashlib.sha256(b"interference:" + interference_result.cache_key.encode())
    for b, sid in enumerate(stander_ids, start=n):
        rec = prep.stander_records[sid]
        if rec.bbox is None:
            raise ValueError(f"StanderRecord {sid} has no bbox; re-run Preprocessing")
        gate_ids = set(rec.prerequisites)
        if config.stander_gate == "prerequisites+contacts":
            gate_ids |= set(rec.contact_neighbors)
        unknown = gate_ids - set(index)
        if unknown:
            raise ValueError(f"Gate of stander {sid} names components not in the interference matrix: "
                             f"{sorted(unknown)}")
        stander_dir[b] = _unit(_vec3(rec.dismantle_direction))
        stander_gate[b] = {index[g] for g in gate_ids}
        # A stander touches its contact neighbours whatever the gate mode (connectivity rule).
        for g in rec.contact_neighbors:
            if g not in index:
                raise ValueError(f"Contact neighbour {g} of stander {sid} is not in the interference matrix")
            contact[b, index[g]] = contact[index[g], b] = True
        bbox_lo.append(_vec3(rec.bbox[0]))
        bbox_hi.append(_vec3(rec.bbox[1]))
        h.update(json.dumps({
            "id": sid,
            "prerequisites": sorted(rec.prerequisites),
            "contact_neighbors": sorted(rec.contact_neighbors),
            "direction": [round(x, 9) for x in _vec3(rec.dismantle_direction)],
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
        stander_dir=stander_dir,
        stander_gate=stander_gate,
        bbox_lo=np.array(bbox_lo, dtype=float).reshape(-1, 3),
        bbox_hi=np.array(bbox_hi, dtype=float).reshape(-1, 3),
        assembly_lo=assembly_lo,
        assembly_hi=assembly_hi,
        cache_key=h.hexdigest(),
        contact=contact,
    )


# ---------------------------------------------------------------------------
# Persistence and exports
# ---------------------------------------------------------------------------

NPZ_NAME = "aog.npz"
JSON_NAME = "aog.json"
_ARRAY_FIELDS = ("is_stander", "directions", "neg", "node_mask", "node_level", "level_offsets",
                 "edge_src", "edge_dst", "edge_comp", "edge_dirs", "out_ptr", "live", "dead_ends",
                 "travel")


def save_aog(aog: AoG, cache_dir: str | Path) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / NPZ_NAME, **{name: getattr(aog, name) for name in _ARRAY_FIELDS})
    meta = {
        "cache_key": aog.cache_key,
        "component_ids": aog.component_ids,
        "component_labels": aog.component_labels,
        "config": asdict(aog.config),
        "stats": aog.stats,
        "complete": aog.complete,
        "root": aog.root,
        "terminal": aog.terminal,
        "dead_ends": [int(v) for v in aog.dead_ends],
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    tmp = cache_dir / (JSON_NAME + ".tmp")
    tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(tmp, cache_dir / JSON_NAME)


def load_aog(cache_dir: str | Path, cache_key: str | None = None) -> AoG | None:
    """Load a cached AoG; None if absent, unreadable, or (when given) the key does not match."""
    cache_dir = Path(cache_dir)
    json_path, npz_path = cache_dir / JSON_NAME, cache_dir / NPZ_NAME
    if not (json_path.exists() and npz_path.exists()):
        return None
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
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
            **data,
        )
    except Exception as exc:  # noqa: BLE001  (corrupt cache: recompute rather than fail)
        log.warning("Could not load cached AoG (%s); recomputing", exc)
        return None


def _fmt_vec(v) -> str:
    return "(" + ", ".join(f"{round(float(c), 6) + 0.0:.6f}" for c in v) + ")"


def _state_label(aog: AoG, v: int) -> str:
    return "; ".join(aog.component_labels[c] for c in aog.state_components(v)) or "(empty)"


def export_csv_dot(aog: AoG, out_dir: str | Path) -> None:
    """aog_nodes.csv, aog_edges.csv and aog.dot for the thesis (small graphs only)."""
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
    s = aog.stats
    log.info("AoG: N=%d (n=%d movable, m=%d standers), D'=%d, V=%d, E=%d, complete=%s, %.2f s, %.1f MB",
             s["N"], s["n_movable"], s["n_standers"], s["D_prime"], s["V"], s["E"], aog.complete,
             s["time_s"], s["array_bytes"] / 1e6)
    log.info("AoG nodes per level: %s", s["nodes_per_level"])
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
        log.warning("AoG is incomplete: the root cannot reach the fully disassembled state")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _default_cache_dir(prep) -> Path:
    stem = Path(prep.step_path).stem if prep.step_path is not None else prep.reduced_assembly.Name
    return REPO_ROOT / "output" / stem / "aog"


def build_aog(
    interference_result,
    prep,
    config: AoGConfig | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> AoG:
    """Return the AoG for the pipeline objects, loading it from cache when the key matches."""
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
