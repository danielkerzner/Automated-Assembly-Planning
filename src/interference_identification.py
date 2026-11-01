"""Interference Identification: the pairwise interference matrix.

For every ordered pair of reduced-assembly components
(i, j) and every candidate direction d_k, decides whether i can translate along d_k all the way
past j without penetrating it, using the verified collision test in collision.py. The result is
the boolean tensor free[i, j, k], computed once per assembly (cached on disk) and queried by
the AoG generator without further geometry calls.

Two properties make the pairwise matrix sufficient and halve its cost:
  - Composition: i can leave sub-assembly S along d_k iff free[i, j, k] for every j in S.
  - Antisymmetry: free[j, i, neg[k]] == free[i, j, k], so only the i < j triangle is computed.

It also records which pairs physically touch (contact[i, j]: static gap <= contact_threshold,
default 0.1 mm so clearance fits count as contact). The AoG generator uses this contact graph to
drop sub-assemblies whose parts would float apart.
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
from typing import Callable, Iterable

import FreeCAD
import numpy as np

from collision import CollisionConfig, HardOverlapError, Stats, classify_pair, is_free_detail, unit
from preprocessing import PreprocessingResult, _leaf_components

# Veryfing existing logger 
log = logging.getLogger("aap.interference")

REPO_ROOT = Path(__file__).resolve().parent.parent
NEG_TOL = 1e-6
# Static gap (mm) up to which two movable parts count as touching. Larger than Preprocessing's
# stander contact_threshold (0.01) so clearance fits count: industrial_fastner's part1 sits in
# part2/part3 with a 0.075 mm gap.
CONTACT_THRESHOLD = 0.1

# Called after each pair's checkpoint is written, with the
# number of pairs completed so far and the pair (i, j). Raising from it simulates an interrupt.
_after_pair_hook: Callable[[int, tuple[int, int]], None] | None = None

# Dataclass for the outcome of a single collision test between two parts along a direction.
@dataclass
class InterferenceResult:
    component_ids: list[str]  # obj.Name, index = matrix row/column
    component_labels: list[str]  # obj.Label, display only
    directions: np.ndarray  # (D, 3) float, unit vectors, Preprocessing order
    neg: np.ndarray  # (D,) int, dirs[neg[k]] == -dirs[k]
    free: np.ndarray  # (n, n, D) bool, free[i, j, k]: i can leave along dirs[k] past j
    pair_info: dict[tuple[int, int], dict]  # i < j: {"kind", "V0", "threshold"}
    exception_cells: list[tuple[int, int, int]]  # common failed twice -> reported blocked
    caught_exceptions: list[dict]  # every caught backend exception (call, pair, k, t, retry_ok)
    outcome_detail: dict[tuple[int, int, int], dict]  # i < j only: phase, iterations, t_block, volume
    config: CollisionConfig
    stats: dict  # timings, iteration counts, broad-phase share, gap-travel share
    cache_key: str
    # (n, n) bool, symmetric, diagonal False: i and j touch (static gap <= contact_threshold).
    contact: np.ndarray | None = None
    contact_threshold: float | None = None  # mm
    gaps: np.ndarray | None = None  # (n, n) float, static distance between i and j, mm

    # queries used by the AoG generator !!!!!NO GEOMETRY CALLS!!!!! 

    def index_of(self, component_id: str) -> int:
        """Index of the component with the given obj.Name in the matrix."""
        return self.component_ids.index(component_id)

    def free_directions(self, i: int, subset: Iterable[int]) -> np.ndarray:
        """Indices k along which i can leave the sub-assembly `subset` (i may be in subset)."""
        members = list(subset)
        if not members:
            return np.arange(len(self.directions))
        return np.flatnonzero(np.all(self.free[i, members, :], axis=0))

    def removable(self, subset: Iterable[int]) -> dict[int, np.ndarray]:
        """{i: free direction indices} for every i in subset with at least one free direction."""
        members = list(subset)
        result = {}
        for i in members:
            dirs = self.free_directions(i, members)
            if len(dirs):
                result[i] = dirs
        return result

    def blockers(self, i: int, k: int, subset: Iterable[int]) -> list[int]:
        """Members of subset that block i along dirs[k] (for debugging and the thesis)."""
        return [j for j in subset if j != i and not self.free[i, j, k]]

    def contact_neighbors(self, i: int) -> list[int]:
        """Components that physically touch i."""
        return [int(j) for j in np.flatnonzero(self.contact[i])]

    def is_connected(self, subset: Iterable[int]) -> bool:
        """True if the members of `subset` form one group through physical contacts."""
        return _connected(self.contact, list(subset))


# ---------------------------------------------------------------------------
# Phase 2 (docs/subassembly-spec.md section 10): derived tensors for an automatic sub-assembly
# group. Pure numpy, no geometry calls -- neither constructor re-runs collision.py.
# ---------------------------------------------------------------------------


def merge_groups(interf: InterferenceResult, group_ids: list[str], new_id: str, new_label: str) -> InterferenceResult:
    """Derived InterferenceResult with `group_ids` replaced by one component `new_id` (the group,
    treated as one rigid body). `free[new_id, j, k]` = AND over group members of `free[member, j,
    k]` (the matrix's own documented composition rule); `free[j, new_id, k]` is never computed
    separately -- it follows from the antisymmetric identity `free[j, i, neg[k]] == free[i, j,
    k]`. `contact[new_id, j]` = OR over members, `gaps[new_id, j]` = min over members. Diagnostic-
    only fields (`pair_info`, `exception_cells`, `caught_exceptions`, `outcome_detail`) are not
    carried over -- they describe individual collision calls, none of which are repeated here, so
    an empty value is more honest than a stale or synthesized one. Used both to rank Phase 2 merge
    candidates (on a throwaway copy) and to commit the chosen one."""
    ids = list(interf.component_ids)
    group_idx = sorted(interf.index_of(g) for g in group_ids)
    group_set = set(group_idx)
    if len(group_idx) < 2:
        raise ValueError(f"merge_groups needs at least 2 members, got {group_ids}")
    if new_id in ids:
        raise ValueError(f"new_id {new_id!r} already exists in this InterferenceResult")

    keep = [i for i in range(len(ids)) if i not in group_set]
    new_ids = [ids[i] for i in keep] + [new_id]
    new_labels = [interf.component_labels[i] for i in keep] + [new_label]
    n_new = len(new_ids)
    D = len(interf.directions)

    free = np.zeros((n_new, n_new, D), dtype=bool)
    free[np.ix_(range(len(keep)), range(len(keep)), range(D))] = \
        interf.free[np.ix_(keep, keep, range(D))]
    merged_row = np.ones((len(keep), D), dtype=bool)  # free[new_id, kept, k]
    for g in group_idx:
        merged_row &= interf.free[g, keep, :]
    free[-1, :-1, :] = merged_row
    for b in range(len(keep)):
        free[b, -1, interf.neg] = merged_row[b, :]  # antisymmetric identity, vectorized over k
    free[-1, -1, :] = True  # self-diagonal, as in the original matrix (free_directions may pass
                              # a subset that includes the component itself)

    contact = gaps = None
    if interf.contact is not None:
        contact = np.zeros((n_new, n_new), dtype=bool)
        contact[np.ix_(range(len(keep)), range(len(keep)))] = interf.contact[np.ix_(keep, keep)]
        merged_contact = np.zeros(len(keep), dtype=bool)
        for g in group_idx:
            merged_contact |= interf.contact[g, keep]
        contact[-1, :-1] = merged_contact
        contact[:-1, -1] = merged_contact
    if interf.gaps is not None:
        gaps = np.full((n_new, n_new), np.inf, dtype=float)
        gaps[np.ix_(range(len(keep)), range(len(keep)))] = interf.gaps[np.ix_(keep, keep)]
        merged_gaps = np.full(len(keep), np.inf, dtype=float)
        for g in group_idx:
            merged_gaps = np.minimum(merged_gaps, interf.gaps[g, keep])
        gaps[-1, :-1] = merged_gaps
        gaps[:-1, -1] = merged_gaps

    cache_key = hashlib.sha256(
        f"merge:v1;parent:{interf.cache_key};group:{sorted(group_ids)};new_id:{new_id};"
        f"ids:{'|'.join(new_ids)}".encode()
    ).hexdigest()

    return InterferenceResult(
        component_ids=new_ids, component_labels=new_labels,
        directions=interf.directions, neg=interf.neg, free=free,
        pair_info={}, exception_cells=[], caught_exceptions=[], outcome_detail={},
        config=interf.config, stats=dict(interf.stats), cache_key=cache_key,
        contact=contact, contact_threshold=interf.contact_threshold, gaps=gaps,
    )


def sub_interference(interf: InterferenceResult, group_ids: list[str]) -> InterferenceResult:
    """Restricted InterferenceResult for one Phase 2 group's own child level: the sub-block
    `free[A, A, :]` (A = group_ids), same `directions`/`neg` as the parent (not rediscovered -- the
    tensor stays indexed by the parent level's candidate-direction list, per spec)."""
    idx = sorted(interf.index_of(g) for g in group_ids)
    D = len(interf.directions)
    ids = [interf.component_ids[i] for i in idx]
    labels = [interf.component_labels[i] for i in idx]
    free = interf.free[np.ix_(idx, idx, range(D))]
    contact = interf.contact[np.ix_(idx, idx)] if interf.contact is not None else None
    gaps = interf.gaps[np.ix_(idx, idx)] if interf.gaps is not None else None
    cache_key = hashlib.sha256(
        f"sub:v1;parent:{interf.cache_key};members:{sorted(group_ids)}".encode()
    ).hexdigest()
    return InterferenceResult(
        component_ids=ids, component_labels=labels,
        directions=interf.directions, neg=interf.neg, free=free,
        pair_info={}, exception_cells=[], caught_exceptions=[], outcome_detail={},
        config=interf.config, stats=dict(interf.stats), cache_key=cache_key,
        contact=contact, contact_threshold=interf.contact_threshold, gaps=gaps,
    )


# Setup: components, directions, cache key, contacts

# Auxiliary function to get the leaf components of a FreeCAD document.
def _components(doc: "FreeCAD.Document"):
    """Return the reduced assembly's leaf components in a reproducible order,
      with their ids, labels and shapes."""
    comps = sorted(_leaf_components(doc), key=lambda obj: obj.Name)
    ids = [obj.Name for obj in comps]
    labels = [obj.Label for obj in comps]
    # The reduced doc already stores each shape at its global placement.
    shapes = [obj.Shape.copy() for obj in comps]
    return ids, labels, shapes

# Auxiliary function to extract the candidate directions 
# and their negation indices for antisymmetry from preprocessing.
def _directions(candidates) -> tuple[list[FreeCAD.Vector], np.ndarray]:
    """Return the candidate directions and the neg array for antisymmetry."""
    dirs = [unit(d) for d in candidates]
    neg = []
    for k, d in enumerate(dirs):
        match = [m for m, e in enumerate(dirs) if (e + d).Length < NEG_TOL]
        if not match:
            raise ValueError(
                f"Candidate direction {k} {tuple(d)} has no negative in the list; Preprocessing "
                "always emits both signs, so this is a bug upstream."
            )
        neg.append(match[0])
    return dirs, np.array(neg, dtype=int)


# Function to compute a cache key based on the preprocessing result, 
# component ids, shapes, directions, and collision configuration. 
# This key is used to determine if a cached interference matrix can be reused 
# or if it needs to be recomputed.
def _connected(contact: np.ndarray, members: list[int]) -> bool:
    """Flood fill over the contact graph restricted to `members`."""
    if len(members) <= 1:
        return True
    inside = set(members)
    seen, stack = {members[0]}, [members[0]]
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(contact[i]):
            j = int(j)
            if j in inside and j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) == len(inside)


def compute_contacts(shapes, labels, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Static gap between every pair and the contact matrix (gap <= threshold).

    distToShape is 0 for touching or overlapping (tolerance) pairs. If it fails, the pair counts
    as in contact when the bounding boxes, enlarged by the threshold, intersect (logged).
    """
    n = len(shapes)
    gaps = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                gap = shapes[i].distToShape(shapes[j])[0]
            except Exception as exc:  # noqa: BLE001  (OCC failure on touching geometry)
                bb = shapes[i].BoundBox
                bb.enlarge(threshold)
                gap = 0.0 if bb.intersect(shapes[j].BoundBox) else float("inf")
                log.warning("distToShape failed for %s / %s (%s); contact decided by bounding boxes: %s",
                            labels[i], labels[j], exc, gap == 0.0)
            gaps[i, j] = gaps[j, i] = gap
    contact = gaps <= threshold
    np.fill_diagonal(contact, False)
    return contact, gaps


def compute_cache_key(
    prep: PreprocessingResult, ids: list[str], shapes, dirs, config: CollisionConfig
) -> str:
    h = hashlib.sha256()
    if prep.step_path is not None:
        h.update(b"step:" + Path(prep.step_path).read_bytes())
    else:  # assembly built in code: hash the geometry itself
        for shape in shapes:
            h.update(b"shape:" + shape.exportBrepToString().encode())
    if prep.bom_path is not None:
        h.update(b"bom:" + Path(prep.bom_path).read_bytes())
    for d in dirs:
        h.update(f"dir:{round(d.x, 9)!r},{round(d.y, 9)!r},{round(d.z, 9)!r};".encode())
    h.update(("ids:" + "|".join(ids)).encode())
    h.update(("config:" + json.dumps(asdict(config), sort_keys=True)).encode())
    # The contact matrix is not part of the key: it does not affect `free`. It is stored with its
    # threshold and recomputed on load when that differs (see identify_interferences).
    return h.hexdigest()


def _default_cache_dir(prep: PreprocessingResult) -> Path:
    stem = Path(prep.step_path).stem if prep.step_path is not None else prep.reduced_assembly.Name
    return REPO_ROOT / "output" / stem / "interference"


# Name declarations for the cache files, interference matrix and related data.
NPZ_NAME = "interference.npz"
JSON_NAME = "interference.json"
CHECKPOINT_NAME = "interference.checkpoint.json"
TABLE_NAME = "interference_table.csv"


def _key_str(*idx: int) -> str:
    """String key for a pair or triple of indices, used in JSON dicts."""
    return ",".join(str(x) for x in idx)


def _parse_key(s: str) -> tuple[int, ...]:
    """Parse a string key back into a tuple of indices."""
    return tuple(int(x) for x in s.split(","))


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON to a temporary file and then atomically rename it to the target path."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# Main function for loading cached interference matrices, saving new results, and exporting the matrix to a CSV table.
def _try_load_cache(cache_dir: Path, cache_key: str, config: CollisionConfig) -> InterferenceResult | None:
    """Return the cached interference matrix if it exists and matches the cache key, else None."""
    json_path, npz_path = cache_dir / JSON_NAME, cache_dir / NPZ_NAME
    if not (json_path.exists() and npz_path.exists()):
        return None
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        if meta.get("cache_key") != cache_key:
            log.info("Cached interference matrix is stale (key mismatch); recomputing")
            return None
        arrays = np.load(npz_path)
        return InterferenceResult(
            component_ids=meta["component_ids"],
            component_labels=meta["component_labels"],
            directions=arrays["directions"],
            neg=arrays["neg"],
            free=arrays["free"],
            pair_info={_parse_key(k): v for k, v in meta["pair_info"].items()},
            exception_cells=[tuple(c) for c in meta["exception_cells"]],
            caught_exceptions=meta["caught_exceptions"],
            outcome_detail={_parse_key(k): v for k, v in meta["outcome_detail"].items()},
            config=config,
            stats=meta["stats"],
            cache_key=cache_key,
            contact=arrays["contact"] if "contact" in arrays.files else None,
            contact_threshold=meta.get("contact_threshold"),
            gaps=arrays["gaps"] if "gaps" in arrays.files else None,
        )
    except Exception as exc:  # noqa: BLE001  (corrupt cache: recompute rather than fail)
        log.warning("Could not load cached interference matrix (%s); recomputing", exc)
        return None

# Function to save cached interference results to disk, 
# including the boolean tensor of free directions, 
#  component information, and statistics.
def _save(result: InterferenceResult, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / NPZ_NAME, free=result.free, directions=result.directions, neg=result.neg,
             contact=result.contact, gaps=result.gaps)
    _write_json_atomic(cache_dir / JSON_NAME, {
        "cache_key": result.cache_key, # Cache Key
        "component_ids": result.component_ids, # Component IDs
        "component_labels": result.component_labels, # Component Labels
        "pair_info": {_key_str(*k): v for k, v in result.pair_info.items()}, # Pair Information
        "exception_cells": [list(c) for c in result.exception_cells], # Exception Cells (if there are any)
        "caught_exceptions": result.caught_exceptions, # Caught Exceptions
        "outcome_detail": {_key_str(*k): v for k, v in result.outcome_detail.items()}, # Outcome Detail
        "config": asdict(result.config), # Collision Configuration
        "contact_threshold": result.contact_threshold, # Contact threshold (mm) used for `contact`
        "contact_pairs": [[result.component_ids[i], result.component_ids[j]]
                          for i, j in zip(*np.nonzero(np.triu(result.contact)))], # Touching pairs
        "stats": result.stats, # Statistics
        "freecad_version": ".".join(FreeCAD.Version()[:3]), # FreeCAD Version (In case of exporting)
        "date": datetime.datetime.now().isoformat(timespec="seconds"), # Date of Save/Export
    })


def _fmt_vec(v) -> str:
    """Format a vector as a string with 6 decimal places, 
    ensuring that -0.0 is displayed as 0.0."""
     # +0.0 turns -0.0 into 0.0 so a zero component prints the same in (i, j) and (j, i).
    return "(" + ", ".join(f"{round(float(c), 6) + 0.0:.6f}" for c in v) + ")"

# Formats table and exports to CSV, ensuring that the output is human-readable.
def export_table(result: InterferenceResult, path: Path) -> None:
    """Matrix in the layout of the thesis figure: cell (i, j) = free directions of i past j.
    Written as utf-8-sig: the BOM makes Excel on Windows read the "—" diagonal correctly.
    """
    n = len(result.component_ids)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([""] + result.component_labels)
        for i in range(n):
            row = [result.component_labels[i]]
            for j in range(n):
                if i == j:
                    row.append("—")
                else:
                    ks = np.flatnonzero(result.free[i, j, :])
                    row.append("; ".join(_fmt_vec(result.directions[k]) for k in ks))
            writer.writerow(row)


# Summarizes statistics for export.
def _stats_summary(stats: Stats, pair_times: dict[str, float], total_time: float) -> dict:
    stepped = len(stats.times)
    return {
        "tests": stats.calls, 
        "broad_phase_settled": stats.broad_free,
        "broad_phase_share": stats.broad_free / stats.calls if stats.calls else 0.0,
        "stepped_tests": stepped,
        "mean_test_time_s": sum(stats.times) / stepped if stepped else 0.0,
        "max_test_time_s": max(stats.times) if stepped else 0.0,
        "iterations": stats.iterations,
        "mean_iterations": stats.iterations / stepped if stepped else 0.0,
        "max_iterations": max(stats.iters_per_test) if stepped else 0,
        "common_calls": stats.commons,
        "distToShape_calls": stats.dists,
        "gap_travel_share": stats.gap_travel / stats.travel if stats.travel else 0.0,
        "pair_times_s": pair_times,
        "matrix_time_s": sum(pair_times.values()),
        "total_time_s": total_time,
    }

# Verifies and handles static overlaps for all pairs of components.
# Static overlaps happen when two components occupy the same space even before 
# the colision analyses (!Should be avoided!).
# Raises error if overlaps are too big (< overlap_tol = 1mm^3) 
def _classify_all(ids, labels, shapes, config) -> dict[tuple[int, int], dict]:
    """Static-overlap pre-check for every i < j. Raises one error listing every hard overlap."""
    n = len(shapes)
    pair_info, hard = {}, []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                kind, v0, threshold = classify_pair(shapes[i], shapes[j], config)
            except HardOverlapError:
                hard.append((ids[i], ids[j], shapes[i].common(shapes[j]).Volume))
                continue
            pair_info[(i, j)] = {"kind": kind, "V0": v0, "threshold": threshold}
            if kind == "tolerance":
                log.warning("Tolerance pair %s / %s (%s / %s): static overlap V0=%.4f mm^3, blocking threshold=%.4f",
                            labels[i], labels[j], ids[i], ids[j], v0, threshold)
    if hard:
        listing = "; ".join(f"{a} / {b} (V0={v:.4f} mm^3)" for a, b, v in hard)
        raise HardOverlapError(
            f"{len(hard)} pair(s) overlap statically by more than overlap_tol={config.overlap_tol} mm^3 "
            f"(fix the model or mark the part as stander): {listing}"
        )
    return pair_info

# Main function to identify interferences between components in a reduced assembly.
def identify_interferences(
    prep: PreprocessingResult,
    config: CollisionConfig | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    contact_threshold: float | None = None,
) -> InterferenceResult:
    """Build (or load from cache) the interference matrix for Preprocessing's reduced assembly.

    `force=True` ignores both a finished cache and an in-progress checkpoint.
    `contact_threshold` (mm) defaults to CONTACT_THRESHOLD; a cached matrix computed with another
    threshold keeps its `free` tensor and only has its contacts recomputed.
    """

    # Setup and cache key computation
    config = config or CollisionConfig()
    if contact_threshold is None:
        contact_threshold = CONTACT_THRESHOLD
    t_start = time.perf_counter()
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir(prep)

    # Load cached interference matrix if available and valid
    ids, labels, shapes = _components(prep.reduced_assembly)
    dirs, neg = _directions(prep.candidate_directions)
    n, D = len(ids), len(dirs)
    cache_key = compute_cache_key(prep, ids, shapes, dirs, config)

    # If not forcing a recomputation, attempt to load the cached interference matrix.
    if not force:
        cached = _try_load_cache(cache_dir, cache_key, config)
        if cached is not None:
            log.info("Loaded cached interference matrix from %s (%d components x %d directions)",
                     cache_dir, n, D)
            if cached.contact is None or cached.contact_threshold != contact_threshold:
                # Cache written before contacts were recorded, or with another threshold:
                # compute the contacts only (cheap) instead of the whole matrix.
                log.info("Computing the contact matrix for the cached result (threshold %.4g mm)",
                         contact_threshold)
                cached.contact, cached.gaps = compute_contacts(shapes, labels, contact_threshold)
                cached.contact_threshold = contact_threshold
                _save(cached, cache_dir)
                _log_contacts(cached)
            return cached

    
    pair_info = _classify_all(ids, labels, shapes, config)

    # Initialize the free matrix, exception cells, outcome details, pair times, and statistics.
    free = np.zeros((n, n, D), dtype=bool)
    exception_cells: list[tuple[int, int, int]] = []
    outcome_detail: dict[tuple[int, int, int], dict] = {}
    pair_times: dict[str, float] = {}
    stats = Stats()
    completed: set[tuple[int, int]] = set()
    rows: dict[str, list[bool]] = {}

    # Checkpoint handling: if a checkpoint exists and matches the cache key, resume from it.
    checkpoint_path = cache_dir / CHECKPOINT_NAME
    if not force and checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Ignoring unreadable checkpoint %s (%s)", checkpoint_path, exc)
            ckpt = None
        if ckpt and ckpt.get("cache_key") == cache_key:
            rows = ckpt["rows"]
            for key, row in rows.items():
                i, j = _parse_key(key)
                completed.add((i, j))
                free[i, j, :] = row
                free[j, i, neg] = row
            exception_cells = [tuple(c) for c in ckpt["exception_cells"]]
            outcome_detail = {_parse_key(k): v for k, v in ckpt["outcome_detail"].items()}
            pair_times = ckpt["pair_times"]
            stats = Stats(**ckpt["stats"])
            log.info("Resuming from checkpoint: %d of %d pairs already done", len(completed), n * (n - 1) // 2)
        elif ckpt:
            log.info("Ignoring checkpoint with a different cache key")

    # Main loop over all pairs of components and candidate directions to determine interferences.
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    # Loop over each pair of components (i, j) to check for interferences along all candidate directions.
    for i, j in pairs:
        # Verify if the pair has already been completed in the checkpoint; if so, skip it.
        if (i, j) in completed:
            continue
        threshold = pair_info[(i, j)]["threshold"]
        t0 = time.perf_counter()
        # Loop over all candidate directions for the current pair of components.
        for k, d in enumerate(dirs):
            n_exc = len(stats.exceptions)
            try:
                # Call the collision test for the current pair of components along the current direction.
                out = is_free_detail(shapes[i], shapes[j], d, config, stats, threshold)
            except Exception as exc:
                raise RuntimeError(
                    f"Collision test failed for pair {ids[i]} / {ids[j]}, direction {k} {tuple(d)}: {exc}"
                ) from exc
            # Obtain results and update the free matrix and exception information accordingly.
            for e in stats.exceptions[n_exc:]:
                e.update(pair=[ids[i], ids[j]], k=k)
            free[i, j, k] = out.free
            free[j, i, neg[k]] = out.free  # antisymmetry, no second test

            # Exception handling: if the collision test failed twice for this cell, mark it as an exception cell.
            if out.exception_cell:
                exception_cells.append((i, j, k))
                log.warning("Exception cell %s / %s dir %d %s: common failed twice, reported blocked",
                            labels[i], labels[j], k, tuple(d))
            outcome_detail[(i, j, k)] = {
                "phase": out.phase, "iterations": out.iterations,
                "t_block": out.t_block, "volume": out.volume,
            }
        # Calculate the time taken for the current pair and update the pair_times dictionary.
        pair_time = time.perf_counter() - t0
        pair_times[_key_str(i, j)] = pair_time

        # Set the pair as completed, update the rows dictionary, and write the checkpoint to disk.
        completed.add((i, j))
        rows[_key_str(i, j)] = [bool(x) for x in free[i, j, :]]
        cache_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(checkpoint_path, {
            "cache_key": cache_key,
            "rows": rows,
            "exception_cells": [list(c) for c in exception_cells],
            "outcome_detail": {_key_str(*k): v for k, v in outcome_detail.items()},
            "pair_times": pair_times,
            "stats": asdict(stats),
        })
        # Add to log for user information about the progress
        log.info("Pair %d/%d %s / %s: %.1f s, %d/%d directions free",
                 len(completed), len(pairs), labels[i], labels[j], pair_time, int(free[i, j].sum()), D)
        if _after_pair_hook is not None:
            _after_pair_hook(len(completed), (i, j))

    # Diagonal: neutral element of AND, so queries may include i itself.
    for i in range(n):
        free[i, i, :] = True

    contact, gaps = compute_contacts(shapes, labels, contact_threshold)

    # Store results
    result = InterferenceResult(
        component_ids=ids,
        component_labels=labels,
        directions=np.array([[d.x, d.y, d.z] for d in dirs], dtype=float),
        neg=neg,
        free=free,
        pair_info=pair_info,
        exception_cells=exception_cells,
        caught_exceptions=stats.exceptions,
        outcome_detail=outcome_detail,
        config=config,
        stats=_stats_summary(stats, pair_times, time.perf_counter() - t_start),
        cache_key=cache_key,
        contact=contact,
        contact_threshold=contact_threshold,
        gaps=gaps,
    )
    _save(result, cache_dir)
    export_table(result, cache_dir / TABLE_NAME)
    checkpoint_path.unlink(missing_ok=True)
    _log_summary(result)
    return result

# Function to log a summary of the interference results, including the number of components, 
# directions, pairs, and statistics about the tests performed. For user information and debugging.
def _log_summary(result: InterferenceResult) -> None:
    n, D = len(result.component_ids), len(result.directions)
    labels = result.component_labels
    for i in range(n):
        for j in range(n):
            if i != j and not result.free[i, j].any():
                log.warning("No free direction for %s past %s (interlocked, missing candidate direction "
                            "or misclassified; see collision test E4)", labels[i], labels[j])
    s = result.stats
    tolerance = [(labels[i], labels[j], info["V0"]) for (i, j), info in result.pair_info.items()
                 if info["kind"] == "tolerance"]
    log.info("Interference matrix: n=%d, D=%d, pairs=%d, total %.1f s", n, D, n * (n - 1) // 2, s["total_time_s"])
    log.info("Tolerance pairs: %s", ", ".join(f"{a}/{b} (V0={v:.4f})" for a, b, v in tolerance) or "none")
    log.info("Exception cells: %s", result.exception_cells or "none")
    log.info("Broad-phase share %.1f%%, mean iterations per stepped test %.1f",
             100 * s["broad_phase_share"], s["mean_iterations"])
    removable = result.removable(range(n))
    log.info("Removable with all components present: %s", ", ".join(
        f"{labels[i]}: {[int(k) for k in ks]}" for i, ks in removable.items()) or "none")
    _log_contacts(result)


def _log_contacts(result: InterferenceResult) -> None:
    labels, n = result.component_labels, len(result.component_ids)
    pairs = [(labels[i], labels[j]) for i, j in zip(*np.nonzero(np.triu(result.contact)))]
    log.info("Contacts (gap <= %.4g mm): %s", result.contact_threshold,
             ", ".join(f"{a}/{b}" for a, b in pairs) or "none")
    for i in range(n):
        if not result.contact[i].any():
            log.warning("%s touches no other movable component (it may only be held by standers)", labels[i])
    if not result.is_connected(range(n)):
        log.info("Movable components alone are not one connected group (the AoG also uses stander contacts)")
