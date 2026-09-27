"""Interference Identification: the pairwise interference matrix.

Implements docs/interference-spec.md. For every ordered pair of reduced-assembly components
(i, j) and every candidate direction d_k, decides whether i can translate along d_k all the way
past j without penetrating it, using the verified collision test in collision.py. The result is
the boolean tensor free[i, j, k], computed once per assembly (cached on disk) and queried by
the AoG generator without further geometry calls.

Two properties make the pairwise matrix sufficient and halve its cost:
  - Composition: i can leave sub-assembly S along d_k iff free[i, j, k] for every j in S.
  - Antisymmetry: free[j, i, neg[k]] == free[i, j, k], so only the i < j triangle is computed.
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

log = logging.getLogger("aap.interference")

REPO_ROOT = Path(__file__).resolve().parent.parent
NEG_TOL = 1e-6

# Test hook (acceptance test I7): called after each pair's checkpoint is written, with the
# number of pairs completed so far and the pair (i, j). Raising from it simulates an interrupt.
_after_pair_hook: Callable[[int, tuple[int, int]], None] | None = None


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

    # --- queries used by the AoG generator (no geometry calls) ---

    def index_of(self, component_id: str) -> int:
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


# ---------------------------------------------------------------------------
# Setup: components, directions, cache key
# ---------------------------------------------------------------------------


def _components(doc: "FreeCAD.Document"):
    comps = sorted(_leaf_components(doc), key=lambda obj: obj.Name)
    ids = [obj.Name for obj in comps]
    labels = [obj.Label for obj in comps]
    # The reduced doc already stores each shape at its global placement.
    shapes = [obj.Shape.copy() for obj in comps]
    return ids, labels, shapes


def _directions(candidates) -> tuple[list[FreeCAD.Vector], np.ndarray]:
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
    return h.hexdigest()


def _default_cache_dir(prep: PreprocessingResult) -> Path:
    stem = Path(prep.step_path).stem if prep.step_path is not None else prep.reduced_assembly.Name
    return REPO_ROOT / "output" / stem / "interference"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

NPZ_NAME = "interference.npz"
JSON_NAME = "interference.json"
CHECKPOINT_NAME = "interference.checkpoint.json"
TABLE_NAME = "interference_table.csv"


def _key_str(*idx: int) -> str:
    return ",".join(str(x) for x in idx)


def _parse_key(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _try_load_cache(cache_dir: Path, cache_key: str, config: CollisionConfig) -> InterferenceResult | None:
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
        )
    except Exception as exc:  # noqa: BLE001  (corrupt cache: recompute rather than fail)
        log.warning("Could not load cached interference matrix (%s); recomputing", exc)
        return None


def _save(result: InterferenceResult, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / NPZ_NAME, free=result.free, directions=result.directions, neg=result.neg)
    _write_json_atomic(cache_dir / JSON_NAME, {
        "cache_key": result.cache_key,
        "component_ids": result.component_ids,
        "component_labels": result.component_labels,
        "pair_info": {_key_str(*k): v for k, v in result.pair_info.items()},
        "exception_cells": [list(c) for c in result.exception_cells],
        "caught_exceptions": result.caught_exceptions,
        "outcome_detail": {_key_str(*k): v for k, v in result.outcome_detail.items()},
        "config": asdict(result.config),
        "stats": result.stats,
        "freecad_version": ".".join(FreeCAD.Version()[:3]),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    })


def _fmt_vec(v) -> str:
    # +0.0 turns -0.0 into 0.0 so a zero component prints the same in (i, j) and (j, i).
    return "(" + ", ".join(f"{round(float(c), 6) + 0.0:.6f}" for c in v) + ")"


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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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


def identify_interferences(
    prep: PreprocessingResult,
    config: CollisionConfig | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> InterferenceResult:
    """Build (or load from cache) the interference matrix for Preprocessing's reduced assembly.

    `force=True` ignores both a finished cache and an in-progress checkpoint.
    """
    config = config or CollisionConfig()
    t_start = time.perf_counter()
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir(prep)

    ids, labels, shapes = _components(prep.reduced_assembly)
    dirs, neg = _directions(prep.candidate_directions)
    n, D = len(ids), len(dirs)
    cache_key = compute_cache_key(prep, ids, shapes, dirs, config)

    if not force:
        cached = _try_load_cache(cache_dir, cache_key, config)
        if cached is not None:
            log.info("Loaded cached interference matrix from %s (%d components x %d directions)",
                     cache_dir, n, D)
            return cached

    pair_info = _classify_all(ids, labels, shapes, config)

    free = np.zeros((n, n, D), dtype=bool)
    exception_cells: list[tuple[int, int, int]] = []
    outcome_detail: dict[tuple[int, int, int], dict] = {}
    pair_times: dict[str, float] = {}
    stats = Stats()
    completed: set[tuple[int, int]] = set()
    rows: dict[str, list[bool]] = {}

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

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        if (i, j) in completed:
            continue
        threshold = pair_info[(i, j)]["threshold"]
        t0 = time.perf_counter()
        for k, d in enumerate(dirs):
            n_exc = len(stats.exceptions)
            try:
                out = is_free_detail(shapes[i], shapes[j], d, config, stats, threshold)
            except Exception as exc:
                raise RuntimeError(
                    f"Collision test failed for pair {ids[i]} / {ids[j]}, direction {k} {tuple(d)}: {exc}"
                ) from exc
            for e in stats.exceptions[n_exc:]:
                e.update(pair=[ids[i], ids[j]], k=k)
            free[i, j, k] = out.free
            free[j, i, neg[k]] = out.free  # antisymmetry, no second test
            if out.exception_cell:
                exception_cells.append((i, j, k))
                log.warning("Exception cell %s / %s dir %d %s: common failed twice, reported blocked",
                            labels[i], labels[j], k, tuple(d))
            outcome_detail[(i, j, k)] = {
                "phase": out.phase, "iterations": out.iterations,
                "t_block": out.t_block, "volume": out.volume,
            }
        pair_time = time.perf_counter() - t0
        pair_times[_key_str(i, j)] = pair_time
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
        log.info("Pair %d/%d %s / %s: %.1f s, %d/%d directions free",
                 len(completed), len(pairs), labels[i], labels[j], pair_time, int(free[i, j].sum()), D)
        if _after_pair_hook is not None:
            _after_pair_hook(len(completed), (i, j))

    # Diagonal: neutral element of AND, so queries may include i itself.
    for i in range(n):
        free[i, i, :] = True

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
    )
    _save(result, cache_dir)
    export_table(result, cache_dir / TABLE_NAME)
    checkpoint_path.unlink(missing_ok=True)
    _log_summary(result)
    return result


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
