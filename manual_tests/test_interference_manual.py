"""Manual smoke test for Interference Identification (src/interference_identification.py).

Counterpart of manual_tests/test_preprocessing_manual.py. Not an acceptance suite that compares against
a reference result. This script runs Preprocessing + Interference Identification end to end on one assembly and dumps everything in a form you can inspect by eye:

  output/industrial_fastner/interference/
    interference.npz            the matrix itself (free, directions, neg) and the contact matrix
                                (contact, gaps), written by the module
    interference.json           module metadata (pair classes, exceptions, stats, cache key)
    interference_table.csv      matrix in the thesis-figure layout: cell (i, j) = directions along
                                which row part i can leave past column part j
    interference_report.json    this script's readable report (see below)

The report lists, with part labels and direction vectors instead of indices:
  - components (label <-> FreeCAD Name) and the numbered candidate directions
  - static-overlap class of every pair (normal / tolerance, V0, blocking threshold)
  - for every ordered pair, the free directions of i past j
  - which parts are removable with the whole assembly present, and along which directions
  - for every part and every direction, which parts block it in the full assembly
  - pairs with no free direction at all, exception cells, timing statistics
  - the contact matrix: threshold used, touching pairs with their static gap, the full gap matrix,
    contact neighbours of every part, parts touching nothing, and whether the movable parts form
    one connected group
  - three quick consistency checks (antisymmetry, diagonal, contact symmetry)

The matrix is cached: a second run with the same STEP, BOM and settings loads it instantly.
Use --force to recompute.

Run (from repo root, environmentAAP active):

    python manual_tests/test_interference_manual.py

or point it at a different STEP/BOM pair and output folder:

    python manual_tests/test_interference_manual.py path/to.stp path/to/bom.csv tests/out_dir --step 1.0

--step (the collision step h, mm) defaults to 5.0, which is only calibrated for
industrial_fastner (collision test E6). For another assembly, calibrate h first.

Expected result: no error; the console prints the number of components and directions, how many pairs are free /
blocked, the removable parts with the whole assembly present, the contact summary and the timing (several minutes
the first time, instant when cached). The three consistency checks at the end must pass: free is antisymmetric
(free[i, j, k] == free[j, i, neg[k]]), the diagonal is free, and the contact matrix is symmetric.
interference_report.json and interference_table.csv are written next to the cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import freecad  # noqa: F401  (sets up the FreeCAD lib path in the conda env; must come first)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from collision import CollisionConfig  # noqa: E402
from interference_identification import identify_interferences  # noqa: E402
from preprocessing import run_preprocessing  # noqa: E402

DEFAULT_STEP = REPO_ROOT / "data" / "industrial_fastner" / "industrial_fastner.stp"
DEFAULT_BOM = REPO_ROOT / "data" / "industrial_fastner" / "generated_bom.csv"
DEFAULT_OUT = REPO_ROOT / "output" / "industrial_fastner"
DEFAULT_H = 5.0


def _vec(v) -> list[float]:
    # round for readability; +0.0 turns -0.0 into 0.0
    return [round(float(c), 6) + 0.0 for c in v]


def _dirs(result, ks) -> list[list[float]]:
    return [_vec(result.directions[k]) for k in ks]


def _gap(g: float) -> float | None:
    # JSON has no infinity; an infinite gap only comes from a failed distToShape (bounding-box fallback)
    return None if not np.isfinite(g) else round(float(g), 6)


def build_contacts(result) -> dict:
    """Contact matrix section of the report (movable parts only; standers come from their records)."""
    if result.contact is None:
        return {"available": False}
    n = len(result.component_ids)
    labels = result.component_labels
    contact, gaps = result.contact, result.gaps
    touching = [
        {"pair": f"{labels[i]} / {labels[j]}", "gap_mm": _gap(gaps[i, j])}
        for i in range(n) for j in range(i + 1, n) if contact[i, j]
    ]
    return {
        "available": True,
        "threshold_mm": result.contact_threshold,
        "touching_pairs": touching,
        "contact_matrix": {labels[i]: {labels[j]: bool(contact[i, j]) for j in range(n)} for i in range(n)},
        "gap_matrix_mm": {labels[i]: {labels[j]: _gap(gaps[i, j]) for j in range(n)} for i in range(n)},
        "neighbors": {labels[i]: [labels[j] for j in result.contact_neighbors(i)] for i in range(n)},
        "touching_nothing": [labels[i] for i in range(n) if not contact[i].any()],
        "movable_parts_connected": result.is_connected(range(n)),
    }


def build_report(result) -> dict:
    n = len(result.component_ids)
    D = len(result.directions)
    labels = result.component_labels
    everyone = list(range(n))

    # Free directions of i past j, for every ordered pair.
    pairwise = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ks = np.flatnonzero(result.free[i, j, :])
            pairwise[f"{labels[i]} past {labels[j]}"] = {
                "count": int(len(ks)),
                "directions": _dirs(result, ks),
            }

    # Removable parts with the whole assembly present.
    removable = {
        labels[i]: {"direction_indices": [int(k) for k in ks], "directions": _dirs(result, ks)}
        for i, ks in result.removable(everyone).items()
    }

    # Blockers of every part along every direction in the full assembly.
    blockers = {}
    for i in range(n):
        per_dir = {}
        for k in range(D):
            b = result.blockers(i, k, everyone)
            per_dir[f"{k} {_vec(result.directions[k])}"] = [labels[j] for j in b] or "free"
        blockers[labels[i]] = per_dir

    # Pairs with no free direction at all (interlocked, missing direction or misclassified).
    no_free = [
        f"{labels[i]} past {labels[j]}"
        for i in range(n) for j in range(n)
        if i != j and not result.free[i, j].any()
    ]

    # Consistency checks (the acceptance tests cover these properly; this is a quick sanity look).
    mismatches = sum(
        1
        for i in range(n) for j in range(n) if i != j
        for k in range(D)
        if result.free[j, i, result.neg[k]] != result.free[i, j, k]
    )
    diagonal_ok = bool(all(result.free[i, i, :].all() for i in range(n)))
    contact_symmetric = None if result.contact is None else bool(
        np.array_equal(result.contact, result.contact.T) and not result.contact.diagonal().any()
    )

    return {
        "component_count": n,
        "components": [
            {"index": i, "label": labels[i], "name": result.component_ids[i]} for i in range(n)
        ],
        "direction_count": D,
        "directions": {str(k): _vec(result.directions[k]) for k in range(D)},
        "config": {
            "step": result.config.step,
            "volume_epsilon": result.config.volume_epsilon,
            "overlap_tol": result.config.overlap_tol,
            "margin": result.config.margin,
            "gap_skip": result.config.gap_skip,
            "use_broad_phase": result.config.use_broad_phase,
        },
        "pair_classes": {
            f"{labels[i]} / {labels[j]}": info for (i, j), info in sorted(result.pair_info.items())
        },
        "removable_with_all_parts": removable,
        "free_directions_per_pair": pairwise,
        "blockers_in_full_assembly": blockers,
        "pairs_with_no_free_direction": no_free,
        "exception_cells": [
            {"pair": f"{labels[i]} / {labels[j]}", "direction": k, "vector": _vec(result.directions[k])}
            for i, j, k in result.exception_cells
        ],
        "caught_backend_exceptions": len(result.caught_exceptions),
        "contacts": build_contacts(result),
        "checks": {
            "antisymmetry_mismatches": mismatches,
            "diagonal_all_true": diagonal_ok,
            "contact_symmetric_diagonal_false": contact_symmetric,
        },
        "stats": {k: v for k, v in result.stats.items() if k != "pair_times_s"},
        "pair_times_s": result.stats.get("pair_times_s", {}),
        "cache_key": result.cache_key,
    }


def print_summary(result, report: dict, elapsed: float, cache_dir: Path, report_path: Path) -> None:
    labels = result.component_labels
    print()
    print(f"Components: {report['component_count']}   Candidate directions: {report['direction_count']}   "
          f"h = {result.config.step} mm")
    tolerance = [(p, c) for p, c in report["pair_classes"].items() if c["kind"] == "tolerance"]
    print(f"Tolerance pairs: {len(tolerance)}")
    for p, c in tolerance:
        print(f"  {p}: V0 = {c['V0']:.4f} mm^3, blocking threshold = {c['threshold']:.4f}")

    print("Removable with all parts present:")
    if report["removable_with_all_parts"]:
        for label, info in report["removable_with_all_parts"].items():
            dirs = ", ".join(f"{k}:{v}" for k, v in zip(info["direction_indices"], info["directions"]))
            print(f"  {label}: {dirs}")
    else:
        print("  none  <-- the AoG would have nowhere to start")

    n = len(labels)
    print("Free directions per pair (count, row = moving part, column = obstacle):")
    width = max(len(s) for s in labels)
    print("  " + " " * width + "  " + "  ".join(f"{j:>3}" for j in range(n)))
    for i in range(n):
        cells = "  ".join("  -" if i == j else f"{int(result.free[i, j].sum()):>3}" for j in range(n))
        print(f"  {labels[i]:<{width}}  {cells}   ({i})")

    if report["pairs_with_no_free_direction"]:
        print("Pairs with NO free direction (check these):")
        for p in report["pairs_with_no_free_direction"]:
            print(f"  {p}")
    contacts = report["contacts"]
    if contacts["available"]:
        print(f"Contact matrix (x = touching, static gap <= {contacts['threshold_mm']} mm):")
        print("  " + " " * width + "  " + "  ".join(f"{j:>3}" for j in range(n)))
        for i in range(n):
            cells = "  ".join("  -" if i == j else ("  x" if result.contact[i, j] else "  .") for j in range(n))
            print(f"  {labels[i]:<{width}}  {cells}   ({i})")
        print("Touching pairs (static gap):")
        for p in contacts["touching_pairs"]:
            gap = "inf" if p["gap_mm"] is None else f"{p['gap_mm']:.4f} mm"
            print(f"  {p['pair']}: {gap}")
        if contacts["touching_nothing"]:
            print(f"Touching no other movable part (held only by standers?): {', '.join(contacts['touching_nothing'])}")
        print(f"Movable parts form one connected group: {contacts['movable_parts_connected']}")
    else:
        print("Contact matrix: not available (cached result without contacts)")

    print(f"Exception cells: {len(report['exception_cells'])}   "
          f"caught backend exceptions: {report['caught_backend_exceptions']}")
    chk = report["checks"]
    print(f"Checks: antisymmetry mismatches = {chk['antisymmetry_mismatches']}, "
          f"diagonal all true = {chk['diagonal_all_true']}, "
          f"contact symmetric with empty diagonal = {chk['contact_symmetric_diagonal_false']}")
    s = report["stats"]
    print(f"Time: {elapsed:.1f} s this run, including preprocessing "
          f"(the matrix itself took {s.get('matrix_time_s', 0.0):.1f} s when it was computed; "
          f"a short run time means it was loaded from the cache)")
    print(f"Matrix files in {cache_dir}")
    print(f"Full report written to {report_path}")


def main(step_path: Path, bom_path: Path, out_dir: Path, h: float, force: bool) -> None:
    """Pipeline: run_preprocessing -> identify_interferences (for every ordered pair of parts and every candidate
    direction, a collision sweep with step h decides whether the part can leave past the other: the `free`
    tensor; plus touching pairs and gaps: `contact`, `gaps`; cached on disk) -> readable report and summary."""
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "interference"

    t0 = time.perf_counter()
    print(f"Preprocessing {step_path.name} ...", flush=True)
    prep = run_preprocessing(step_path, bom_path)
    print(f"Interference Identification (h = {h}) ...", flush=True)
    result = identify_interferences(prep, CollisionConfig(step=h), cache_dir, force=force)
    elapsed = time.perf_counter() - t0

    report = build_report(result)
    report_path = cache_dir / "interference_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(result, report, elapsed, cache_dir, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Interference Identification and dump a readable report.")
    parser.add_argument("paths", nargs="*", type=Path, help="[step_path bom_path out_dir]")
    parser.add_argument("--step", type=float, default=DEFAULT_H,
                        help=f"collision step h in mm (default {DEFAULT_H}, calibrated for industrial_fastner only)")
    parser.add_argument("--force", action="store_true", help="recompute even if a cached matrix exists")
    args = parser.parse_args()
    if len(args.paths) == 0:
        step, bom, out = DEFAULT_STEP, DEFAULT_BOM, DEFAULT_OUT
    elif len(args.paths) == 3:
        step, bom, out = args.paths
    else:
        parser.error("give either no paths or all three: step_path bom_path out_dir")
    main(step, bom, out, args.step, args.force)
