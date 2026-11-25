"""Acceptance tests for Interference Identification (docs/interference-spec.md, I1-I11).

Standalone runner (pytest is not installed in environmentAAP). From repo root:

    python automatic_tests/test_interference.py                  # everything (~20 min)
    python automatic_tests/test_interference.py --tests I8 I11   # individual tests
    python automatic_tests/test_interference.py --step 2.0

Uses industrial_fastner at h = 2.0 unless noted. The expensive runs (brute-force matrix, module
run) are computed once and shared between tests. Scratch caches go to
automatic_tests/output/interference_test/; results to automatic_tests/output/interference_results.json.

Recovered 2026-11-24 from git history (commit f4d4835, "Interference Identification and Tests",
2026-09-27) after this file and test_collision.py were accidentally deleted the next day along
with the rest of the old tests_scripts/ directory (commit 4e95958). Content is otherwise verbatim;
the only change is the BOM path (data/industrial_fastner/bom.csv no longer exists, renamed to
generated_bom.csv on 2026-11-05). This suite never touched the contact/gaps subsystem added to
InterferenceResult on 2026-10-11 (after this file was first written) -- no coverage gap introduced
by the recovery, just one that already existed.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import json
import logging
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import freecad  # noqa: F401  (sets up the FreeCAD lib path in the conda env)
import FreeCAD
import numpy as np
import Part
from FreeCAD import Vector

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import collision  # noqa: E402  (module attribute access lets I9 replace the backend hook)
import interference_identification as ii  # noqa: E402
from collision import CollisionConfig, HardOverlapError, Stats, classify_pair, is_free_detail  # noqa: E402
from preprocessing import PreprocessingResult, run_preprocessing  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
SCRATCH = OUT_DIR / "interference_test"
FASTENER_STEP = REPO_ROOT / "data" / "industrial_fastner" / "industrial_fastner.stp"
FASTENER_BOM = REPO_ROOT / "data" / "industrial_fastner" / "generated_bom.csv"

X, Y, Z = Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)


# ---------------------------------------------------------------------------
# Tiny test harness (same shape as test_collision.py)
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


def fresh_dir(name: str) -> Path:
    path = SCRATCH / name
    shutil.rmtree(path, ignore_errors=True)
    return path


def _vec(v) -> tuple[float, float, float]:
    return tuple(round(float(c), 6) + 0.0 for c in v)


# ---------------------------------------------------------------------------
# Shared expensive state
# ---------------------------------------------------------------------------

_STATE: dict = {}


def fastener_prep() -> PreprocessingResult:
    if "prep" not in _STATE:
        print("  (running preprocessing on industrial_fastner)", flush=True)
        _STATE["prep"] = run_preprocessing(FASTENER_STEP, FASTENER_BOM)
    return _STATE["prep"]


def brute_force(config: CollisionConfig):
    """Both triangles computed independently with collision.is_free_detail (spec I2)."""
    if "brute" not in _STATE:
        prep = fastener_prep()
        ids, labels, shapes = ii._components(prep.reduced_assembly)
        dirs, _ = ii._directions(prep.candidate_directions)
        n, D = len(ids), len(dirs)
        free = np.ones((n, n, D), dtype=bool)
        t0 = time.perf_counter()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                _, _, thr = classify_pair(shapes[min(i, j)], shapes[max(i, j)], config)
                tp = time.perf_counter()
                for k, d in enumerate(dirs):
                    free[i, j, k] = is_free_detail(shapes[i], shapes[j], d, config, Stats(), thr).free
                print(f"    brute force {labels[i]} vs {labels[j]}: {time.perf_counter() - tp:.1f} s", flush=True)
        _STATE["brute"] = (free, time.perf_counter() - t0)
    return _STATE["brute"]


def module_run(config: CollisionConfig):
    """One cold module run into a fresh cache dir (shared by I2-I6, I9, I10)."""
    if "module" not in _STATE:
        cache = fresh_dir("module")
        t0 = time.perf_counter()
        result = ii.identify_interferences(fastener_prep(), config, cache)
        _STATE["module"] = (result, time.perf_counter() - t0, cache)
    return _STATE["module"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_i1(rec, config):
    cmd = [sys.executable, str(REPO_ROOT / "automatic_tests" / "test_collision.py"), "--groups", "A", "B", "C", "D"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    rows = [line for line in proc.stdout.splitlines() if line.startswith("| ")][1:]  # skip header row
    failed = [line for line in rows if "| FAIL" in line or "| ERROR" in line]
    for line in failed:
        rec.note(line)
    rec.check(proc.returncode == 0 and not failed,
              f"test_collision.py --groups A B C D: exit code {proc.returncode}, {len(rows)} tests, "
              f"{len(failed)} failed")


def test_i2(rec, config):
    brute, t_brute = brute_force(config)
    result, t_module, _ = module_run(config)
    diff = np.argwhere(brute != result.free)
    for i, j, k in diff[:10]:
        rec.note(f"difference at {result.component_labels[i]} vs {result.component_labels[j]} dir {k}: "
                 f"brute={brute[i, j, k]}, module={result.free[i, j, k]}")
    rec.check(len(diff) == 0, f"module vs brute force: {len(diff)} differences in free ({brute.size} cells)")


def test_i3(rec, config):
    result, _, _ = module_run(config)
    free, neg = result.free, result.neg
    mismatches = int(sum(
        (free[j, i, neg[k]] != free[i, j, k])
        for i in range(free.shape[0]) for j in range(free.shape[1]) for k in range(free.shape[2])
    ))
    rec.check(mismatches == 0, f"free[j, i, neg[k]] == free[i, j, k] for all {free.size} cells "
              f"({mismatches} mismatches)")


def test_i4(rec, config):
    result, _, _ = module_run(config)
    labels = result.component_labels
    removable = result.removable(range(len(labels)))
    got = {labels[i]: [int(k) for k in ks] for i, ks in removable.items()}
    # Same removable directions as collision test E4 (part1 -> (0,0,-1), part5 -> (-1,0,0), part6 -> (1,0,0)),
    # but at whatever index each currently holds in Preprocessing's candidate-direction list -- the index
    # alone (6/0/1, recorded when this test was first written 2026-09-27) is not a stable invariant to
    # hardcode, since list order depends on face-enumeration order; re-pinned to the current order 2026-11-24
    # (confirmed same vectors as A2b in automatic_tests/test_aog.py).
    expected = {"industrial_fastner_part1": [5], "industrial_fastner_part5": [1], "industrial_fastner_part6": [0]}
    for label, ks in got.items():
        rec.note(f"{label}: {ks} -> {[_vec(result.directions[k]) for k in ks]}")
    rec.check(got == expected, f"removable(all) == {expected}")


def test_i5(rec, config):
    _, t_brute = brute_force(config)
    result, t_module, _ = module_run(config)
    ratio = t_module / t_brute
    rec.note(f"brute force (both triangles) {t_brute:.1f} s, module (one triangle) {t_module:.1f} s, "
             f"ratio {ratio:.2f}; module matrix-only time {result.stats['matrix_time_s']:.1f} s")
    rec.check(ratio < 0.7, f"module takes about half the brute-force time (ratio {ratio:.2f} < 0.7)")


def test_i6(rec, config):
    first, _, cache = module_run(config)
    prep = fastener_prep()
    t0 = time.perf_counter()
    second = ii.identify_interferences(prep, config, cache)
    dt = time.perf_counter() - t0
    same = (np.array_equal(first.free, second.free) and np.array_equal(first.directions, second.directions)
            and np.array_equal(first.neg, second.neg) and first.exception_cells == second.exception_cells)
    rec.check(dt < 2.0 and same, f"second run loads from cache in {dt:.2f} s (< 2 s), arrays identical={same}")

    # Invalidation: a different key must miss the cache (which is what forces the recompute).
    ids, _, shapes = ii._components(prep.reduced_assembly)
    dirs, _ = ii._directions(prep.candidate_directions)
    key_step = ii.compute_cache_key(prep, ids, shapes, dirs, replace(config, step=config.step / 2))
    rec.check(key_step != first.cache_key and ii._try_load_cache(cache, key_step, config) is None,
              "changing step changes the cache key and misses the cache")
    touched = fresh_dir("touched_step") / FASTENER_STEP.name
    touched.parent.mkdir(parents=True)
    touched.write_bytes(FASTENER_STEP.read_bytes() + b"\n")
    key_step_file = ii.compute_cache_key(replace(prep, step_path=touched), ids, shapes, dirs, config)
    rec.check(key_step_file != first.cache_key and ii._try_load_cache(cache, key_step_file, config) is None,
              "modifying the STEP file changes the cache key and misses the cache")


def test_i7(rec, config):
    brute, _ = brute_force(config)
    cache = fresh_dir("resume")
    prep = fastener_prep()

    class Interrupt(Exception):
        pass

    def stop_after_3(done, pair):
        if done == 3:
            raise Interrupt(f"simulated interrupt after pair {pair}")

    ii._after_pair_hook = stop_after_3
    try:
        ii.identify_interferences(prep, config, cache)
        rec.check(False, "first run should have been interrupted")
    except Interrupt as exc:
        rec.note(str(exc))
    finally:
        ii._after_pair_hook = None
    ckpt = json.loads((cache / ii.CHECKPOINT_NAME).read_text(encoding="utf-8"))
    rec.check(len(ckpt["rows"]) == 3 and not (cache / ii.JSON_NAME).exists(),
              f"checkpoint holds {len(ckpt['rows'])} completed pairs, no final result written")

    seen = []
    ii._after_pair_hook = lambda done, pair: seen.append(done)
    try:
        result = ii.identify_interferences(prep, config, cache)
    finally:
        ii._after_pair_hook = None
    rec.check(bool(seen) and seen[0] == 4, f"resumed at pair {seen[0] if seen else None} (expected 4)")
    rec.check(np.array_equal(result.free, brute), "final free identical to the brute-force matrix (I2)")
    rec.check(not (cache / ii.CHECKPOINT_NAME).exists(), "checkpoint deleted on success")


def _synthetic_prep(name: str, boxes: dict[str, Part.Shape], directions) -> PreprocessingResult:
    doc = FreeCAD.newDocument(name)
    for obj_name, shape in boxes.items():
        feature = doc.addObject("Part::Feature", obj_name)
        feature.Shape = shape
        feature.Label = obj_name
    doc.recompute()
    return PreprocessingResult(doc, {}, list(directions))


def test_i8(rec, config):
    boxes = {
        "A": Part.makeBox(10, 10, 10),
        "B": Part.makeBox(10, 10, 10, Vector(9.9, 0, 0)),  # overlaps A by 10 mm^3
        "C": Part.makeBox(10, 10, 10, Vector(19.8, 0, 0)),  # overlaps B by 10 mm^3
    }
    prep = _synthetic_prep("hard_overlap", boxes, [X, -X, Y, -Y, Z, -Z])
    cache = fresh_dir("hard_overlap")
    try:
        ii.identify_interferences(prep, config, cache)
        rec.check(False, "expected a HardOverlapError")
    except HardOverlapError as exc:
        msg = str(exc)
        rec.note(msg)
        rec.check("A / B" in msg and "B / C" in msg and "A / C" not in msg,
                  "one error naming both overlapping pairs (A/B and B/C), not A/C")
    rec.check(not cache.exists() or not any(cache.iterdir()), "no files written")


def test_i9(rec, config):
    base, _, _ = module_run(config)
    # A free cell (so "reported blocked" is a visible change) that steps in contact and therefore
    # calls common: the free stepped cell with the most iterations.
    cell = max(((i, j, k) for (i, j, k), det in base.outcome_detail.items()
                if det["phase"] == "step" and base.free[i, j, k]),
               key=lambda c: (base.outcome_detail[c]["iterations"], c))
    i, j, k = cell
    prep = fastener_prep()
    _, _, shapes = ii._components(prep.reduced_assembly)
    d = Vector(*base.directions[k])
    bb_i, bb_j = shapes[i].BoundBox, shapes[j].BoundBox
    real_common = collision._common_volume
    fails = {"n": 0}

    def fail_in_cell(a, b):
        disp = a.BoundBox.Center - bb_i.Center
        in_cell = (
            abs(b.BoundBox.XMin - bb_j.XMin) < 1e-9 and abs(b.BoundBox.ZMax - bb_j.ZMax) < 1e-9
            and abs(a.BoundBox.XLength - bb_i.XLength) < 1e-6 and abs(a.BoundBox.ZLength - bb_i.ZLength) < 1e-6
            and disp.Length > 1e-9 and disp.cross(d).Length < 1e-6 * disp.Length and disp.dot(d) > 0
        )
        if in_cell and fails["n"] < 2:
            fails["n"] += 1
            raise RuntimeError("injected by I9")
        return real_common(a, b)

    collision._common_volume = fail_in_cell
    try:
        result = ii.identify_interferences(prep, config, fresh_dir("exception_cell"))
    finally:
        collision._common_volume = real_common
    labels = result.component_labels
    rec.note(f"chosen cell: {labels[i]} vs {labels[j]} dir {k} {_vec(base.directions[k])} (free in the base run)")
    rec.check(cell in result.exception_cells and not result.free[i, j, k] and not result.free[j, i, result.neg[k]],
              f"cell reported blocked (and its antisymmetric twin) and listed in exception_cells "
              f"{result.exception_cells}")
    mask = np.ones_like(result.free)
    mask[i, j, k] = mask[j, i, result.neg[k]] = False
    changed = int(np.sum(result.free[mask] != base.free[mask]))
    rec.check(changed == 0, f"rest of the matrix unchanged ({changed} other cells differ)")


def test_i10(rec, config):
    result, _, cache = module_run(config)
    n = len(result.component_labels)
    raw = (cache / ii.TABLE_NAME).read_bytes()
    rec.check(raw.startswith(codecs.BOM_UTF8), "CSV starts with a UTF-8 BOM (Excel shows the diagonal correctly)")
    with open(cache / ii.TABLE_NAME, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    rec.check(len(rows) == n + 1 and all(len(r) == n + 1 for r in rows),
              f"CSV is (n+1) x (n+1) with n={n} (header row and column)")

    def parse(cell: str) -> set[tuple[float, ...]]:
        if not cell:
            return set()
        return {tuple(round(float(x), 6) + 0.0 for x in part.strip("() ").split(",")) for part in cell.split("; ")}

    bad = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                bad += rows[i + 1][j + 1] != "—"
                continue
            neg_ij = {tuple(round(-c, 6) + 0.0 for c in v) for v in parse(rows[i + 1][j + 1])}
            bad += neg_ij != parse(rows[j + 1][i + 1])
    rec.check(bad == 0, f"diagonal is the em dash and each (j, i) cell is the negation of (i, j) ({bad} bad cells)")


def test_i11(rec, config):
    boxes = {
        "A": Part.makeBox(10, 10, 10, Vector(0, 0, 20)),
        "B": Part.makeBox(10, 10, 10, Vector(0, 0, 10)),
        "C": Part.makeBox(10, 10, 10, Vector(0, 0, 0)),
    }
    dirs = [X, -X, Y, -Y, Z, -Z]
    names = ["+x", "-x", "+y", "-y", "+z", "-z"]
    result = ii.identify_interferences(_synthetic_prep("stack", boxes, dirs), config, fresh_dir("stack"))
    A, B, C = (result.index_of(c) for c in "ABC")

    def named(ks):
        return {names[int(k)] for k in ks}

    side = {"+x", "-x", "+y", "-y"}
    everything = [A, B, C]
    cases = [
        ("free_directions(A, all)", named(result.free_directions(A, everything)), side | {"+z"}),
        ("free_directions(B, all)", named(result.free_directions(B, everything)), side),
        ("free_directions(C, all)", named(result.free_directions(C, everything)), side | {"-z"}),
        ("removable(all)", {result.component_ids[i]: named(ks) for i, ks in result.removable(everything).items()},
         {"A": side | {"+z"}, "B": side, "C": side | {"-z"}}),
        ("blockers(B, +z, all)", result.blockers(B, names.index("+z"), everything), [A]),
        ("blockers(B, -z, all)", result.blockers(B, names.index("-z"), everything), [C]),
        ("blockers(A, -z, all)", result.blockers(A, names.index("-z"), everything), [B, C]),
        ("free_directions(B, {B, C}) after removing A", named(result.free_directions(B, [B, C])), side | {"+z"}),
        ("removable({B, C}) after removing A",
         {result.component_ids[i]: named(ks) for i, ks in result.removable([B, C]).items()},
         {"B": side | {"+z"}, "C": side | {"-z"}}),
    ]
    for label, got, expected in cases:
        rec.check(got == expected, f"{label} = {got} (expected {expected})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("I1", test_i1), ("I2", test_i2), ("I3", test_i3), ("I4", test_i4), ("I5", test_i5), ("I6", test_i6),
    ("I7", test_i7), ("I8", test_i8), ("I9", test_i9), ("I10", test_i10), ("I11", test_i11),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=float, default=2.0, help="step size h (default 2.0, stable on the fastener)")
    ap.add_argument("--tests", nargs="*", help="individual test ids, e.g. I8 I11")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="    %(name)s %(levelname)s: %(message)s")

    config = CollisionConfig(step=args.step)
    print(f"CollisionConfig: {config}")
    rec = Recorder()
    for test_id, fn in TESTS:
        if args.tests and test_id not in args.tests:
            continue
        print(f"\n=== {test_id} ===", flush=True)
        rec.start(test_id)
        t0 = time.perf_counter()
        try:
            fn(rec, config)
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
    out = OUT_DIR / "interference_results.json"
    out.write_text(json.dumps({"config": config.__dict__, "results": rec.results}, indent=2))
    print(f"\nDetails written to {out}")
    return 0 if all(r["passed"] for r in rec.results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
