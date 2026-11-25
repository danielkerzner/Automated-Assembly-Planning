"""Acceptance tests for Preprocessing Part 1, standard handling (S1-S13, docs/preprocessing-spec.md).

[SUPP] Automatic test file: not needed to run the pipeline. Module under test: `src/preprocessing.py`
(`process_standards`, `load_standard_categories`, `load_bom`). Every case builds a tiny synthetic FreeCAD
scene by hand (`Part.makeBox`/`makeCylinder`, no STEP file) and calls `process_standards` directly -- the
cheapest entry point for these cases, since none of them need candidate-direction discovery or a reduced
assembly (`run_preprocessing_on_document`'s Part 2). `CollisionConfig(step=1.0)` is used throughout (smaller
than the pipeline default of 5 mm) so the sweeps never step over any of these small features.

Standalone runner (pytest is not installed in environmentAAP). From repo root:

    python automatic_tests/test_preprocessing.py                # everything (seconds, no STEP/BOM files)
    python automatic_tests/test_preprocessing.py --tests S1 S5   # individual tests

Scratch output (S10's temporary BOM CSVs) goes to automatic_tests/output/preprocessing_test/; results to
automatic_tests/output/preprocessing_results.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import freecad  # noqa: E402,F401  (bootstrap for the conda build)

import FreeCAD  # noqa: E402
import Part  # noqa: E402
from FreeCAD import Vector  # noqa: E402

from collision import CollisionConfig  # noqa: E402
from preprocessing import PreprocessingConfig, load_standard_categories, process_standards  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
SCRATCH = OUT_DIR / "preprocessing_test"

Z = Vector(0, 0, 1)
CCFG = CollisionConfig(step=1.0)  # smaller than the 5 mm pipeline default: these parts are a few mm across


# ---------------------------------------------------------------------------
# Tiny test harness (same shape as test_aog.py / test_subassembly.py)
# ---------------------------------------------------------------------------


class Recorder:
    """Collects PASS/FAIL checks, notes and errors per test id for the summary table and the JSON file."""

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


class LogCapture(logging.Handler):
    """[SUPP] Context manager that records the log records of one named logger."""

    def __init__(self, logger_name: str):
        super().__init__(level=logging.INFO)
        self.logger_name = logger_name
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def __enter__(self):
        logging.getLogger(self.logger_name).addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger(self.logger_name).removeHandler(self)


def fresh_dir(name: str) -> Path:
    # [SUPP] Empty scratch folder under output/preprocessing_test/.
    path = SCRATCH / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------


def doc_with(name: str, parts: dict[str, "Part.Shape"]) -> "FreeCAD.Document":
    """[SUPP] A fresh FreeCAD document with one Part::Feature per (Name == Label, shape) entry."""
    doc = FreeCAD.newDocument(name)
    for part_name, shape in parts.items():
        obj = doc.addObject("Part::Feature", part_name)
        obj.Shape = shape
        obj.Label = part_name
    doc.recompute()
    return doc


def hex_prism(across_flats: float, height: float, base: Vector, axis_dir: Vector = Z) -> "Part.Shape":
    """[SUPP] A hex prism (nut blank) of the given across-flats distance, centred on `base`, extruded `height`
    along `axis_dir` from `base`."""
    r = across_flats / math.sqrt(3)  # circumradius of a regular hexagon from its across-flats (inradius*2)
    pts = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3  # flat-to-flat orientation (flats along +-x/+-y-ish)
        pts.append(Vector(base.x + r * math.cos(angle), base.y + r * math.sin(angle), base.z))
    pts.append(pts[0])
    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    return face.extrude(axis_dir * height)


def annulus(outer_r: float, inner_r: float, height: float, base: Vector, axis_dir: Vector = Z) -> "Part.Shape":
    """[SUPP] A round nut/washer blank: outer cylinder minus a concentric bore."""
    outer = Part.makeCylinder(outer_r, height, base, axis_dir)
    inner = Part.makeCylinder(inner_r, height * 1.2, base + axis_dir * (-height * 0.1), axis_dir)
    return outer.cut(inner)


def screw(shank_r: float, shank_len: float, head_r: float, head_len: float, base: Vector,
         axis_dir: Vector = Z) -> "Part.Shape":
    """[SUPP] A screw: a shank from `base` along `axis_dir`, topped by a wider head."""
    shank = Part.makeCylinder(shank_r, shank_len, base, axis_dir)
    head = Part.makeCylinder(head_r, head_len, base + axis_dir * shank_len, axis_dir)
    return shank.fuse(head).removeSplitter()


def cap_ring(outer_r: float, hole_r: float, height: float, base: Vector, axis_dir: Vector = Z) -> "Part.Shape":
    """[SUPP] A round, coaxial disc resting flush against a standard's end face (zero gap at rest), used only
    to give the P4 direction probe something asymmetric to react to. A full-revolution shape, so rotating
    whatever it rests against about the shared axis never changes their overlap (always 0) -- it cannot
    affect a P8 rotation-lock test, only the P4 translation probe."""
    return annulus(outer_r, hole_r, height, base, axis_dir)


def wedge(r_outer: float, r_inner: float, half_angle_deg: float, height: float, base: Vector,
         axis_dir: Vector = Z) -> "Part.Shape":
    """[SUPP] An annular pie-slice (radius r_inner..r_outer, angle +-half_angle_deg about axis_dir, centred on
    angle 0), for building a small notch or a small key that is NOT rotationally symmetric."""
    outer = Part.makeCylinder(r_outer, height, base, axis_dir, 2 * half_angle_deg).rotated(base, axis_dir, -half_angle_deg)
    if r_inner <= 0:
        return outer
    inner = Part.makeCylinder(r_inner, height, base, axis_dir, 2 * half_angle_deg).rotated(base, axis_dir, -half_angle_deg)
    return outer.cut(inner)


# ---------------------------------------------------------------------------
# S1 - screw head on a plate, shaft in a clearance hole
# ---------------------------------------------------------------------------


def _s1_scene(hole_gap: float):
    shaft_r = 3.0
    plate = Part.makeBox(40, 40, 10).cut(Part.makeCylinder(shaft_r + hole_gap, 12, Vector(20, 20, -1), Z))
    scr = screw(shaft_r, 15, 6, 4, Vector(20, 20, -5), Z)
    return doc_with(f"s1_{hole_gap}", {"plate": plate, "screw": scr})


def test_s1(rec, args):
    """S1 - screw head on a plate, shaft in a clearance hole.
    What: free side = outward (away from the plate); holds include the plate only if the hole's radial gap
    (hole radius - shaft radius) is below `perpendicular_displacement` (0.5 mm): a 0.3 mm gap makes the plate
    a hold (the lateral probe closes the gap), a 0.8 mm gap does not.
    """
    for gap, expect_hold in [(0.3, True), (0.8, False)]:
        doc = _s1_scene(gap)
        records, _, _ = process_standards(doc, {"screw"}, PreprocessingConfig(), collision_config=CCFG,
                                          bom_categories={"screw": "screw"})
        rec_ = records["screw"]
        rec.check(rec_.category == 1, f"gap={gap}: stays category 1 (screw), got {rec_.category}")
        rec.check(len(rec_.options) == 1, f"gap={gap}: one removal option (got {len(rec_.options)})")
        got_hold = "plate" in rec_.holds
        rec.check(got_hold == expect_hold,
                  f"gap={gap}: plate in holds={got_hold} (expected {expect_hold}), holds={rec_.holds}")
        if rec_.options:
            rec.note(f"gap={gap}: direction={rec_.options[0].direction}, blockers={rec_.options[0].blockers}")
        FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# S2 - screw in an undersized threaded hole (static overlap)
# ---------------------------------------------------------------------------


def test_s2(rec, args):
    """S2 - screw in an undersized threaded hole (static overlap): not blocked by its seat; seat is a hold.

    The seat's own (large, constant-along-the-engagement) static overlap would, on its own, make the P4
    direction probe ambiguous (a tiny +-0.1 mm shift barely perturbs it either way) -- a separate "stop" face
    the head rests against with zero gap resolves the direction cleanly, the same mechanism S1 uses.
    """
    shaft_r = 3.0
    seat = Part.makeBox(40, 40, 10).cut(Part.makeCylinder(shaft_r - 0.5, 12, Vector(20, 20, -1), Z))  # z [0, 10]
    stop = Part.makeBox(40, 40, 5, Vector(0, 0, 10)).cut(  # z [10, 15], clearance hole (not undersized)
        Part.makeCylinder(shaft_r + 0.5, 7, Vector(20, 20, 9), Z))
    scr = screw(shaft_r, 20, 6, 4, Vector(20, 20, -5), Z)  # shank z [-5, 15], head rests on `stop`, z [15, 19]
    doc = doc_with("s2", {"seat": seat, "stop": stop, "screw": scr})
    records, _, _ = process_standards(doc, {"screw"}, PreprocessingConfig(), collision_config=CCFG,
                                      bom_categories={"screw": "screw"})
    rec_ = records["screw"]
    rec.check(rec_.category == 1, f"stays category 1 (got {rec_.category})")
    rec.check("seat" in rec_.holds, f"seat is a hold (got {rec_.holds})")
    if rec_.options:
        rec.check("seat" not in rec_.options[0].blockers, f"seat is not a blocker (got {rec_.options[0].blockers})")
    FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# S3 - cover resting on a screw head
# ---------------------------------------------------------------------------


def test_s3(rec, args):
    """S3 - cover resting on a screw head: cover is a blocker, not a hold.

    The cover sits a couple of mm clear of the head at rest, not touching it: if it touched (zero gap), the
    P4 probe (+-0.1 mm) would see the escape side as newly blocked too (by the cover) and the demotion-to-
    category-3 "both sides blocked" path would fire instead of resolving a direction -- the cover's own
    (unconditional) blocking only needs to be caught by the later full sweep, which it comfortably is.
    """
    shaft_r = 3.0
    plate = Part.makeBox(40, 40, 10).cut(Part.makeCylinder(shaft_r + 1.0, 12, Vector(20, 20, -1), Z))
    scr = screw(shaft_r, 15, 6, 4, Vector(20, 20, -5), Z)  # head z [10, 14]
    cover = Part.makeBox(30, 30, 5, Vector(5, 5, 16))  # 2 mm clear of the head top (z = 14)
    doc = doc_with("s3", {"plate": plate, "screw": scr, "cover": cover})
    records, _, _ = process_standards(doc, {"screw"}, PreprocessingConfig(), collision_config=CCFG,
                                      bom_categories={"screw": "screw"})
    rec_ = records["screw"]
    rec.check(rec_.category == 1, f"stays category 1 (got {rec_.category})")
    rec.check("cover" not in rec_.holds, f"cover is not a hold (got {rec_.holds})")
    if rec_.options:
        rec.check("cover" in rec_.options[0].blockers,
                  f"cover is a blocker (got {rec_.options[0].blockers})")
    FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# S4 - screw whose both sides are blocked / both free
# ---------------------------------------------------------------------------


def test_s4(rec, args):
    """S4 - screw whose both sides are blocked, or both free: category 3 with a warning, either way."""
    # Both free: a screw floating in open space (no plate at all).
    with LogCapture("aap.preprocessing") as cap:
        doc = doc_with("s4_free", {"screw": screw(3, 15, 6, 4, Vector(0, 0, 0), Z)})
        records, _, _ = process_standards(doc, {"screw"}, PreprocessingConfig(), collision_config=CCFG,
                                          bom_categories={"screw": "screw"})
    rec.check(records["screw"].category == 3, f"both free: category 3 (got {records['screw'].category})")
    rec.check(any("both axis sides free or both blocked" in r.getMessage() for r in cap.records),
              "both free: warning logged")
    FreeCAD.closeDocument(doc.Name)

    # Both blocked: sandwiched tightly between two plates (touching both ends, no clearance anywhere).
    with LogCapture("aap.preprocessing") as cap:
        bottom = Part.makeBox(20, 20, 5, Vector(-10, -10, -5))
        top = Part.makeBox(20, 20, 5, Vector(-10, -10, 20))
        scr = screw(3, 15, 6, 5, Vector(0, 0, 0), Z)  # shank z[0,15], head z[15,20], flush against `top`
        doc = doc_with("s4_blocked", {"bottom": bottom, "top": top, "screw": scr})
        records, _, _ = process_standards(doc, {"screw"}, PreprocessingConfig(), collision_config=CCFG,
                                          bom_categories={"screw": "screw"})
    rec.check(records["screw"].category == 3, f"both blocked: category 3 (got {records['screw'].category})")
    rec.check(any("both axis sides free or both blocked" in r.getMessage() for r in cap.records),
              "both blocked: warning logged")
    FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# S5 / S6 - nut rotation lock (hex pocket vs. open air)
# ---------------------------------------------------------------------------


def _s5_s6_scene(with_pocket: bool):
    bore_r, across_flats, nut_h = 3.0, 10.0, 6.0
    scr = screw(bore_r - 0.3, 30, 6, 4, Vector(0, 0, 0), Z)  # threaded engagement: shank slightly > nut bore
    nut = hex_prism(across_flats, nut_h, Vector(0, 0, 10), Z).cut(
        Part.makeCylinder(bore_r, nut_h * 1.2, Vector(0, 0, 10 - nut_h * 0.1), Z))
    # A plain shank has the same cross-section along its whole engaged length, so the nut's P4 probe (+-0.1
    # mm) sees the same large overlap either way and cannot resolve a direction on its own; a cap resting
    # flush on the nut's far face (zero gap) gives it one clearly free side and one clearly blocked side,
    # without affecting the rotation-lock test (round and coaxial: rotating the nut about the shared axis
    # never changes their overlap).
    cap = cap_ring(across_flats * 0.7, bore_r + 0.2, 1.0, Vector(0, 0, 16), Z)
    parts = {"screw": scr, "nut": nut, "cap": cap}
    if with_pocket:
        # A housing with a matching hex pocket around the nut; none of 15/30/45 deg is a multiple of the
        # hex's own 60 deg symmetry, so rotating the nut by any of them collides with the pocket wall.
        block = Part.makeBox(30, 30, nut_h, Vector(-15, -15, 10))
        pocket = hex_prism(across_flats + 0.2, nut_h * 1.2, Vector(0, 0, 10 - nut_h * 0.1), Z)
        housing = block.cut(pocket)
        parts["housing"] = housing
    return doc_with(f"s5s6_{with_pocket}", parts)


def test_s5(rec, args):
    """S5 - nut in a hex pocket on a standard screw: locked; partner = the screw; AoG removes the screw first."""
    doc = _s5_s6_scene(with_pocket=True)
    records, _, _ = process_standards(doc, {"screw", "nut"}, PreprocessingConfig(), collision_config=CCFG,
                                      bom_categories={"screw": "screw", "nut": "nut"})
    rec_ = records.get("nut")
    rec.check(rec_ is not None and rec_.category == 2, f"stays category 2 (nut) (got {rec_})")
    if rec_ is not None:
        rec.check(rec_.partner == "screw", f"partner = screw (got {rec_.partner})")
        rec.check(rec_.nut_rule == "locked", f"nut_rule = locked (got {rec_.nut_rule})")
    FreeCAD.closeDocument(doc.Name)


def test_s6(rec, args):
    """S6 - nut free to turn on a standard screw: free; AoG has no screw-first path."""
    doc = _s5_s6_scene(with_pocket=False)
    records, _, _ = process_standards(doc, {"screw", "nut"}, PreprocessingConfig(), collision_config=CCFG,
                                      bom_categories={"screw": "screw", "nut": "nut"})
    rec_ = records.get("nut")
    rec.check(rec_ is not None and rec_.category == 2, f"stays category 2 (nut) (got {rec_})")
    if rec_ is not None:
        rec.check(rec_.partner == "screw", f"partner = screw (got {rec_.partner})")
        rec.check(rec_.nut_rule == "free", f"nut_rule = free (got {rec_.nut_rule})")
    FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# S7 - nut with nothing in its bore
# ---------------------------------------------------------------------------


def test_s7(rec, args):
    """S7 - nut with nothing in its bore: category 3 (demoted; no partner above pairing_min_fill)."""
    nut = annulus(5.0, 3.0, 6.0, Vector(0, 0, 0), Z)
    with LogCapture("aap.preprocessing") as cap:
        doc = doc_with("s7", {"nut": nut})
        records, _, _ = process_standards(doc, {"nut"}, PreprocessingConfig(), collision_config=CCFG,
                                          bom_categories={"nut": "nut"})
    rec.check(records["nut"].category == 3, f"demoted to category 3 (got {records['nut'].category})")
    rec.check(any("nothing occupies its bore" in r.getMessage() for r in cap.records),
              "demotion logged (P7)")
    FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# S8 - ring between two plates on a shaft
# ---------------------------------------------------------------------------


def test_s8(rec, args):
    """S8 - ring between two plates on a shaft: two options with different blocker sets; the shaft is not a
    blocker on either (its clearance bore clears it)."""
    ring = annulus(6.0, 4.0, 10.0, Vector(0, 0, 0), Z)  # z [0, 10]
    plate_below = Part.makeBox(30, 30, 5, Vector(-15, -15, -5))  # z [-5, 0], touches ring's bottom
    plate_above = Part.makeBox(30, 30, 5, Vector(-15, -15, 10))  # z [10, 15], touches ring's top
    shaft = Part.makeCylinder(3.5, 40, Vector(0, 0, -15), Z)  # passes through everything, clears the ring's bore
    doc = doc_with("s8", {"ring": ring, "below": plate_below, "above": plate_above, "shaft": shaft})
    records, _, _ = process_standards(doc, {"ring"}, PreprocessingConfig(), collision_config=CCFG,
                                      bom_categories={})
    rec_ = records["ring"]
    rec.check(rec_.category == 3 and len(rec_.options) == 2,
              f"category 3, two options (got category={rec_.category}, {len(rec_.options)} option(s))")
    blocker_sets = [opt.blockers for opt in rec_.options]
    rec.check(any(b == {"below"} for b in blocker_sets) and any(b == {"above"} for b in blocker_sets),
              f"one option blocked only by 'below', the other only by 'above' (got {blocker_sets})")
    rec.check(all("shaft" not in b for b in blocker_sets), f"shaft is not a blocker on either option "
              f"(got {blocker_sets})")
    FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# S9 - washer between nut and screw head
# ---------------------------------------------------------------------------


def test_s9(rec, args):
    """S9 - washer between nut and screw head: blockers on each side are the nut and the head (standards)."""
    bore_r = 3.0
    washer = annulus(6.0, bore_r + 0.3, 4.0, Vector(0, 0, 10), Z)  # z [10, 14]
    scr = screw(bore_r - 0.3, 10, 6, 4, Vector(0, 0, -4), Z)  # head z [-4, 0], shank z [0, 10]
    nut = annulus(6.0, bore_r - 0.3, 4.0, Vector(0, 0, 14), Z)  # z [14, 18], threaded onto the shank's top
    doc = doc_with("s9", {"washer": washer, "screw": scr, "nut": nut})
    records, _, _ = process_standards(doc, {"screw", "nut", "washer"}, PreprocessingConfig(), collision_config=CCFG,
                                      bom_categories={"screw": "screw", "nut": "nut"})
    rec_ = records.get("washer")
    rec.check(rec_ is not None and rec_.category == 3 and len(rec_.options) == 2,
              f"washer: category 3, two options (got {rec_})")
    if rec_ is not None:
        blocker_sets = [opt.blockers for opt in rec_.options]
        rec.check(any(b == {"screw"} for b in blocker_sets) and any(b == {"nut"} for b in blocker_sets),
                  f"one option blocked only by the screw, the other only by the nut (got {blocker_sets})")
    FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# S10 - BOM validation errors
# ---------------------------------------------------------------------------


def _write_bom(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_s10(rec, args):
    """S10 - BOM validation: both screw and nut set is an error; flags on a non-standard row are ignored
    (warning); a BOM without the screw/nut columns makes every standard category 3."""
    scratch = fresh_dir("s10")
    fieldnames = ["label", "is_standard", "screw", "nut"]

    both_bom = scratch / "both.csv"
    _write_bom(both_bom, fieldnames, [{"label": "s1", "is_standard": "true", "screw": "true", "nut": "true"}])
    try:
        load_standard_categories(both_bom)
        rec.check(False, "both screw and nut true should raise ValueError")
    except ValueError as exc:
        rec.check("sets both 'screw' and 'nut' true" in str(exc), f"error message matches (got: {exc})")

    ignored_bom = scratch / "ignored.csv"
    _write_bom(ignored_bom, fieldnames, [{"label": "p1", "is_standard": "false", "screw": "", "nut": "true"}])
    with LogCapture("aap.preprocessing") as cap:
        cats = load_standard_categories(ignored_bom)
    rec.check(cats == {}, f"flag on a non-standard row: no entry in the result (got {cats})")
    rec.check(any("is_standard' is not true; ignored" in r.getMessage() for r in cap.records),
              "flag on a non-standard row: warning logged")

    no_cols_bom = scratch / "no_cols.csv"
    _write_bom(no_cols_bom, ["label", "is_standard"], [{"label": "s1", "is_standard": "true"}])
    cats = load_standard_categories(no_cols_bom)
    rec.check(cats == {"s1": None}, f"no screw/nut columns: category None (got {cats})")

    scr = screw(3, 15, 6, 4, Vector(0, 0, 0), Z)
    doc = doc_with("s10", {"s1": scr})
    records, _, _ = process_standards(doc, {"s1"}, PreprocessingConfig(), collision_config=CCFG,
                                      bom_categories=cats)
    rec.check(records["s1"].category == 3, f"end-to-end: lands in category 3 (got {records['s1'].category})")
    FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# S11 - nut on a non-standard stud
# ---------------------------------------------------------------------------


def _s11_scene(with_pocket: bool):
    bore_r, across_flats, nut_h = 3.0, 10.0, 6.0
    stud = Part.makeCylinder(bore_r - 0.3, 30, Vector(0, 0, 0), Z)  # NOT flagged as a standard
    nut = hex_prism(across_flats, nut_h, Vector(0, 0, 10), Z).cut(
        Part.makeCylinder(bore_r, nut_h * 1.2, Vector(0, 0, 10 - nut_h * 0.1), Z))
    cap = cap_ring(across_flats * 0.7, bore_r + 0.2, 1.0, Vector(0, 0, 16), Z)  # see _s5_s6_scene
    parts = {"stud": stud, "nut": nut, "cap": cap}
    if with_pocket:
        block = Part.makeBox(30, 30, nut_h, Vector(-15, -15, 10))
        pocket = hex_prism(across_flats + 0.2, nut_h * 1.2, Vector(0, 0, 10 - nut_h * 0.1), Z)
        parts["housing"] = block.cut(pocket)
    return doc_with(f"s11_{with_pocket}", parts)


def test_s11(rec, args):
    """S11 - nut on a non-standard stud (free / locked): partner = the stud (movable), even though the stud
    is never flagged as a standard."""
    for with_pocket, expect_rule in [(False, "free"), (True, "locked")]:
        doc = _s11_scene(with_pocket)
        records, _, _ = process_standards(doc, {"nut"}, PreprocessingConfig(), collision_config=CCFG,
                                          bom_categories={"nut": "nut"})
        rec_ = records.get("nut")
        rec.check(rec_ is not None and rec_.category == 2, f"pocket={with_pocket}: stays category 2 (got {rec_})")
        if rec_ is not None:
            rec.check(rec_.partner == "stud", f"pocket={with_pocket}: partner = stud (got {rec_.partner})")
            rec.check(rec_.nut_rule == expect_rule,
                      f"pocket={with_pocket}: nut_rule = {expect_rule} (got {rec_.nut_rule})")
        FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# S12 - hex pocket belonging to the partner itself; round nut
# ---------------------------------------------------------------------------


def test_s12(rec, args):
    """S12 - hex pocket belonging to the partner itself, round nut: locked still detected (partner clipped
    only inside the bore); bore radius taken from the internal cylinder (P2's dominant-by-area axis would
    pick the nut's OUTER face, but P7's bore detection must independently find the inner one).

    A *perfectly* round nut can never be detected as rotation-locked: rotating a body of revolution about
    its own axis occupies exactly the same points, so its overlap with anything is rotation-invariant by
    construction, whatever the fixed geometry around it looks like. The nut's dominant face (by area, what
    P2 reads) is still its big outer cylinder, but it also carries one small notch -- small enough not to
    compete with that face's area -- aligned at rest with a key that belongs to the partner (survives the
    bore clip: its radius is well beyond bore_radius + bore_clip_margin) and sits in the notch's clearance
    only there; any rotation swings the notch away and brings the nut's plain round body into the key.
    """
    bore_r, outer_r, nut_h = 3.0, 8.0, 6.0  # outer face area >> inner face area: P2 picks the outer cylinder
    base = Vector(0, 0, 10)
    notch = wedge(outer_r + 1.0, outer_r - 2.0, 5.0, nut_h, base)  # radius [6, 9], angle +-5 deg, z [10, 16]
    nut = annulus(outer_r, bore_r, nut_h, base, Z).cut(notch)
    cap = cap_ring(outer_r + 1.0, bore_r + 0.2, 1.0, base + Vector(0, 0, nut_h), Z)  # resolves the P4 probe
    shank = Part.makeCylinder(bore_r - 0.3, 20, Vector(0, 0, 0), Z)  # z [0, 20], engages the nut's bore
    key = wedge(outer_r + 0.5, outer_r - 1.5, 3.0, 2.0, base + Vector(0, 0, 2))  # radius [6.5, 8.5], +-3 deg,
    scr = shank.fuse(key).removeSplitter()                                      # z [12, 14] -- inside the notch
    doc = doc_with("s12", {"screw": scr, "nut": nut, "cap": cap})
    records, _, _ = process_standards(doc, {"screw", "nut"}, PreprocessingConfig(), collision_config=CCFG,
                                      bom_categories={"screw": "screw", "nut": "nut"})
    rec_ = records.get("nut")
    rec.check(rec_ is not None and rec_.category == 2, f"stays category 2 (got {rec_})")
    if rec_ is not None:
        rec.check(rec_.partner == "screw", f"partner = screw, found via the inner bore face (got {rec_.partner})")
        rec.check(rec_.nut_rule == "locked", f"nut_rule = locked, via the partner's own hex collar "
                  f"(got {rec_.nut_rule})")
    FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# S13 - washer touching a nut's face
# ---------------------------------------------------------------------------


def test_s13(rec, args):
    """S13 - washer touching a nut's face: not a partner (does not occupy the bore)."""
    bore_r, across_flats, nut_h = 3.0, 10.0, 6.0
    scr = screw(bore_r - 0.3, 20, 6, 4, Vector(0, 0, 0), Z)
    nut = hex_prism(across_flats, nut_h, Vector(0, 0, 10), Z).cut(
        Part.makeCylinder(bore_r, nut_h * 1.2, Vector(0, 0, 10 - nut_h * 0.1), Z))
    # Washer stacked against the nut's top face; its own hole is wider than the nut's bore, so none of its
    # material falls inside the bore cylinder even though it axially overlaps the nut's band.
    washer = annulus(across_flats * 0.6, bore_r + 1.0, 2.0, Vector(0, 0, 16), Z)  # z [16, 18]
    doc = doc_with("s13", {"screw": scr, "nut": nut, "washer": washer})
    records, _, _ = process_standards(doc, {"screw", "nut", "washer"}, PreprocessingConfig(), collision_config=CCFG,
                                      bom_categories={"screw": "screw", "nut": "nut"})
    rec_ = records.get("nut")
    rec.check(rec_ is not None and rec_.category == 2, f"stays category 2 (got {rec_})")
    if rec_ is not None:
        rec.check(rec_.partner == "screw", f"partner is still the screw, not the washer (got {rec_.partner})")
    FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("S1", test_s1, "core"), ("S2", test_s2, "core"), ("S3", test_s3, "core"), ("S4", test_s4, "core"),
    ("S5", test_s5, "core"), ("S6", test_s6, "core"), ("S7", test_s7, "core"), ("S8", test_s8, "core"),
    ("S9", test_s9, "core"), ("S10", test_s10, "core"), ("S11", test_s11, "core"), ("S12", test_s12, "core"),
    ("S13", test_s13, "core"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="*", help="individual test ids, e.g. S1 S5")
    ap.add_argument("--groups", nargs="*", default=["core"], choices=["core"], help="core (default, only group)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="    %(name)s %(levelname)s: %(message)s")

    rec = Recorder()
    for test_id, fn, group in TESTS:
        if args.tests and test_id not in args.tests:
            continue
        if not args.tests and group not in args.groups:
            continue
        print(f"\n=== {test_id} ===", flush=True)
        rec.start(test_id)
        t0 = time.perf_counter()
        try:
            fn(rec, args)
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
    out = OUT_DIR / "preprocessing_results.json"
    out.write_text(json.dumps({"results": rec.results}, indent=2))
    print(f"\nDetails written to {out}")
    return 0 if all(r["passed"] for r in rec.results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
