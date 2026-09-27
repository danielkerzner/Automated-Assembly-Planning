"""Pre-implementation verification of the single-pair collision test (docs/collisiontest.md).

Exercises `is_free(i, j, d)` from src/collision.py on the FreeCAD / OpenCascade backend; the
interference matrix in src/interference_identification.py is built on top of it. Method: discrete stepping
with gap-skipping (the swept-volume method was abandoned, see the History section of the spec).
Groups A-D use hand-made Part.makeBox / Part.makeCylinder geometry whose expected result can be
checked by hand; group E runs on the industrial_fastner reduced assembly produced by
Preprocessing.

Standalone runner (pytest is not installed in environmentAAP). From repo root:

    python tests_scripts/test_collision.py                     # everything
    python tests_scripts/test_collision.py --groups A B C D     # subset of groups
    python tests_scripts/test_collision.py --tests E3 E6       # individual tests
    python tests_scripts/test_collision.py --step 0.5 --margin 1.0 --no-gap-skip

Prints a PASS/FAIL line per sub-check, then the results-log table from the spec. The same
table plus all details is written to tests_scripts/output/collision_results_step.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import freecad  # noqa: F401  (sets up the FreeCAD lib path in the conda env)
import FreeCAD
import Part
from FreeCAD import Vector

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


OUT_DIR = Path(__file__).resolve().parent / "output"
FASTENER_STEP = REPO_ROOT / "data" / "industrial_fastner" / "industrial_fastner.stp"
FASTENER_BOM = REPO_ROOT / "data" / "industrial_fastner" / "bom.csv"

# The collision test under verification is the production code in src/collision.py.
import collision  # noqa: E402  (module attribute access lets A5 replace the backend hooks)
from collision import (  # noqa: E402
    CollisionConfig,
    HardOverlapError,
    Outcome,
    Stats,
    _extent,
    broad_phase,
    classify_pair,
    is_free,
    is_free_detail,
    sweep_length,
    unit,
)


# ---------------------------------------------------------------------------
# Tiny test harness
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


def _fmt(v: Vector) -> str:
    return f"({v.x:.4g}, {v.y:.4g}, {v.z:.4g})"


def expect_free(rec: Recorder, name: str, si, sj, d, expected: bool, params: CollisionConfig, stats=None, threshold=None):
    out = is_free_detail(si, sj, d, params, stats, threshold)
    exp = "free" if expected else "blocked"
    got = "free" if out.free else "blocked"
    extra = f", blocked at t={out.t_block:.3f} vol={out.volume:.3e}" if out.t_block is not None else ""
    rec.check(
        out.free == expected,
        f"{name} along {_fmt(d)}: expected {exp}, got {got} "
        f"({out.phase}, L={out.sweep_length:.3f}, iters={out.iterations}{extra})",
    )
    return out


def box(dx, dy, dz, x=0.0, y=0.0, z=0.0) -> Part.Shape:
    return Part.makeBox(dx, dy, dz, Vector(x, y, z))


X, Y, Z = Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)


# ---------------------------------------------------------------------------
# Geometry fixtures (shared by groups B, C and D)
# ---------------------------------------------------------------------------


def fx_b1():
    return box(10, 10, 10, z=10), box(10, 10, 10)


def fx_b2():
    return box(10, 10, 10), box(10, 10, 10, x=10)


def fx_plate_with_hole():
    plate = box(40, 40, 10)
    hole = Part.makeCylinder(5, 10, Vector(20, 20, 0), Z)
    return plate.cut(hole)


def fx_pin(radius):
    return Part.makeCylinder(radius, 30, Vector(20, 20, -10), Z)


def fx_headed_pin():
    shaft = Part.makeCylinder(5, 20, Vector(20, 20, -10), Z)  # z in [-10, 10]
    head = Part.makeCylinder(8, 3, Vector(20, 20, 10), Z)  # rests on plate top, z in [10, 13]
    return shaft.fuse(head).removeSplitter()


def fx_c4():
    return box(0.5, 10, 10), box(0.4, 10, 10, x=5)


def fx_l_shape():
    arm_x = box(20, 10, 10)  # x in [0,20], y in [0,10]
    arm_y = box(10, 20, 10)  # x in [0,10], y in [0,20]
    l_shape = arm_x.fuse(arm_y).removeSplitter()
    nested = box(10, 10, 10, x=10, y=10)  # touches both inner faces of the L
    return l_shape, nested


def contact_cases() -> list[tuple[str, Part.Shape, Part.Shape, Vector, bool]]:
    """Every (config, moving, fixed, direction, expected_free) case from groups B and C.

    C4 is included only for the h = 0.5 row (blocked); callers that change h must skip it.
    """
    cases = []
    a, b = fx_b1()
    for d, exp in [(Z, True), (-Z, False), (X, True), (-X, True), (Y, True), (-Y, True)]:
        cases.append(("B1 A", a, b, d, exp))
    for d, exp in [(-Z, True), (Z, False)]:
        cases.append(("B1 B", b, a, d, exp))

    a, b = fx_b2()
    for d, exp in [(Y, True), (-Y, True), (Z, True), (-Z, True), (X, False)]:
        cases.append(("B2 A", a, b, d, exp))

    plate = fx_plate_with_hole()
    diag = unit(Vector(1, 0, 1))
    for tag, r in (("B3", 5.0), ("B4", 4.9)):
        pin = fx_pin(r)
        for d, exp in [(Z, True), (-Z, True), (X, False), (-X, False), (Y, False), (-Y, False), (diag, False)]:
            cases.append((f"{tag} pin", pin, plate, d, exp))

    headed = fx_headed_pin()
    cases.append(("B5 headed pin", headed, plate, Z, True))
    cases.append(("B5 headed pin", headed, plate, -Z, False))

    a, b = fx_b1()
    cases.append(("C1 A", a, b, Vector(1, -4.6e-15, 0), True))
    cases.append(("C2 A tilt into B", a, b, unit(Vector(1, 0, -1e-3)), False))
    cases.append(("C2 A tilt away", a, b, unit(Vector(1, 0, 1e-3)), True))

    cases.append(("C3 A vs thin plate", box(10, 10, 10), box(0.1, 10, 10, x=100), X, False))

    a, b = fx_c4()
    cases.append(("C4 thin plates (h=0.5)", a, b, X, False))

    l_shape, nested = fx_l_shape()
    for d, exp in [(-X, True), (-Y, True), (X, False), (Y, False), (Z, True), (-Z, True)]:
        cases.append(("C7 L", l_shape, nested, d, exp))
    return cases


def _at_step(params: CollisionConfig, h: float) -> CollisionConfig:
    return replace(params, step=h)


# ---------------------------------------------------------------------------
# Group A: backend behaviour
# ---------------------------------------------------------------------------


def test_a1(rec: Recorder, params: CollisionConfig):
    v = FreeCAD.Version()
    rec.note(f"FreeCAD version: {'.'.join(v[:3])} ({v[3]})")
    rec.check(hasattr(Part, "makeBox"), "import freecad; import FreeCAD, Part succeeded")


def test_a2(rec, params):
    b = box(10, 10, 10)
    s = b.translated(Vector(5, 0, 0))
    rec.check(abs(b.BoundBox.XMin) < 1e-12, f"original unchanged: XMin={b.BoundBox.XMin}")
    rec.check(abs(s.BoundBox.XMin - 5) < 1e-12, f"translated copy: XMin={s.BoundBox.XMin}")


def test_a3(rec, params):
    a = box(10, 10, 10)
    d_apart = a.distToShape(box(10, 10, 10, x=15))[0]
    d_touch = a.distToShape(box(10, 10, 10, x=10))[0]
    d_over = a.distToShape(box(10, 10, 10, x=5))[0]
    rec.check(abs(d_apart - 5.0) < 1e-9, f"5 mm apart: distance={d_apart}")
    rec.check(abs(d_touch) < 1e-7, f"touching: distance={d_touch:.3e}")
    rec.check(abs(d_over) < 1e-7, f"overlapping: distance={d_over:.3e}")


def test_a4(rec, params):
    a = box(10, 10, 10)
    v_touch = a.common(box(10, 10, 10, x=10)).Volume
    v_over = a.common(box(10, 10, 10, x=9)).Volume
    rec.check(v_touch <= params.volume_epsilon, f"touching: common volume={v_touch:.3e}")
    err = abs(v_over - 100) / 100
    rec.check(err < 1e-9, f"1 mm overlap: volume={v_over:.12f}, rel err={err:.2e}")


class _FailingHook:
    """Replaces a backend hook; raises on the call numbers in `fail_on` (1-based)."""

    def __init__(self, real, fail_on, every: int | None = None):
        self.real, self.fail_on, self.every, self.n, self.raised = real, set(fail_on), every, 0, 0

    def __call__(self, a, b):
        self.n += 1
        if self.n in self.fail_on or (self.every and self.n % self.every == 0):
            self.raised += 1
            raise RuntimeError("StdFail_NotDone: (injected by A5)")
        return self.real(a, b)


def test_a5(rec, params):
    cases = [c for c in contact_cases() if c[0].startswith("B")]
    baseline = [is_free(si, sj, d, params) for _, si, sj, d, _ in cases]

    # 1. distToShape raises on every 3rd call across B1-B5: results unchanged, every one logged.
    real_dist = collision._distance
    hook = _FailingHook(real_dist, (), every=3)
    collision._distance = hook
    stats = Stats()
    try:
        forced = [is_free(si, sj, d, params, stats) for _, si, sj, d, _ in cases]
    finally:
        collision._distance = real_dist
    logged = sum(e["call"] == "distToShape" for e in stats.exceptions)
    rec.check(forced == baseline, f"B1-B5 with distToShape failing every 3rd call: results identical "
              f"({sum(a != b for a, b in zip(forced, baseline))} of {len(cases)} differ)")
    rec.check(hook.raised > 0 and logged == hook.raised,
              f"every forced distToShape exception logged ({logged} logged / {hook.raised} raised)")

    # 2. common raises once: the retry at t + h/100 succeeds and the result is unchanged.
    real_common = collision._common_volume
    for name, (si, sj), d, exp in [("B3 pin along bore", (fx_pin(5.0), fx_plate_with_hole()), Z, True), ("B2 A into B", fx_b2(), X, False)]:
        stats = Stats()
        collision._common_volume = _FailingHook(real_common, {1})
        try:
            out = is_free_detail(si, sj, d, params, stats)
        finally:
            collision._common_volume = real_common
        exc = [e for e in stats.exceptions if e["call"] == "common"]
        rec.check(out.free == exp and not out.exception_cell and len(exc) == 1 and exc[0]["retry_ok"],
                  f"{name} {_fmt(d)}: common fails once -> retry ok={exc[0]['retry_ok'] if exc else None}, "
                  f"result {'free' if out.free else 'blocked'} (expected {'free' if exp else 'blocked'})")

    # 3. common raises twice: the cell is reported blocked and listed as an exception cell.
    # (B3 pin along its bore: free, and not settled by the broad phase, so common is called.)
    si, sj = fx_pin(5.0), fx_plate_with_hole()
    stats = Stats()
    collision._common_volume = _FailingHook(real_common, {1, 2})
    try:
        out = is_free_detail(si, sj, Z, params, stats)
    finally:
        collision._common_volume = real_common
    exc = [e for e in stats.exceptions if e["call"] == "common"]
    rec.check(not out.free and out.exception_cell and len(exc) == 2 and exc[0]["retry_ok"] is False,
              f"B3 pin along bore (normally free) {_fmt(Z)}: common fails twice -> "
              f"{'free' if out.free else 'blocked'}, exception_cell={out.exception_cell}, logged={len(exc)}")


# ---------------------------------------------------------------------------
# Group B: contact semantics
# ---------------------------------------------------------------------------


def _run_cases(rec: Recorder, params: CollisionConfig, prefix: str):
    for name, si, sj, d, exp in contact_cases():
        if name.startswith(prefix):
            expect_free(rec, name, si, sj, d, exp, params)


def test_b1(rec, params):
    _run_cases(rec, params, "B1")


def test_b2(rec, params):
    _run_cases(rec, params, "B2")


def test_b3(rec, params):
    _run_cases(rec, params, "B3")


def test_b4(rec, params):
    _run_cases(rec, params, "B4")


def test_b5(rec, params):
    _run_cases(rec, params, "B5")


# ---------------------------------------------------------------------------
# Group C: stepping and numerical robustness
# ---------------------------------------------------------------------------


def test_c1(rec, params):
    _run_cases(rec, params, "C1")


def test_c2(rec, params):
    a, b = fx_b1()
    d_in = unit(Vector(1, 0, -1e-3))
    for h in (0.5, 1.0):
        out = expect_free(rec, f"C2 tilt into B (h={h})", a, b, d_in, False, _at_step(params, h))
        if out.t_block is not None:
            # A's x-shift at travel t is t*d.x; the spec formula uses the shift along x.
            s = out.t_block * d_in.x
            formula = 0.01 * s * (10 - s)
            rec.note(f"h={h}: first blocking t={out.t_block:.4f}, volume={out.volume:.6f}, "
                     f"formula 0.01*t*(10-t)={formula:.6f}")
    expect_free(rec, "C2 tilt away", a, b, unit(Vector(1, 0, 1e-3)), True, params)


def test_c3(rec, params):
    stats = Stats()
    out = expect_free(rec, "C3 A vs thin plate", box(10, 10, 10), box(0.1, 10, 10, x=100), X, False, params, stats)
    rec.note(f"iterations={out.iterations}, distToShape calls={stats.dists}, common calls={stats.commons}, "
             f"gap travel={stats.gap_travel:.2f} mm")
    rec.check(out.iterations <= 10, f"gap skipping keeps iterations small ({out.iterations} <= 10)")


def test_c4(rec, params):
    a, b = fx_c4()
    for h, exp in [(0.25, False), (0.5, False), (0.8, False), (1.0, True), (2.0, True)]:
        label = "blocked" if not exp else "missed (reported free, expected limitation)"
        expect_free(rec, f"C4 h={h} -> {label}", a, b, X, exp, replace(params, step=h, gap_skip=True))


def test_c5(rec, params):
    a = box(10, 10, 10)
    behind = box(10, 10, 10, x=-30)
    stats = Stats()
    out = is_free_detail(a, behind, X, params, stats, threshold=params.volume_epsilon)
    rec.check(out.free and out.phase == "broad" and stats.commons == 0 and stats.dists == 0,
              f"B behind A: free by broad phase, common calls={stats.commons}, distToShape calls={stats.dists}")

    front = box(10, 10, 10, x=30, y=5)
    for d in (X, unit(Vector(1, 0.2, 0.1))):
        L = sweep_length(a, front, d, params.margin)
        a_min, _ = _extent(a.translated(d * L).BoundBox, d)
        _, b_max = _extent(front.BoundBox, d)
        rec.check(a_min > b_max, f"{_fmt(d)}: at t=L={L:.3f}, A.min={a_min:.3f} > B.max={b_max:.3f}")
    expect_free(rec, "C5 B in front, half-overlapping in projection", a, front, X, False, params)


def _d2_pairs(n_pairs: int = 200, seed: int = 12345):
    rng = random.Random(seed)
    pairs = []
    for _ in range(n_pairs):
        si, sj = _random_solid(rng), _random_solid(rng)
        d = unit(Vector(rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))
        pairs.append((si, sj, d))
    return pairs


def overlap_window(si, sj, d, threshold: float, margin: float, fine: float) -> float:
    """Longest contiguous travel interval (fine-step scan) where the overlap exceeds threshold."""
    length = sweep_length(si, sj, d, margin)
    best = run = 0
    t = 0.0
    while t <= length:
        if collision._common_volume(si.translated(d * t), sj) > threshold:
            run += 1
            best = max(best, run)
        else:
            run = 0
        t += fine
    return best * fine


def _c6_mismatch(rec, label, si, sj, d, g, p, thr, h, margin) -> bool:
    """Log a mismatch; return True if it is a real failure (window >= h)."""
    window = overlap_window(si, sj, d, thr, margin, h / 50)
    explained = window < h
    rec.note(f"mismatch {label} {_fmt(d)}: gap-skip={g}, plain={p}, overlap window={window:.3f} mm "
             f"({'< h, known limitation' if explained else '>= h, REAL FAILURE'})")
    return not explained


def test_c6(rec, params):
    h = 0.5
    p_gap = replace(params, step=h, gap_skip=True)
    p_plain = replace(params, step=h, gap_skip=False)
    s_gap, s_plain = Stats(), Stats()
    mismatches = real = 0
    for name, si, sj, d, _ in contact_cases():
        g = is_free(si, sj, d, p_gap, s_gap)
        p = is_free(si, sj, d, p_plain, s_plain)
        if g != p:
            mismatches += 1
            _, _, thr = classify_pair(si, sj, params)
            real += _c6_mismatch(rec, name, si, sj, d, g, p, thr, h, params.margin)
    rec.note(f"B/C cases: iterations gap-skip={s_gap.iterations}, plain={s_plain.iterations}")

    r_gap, r_plain = Stats(), Stats()
    skipped = 0
    for k, (si, sj, d) in enumerate(_d2_pairs()):
        try:
            _, _, thr = classify_pair(si, sj, params)
        except HardOverlapError:
            skipped += 1
            continue
        g = is_free(si, sj, d, p_gap, r_gap, thr)
        p = is_free(si, sj, d, p_plain, r_plain, thr)
        if g != p:
            mismatches += 1
            real += _c6_mismatch(rec, f"D2 pair {k}", si, sj, d, g, p, thr, h, params.margin)
    rec.note(f"D2 set: {skipped} pairs skipped as hard overlaps; iterations gap-skip={r_gap.iterations}, "
             f"plain={r_plain.iterations}")
    rec.check(real == 0, f"every gap-skip/plain mismatch has an overlap window < h "
              f"({mismatches} mismatches, {real} real failures)")


def test_c7(rec, params):
    _run_cases(rec, params, "C7")


# ---------------------------------------------------------------------------
# Group D: structural properties and overlap rule
# ---------------------------------------------------------------------------


def test_d1(rec, params):
    cases = contact_cases()
    mismatches = 0
    for name, si, sj, d, _ in cases:
        p = _at_step(params, 0.5) if name.startswith("C4") else params
        fwd = is_free(si, sj, d, p)
        bwd = is_free(sj, si, -d, p)
        if fwd != bwd:
            mismatches += 1
            rec.check(False, f"{name} {_fmt(d)}: is_free(i,j,d)={fwd} but is_free(j,i,-d)={bwd}")
    rec.check(mismatches == 0, f"antisymmetry over {len(cases)} cases, mismatches={mismatches}")


def _random_solid(rng: random.Random) -> Part.Shape:
    if rng.random() < 0.5:
        s = Part.makeBox(rng.uniform(2, 15), rng.uniform(2, 15), rng.uniform(2, 15))
    else:
        s = Part.makeCylinder(rng.uniform(1, 7), rng.uniform(2, 20))
    axis = unit(Vector(rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))
    rot = FreeCAD.Rotation(axis, rng.uniform(0, 360))
    pos = Vector(rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-20, 20))
    s.Placement = FreeCAD.Placement(pos, rot)
    return s


def test_d2(rec, params):
    stepping_only = replace(params, use_broad_phase=False)
    pairs = _d2_pairs()
    broad_hits = contradictions = 0
    for k, (si, sj, d) in enumerate(pairs):
        if broad_phase(si, sj, d) is None:
            continue
        broad_hits += 1
        if not is_free(si, sj, d, stepping_only):
            contradictions += 1
            rec.check(False, f"pair {k}: broad phase says free, stepping says blocked (d={_fmt(d)})")
    rec.note(f"{broad_hits}/{len(pairs)} pairs settled by broad phase")
    rec.check(contradictions == 0, f"broad phase never contradicts stepping (contradictions={contradictions})")


def test_d3(rec, params):
    a = box(10, 10, 10)
    try:
        kind, v0, _ = classify_pair(a, box(10, 10, 10, x=9.9), params)
        rec.check(False, f"hard error expected, got {kind} (V0={v0:.4f})")
    except HardOverlapError as exc:
        rec.check(True, f"0.1 mm overlap raises hard error: {exc}")

    b = box(10, 10, 10, x=9.995)
    kind, v0, thr = classify_pair(a, b, params)
    rec.check(kind == "tolerance" and abs(v0 - 0.5) < 1e-6 and abs(thr - 1.0) < 1e-6,
              f"0.005 mm overlap: kind={kind}, V0={v0:.6f}, threshold={thr:.6f}")
    for d in (Y, -Y, Z, -Z):
        expect_free(rec, "D3 tolerance pair, sliding", a, b, d, True, params, threshold=thr)
    expect_free(rec, "D3 tolerance pair, away", a, b, -X, True, params, threshold=thr)
    expect_free(rec, "D3 tolerance pair, into B", a, b, X, False, params, threshold=thr)

    a2, b2 = fx_b2()
    kind, v0, _ = classify_pair(a2, b2, params)
    rec.check(kind == "normal", f"B2 touching: kind={kind}, V0={v0:.3e}")


# ---------------------------------------------------------------------------
# Group E: real geometry (industrial_fastner reduced assembly)
# ---------------------------------------------------------------------------

_E_CACHE: dict = {}


def _fastener():
    if "shapes" not in _E_CACHE:
        from preprocessing import _leaf_components, run_preprocessing

        print("  (re-running preprocessing on the corrected assembly)", flush=True)
        result = run_preprocessing(FASTENER_STEP, FASTENER_BOM)
        comps = _leaf_components(result.reduced_assembly)
        _E_CACHE["labels"] = [o.Label for o in comps]
        _E_CACHE["shapes"] = [o.Shape.copy() for o in comps]
        _E_CACHE["dirs"] = [unit(d) for d in result.candidate_directions]
    return _E_CACHE["labels"], _E_CACHE["shapes"], _E_CACHE["dirs"]


def _pair_classes(params: CollisionConfig):
    """{(i, j): (kind, V0, threshold)} for i < j; kind 'hard' has threshold None."""
    if "classes" not in _E_CACHE:
        labels, shapes, _ = _fastener()
        classes = {}
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                try:
                    classes[(i, j)] = classify_pair(shapes[i], shapes[j], params)
                except HardOverlapError:
                    classes[(i, j)] = ("hard", shapes[i].common(shapes[j]).Volume, None)
        _E_CACHE["classes"] = classes
    return _E_CACHE["classes"]


def _neg_index(dirs: list[Vector], tol: float = 1e-6) -> list[int | None]:
    neg = []
    for d in dirs:
        match = [k for k, e in enumerate(dirs) if (e + d).Length < tol]
        neg.append(match[0] if match else None)
    return neg


def _full_matrix(params: CollisionConfig):
    """Brute-force n x n x D matrix (both triangles). Cached per (h, gap_skip, margin)."""
    key = ("matrix", params.step, params.gap_skip, params.margin)
    if key not in _E_CACHE:
        labels, shapes, dirs = _fastener()
        classes = _pair_classes(params)
        n, D = len(shapes), len(dirs)
        stats = Stats()
        B = [[[None] * D for _ in range(n)] for _ in range(n)]
        exc_cells = []  # (i, j, k): common failed twice, cell reported blocked
        errors = []
        t_all = time.perf_counter()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                kind, _, thr = classes[(min(i, j), max(i, j))]
                if kind == "hard":
                    continue
                t0 = time.perf_counter()
                for k, d in enumerate(dirs):
                    n_exc = len(stats.exceptions)
                    try:
                        out = is_free_detail(shapes[i], shapes[j], d, params, stats, thr)
                        B[i][j][k] = out.free
                        if out.exception_cell:
                            exc_cells.append((i, j, k))
                        for e in stats.exceptions[n_exc:]:
                            e.update(pair=f"{labels[i]} vs {labels[j]}", k=k)
                    except Exception as exc:  # noqa: BLE001  (left as None, reported by E1)
                        errors.append(f"{labels[i]} vs {labels[j]} dir {k} {_fmt(d)}: {type(exc).__name__}: {exc}")
                print(f"    [h={params.step}] {labels[i]} vs {labels[j]}: {time.perf_counter() - t0:.1f} s",
                      flush=True)
        _E_CACHE[key] = (B, stats, errors, time.perf_counter() - t_all)
        _E_CACHE[("exc_cells",) + key[1:]] = exc_cells
    return _E_CACHE[key]


def test_e1(rec, params):
    labels, shapes, dirs = _fastener()
    rec.note(f"{len(shapes)} reduced components x {len(dirs)} candidate directions, h={params.step}")
    _, stats, errors, total = _full_matrix(params)
    for e in errors:
        rec.note(f"exception: {e}")
    for e in stats.exceptions:
        rec.note(f"caught backend exception: {e}")
    rec.check(not errors, f"all pairs x directions ran without exceptions ({len(errors)} errors, {total:.1f} s)")


def test_e2(rec, params):
    labels, shapes, dirs = _fastener()
    neg = _neg_index(dirs)
    missing = [k for k, m in enumerate(neg) if m is None]
    rec.check(not missing, f"every candidate direction has its negation in the list (missing: {missing})")
    B, _, _, _ = _full_matrix(params)
    n, mismatches, compared = len(shapes), 0, 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(len(dirs)):
                if neg[k] is None or B[i][j][k] is None or B[j][i][neg[k]] is None:
                    continue
                compared += 1
                if B[j][i][neg[k]] != B[i][j][k]:
                    mismatches += 1
                    rec.note(f"mismatch: {labels[i]} vs {labels[j]} dir {k} {_fmt(dirs[k])}: "
                             f"B[i,j,k]={B[i][j][k]}, B[j,i,neg]={B[j][i][neg[k]]}")
    rec.check(mismatches == 0, f"antisymmetry B[j,i,neg[k]] == B[i,j,k]: {mismatches} mismatches / {compared}")


def test_e3(rec, params):
    labels, _, _ = _fastener()
    hard = 0
    for (i, j), (kind, v0, thr) in _pair_classes(params).items():
        if kind != "normal":
            thr_txt = f", threshold={thr:.4f}" if thr is not None else ""
            rec.note(f"{kind}: {labels[i]} / {labels[j]} V0={v0:.4f} mm^3{thr_txt}")
        hard += kind == "hard"
    rec.check(hard == 0, f"no hard-error overlaps (found {hard})")


def test_e4(rec, params):
    labels, shapes, dirs = _fastener()
    B, _, _, _ = _full_matrix(params)
    n = len(shapes)
    for i in range(n):
        partners = [labels[j] for j in range(n) if j != i and any(x is True for x in B[i][j])]
        rec.check(bool(partners), f"{labels[i]} can separate from at least one partner ({len(partners)}/{n - 1})")
    for i in range(n):
        for j in range(n):
            if i != j and not any(x is True for x in B[i][j]):
                rec.note(f"no free direction: {labels[i]} moving vs {labels[j]}")
    for i in range(n):
        free_dirs = [k for k in range(len(dirs)) if all(B[i][j][k] is True for j in range(n) if j != i)]
        rec.note(f"{labels[i]}: globally free directions = {free_dirs}")


def test_e5(rec, params):
    _, stats, _, total = _full_matrix(params)
    t, its = stats.times, stats.iters_per_test
    share = stats.broad_free / stats.calls if stats.calls else 0.0
    rec.note(f"tests={stats.calls}, broad-phase settled={stats.broad_free} ({share:.1%}), stepped={len(t)}")
    if t:
        rec.note(f"stepped test time: mean={1000 * sum(t) / len(t):.1f} ms, max={1000 * max(t):.1f} ms, "
                 f"matrix total={total:.1f} s")
        rec.note(f"iterations per stepped test: mean={sum(its) / len(its):.1f}, max={max(its)}; "
                 f"common calls={stats.commons}, distToShape calls={stats.dists}")
    if stats.travel:
        rec.note(f"share of travel covered by gap jumps: {stats.gap_travel / stats.travel:.1%}")
    rec.check(True, "timing recorded")


def test_e6(rec, params):
    labels, shapes, dirs = _fastener()
    steps = [2.0, 1.0, 0.5, 0.25]
    mats = {}
    for h in steps:
        B, stats, errors, total = _full_matrix(_at_step(params, h))
        mats[h] = B
        rec.note(f"h={h}: matrix time={total:.1f} s, iterations={stats.iterations}, "
                 f"uncaught={len(errors)}, caught backend exceptions={len(stats.exceptions)}")
        for e in errors + stats.exceptions:
            rec.note(f"h={h} exception: {e}")
    n, D = len(shapes), len(dirs)
    changes = {}
    for a, b in zip(steps, steps[1:]):
        cells = [(i, j, k) for i in range(n) for j in range(n) if i != j for k in range(D)]
        # A cell that raised at either h (None) is an exception, not a step-size change.
        exc = set(_E_CACHE.get(("exc_cells", a, params.gap_skip, params.margin), [])) | set(
            _E_CACHE.get(("exc_cells", b, params.gap_skip, params.margin), []))
        errored = [c for c in cells if c in exc
                   or mats[a][c[0]][c[1]][c[2]] is None or mats[b][c[0]][c[1]][c[2]] is None]
        diff = [
            (labels[i], labels[j], k) for i, j, k in cells
            if (i, j, k) not in errored and mats[a][i][j][k] != mats[b][i][j][k]
        ]
        changes[(a, b)] = len(diff)
        rec.note(f"h {a} -> {b}: {len(diff)} cells changed, {len(errored)} cells skipped (exception)"
                 + (f", e.g. {diff[:3]}" if diff else ""))
    # h is stable if every refinement below it (h -> h/2 -> ...) changes nothing. The smallest h
    # has no refinement to compare against, so it can never be confirmed stable.
    stable = [
        a for a in steps[:-1]
        if all(changes[(x, y)] == 0 for x, y in zip(steps, steps[1:]) if x <= a)
    ]
    chosen = max(stable) if stable else None
    rec.note(f"largest h after which the matrix no longer changes: {chosen}")
    rec.check(chosen is not None, "a stable step size exists within the tested range")


def test_e6c(rec, params):
    """Re-run the E6 exception cell (part2 vs part3 along +y, h = 0.25) with the fallback in place."""
    labels, shapes, dirs = _fastener()
    i, j = labels.index("industrial_fastner_part2"), labels.index("industrial_fastner_part3")
    k = next(n for n, e in enumerate(dirs) if (e - Y).Length < 1e-6)
    kn = _neg_index(dirs)[k]
    _, _, thr = classify_pair(shapes[i], shapes[j], params)
    results = {}
    for h in (0.5, 0.25):
        for tag, si, sj, d in (("fwd", shapes[i], shapes[j], dirs[k]), ("bwd", shapes[j], shapes[i], dirs[kn])):
            stats = Stats()
            out = is_free_detail(si, sj, d, _at_step(params, h), stats, thr)
            results[(h, tag)] = out
            blk = f", blocked at t={out.t_block:.3f} vol={out.volume:.3f}" if out.t_block is not None else ""
            rec.note(f"h={h} {tag} {_fmt(d)}: {'free' if out.free else 'blocked'}{blk}, iters={out.iterations}, "
                     f"exception_cell={out.exception_cell}, caught={stats.exceptions}")
    rec.check(all(not o.exception_cell for o in results.values()), "no exception cell with the fallback")
    rec.check(results[(0.25, "fwd")].free == results[(0.5, "fwd")].free,
              f"h=0.25 matches h=0.5 (both {'free' if results[(0.5, 'fwd')].free else 'blocked'})")
    rec.check(results[(0.25, "fwd")].free == results[(0.25, "bwd")].free, "antisymmetry holds at h=0.25")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("A1", test_a1), ("A2", test_a2), ("A3", test_a3), ("A4", test_a4), ("A5", test_a5),
    ("B1", test_b1), ("B2", test_b2), ("B3", test_b3), ("B4", test_b4), ("B5", test_b5),
    ("C1", test_c1), ("C2", test_c2), ("C3", test_c3), ("C4", test_c4), ("C5", test_c5),
    ("C6", test_c6), ("C7", test_c7),
    ("D1", test_d1), ("D2", test_d2), ("D3", test_d3),
    ("E1", test_e1), ("E2", test_e2), ("E3", test_e3), ("E4", test_e4), ("E5", test_e5), ("E6", test_e6),
    ("E6c", test_e6c),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("step",), default="step")
    ap.add_argument("--step", type=float, default=CollisionConfig.step, help="step size h")
    ap.add_argument("--no-gap-skip", action="store_true")
    ap.add_argument("--margin", type=float, default=CollisionConfig.margin)
    ap.add_argument("--epsilon", type=float, default=CollisionConfig.volume_epsilon)
    ap.add_argument("--groups", nargs="*", default=list("ABCDE"))
    ap.add_argument("--tests", nargs="*", help="individual test ids, e.g. E3 E6 (overrides --groups)")
    args = ap.parse_args()

    params = CollisionConfig(step=args.step, volume_epsilon=args.epsilon, margin=args.margin,
                    gap_skip=not args.no_gap_skip)
    print(f"CollisionConfig: {params}")
    rec = Recorder()
    for test_id, fn in TESTS:
        if args.tests:
            if test_id not in args.tests:
                continue
        elif test_id[0] not in args.groups:
            continue
        print(f"\n=== {test_id} ===", flush=True)
        rec.start(test_id)
        t0 = time.perf_counter()
        try:
            fn(rec, params)
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
    out = OUT_DIR / f"collision_results_{args.method}.json"
    out.write_text(json.dumps({"params": params.__dict__, "results": rec.results}, indent=2))
    print(f"\nDetails written to {out}")
    return 0 if all(r["passed"] for r in rec.results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
