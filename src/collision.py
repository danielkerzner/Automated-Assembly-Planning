"""Single-pair collision test: can part i translate along d all the way past part j?

Method: discrete stepping with gap-skipping:

    0. Static-overlap pre-check (classify_pair): hard error / tolerance pair / normal pair.
    1. Broad phase on bounding boxes projected onto the plane perpendicular to d.
    2. Sweep length L carries i fully past j along d.
    3. Step i along d. Where the parts are `delta` apart, jump by delta instead of h;
       otherwise step by h and check the overlap volume against the pair's threshold.

Contact counts as free: only penetration above the threshold blocks. Known limitation: a
collision whose overlap window along d is shorter than h can be missed (grazing contacts).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import FreeCAD  # noqa: F401  (Part needs the FreeCAD runtime loaded first)
import Part
from FreeCAD import Vector

from preprocessing import PreprocessingConfig

# Dataclass for collision configuration parameters. All values are tunable, but the
# default values are based on the industrial fastner case study.
@dataclass
class CollisionConfig:
    # Step size h. Industrial_fastner is stable at 2.0, 0.5 is the conservative default.
    step: float = 5

    # Minimum threshold to consider a contact
    volume_epsilon: float = PreprocessingConfig().volume_epsilon
    # Static overlap above this (mm^3) is a hard error: fix the model or mark the part stander.
    overlap_tol: float = 1.0

    # Configurations of variating stepping behavior. 
    margin: float = 1.0
    gap_skip: bool = True
    use_broad_phase: bool = True


# Dataclass for tracking collision test statistics
@dataclass
class Stats:
    calls: int = 0
    broad_free: int = 0
    iterations: int = 0
    commons: int = 0
    dists: int = 0
    travel: float = 0.0
    gap_travel: float = 0.0
    times: list[float] = field(default_factory=list)  # per stepped (non-broad) test
    iters_per_test: list[int] = field(default_factory=list)
    # Every caught backend exception: {"call", "t", "d", "retry_ok"}.
    exceptions: list[dict] = field(default_factory=list)

# Dataclass for the outcome of a collision test between two parts along a direction.
@dataclass
class Outcome:
    free: bool # True if parts are free
    phase: str  # "broad" or "step"
    sweep_length: float = 0.0 # distance along d to carry i fully past j
    iterations: int = 0 # number of steps taken
    t_block: float | None = None # position along d where blocking occurs
    volume: float = 0.0  # overlap volume at the blocking step
    exception_cell: bool = False  # common failed twice -> reported blocked (conservative)


class HardOverlapError(Exception):
    """Static overlap above overlap_tol: the model must be fixed or the part marked as stander."""

# Auxiliary functions for vector math and bounding box operations, used in the collision tests.
def unit(v: Vector) -> Vector:
    """Normalization of a vector"""
    return Vector(v).normalize()


def _fmt(v: Vector) -> str:
    """Vector formatting for logging"""
    return f"({v.x:.4g}, {v.y:.4g}, {v.z:.4g})"


def _corners(bb) -> list[Vector]:
    """Return the 8 corners of a bounding box."""
    return [
        Vector(x, y, z)
        for x in (bb.XMin, bb.XMax)
        for y in (bb.YMin, bb.YMax)
        for z in (bb.ZMin, bb.ZMax)
    ]


def _extent(bb, d: Vector) -> tuple[float, float]:
    """Return the min and max of the dot product of d with the corners of bb."""
    vals = [d.dot(p) for p in _corners(bb)]
    return min(vals), max(vals)


def _perp_basis(d: Vector) -> tuple[Vector, Vector]:
    """Return two unit vectors perpendicular to d and to each other."""
    helper = Vector(1, 0, 0) if abs(d.x) < 0.9 else Vector(0, 1, 0)
    u = d.cross(helper).normalize()
    w = d.cross(u).normalize()
    return u, w


def broad_phase(shape_i: Part.Shape, shape_j: Part.Shape, d: Vector, tol: float = 1e-9) -> str | None:
    """Return a reason string if the pair is free along +d without any geometry query, else None."""
    bi, bj = shape_i.BoundBox, shape_j.BoundBox
    for axis in _perp_basis(d):
        imin, imax = _extent(bi, axis)
        jmin, jmax = _extent(bj, axis)
        if imax <= jmin + tol or jmax <= imin + tol:
            return "projections disjoint"
    _, jmax_d = _extent(bj, d)
    imin_d, _ = _extent(bi, d)
    if jmax_d <= imin_d + tol:
        return "j behind i"
    return None


def sweep_length(shape_i: Part.Shape, shape_j: Part.Shape, d: Vector, margin: float) -> float:
    """Return the distance along d to carry i fully past j, plus margin."""
    _, jmax_d = _extent(shape_j.BoundBox, d)
    imin_d, _ = _extent(shape_i.BoundBox, d)
    return jmax_d - imin_d + margin


def classify_pair(shape_i: Part.Shape, shape_j: Part.Shape, config: CollisionConfig) -> tuple[str, float, float]:
    """Static-overlap pre-check (step 0). Returns (kind, V0, blocking threshold)."""
    v0 = shape_i.common(shape_j).Volume
    if v0 > config.overlap_tol:
        raise HardOverlapError(f"static overlap {v0:.4f} mm^3 > overlap_tol {config.overlap_tol}")
    if v0 > config.volume_epsilon:
        return "tolerance", v0, max(config.volume_epsilon, 2 * v0)
    return "normal", v0, config.volume_epsilon


# Backend calls go through these module-level hooks so tests can inject failures
# (replace them as `collision._distance = ...`, not as a local name).
def _distance(a: Part.Shape, b: Part.Shape) -> float:
    return a.distToShape(b)[0]


def _common_volume(a: Part.Shape, b: Part.Shape) -> float:
    return a.common(b).Volume


def safe_common(shape_i, shape_j, d: Vector, t: float, h: float, stats: Stats) -> float | None:
    """common at t; on exception retry once at t + h/100; None if that fails too."""
    for attempt, pos in enumerate((t, t + h / 100)):
        stats.commons += 1
        try:
            return _common_volume(shape_i.translated(d * pos), shape_j)
        except Exception:  # noqa: BLE001  (OCCError / RuntimeError from OCC)
            # retry_ok is filled in by is_free_detail once the retry's outcome is known.
            stats.exceptions.append({"call": "common", "t": pos, "d": _fmt(d), "retry_ok": None})
            if attempt == 1:
                return None
    return None


def is_free_detail(
    shape_i: Part.Shape,
    shape_j: Part.Shape,
    d: Vector,
    config: CollisionConfig,
    stats: Stats | None = None,
    threshold: float | None = None,
    max_length: float | None = None,
) -> Outcome:
    """Core of the colision test.

    `max_length`, when given, caps how far i is swept (still at least the broad-phase check):
    i is free as soon as it travels `max_length` even if it has not fully cleared j. A caller
    that needs a *shorter* stopping distance than "fully past j" (e.g. a part that only needs to
    travel a fixed multiple of its own length, docs/preprocessing-spec.md P5) passes it here
    instead of re-implementing the stepped sweep.
    """

    stats = stats if stats is not None else Stats()
    stats.calls += 1
    d = unit(d)

    # Step 0: static overlap pre-check
    # Step 1: broad phase on bounding boxes projected onto the plane perpendicular to d.
    if config.use_broad_phase and broad_phase(shape_i, shape_j, d) is not None:
        stats.broad_free += 1
        return Outcome(True, "broad")
    if threshold is None:
        _, _, threshold = classify_pair(shape_i, shape_j, config)

    # Step 2: step i along d, checking overlap volume at each step.
    t0 = time.perf_counter()
    h = config.step
    length = sweep_length(shape_i, shape_j, d, config.margin)
    if max_length is not None:
        length = min(length, max_length)
    t, iters = 0.0, 0
    outcome = None
    
    while t < length:
        iters += 1
        delta = 0.0
        if config.gap_skip:
            stats.dists += 1
            try:
                delta = _distance(shape_i.translated(d * t), shape_j)
            except Exception:  # noqa: BLE001  (e.g. StdFail_NotDone on touching geometry)
                delta = 0.0  # give up the jump, take a normal step
                stats.exceptions.append({"call": "distToShape", "t": t, "d": _fmt(d), "retry_ok": None})

        # If the gap is larger than the step size, jump by delta instead of h.
        if delta > h:
            t += delta
            stats.gap_travel += delta
            continue
        t += h
        # Step 3: check the overlap volume against the pair's threshold.
        n_exc = len(stats.exceptions)
        vol = safe_common(shape_i, shape_j, d, t, h, stats)
        if len(stats.exceptions) > n_exc:
            stats.exceptions[n_exc]["retry_ok"] = vol is not None
        if vol is None:
            outcome = Outcome(False, "step", length, iters, t, 0.0, exception_cell=True)
            break
        if vol > threshold:
            outcome = Outcome(False, "step", length, iters, t, vol)
            break
    if outcome is None:
        outcome = Outcome(True, "step", length, iters)
    # Update stats and return the outcome.
    stats.iterations += iters
    stats.iters_per_test.append(iters)
    stats.travel += min(t, length)
    stats.times.append(time.perf_counter() - t0)
    return outcome


def is_free(shape_i, shape_j, d, config: CollisionConfig, stats: Stats | None = None, threshold=None,
           max_length: float | None = None) -> bool:
    return is_free_detail(shape_i, shape_j, d, config, stats, threshold, max_length).free
