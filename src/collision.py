"""Single-pair collision test: can part i translate along direction d all the way past part j?

Used by: Preprocessing (standard/assembly-box checks) and Interference Identification (the pairwise
blocking matrix).

Inputs
    - Two FreeCAD `Part.Shape` objects at global placement (i is the moving part, j the obstacle).
    - A direction vector d (normalised internally).
    - A `CollisionConfig` (step size, margin, thresholds, switches).
    - Optionally a `Stats` object that accumulates counters, a precomputed overlap threshold for the pair,
      and `max_length` to cap the sweep.
Outputs
    - `Outcome` (from `is_free_detail`): free or blocked, which phase decided it, and where/how it blocked.
    - `is_free` returns just the boolean.
    - `HardOverlapError` is raised when i and j already overlap by more than `overlap_tol` before moving.

Algorithm (discrete stepping with gap-skipping)
    0. Static-overlap pre-check (classify_pair): hard error / tolerance pair / normal pair.
    1. Broad phase on bounding boxes projected onto the plane perpendicular to d.
    2. Sweep length L carries i fully past j along d.
    3. Step i along d. Where the parts are `delta` apart, jump by delta instead of h; otherwise step by h
       and check the overlap volume against the pair's threshold.

Contact counts as free: only penetration above the threshold blocks. Known limitation: a collision whose
overlap window along d is shorter than h can be missed (grazing contacts).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import FreeCAD  # noqa: F401  (Part needs the FreeCAD runtime loaded first)
import Part
from FreeCAD import Vector

from preprocessing import PreprocessingConfig


@dataclass
class CollisionConfig:
    """Configuration of the collision test. Input to every function in this module.

    All values are tunable. The defaults were validated on the case studies in `data/`
    (industrial_fastner, lunar_*, nonlinearsimple, obliquesimple).
    """

    # float, mm. Step size h of the sweep: smaller = slower = safer against missed grazing collisions but slower.
    # Recommended: 5 (every assembly in data/ disassembles correctly with it and the interference matrix did
    #  not change with a smaller value. For other examples, it might be prudent to compare the dimensions
    # with the used examples and tun this value accordingly).
    step: float = 5

    # float, mm^3. Overlap volume below this is numerical noise from touching geometry, not a contact.
    # Inherited from PreprocessingConfig so both stages use the same noise floor. Recommended: keep it (1e-6).
    volume_epsilon: float = PreprocessingConfig().volume_epsilon
    # float, mm^3. Static overlap above this is a hard error (HardOverlapError): fix the model or mark the
    # part as standard. Recommended: 1.0.
    overlap_tol: float = 1.0

    # float, mm. Extra distance added to the sweep length so i ends clearly past j. Recommended: 1.0.
    margin: float = 1.0
    # bool. Jump by the measured gap instead of stepping when the parts are further apart than h
    # (large speed-up, no effect on the result). Recommended: True.
    gap_skip: bool = True
    # bool. Settle pairs whose projected bounding boxes are disjoint without any geometry query.
    # Recommended: True.
    use_broad_phase: bool = True


@dataclass
class Stats:
    """Counters accumulated over many collision tests (output, optional input to `is_free_detail`).

    Pass one shared instance to collect statistics for a whole interference run; it is also where
    backend exceptions are recorded so they can be reported afterwards.
    """

    calls: int = 0           # int. Number of is_free_detail calls (all pairs/directions).
    broad_free: int = 0      # int. Calls settled as free by the broad phase (no geometry query needed).
    iterations: int = 0      # int. Total sweep steps taken over all stepped tests.
    commons: int = 0         # int. Number of boolean `common` volume queries (the expensive operation).
    dists: int = 0           # int. Number of `distToShape` gap queries (gap-skipping).
    travel: float = 0.0      # float, mm. Total distance swept by stepped tests.
    gap_travel: float = 0.0  # float, mm. Part of `travel` covered by gap jumps instead of steps.
    times: list[float] = field(default_factory=list)  # seconds per stepped (non-broad) test.
    iters_per_test: list[int] = field(default_factory=list)  # steps per stepped test.
    # One dict per caught backend exception: {"call", "t", "d", "retry_ok"}.
    exceptions: list[dict] = field(default_factory=list)


@dataclass
class Outcome:
    """Result of one collision test of part i against part j along a direction (output)."""

    free: bool                    # bool. True if i can translate past j along d.
    phase: str                    # str. "broad" (decided by bounding boxes) or "step" (decided by sweeping).
    sweep_length: float = 0.0     # float, mm. Distance along d that carries i fully past j (plus margin).
    iterations: int = 0           # int. Steps taken before the decision.
    t_block: float | None = None  # float, mm, or None. Position along d where blocking was detected.
    volume: float = 0.0           # float, mm^3. Overlap volume at the blocking step (0 if free/exception).
    exception_cell: bool = False  # bool. The geometry backend failed twice; reported as blocked (conservative).


class HardOverlapError(Exception):
    """Static overlap above overlap_tol: the model must be fixed or the part marked as standard."""


# ---------------------------------------------------------------------------------------------------
# Auxiliary functions: vector maths and bounding-box helpers.
# ---------------------------------------------------------------------------------------------------

def unit(v: Vector) -> Vector:
    """[AUX] Return v as a unit vector (copy; the input is not modified)."""
    return Vector(v).normalize()


def _fmt(v: Vector) -> str:
    """[SUPP] Format a vector as text for the exception log."""
    return f"({v.x:.4g}, {v.y:.4g}, {v.z:.4g})"


def _corners(bb) -> list[Vector]:
    """[AUX] Return the 8 corners of a bounding box."""
    return [
        Vector(x, y, z)
        for x in (bb.XMin, bb.XMax)
        for y in (bb.YMin, bb.YMax)
        for z in (bb.ZMin, bb.ZMax)
    ]


def _extent(bb, d: Vector) -> tuple[float, float]:
    """[AUX] Return (min, max) of the projection of the box onto d, i.e. its extent along that axis."""
    vals = [d.dot(p) for p in _corners(bb)]
    return min(vals), max(vals)


def _perp_basis(d: Vector) -> tuple[Vector, Vector]:
    """[AUX] Return two unit vectors perpendicular to d and to each other (the plane the broad phase uses)."""
    helper = Vector(1, 0, 0) if abs(d.x) < 0.9 else Vector(0, 1, 0)
    u = d.cross(helper).normalize()
    w = d.cross(u).normalize()
    return u, w


def broad_phase(shape_i: Part.Shape, shape_j: Part.Shape, d: Vector, tol: float = 1e-9) -> str | None:
    """[AUX] Cheap free-check on bounding boxes. Returns a reason string if i is free along +d, else None.

    Free when the boxes' projections onto the plane perpendicular to d are disjoint (i never meets j), or
    when j lies entirely behind i along d. None means "unknown, run the stepped test".
    """
    bi, bj = shape_i.BoundBox, shape_j.BoundBox
    # Overlap test on both perpendicular axes: a gap on either one means the sweep tube misses j.
    for axis in _perp_basis(d):
        imin, imax = _extent(bi, axis)
        jmin, jmax = _extent(bj, axis)
        if imax <= jmin + tol or jmax <= imin + tol:
            return "projections disjoint"
    # j entirely behind i along d: moving forward can never reach it.
    _, jmax_d = _extent(bj, d)
    imin_d, _ = _extent(bi, d)
    if jmax_d <= imin_d + tol:
        return "j behind i"
    return None


def sweep_length(shape_i: Part.Shape, shape_j: Part.Shape, d: Vector, margin: float) -> float:
    """[AUX] Distance along d that carries i fully past j (back of j minus front of i), plus margin."""
    _, jmax_d = _extent(shape_j.BoundBox, d)
    imin_d, _ = _extent(shape_i.BoundBox, d)
    return jmax_d - imin_d + margin


def classify_pair(shape_i: Part.Shape, shape_j: Part.Shape, config: CollisionConfig) -> tuple[str, float, float]:
    """[AUX] Static-overlap pre-check (step 0). Returns (kind, V0, blocking threshold).

    V0 is the overlap volume of the parts at rest. Above `overlap_tol` raises HardOverlapError. Above
    `volume_epsilon` the pair is a "tolerance" pair (slightly interpenetrating, e.g. a press fit) and its
    blocking threshold is raised to 2*V0 so the resting overlap itself does not count as a collision;
    otherwise it is a "normal" pair with threshold `volume_epsilon`.
    """
    v0 = shape_i.common(shape_j).Volume
    if v0 > config.overlap_tol:
        raise HardOverlapError(f"static overlap {v0:.4f} mm^3 > overlap_tol {config.overlap_tol}")
    if v0 > config.volume_epsilon:
        return "tolerance", v0, max(config.volume_epsilon, 2 * v0)
    return "normal", v0, config.volume_epsilon


# Backend calls go through these module-level hooks so tests can inject failures
# (replace them as `collision._distance = ...`, not as a local name).
def _distance(a: Part.Shape, b: Part.Shape) -> float:
    """[SUPP] Minimum distance between two shapes (hook for tests to inject failures)."""
    return a.distToShape(b)[0]


def _common_volume(a: Part.Shape, b: Part.Shape) -> float:
    """[SUPP] Volume of the boolean intersection of two shapes (hook for tests to inject failures)."""
    return a.common(b).Volume


def safe_common(shape_i, shape_j, d: Vector, t: float, h: float, stats: Stats) -> float | None:
    """[AUX] Overlap volume of i moved by t along d with j. Retries once at t + h/100 if OCC fails.

    OCC boolean operations occasionally fail on touching geometry; a tiny shift usually avoids it. Each
    failure is logged in `stats.exceptions`. Returns None when the retry fails too.
    """
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


# ---------------------------------------------------------------------------------------------------
# Main function: the stepped collision test.
# ---------------------------------------------------------------------------------------------------

def is_free_detail(
    shape_i: Part.Shape,
    shape_j: Part.Shape,
    d: Vector,
    config: CollisionConfig,
    stats: Stats | None = None,
    threshold: float | None = None,
    max_length: float | None = None,
) -> Outcome:
    """[MAIN] Core collision test: can i translate along d past j without penetrating it?

    Inputs
        shape_i, shape_j: moving part and obstacle, both at global placement.
        d: translation direction (any length; normalised here).
        config: step size, margin and switches (see CollisionConfig).
        stats: shared counters; a throwaway one is created if omitted.
        threshold: overlap volume (mm^3) above which the pair counts as colliding. Pass the value from
            `classify_pair` when the caller already ran it (avoids a second boolean per call); if None
            it is computed here, which also raises HardOverlapError for hard static overlaps.
        max_length: caps how far i is swept (the broad-phase check still applies): i is free as soon as it
            has travelled `max_length` even if it has not fully cleared j. A caller that needs a shorter
            stopping distance than "fully past j" (e.g. a part that only needs to travel a fixed multiple
            of its own length, rule P5) passes it here instead of re-implementing
            the stepped sweep.
    Output
        An `Outcome`: free (True/False), the deciding phase, and for blocked results the position along d
        and overlap volume of the first blocking step. A backend failure that survives the retry is
        reported as blocked with `exception_cell=True` (conservative).
    """

    stats = stats if stats is not None else Stats()
    stats.calls += 1
    d = unit(d)

    # Step 1: broad phase on bounding boxes projected onto the plane perpendicular to d. Pairs settled here
    # need no geometry query at all, which is most pairs in a typical assembly.
    if config.use_broad_phase and broad_phase(shape_i, shape_j, d) is not None:
        stats.broad_free += 1
        return Outcome(True, "broad")
    # Step 0 (static pre-check): runs only for pairs the broad phase could not settle. It fixes the blocking
    # threshold, or raises HardOverlapError for a hard overlap.
    if threshold is None:
        _, _, threshold = classify_pair(shape_i, shape_j, config)

    # Step 2: set up the sweep. The length takes i fully past j (plus margin), optionally capped.
    t0 = time.perf_counter()
    h = config.step
    length = sweep_length(shape_i, shape_j, d, config.margin)
    if max_length is not None:
        length = min(length, max_length)
    t, iters = 0.0, 0
    outcome = None

    while t < length:
        iters += 1
        # Gap measurement: how far is i from j right now? Used to skip empty space in one jump.
        delta = 0.0
        if config.gap_skip:
            stats.dists += 1
            try:
                delta = _distance(shape_i.translated(d * t), shape_j)
            except Exception:  # noqa: BLE001  (e.g. StdFail_NotDone on touching geometry)
                delta = 0.0  # give up the jump, take a normal step
                stats.exceptions.append({"call": "distToShape", "t": t, "d": _fmt(d), "retry_ok": None})

        # If the gap is larger than the step size, jump by delta instead of h: nothing can collide in between.
        if delta > h:
            t += delta
            stats.gap_travel += delta
            continue
        t += h
        # Step 3: measure the overlap at the new position and compare it with the pair's threshold.
        n_exc = len(stats.exceptions)
        vol = safe_common(shape_i, shape_j, d, t, h, stats)
        if len(stats.exceptions) > n_exc:
            stats.exceptions[n_exc]["retry_ok"] = vol is not None
        # Backend failed twice: treat the pair as blocked (safe side) and flag the cell.
        if vol is None:
            outcome = Outcome(False, "step", length, iters, t, 0.0, exception_cell=True)
            break
        # Penetration above the threshold: blocked at this position.
        if vol > threshold:
            outcome = Outcome(False, "step", length, iters, t, vol)
            break
    # Reached the end of the sweep without a blocking overlap: free.
    if outcome is None:
        outcome = Outcome(True, "step", length, iters)
    # Update the shared statistics, then return.
    stats.iterations += iters
    stats.iters_per_test.append(iters)
    stats.travel += min(t, length)
    stats.times.append(time.perf_counter() - t0)
    return outcome


def is_free(shape_i, shape_j, d, config: CollisionConfig, stats: Stats | None = None, threshold=None,
           max_length: float | None = None) -> bool:
    """[AUX] Boolean shortcut for `is_free_detail`: True if i can translate along d past j."""
    return is_free_detail(shape_i, shape_j, d, config, stats, threshold, max_length).free
