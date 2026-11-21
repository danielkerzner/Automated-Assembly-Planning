"""Preprocessing: standard handling (Part 1) and candidate-direction discovery (Part 2).

Rules are numbered P1-P12 and referred to as P<n> in the comments.

Inputs
    - A STEP assembly loaded as a FreeCAD document (`load_assembly`), or one level's document built by
      `levels.build_level_document`.
    - The BOM CSV: which components are standards (`is_standard`) and, for standards, whether they are screws
      or nuts (`screw` / `nut` columns). Read by `load_bom` and `load_standard_categories`.
    - A `PreprocessingConfig` (thresholds) and a `collision_config` (`collision.CollisionConfig`).
Outputs
    - A `PreprocessingResult` holding: the reduced assembly (a new FreeCAD document with every standard
      removed), one `StandardRecord` per standard (how and when it can be removed), the list of candidate
      disassembly directions, and the whole-assembly bounding box. Consumed by Interference Identification,
      the AoG generator and the per-level orchestration in `levels.py`.

What this module does
    Part 1, standard handling. Standards (screws, nuts, pins...) are not moved by the optimizer as ordinary
    parts. Each is classified as category 1 (screw), 2 (nut) or 3 (other) from the BOM flags, and the module
    works out for each one (a) which direction(s) it leaves along, (b) which components must already be
    gone for it to leave (`blockers`), (c) for screws, which components must still be present (`holds`),
    and (d) for nuts, which component occupies the bore (`partner`) and whether the nut is rotation-locked
    (`nut_rule`). All collision questions reuse the stepped sweep of `collision.py`. The standards are then
    stripped out of the assembly.
    Part 2, candidate directions. From the remaining (reduced) assembly, build the list of directions the
    parts may translate along: the six main axes, plus cylinder axes and planar-face normals taken only from
    faces that touch another component (stages 1-3 in `discover_candidate_directions`).
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import FreeCAD
import Part
import Import

log = logging.getLogger("aap.preprocessing")

_TRUE = ("1", "true", "yes", "y", "t")  # accepted spellings of "true" in BOM CSV cells (compared lower-case)


@dataclass
class PreprocessingConfig:
    """Thresholds of the preprocessing stage. Config input of `process_standards` and
    `discover_candidate_directions`.

    All values are tunable; the defaults were calibrated on the industrial_fastner case study. Fields marked
    P<n> are the numbered preprocessing rules (P1-P12).
    """

    # float, mm^3. Intersection volume below this is numerical noise from touching (not interfering)
    # geometry, not a real contact. Also the noise floor of `collision.CollisionConfig`. Recommended: 1e-6.
    volume_epsilon: float = 1e-6
    # float, mm. Gap below which a non-standard counts as physically touching a standard (feeds
    # `contact_neighbors`, i.e. assembly-order contact rather than the directional dismantle walk). Part 2
    # reuses it to decide whether a face touches another component at all. Recommended: 0.01.
    contact_threshold: float = 0.01

    # str, "none" or "exterior". How a fused sub-assembly component contributes candidate directions
    #: "none" = nothing from its internal geometry; "exterior" = its fused
    # solid's faces are scanned like any other component's (only exterior faces exist on a fused solid).
    # Recommended: "none".
    compound_directions: str = "none"

    # float, degrees. Part 2 stage 3 only: a planar-face normal is dropped if it is within this angle of a
    # direction already in the list. Stops many slightly misaligned faces from inflating the number of
    # candidate directions. Recommended: 15.0.
    direction_min_angle_deg: float = 15.0

    # Part 1 parameters. All are "unconfirmed defaults, calibrate on industrial_fastner":
    probe_displacement: float = 0.1  # mm, axial displacement for the direction-sign test (P4)
    perpendicular_displacement: float = 0.5  # mm, lateral displacement for screw holds (P6)
    perpendicular_directions: int = 8  # number of evenly spaced lateral directions around the axis (P6)
    nut_rotation_angles_deg: tuple[float, ...] = (15.0, 30.0, 45.0)  # angles of the nut rotation-lock test (P8)
    pairing_min_fill: float = 0.25  # fraction of the bore volume a partner must occupy to be a partner (P7)
    bore_clip_margin: float = 0.2  # mm, radial margin added to the bore when clipping the partner (P8)
    # float, multiples of the screw's own length. A screw only needs to travel this far along its direction
    # or clear the whole-assembly bbox, whichever comes first (P5 extension). Recommended: 1.5.
    screw_travel_factor: float = 1.5


@dataclass
class RemovalOption:
    """One way a standard can leave the assembly (part of a `StandardRecord`; output of `process_standards`).

    Screws and nuts have one option, category-3 standards usually have two (both senses of the axis).
    """

    direction: FreeCAD.Vector  # Vector (unit). Direction the standard translates along to leave.
    exit_point: FreeCAD.Vector  # Vector, global mm. Where the standard's axis leaves the assembly bbox (drives the AoG travel model).
    blockers: set[str]  # set of FreeCAD object Names that must already be removed before it can leave.
    # float, mm, or None. Screws only (P5 extension): the standard only needs to travel
    # min(screw_travel_factor * its own length, distance to clear the assembly bbox) plus margin, rather
    # than always clearing the whole assembly. None = no override, use the bbox-only distance as before
    # (nuts, category-3 standards and any screw this was not computed for).
    max_travel: float | None = None


@dataclass
class StandardRecord:
    """Contact and dismantle information for one standard (output of `process_standards`).

    Replaces the old prerequisites/dismantle_direction/exit_point fields: "present" is now `holds`,
    "absent" is now a RemovalOption's `blockers`. The AoG generator reads it to build the standard
    transitions.
    """

    category: int  # int. 1 = screw, 2 = nut, 3 = other.
    options: list[RemovalOption]  # list. 1 option for categories 1/2, usually 2 for category 3.

    # set of Names. Non-standard components that must still be PRESENT while this standard is removed
    # (screws only, P6). Category 2/3 standards never have holds.
    holds: set[str]

    # set of Names. All non-standard components physically touching the standard (a full contact walk, not
    # directional, P10). Feeds the AoG connectivity rule (D11); it is not a gate.
    contact_neighbors: set[str]

    # (min, max) pair of global-coordinate Vectors, or None. The standard's own bounding box inside the
    # assembly, stored because the AoG travel model has no other access to standard geometry.
    bbox: tuple[FreeCAD.Vector, FreeCAD.Vector] | None = None

    partner: str | None = None  # str or None. Nuts only (P7): Name of whatever occupies the bore (standard or not).
    nut_rule: str | None = None  # "locked" | "free" | None. Nuts only (P8): whether the nut is rotation-locked (e.g. hex pocket).

    # Read-only compatibility properties for single-option standards (categories 1/2), kept for downstream
    # code not yet migrated to `options`. They raise on a category-3 standard (two options, no single answer).
    @property
    def dismantle_direction(self) -> FreeCAD.Vector:
        if len(self.options) != 1:
            raise AttributeError("dismantle_direction is only defined for a single-option standard "
                                  "(category 1/2); this one has multiple options, use .options")
        return self.options[0].direction

    @property
    def exit_point(self) -> FreeCAD.Vector:
        if len(self.options) != 1:
            raise AttributeError("exit_point is only defined for a single-option standard "
                                  "(category 1/2); this one has multiple options, use .options")
        return self.options[0].exit_point


@dataclass
class PreprocessingResult:
    """Everything Preprocessing hands to the later stages (output of `run_preprocessing_on_document`)."""

    # FreeCAD Document. The assembly with every standard removed (movable parts only, at global placement).
    # Named '<doc>_reduced'; `shapes.component_shapes` relies on that name to find the original document.
    reduced_assembly: "FreeCAD.Document"

    # dict {object Name: StandardRecord}. One record per standard.
    standard_records: dict[str, StandardRecord]

    # list of unit Vectors. Candidate disassembly directions discovered from the reduced assembly.
    candidate_directions: list[FreeCAD.Vector]

    # (min, max) pair of Vectors, or None. Whole-assembly bounding box including standards, computed before
    # they are stripped. Later stages (e.g. the AoG travel model) must measure against this fixed box and
    # never one rebuilt from a sub-assembly.
    assembly_bbox: tuple[FreeCAD.Vector, FreeCAD.Vector] | None = None

    # Path or None. The input files, so later stages can key their caches on the file contents.
    step_path: Path | None = None
    bom_path: Path | None = None

    # Sub-assembly identity; None / empty for a flat, non-nested run.
    level_id: str | None = None  # str. The level's container id ("__root__" or an object Name).
    level_label: str | None = None  # str. The container's Label (human-readable name).
    compound_ids: frozenset[str] = field(default_factory=frozenset)  # ids of the fused child-container components at this level.
    members: dict[str, list[str]] = field(default_factory=dict)  # dict {compound id: descendant leaf ids}.

    # dict {auto_<n> id: BOM `group` value}. Phase 2 automatic sub-assemblies: the group the merged component
    # uses in Optimization (the shared group of its members if they all agree, else its own id).
    # Set by `levels._commit_merge`, read by `optimization.collect_tables`; empty otherwise.
    auto_group_override: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------
# Loading (auxiliary)
# ---------------------------------------------------------------------------------------------------

def load_assembly(step_path: str | Path) -> "FreeCAD.Document":
    """[AUX] Import a STEP assembly into a new FreeCAD document, one Part::Feature per solid.

    The import preference `ReadShapeCompoundMode` is switched off so every solid becomes its own object
    instead of one compound.
    """
    step_path = Path(step_path)
    FreeCAD.ParamGet(
        "User parameter:BaseApp/Preferences/Mod/Import/hSTEP"
    ).SetBool("ReadShapeCompoundMode", False)
    doc = FreeCAD.newDocument(step_path.stem)
    Import.insert(str(step_path), doc.Name)
    return doc


def load_bom(bom_path: str | Path) -> set[str]:
    """[AUX] Load the BOM and return the set of component Labels flagged as standards.

    CSV with a `label` column, matched exactly against imported Part::Feature `Label` values. An optional
    `is_standard` column (true/false/1/0/yes/no) restricts which rows count; if the column is absent every
    listed row is a standard.
    """
    standards: set[str] = set()
    with open(bom_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_flag_col = reader.fieldnames is not None and "is_standard" in reader.fieldnames
        for row in reader:
            label = (row.get("label") or "").strip()
            if not label:
                continue
            if has_flag_col:
                flag = (row.get("is_standard") or "").strip().lower()
                if flag not in _TRUE:
                    continue
            standards.add(label)
    return standards


def load_standard_categories(bom_path: str | Path) -> dict[str, str | None]:
    """[AUX] Load the BOM's `screw`/`nut` flags (P1). Returns {label: "screw" | "nut" | None}.

    Only rows with `is_standard` true appear; None means category 3 ("other"). A BOM without `screw`/`nut`
    columns makes every standard category 3. Both flags on one row is a load error (ValueError); a flag on a
    non-standard row is ignored with a warning.
    """
    categories: dict[str, str | None] = {}
    with open(bom_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_screw_col = "screw" in fieldnames
        has_nut_col = "nut" in fieldnames
        for row in reader:
            label = (row.get("label") or "").strip()
            if not label:
                continue
            is_standard = (row.get("is_standard") or "").strip().lower() in _TRUE
            screw = has_screw_col and (row.get("screw") or "").strip().lower() in _TRUE
            nut = has_nut_col and (row.get("nut") or "").strip().lower() in _TRUE
            if not is_standard:
                if screw or nut:
                    log.warning("BOM row %r sets 'screw'/'nut' but 'is_standard' is not true; ignored", label)
                continue
            if screw and nut:
                raise ValueError(f"BOM row {label!r} sets both 'screw' and 'nut' true; a standard "
                                  "can only be one category")
            categories[label] = "screw" if screw else ("nut" if nut else None)
    return categories


# Functions whose name starts with an underscore are internal helpers, not part of the public API.

# ---------------------------------------------------------------------------------------------------
# Geometry helpers (auxiliary)
# ---------------------------------------------------------------------------------------------------

def _leaf_components(doc: "FreeCAD.Document") -> list:
    """[AUX] All leaf Part::Feature objects of the document that have a non-null shape with at least one solid."""
    return [
        obj
        for obj in doc.Objects
        if obj.TypeId == "Part::Feature"
        and getattr(obj, "Shape", None) is not None
        and not obj.Shape.isNull()
        and len(obj.Shape.Solids) > 0
    ]


def _global_shape(obj) -> Part.Shape:
    """[AUX] Copy of the object's shape moved to its global placement (the original is not modified)."""
    shape = obj.Shape.copy()
    shape.Placement = obj.getGlobalPlacement()
    return shape


def _assembly_bounding_box(objs):
    """[AUX] Union of the global bounding boxes of all given objects."""
    bbox = None
    for obj in objs:
        b = _global_shape(obj).BoundBox
        bbox = b if bbox is None else bbox.united(b)
    return bbox


def _bbox_to_tuple(bbox) -> tuple[FreeCAD.Vector, FreeCAD.Vector]:
    """[AUX] Convert a FreeCAD BoundBox to a (min, max) pair of Vectors (the form stored in the records)."""
    return (
        FreeCAD.Vector(bbox.XMin, bbox.YMin, bbox.ZMin),
        FreeCAD.Vector(bbox.XMax, bbox.YMax, bbox.ZMax),
    )


def _dominant_cylindrical_axis(shape: Part.Shape):
    """[AUX] (base_point, unit_direction, radius) of the shape's largest cylindrical face, or None (P2).

    For a screw or nut this is its own axis, the line it is turned and removed along.
    """
    cyl_faces = [f for f in shape.Faces if f.Surface.TypeId == "Part::GeomCylinder"]
    if not cyl_faces:
        return None
    face = max(cyl_faces, key=lambda f: f.Area)
    surf = face.Surface
    direction = FreeCAD.Vector(surf.Axis).normalize()
    return surf.Center, direction, surf.Radius


def _bbox_overlap(a, b, margin: float = 0.0) -> bool:
    """[AUX] True if bounding boxes `a` and `b` are within `margin` of overlapping.

    A cheap necessary condition: if the boxes do not overlap (even expanded by `margin`) the shapes cannot
    intersect or be within `margin` either, so an exact boolean or distance query can be skipped. It never
    changes a result, only how many exact queries are needed to get it.
    """
    return not (
        a.XMin - margin > b.XMax or a.XMax + margin < b.XMin
        or a.YMin - margin > b.YMax or a.YMax + margin < b.YMin
        or a.ZMin - margin > b.ZMax or a.ZMax + margin < b.ZMin
    )


def _contact_neighbors(
    standard_shape: Part.Shape, candidates: list, candidate_shapes: dict[str, Part.Shape], config: PreprocessingConfig
) -> set[str]:
    """[AUX] Names of the candidates within `contact_threshold` of the standard (P10): its contact neighbours.

    Uses the bounding-box test first, then `distToShape` for the survivors.
    """
    standard_bbox = standard_shape.BoundBox
    neighbors: set[str] = set()
    for obj in candidates:
        shape = candidate_shapes[obj.Name]
        if not _bbox_overlap(standard_bbox, shape.BoundBox, config.contact_threshold):
            continue
        distance, _, _ = standard_shape.distToShape(shape)
        if distance <= config.contact_threshold:
            neighbors.add(obj.Name)
    return neighbors


# --- Part 1 helpers -------------------------------------

def _v0(a_name: str, b_name: str, shapes: dict[str, Part.Shape], cache: dict[tuple[str, str], float]) -> float:
    """[AUX] Static overlap volume of two components at rest (P3), cached.

    It is the baseline every displaced, swept or rotated collision test below is compared against.
    """
    key = (a_name, b_name)
    if key not in cache:
        cache[key] = shapes[a_name].common(shapes[b_name]).Volume
    return cache[key]


def _collides(overlap_volume: float, v0: float, config: PreprocessingConfig) -> bool:
    """[AUX] P3: a moved, swept or rotated copy collides with j iff its overlap exceeds max(eps, 2*V0)."""
    return overlap_volume > max(config.volume_epsilon, 2 * v0)


def _perp_basis(d: FreeCAD.Vector) -> tuple[FreeCAD.Vector, FreeCAD.Vector]:
    """[AUX] Two unit vectors perpendicular to d and to each other (same construction as collision._perp_basis)."""
    helper = FreeCAD.Vector(1, 0, 0) if abs(d.x) < 0.9 else FreeCAD.Vector(0, 1, 0)
    u = d.cross(helper).normalize()
    w = d.cross(u).normalize()
    return u, w


def _bore_radius(shape: Part.Shape, axis_base: FreeCAD.Vector, axis_dir: FreeCAD.Vector) -> float | None:
    """[AUX] Radius of the nut's largest *internal* cylindrical face, its bore (P7), or None if it has none.

    Internal means the face normal points toward the axis. This is not always P2's dominant cylinder: on a
    hex nut the bore is also the largest cylinder, but on a round or knurled nut the outside is larger.
    """
    best = None
    for face in shape.Faces:
        if face.Surface.TypeId != "Part::GeomCylinder":
            continue
        # `face.CenterOfMass` collapses onto the axis for a full-revolution cylindrical face (the
        # radially symmetric points average out), so it can't give a radial direction -- sample an
        # actual surface point at the middle of the face's own parameter range instead.
        u0, u1, v0, v1 = face.ParameterRange
        u, v = (u0 + u1) / 2, (v0 + v1) / 2
        point = face.Surface.value(u, v)
        normal = face.normalAt(u, v)
        t = (point - axis_base).dot(axis_dir)
        axis_point = axis_base + axis_dir * t
        radial = point - axis_point
        if radial.Length < 1e-9:
            continue
        radial.normalize()
        if normal.dot(radial) >= 0:
            continue  # outward-facing cylindrical face, not a bore
        if best is None or face.Area > best[0]:
            best = (face.Area, face.Surface.Radius)
    return best[1] if best is not None else None


def _axial_extent(shape: Part.Shape, base: FreeCAD.Vector, axis_dir: FreeCAD.Vector) -> tuple[float, float]:
    """[AUX] Min and max signed projection of the shape's vertices onto the axis through `base`.

    This is the standard's own extent along its axis; it sizes the bore solid (P7) to "the nut's length" and
    gives a screw's own length for the travel cap.
    """
    projections = [(v.Point - base).dot(axis_dir) for v in shape.Vertexes]
    return min(projections), max(projections)


def _bbox_exit_distance(shape: Part.Shape, direction: FreeCAD.Vector, assembly_bbox) -> float:
    """[AUX] Raw distance along `direction` for the shape's own extent to clear the whole-assembly bounding box.

    No margin is added (callers add their own). It uses the same per-axis projection convention as
    `collision.sweep_length`, treating the assembly box as the "other shape". The P5 screw-travel extension
    only needs a safe upper bound for how far the sweep must run, so the fuller multi-axis formula of
    `aog_generator.compute_travel_table` is not repeated here.
    """
    from collision import _extent

    _, box_max_d = _extent(assembly_bbox, direction)
    shape_min_d, _ = _extent(shape.BoundBox, direction)
    return box_max_d - shape_min_d


def _axial_solid(radius: float, base: FreeCAD.Vector, axis_dir: FreeCAD.Vector,
                 t_min: float, t_max: float) -> Part.Shape:
    """[AUX] Cylinder of `radius` centred on the axis from `t_min` to `t_max` along `axis_dir` from `base`.

    Used as the bore solid (P7) and, with a larger radius, to clip the partner (P8).
    """
    length = t_max - t_min
    start = base + axis_dir * t_min
    return Part.makeCylinder(radius, length, start, axis_dir)


def _find_partner(
    nut_shape: Part.Shape, bore_solid: Part.Shape, bore_volume: float, candidates: list,
    shapes: dict[str, Part.Shape], config: PreprocessingConfig, name: str,
) -> str | None:
    """[AUX] P7: the nut's partner is whatever occupies its bore, standard or not.

    It is the component with the largest overlap against the bore solid, provided that overlap is at least
    `pairing_min_fill` of the bore volume. A component merely touching the nut (a washer around the shaft)
    does not fill the bore. Returns None if nothing qualifies, so the nut becomes category 3.
    """
    best_name, best_overlap = None, 0.0
    bore_bbox = bore_solid.BoundBox
    for other in candidates:
        if other.Name == name:
            continue
        other_shape = shapes[other.Name]
        if not _bbox_overlap(bore_bbox, other_shape.BoundBox):
            continue
        overlap = bore_solid.common(other_shape).Volume
        if overlap > best_overlap:
            best_overlap, best_name = overlap, other.Name
    if best_name is None or best_overlap < config.pairing_min_fill * bore_volume:
        return None
    return best_name


def _signed_axis_direction(
    shape: Part.Shape, axis_dir: FreeCAD.Vector, displacement: float, candidates: list,
    shapes: dict[str, Part.Shape], v0_cache: dict, config: PreprocessingConfig, name: str,
) -> FreeCAD.Vector | None:
    """[AUX] P4: pick the sense of the axis the standard can leave along, by a small probe displacement.

    The shape is moved by `displacement` along each side of the axis; a side "collides" (P3) if the moved
    copy hits any other candidate, standards included. Returns the direction of the single free side, or
    None if both or neither side is free (the caller then falls back to category 3).
    """
    collides_side = {}
    for sign in (1.0, -1.0):
        moved = shape.translated(axis_dir * sign * displacement)
        moved_bbox = moved.BoundBox
        collided = False
        for other in candidates:
            if other.Name == name:
                continue
            other_shape = shapes[other.Name]
            if not _bbox_overlap(moved_bbox, other_shape.BoundBox):
                continue
            overlap = moved.common(other_shape).Volume
            if _collides(overlap, _v0(name, other.Name, shapes, v0_cache), config):
                collided = True
                break
        collides_side[sign] = collided
    free_sides = [s for s, c in collides_side.items() if not c]
    return axis_dir * free_sides[0] if len(free_sides) == 1 else None


def _sweep_blockers(
    shape: Part.Shape, direction: FreeCAD.Vector, candidates: list, shapes: dict[str, Part.Shape],
    v0_cache: dict, collision_config, config: PreprocessingConfig, name: str, stats,
    max_length: float | None = None,
) -> set[str]:
    """[AUX] P5: the blockers of a standard along `direction`.

    Every candidate the standard's solid cannot sweep past (the collision core's stepped sweep; contact
    counts as free; threshold per P3) is a blocker. `max_length`, when given, caps the sweep (screw-travel
    extension): a component the standard would only reach after that distance is not a blocker, since the
    standard is already clear of the assembly by then.
    """
    from collision import is_free_detail

    blockers: set[str] = set()
    for other in candidates:
        if other.Name == name:
            continue
        other_shape = shapes[other.Name]
        threshold = max(config.volume_epsilon, 2 * _v0(name, other.Name, shapes, v0_cache))
        outcome = is_free_detail(shape, other_shape, direction, collision_config, stats, threshold=threshold,
                                 max_length=max_length)
        if not outcome.free:
            blockers.add(other.Name)
    return blockers


def _screw_holds(
    shape: Part.Shape, axis_dir: FreeCAD.Vector, candidates: list, shapes: dict[str, Part.Shape],
    v0_cache: dict, config: PreprocessingConfig, name: str,
) -> set[str]:
    """[AUX] P6: the non-standard components a screw holds in place (they must still be present when it leaves).

    A component j is a hold if it statically overlaps the screw, or if displacing the screw sideways (any of
    `perpendicular_directions` evenly spaced directions around its axis) makes it collide with j.
    """
    holds: set[str] = set()
    for other in candidates:
        if other.Name == name:
            continue
        if _v0(name, other.Name, shapes, v0_cache) > config.volume_epsilon:
            holds.add(other.Name)

    # Lateral probes: a screw passing through a hole is stopped by the hole wall when pushed sideways.
    u, w = _perp_basis(axis_dir)
    n = max(config.perpendicular_directions, 1)
    for i in range(n):
        angle = 2 * math.pi * i / n
        perp_dir = (u * math.cos(angle) + w * math.sin(angle)).normalize()
        moved = shape.translated(perp_dir * config.perpendicular_displacement)
        moved_bbox = moved.BoundBox
        for other in candidates:
            if other.Name == name or other.Name in holds:
                continue
            other_shape = shapes[other.Name]
            if not _bbox_overlap(moved_bbox, other_shape.BoundBox):
                continue
            overlap = moved.common(other_shape).Volume
            if _collides(overlap, _v0(name, other.Name, shapes, v0_cache), config):
                holds.add(other.Name)
    return holds


def _nut_rotation_locked(
    nut_shape: Part.Shape, base: FreeCAD.Vector, axis_dir: FreeCAD.Vector, candidates: list,
    shapes: dict[str, Part.Shape], v0_cache: dict, config: PreprocessingConfig, name: str,
    partner_name: str, partner_clip_shape: Part.Shape,
) -> bool:
    """[AUX] P8: True if the nut cannot rotate about its own axis (it sits in a pocket), False if it can.

    The nut is rotated by each of `nut_rotation_angles_deg` (both signs) against every other component,
    standards included (P3); any collision means locked. The partner is tested against its bore-clipped shape
    (`partner_clip_shape`, thread engagement cut out) so normal thread contact does not cause a false lock,
    while a hex pocket belonging to the partner itself still counts.
    """
    for angle_deg in config.nut_rotation_angles_deg:
        for sign in (1.0, -1.0):
            rotated = nut_shape.rotated(base, axis_dir, angle_deg * sign)
            rotated_bbox = rotated.BoundBox
            for other in candidates:
                if other.Name == name:
                    continue
                if other.Name == partner_name:
                    other_shape, v0 = partner_clip_shape, 0.0  # clip already removes the rest baseline
                else:
                    other_shape, v0 = shapes[other.Name], _v0(name, other.Name, shapes, v0_cache)
                if not _bbox_overlap(rotated_bbox, other_shape.BoundBox):
                    continue
                overlap = rotated.common(other_shape).Volume
                if _collides(overlap, v0, config):
                    return True
    return False


# ---------------------------------------------------------------------------------------------------
# Main function, Part 1: standard handling
# ---------------------------------------------------------------------------------------------------

def process_standards(
    doc: "FreeCAD.Document",
    bom_labels: set[str],
    config: PreprocessingConfig,
    collision_config=None,
    bom_categories: dict[str, str | None] | None = None,
    compound_ids: frozenset[str] = frozenset(),
) -> tuple[dict[str, StandardRecord], set[str], tuple[FreeCAD.Vector, FreeCAD.Vector]]:
    """[MAIN] Part 1: classify every standard and work out how, along which direction and when it can leave.

    Inputs
        doc: the FreeCAD document (flat assembly or one level's document).
        bom_labels: Labels of the components flagged as standards (`load_bom`).
        config: thresholds (`PreprocessingConfig`).
        collision_config: `collision.CollisionConfig` for the sweeps; default if None.
        bom_categories: {label: "screw" | "nut" | None} from `load_standard_categories`. None or missing
            entries make a standard category 3 (a slow default: nothing is simplified).
        compound_ids: Names of this document's fused sub-assembly components. A standard that is one of them
            (a standard sub-assembly) tries the cheap P4 probe test before full sweeps (see pass 4).
    Outputs
        (records, standard_ids, assembly_bbox): one `StandardRecord` per standard, the set of standard object
        Names, and the whole-assembly bounding box (min, max) including standards.

    Passes: (1) find each standard's axis and BOM category; (1.5) find each nut's bore and partner; (2) screws;
    (3) nuts; (4) everything else; then the contact neighbours of every standard. A screw or nut whose
    direction test fails is demoted to category 3. Rules P1-P12 are the numbered preprocessing rules. Unlike the
    rest of the pipeline the nut pass needs nothing from the screw pass: the precedence it derives lives on
    the nut's own record (`partner`/`nut_rule`) and the AoG reads it from there.
    """
    from collision import CollisionConfig, Stats

    collision_config = collision_config or CollisionConfig()
    bom_categories = bom_categories or {}

    # Common data: all leaves, split into standards and movable parts, with their global shapes.
    all_components = _leaf_components(doc)
    standard_objs = [obj for obj in all_components if obj.Label in bom_labels]
    standard_ids = {obj.Name for obj in standard_objs}
    assembly_bbox = _assembly_bounding_box(all_components)
    non_standards = [obj for obj in all_components if obj.Name not in standard_ids]
    shapes = {obj.Name: _global_shape(obj) for obj in all_components}
    v0_cache: dict[tuple[str, str], float] = {}
    stats = Stats()

    # Pass 1: axis (P2) + initial category (P1) for every standard, from the BOM flags.
    axes: dict[str, tuple] = {}
    category: dict[str, int] = {}
    for obj in standard_objs:
        axis = _dominant_cylindrical_axis(shapes[obj.Name])
        if axis is None:
            raise ValueError(f"No cylindrical feature found on standard '{obj.Label}'")
        axes[obj.Name] = axis
        category[obj.Name] = {"screw": 1, "nut": 2}.get(bom_categories.get(obj.Label), 3)

    records: dict[str, StandardRecord] = {}

    # Pass 1.5: nut bore/partner info, computed early (before screws) purely so screws know which
    # nuts are partnered to them (P11 exception below) -- nothing else about nut processing
    # depends on screws, so moving this earlier changes no result, it only makes the mapping
    # available sooner. A nut demoted to category 3 here (no partner) gets no entry.
    nut_bore: dict[str, tuple[float, Part.Shape, float]] = {}  # name -> (bore_radius, bore_solid, bore_volume)
    partner_of: dict[str, str] = {}       # nut name -> partner name
    nuts_of_partner: dict[str, set[str]] = {}  # partner name -> set of nut names
    for obj in standard_objs:
        name = obj.Name
        if category[name] != 2:
            continue
        base, axis_dir, _radius = axes[name]
        shape = shapes[name]
        # Bore solid: a cylinder filling the nut's hole over its whole length; its occupant is the partner.
        bore_radius = _bore_radius(shape, base, axis_dir)
        if bore_radius is None:
            raise ValueError(f"No internal (bore) cylindrical face found on nut standard '{obj.Label}'")
        t_min, t_max = _axial_extent(shape, base, axis_dir)
        bore_solid = _axial_solid(bore_radius, base, axis_dir, t_min, t_max)
        bore_volume = bore_solid.Volume
        partner = _find_partner(shape, bore_solid, bore_volume, all_components, shapes, config, name)
        if partner is None:
            log.info("Nut standard %r: nothing occupies its bore above pairing_min_fill (P7); "
                     "treating as category 3", obj.Label)
            category[name] = 3
            continue
        nut_bore[name] = (bore_radius, bore_solid, bore_volume)
        partner_of[name] = partner
        nuts_of_partner.setdefault(partner, set()).add(name)

    # Pass 2: screws. A nut's own partner always physically overlaps/grazes it along the axis as
    # it (un)screws -- that is not a blocking collision, it is already modeled by the nut's own
    # `partner`/`nut_rule` (P7/P8), so a screw excludes any nut partnered to it from its blocker
    # sweep, the same way a nut excludes its partner below. Every *other* standard is a fair
    # candidate (P11 revised: a screw/nut can be blocked by an unrelated standard, confirmed on
    # `lunar_bodylager` where a nut's real removal path collides with an unrelated spacer standard).
    for obj in standard_objs:
        name = obj.Name
        if category[name] != 1:
            continue
        base, axis_dir, _radius = axes[name]
        shape = shapes[name]
        # Removal direction: the single free side of the axis (P4); otherwise demote to category 3.
        d = _signed_axis_direction(shape, axis_dir, config.probe_displacement, all_components,
                                   shapes, v0_cache, config, name)
        if d is None:
            log.warning("Screw standard %r: both axis sides free or both blocked (P4); treating "
                       "as category 3", obj.Label)
            category[name] = 3
            continue
        # Holds (P6): the movable parts this screw keeps in place.
        holds = _screw_holds(shape, axis_dir, non_standards, shapes, v0_cache, config, name)
        # P5 extension: the screw only needs to travel screw_travel_factor times its own length
        # along d, or clear the whole-assembly bbox -- whichever comes first (plus the usual sweep
        # margin) -- rather than always sweeping all the way to the bbox.
        t_min, t_max = _axial_extent(shape, base, axis_dir)
        own_length = t_max - t_min
        bbox_exit = _bbox_exit_distance(shape, d, assembly_bbox)
        max_travel = min(config.screw_travel_factor * own_length, bbox_exit) + collision_config.margin
        # Blockers (P5): what the screw cannot sweep past along d, minus its holds and its own nuts.
        own_nuts = nuts_of_partner.get(name, set())
        blocker_candidates = [o for o in all_components if o.Name not in own_nuts]
        blockers = _sweep_blockers(shape, d, blocker_candidates, shapes, v0_cache, collision_config,
                                   config, name, stats, max_length=max_travel) - holds
        exit_point = assembly_bbox.getIntersectionPoint(base, d)
        records[name] = StandardRecord(
            category=1,
            options=[RemovalOption(direction=d, exit_point=exit_point, blockers=blockers, max_travel=max_travel)],
            holds=holds, contact_neighbors=set(), bbox=_bbox_to_tuple(shape.BoundBox),
        )

    # Pass 3: nuts. The partner (P7, generalised) can be any component -- standard or movable -- so
    # this pass needs nothing from pass 2; the precedence it derives (P8) lives entirely on the
    # nut's own record (`partner`/`nut_rule`), never mutating anyone else's.
    for obj in standard_objs:
        name = obj.Name
        if category[name] != 2:
            continue
        base, axis_dir, _radius = axes[name]
        shape = shapes[name]
        bore_radius, bore_solid, bore_volume = nut_bore[name]
        t_min, t_max = _axial_extent(shape, base, axis_dir)
        partner = partner_of[name]
        # Removal direction (P4); the nut should leave opposite to its partner screw (sanity check only).
        d = _signed_axis_direction(shape, axis_dir, config.probe_displacement, all_components,
                                   shapes, v0_cache, config, name)
        if d is None:
            log.warning("Nut standard %r: both axis sides free or both blocked (P4); treating as "
                       "category 3", obj.Label)
            category[name] = 3
            continue
        partner_rec = records.get(partner)
        if partner_rec is not None and len(partner_rec.options) == 1 and d.dot(partner_rec.options[0].direction) > 0:
            log.warning("Nut standard %r's direction is not opposite its partner %r's (P1 sanity "
                       "check); keeping it anyway", obj.Label, partner)
        # P11 revised (see pass 2 comment): every other component, standards included, is a candidate
        # blocker -- except the partner itself, which always overlaps/grazes the nut along its own
        # axis and is already governed by `partner`/`nut_rule`, not by a translation blocker
        # (including it here would contradict `nut_rule` for at least one of the two).
        blocker_candidates = [o for o in all_components if o.Name != partner]
        blockers = _sweep_blockers(shape, d, blocker_candidates, shapes, v0_cache, collision_config,
                                   config, name, stats)
        # Rotation lock (P8): rotate against everything, the partner clipped at the thread engagement.
        clip_solid = _axial_solid(bore_radius + config.bore_clip_margin, base, axis_dir, t_min, t_max)
        partner_clip_shape = shapes[partner].cut(clip_solid)
        locked = _nut_rotation_locked(shape, base, axis_dir, standard_objs + non_standards, shapes,
                                      v0_cache, config, name, partner, partner_clip_shape)
        nut_rule = "locked" if locked else "free"
        exit_point = assembly_bbox.getIntersectionPoint(base, d)
        records[name] = StandardRecord(
            category=2, options=[RemovalOption(direction=d, exit_point=exit_point, blockers=blockers)],
            holds=set(), contact_neighbors=set(), bbox=_bbox_to_tuple(shape.BoundBox),
            partner=partner, nut_rule=nut_rule,
        )

    # Pass 4: everything else -- originally category 3, plus any screw/nut demoted above. Both senses of
    # the dominant axis are kept as removal options, each with its own blockers.
    for obj in standard_objs:
        name = obj.Name
        if category[name] != 3:
            continue
        base, axis_dir, _radius = axes[name]
        shape = shapes[name]
        others = [o for o in standard_objs + non_standards if o.Name != name]
        options = []
        probe_dir = None
        sweep_config = collision_config
        if name in compound_ids:
            # Standard sub-assembly: try the cheap P4
            # probe-displacement sign test (same helper as screws/nuts) before paying for a full
            # sweep in both directions. Exactly one free side at the probe displacement is taken as
            # the unique direction, and the opposite side's full sweep is skipped entirely. Both
            # free or both blocked: fall back to the original two-sweep pass.
            probe_dir = _signed_axis_direction(shape, axis_dir, config.probe_displacement,
                                               standard_objs + non_standards, shapes, v0_cache, config, name)
            # gap_skip off for this sweep (measured on lunar_motor_tire_complete: distToShape()
            # between two large fused shapes near contact cost 18-90 s per call, vs. 0.7-3.5 s for
            # common() on the same pair -- gap_skip's per-step distToShape() dominated the whole
            # sweep). Fixed-step common()-only checks still cover the entire travel distance, so
            # the correctness guarantee is the same, just without the expensive distance query.
            sweep_config = replace(collision_config, gap_skip=False)
        if probe_dir is not None:
            blockers = _sweep_blockers(shape, probe_dir, others, shapes, v0_cache, sweep_config,
                                       config, name, stats)
            exit_point = assembly_bbox.getIntersectionPoint(base, probe_dir)
            options = [RemovalOption(direction=probe_dir, exit_point=exit_point, blockers=blockers)]
            log.info("Standard subassembly %r: probe test (P4) found a unique free side; skipping "
                     "the opposite direction's full sweep", obj.Label)
        else:
            if name in compound_ids:
                log.info("Standard subassembly %r: probe test (P4) found both sides free or both "
                         "blocked; falling back to a full sweep in both directions", obj.Label)
            for sign in (1.0, -1.0):
                d = axis_dir * sign
                blockers = _sweep_blockers(shape, d, others, shapes, v0_cache, sweep_config,
                                           config, name, stats)
                exit_point = assembly_bbox.getIntersectionPoint(base, d)
                options.append(RemovalOption(direction=d, exit_point=exit_point, blockers=blockers))
        records[name] = StandardRecord(
            category=3, options=options, holds=set(), contact_neighbors=set(),
            bbox=_bbox_to_tuple(shape.BoundBox),
        )

    # contact_neighbors (P10): every standard against the non-standard pool.
    for obj in standard_objs:
        name = obj.Name
        records[name] = replace(
            records[name], contact_neighbors=_contact_neighbors(shapes[name], non_standards, shapes, config),
        )

    # Summary log: how many standards ended up in each category (after demotions).
    n_screw = sum(1 for c in category.values() if c == 1)
    n_nut = sum(1 for c in category.values() if c == 2)
    n_other = sum(1 for c in category.values() if c == 3)
    log.info("Processed %d standard(s): %d screw, %d nut, %d other; %.2f s",
             len(standard_objs), n_screw, n_nut, n_other, sum(stats.times))

    return records, standard_ids, _bbox_to_tuple(assembly_bbox)


def build_standard_reduced_assembly(doc: "FreeCAD.Document", standard_ids: set[str]) -> "FreeCAD.Document":
    """[AUX] Build the reduced document: every leaf except the standards, copied at its global placement.

    The new document is named '<doc>_reduced' (`shapes.component_shapes` depends on that name). Object
    Names and Labels are kept so later stages can match components by id.
    """
    reduced = FreeCAD.newDocument(f"{doc.Name}_reduced")
    for obj in _leaf_components(doc):
        if obj.Name in standard_ids:
            continue
        feature = reduced.addObject("Part::Feature", obj.Name)
        feature.Shape = _global_shape(obj)
        feature.Label = obj.Label
    reduced.recompute()
    return reduced


# ---------------------------------------------------------------------------------------------------
# Part 2: candidate directions
# ---------------------------------------------------------------------------------------------------

# The six main axes: always the first candidates, so the list is never empty.
_MAIN_AXES = (
    FreeCAD.Vector(1, 0, 0), FreeCAD.Vector(-1, 0, 0),
    FreeCAD.Vector(0, 1, 0), FreeCAD.Vector(0, -1, 0),
    FreeCAD.Vector(0, 0, 1), FreeCAD.Vector(0, 0, -1),
)


def _direction_key(v: FreeCAD.Vector) -> tuple[float, float, float]:
    """[AUX] Hashable key of a direction (components rounded to 9 decimals) for exact-duplicate detection."""
    return (round(v.x, 9), round(v.y, 9), round(v.z, 9))


def _face_in_contact(face, obj_name: str, components, threshold: float) -> bool:
    """[AUX] True if `face` is within `threshold` of some other component's shape.

    `components` must already be in the same (global) coordinates as `face`, as they are in
    `discover_candidate_directions` (shapes in the reduced document are global). A bounding-box test skips
    the expensive `distToShape` call for components that cannot be within range.
    """
    face_bbox = face.BoundBox
    for other in components:
        if other.Name == obj_name:
            continue
        if not _bbox_overlap(face_bbox, other.Shape.BoundBox, threshold):
            continue
        if face.distToShape(other.Shape)[0] <= threshold:
            return True
    return False


def discover_candidate_directions(
    reduced_doc: "FreeCAD.Document",
    skip_ids: frozenset[str] = frozenset(),
    config: PreprocessingConfig | None = None,
) -> list[FreeCAD.Vector]:
    """[MAIN] Part 2: build the list of candidate disassembly directions from the reduced assembly.

    Inputs
        reduced_doc: the reduced document (standards already removed).
        skip_ids: Names of fused sub-assembly components whose internal geometry must not contribute
            directions (used when `compound_directions == "none"`).
        config: thresholds (`contact_threshold`, `direction_min_angle_deg`).
    Output
        A list of unit Vectors, the six main axes first, then the additions of stages 2 and 3.

    Stages
        1. The six main axes (+-x, +-y, +-z), always included first.
        2. Axes of cylindrical faces that touch another component (within `contact_threshold`),
           deduplicated by exact signed-vector equality against everything already in the list.
        3. Normals of planar faces with the same contact requirement; a normal is dropped if it lies within
           `direction_min_angle_deg` of any direction already in the list.
    A face touching nothing is not a mating interface and is unlikely to be a real separation axis; that
    contact requirement is the main source of count reduction. In both stages the cheap duplicate check runs
    before the expensive contact check (most faces repeat a handful of directions). Contact is checked only
    inside the reduced assembly (standards excluded as sources and as partners): checking against the removed
    standards was tried and reverted, it recovered no directions on `lunar_bodylager` and cost runtime. If an
    assembly needs it, `direction_min_angle_deg` or `contact_threshold` are the tunables to revisit first.
    """
    config = config or PreprocessingConfig()
    min_angle = math.radians(config.direction_min_angle_deg)

    directions: list[FreeCAD.Vector] = []
    seen: set[tuple[float, float, float]] = set()

    def add(v: FreeCAD.Vector) -> None:
        key = _direction_key(v)
        if key in seen:
            return
        seen.add(key)
        directions.append(v)

    # Stage 1: main axes.
    for axis in _MAIN_AXES:
        add(axis)

    components = [obj for obj in _leaf_components(reduced_doc) if obj.Name not in skip_ids]

    # Stage 2: cylindrical features, only from faces touching another component, exact dedup only.
    # The direction is cheap to derive and compare; the contact check is not, so check first
    # whether this axis (both signs) is already in the list -- most faces across a real assembly
    # repeat a handful of directions, and a redundant one can never add anything regardless of
    # contact.
    for obj in components:
        for face in obj.Shape.Faces:
            if face.Surface.TypeId != "Part::GeomCylinder":
                continue
            axis = FreeCAD.Vector(face.Surface.Axis).normalize()
            if _direction_key(axis) in seen and _direction_key(-axis) in seen:
                continue
            if not _face_in_contact(face, obj.Name, components, config.contact_threshold):
                continue
            add(axis)
            add(-axis)

    # Stage 3: planar features, same early-exit idea, but the dedup test is angular (within
    # `min_angle` of the list built so far, not exact equality) -- skip the contact check only
    # when both signs are already within tolerance of something already accepted.
    for obj in components:
        for face in obj.Shape.Faces:
            if face.Surface.TypeId != "Part::GeomPlane":
                continue
            normal = FreeCAD.Vector(face.Surface.Axis).normalize()
            pending = [
                candidate
                for candidate in (normal, -normal)
                if not any(candidate.getAngle(existing) < min_angle for existing in directions)
            ]
            if not pending:
                continue
            if not _face_in_contact(face, obj.Name, components, config.contact_threshold):
                continue
            for candidate in pending:
                # Re-check: an earlier candidate from this same face (or an earlier face in this
                # same loop) may have just been added, making this one redundant now too.
                if any(candidate.getAngle(existing) < min_angle for existing in directions):
                    continue
                add(candidate)

    return directions


# ---------------------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------------------

def _has_nested_containers(doc: "FreeCAD.Document") -> bool:
    """[SUPP] True if the document has more than one top-level App::Part or a nested one (only used for a warning)."""
    roots = [o for o in doc.RootObjects if o.TypeId == "App::Part"]
    if len(roots) != 1:
        return len(roots) > 0

    def has_nested(obj) -> bool:
        return any(c.TypeId == "App::Part" or has_nested(c) for c in getattr(obj, "Group", []))

    return has_nested(roots[0])


def run_preprocessing_on_document(
    doc: "FreeCAD.Document",
    bom_labels: set[str],
    config: PreprocessingConfig | None = None,
    *,
    collision_config=None,
    bom_categories: dict[str, str | None] | None = None,
    step_path: Path | None = None,
    bom_path: Path | None = None,
    level_id: str | None = None,
    level_label: str | None = None,
    compound_ids: frozenset[str] = frozenset(),
    members: dict[str, list[str]] | None = None,
) -> PreprocessingResult:
    """[MAIN] Run Part 1 (standards) and Part 2 (candidate directions) on an already-loaded document.

    `doc` may be the flat, originally loaded assembly, or one level's document from
    `levels.build_level_document`. The remaining arguments:
      - `collision_config`: sweep settings for Part 1 (default `CollisionConfig()` if None).
      - `bom_categories`: from `load_standard_categories`. If omitted, every standard is category 3, which
        works but is much slower (no screw/nut shortcuts); `main.py` always passes it.
      - `step_path`, `bom_path`: stored in the result so later stages can key caches on the files.
      - `level_id`, `level_label`, `compound_ids`, `members`: sub-assembly identity of this level, copied
        into the result (empty for a flat run).
    Returns a `PreprocessingResult`. Raises ValueError for an invalid `compound_directions`.
    """
    config = config or PreprocessingConfig()
    if config.compound_directions not in ("none", "exterior"):
        raise ValueError(f"compound_directions must be 'none' or 'exterior', got {config.compound_directions!r}")
    members = members or {}

    # Part 1: classify and process the standards, then strip them from the assembly.
    standard_records, standard_ids, assembly_bbox = process_standards(
        doc, bom_labels, config, collision_config=collision_config, bom_categories=bom_categories,
        compound_ids=compound_ids)
    reduced_assembly = build_standard_reduced_assembly(doc, standard_ids)

    # Part 2: candidate directions from the reduced assembly. Under "none" the fused sub-assembly
    # components contribute nothing.
    skip_ids = compound_ids if config.compound_directions == "none" else frozenset()
    candidate_directions = discover_candidate_directions(reduced_assembly, skip_ids=skip_ids, config=config)
    # Fallback to "exterior" when the list is empty. As written it never triggers: the six main axes are
    # always in the list, so `candidate_directions` is never empty (see CLAUDE.md, Open mismatches).
    if not candidate_directions and compound_ids and config.compound_directions == "none":
        log.warning(
            "No candidate directions found with compound_directions='none' (every direction-bearing "
            "face belonged to a fused sub-assembly component); retrying with 'exterior'"
        )
        candidate_directions = discover_candidate_directions(reduced_assembly, skip_ids=frozenset(), config=config)

    return PreprocessingResult(
        reduced_assembly, standard_records, candidate_directions,
        assembly_bbox=assembly_bbox,
        step_path=step_path, bom_path=bom_path,
        level_id=level_id, level_label=level_label,
        compound_ids=frozenset(compound_ids), members=dict(members),
    )


def run_preprocessing(
    step_path: str | Path,
    bom_path: str | Path,
    config: PreprocessingConfig | None = None,
    collision_config=None,
) -> PreprocessingResult:
    """[AUX] Flat-assembly entry point: load the STEP and BOM, then call `run_preprocessing_on_document`.

    Correct only for a flat assembly (no nested sub-assemblies); a nested one goes through
    `hierarchy.build_hierarchy` + `levels.solve_level` instead. A warning is
    logged if nested containers are detected. Used by `main.py` for flat runs and by the manual tests.
    """
    doc = load_assembly(step_path)
    bom_labels = load_bom(bom_path)
    bom_categories = load_standard_categories(bom_path)
    if _has_nested_containers(doc):
        log.warning(
            "This assembly has nested sub-assemblies (App::Part); run_preprocessing only sees the "
            "flat leaf list and ignores the hierarchy. Use hierarchy.build_hierarchy + "
            "levels.solve_level for a correct nested run."
        )
    return run_preprocessing_on_document(
        doc, bom_labels, config, collision_config=collision_config, bom_categories=bom_categories,
        step_path=Path(step_path), bom_path=Path(bom_path),
    )
