"""Preprocessing: stander handling + candidate direction discovery.

Part 1 (stander handling) follows docs/preprocessing-spec.md (rewritten 2026-11-04, nut
partnering generalised the same evening): standers split into three classes -- screw, nut,
other -- each with its own dismantle-direction and blocker rule, reusing the collision core
(collision.py) instead of an inflated probe cylinder. Part 2 (candidate directions) is unchanged
(spec decision P12).
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

_TRUE = ("1", "true", "yes", "y", "t")


# Dataclass for preprocessing configuration parameters. All values are tunable, but the
# default values are based on the industrial fastner case study.
@dataclass
class PreprocessingConfig:
    # Intersection volume below this is treated as numerical noise from
    # touching (not interfering) geometry, not a real contact.
    volume_epsilon: float = 1e-6
    # Minimum gap below which a non-stander component counts as physically
    # touching the stander (assembly-order contact, not the directional
    # dismantle-path walk). Also reused by Part 2 stage 3 to decide whether a
    # planar face is touching another component at all (see discover_candidate_directions).
    contact_threshold: float = 0.01

    # Candidate-direction discovery (Part 2) and a fused sub-assembly component (docs/subassembly-spec.md):
    # "none" = it contributes no directions from its internal geometry; "exterior" = its fused solid's
    # faces are scanned like any other component's (they are exterior faces only).
    compound_directions: str = "none"

    # Part 2 step 3 only (planar-feature normals): a candidate is discarded if it is within this
    # many degrees of any direction already in the list (main axes, cylindrical axes, or an
    # earlier-accepted planar normal). Keeps near-duplicate directions from many slightly
    # misaligned faces from blowing up the candidate-direction count.
    direction_min_angle_deg: float = 15.0

    # Part 1 (docs/preprocessing-spec.md), all "unconfirmed defaults, calibrate on
    # industrial_fastner":
    probe_displacement: float = 0.1  # mm, axial displacement for the direction-sign test (P4)
    perpendicular_displacement: float = 0.5  # mm, lateral displacement for screw holds (P6)
    perpendicular_directions: int = 8  # evenly spaced directions around the axis (P6)
    nut_rotation_angles_deg: tuple[float, ...] = (15.0, 30.0, 45.0)  # rotation lock test (P8)
    pairing_min_fill: float = 0.25  # min fraction of bore volume a partner must occupy (P7)
    bore_clip_margin: float = 0.2  # mm, radial margin added to the bore when clipping the partner (P8)
    # A screw only needs to travel this many times its own length along its direction, or clear
    # the whole-assembly bbox -- whichever comes first (P5 extension, added 2026-11-05).
    screw_travel_factor: float = 1.5


# Dataclass for one removal option of a stander: a direction, where its trajectory exits the
# assembly bounding box, and the components that must already be gone before it can leave.
@dataclass
class RemovalOption:
    direction: FreeCAD.Vector
    exit_point: FreeCAD.Vector
    blockers: set[str]  # component Names that must already be removed
    # Screws only (docs/preprocessing-spec.md P5 extension, added 2026-11-05): the stander only
    # needs to travel min(screw_travel_factor * its own length along `direction`, the distance to
    # clear the whole-assembly bbox), plus margin -- whichever comes first -- rather than always
    # clearing the whole assembly. None = no override, use the bbox-only distance (D10) as before
    # (nuts, category-3 "other", and any screw this wasn't computed for).
    max_travel: float | None = None


# Dataclass for storing the contact and dismantle information for a stander component. Replaces
# the old prerequisites/dismantle_direction/exit_point fields (docs/preprocessing-spec.md,
# "Data model"): "present" is now `holds`, "absent" is now a RemovalOption's `blockers`.
@dataclass
class StanderRecord:
    category: int  # 1 screw, 2 nut, 3 other
    options: list[RemovalOption]  # 1 for categories 1/2, 2 for category 3

    # Non-stander components that must still be PRESENT while this stander is removed (screws
    # only; P6). Category 2/3 standers never have holds.
    holds: set[str]

    # All non-stander components physically touching the stander (full contact walk, not
    # directional). Unchanged meaning from the old spec (P10): feeds the AoG connectivity rule
    # (D11), it is not a gate any more.
    contact_neighbors: set[str]

    # Global (min, max) bounding box of the stander's own shape inside the assembly. Needed by
    # the AoG travel model, which has no other access to stander geometry.
    bbox: tuple[FreeCAD.Vector, FreeCAD.Vector] | None = None

    partner: str | None = None  # nuts only (P7): Name of whatever occupies the bore, stander or not
    nut_rule: str | None = None  # "locked" | "free" | None (P8)

    # Read-only compatibility properties for single-option standers (categories 1/2), for any
    # downstream code not yet migrated to `options` (docs/preprocessing-spec.md, "Data
    # model"). Raise on a category-3 stander (two options; there is no single answer).
    @property
    def dismantle_direction(self) -> FreeCAD.Vector:
        if len(self.options) != 1:
            raise AttributeError("dismantle_direction is only defined for a single-option stander "
                                  "(category 1/2); this one has multiple options, use .options")
        return self.options[0].direction

    @property
    def exit_point(self) -> FreeCAD.Vector:
        if len(self.options) != 1:
            raise AttributeError("exit_point is only defined for a single-option stander "
                                  "(category 1/2); this one has multiple options, use .options")
        return self.options[0].exit_point

#Dataclass for preprocessing results. Reduxed assembly + Stander records + candidate directions.
#Also stores the input file paths for caching purposes.
@dataclass
class PreprocessingResult:
    # Reduced assembly with all standers removed
    reduced_assembly: "FreeCAD.Document"

    # stander_records
    stander_records: dict[str, StanderRecord]

    #Candidate directions discovered from the reduced assembly.
    candidate_directions: list[FreeCAD.Vector]

    # Whole-assembly bounding box (global min, max), all components including standers,
    # computed before standers are stripped. Downstream modules (e.g. the AoG's travel
    # model) must measure against this fixed box, never one rebuilt from a sub-assembly.
    assembly_bbox: tuple[FreeCAD.Vector, FreeCAD.Vector] | None = None

    # Input files, so later stages can key caches on their contents.
    step_path: Path | None = None
    bom_path: Path | None = None

    # Sub-assembly identity (docs/subassembly-spec.md); None/empty for a flat, non-nested run.
    level_id: str | None = None  # the level's container id ("__root__" or an obj.Name)
    level_label: str | None = None  # the container's Label
    compound_ids: frozenset[str] = field(default_factory=frozenset)  # fused child-container component ids
    members: dict[str, list[str]] = field(default_factory=dict)  # compound id -> descendant leaf ids

    # Phase 2 (docs/subassembly-spec.md section 10): auto_<n> id -> the BOM M7 `group` value it
    # should use in Optimization -- the shared group of its direct merge members if they all agree,
    # else its own id (levels._commit_merge sets this; see optimization.collect_tables).
    auto_group_override: dict[str, str] = field(default_factory=dict)


def load_assembly(step_path: str | Path) -> "FreeCAD.Document":
    """Import a STEP assembly with one Part::Feature per solid"""
    step_path = Path(step_path)
    FreeCAD.ParamGet(
        "User parameter:BaseApp/Preferences/Mod/Import/hSTEP"
    ).SetBool("ReadShapeCompoundMode", False)
    doc = FreeCAD.newDocument(step_path.stem)
    Import.insert(str(step_path), doc.Name)
    return doc


def load_bom(bom_path: str | Path) -> set[str]:
    """Load the external BOM, return the set of component Labels flagged as standers.

    CSV with a `label` column, matched exactly against imported Part::Feature
    `Label` values. An optional `is_stander` column (true/false/1/0/yes/no)
    restricts which rows count; if the column is absent every listed row is a
    stander.
    """
    standers: set[str] = set()
    with open(bom_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_flag_col = reader.fieldnames is not None and "is_stander" in reader.fieldnames
        for row in reader:
            label = (row.get("label") or "").strip()
            if not label:
                continue
            if has_flag_col:
                flag = (row.get("is_stander") or "").strip().lower()
                if flag not in _TRUE:
                    continue
            standers.add(label)
    return standers


def load_stander_categories(bom_path: str | Path) -> dict[str, str | None]:
    """Load the BOM's `screw`/`nut` category flags (docs/preprocessing-spec.md P1).

    Returns `{label: "screw" | "nut" | None}` for every row with `is_stander` true; `None` means
    category 3 ("other"). Rows without `is_stander` true are not in the returned dict at all
    (`process_standers` only consults it for actual standers). A BOM without `screw`/`nut`
    columns makes every stander category 3 (empty dict values). Both flags true on one row is a
    load error; a flag set on a non-stander row is ignored with a warning.
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
            is_stander = (row.get("is_stander") or "").strip().lower() in _TRUE
            screw = has_screw_col and (row.get("screw") or "").strip().lower() in _TRUE
            nut = has_nut_col and (row.get("nut") or "").strip().lower() in _TRUE
            if not is_stander:
                if screw or nut:
                    log.warning("BOM row %r sets 'screw'/'nut' but 'is_stander' is not true; ignored", label)
                continue
            if screw and nut:
                raise ValueError(f"BOM row {label!r} sets both 'screw' and 'nut' true; a stander "
                                  "can only be one category")
            categories[label] = "screw" if screw else ("nut" if nut else None)
    return categories

"""Functions marked with _ are internal helpers and not part of the public API."""
def _leaf_components(doc: "FreeCAD.Document") -> list:
    """Return all leaf Part::Feature objects in the document 
    that have a non-null shape and at least one solid."""
    return [
        obj
        for obj in doc.Objects
        if obj.TypeId == "Part::Feature"
        and getattr(obj, "Shape", None) is not None
        and not obj.Shape.isNull()
        and len(obj.Shape.Solids) > 0
    ]


def _global_shape(obj) -> Part.Shape:
    """Return a copy of the object's shape in global coordinates."""
    shape = obj.Shape.copy()
    shape.Placement = obj.getGlobalPlacement()
    return shape


def _assembly_bounding_box(objs):
    """Return the bounding box of the assembly in global coordinates."""
    bbox = None
    for obj in objs:
        b = _global_shape(obj).BoundBox
        bbox = b if bbox is None else bbox.united(b)
    return bbox


def _bbox_to_tuple(bbox) -> tuple[FreeCAD.Vector, FreeCAD.Vector]:
    """Convert a FreeCAD BoundBox to a (min, max) pair of global-coordinate vectors."""
    return (
        FreeCAD.Vector(bbox.XMin, bbox.YMin, bbox.ZMin),
        FreeCAD.Vector(bbox.XMax, bbox.YMax, bbox.ZMax),
    )


def _dominant_cylindrical_axis(shape: Part.Shape):
    """Return (base_point, unit_direction, radius) of the largest cylindrical face, or None."""
    cyl_faces = [f for f in shape.Faces if f.Surface.TypeId == "Part::GeomCylinder"]
    if not cyl_faces:
        return None
    face = max(cyl_faces, key=lambda f: f.Area)
    surf = face.Surface
    direction = FreeCAD.Vector(surf.Axis).normalize()
    return surf.Center, direction, surf.Radius


def _bbox_overlap(a, b, margin: float = 0.0) -> bool:
    """True if bounding boxes `a` and `b` are within `margin` of overlapping.

    A cheap necessary-condition check: if the boxes don't overlap (even expanded by `margin`),
    the actual shapes can't possibly intersect or be within `margin` of each other either, so an
    expensive exact boolean/distance computation can be skipped safely -- this never changes the
    result, only how many exact computations are needed to get it.
    """
    return not (
        a.XMin - margin > b.XMax or a.XMax + margin < b.XMin
        or a.YMin - margin > b.YMax or a.YMax + margin < b.YMin
        or a.ZMin - margin > b.ZMax or a.ZMax + margin < b.ZMin
    )


def _contact_neighbors(
    stander_shape: Part.Shape, candidates: list, candidate_shapes: dict[str, Part.Shape], config: PreprocessingConfig
) -> set[str]:
    """Return the set of candidate component Names that are in contact with the stander (P10,
    unchanged from the old spec)."""
    stander_bbox = stander_shape.BoundBox
    neighbors: set[str] = set()
    for obj in candidates:
        shape = candidate_shapes[obj.Name]
        if not _bbox_overlap(stander_bbox, shape.BoundBox, config.contact_threshold):
            continue
        distance, _, _ = stander_shape.distToShape(shape)
        if distance <= config.contact_threshold:
            neighbors.add(obj.Name)
    return neighbors


# --- Part 1 helpers (docs/preprocessing-spec.md) -------------------------------------

def _v0(a_name: str, b_name: str, shapes: dict[str, Part.Shape], cache: dict[tuple[str, str], float]) -> float:
    """Static overlap volume at rest (P3), cached -- the baseline every displaced/swept/rotated
    collision test below is compared against."""
    key = (a_name, b_name)
    if key not in cache:
        cache[key] = shapes[a_name].common(shapes[b_name]).Volume
    return cache[key]


def _collides(overlap_volume: float, v0: float, config: PreprocessingConfig) -> bool:
    """P3: a moved/swept/rotated copy collides with j iff its overlap exceeds max(eps, 2*V0)."""
    return overlap_volume > max(config.volume_epsilon, 2 * v0)


def _perp_basis(d: FreeCAD.Vector) -> tuple[FreeCAD.Vector, FreeCAD.Vector]:
    """Return two unit vectors perpendicular to d and to each other."""
    helper = FreeCAD.Vector(1, 0, 0) if abs(d.x) < 0.9 else FreeCAD.Vector(0, 1, 0)
    u = d.cross(helper).normalize()
    w = d.cross(u).normalize()
    return u, w


def _bore_radius(shape: Part.Shape, axis_base: FreeCAD.Vector, axis_dir: FreeCAD.Vector) -> float | None:
    """P7/implementation notes: radius of the nut's largest *internal* cylindrical face (its
    bore) -- the face whose normal points toward the axis, not away from it. This is not always
    the same face as P2's dominant cylinder: for a hex nut the bore is also the largest cylinder,
    but for a round/knurled nut the outside is larger, so the bore must be picked separately.
    Returns None if the shape has no internal cylindrical face at all."""
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
    """Min/max signed projection of every vertex onto `axis_dir` through `base` -- the stander's
    own extent along its axis, used to size the bore solid (P7) to "the nut's length"."""
    projections = [(v.Point - base).dot(axis_dir) for v in shape.Vertexes]
    return min(projections), max(projections)


def _bbox_exit_distance(shape: Part.Shape, direction: FreeCAD.Vector, assembly_bbox) -> float:
    """Raw distance (no margin -- callers add their own) along `direction` for `shape`'s own
    extent to clear the whole-assembly bounding box (a `FreeCAD.BoundBox`), same per-axis
    projection convention as `collision.sweep_length` (treating the assembly box as the "other
    shape"). Used by the P5 screw-travel extension (2026-11-05) to find "the distance to clear
    the assembly bbox" without re-deriving `aog_generator.compute_travel_table`'s fuller
    multi-axis formula here: a cutoff for how far the stepped sweep needs to run only has to be a
    safe upper bound, not exact."""
    from collision import _extent

    _, box_max_d = _extent(assembly_bbox, direction)
    shape_min_d, _ = _extent(shape.BoundBox, direction)
    return box_max_d - shape_min_d


def _axial_solid(radius: float, base: FreeCAD.Vector, axis_dir: FreeCAD.Vector,
                 t_min: float, t_max: float) -> Part.Shape:
    """A cylinder of `radius`, centered on the axis, spanning from `t_min` to `t_max` along
    `axis_dir` from `base` -- the bore solid (P7) or its clipped-partner counterpart (P8)."""
    length = t_max - t_min
    start = base + axis_dir * t_min
    return Part.makeCylinder(radius, length, start, axis_dir)


def _find_partner(
    nut_shape: Part.Shape, bore_solid: Part.Shape, bore_volume: float, candidates: list,
    shapes: dict[str, Part.Shape], config: PreprocessingConfig, name: str,
) -> str | None:
    """P7 (generalised 2026-11-04): the partner is whatever occupies the nut's bore, stander or
    not -- the component with the largest overlap against the bore solid, provided that overlap
    is at least `pairing_min_fill` of the bore volume. A component merely touching the nut (a
    washer ringed around the shaft) does not occupy the bore and is never a candidate. No
    candidate above the threshold: the nut is category 3."""
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
    """P4: move `shape` by `displacement` along each axis side; a side "collides" (P3) if the
    moved copy collides with any other candidate, standers included. Returns the single free
    side's direction, or None if both or neither side is free (caller falls back to category 3)."""
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
    """P5: every candidate the stander's own solid cannot sweep past along `direction` (the
    collision core's stepped sweep, contact counts as free, threshold per P3) is a blocker.

    `max_length`, when given, caps the sweep (2026-11-05 screw-travel extension): a component the
    stander would only reach *after* `max_length` is not a blocker, since the stander is
    considered clear of the assembly by then regardless.
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
    """P6: a non-stander j is a hold if it statically overlaps the screw, or if displacing the
    screw perpendicular to its axis (any of `perpendicular_directions` evenly spaced directions)
    makes it collide (P3) with j. Non-stander candidates only -- holds are screws-only."""
    holds: set[str] = set()
    for other in candidates:
        if other.Name == name:
            continue
        if _v0(name, other.Name, shapes, v0_cache) > config.volume_epsilon:
            holds.add(other.Name)

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
    """P8 (generalised 2026-11-04): rotate the nut about its own axis by each of
    `nut_rotation_angles_deg` (both signs) against every other component, standers included
    (P3). The partner is tested too, but against its bore-clipped shape (`partner_clip_shape`,
    the thread engagement cut out) so normal thread contact cannot cause a false lock while a
    hex pocket belonging to the partner itself still counts. Any collision at any tested angle =
    locked (hex pocket)."""
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


def process_standers(
    doc: "FreeCAD.Document",
    bom_labels: set[str],
    config: PreprocessingConfig,
    collision_config=None,
    bom_categories: dict[str, str | None] | None = None,
    compound_ids: frozenset[str] = frozenset(),
) -> tuple[dict[str, StanderRecord], set[str], tuple[FreeCAD.Vector, FreeCAD.Vector]]:
    """Return stander records, the set of stander object Names, and the whole-assembly bbox.

    Implements docs/preprocessing-spec.md Part 1 (P1-P12): standers are classified into three
    categories from the BOM's `screw`/`nut` flags (P1). Nut partnering (P7, generalised
    2026-11-04) is whatever occupies the nut's bore by volume, stander or movable component
    alike, so unlike the rest of the pipeline the nut pass needs nothing precomputed by the
    screw pass -- the precedence it derives (P8) lives entirely on the nut's own record
    (`partner`/`nut_rule`), the AoG reads it from there instead of it being pushed into anyone
    else's blockers.

    `compound_ids` names this document's fused sub-assembly components (docs/subassembly-
    spec.md): for a stander whose id is in there (a stander subassembly, added 2026-11-11), pass
    4 tries the cheap P4 probe-displacement sign test before paying for a full sweep in both
    directions (docs/stander_processing-spec.md section 14) -- a plain leaf "other" stander is
    unaffected (its shape is small enough that two full sweeps are already cheap).
    """
    from collision import CollisionConfig, Stats

    collision_config = collision_config or CollisionConfig()
    bom_categories = bom_categories or {}

    all_components = _leaf_components(doc)
    stander_objs = [obj for obj in all_components if obj.Label in bom_labels]
    stander_ids = {obj.Name for obj in stander_objs}
    assembly_bbox = _assembly_bounding_box(all_components)
    non_standers = [obj for obj in all_components if obj.Name not in stander_ids]
    shapes = {obj.Name: _global_shape(obj) for obj in all_components}
    v0_cache: dict[tuple[str, str], float] = {}
    stats = Stats()

    # Pass 1: axis (P2, unchanged) + initial category for every stander, from the BOM flags.
    axes: dict[str, tuple] = {}
    category: dict[str, int] = {}
    for obj in stander_objs:
        axis = _dominant_cylindrical_axis(shapes[obj.Name])
        if axis is None:
            raise ValueError(f"No cylindrical feature found on stander '{obj.Label}'")
        axes[obj.Name] = axis
        category[obj.Name] = {"screw": 1, "nut": 2}.get(bom_categories.get(obj.Label), 3)

    records: dict[str, StanderRecord] = {}

    # Pass 1.5: nut bore/partner info, computed early (before screws) purely so screws know which
    # nuts are partnered to them (P11 exception below) -- nothing else about nut processing
    # depends on screws, so moving this earlier changes no result, it only makes the mapping
    # available sooner. A nut demoted to category 3 here (no partner) gets no entry.
    nut_bore: dict[str, tuple[float, Part.Shape, float]] = {}  # name -> (bore_radius, bore_solid, bore_volume)
    partner_of: dict[str, str] = {}       # nut name -> partner name
    nuts_of_partner: dict[str, set[str]] = {}  # partner name -> set of nut names
    for obj in stander_objs:
        name = obj.Name
        if category[name] != 2:
            continue
        base, axis_dir, _radius = axes[name]
        shape = shapes[name]
        bore_radius = _bore_radius(shape, base, axis_dir)
        if bore_radius is None:
            raise ValueError(f"No internal (bore) cylindrical face found on nut stander '{obj.Label}'")
        t_min, t_max = _axial_extent(shape, base, axis_dir)
        bore_solid = _axial_solid(bore_radius, base, axis_dir, t_min, t_max)
        bore_volume = bore_solid.Volume
        partner = _find_partner(shape, bore_solid, bore_volume, all_components, shapes, config, name)
        if partner is None:
            log.info("Nut stander %r: nothing occupies its bore above pairing_min_fill (P7); "
                     "treating as category 3", obj.Label)
            category[name] = 3
            continue
        nut_bore[name] = (bore_radius, bore_solid, bore_volume)
        partner_of[name] = partner
        nuts_of_partner.setdefault(partner, set()).add(name)

    # Pass 2: screws. A nut's own partner always physically overlaps/grazes it along the axis as
    # it (un)screws -- that is not a blocking collision, it is already modeled by the nut's own
    # `partner`/`nut_rule` (P7/P8), so a screw excludes any nut partnered to it from its blocker
    # sweep, the same way a nut excludes its partner below. Every *other* stander is now a fair
    # candidate (P11 revised: a screw/nut can be blocked by an unrelated stander, confirmed on
    # `lunar_bodylager` where a nut's real removal path collides with an unrelated spacer stander
    # that categories 1/2 previously could never see).
    for obj in stander_objs:
        name = obj.Name
        if category[name] != 1:
            continue
        base, axis_dir, _radius = axes[name]
        shape = shapes[name]
        d = _signed_axis_direction(shape, axis_dir, config.probe_displacement, all_components,
                                   shapes, v0_cache, config, name)
        if d is None:
            log.warning("Screw stander %r: both axis sides free or both blocked (P4); treating "
                       "as category 3", obj.Label)
            category[name] = 3
            continue
        holds = _screw_holds(shape, axis_dir, non_standers, shapes, v0_cache, config, name)
        # P5 extension (2026-11-05): the screw only needs to travel screw_travel_factor times its
        # own length along d, or clear the whole-assembly bbox -- whichever comes first (plus the
        # usual sweep margin) -- rather than always sweeping all the way to the bbox.
        t_min, t_max = _axial_extent(shape, base, axis_dir)
        own_length = t_max - t_min
        bbox_exit = _bbox_exit_distance(shape, d, assembly_bbox)
        max_travel = min(config.screw_travel_factor * own_length, bbox_exit) + collision_config.margin
        own_nuts = nuts_of_partner.get(name, set())
        blocker_candidates = [o for o in all_components if o.Name not in own_nuts]
        blockers = _sweep_blockers(shape, d, blocker_candidates, shapes, v0_cache, collision_config,
                                   config, name, stats, max_length=max_travel) - holds
        exit_point = assembly_bbox.getIntersectionPoint(base, d)
        records[name] = StanderRecord(
            category=1,
            options=[RemovalOption(direction=d, exit_point=exit_point, blockers=blockers, max_travel=max_travel)],
            holds=holds, contact_neighbors=set(), bbox=_bbox_to_tuple(shape.BoundBox),
        )

    # Pass 3: nuts. The partner (P7, generalised 2026-11-04) can be any component -- stander or
    # movable -- so this pass needs nothing from pass 2; the precedence it derives (P8) lives
    # entirely on the nut's own record (`partner`/`nut_rule`), never mutating anyone else's.
    for obj in stander_objs:
        name = obj.Name
        if category[name] != 2:
            continue
        base, axis_dir, _radius = axes[name]
        shape = shapes[name]
        bore_radius, bore_solid, bore_volume = nut_bore[name]
        t_min, t_max = _axial_extent(shape, base, axis_dir)
        partner = partner_of[name]
        d = _signed_axis_direction(shape, axis_dir, config.probe_displacement, all_components,
                                   shapes, v0_cache, config, name)
        if d is None:
            log.warning("Nut stander %r: both axis sides free or both blocked (P4); treating as "
                       "category 3", obj.Label)
            category[name] = 3
            continue
        partner_rec = records.get(partner)
        if partner_rec is not None and len(partner_rec.options) == 1 and d.dot(partner_rec.options[0].direction) > 0:
            log.warning("Nut stander %r's direction is not opposite its partner %r's (P1 sanity "
                       "check); keeping it anyway", obj.Label, partner)
        # P11 revised (see pass 2 comment): every other component, standers included, is now a
        # candidate blocker -- except the partner itself, which always overlaps/grazes the nut
        # along its own axis and is already governed by `partner`/`nut_rule`, not by a translation
        # blocker (including it here would contradict `nut_rule` for at least one of the two).
        blocker_candidates = [o for o in all_components if o.Name != partner]
        blockers = _sweep_blockers(shape, d, blocker_candidates, shapes, v0_cache, collision_config,
                                   config, name, stats)
        clip_solid = _axial_solid(bore_radius + config.bore_clip_margin, base, axis_dir, t_min, t_max)
        partner_clip_shape = shapes[partner].cut(clip_solid)
        locked = _nut_rotation_locked(shape, base, axis_dir, stander_objs + non_standers, shapes,
                                      v0_cache, config, name, partner, partner_clip_shape)
        nut_rule = "locked" if locked else "free"
        exit_point = assembly_bbox.getIntersectionPoint(base, d)
        records[name] = StanderRecord(
            category=2, options=[RemovalOption(direction=d, exit_point=exit_point, blockers=blockers)],
            holds=set(), contact_neighbors=set(), bbox=_bbox_to_tuple(shape.BoundBox),
            partner=partner, nut_rule=nut_rule,
        )

    # Pass 4: everything else -- originally category 3, plus any screw/nut demoted above.
    for obj in stander_objs:
        name = obj.Name
        if category[name] != 3:
            continue
        base, axis_dir, _radius = axes[name]
        shape = shapes[name]
        others = [o for o in stander_objs + non_standers if o.Name != name]
        options = []
        probe_dir = None
        sweep_config = collision_config
        if name in compound_ids:
            # Stander subassembly (added 2026-11-11, docs/stander_processing-spec.md section 14):
            # try the cheap P4 probe-displacement sign test (same helper as screws/nuts) before
            # paying for a full sweep in both directions. Exactly one free side at the probe
            # displacement -- the same "is this the unique dismantle direction" test used for
            # screws -- is taken as that unique direction, and the opposite side's full sweep is
            # skipped entirely. Both free or both blocked: fall back to the original two-sweep pass.
            probe_dir = _signed_axis_direction(shape, axis_dir, config.probe_displacement,
                                               stander_objs + non_standers, shapes, v0_cache, config, name)
            # gap_skip off for this sweep (measured 2026-11-11 on lunar_motor_tire_complete:
            # distToShape() between two large fused shapes near contact cost 18-90 s per call,
            # vs. 0.7-3.5 s for common() on the same pair -- gap_skip's per-step distToShape()
            # dominated the whole sweep). Fixed-step common()-only checks still cover the entire
            # travel distance (unlike a single short jump, which could miss a blocker further down
            # the path), so this keeps the same correctness guarantee, just without the expensive
            # distance query.
            sweep_config = replace(collision_config, gap_skip=False)
        if probe_dir is not None:
            blockers = _sweep_blockers(shape, probe_dir, others, shapes, v0_cache, sweep_config,
                                       config, name, stats)
            exit_point = assembly_bbox.getIntersectionPoint(base, probe_dir)
            options = [RemovalOption(direction=probe_dir, exit_point=exit_point, blockers=blockers)]
            log.info("Stander subassembly %r: probe test (P4) found a unique free side; skipping "
                     "the opposite direction's full sweep", obj.Label)
        else:
            if name in compound_ids:
                log.info("Stander subassembly %r: probe test (P4) found both sides free or both "
                         "blocked; falling back to a full sweep in both directions", obj.Label)
            for sign in (1.0, -1.0):
                d = axis_dir * sign
                blockers = _sweep_blockers(shape, d, others, shapes, v0_cache, sweep_config,
                                           config, name, stats)
                exit_point = assembly_bbox.getIntersectionPoint(base, d)
                options.append(RemovalOption(direction=d, exit_point=exit_point, blockers=blockers))
        records[name] = StanderRecord(
            category=3, options=options, holds=set(), contact_neighbors=set(),
            bbox=_bbox_to_tuple(shape.BoundBox),
        )

    # contact_neighbors (P10, unchanged): every stander against the non-stander pool.
    for obj in stander_objs:
        name = obj.Name
        records[name] = replace(
            records[name], contact_neighbors=_contact_neighbors(shapes[name], non_standers, shapes, config),
        )

    n_screw = sum(1 for c in category.values() if c == 1)
    n_nut = sum(1 for c in category.values() if c == 2)
    n_other = sum(1 for c in category.values() if c == 3)
    log.info("Processed %d stander(s): %d screw, %d nut, %d other; %.2f s",
             len(stander_objs), n_screw, n_nut, n_other, sum(stats.times))

    return records, stander_ids, _bbox_to_tuple(assembly_bbox)



def build_stander_reduced_assembly(doc: "FreeCAD.Document", stander_ids: set[str]) -> "FreeCAD.Document":
    """Remove the stander components from the assembly"""
    reduced = FreeCAD.newDocument(f"{doc.Name}_reduced")
    for obj in _leaf_components(doc):
        if obj.Name in stander_ids:
            continue
        feature = reduced.addObject("Part::Feature", obj.Name)
        feature.Shape = _global_shape(obj)
        feature.Label = obj.Label
    reduced.recompute()
    return reduced



_MAIN_AXES = (
    FreeCAD.Vector(1, 0, 0), FreeCAD.Vector(-1, 0, 0),
    FreeCAD.Vector(0, 1, 0), FreeCAD.Vector(0, -1, 0),
    FreeCAD.Vector(0, 0, 1), FreeCAD.Vector(0, 0, -1),
)


def _direction_key(v: FreeCAD.Vector) -> tuple[float, float, float]:
    return (round(v.x, 9), round(v.y, 9), round(v.z, 9))


def _face_in_contact(face, obj_name: str, components, threshold: float) -> bool:
    """True if `face` is within `threshold` of some other component's shape.

    `components` are assumed already in the same (global) coordinates as `face`, as they are in
    `discover_candidate_directions` (shapes copied into the reduced assembly document are global,
    see `build_stander_reduced_assembly`). A bounding-box check skips the expensive `distToShape`
    call for every component pair that cannot possibly be within range.
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
    """Discover candidate dismantle directions from the reduced assembly, built up in three stages:

    1. The six main axes (+-x, +-y, +-z), always included first.
    2. Cylindrical-feature axes, but only from faces actually touching another component (within
       `config.contact_threshold`), deduplicated by exact signed-vector equality against
       everything already in the list (main axes + other cylindrical axes). No angular tolerance
       here -- the CAD kernel returns exact analytic values for this feature type.
    3. Planar-feature normals, same contact requirement as stage 2. Of the surviving faces, a
       normal is discarded if within `config.direction_min_angle_deg` of any direction already in
       the list (main axes, cylindrical axes, or a planar normal accepted earlier in this same
       pass). This angular pruning is what keeps near-duplicate directions from many slightly
       misaligned faces from blowing up the candidate count; stage 2 uses exact dedup only, no
       angular tolerance.

    A face touching nothing (stages 2 and 3 alike) is not a mating interface, so it is unlikely to
    be a real separation axis -- that contact requirement is the main source of count reduction;
    the angular pruning in stage 3 is secondary, on top of it.

    In both stages, the dedup check runs *before* the contact check, not after: deriving a face's
    direction and comparing it to the list already built is cheap, `_face_in_contact` is not, and
    most faces across a real assembly repeat a handful of directions. A direction that is already
    (or, for stage 3, within tolerance of) something in the list can never change the result
    regardless of contact, so there is nothing to gain by running the expensive check first.

    Contact is checked only within the reduced assembly (standers excluded, as direction sources
    and as contact partners alike). Checking against the removed standers too was tried and
    reverted: it did not recover any directions on `lunar_bodylager` (the actual mating contact
    there turns out to sit on fillet/blend faces, not the plain planar/cylindrical ones this scan
    looks at), and it cost a real amount of extra runtime for every assembly, including ones where
    it would never have mattered. If an assembly genuinely needs this, `direction_min_angle_deg` or
    `contact_threshold` are the tunables to revisit first.

    `skip_ids` (fused sub-assembly components under `compound_directions="none"`) contribute no
    directions from their internal geometry, at any stage.
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

    # 1. Main axes.
    for axis in _MAIN_AXES:
        add(axis)

    components = [obj for obj in _leaf_components(reduced_doc) if obj.Name not in skip_ids]

    # 2. Cylindrical features: only from faces touching another component, exact dedup only.
    # The direction is cheap to derive and compare; the contact check is not, so check first
    # whether this axis (both signs) is already in the list -- most faces across a real assembly
    # repeat a handful of directions, and a redundant one can never add anything regardless of
    # contact, so there is no need to pay for `_face_in_contact` to find that out.
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

    # 3. Planar features: same early-exit idea, but the dedup test is angular (within
    #    `min_angle` of the list built so far, not exact equality) -- skip the contact check only
    #    when both signs are already within tolerance of something already accepted.
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


def _has_nested_containers(doc: "FreeCAD.Document") -> bool:
    """True if the document has more than one top-level App::Part, or a nested one below it."""
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
    """Run Part 1 (standers) and Part 2 (candidate directions) on an already-loaded document.

    `doc` may be the flat, originally loaded assembly, or one level's document from
    `levels.build_level_document` (docs/subassembly-spec.md). `compound_ids` names this
    document's fused sub-assembly components, if any. `collision_config` and `bom_categories`
    (from `load_stander_categories`) feed Part 1's new screw/nut/other classification
    (docs/preprocessing-spec.md); both default to "every stander is category 3" if omitted.
    """
    config = config or PreprocessingConfig()
    if config.compound_directions not in ("none", "exterior"):
        raise ValueError(f"compound_directions must be 'none' or 'exterior', got {config.compound_directions!r}")
    members = members or {}

    stander_records, stander_ids, assembly_bbox = process_standers(
        doc, bom_labels, config, collision_config=collision_config, bom_categories=bom_categories,
        compound_ids=compound_ids)
    reduced_assembly = build_stander_reduced_assembly(doc, stander_ids)

    skip_ids = compound_ids if config.compound_directions == "none" else frozenset()
    candidate_directions = discover_candidate_directions(reduced_assembly, skip_ids=skip_ids, config=config)
    if not candidate_directions and compound_ids and config.compound_directions == "none":
        log.warning(
            "No candidate directions found with compound_directions='none' (every direction-bearing "
            "face belonged to a fused sub-assembly component); retrying with 'exterior'"
        )
        candidate_directions = discover_candidate_directions(reduced_assembly, skip_ids=frozenset(), config=config)

    return PreprocessingResult(
        reduced_assembly, stander_records, candidate_directions,
        assembly_bbox=assembly_bbox,
        step_path=step_path, bom_path=bom_path,
        level_id=level_id, level_label=level_label,
        compound_ids=frozenset(compound_ids), members=dict(members),
    )


# Main body of preprocessing: flat-assembly entry point. Loads the STEP assembly and BOM, then
# delegates to run_preprocessing_on_document. Only correct for a flat assembly (no nested
# sub-assemblies); a nested one should go through hierarchy.build_hierarchy + levels.solve_level
# instead (docs/subassembly-spec.md).
def run_preprocessing(
    step_path: str | Path,
    bom_path: str | Path,
    config: PreprocessingConfig | None = None,
    collision_config=None,
) -> PreprocessingResult:
    doc = load_assembly(step_path)
    bom_labels = load_bom(bom_path)
    bom_categories = load_stander_categories(bom_path)
    if _has_nested_containers(doc):
        log.warning(
            "This assembly has nested sub-assemblies (App::Part); run_preprocessing only sees the "
            "flat leaf list and ignores the hierarchy. Use hierarchy.build_hierarchy + "
            "levels.solve_level for a correct nested run (docs/subassembly-spec.md)."
        )
    return run_preprocessing_on_document(
        doc, bom_labels, config, collision_config=collision_config, bom_categories=bom_categories,
        step_path=Path(step_path), bom_path=Path(bom_path),
    )
