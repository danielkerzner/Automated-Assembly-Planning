"""Preprocessing: stander handling + candidate direction discovery."""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import FreeCAD
import Part
import Import

log = logging.getLogger("aap.preprocessing")


# Dataclass for preprocessing configuration parameters. All values are tunable, but the
# default values are based on the industrial fastner case study.
@dataclass
class PreprocessingConfig:
    # Radial distance beyond a stander's shaft surface used for the contact
    # walk (spec Part 1 step 3).
    radial_threshold: float = 1.0
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

# Dataclass for storing the contact and dismantle information for a stander component. 
# This is used to store the prerequisites, exit point, dismantle direction, and contact neighbors.
@dataclass
class StanderRecord:
    # Components physically along the stander's chosen exit path (directional collision
    # walk). Per the AoG stander-precedence rule (docs/aog-spec.md D4), a stander is
    # assembled after these are in place, so in disassembly it must come off BEFORE
    # them, while they are still present -- not the other way around.
    prerequisites: set[str]

    #Exit point of the stander in the assembly. Used for creating the dismatle path.
    exit_point: FreeCAD.Vector

    #Dismantle direction of the stander.
    dismantle_direction: FreeCAD.Vector

    # All non-stander components physically touching the stander (full contact walk,
    # not directional). Same precedence direction as prerequisites: the stander must be
    # removed before these, not after.
    contact_neighbors: set[str]

    #Global (min, max) bounding box of the stander's own shape inside the assembly.
    #Needed by the AoG travel model, which has no other access to stander geometry.
    bbox: tuple[FreeCAD.Vector, FreeCAD.Vector] | None = None

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
                if flag not in ("1", "true", "yes", "y", "t"):
                    continue
            standers.add(label)
    return standers

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


def _collect_axis_collisions(
    probe: Part.Shape, candidates: list, candidate_shapes: dict[str, Part.Shape], config: PreprocessingConfig
):
    """Return (count, volume, colliding_ids) of candidates that intersect the probe."""
    probe_bbox = probe.BoundBox
    count = 0
    volume = 0.0
    colliding_ids: set[str] = set()
    for obj in candidates:
        shape = candidate_shapes[obj.Name]
        if not _bbox_overlap(probe_bbox, shape.BoundBox):
            continue
        intersection = probe.common(shape)
        if intersection.Volume > config.volume_epsilon:
            count += 1
            volume += intersection.Volume
            colliding_ids.add(obj.Name)
    return count, volume, colliding_ids


def _contact_neighbors(
    stander_shape: Part.Shape, candidates: list, candidate_shapes: dict[str, Part.Shape], config: PreprocessingConfig
) -> set[str]:
    """Return the set of candidate component Names that are in contact with the stander."""
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


def determine_dismantle_direction(
    stander_obj,
    candidates: list,
    candidate_shapes: dict[str, Part.Shape],
    assembly_bbox,
    config: PreprocessingConfig,
) -> StanderRecord:
    """Determine the dismantle direction for a stander object.

    `candidates` (and their precomputed global `candidate_shapes`) are the non-stander leaf
    components -- the same list for every stander, computed once by `process_standers` rather
    than per call, since copying+placing a shape is not free and there is nothing stander-specific
    about it.
    """
    shape = _global_shape(stander_obj)
    axis = _dominant_cylindrical_axis(shape)
    if axis is None:
        raise ValueError(f"No cylindrical feature found on stander '{stander_obj.Label}'")
    base, direction, radius = axis
    probe_radius = radius + config.radial_threshold

    # Identify colisions with the probe in both directions along the axis.
    # The direction with fewer collisions is preferred.
    per_direction = {}
    for sign in (1.0, -1.0):

        # Compute the start point of the probe along the axis.
        d = direction * sign
        extent = max(d.dot(v.Point - base) for v in shape.Vertexes)
        probe_start = base + d * extent

        #Compute the exit point of the probe from the assembly bounding box.
        exit_point = assembly_bbox.getIntersectionPoint(probe_start, d)
        height = (exit_point - probe_start).Length
        if height <= 0:
            per_direction[sign] = (0, 0.0, set())
            continue

        #Compute the probe shape as a cylinder extending from the probe start point to the exit point.
        #Calculate the amount of interference with the candidate components in that direction.
        probe = Part.makeCylinder(probe_radius, height, probe_start, d)
        per_direction[sign] = _collect_axis_collisions(probe, candidates, candidate_shapes, config)

    #Choose the direction with fewer collisions, or if equal, the one with less total volume of interference.
    count_pos, volume_pos, _ = per_direction[1.0]
    count_neg, volume_neg, _ = per_direction[-1.0]
    if count_pos != count_neg:
        winner = 1.0 if count_pos < count_neg else -1.0
    else:
        winner = 1.0 if volume_pos <= volume_neg else -1.0

    chosen_direction = direction * winner
    prerequisites = per_direction[winner][2]

    #Recalculate exit point in the chosen direction for the final record.
    exit_point = assembly_bbox.getIntersectionPoint(base, chosen_direction)
    contact_neighbors = _contact_neighbors(shape, candidates, candidate_shapes, config)

    return StanderRecord(
        prerequisites=prerequisites,
        exit_point=exit_point,
        dismantle_direction=chosen_direction,
        contact_neighbors=contact_neighbors,
        bbox=_bbox_to_tuple(shape.BoundBox),
    )


def process_standers(
    doc: "FreeCAD.Document", bom_labels: set[str], config: PreprocessingConfig
) -> tuple[dict[str, StanderRecord], set[str], tuple[FreeCAD.Vector, FreeCAD.Vector]]:
    """Return stander records, the set of stander object Names, and the whole-assembly bbox."""
    all_components = _leaf_components(doc)
    stander_objs = [obj for obj in all_components if obj.Label in bom_labels]
    stander_ids = {obj.Name for obj in stander_objs}
    assembly_bbox = _assembly_bounding_box(all_components)

    # Standers never test against each other (spec: stander-stander contact is ignored for
    # ordering), so every stander's candidate pool is exactly the same non-stander leaf set --
    # compute it, and its global shapes, once rather than per stander.
    candidates = [obj for obj in all_components if obj.Name not in stander_ids]
    candidate_shapes = {obj.Name: _global_shape(obj) for obj in candidates}

    records = {
        obj.Name: determine_dismantle_direction(
            obj, candidates, candidate_shapes, assembly_bbox, config
        )
        for obj in stander_objs
    }
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
    document's fused sub-assembly components, if any.
    """
    config = config or PreprocessingConfig()
    if config.compound_directions not in ("none", "exterior"):
        raise ValueError(f"compound_directions must be 'none' or 'exterior', got {config.compound_directions!r}")
    members = members or {}

    stander_records, stander_ids, assembly_bbox = process_standers(doc, bom_labels, config)
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
) -> PreprocessingResult:
    doc = load_assembly(step_path)
    bom_labels = load_bom(bom_path)
    if _has_nested_containers(doc):
        log.warning(
            "This assembly has nested sub-assemblies (App::Part); run_preprocessing only sees the "
            "flat leaf list and ignores the hierarchy. Use hierarchy.build_hierarchy + "
            "levels.solve_level for a correct nested run (docs/subassembly-spec.md)."
        )
    return run_preprocessing_on_document(
        doc, bom_labels, config, step_path=Path(step_path), bom_path=Path(bom_path),
    )
