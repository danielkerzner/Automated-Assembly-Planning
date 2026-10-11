"""Preprocessing: stander handling + candidate direction discovery."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import FreeCAD
import Part
import Import


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
    # dismantle-path walk).
    contact_threshold: float = 0.01

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


def _collect_axis_collisions(probe: Part.Shape, candidates, config: PreprocessingConfig):
    """Return (count, volume, colliding_ids) of candidates that intersect the probe."""
    count = 0
    volume = 0.0
    colliding_ids: set[str] = set()
    for obj in candidates:
        intersection = probe.common(_global_shape(obj))
        if intersection.Volume > config.volume_epsilon:
            count += 1
            volume += intersection.Volume
            colliding_ids.add(obj.Name)
    return count, volume, colliding_ids


def _contact_neighbors(stander_shape: Part.Shape, candidates, config: PreprocessingConfig) -> set[str]:
    """Return the set of candidate component Names that are in contact with the stander."""
    neighbors: set[str] = set()
    for obj in candidates:
        distance, _, _ = stander_shape.distToShape(_global_shape(obj))
        if distance <= config.contact_threshold:
            neighbors.add(obj.Name)
    return neighbors


def determine_dismantle_direction(
    stander_obj,
    all_components: list,
    stander_ids: set[str],
    assembly_bbox,
    config: PreprocessingConfig,
) -> StanderRecord:
    """Determine the dismantle direction for a stander object."""
    shape = _global_shape(stander_obj)
    axis = _dominant_cylindrical_axis(shape)
    if axis is None:
        raise ValueError(f"No cylindrical feature found on stander '{stander_obj.Label}'")
    base, direction, radius = axis
    probe_radius = radius + config.radial_threshold

    # Never test a stander against itself or against other standers.
    candidates = [
        obj
        for obj in all_components
        if obj.Name != stander_obj.Name and obj.Name not in stander_ids
    ]

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
        per_direction[sign] = _collect_axis_collisions(probe, candidates, config)

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
    contact_neighbors = _contact_neighbors(shape, candidates, config)

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

    records = {
        obj.Name: determine_dismantle_direction(
            obj, all_components, stander_ids, assembly_bbox, config
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



def _face_direction(face) -> FreeCAD.Vector | None:
    """Determine the direction of a face if it is cylindrical or planar."""
    surf = face.Surface
    if surf.TypeId in ("Part::GeomCylinder", "Part::GeomPlane"):
        return FreeCAD.Vector(surf.Axis).normalize()
    return None


def discover_candidate_directions(reduced_doc: "FreeCAD.Document") -> list[FreeCAD.Vector]:
    seen: dict[tuple[float, float, float], FreeCAD.Vector] = {}
    """Discover candidate dismantle directions from the reduced assembly."""
    for obj in _leaf_components(reduced_doc):
        for face in obj.Shape.Faces:
            direction = _face_direction(face)
            if direction is None:
                continue
            for candidate in (direction, -direction):
                key = (round(candidate.x, 9), round(candidate.y, 9), round(candidate.z, 9))
                seen.setdefault(key, candidate)
    return list(seen.values())


# Main body of preprocessing. 
# Loads the STEP assembly and BOM, processes standers, 
# builds a reduced assembly, and discovers candidate directions.
def run_preprocessing(
    step_path: str | Path,
    bom_path: str | Path,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    config = config or PreprocessingConfig()
    doc = load_assembly(step_path)
    bom_labels = load_bom(bom_path)
    stander_records, stander_ids, assembly_bbox = process_standers(doc, bom_labels, config)
    reduced_assembly = build_stander_reduced_assembly(doc, stander_ids)
    candidate_directions = discover_candidate_directions(reduced_assembly)
    return PreprocessingResult(
        reduced_assembly, stander_records, candidate_directions,
        assembly_bbox=assembly_bbox,
        step_path=Path(step_path), bom_path=Path(bom_path),
    )
