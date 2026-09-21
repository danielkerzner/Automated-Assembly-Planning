"""Preprocessing: stander handling + candidate direction discovery.

Implements docs/preprocessing-spec.md. Sequential, not parallel: Part 2 runs on
the stander-reduced assembly Part 1 produces, not on the original.

Geometry backend is FreeCAD (Part module). A "component" is a leaf Part::Feature
object produced by importing a STEP assembly with ReadShapeCompoundMode off, so
every distinct solid becomes its own object instead of one compound per file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import FreeCAD
import Part
import Import


@dataclass
class PreprocessingConfig:
    # Radial distance beyond a stander's shaft surface used for the contact
    # walk (spec Part 1 step 3). Open/tunable per spec — start small, calibrate
    # against the Vulkan case study.
    radial_threshold: float = 1.0
    # Intersection volume below this is treated as numerical noise from
    # touching (not interfering) geometry, not a real contact.
    volume_epsilon: float = 1e-6
    # Minimum gap below which a non-stander component counts as physically
    # touching the stander (assembly-order contact, not the directional
    # dismantle-path walk). Tunable — start small, calibrate against Vulkan.
    contact_threshold: float = 0.01


@dataclass
class StanderRecord:
    prerequisites: set[str]
    exit_point: FreeCAD.Vector
    dismantle_direction: FreeCAD.Vector
    # Every non-stander component in direct contact with this stander,
    # independent of dismantle direction — what must already be assembled
    # before this stander can be added.
    contact_neighbors: set[str]


@dataclass
class PreprocessingResult:
    reduced_assembly: "FreeCAD.Document"
    stander_records: dict[str, StanderRecord]
    candidate_directions: list[FreeCAD.Vector]


def load_assembly(step_path: str | Path) -> "FreeCAD.Document":
    """Import a STEP assembly with one Part::Feature per solid (not one compound)."""
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


def _leaf_components(doc: "FreeCAD.Document") -> list:
    return [
        obj
        for obj in doc.Objects
        if obj.TypeId == "Part::Feature"
        and getattr(obj, "Shape", None) is not None
        and not obj.Shape.isNull()
        and len(obj.Shape.Solids) > 0
    ]


def _global_shape(obj) -> Part.Shape:
    shape = obj.Shape.copy()
    shape.Placement = obj.getGlobalPlacement()
    return shape


def _assembly_bounding_box(objs):
    bbox = None
    for obj in objs:
        b = _global_shape(obj).BoundBox
        bbox = b if bbox is None else bbox.united(b)
    return bbox


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
    shape = _global_shape(stander_obj)
    axis = _dominant_cylindrical_axis(shape)
    if axis is None:
        raise ValueError(f"No cylindrical feature found on stander '{stander_obj.Label}'")
    base, direction, radius = axis
    probe_radius = radius + config.radial_threshold

    # Never test a stander against itself or against other standers (spec
    # Part 1 step 5: stander-stander contact must not drive ordering).
    candidates = [
        obj
        for obj in all_components
        if obj.Name != stander_obj.Name and obj.Name not in stander_ids
    ]

    per_direction = {}
    for sign in (1.0, -1.0):
        d = direction * sign
        extent = max(d.dot(v.Point - base) for v in shape.Vertexes)
        probe_start = base + d * extent
        exit_point = assembly_bbox.getIntersectionPoint(probe_start, d)
        height = (exit_point - probe_start).Length
        if height <= 0:
            per_direction[sign] = (0, 0.0, set())
            continue
        probe = Part.makeCylinder(probe_radius, height, probe_start, d)
        per_direction[sign] = _collect_axis_collisions(probe, candidates, config)

    count_pos, volume_pos, _ = per_direction[1.0]
    count_neg, volume_neg, _ = per_direction[-1.0]
    if count_pos != count_neg:
        winner = 1.0 if count_pos < count_neg else -1.0
    else:
        winner = 1.0 if volume_pos <= volume_neg else -1.0

    chosen_direction = direction * winner
    prerequisites = per_direction[winner][2]
    exit_point = assembly_bbox.getIntersectionPoint(base, chosen_direction)
    contact_neighbors = _contact_neighbors(shape, candidates, config)

    return StanderRecord(
        prerequisites=prerequisites,
        exit_point=exit_point,
        dismantle_direction=chosen_direction,
        contact_neighbors=contact_neighbors,
    )


def process_standers(
    doc: "FreeCAD.Document", bom_labels: set[str], config: PreprocessingConfig
) -> tuple[dict[str, StanderRecord], set[str]]:
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
    return records, stander_ids


def build_stander_reduced_assembly(doc: "FreeCAD.Document", stander_ids: set[str]) -> "FreeCAD.Document":
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
    surf = face.Surface
    if surf.TypeId in ("Part::GeomCylinder", "Part::GeomPlane"):
        return FreeCAD.Vector(surf.Axis).normalize()
    return None


def discover_candidate_directions(reduced_doc: "FreeCAD.Document") -> list[FreeCAD.Vector]:
    seen: dict[tuple[float, float, float], FreeCAD.Vector] = {}
    for obj in _leaf_components(reduced_doc):
        for face in obj.Shape.Faces:
            direction = _face_direction(face)
            if direction is None:
                continue
            for candidate in (direction, -direction):
                key = (round(candidate.x, 9), round(candidate.y, 9), round(candidate.z, 9))
                seen.setdefault(key, candidate)
    return list(seen.values())


def run_preprocessing(
    step_path: str | Path,
    bom_path: str | Path,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    config = config or PreprocessingConfig()
    doc = load_assembly(step_path)
    bom_labels = load_bom(bom_path)
    stander_records, stander_ids = process_standers(doc, bom_labels, config)
    reduced_assembly = build_stander_reduced_assembly(doc, stander_ids)
    candidate_directions = discover_candidate_directions(reduced_assembly)
    return PreprocessingResult(reduced_assembly, stander_records, candidate_directions)
