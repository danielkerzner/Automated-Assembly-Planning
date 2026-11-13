"""Tool check: does a stander's tool touch the rest of the assembly? (docs/robotic-gripper_spec.md section 7)

Independent module (also meant for human tools in manual mode later). Boolean only, no score.

Model: each tool is a STEP file, attached ("glued") to the stander along its disassembly axis in a single
orientation, and checked statically in the assembled state against every other component present in
that state, with the collision core's boolean (`collision._common_volume`). Contact = a common volume
above `RobotConfig.tool_contact_volume_tol` (touching faces give ~0 volume and do not count).

Conventions (confirmed with the author 2026-11-12; spec 13.4):
  - Tool STEP frame: the tool's working axis is its +Z axis through the origin, pointing from the robot
    flange towards the working element (bit, socket, pin), which connects to the stander in the max Z
    direction. Interface plane = the bbox max Z of the solid that the Z axis line meets highest. The
    robot body and its off-axis motor housing are taller but never on the axis. For a hollow socket the
    axis meets only the cavity floor, so the rim (that solid's bbox max Z) is the plane. Measured:
    screwing_effector 241 mm (bit), nut_effector 239 mm (socket), bushing_effector 235 mm (pin).
  - Engagement point on the stander: its outermost point along the removal direction d (screw head
    face, nut top), on the line through its centre of mass along d.
  - Overlap (`RobotConfig.tool_overlap_fraction`, per stander category): the stander sits this fraction
    of its own extent along d past the tool's interface plane. Screws 0 (the bit just touches the head),
    nuts and bushings 0.5 (a 12 mm nut has 6 mm inside the socket). The extent is measured along d from
    the stander's tessellation, which equals its bounding-box extent for an axis-aligned d.
  - Placement: tool +Z is turned onto -d (the tool comes from outside, along the removal direction)
    with the shortest rotation, so there is no extra spin about the axis; the interface plane sits at
    the engagement point moved by the overlap into the stander along -d.
  - Senses: every free direction of the stander's AoG edge is a sense; the tool is placed on the side
    the stander leaves towards. The stander itself is ignored in the contact test.
  - Tool per stander category: `RobotConfig.tools` (1 screw -> screwing_effector.step, 2 nut ->
    nut_effector.stp, 3 other/bushing -> bushing_effector.stp). A category mapped to None is reported
    `checked = False` and counts as feasible (the spec's stub behaviour).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import FreeCAD  # noqa: F401
import Part
from FreeCAD import Vector

import collision
from preprocessing import _bbox_overlap
from robot_config_robotic import RobotConfig

log = logging.getLogger("aap.robot.tools")


@dataclass
class Tool:
    name: str
    path: Path
    solids: list  # Part.Shape in the tool frame
    tip: Vector  # bit tip in the tool frame (on the +Z axis)


def load_tool(path: str | Path) -> Tool:
    import Import

    path = Path(path)
    doc = FreeCAD.newDocument(f"tool_{path.stem}".replace("-", "_"))
    try:
        Import.insert(str(path), doc.Name)
        doc.recompute()
        solids = []
        for obj in doc.Objects:
            if not obj.isDerivedFrom("Part::Feature") or obj.Shape.isNull() or not obj.Shape.Solids:
                continue
            if obj.InList and any(p.isDerivedFrom("Part::Feature") for p in obj.InList):
                continue  # a sub-shape of another feature
            shape = obj.Shape.copy()
            shape.Placement = obj.getGlobalPlacement()
            solids.extend(s.copy() for s in shape.Solids)
    finally:
        FreeCAD.closeDocument(doc.Name)
    if not solids:
        raise ValueError(f"Tool STEP {path} has no solids")
    zmin = min(s.BoundBox.ZMin for s in solids) - 10
    zmax = max(s.BoundBox.ZMax for s in solids) + 10
    axis = Part.LineSegment(Vector(0, 0, zmin), Vector(0, 0, zmax)).toShape()
    hit, element = None, None  # highest point where the Z axis meets material, and that solid
    for s in solids:
        try:
            cut = s.common(axis)
        except Exception:  # noqa: BLE001
            continue
        for v in cut.Vertexes:
            if hit is None or v.Point.z > hit:
                hit, element = v.Point.z, s
    if element is None:
        raise ValueError(f"Tool STEP {path}: its +Z axis meets no solid; cannot find the working element")
    # Interface plane: the element's max Z (a socket's rim, not its cavity floor). optimalBoundingBox, not BoundBox:
    # the plain BoundBox of these STEP solids can be far too loose (the robot body reports z 339.85, really 188).
    top = element.optimalBoundingBox().ZMax
    log.info("Tool %s: %d solid(s), working element meets the axis up to z = %.2f mm, interface plane z = %.2f mm",
             path.name, len(solids), hit, top)
    return Tool(name=path.stem, path=path, solids=solids, tip=Vector(0, 0, top))


def engagement(shape, d: Vector) -> tuple[Vector, float]:
    """(outermost point of the stander along d on the line through its centre of mass, its extent along d)."""
    pts, _ = shape.tessellate(0.1)
    proj = [p.dot(d) for p in pts]
    top, bottom = max(proj), min(proj)
    com = shape.CenterOfMass if hasattr(shape, "CenterOfMass") else shape.BoundBox.Center
    return com + d * (top - com.dot(d)), top - bottom


def engagement_point(shape, d: Vector) -> Vector:
    """Outermost point of the stander along d, on the line through its centre of mass along d."""
    return engagement(shape, d)[0]


def place_tool(tool: Tool, engage: Vector, d: Vector, depth: float = 0.0) -> list:
    d = Vector(d)
    d.normalize()
    rot = FreeCAD.Rotation(Vector(0, 0, 1), d * -1)
    tip_world = engage - d * depth
    base = tip_world - rot.multVec(tool.tip)
    pl = FreeCAD.Placement(base, rot)
    out = []
    for s in tool.solids:
        c = s.copy()
        c.Placement = pl.multiply(c.Placement)
        out.append(c)
    return out


@dataclass
class ToolStep:
    state: int
    component: int
    label: str
    tool: str | None
    checked: bool
    senses: dict = field(default_factory=dict)  # direction index -> contact (bool)
    contact_with: dict = field(default_factory=dict)  # direction index -> [labels touched]
    feasible: bool = True  # at least one free sense without contact (or not checked)
    sense: int | None = None  # the sense reported for the step

    def to_json(self, directions=None) -> dict:
        def vec(k):
            return None if directions is None else [float(v) for v in directions[k]]
        return {"state": self.state, "component": self.component, "label": self.label, "tool": self.tool,
                "checked": self.checked, "feasible": self.feasible,
                "sense": self.sense, "sense_direction": None if self.sense is None else vec(self.sense),
                "senses": [{"direction_index": int(k), "direction": vec(k), "contact": bool(v),
                            "contact_with": self.contact_with.get(k, [])} for k, v in self.senses.items()]}


class ToolChecker:
    """tool_contact(state, stander, sense) over AoG component indices; pairwise results are cached."""

    def __init__(self, shapes: list, labels: list[str], is_stander: list[bool], categories: dict[int, int],
                 directions, config: RobotConfig):
        self.shapes, self.labels, self.is_stander = shapes, labels, list(is_stander)
        self.categories, self.directions, self.config = categories, directions, config
        self._tools: dict[int, Tool | None] = {}
        self._placed: dict[tuple[int, int], list] = {}
        self._pair: dict[tuple[int, int, int], bool] = {}
        self._bboxes: dict = {}
        self._steps: dict[tuple[int, int, int | None], ToolStep] = {}

    def tool_for(self, c: int) -> Tool | None:
        cat = self.categories.get(c)
        if cat not in self._tools:
            fname = self.config.tools.get(cat) if cat is not None else None
            if fname is None:
                self._tools[cat] = None
            else:
                p = Path(fname)
                p = p if p.is_absolute() else Path(self.config.tool_dir) / p
                self._tools[cat] = load_tool(p)
        return self._tools[cat]

    def _tool_solids(self, c: int, k: int) -> list:
        if (c, k) not in self._placed:
            tool = self.tool_for(c)
            d = Vector(*[float(v) for v in self.directions[k]])
            d.normalize()
            engage, extent = engagement(self.shapes[c], d)
            depth = float(self.config.tool_overlap_fraction.get(self.categories.get(c), 0.0)) * extent
            self._placed[(c, k)] = place_tool(tool, engage, d, depth)
        return self._placed[(c, k)]

    def _tight_bbox(self, key, shape):
        """optimalBoundingBox, cached: the plain BoundBox can be far too loose (B-spline faces), which only
        costs extra exact booleans in the pre-filter below, never a wrong answer."""
        if key not in self._bboxes:
            self._bboxes[key] = shape.optimalBoundingBox()
        return self._bboxes[key]

    def _touches(self, c: int, k: int, o: int) -> bool:
        key = (c, k, o)
        if key not in self._pair:
            other = self.shapes[o]
            other_bb = self._tight_bbox(("comp", o), other)
            hit = False
            for idx, s in enumerate(self._tool_solids(c, k)):
                if not _bbox_overlap(self._tight_bbox(("tool", c, k, idx), s), other_bb):
                    continue
                try:
                    vol = collision._common_volume(s, other)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Tool common() failed for %s vs %s (%s); counted as contact", self.labels[c],
                                self.labels[o], exc)
                    vol = float("inf")
                if vol > self.config.tool_contact_volume_tol:
                    hit = True
                    break
            self._pair[key] = hit
        return self._pair[key]

    def tool_contact(self, state: int, c: int, k: int) -> bool:
        """True = the tool on stander c, approaching along removal direction k, touches a present component."""
        if self.tool_for(c) is None:
            return False
        return bool(self._contacts(state, c, k))

    def _contacts(self, state: int, c: int, k: int) -> list[str]:
        return [self.labels[o] for o in range(len(self.shapes))
                if o != c and (state >> o) & 1 and self._touches(c, k, o)]

    def step(self, state: int, c: int, senses: list[int], preferred: int | None = None) -> ToolStep:
        key = (int(state), int(c), preferred)
        if key in self._steps:
            return self._steps[key]
        tool = self.tool_for(c)
        out = ToolStep(state=int(state), component=int(c), label=self.labels[c],
                       tool=None if tool is None else tool.name, checked=tool is not None)
        if tool is None:
            out.sense = preferred if preferred is not None else (senses[0] if senses else None)
        else:
            for k in senses:
                hits = self._contacts(state, c, int(k))
                out.senses[int(k)] = bool(hits)
                if hits:
                    out.contact_with[int(k)] = hits
            free = [k for k, hit in out.senses.items() if not hit]
            out.feasible = bool(free)
            out.sense = preferred if preferred in free else (free[0] if free else preferred)
        self._steps[key] = out
        return out
