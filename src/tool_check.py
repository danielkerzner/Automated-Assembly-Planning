"""Tool check: does a stander's tool touch the rest of the assembly? (docs/robotic-gripper_spec.md section 7)

Independent module (also meant for human tools in manual mode later). Boolean only, no score.

Model: each tool is a STEP file, attached ("glued") to the stander along its disassembly axis in a single
orientation, and checked statically in the assembled state against every other component present in
that state, with the collision core's boolean (`collision._common_volume`). Contact = a common volume
above `RobotConfig.tool_contact_volume_tol` (touching faces give ~0 volume and do not count).

Conventions (defined here, 2026-11-12, to confirm with the author -- spec 7 left them open):
  - Tool STEP frame: the tool's working axis is its +Z axis through the origin, pointing from the robot
    flange towards the bit. The bit tip is the highest point where the Z axis line meets the tool's
    solids (computed, not hard-coded; for tools/gripper/screwing_effector.step it is about z = 241 mm).
  - Engagement point on the stander: its outermost point along the removal direction d (the screw head
    face), on the line through its centre of mass along d.
  - Placement: tool +Z is turned onto -d (the tool comes from outside, along the removal direction)
    with the shortest rotation, so there is no extra spin about the axis; the bit tip sits on the
    engagement point, moved `tool_engage_depth_mm` into the stander along -d (default 0).
  - Senses: every free direction of the stander's AoG edge is a sense; the tool is placed on the side
    the stander leaves towards.
  - Tool per stander category: `RobotConfig.tools` (1 screw -> screwing_effector.step; nut and other
    tools do not exist yet). A stander without a tool is reported `checked = False` and counts as
    feasible (the spec's stub behaviour).
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
    top = None
    for s in solids:
        try:
            cut = s.common(axis)
        except Exception:  # noqa: BLE001
            continue
        for v in cut.Vertexes:
            if top is None or v.Point.z > top:
                top = v.Point.z
    if top is None:
        raise ValueError(f"Tool STEP {path}: its +Z axis meets no solid; cannot find the bit tip")
    log.info("Tool %s: %d solid(s), bit tip at z = %.2f mm", path.name, len(solids), top)
    return Tool(name=path.stem, path=path, solids=solids, tip=Vector(0, 0, top))


def engagement_point(shape, d: Vector) -> Vector:
    """Outermost point of the stander along d, on the line through its centre of mass along d."""
    pts, _ = shape.tessellate(0.1)
    top = max(p.dot(d) for p in pts)
    com = shape.CenterOfMass if hasattr(shape, "CenterOfMass") else shape.BoundBox.Center
    return com + d * (top - com.dot(d))


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
            engage = engagement_point(self.shapes[c], d)
            self._placed[(c, k)] = place_tool(tool, engage, d, self.config.tool_engage_depth_mm)
        return self._placed[(c, k)]

    def _touches(self, c: int, k: int, o: int) -> bool:
        key = (c, k, o)
        if key not in self._pair:
            other = self.shapes[o]
            hit = False
            for s in self._tool_solids(c, k):
                if not _bbox_overlap(s.BoundBox, other.BoundBox):
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
