"""Show how a robot tool sits on its stander (docs/robotic-gripper_spec.md section 7 / 13.4).

Places the tool exactly as `src/tool_check.py` does (same functions: interface plane, overlap rule, placement)
and draws it in two panels:

  left   3D view around the stander: stander blue, tool dark grey, other present components light grey,
         those the tool collides with red. Only triangles near the stander are drawn (fast, readable).
  right  cross-section through the tool axis (outlines): this is where you see whether the tool's pin or
         socket really fits the stander (a pin inside the nut's bore, a bit on the screw head, ...).

The title gives the numbers: how much the tool overlaps the stander itself (the contact test ignores the
stander, so only this tells you whether the pin fits; a pin inside a bore overlaps ~0 mm^3) and the
collision volume with every other present component.

The stander's tool, approach direction (sense) and assembly state are read from a robot-mode plan.json, so no
preprocessing run is needed. Keys: right / n = next sense of this stander, left / p = previous, q = close.
Nothing is written to disk unless --save is given.

Run in environmentAAP from the repo root:

    python extra_functions/visualize_tool.py output/industrial_fastner_robot data/industrial_fastner/industrial_fastner.stp industrial_fastner_part7
    python extra_functions/visualize_tool.py <output_dir> <step> <stander label> --removed industrial_fastner_part6
    python extra_functions/visualize_tool.py <output_dir> <step> <stander label> --all-present --save tool.png

Options: --removed LABEL...  components to treat as already removed (all others present);
         --all-present       every component present (fully assembled);
         default             the state of the stander's step in the chosen sequence (else any recorded one).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _find_step(plan: dict, label: str) -> dict:
    for s in plan["robot"]["steps"]:
        if s["component_label"] == label:
            return s
    for seq in plan["robot"]["loop"]["sequences"]:
        for s in seq["steps"]:
            if s["component_label"] == label:
                return s
    raise ValueError(f"{label!r} is not a step of any recorded sequence in plan.json")


def _mesh(shape, deflection: float):
    V, T = shape.tessellate(deflection)
    return np.array([[v.x, v.y, v.z] for v in V], float).reshape(-1, 3), np.array(T, int).reshape(-1, 3)


def _crop(V, T, lo, hi):
    """Triangles with all three corners in the box [lo, hi] (a centroid test lets long slivers poke out)."""
    if not len(T):
        return np.zeros((0, 3, 3))
    tri = V[T]
    keep = np.all((tri >= lo) & (tri <= hi), axis=(1, 2))
    return tri[keep]


def _section_lines(shape, plane_face, origin, e_axis, e_perp, samples: int = 60):
    """Outline of shape cut by plane_face, as 2D polylines (coordinates along e_axis, e_perp)."""
    try:
        sec = shape.section(plane_face)
    except Exception:  # noqa: BLE001
        return []
    lines = []
    for edge in sec.Edges:
        try:
            pts = edge.discretize(Number=samples)
        except Exception:  # noqa: BLE001
            continue
        P = np.array([[p.x, p.y, p.z] for p in pts]) - origin
        lines.append(np.c_[P @ e_axis, P @ e_perp])
    return lines


def show_tool(output_dir, step_path, stander_label: str, removed: list[str] | None = None, all_present: bool = False,
              deflection: float = 0.3, save: str | None = None, show: bool = True):
    import FreeCAD  # noqa: F401
    import Part
    from FreeCAD import Vector
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    import collision
    from preprocessing import _global_shape, _leaf_components, load_assembly
    from robot_config_robotic import RobotConfig
    from tool_check import engagement, load_tool, place_tool

    plan = json.loads((Path(output_dir) / "plan.json").read_text(encoding="utf-8"))
    if "robot" not in plan:
        raise ValueError(f"{output_dir}/plan.json has no 'robot' block; run main.py with --mode robot first")
    step = _find_step(plan, stander_label)
    if step["end_effector"] == "gripper":
        raise ValueError(f"{stander_label!r} is a gripper step, not a tool step")
    tool_name = step.get("tool")
    if not tool_name:
        raise ValueError(f"{stander_label!r} has no tool in plan.json (category mapped to None)")
    cfg = RobotConfig()
    cat = next((c for c, f in cfg.tools.items() if f and Path(f).stem == tool_name), None)
    if cat is None:
        raise ValueError(f"tool {tool_name!r} is not in RobotConfig.tools {cfg.tools}")
    tool = load_tool(Path(cfg.tool_dir) / cfg.tools[cat])
    fraction = float(cfg.tool_overlap_fraction.get(cat, 0.0))

    # components and the state
    lsteps = plan["level"]["steps"]
    comp_id = {s["component"]: s["component_id"] for s in lsteps}
    label = {s["component_id"]: s["component_label"] for s in lsteps}
    by_label = {v: k for k, v in label.items()}
    doc = load_assembly(str(step_path))
    shapes = {o.Name: _global_shape(o) for o in _leaf_components(doc) if o.Name in label}
    me = step["component_id"]
    if all_present or removed:
        gone = {by_label[r] for r in (removed or []) if r in by_label}
        bad = [r for r in (removed or []) if r not in by_label]
        if bad:
            raise ValueError(f"unknown labels in --removed: {bad}")
        present = [cid for cid in label if cid not in gone]
        state_txt = "all present" if not gone else "removed: " + ", ".join(label[g] for g in gone)
    else:
        present = [comp_id[b] for b in comp_id if (step["state"] >> b) & 1]
        state_txt = f"state of the recorded step (state {step['state']})"
    others = [cid for cid in present if cid != me]
    senses = step["senses"]
    st_shape = shapes[me]
    st_vol = st_shape.Volume

    # place the tool once per sense, measure overlaps
    cases = []
    for sense in senses:
        d = Vector(*sense["direction"])
        d.normalize()
        engage, extent = engagement(st_shape, d)
        placed = place_tool(tool, engage, d, fraction * extent)
        own = sum(collision._common_volume(s, st_shape) for s in placed)
        hits = {}
        for cid in others:
            v = sum(collision._common_volume(s, shapes[cid]) for s in placed)
            if v > cfg.tool_contact_volume_tol:
                hits[cid] = v
        cases.append(dict(d=np.array(sense["direction"], float), engage=np.array([engage.x, engage.y, engage.z]),
                          extent=extent, placed=placed, own=own, hits=hits))

    bb = st_shape.optimalBoundingBox()
    c0 = np.array([bb.Center.x, bb.Center.y, bb.Center.z])
    half = max(bb.XLength, bb.YLength, bb.ZLength) * 1.6 + 10
    lo, hi = c0 - half, c0 + half
    st_tri = _crop(*_mesh(st_shape, deflection), lo, hi)
    other_tri = {cid: _crop(*_mesh(shapes[cid], deflection), lo, hi) for cid in others}

    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    state = {"i": max(0, next((i for i, s in enumerate(senses) if s["direction_index"] == step.get("sense")), 0))}

    def draw():
        ax.clear()
        ax2.clear()
        case = cases[state["i"]]
        for cid, tri in other_tri.items():
            if len(tri):
                hit = cid in case["hits"]
                ax.add_collection3d(Poly3DCollection(tri, facecolor="tab:red" if hit else "lightgrey",
                                                     edgecolor="none", alpha=0.55 if hit else 0.15))
        ax.add_collection3d(Poly3DCollection(st_tri, facecolor="tab:blue", edgecolor="none", alpha=0.75))
        for s in case["placed"]:
            tri = _crop(*_mesh(s, deflection), lo, hi)
            if len(tri):
                ax.add_collection3d(Poly3DCollection(tri, facecolor="dimgray", edgecolor="none", alpha=0.45))
        e, d = case["engage"], case["d"]
        ax.quiver(*e, *(d * half * 0.8), color="black", arrow_length_ratio=0.1, linewidth=1.5)
        for setlim, k in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setlim(lo[k], hi[k])
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")

        # section through the axis: the plane spanned by d and the world axis most perpendicular to it
        world = np.eye(3)
        u = world[int(np.argmin(np.abs(world @ d)))]
        u = u - (u @ d) * d
        u /= np.linalg.norm(u)
        n = np.cross(d, u)
        size = 4 * half
        corner = e - d * size / 2 - u * size / 2
        face = Part.makePlane(size, size, Vector(*corner), Vector(*n), Vector(*d))
        for cid in others:
            col = "tab:red" if cid in case["hits"] else "#9aa0a6"
            for ln in _section_lines(shapes[cid], face, e, d, u):
                ax2.plot(ln[:, 0], ln[:, 1], color=col, linewidth=1.0 if cid not in case["hits"] else 1.6)
        for ln in _section_lines(st_shape, face, e, d, u):
            ax2.plot(ln[:, 0], ln[:, 1], color="tab:blue", linewidth=1.8)
        for s in case["placed"]:
            for ln in _section_lines(s, face, e, d, u):
                ax2.plot(ln[:, 0], ln[:, 1], color="black", linewidth=1.3)
        iface = -fraction * case["extent"]  # interface plane along the axis, relative to the engagement point
        ax2.axvline(0, color="tab:blue", linestyle=":", linewidth=1)
        ax2.axvline(iface, color="black", linestyle="--", linewidth=1)
        ax2.axhline(0, color="gray", linestyle="-.", linewidth=0.6)
        ax2.set_xlim(-half, half)
        ax2.set_ylim(-half, half)
        ax2.set_aspect("equal")
        ax2.set_xlabel("along the removal direction from the stander's outer face [mm]  (tool comes from +)")
        ax2.set_ylabel(f"across, along world {'xyz'[int(np.argmax(np.abs(u)))]} [mm]")
        ax2.set_title("Section through the tool axis: blue stander, black tool, red = collision, grey = others\n"
                      "dotted: stander's outer face, dashed: tool interface plane", fontsize=9)

        hits = ", ".join(f"{label[c]} {v:.1f} mm^3" for c, v in case["hits"].items()) or "none"
        fig.suptitle(f"{label[me]} with {tool.name} from {np.round(d, 3).tolist()}  "
                     f"(sense {state['i'] + 1}/{len(cases)}, {state_txt})\n"
                     f"tool inside the stander itself: {case['own']:.1f} mm^3 ({100 * case['own'] / st_vol:.1f} % of its "
                     f"volume; ignored by the contact test)  |  overlap rule {fraction:g} x {case['extent']:.2f} mm\n"
                     f"collisions with present components: {hits}", fontsize=9)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "n"):
            state["i"] = (state["i"] + 1) % len(cases)
        elif event.key in ("left", "p"):
            state["i"] = (state["i"] - 1) % len(cases)
        else:
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    manager = getattr(fig.canvas, "manager", None)  # no Ctrl + wheel / pinch zoom (matplotlib >= 3.10)
    if manager is not None and getattr(manager, "scroll_handler_id", None) is not None:
        fig.canvas.mpl_disconnect(manager.scroll_handler_id)
        manager.scroll_handler_id = None
    draw()
    for case in cases:
        print(f"{label[me]} with {tool.name} from {case['d'].tolist()}: tool inside the stander {case['own']:.2f} mm^3 "
              f"({100 * case['own'] / st_vol:.1f} %), collisions: "
              + (", ".join(f"{label[c]} {v:.1f} mm^3" for c, v in case['hits'].items()) or "none"))
    if save:
        fig.savefig(save, dpi=150)
    if show:
        plt.show()
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", help="folder with the robot-mode plan.json")
    ap.add_argument("step_path", help="the STEP file the plan was made from")
    ap.add_argument("stander", help="Label of the stander, e.g. industrial_fastner_part7")
    ap.add_argument("--removed", nargs="+", help="components already removed (all others present)")
    ap.add_argument("--all-present", action="store_true", help="every component present")
    ap.add_argument("--deflection", type=float, default=0.3, help="tessellation deflection in mm")
    ap.add_argument("--save", help="also write the figure to this image file")
    args = ap.parse_args()
    show_tool(args.output_dir, args.step_path, args.stander, args.removed, args.all_present, args.deflection, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
