"""Interactive viewer for the robot-mode grasps of the chosen sequence (docs/robotic-gripper_spec.md section 8).

Reads `<output_dir>/plan.json` (written by `main.py --mode robot`) and the STEP file, and shows one step at a
time in a matplotlib 3D window, in assembly order:

  - grey, transparent      the components present in the step's state (the gripper's obstacles)
  - green / red            the component of the step (feasible / infeasible)
  - dashed arrow           its removal motion (opposite the insertion `direction`, `travel` mm long)
  - point + 3 arrows       the TCP and the grasp frame: X red (edge normal), Y green (edge direction),
                           Z blue (plane normal = closing direction)
  - two dark rectangles    the finger pads, schematic: at TCP +/- grasp_width/2 along Z, sized from the
                           gripper YAML (a_pg along Y, b_pg along X); not the real finger geometry

Steps without a pose (infeasible gripper steps, tool steps) show only the parts and the motion arrow.

Right panel: per gripper step, the score with the part alone (S_alone, light blue) and in the step's assembled
state (S, green = feasible / red = infeasible; "n/a" = not evaluated because the part fails even alone), the
absolute floor (dashed) and the relative threshold x S_alone (black tick on the S bar); the current step is
highlighted. Tool steps are marked "tool".
Keys: right / n = next step, left / p = previous step, q = close. Nothing is written to disk.

Run in environmentAAP from the repo root:

    python extra_functions/visualize_grasps.py output/industrial_fastner_robot data/industrial_fastner/industrial_fastner.stp
    python extra_functions/visualize_grasps.py <output_dir> <step> --step 4 --deflection 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _finger_size(plan: dict) -> tuple[float, float]:
    """(pad size along Y, pad size along X) in mm from the gripper YAML in plan.json, else Franka defaults."""
    try:
        import yaml
        path = plan["robot"]["engine"]["gripper_yaml"]
        with open(path, "r", encoding="utf-8") as f:
            p = yaml.safe_load(f)
        return float(p["a_pg"]), float(p["b_pg"])
    except Exception:  # noqa: BLE001
        return 25.0, 18.0


def _meshes(step_path: str, ids: set[str], deflection: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """component id (obj.Name) -> (vertices (n, 3), triangles (m, 3)) at global placement."""
    from preprocessing import _global_shape, _leaf_components, load_assembly

    doc = load_assembly(step_path)
    out = {}
    for obj in _leaf_components(doc):
        if obj.Name not in ids:
            continue
        V, T = _global_shape(obj).tessellate(deflection)
        out[obj.Name] = (np.array([[v.x, v.y, v.z] for v in V], float), np.array(T, int).reshape(-1, 3))
    missing = ids - set(out)
    if missing:
        print(f"warning: no shape found in the STEP for {sorted(missing)} (nested assembly? robot mode is flat only)")
    return out


def show_grasps(output_dir: str | Path, step_path: str | Path, start_step: int = 1, deflection: float = 0.5) -> None:
    """Open the viewer on `<output_dir>/plan.json` (robot mode) with the shapes of `step_path`."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plan = json.loads((Path(output_dir) / "plan.json").read_text(encoding="utf-8"))
    if "robot" not in plan:
        raise ValueError(f"{output_dir}/plan.json has no 'robot' block; run main.py with --mode robot first")
    lsteps, rsteps = plan["level"]["steps"], plan["robot"]["steps"]
    comp_id = {s["component"]: s["component_id"] for s in lsteps}  # AoG bit -> obj.Name
    label = {s["component_id"]: s["component_label"] for s in lsteps}
    meshes = _meshes(str(step_path), set(comp_id.values()), deflection)
    pad_y, pad_x = _finger_size(plan)

    allv = np.vstack([v for v, _ in meshes.values()])
    lo, hi = allv.min(0), allv.max(0)
    center, span = (lo + hi) / 2, float((hi - lo).max())
    arrow = 0.3 * span

    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    axs = fig.add_subplot(1, 2, 2)
    state = {"i": min(max(start_step, 1), len(rsteps)) - 1}
    thr = plan["robot"].get("thresholds", {})

    def draw_scores(i):
        """Per step: S in the assembled state and S_alone, with the feasibility rule; current step highlighted."""
        axs.clear()
        x = np.arange(len(rsteps))
        w = 0.38
        floor, rel = thr.get("grasp_abs_floor"), thr.get("grasp_rel_threshold")
        for k, rs in enumerate(rsteps):
            if rs["end_effector"] != "gripper":
                axs.text(k, 0.02, "tool", ha="center", va="bottom", fontsize=7, rotation=90, color="gray")
                continue
            alone, ctx = rs.get("score_alone") or 0.0, rs.get("score")
            axs.bar(k - w / 2, alone, w, color="lightsteelblue", edgecolor="steelblue",
                    label="S alone" if k == 0 or "S alone" not in axs.get_legend_handles_labels()[1] else None)
            if ctx is None:  # not evaluated: fails even alone
                axs.text(k + w / 2, 0.02, "n/a", ha="center", va="bottom", fontsize=7, rotation=90, color="tab:red")
            else:
                axs.bar(k + w / 2, ctx, w, color="tab:green" if rs["feasible"] else "tab:red",
                        label=None if "S in state" in axs.get_legend_handles_labels()[1] else "S in state")
                axs.text(k + w / 2, ctx + 0.01, f"{ctx:.2f}", ha="center", va="bottom", fontsize=7)
            axs.text(k - w / 2, alone + 0.01, f"{alone:.2f}", ha="center", va="bottom", fontsize=7)
            if rel is not None and alone > 0:  # S must also reach rel * S_alone
                axs.plot([k + 0.02, k + w + 0.02 - 0.04], [rel * alone] * 2, color="black", linewidth=1.5,
                         label=None if "rel. threshold x S alone" in axs.get_legend_handles_labels()[1]
                         else "rel. threshold x S alone")
        if floor is not None:
            axs.axhline(floor, color="black", linestyle="--", linewidth=1, label=f"absolute floor {floor}")
        axs.axvspan(i - 0.5, i + 0.5, color="gold", alpha=0.25, zorder=0)
        axs.set_xticks(x)
        names = [label.get(s["component_id"], s["component_id"]) for s in rsteps]
        prefix = os.path.commonprefix(names) if len(names) > 1 else ""
        cut = max(prefix.rfind("_"), prefix.rfind("-")) + 1  # keep whole words ("part3", not "3")
        axs.set_xticklabels([f"{s['step']}\n{n[cut:] or n}" for s, n in zip(rsteps, names)], fontsize=7)
        axs.set_ylim(0, 1.08)
        axs.set_ylabel("grasp score (engine, 0..1)")
        axs.set_title("Grasp scores per step (assembly order); green/red = feasible/infeasible", fontsize=9)
        axs.legend(fontsize=7, loc="upper right")

    def draw():
        ax.clear()
        i = state["i"]
        draw_scores(i)
        ls, rs = lsteps[i], rsteps[i]
        me = rs["component_id"]
        present = [cid for bit, cid in comp_id.items() if (rs["state"] >> bit) & 1]
        for cid in present:
            if cid not in meshes:
                continue
            V, T = meshes[cid]
            own = cid == me
            color = ("tab:green" if rs["feasible"] else "tab:red") if own else "lightgrey"
            ax.add_collection3d(Poly3DCollection(V[T], facecolor=color, edgecolor="none",
                                                 alpha=0.65 if own else 0.12))
        if me in meshes:  # removal motion: from the part's centre, opposite the insertion direction
            c = meshes[me][0].mean(0)
            d = -np.asarray(ls["direction"], float) * ls["travel"]
            ax.quiver(*c, *d, color="black", linestyle="--", arrow_length_ratio=0.08, linewidth=1.2)
        info = ""
        if rs.get("tcp") is not None:
            tcp = np.asarray(rs["tcp"], float)
            X, Y, Z = (np.asarray(a, float) for a in rs["axes"])
            ax.scatter(*tcp, color="black", s=30)
            for a, col in ((X, "red"), (Y, "green"), (Z, "blue")):
                ax.quiver(*tcp, *(a * arrow), color=col, arrow_length_ratio=0.15, linewidth=2)
            for sgn in (1, -1):
                pc = tcp + sgn * rs["grasp_width"] / 2 * Z
                corners = [pc + sx * pad_x / 2 * X + sy * pad_y / 2 * Y for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
                ax.add_collection3d(Poly3DCollection([corners], facecolor="dimgray", edgecolor="black", alpha=0.8))
            info = (f"S={rs['score']:.3f}  S_rel={rs['score_relative']:.3f}  width={rs['grasp_width']:.1f} mm"
                    + (f"  rotation checked {rs['angles']}" if rs.get("rotation_checked") else ""))
        elif rs["end_effector"] == "gripper":
            info = f"no grasp: {rs.get('reason')}"
        else:
            info = f"tool step ({rs['end_effector']}), sense {rs.get('sense_direction')}"
        ax.set_title(f"Step {ls['step']}/{len(lsteps)}: {label.get(me, me)}  [{rs['end_effector']}, "
                     f"{'feasible' if rs['feasible'] else 'INFEASIBLE'}]\n{info}\n"
                     "(right/n next, left/p previous, q close)", fontsize=9)
        for setlim, k in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setlim(center[k] - span * 0.6, center[k] + span * 0.6)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "n"):
            state["i"] = (state["i"] + 1) % len(rsteps)
        elif event.key in ("left", "p"):
            state["i"] = (state["i"] - 1) % len(rsteps)
        else:
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", help="folder with the robot-mode plan.json")
    ap.add_argument("step_path", help="the STEP file the plan was made from")
    ap.add_argument("--step", type=int, default=1, help="assembly step to open first (default 1)")
    ap.add_argument("--deflection", type=float, default=0.5, help="tessellation deflection in mm (larger = faster)")
    args = ap.parse_args()
    show_grasps(args.output_dir, args.step_path, args.step, args.deflection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
