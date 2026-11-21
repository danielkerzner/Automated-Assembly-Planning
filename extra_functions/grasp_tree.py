"""Draw the robot-mode grasp results as a disassembly tree with thumbnails (AND/OR-graph style).

[SUPP] Visualisation script, not part of the planning pipeline.

Inputs
    - `<output_dir>/plan.json` written by `main.py --mode robot` (needs its `robot` block).
    - The STEP file the plan was made from (only for the thumbnails).
    - Options: `--chosen-only`, `--deflection` (mm, default 1.0), `--save <file>`, `--part-size` (default 0.6).
Outputs
    - A matplotlib figure (shown, and written to `--save` if given). Nothing else is written.

How it works
    `collect` gathers the (state, component) edges of every recorded sequence from plan.json, `layout`
    places them in rows (one row per number of removed components, children under the barycentre of
    their parents), `_thumbnail` renders each node's components, and `draw_tree` draws boxes, arrows and
    score labels.

Reads `<output_dir>/plan.json` (written by `main.py --mode robot`) and the STEP file. Shows the part of the AoG that
the robot loop explored (the union of every recorded sequence; `--chosen-only` for just the chosen one):

  rows            level k = k components removed; the full assembly at the top
  framed node     a sub-assembly (AoG state), with a thumbnail of its components. A one-component state shows
                  that part's score alone.
  leaf node       the component removed by one AoG edge, drawn next to the remaining sub-assembly, with its
                  score alone (S_alone)
  edge label      on the edge to the removed part: its grasp score in that state (S), coloured red -> yellow ->
                  green; "-" = no grasp (S = 0 or not evaluated because the part fails even alone),
                  "tool" = a standard removed by a tool (no grasp score): green = the tool check passed,
                  red = the tool collides in every free sense. A red frame = infeasible step.
  plain edge      parent -> remaining sub-assembly
  sub-assembly    its own grasp score (red frame = not graspable), ONLY if the plan was made with
  score           `main.py --mode robot --check-subassembly-grasp` (robot.subassembly_scores); a plan from a
                  run without that flag has no sub-assembly scores, and those boxes stay blank
  yellow box      standards
  bold edges      the chosen sequence

Arrows point upward (assembly direction), as in the original sketch. Only edges a recorded sequence used are
drawn: the loop evaluates (state, component) edges only when a candidate sequence uses them, so other AoG edges
have no score.

Run in environmentAAP from the repo root:

    python extra_functions/grasp_tree.py output/industrial_fastner_robot_nodes data/industrial_fastner/industrial_fastner.stp
    python extra_functions/grasp_tree.py <output_dir> <step> --chosen-only
    python extra_functions/grasp_tree.py <output_dir> <step> --save tree.png      # also write the image
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

LIGHT = np.array([0.4, -0.5, 0.75])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def _meshes(step_path: str, ids: set[str], deflection: float) -> dict:
    """[AUX] Tessellate the listed components of the STEP at global placement: component id -> (vertices, triangles)."""
    from preprocessing import _global_shape, _leaf_components, load_assembly

    doc = load_assembly(step_path)
    out = {}
    for obj in _leaf_components(doc):
        if obj.Name in ids:
            V, T = _global_shape(obj).tessellate(deflection)
            out[obj.Name] = (np.array([[v.x, v.y, v.z] for v in V], float), np.array(T, int).reshape(-1, 3))
    return out


def _thumbnail(meshes: dict, ids: list[str], color, px: int = 140, elev: float = 25, azim: float = -60,
               part_size: float = 1.0) -> np.ndarray:
    """[AUX] RGBA image of the given components, Lambert-shaded, autoscaled to them. `part_size` scales the drawn
    components inside the (unchanged) image: 1.0 = filling it, 0.6 = 60 % of that, with more empty margin."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgb
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = Figure(figsize=(px / 100, px / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    pts = []
    for cid in ids:
        if cid not in meshes:
            continue
        V, T = meshes[cid]
        tri = V[T]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        shade = 0.35 + 0.65 * np.abs(n @ LIGHT)
        base = np.array(to_rgb(color(cid) if callable(color) else color))
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(shade[:, None] * base, 0, 1), edgecolor="none"))
        pts.append(V)
    if pts:
        P = np.vstack(pts)
        c, r = (P.min(0) + P.max(0)) / 2, float((P.max(0) - P.min(0)).max()) / 2 * 0.85 / max(part_size, 1e-3)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba()).copy()


def _score_color(s):
    """[AUX] Colour of a grasp score 0..1 on a red -> yellow -> green map."""
    import matplotlib
    return matplotlib.colormaps["RdYlGn"](0.1 + 0.85 * float(np.clip(s, 0, 1)))


def collect(plan: dict, chosen_only: bool = False):
    """[MAIN] Read the explored AoG edges from plan.json.

    Returns (bit -> id, id -> label, standard ids, edges {(S, bit): step json}, chosen edge set). An edge is
    (AoG state before the removal, component bit); the last part alone is not an edge (it is the one-component
    state node). With `chosen_only` only the chosen sequence's edges are kept.
    """
    lsteps = plan["level"]["steps"]
    bit_of = {s["component_id"]: s["component"] for s in lsteps}
    comp_id = {b: c for c, b in bit_of.items()}
    label = {s["component_id"]: s["component_label"] for s in lsteps}
    standards = {s["component_id"] for s in lsteps if s["is_standard"]}
    edges, chosen = {}, set()
    for seq in plan["robot"]["loop"]["sequences"]:
        if chosen_only and not seq["chosen"]:
            continue
        for st in seq["steps"]:
            key = (int(st["state"]), bit_of[st["component_id"]])
            if bin(key[0]).count("1") <= 1:
                continue  # the last part alone: shown as the one-component state node
            edges.setdefault(key, st)
            if seq["chosen"]:
                chosen.add(key)
    return comp_id, label, standards, edges, chosen


ITEM_GAP = 1.0  # centre distance of the boxes inside one group (removed parts + the sub-assembly they leave)
GROUP_GAP = 0.7  # extra space between groups
ROW_GAP = 1.45  # vertical distance between rows


def layout(edges: dict, n: int):
    """[MAIN] Rows of items, ("state", S) or ("part", (S, bit)), and their x positions.

    Each row is a sequence of groups: the parts removed into one sub-assembly, then that sub-assembly. Groups are
    ordered by the mean x of their parents (barycentre) and placed so the sub-assembly sits under its parents where
    possible, pushed right only as far as needed to avoid overlaps; each row is then shifted to keep the average
    displacement from those targets at zero."""
    root = max({s for s, _ in edges}, key=lambda s: bin(s).count("1"))
    level = lambda s: n - bin(s).count("1")
    children: dict[int, list[tuple[int, int]]] = {}
    for (s, b) in edges:
        children.setdefault(level(s) + 1, []).append((s, b))
    x = {("state", root): 0.0}
    rows = [[("state", root)]]
    for L in range(level(root) + 1, n):
        es = children.get(L, [])
        if not es:
            break
        targets: dict[int, list[tuple[int, int]]] = {}
        for s, b in es:
            targets.setdefault(s & ~(1 << b), []).append((s, b))
        bary = {t: float(np.mean([x[("state", s)] for s, _ in inc])) for t, inc in targets.items()}
        row, placed, wanted = [], [], []
        right = -np.inf
        for t in sorted(targets, key=lambda t: (bary[t], t)):
            parts = [("part", e) for e in sorted(targets[t], key=lambda e: x[("state", e[0])])]
            group = parts + [("state", t)]
            # the sub-assembly (last box) under its parents' barycentre; parts to its left
            left = max(bary[t] - ITEM_GAP * len(parts), right + GROUP_GAP + ITEM_GAP if row else -np.inf)
            for k, item in enumerate(group):
                x[item] = left + ITEM_GAP * k
            right = x[group[-1]]
            placed.append(x[group[-1]])
            wanted.append(bary[t])
            row.extend(group)
        shift = float(np.mean(np.array(wanted) - np.array(placed)))
        for item in row:
            x[item] += shift
        rows.append(row)
    return rows, x


def draw_tree(output_dir: str | Path, step_path: str | Path, chosen_only: bool = False, deflection: float = 1.0,
              save: str | None = None, show: bool = True, part_size: float = 0.6):
    """[MAIN] Draw the tree for `<output_dir>/plan.json` and return the figure.

    `part_size`: size of the drawn components inside their (fixed-size) boxes, 1.0 = filling the thumbnail.
    Order: layout, edges (arrows and score labels) first so they lie underneath, then the node boxes with
    thumbnails and scores, row numbers and the legend title.
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import FancyBboxPatch

    plan = json.loads((Path(output_dir) / "plan.json").read_text(encoding="utf-8"))
    if "robot" not in plan:
        raise ValueError(f"{output_dir}/plan.json has no 'robot' block; run main.py with --mode robot first")
    comp_id, label, standards, edges, chosen = collect(plan, chosen_only)
    if not edges:
        raise ValueError("No recorded steps to draw")
    n = len(comp_id)
    alone = {label_id: v for label_id, v in plan["robot"].get("alone_scores", {}).items()}  # keyed by label
    meshes = _meshes(str(step_path), set(comp_id.values()), deflection)
    rows, x = layout(edges, n)

    xs = np.array(list(x.values()))
    x_lo, x_hi = float(xs.min()) - 0.6, float(xs.max()) + 0.6
    if x_hi - x_lo < 4:  # leave room for the title on narrow trees
        mid = (x_lo + x_hi) / 2
        x_lo, x_hi = mid - 2, mid + 2
    height = ROW_GAP * (len(rows) - 1) + 1.2
    fig, ax = plt.subplots(figsize=(1.25 * (x_hi - x_lo) + 0.8, 1.25 * height + 1.0))
    ax.set_xlim(x_lo - 0.4, x_hi)
    ax.set_ylim(-ROW_GAP * (len(rows) - 1) - 0.6, 0.75)
    ax.set_aspect("equal")  # square node boxes whatever the tree's width
    ax.set_axis_off()
    row_of = {item: i for i, r in enumerate(rows) for item in r}
    y = lambda item: -ROW_GAP * row_of[item]
    thumbs: dict = {}

    def ids_of(state):
        return [comp_id[b] for b in range(n) if (state >> b) & 1]

    def node(item, ids, color, text, standard=False, frame="black", lw=0.8):
        key = (tuple(ids), str(color))
        if key not in thumbs:
            thumbs[key] = _thumbnail(meshes, ids, color, part_size=part_size)
        cx, cy = x[item], y(item)
        ax.add_patch(FancyBboxPatch((cx - 0.42, cy - 0.36), 0.84, 0.72, boxstyle="round,pad=0.02",
                                    facecolor="#fff59d" if standard else "#f4f6fa", edgecolor=frame, linewidth=lw,
                                    zorder=1))
        ax.add_artist(AnnotationBbox(OffsetImage(thumbs[key], zoom=0.42), (cx, cy), frameon=False, zorder=2))
        if text:
            ax.text(cx, cy, text, ha="center", va="center", fontsize=8, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"))

    def arrow(child, parent, bold):
        ax.annotate("", xy=(x[parent], y(parent) - 0.38), xytext=(x[child], y(child) + 0.38),
                    arrowprops=dict(arrowstyle="-|>", color="black" if bold else "#b0b4ba", lw=2.0 if bold else 0.7,
                                    shrinkA=0, shrinkB=0), zorder=0.5 if bold else 0)

    # Edges first (underneath), then nodes.
    for (s, b), st in edges.items():
        t = s & ~(1 << b)
        bold = (s, b) in chosen
        arrow(("state", t), ("state", s), bold)
        arrow(("part", (s, b)), ("state", s), bold)
        # label near the part's box (30% of the way up), so it reads as that part's and crossings rarely collide
        (px, py), (qx, qy) = (x[("part", (s, b))], y(("part", (s, b))) + 0.38), (x[("state", s)], y(("state", s)) - 0.38)
        mx, my = px + 0.3 * (qx - px), py + 0.3 * (qy - py)
        if st["end_effector"] != "gripper":
            # the tool check alone (part_feasible), not the edge's sub-assembly checks: green = tool fits, red = contact
            txt, fc = "tool", ("#66bb6a" if st.get("part_feasible", st["feasible"]) else "#e53935")
        elif st.get("score") is None or st["score"] <= 0:
            txt, fc = "-", "#b71c1c"
        else:
            txt, fc = f"{st['score']:.2f}", _score_color(st["score"])
        ax.text(mx, my, txt, ha="center", va="center", fontsize=8, zorder=4,
                color="white" if txt == "-" else "black",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=fc, edgecolor="black" if bold else "gray",
                          linewidth=1.4 if bold else 0.5))

    # Optional sub-assembly check (robot.subassembly_scores): sub-assembly nodes show their own grasp score.
    sub = plan["robot"].get("subassembly_scores") or {}
    if not sub:
        print(f"Note: {output_dir}/plan.json has no sub-assembly scores (robot.subassembly_scores is empty); "
              "sub-assembly boxes are left blank. Re-run main.py with --mode robot --check-subassembly-grasp "
              "to get them.")
    for r in rows:
        for item in r:
            kind, v = item
            if kind == "state":
                ids = ids_of(v)
                ns = sub.get(str(v))
                if len(ids) == 1:  # a single part left
                    a = alone.get(label[ids[0]])
                    ok = ns is None or ns["feasible"]
                    node(item, ids, "#5e8f6b", "" if a is None else f"{a:.2f}", standard=ids[0] in standards,
                         frame="black" if ok else "#c0392b", lw=0.8 if ok else 1.8)
                elif ns is not None:
                    node(item, ids, "#9aa3ad", f"{ns['score']:.2f}", frame="black" if ns["feasible"] else "#c0392b",
                         lw=0.8 if ns["feasible"] else 1.8)
                else:
                    node(item, ids, "#9aa3ad", "")
            else:
                s, b = v
                cid = comp_id[b]
                st = edges[(s, b)]
                a = alone.get(label[cid])
                ok = st.get("part_feasible", st["feasible"])  # the part's own check (edge feasibility may add nodes)
                col = "#c9a227" if cid in standards else (_score_color(st["score"]) if st.get("score") else "#c0392b")
                node(item, [cid], col, "" if a is None else f"{a:.2f}", standard=cid in standards,
                     frame="black" if ok else "#c0392b", lw=0.8 if ok else 1.8)

    for i in range(len(rows)):
        ax.text(x_lo - 0.1, -ROW_GAP * i, str(i), ha="right", va="center", fontsize=16)
    n_seq = len(plan["robot"]["loop"]["sequences"])
    what = "chosen sequence" if chosen_only else f"{n_seq} recorded sequences"
    sub_txt = ("sub-assembly node = its own grasp score (red frame = not graspable)\n" if sub else "")
    ax.set_title(f"{plan['assembly']}: grasp scores ({what}, chosen in bold)\n"
                 f"part node = score alone; edge label = score in that state\n{sub_txt}"
                 f"yellow = standard; \"tool\" green = tool removal possible, red = tool collides\n"
                 f"red frame on a part = infeasible part step", fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    if show:
        plt.show()
    return fig


def main() -> int:
    """[AUX] Command line (arguments in the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", help="folder with the robot-mode plan.json")
    ap.add_argument("step_path", help="the STEP file the plan was made from")
    ap.add_argument("--chosen-only", action="store_true", help="draw only the chosen sequence")
    ap.add_argument("--deflection", type=float, default=1.0, help="tessellation deflection in mm (larger = faster)")
    ap.add_argument("--save", help="also write the figure to this image file")
    ap.add_argument("--part-size", type=float, default=0.6,
                    help="size of the components inside their boxes (1.0 = filling the thumbnail; default 0.6)")
    args = ap.parse_args()
    draw_tree(args.output_dir, args.step_path, args.chosen_only, args.deflection, args.save,
              part_size=args.part_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
