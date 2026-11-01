"""Acceptance tests for the Illustration Generator (docs/illustration-spec.md, L1-L13).

Standalone runner (pytest is not installed in environmentAAP). From repo root:

    python automatic_tests/test_illustration.py                       # core group (no FreeCAD run)
    python automatic_tests/test_illustration.py --tests L1 L5         # individual tests
    python automatic_tests/test_illustration.py --groups core real    # + industrial_fastner end to end (L11)
    python automatic_tests/test_illustration.py --groups core scale   # + L13 at N = 20 (core runs it at N = 8)

Groups:
  core   synthetic meshes (boxes, a ring, a rod through a plate, a closed enclosure, spheres); no FreeCAD.
  real   L11: run_preprocessing on industrial_fastner + the cached optimized sequence
         (tests/industrial_fastner/optimization/optimization_result.json, read only), then generate_illustrations;
         a second run with another colour checks that meshes and layers are reused.
  scale  L13 at N = 20 with the default image sizes.

The suite never writes to tests/industrial_fastner/; a final "cache" check confirms the files it reads are byte-identical.
Scratch output goes to automatic_tests/output/illustration_test/; results to automatic_tests/output/illustration_results.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import tracemalloc
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import illustration_generator as ig  # noqa: E402
from illustration_generator import IllustrationConfig, ViewContext  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
SCRATCH = OUT_DIR / "illustration_test"
FASTENER_DIR = REPO_ROOT / "tests" / "industrial_fastner"
FASTENER_STEP = REPO_ROOT / "data" / "industrial_fastner" / "industrial_fastner.stp"
FASTENER_BOM = REPO_ROOT / "data" / "industrial_fastner" / "bom.csv"
SEQUENCE_FILE = FASTENER_DIR / "optimization" / "optimization_result.json"
# Fallback when the committed cache is absent: the industrial_fastner sequence written by test_optimization.py O13.
SEQUENCE_FALLBACK = OUT_DIR / "optimization_test" / "pipeline" / "optimization" / "optimization_result.json"

SMALL = dict(image_size=(480, 360), search_size=(240, 180))


# ---------------------------------------------------------------------------
# Tiny test harness (same shape as test_optimization.py)
# ---------------------------------------------------------------------------


class Recorder:
    def __init__(self):
        self.results: dict[str, dict] = {}
        self._current: str | None = None

    def start(self, test_id: str):
        self._current = test_id
        self.results[test_id] = {"passed": True, "checks": [], "notes": []}

    def check(self, ok: bool, label: str):
        entry = self.results[self._current]
        entry["checks"].append({"label": label, "passed": bool(ok)})
        if not ok:
            entry["passed"] = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}", flush=True)

    def note(self, text: str):
        self.results[self._current]["notes"].append(text)
        print(f"  note: {text}", flush=True)

    def error(self, exc: BaseException):
        entry = self.results[self._current]
        entry["passed"] = False
        entry["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"  [ERROR] {entry['error']}", flush=True)
        traceback.print_exc()


CACHED_FILES = ["optimization/optimization_result.json"]


def cache_snapshot() -> dict[str, str | None]:
    import hashlib
    out = {}
    for name in CACHED_FILES:
        path = FASTENER_DIR / name
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return out


def fresh_dir(name: str) -> Path:
    path = SCRATCH / name
    shutil.rmtree(path, ignore_errors=True)
    return path


def raises(fn, needle: str) -> bool:
    try:
        fn()
    except ValueError as exc:
        return needle in str(exc)
    return False


# ---------------------------------------------------------------------------
# Synthetic meshes and sequences
# ---------------------------------------------------------------------------

BOX_FACES = [(0, 1, 3), (0, 3, 2), (4, 7, 5), (4, 6, 7), (0, 4, 5), (0, 5, 1), (2, 7, 6), (2, 3, 7),
             (0, 2, 6), (0, 6, 4), (1, 7, 3), (1, 5, 7)]


def box(lo, hi, key="") -> ig.Mesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = [[(lo, hi)[ix][0], (lo, hi)[iy][1], (lo, hi)[iz][2]] for ix in (0, 1) for iy in (0, 1) for iz in (0, 1)]
    return ig.mesh_from_arrays(V, BOX_FACES, key or f"box{lo.tolist()}{hi.tolist()}")


def ring(center, r_in, r_out, h, n=48, key="") -> ig.Mesh:
    """Annulus prism about +z (a ring-shaped part)."""
    c = np.asarray(center, float)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circ = np.stack([np.cos(a), np.sin(a), np.zeros(n)], axis=1)
    V = np.concatenate([c + r_out * circ, c + r_out * circ + [0, 0, h], c + r_in * circ, c + r_in * circ + [0, 0, h]])
    ob, ot, ib, it = 0, n, 2 * n, 3 * n
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [(ob + i, ob + j, ot + j), (ob + i, ot + j, ot + i)]  # outer wall
        F += [(ib + i, it + j, ib + j), (ib + i, it + i, it + j)]  # inner wall
        F += [(it + i, ot + i, ot + j), (it + i, ot + j, it + j)]  # top
        F += [(ib + i, ob + j, ob + i), (ib + i, ib + j, ob + j)]  # bottom
    return ig.mesh_from_arrays(V, F, key or f"ring{c.tolist()}{r_in}{r_out}{h}")


def sphere(center, radius, n_lat=40, n_lon=80, key="") -> ig.Mesh:
    c = np.asarray(center, float)
    th = np.linspace(0, np.pi, n_lat + 1)[1:-1]
    ph = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    ring_pts = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1).reshape(-1, 3)
    V = np.concatenate([[[0, 0, 1]], ring_pts, [[0, 0, -1]]]) * radius + c
    south = len(V) - 1
    F = []
    for j in range(n_lon):
        k = (j + 1) % n_lon
        F.append((0, 1 + j, 1 + k))
        F.append((south, 1 + (n_lat - 2) * n_lon + k, 1 + (n_lat - 2) * n_lon + j))
        for i in range(n_lat - 2):
            a, b = 1 + i * n_lon + j, 1 + i * n_lon + k
            F += [(a, a + n_lon, b + n_lon), (a, b + n_lon, b)]
    return ig.mesh_from_arrays(V, F, key or f"sphere{c.tolist()}{radius}{n_lat}{n_lon}")


def merged(meshes, key="merged") -> ig.Mesh:
    V, F, off = [], [], 0
    for m in meshes:
        V.append(m.vertices)
        F.append(m.triangles + off)
        off += len(m.vertices)
    return ig.mesh_from_arrays(np.concatenate(V), np.concatenate(F), key)


def enclosure(lo, hi, t, key="enclosure") -> ig.Mesh:
    """Closed hollow box made of six wall plates of thickness t."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    walls = []
    for ax in range(3):
        a, b = lo.copy(), hi.copy()
        b[ax] = lo[ax] + t
        walls.append(box(a, b))
        a, b = lo.copy(), hi.copy()
        a[ax] = hi[ax] - t
        walls.append(box(a, b))
    return merged(walls, key)


def step(t, cid, offset, stander=False, gravity=None, cls=None, label=None) -> dict:
    o = np.asarray(offset, float)
    travel = float(np.linalg.norm(o))
    return {"step": t, "component_id": cid, "component_label": label or cid, "is_stander": stander,
            "direction": (-o / travel).tolist(), "travel": travel, "start_offset": o.tolist(),
            "gravity": gravity, "stability_class": cls}


def scene(parts):
    """parts: [(cid, mesh, offset, extra kwargs)] -> (steps, meshes dict, mesh list, offsets)."""
    steps, meshes = [], {}
    for t, (cid, mesh, offset, *kw) in enumerate(parts, start=1):
        steps.append(step(t, cid, offset, **(kw[0] if kw else {})))
        meshes[cid] = mesh
    return steps, meshes, [meshes[s["component_id"]] for s in steps], np.array([s["start_offset"] for s in steps])


def row_scene():
    """Four boxes in a row, inserted from different sides."""
    return scene([
        ("A", box([0, 0, 0], [20, 20, 10]), [0, 0, -40]),
        ("B", box([0, 0, 10], [10, 10, 20]), [0, 0, 40]),
        ("C", box([10, 0, 10], [20, 10, 20]), [40, 0, 0], {"stander": True}),
        ("D", box([0, 10, 10], [20, 20, 16]), [0, 40, 0]),
    ])


def at_pixel(ctx: ViewContext, p) -> tuple[int, int]:
    col, row = ctx.to_pixel(p)
    return int(np.floor(row)), int(np.floor(col))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_l1(rec, args):
    C = IllustrationConfig
    ig.validate_config(C())
    rec.check(True, "default config validates")
    for kw, needle in [({"image_size": (0, 100)}, "image_size"), ({"search_size": (100,)}, "search_size"),
                       ({"azimuths": ()}, "must not be empty"), ({"elevations": ()}, "must not be empty"),
                       ({"elevations": (-10, 20)}, "Elevations"), ({"default_view": (0, -5)}, "Elevations"),
                       ({"vis_min_end": 1.5}, "vis_min_end"), ({"xray_alpha": -0.1}, "xray_alpha"),
                       ({"placed_color": "notacolour"}, "colour"), ({"language": "xx"}, "language"),
                       ({"perturb_step_deg": 0}, "perturb_step_deg"), ({"dpi": 0}, "dpi"),
                       ({"tess_tolerance_rel": 0}, "tess_tolerance_rel")]:
        rec.check(raises(lambda: ig.validate_config(C(**kw)), needle), f"invalid {kw} raises ({needle!r})")

    out = fresh_dir("l1")
    out.mkdir(parents=True)
    no_tool = out / "bom_no_tool.csv"
    no_tool.write_text("label,is_stander\nP1,false\nS1,true\n", encoding="utf-8")
    with_tool = out / "bom.csv"
    with_tool.write_text("label,is_stander,tool,group\nP1,false,,g1\nS1,true,wrench_10,g2\nS2,true,,g2\n",
                         encoding="utf-8")
    bom0, bom1 = ig.read_bom(no_tool), ig.read_bom(with_tool)
    s_p = step(1, "p1", [0, 0, 10], label="P1")
    s_s1 = step(2, "s1", [0, 0, 10], stander=True, label="S1")
    s_s2 = step(3, "s2", [0, 0, 10], stander=True, label="S2")
    rec.check(ig.tool_text(s_s1, bom0) == "by hand", "BOM without a tool column: stander prints 'by hand'")
    rec.check(ig.tool_text(s_s1, bom1) == "wrench_10", "stander tool read from the BOM by Label")
    rec.check(ig.tool_text(s_s2, bom1) == "by hand", "blank tool prints 'by hand'")
    rec.check(ig.tool_text(s_p, bom1) == "by hand", "non-stander prints 'by hand'")
    rec.check(ig.tool_text(step(1, "x", [0, 0, 1], label="unknown"), bom1) == "by hand", "label missing from BOM")
    steps, meshes, _, _ = row_scene()
    rec.check(raises(lambda: ig.render_illustrations(steps, {"A": meshes["A"]}, {}, out / "r", IllustrationConfig(**SMALL)),
                     "No mesh"), "missing mesh raises")
    rec.check(raises(lambda: ig.render_illustrations(steps + steps[:1], meshes, {}, out / "r", IllustrationConfig(**SMALL)),
                     "more than once"), "duplicate component raises")


def test_l2(rec, args):
    m = box([0, 0, 0], [10, 10, 10])
    r, u, v = ig.view_basis(0, 0)
    rec.check(np.allclose(r, [0, 1, 0]) and np.allclose(u, [0, 0, 1]) and np.allclose(v, [-1, 0, 0]),
              "az 0 el 0: right +y, up +z, looking along -x")
    r, u, v = ig.view_basis(90, 0)
    rec.check(np.allclose(r, [-1, 0, 0]) and np.allclose(v, [0, -1, 0]), "az 90 el 0: right -x, looking along -y")
    r, u, v = ig.view_basis(0, 90)
    rec.check(np.allclose(v, [0, 0, -1]), "el 90 looks straight down")
    rng = np.random.default_rng(1)
    ok_basis = ok_axes = ok_shift = ok_center = True
    for az, el in [(0, 0), (90, 0), (-35, 30), (200, 50), (45, 90), (315, 20)]:
        r, u, v = ig.view_basis(az, el)
        B = np.stack([r, u, v])
        ok_basis &= np.allclose(B @ B.T, np.eye(3)) and np.allclose(np.cross(r, u), -v) and abs(r[2]) < 1e-12 \
            and u[2] >= -1e-12
        ctx = ViewContext((az, el), (400, 300), [m], np.array([[0, 0, 50]]))
        p = rng.normal(size=3) * 5
        d = 3.7
        ok_axes &= np.allclose(ctx.to_pixel(p + d * r) - ctx.to_pixel(p), [d * ctx.scale, 0]) and \
            np.allclose(ctx.to_pixel(p + d * u) - ctx.to_pixel(p), [0, -d * ctx.scale]) and \
            np.allclose(ctx.to_pixel(p + d * v), ctx.to_pixel(p))
        c = ctx.center[0] * r + ctx.center[1] * u
        ok_center &= np.allclose(ctx.to_pixel(c), [ctx.W / 2, ctx.H / 2])
        for _ in range(5):
            o = rng.normal(size=3) * 20
            dx, dy, dz = ctx.shift(o)
            ok_shift &= np.allclose([dx, dy], ctx.to_pixel(p + o) - ctx.to_pixel(p)) and np.isclose(dz, v @ o)
    rec.check(ok_basis, "basis orthonormal, right-handed, r horizontal, up towards +z (6 views)")
    rec.check(ok_axes, "unit moves along r / u / v map to (+s, 0) / (0, -s) / (0, 0) pixels")
    rec.check(ok_center, "framing centre maps to the image centre")
    rec.check(ok_shift, "ghost shift equals the projection of start_offset (pixels and depth)")
    ctx = ViewContext((0, 0), (400, 300), [m], np.array([[0, 0, 50]]))
    layer = ctx.layer(0)
    rr, cc = at_pixel(ctx, [10, 5, 5])
    d = layer.depth[rr - layer.y0, cc - layer.x0]
    rec.check(np.isclose(d, -10.0, atol=1e-4), f"depth of the front face x = 10 is v . p = -10 (got {d:.5f})")


def test_l3(rec, args):
    A, B = box([0, 0, 0], [10, 10, 10]), box([5, 5, 5], [15, 15, 15])
    rec.check(A.closed and B.closed, "boxes are closed, outward-oriented meshes")
    ctx = ViewContext((0, 0), (480, 360), [A, B], np.zeros((2, 3)))
    depth, ids, _ = ig.zmerge(ctx.W, ctx.H, [(ctx.layer(0), 0, 0, 0.0, 0), (ctx.layer(1), 0, 0, 0.0, 1)])
    for p, k, label in [([0, 7.5, 7.5], 1, "overlap: B (nearer) wins"), ([0, 2.5, 2.5], 0, "A only"),
                        ([0, 12.5, 12.5], 1, "B only")]:
        rr, cc = at_pixel(ctx, p)
        rec.check(ids[rr, cc] == k, f"ID buffer: {label}")
    rr, cc = at_pixel(ctx, [0, 7.5, 7.5])
    rec.check(np.isclose(depth[rr, cc], -15, atol=1e-4), "overlap depth is B's front face (-15)")

    plate = box([20, -20, -20], [22, 30, 30])
    ctx = ViewContext((0, 0), (480, 360), [plate, A], np.array([[0, 0, 80], [0, 0, 80]]))
    ev = ig.evaluate_view(ctx)
    rec.check(ev["end"][1] == 0.0, f"box behind a plate is fully occluded (end fraction {ev['end'][1]})")
    rec.check(ev["start"][1] == 1.0, "its ghost above the plate is fully visible")

    rg = ring([0, 0, 0], 5, 10, 4)
    rec.check(rg.closed, "ring mesh is closed")
    ctx = ViewContext((0, 90), (480, 360), [rg], np.array([[0, 0, 30]]))
    layer = ctx.layer(0)
    _, own, _ = ig.zmerge(ctx.W, ctx.H, [(layer, 0, 0, 0.0, 0)])
    rec.check(own[at_pixel(ctx, [0, 0, 0])] == -1 and own[at_pixel(ctx, [7.5, 0, 0])] == 0,
              "ring seen from above: hole empty, annulus covered")
    tri = ig.mesh_from_arrays([[0, 0, 0], [10, 0, 0], [0, 10, 0]], [(0, 1, 2)], "tri")
    rec.check(not tri.closed, "a single triangle is not closed")
    up = ViewContext((0, 90), (200, 150), [tri], np.array([[0, 0, 5]])).layer(0).mask.sum()
    down_tri = ig.mesh_from_arrays([[0, 0, 0], [0, 10, 0], [10, 0, 0]], [(0, 1, 2)], "tri2")
    down = ViewContext((0, 90), (200, 150), [down_tri], np.array([[0, 0, 5]])).layer(0).mask.sum()
    rec.check(up > 0 and down == up, "open meshes are not back-face culled (both windings drawn)")


def test_l4(rec, args):
    parts = [box([0, 0, 0], [10, 10, 10]), box([5, 5, 5], [15, 15, 15]), ring([4, 4, 12], 3, 6, 3),
             box([-5, 8, -3], [25, 9, 2])]
    combined = merged(parts)
    worst_mask, worst_depth, worst_id = 0, 0.0, 0
    for view in [(-35, 30), (120, 50), (250, 20)]:
        ctx = ViewContext(view, (480, 360), parts, np.zeros((len(parts), 3)))
        ctx2 = ViewContext(view, (480, 360), [combined], np.zeros((1, 3)))
        depth, ids, _ = ig.zmerge(ctx.W, ctx.H, [(ctx.layer(k), 0, 0, 0.0, k) for k in range(len(parts))])
        d2, id2, _ = ig.zmerge(ctx2.W, ctx2.H, [(ctx2.layer(0), 0, 0, 0.0, 0)])
        both = (ids >= 0) & (id2 >= 0)
        worst_mask = max(worst_mask, int(((ids >= 0) != (id2 >= 0)).sum()))
        worst_depth = max(worst_depth, float(np.abs(depth[both] - d2[both]).max()))
        worst_id = max(worst_id, int((np.abs(depth - d2) > 1e-3)[both].sum()))
    rec.check(worst_mask == 0, f"coverage of the merged layers equals direct rasterization ({worst_mask} px differ)")
    rec.check(worst_depth < 1e-3, f"depth of the merge equals direct rasterization (max diff {worst_depth:.2e})")


def l5_case(part, offset):
    plate = box([10, -20, -20], [12, 20, 20])
    ctx = ViewContext((0, 0), (480, 360), [plate, part], np.array([[0, 0, 80], offset], float))
    ev = ig.evaluate_view(ctx)
    return ev, ig.visible_mask(ev, IllustrationConfig())[1]


def test_l5(rec, args):
    hidden, side = box([0, -3, -3], [4, 3, 3]), box([0, 30, -3], [4, 36, 3])
    ev, vis = l5_case(hidden, [0, 0, 60])
    rec.check(ev["end"][1] < 0.5 and ev["start"][1] >= 0.5 and not vis,
              f"end occluded fails (end {ev['end'][1]:.2f}, start {ev['start'][1]:.2f})")
    ev, vis = l5_case(side, [0, -33, 0])
    rec.check(ev["end"][1] >= 0.5 and ev["start"][1] < 0.5 and not vis,
              f"start occluded fails (end {ev['end'][1]:.2f}, start {ev['start'][1]:.2f})")
    ev, vis = l5_case(side, [60, 0, 0])
    rec.check(ev["end"][1] >= 0.5 and ev["start"][1] >= 0.5 and ev["traj"][1] < 0.01 and not vis,
              f"view along the insertion direction fails (trajectory {ev['traj'][1]:.3f})")
    ev, vis = l5_case(side, [0, 0, 60])
    rec.check(bool(vis) and ev["in_frame"][1], f"clear case passes (end {ev['end'][1]:.2f}, start "
                                                f"{ev['start'][1]:.2f}, trajectory {ev['traj'][1]:.2f})")


def test_l6(rec, args):
    cfg = IllustrationConfig(azimuths=(0, 90, 180, 270), elevations=(20, 50), search_size=(240, 180))
    _, _, mesh_list, offsets = scene([
        ("wall", box([-1, -30, -5], [1, 30, 30]), [0, 0, 60]),
        ("p", box([2, -4, 0], [10, 4, 8]), [0, 0, 50]),
        ("q", box([-10, -4, 0], [-2, 4, 8]), [0, 0, 50]),
        ("r", box([-4, 20, 0], [4, 28, 8]), [0, 40, 0]),
    ])
    glob, _ = ig.select_views(mesh_list, offsets, cfg)
    cands = ig.candidate_views(cfg)
    evs = {}
    for c in cands:
        ev = ig.evaluate_view(ViewContext(c, cfg.search_size, mesh_list, offsets))
        evs[c] = (int(ig.visible_mask(ev, cfg).sum()), float(np.nansum(ev["end"])))
    best_count = max(n for n, _ in evs.values())
    rec.check(evs[glob][0] == best_count, f"global view {glob} shows the maximum number of steps ({best_count})")
    tied = [c for c in cands if evs[c][0] == best_count]
    rec.check(all(evs[glob][1] >= evs[c][1] - 1e-9 for c in tied), "ties broken by the sum of end-visible fractions")
    rec.note(f"visible counts: {sorted(set(n for n, _ in evs.values()))}, {len(tied)} view(s) tied on count")

    _, _, mesh_list, offsets = scene([("a", box([0, 0, 0], [10, 10, 10]), [0, 0, 40]),
                                      ("b", box([30, 30, 0], [40, 40, 10]), [0, 0, 40])])
    glob, _ = ig.select_views(mesh_list, offsets, cfg)
    rec.check(glob == tuple(map(float, cfg.default_view)),
              f"full tie on count and end fractions: default view wins by angular distance (got {glob})")
    cfg2 = replace(cfg, default_view=(80.0, 45.0))
    glob, _ = ig.select_views(mesh_list, offsets, cfg2)
    rec.check(glob == (80.0, 45.0), f"the default view is always the first candidate (got {glob})")


def test_l7(rec, args):
    wall = box([-1, -30, -5], [1, 30, 40])
    _, _, mesh_list, offsets = scene([("wall", wall, [0, 0, 45]), ("p", box([2, -4, 0], [10, 4, 8]), [0, 0, 50]),
                                      ("q", box([-10, -4, 0], [-2, 4, 8]), [0, 0, 50])])
    cfg = IllustrationConfig(azimuths=(0, 180), elevations=(20,), default_view=(0, 20), search_size=(480, 360))
    glob, ch = ig.select_views(mesh_list, offsets, cfg)
    rec.check(glob == (0.0, 20.0) and ch[1]["kind"] == "global", f"global view (0, 20) shows step 2 ({ch[1]['kind']})")
    rec.check(ch[2]["kind"] == "candidate" and ch[2]["view"] == (180.0, 20.0) and ch[2]["visible"],
              f"step 3 hidden behind the wall gets the other candidate ({ch[2]['kind']} {ch[2]['view']})")

    toward = np.array([np.cos(np.radians(20)), 0, np.sin(np.radians(20))])
    _, _, mesh_list, offsets = scene([("a", box([0, 0, 0], [10, 10, 10]), [0, 0, 40]),
                                      ("b", box([0, 20, 0], [10, 30, 10]), (60 * toward).tolist())])
    cfg = IllustrationConfig(azimuths=(0,), elevations=(20,), default_view=(0, 20), search_size=(480, 360),
                             min_trajectory_px_frac=0.05)
    glob, ch = ig.select_views(mesh_list, offsets, cfg)
    rec.check(ch[1]["kind"] == "perturbed" and ch[1]["visible"] and ch[1]["view"] != glob,
              f"step along the only candidate's view direction gets a perturbed view ({ch[1]['kind']} {ch[1]['view']})")
    d = ig.view_angle(ch[1]["view"], glob)
    rec.check(d <= cfg.perturb_step_deg + 1e-6, f"the nearest perturbation that passes is used ({d:.1f} deg)")

    _, _, mesh_list, offsets = scene([("box", enclosure([-10, -10, -10], [10, 10, 10], 2), [0, 0, 60]),
                                      ("inner", box([-4, -4, -4], [4, 4, 4]), [0, 0, 40])])
    cfg = IllustrationConfig(search_size=(160, 120))
    glob, ch = ig.select_views(mesh_list, offsets, cfg)
    rec.check(ch[1]["kind"] == "xray" and ch[1]["xray"] and ch[1]["warnings"] and ch[1]["visible"],
              f"enclosed part gets the x-ray fallback with a warning ({ch[1]['kind']}, {ch[1]['warnings']})")
    rec.check(ch[1]["vis_end_occluded"] == 0.0, "its occluded end fraction is 0 in the chosen view")


def test_l8(rec, args):
    steps, meshes, mesh_list, offsets = row_scene()
    cfg = IllustrationConfig(**SMALL)
    res = ig.render_illustrations(steps, meshes, {}, fresh_dir("l8"), cfg, title="row")
    rec.check([f["component_id"] for f in res.frames] == [s["component_id"] for s in steps]
              and [f["step"] for f in res.frames] == [1, 2, 3, 4], "frames keep the sequence order")
    rec.check(all(Path(f["png"]).exists() for f in res.frames) and Path(res.manual_path).exists(),
              "one PNG per step and the manual are written")
    ctx = ViewContext(res.global_view, cfg.image_size, mesh_list, offsets)
    N = len(steps)
    ok_future = ok_placed = ok_ghost = True
    for s in range(N):
        cov = [ig.zmerge(ctx.W, ctx.H, [(ctx.layer(k), 0, 0, 0.0, 0)])[1] >= 0 for k in range(N)]
        dx, dy, _ = ctx.shift(offsets[s])
        ghost = ig.zmerge(ctx.W, ctx.H, [(ctx.layer(s), int(round(dx)), int(round(dy)), 0.0, 0)])[1] >= 0
        shown = np.any(cov[: s + 1], axis=0) | ghost
        future = np.any(cov[s + 1:], axis=0) if s + 1 < N else np.zeros_like(ghost)
        rgb = ig.compose_frame(ctx, s, cfg, steps[s]["is_stander"])
        white = (rgb > 0.999).all(axis=2)
        near = shown | np.roll(shown, 1, 0) | np.roll(shown, -1, 0) | np.roll(shown, 1, 1) | np.roll(shown, -1, 1)
        ok_future &= bool(white[future & ~near].all())  # outlines may spill one pixel past a shown part
        if s > 0:
            placed_only = np.any(cov[:s], axis=0) & ~cov[s] & ~ghost
            ok_placed &= bool((~white[placed_only]).mean() > 0.99)
        a, b = ig.trajectory_points(ctx, s)
        c = mesh_list[s].centroid
        ok_ghost &= np.allclose(a, ctx.to_pixel(c + offsets[s])) and np.allclose(b, ctx.to_pixel(c))
    rec.check(ok_future, "step t never shows components of later steps")
    rec.check(ok_placed, "step t shows the parts of steps 1..t-1")
    rec.check(ok_ghost, "ghost at final + start_offset; arrow from ghost centroid to part centroid")


def test_l9(rec, args):
    steps, meshes, mesh_list, offsets = row_scene()
    cfg = IllustrationConfig(**SMALL)
    res = ig.render_illustrations(steps, meshes, {}, fresh_dir("l9"), cfg, title="row")
    glob_frames = [f for f in res.frames if f["view"]["is_global"]]
    rec.check(len(glob_frames) >= 2, f"{len(glob_frames)} frames share the global view")
    ctx = ViewContext(res.global_view, cfg.image_size, mesh_list, offsets)
    ctx2 = ViewContext(res.global_view, cfg.image_size, list(reversed(mesh_list)), offsets[::-1])
    rec.check(ctx.scale == ctx2.scale and np.allclose(ctx.center, ctx2.center),
              "framing is a union over all steps (independent of the order)")
    N, ok = len(steps), True
    for s in range(1, N - 1):
        frames = []
        for t in (s, s + 1):
            items = [(ctx.layer(k), 0, 0, 0.0, k) for k in range(t + 1)]
            dx, dy, _ = ctx.shift(offsets[t])
            ghost = ig.zmerge(ctx.W, ctx.H, [(ctx.layer(t), int(round(dx)), int(round(dy)), 0.0, 0)])[1] >= 0
            ids = ig.zmerge(ctx.W, ctx.H, items)[1]
            frames.append((ig.compose_frame(ctx, t, cfg, steps[t]["is_stander"]), ids, ghost))
        (r1, i1, g1), (r2, i2, g2) = frames
        region = (i1 == 0) & (i2 == 0) & ~g1 & ~g2
        # Interior pixels only: outlines next to a newly added part may change.
        inner = region & np.roll(region, 1, 0) & np.roll(region, -1, 0) & np.roll(region, 1, 1) & np.roll(region, -1, 1)
        ok &= inner.sum() > 0 and np.allclose(r1[inner], r2[inner])
    rec.check(ok, "a placed part has the same pixels in consecutive frames (same scale and offset)")


def test_l10(rec, args):
    bom = {"S": {"label": "S", "is_stander": "true", "tool": "wrench_10", "group": "g2"},
           "P": {"label": "P", "is_stander": "false", "tool": "", "group": "g1"}}  # distinct: one frame per step
    steps = [step(1, "p", [0, 0, -20], gravity=[1, 0, 0], cls=2, label="P"),
             step(2, "s", [0, 0, 30], stander=True, gravity=[1, 0, 0], cls=1, label="S"),
             step(3, "q", [10, 0, 0], gravity=[0, 1, 0], cls=2, label="Q")]
    cfg = IllustrationConfig()
    l0, f0 = ig.footnote_lines(steps, 0, bom, cfg)
    l1, f1 = ig.footnote_lines(steps, 1, bom, cfg)
    l2, f2 = ig.footnote_lines(steps, 2, bom, cfg)
    rec.check(l0 == ["Step 1 / 3   P   Tool: by hand"] and not f0["reorient"] and not f0["hold"],
              f"step 1: label and 'by hand' ({l0})")
    rec.check(l1[0] == "Step 2 / 3   S   Tool: wrench_10" and f1["hold"] and l1[-1] == "Hold in place"
              and not f1["reorient"], f"stander: BOM tool, class 1 gets 'Hold in place' ({l1})")
    rec.check(f2["reorient"] and any(x.startswith("Re-orient assembly") and "(0, 1, 0)" in x for x in l2)
              and not f2["hold"], f"gravity change gets the re-orient line ({l2})")
    l1b, f1b = ig.footnote_lines(steps, 1, bom, replace(cfg, mark_unstable=False))
    rec.check(not f1b["hold"] and len(l1b) == 1, "mark_unstable=False drops the hold mark")
    l1c, _ = ig.footnote_lines(steps, 1, bom, replace(cfg, footnote_show_vector=True))
    rec.check(l1c[0].endswith("Move: (0, 0, -30) mm"), f"footnote_show_vector appends the displacement ({l1c[0]})")
    no_g = [dict(s, gravity=None, stability_class=None) for s in steps]
    rec.check(all(ig.footnote_lines(no_g, s, bom, cfg)[1] == {"reorient": False, "hold": False} for s in range(3)),
              "no gravity data: no re-orient or hold marks")
    meshes = {"p": box([0, 0, 0], [10, 10, 5]), "s": box([3, 3, 5], [6, 6, 15]), "q": box([10, 0, 0], [15, 10, 5])}
    res = ig.render_illustrations(steps, meshes, bom, fresh_dir("l10"), IllustrationConfig(**SMALL), title="t")
    rec.check([f["footnote"] for f in res.frames] == [l0, l1, l2] and res.frames[1]["hold"] and res.frames[2]["reorient"],
              "frames report the same footnotes and flags")


def _png_bytes(res):
    return [Path(f["png"]).read_bytes() for f in res.frames]


def test_l12(rec, args):
    steps, meshes, _, _ = row_scene()
    cfg = IllustrationConfig(**SMALL)
    a, b = fresh_dir("l12a"), fresh_dir("l12b")
    r1 = ig.render_illustrations(steps, meshes, {}, a, cfg, title="row")
    stamps = [Path(f["png"]).stat().st_mtime_ns for f in r1.frames]
    r2 = ig.render_illustrations(steps, meshes, {}, a, cfg, title="row")
    rec.check(r2.config_hash == r1.config_hash and [Path(f["png"]).stat().st_mtime_ns for f in r2.frames] == stamps,
              "second run loads the cached frames")
    r3 = ig.render_illustrations(steps, meshes, {}, b, cfg, title="row", force=True)
    rec.check(_png_bytes(r3) == _png_bytes(r1), "identical PNG bytes for the same config (fresh directory)")
    rec.check(Path(r3.manual_path).read_bytes() == Path(r1.manual_path).read_bytes(), "identical manual PDF")

    before = _png_bytes(r1)
    r4 = ig.render_illustrations(steps, meshes, {}, a, replace(cfg, new_color="#00aa00"), title="row")
    rec.check(r4.config_hash != r1.config_hash and _png_bytes(r4) != before, "colour change re-renders frames")
    rec.check(r4.stats.get("layers_rasterized", 0) == 0 and r4.stats.get("views_cached"),
              f"colour change re-rasterizes no layer and reuses the view search ({r4.stats.get('layers_rasterized', 0)})")

    # A reorder may legitimately change the global view; one candidate pins it to test the layer cache.
    fixed = replace(cfg, azimuths=(0,), elevations=(35,), default_view=(0, 35))
    c = fresh_dir("l12c")
    r5 = ig.render_illustrations(steps, meshes, {}, c, fixed, title="row")
    n_layers = len(list((c / "layers").glob("*.npz")))
    order = [0, 2, 1, 3]
    reordered = [dict(steps[i], step=t + 1) for t, i in enumerate(order)]
    r6 = ig.render_illustrations(reordered, meshes, {}, c, fixed, title="row")
    rec.check(r6.config_hash != r5.config_hash and [f["component_id"] for f in r6.frames] == ["A", "C", "B", "D"],
              "sequence change invalidates the frames")
    rec.check(r6.global_view == r5.global_view and r6.stats.get("layers_loaded", 0) == n_layers,
              f"reordering reuses all {n_layers} final-resolution layers of the global view (framing is order "
              f"independent); views: {[f['view_kind'] for f in r6.frames]}")
    rec.check(len(list(c.glob("step_*.png"))) == len(steps), "stale step PNGs from the previous order are removed")


def _scale_case(rec, N, budget_s, image_cfg):
    rng = np.random.default_rng(7)
    parts, axes = [], [[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
    cols = 5
    for i in range(N):
        cx, cy = (i % cols) * 30.0, (i // cols) * 30.0
        offset = (np.array(axes[i % len(axes)]) * rng.uniform(30, 60)).tolist()
        mesh = sphere([cx, cy, 0], 10, n_lat=60, n_lon=120, key=f"scale-sphere-{i}")
        parts.append((f"c{i}", mesh, offset, {"stander": i % 4 == 3, "gravity": [0, 0, -1], "cls": 2}))
    steps, meshes, _, _ = scene(parts)
    rec.note(f"N = {N}: {sum(len(m.triangles) for m in meshes.values())} triangles")
    tracemalloc.start()
    t0 = time.perf_counter()
    res = ig.render_illustrations(steps, meshes, {}, fresh_dir(f"scale{N}"), image_cfg, title="scale")
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rec.check(len(res.frames) == N, f"N = {N}: {N} frames")
    rec.check(wall <= budget_s, f"N = {N} within {budget_s:.0f} s ({wall:.1f} s: search "
                                f"{res.stats['view_search_s']:.1f} s, render {res.stats['render_s']:.1f} s)")
    rec.check(peak < 2 * 1024 ** 3, f"peak traced memory {peak / 1024 ** 2:.0f} MB < 2 GB")
    rec.note(f"views evaluated {res.stats.get('views_evaluated')}, layers rasterized "
             f"{res.stats.get('layers_rasterized')}, kinds {[f['view_kind'] for f in res.frames]}")


def test_l13(rec, args):
    _scale_case(rec, 8, 60.0, IllustrationConfig())
    if "scale" in args.groups:
        _scale_case(rec, 20, 300.0, IllustrationConfig())
    else:
        rec.note("N = 20 skipped (run with --groups scale)")


def test_l11(rec, args):
    if "real" not in args.groups:
        rec.note("skipped (run with --groups real)")
        return
    import freecad  # noqa: F401
    from preprocessing import run_preprocessing

    out = fresh_dir("fastener")
    seq_file = SEQUENCE_FILE if SEQUENCE_FILE.exists() else SEQUENCE_FALLBACK
    if not seq_file.exists():
        rec.check(False, f"no cached industrial_fastner sequence ({SEQUENCE_FILE} or {SEQUENCE_FALLBACK}); "
                         "run test_optimization.py --tests O13 --groups real first")
        return
    rec.note(f"sequence from {seq_file.relative_to(REPO_ROOT)}")
    steps = json.loads(seq_file.read_text(encoding="utf-8"))["steps"]
    prep = run_preprocessing(FASTENER_STEP, FASTENER_BOM)
    t0 = time.perf_counter()
    res = ig.generate_illustrations(steps, prep, out)
    wall = time.perf_counter() - t0
    pngs = sorted(out.glob("step_*.png"))
    blocks = ig.group_blocks(steps, ig.read_bom(FASTENER_BOM), IllustrationConfig())
    rec.check(len(steps) == 8 and len(pngs) == len(res.frames) == len(blocks),
              f"{len(pngs)} PNGs = {len(blocks)} frames for N = {len(steps)} (group blocks {blocks})")
    rec.check(sorted(n for f in res.frames for n in f["steps"]) == list(range(1, 9)), "every step is in one frame")
    rec.check(Path(res.manual_path).exists() and Path(res.manual_path).stat().st_size > 0, "manual.pdf written")
    rec.check(all(f["visible"] or f["warnings"] for f in res.frames), "every frame visible or carries a warning")
    rec.check(all(not f["xray"] or f["warnings"] for f in res.frames), "every x-ray frame carries a warning")
    rec.check(wall < 120, f"end to end in {wall:.1f} s (target about 1 min)")
    rec.note(f"global view {res.global_view}; " + "; ".join(
        f"{f['step']}:{f['view_kind']}" + ("(x-ray)" if f["xray"] else "") for f in res.frames))

    mesh_stamps = {p: p.stat().st_mtime_ns for p in (out / "meshes").glob("*.npz")}
    res2 = ig.generate_illustrations(steps, prep, out, IllustrationConfig(placed_color="#a0a0c0"))
    rec.check({p: p.stat().st_mtime_ns for p in (out / "meshes").glob("*.npz")} == mesh_stamps,
              "colour change does not re-tessellate")
    rec.check(res2.stats.get("layers_rasterized", 0) == 0, "colour change does not re-rasterize layers")


def test_l14(rec, args):
    """Group frames (spec 12): consecutive steps of one BOM group share a frame; only the first movement is judged."""
    groups = ["g1", "g1", "", "", "g2", "g1", "g1"]
    steps = [step(t + 1, f"c{t}", [0, 0, 10], label=f"L{t}") for t in range(len(groups))]
    bom = {f"L{t}": {"label": f"L{t}", "group": g} for t, g in enumerate(groups)}
    cfg = IllustrationConfig(**SMALL)
    blocks = ig.group_blocks(steps, bom, cfg)
    rec.check(blocks == [(0, 1), (2, 2), (3, 3), (4, 4), (5, 6)],
              f"runs of one non-blank group merge, blanks and interrupted groups do not ({blocks})")
    rec.check(ig.group_blocks(steps, bom, replace(cfg, group_frames=False)) == [(t, t) for t in range(7)],
              "group_frames=False gives one frame per step")

    # Frame content: B and C (group 'pins') share a frame, each with its ghost; D comes later.
    steps, meshes, mesh_list, offsets = scene([
        ("A", box([0, 0, 0], [40, 20, 6]), [0, 0, -40]),
        ("B", box([4, 6, 6], [10, 14, 20]), [0, 0, 40]),
        ("C", box([28, 6, 6], [34, 14, 20]), [0, 0, 40], {"stander": True}),
        ("D", box([14, 4, 6], [24, 16, 10]), [0, 30, 0]),
    ])
    bom = {"A": {"group": "base"}, "B": {"group": "pins"}, "C": {"group": "pins", "tool": "wrench_10"},
           "D": {"group": "cover"}}
    out = fresh_dir("l14")
    res = ig.render_illustrations(steps, meshes, bom, out, cfg, title="groups")
    f1 = res.frames[1]
    rec.check(len(res.frames) == 3 and f1["steps"] == [2, 3] and f1["component_ids"] == ["B", "C"]
              and f1["group"] == "pins", f"3 frames, the second covers steps 2-3 ({[f['steps'] for f in res.frames]})")
    rec.check(Path(f1["png"]).name == "step_02-03_pins.png" and len(list(out.glob("step_*.png"))) == 3,
              f"one PNG per frame, named by step range and group ({Path(f1['png']).name})")
    rec.check(f1["footnote"][:2] == ["Steps 2-3 / 4   pins", "Parts: B (by hand), C (wrench_10)"],
              f"block footnote with per-part tools ({f1['footnote'][:2]})")

    ctx = ViewContext((f1["view"]["azimuth"], f1["view"]["elevation"]), cfg.image_size, mesh_list, offsets)
    rgb = ig.compose_frame(ctx, 1, cfg, [False, True], last=2)

    def hue_at(p, color):  # a part's pixels are color * shade: compare normalized RGB around p
        r, c = at_pixel(ctx, p)
        v = rgb[r - 2:r + 3, c - 2:c + 3].reshape(-1, 3).mean(0)
        return bool(np.allclose(v / v.sum(), color / color.sum(), atol=0.03))

    def tint_at(p, color):  # ghosts are blended over white: tinted, red/blue balance leaning like `color`
        r, c = at_pixel(ctx, p)
        v = rgb[r - 2:r + 3, c - 2:c + 3].reshape(-1, 3).mean(0)
        return bool(v.min() < 0.95 and np.sign(v[0] - v[2]) == np.sign(color[0] - color[2]))

    new, stander = ig._rgb(cfg.new_color), ig._rgb(cfg.stander_color)
    rec.check(hue_at(mesh_list[1].centroid, new) and hue_at(mesh_list[2].centroid, stander),
              "both members drawn at their final positions in their own colours")
    rec.check(tint_at(mesh_list[1].centroid + offsets[1], new) and tint_at(mesh_list[2].centroid + offsets[2], stander),
              "both members get a ghost at final + start_offset")
    rec.check(not hue_at(mesh_list[3].centroid, new), "the later step D is not drawn in the group frame")
    arrows = [ig.trajectory_points(ctx, k) for k in (1, 2)]
    rec.check(all(np.linalg.norm(b - a) > 10 for a, b in arrows), "one trajectory per member")

    # Only the first movement must be visible: q (hidden behind the wall from az 0) rides along with p.
    wall = box([-1, -30, -5], [1, 30, 40])
    vcfg = IllustrationConfig(azimuths=(0, 180), elevations=(20,), default_view=(0, 20), search_size=(480, 360))
    _, _, mesh_list, offsets = scene([("wall", wall, [0, 0, 45]), ("p", box([2, -4, 0], [10, 4, 8]), [0, 0, 50]),
                                      ("q", box([-10, -4, 0], [-2, 4, 8]), [0, 0, 50])])
    _, ch = ig.select_views(mesh_list, offsets, vcfg, blocks=[(0, 0), (1, 2)])
    rec.check(len(ch) == 2 and ch[1]["kind"] == "global" and ch[1]["visible"],
              f"the group frame passes on its first movement alone ({ch[1]['kind']})")
    _, ch1 = ig.select_views(mesh_list, offsets, vcfg)
    rec.check(ch1[2]["kind"] == "candidate", "without grouping the hidden second step needs its own view")

    # The other members of a block occlude its first movement (they are drawn in the same frame).
    lead, sibling = box([0, -3, -3], [4, 3, 3]), box([10, -20, -20], [12, 20, 20])
    ctx = ViewContext((0, 0), (480, 360), [lead, sibling], np.array([[0, 0, 60], [0, 0, 80]]))
    alone, grouped = ig.evaluate_view(ctx), ig.evaluate_view(ctx, blocks=[(0, 1)])
    rec.check(alone["end"][0] == 1.0 and grouped["end"][0] == 0.0,
              f"a sibling in front hides the first movement (end {alone['end'][0]:.2f} -> {grouped['end'][0]:.2f})")

    # Footnote of a block: shared tool, move vectors, re-orient inside the block, hold list.
    st = [step(1, "a", [0, 0, -5], gravity=[1, 0, 0], label="A"),
          step(2, "s1", [0, 10, 0], stander=True, gravity=[1, 0, 0], cls=1, label="S1"),
          step(3, "s2", [0, 10, 0], stander=True, gravity=[0, 1, 0], cls=1, label="S2")]
    b2 = {"S1": {"tool": "wrench_10", "group": "f"}, "S2": {"tool": "wrench_10", "group": "f"}}
    lines, flags = ig.block_footnote_lines(st, 1, 2, b2, replace(cfg, footnote_show_vector=True))
    rec.check(lines == ["Steps 2-3 / 3   f   Tool: wrench_10", "Parts: S1, S2",
                        "Move: S1 (0, -10, 0) mm; S2 (0, -10, 0) mm",
                        "Re-orient assembly: gravity along (0, 1, 0)", "Hold in place: S1, S2"]
              and flags == {"reorient": True, "hold": True}, f"block footnote lines and flags ({lines})")
    rec.check(ig.block_footnote_lines(st, 1, 1, b2, cfg) == ig.footnote_lines(st, 1, b2, cfg),
              "a single-step block keeps the per-step footnote")


TESTS = [
    ("L1", test_l1), ("L2", test_l2), ("L3", test_l3), ("L4", test_l4), ("L5", test_l5), ("L6", test_l6),
    ("L7", test_l7), ("L8", test_l8), ("L9", test_l9), ("L10", test_l10), ("L11", test_l11), ("L12", test_l12),
    ("L13", test_l13), ("L14", test_l14),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="*", help="individual test ids, e.g. L1 L5")
    ap.add_argument("--groups", nargs="*", default=["core"], choices=["core", "real", "scale"],
                    help="core (default), real (industrial_fastner end to end), scale (L13 at N = 20)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="    %(name)s %(levelname)s: %(message)s")
    logging.getLogger("aap.illustration").setLevel(logging.ERROR)  # x-ray warnings are expected in L7

    rec = Recorder()
    cache_before = cache_snapshot()
    for test_id, fn in TESTS:
        if args.tests and test_id not in args.tests:
            continue
        print(f"\n=== {test_id} ===", flush=True)
        rec.start(test_id)
        t0 = time.perf_counter()
        try:
            fn(rec, args)
        except Exception as exc:  # noqa: BLE001
            rec.error(exc)
        rec.results[test_id]["seconds"] = round(time.perf_counter() - t0, 3)

    print("\n=== cache ===", flush=True)
    rec.start("cache")
    after = cache_snapshot()
    for name, digest in cache_before.items():
        rec.check(after.get(name) == digest, f"tests/industrial_fastner/{name} byte-identical after the run")
    rec.results["cache"]["seconds"] = 0.0

    print("\n| Test | Result | Time | Notes |\n|---|---|---|---|")
    for test_id, r in rec.results.items():
        status = "PASS" if r["passed"] else ("ERROR" if "error" in r else "FAIL")
        failed = [c["label"] for c in r["checks"] if not c["passed"]]
        notes = "; ".join(failed[:2] + r["notes"][:2] + ([r["error"]] if "error" in r else []))
        print(f"| {test_id} | {status} | {r['seconds']:.1f} s | {notes} |")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "illustration_results.json"
    out.write_text(json.dumps({"groups": args.groups, "results": rec.results}, indent=2))
    print(f"\nDetails written to {out}")
    return 0 if all(r["passed"] for r in rec.results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
