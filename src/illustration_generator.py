"""Illustration Generator: one action diagram per assembly step plus a PDF manual.

Implements docs/illustration-spec.md. Every component is tessellated once (FreeCAD), rasterized per view
into a layer (depth + shade, numpy z-buffer, orthographic), and frames are depth-min merges of layers: the
placed parts (steps 1..t-1), the new part at its final position and its ghost at the start position (the
same layer shifted on screen, exact under an orthographic projection). The view is one global camera for
the whole manual, replaced for a step only when that step fails the visibility rule (spec 6).

Layout:
  IllustrationConfig, validate_config      configuration (spec 3)
  Mesh, mesh_from_arrays                   triangle meshes (spec 5.1)
  view_basis, ViewContext, rasterize       camera, framing, layers (spec 4, 5.2)
  evaluate_view, select_views              visibility rule and view cascade (spec 6)
  compose_frame, footnote_lines            frame content (spec 5.3, 7)
  render_illustrations                     pure numpy/matplotlib entry, with caching (spec 8)
  generate_illustrations                   pipeline entry (FreeCAD shapes -> meshes)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("aap.illustration")

FRAME_MARGIN = 0.06  # framing box margin, fraction of its larger side
FOOTER_FRAC = 0.1  # bottom band of each frame reserved for the footnote
AMBIENT = 0.35  # Lambert shading: ambient share, the rest is one headlight
DEPTH_EPS_REL = 1e-4  # depth tolerance (fraction of the scene diagonal) for touching surfaces
EDGE_JUMP_REL = 0.02  # depth jump (fraction of the scene diagonal) drawn as a line inside one part
BARY_TOL = 1e-7  # barycentric tolerance, closes cracks between neighbouring triangles
CHUNK = 1_000_000  # candidate pixels per rasterizer batch (memory bound)
OUTLINE_RGB = np.array([0.12, 0.12, 0.12])
TRAJECTORY_COLOR = "#1a1a1a"
MESH_VERSION = "1"
LAYER_VERSION = "1"
FRAMES_VERSION = "1"

REPORT_NAME = "illustration_report.json"
VIEWS_NAME = "view_selection.json"
# Config fields the view search depends on (the view cache ignores colours, footnote and page options).
SEARCH_FIELDS = ("search_size", "default_view", "azimuths", "elevations", "vis_min_end", "vis_min_start",
                 "min_trajectory_px_frac", "perturb_step_deg", "perturb_max_deg")
MANUAL_NAME = "manual.pdf"

STRINGS = {
    "en": {
        "step": "Step",
        "tool": "Tool",
        "by_hand": "by hand",
        "move": "Move",
        "reorient": "Re-orient assembly: gravity along {g}",
        "hold": "Hold in place",
        "steps": "{n} steps",
        "final": "Final assembly",
        "columns": ("Part", "Stander", "Tool", "Group"),
        "yes": "yes",
        "no": "no",
    },
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IllustrationConfig:
    image_size: tuple[int, int] = (1600, 1200)  # px, final frame (width, height)
    search_size: tuple[int, int] = (480, 360)  # px, view search
    dpi: int = 200
    tess_tolerance_rel: float = 0.002  # tessellation deflection / assembly bbox diagonal
    default_view: tuple[float, float] = (-35.0, 30.0)  # (azimuth, elevation) deg, start of the search order
    azimuths: tuple = tuple(range(0, 360, 15))
    elevations: tuple = (20, 35, 50)
    vis_min_end: float = 0.5
    vis_min_start: float = 0.5
    min_trajectory_px_frac: float = 0.12  # of the shorter image side
    perturb_step_deg: float = 5.0
    perturb_max_deg: float = 30.0
    xray_alpha: float = 0.25
    placed_color: str = "#bdbdbd"
    new_color: str = "#f28e2b"
    stander_color: str = "#3b75af"
    ghost_alpha: float = 0.35
    show_gravity: bool = True
    mark_unstable: bool = True
    footnote_show_vector: bool = False
    title: str | None = None  # None = assembly name
    language: str = "en"
    force: bool = False


def validate_config(config: IllustrationConfig) -> None:
    """Spec 3 validation; raises ValueError."""
    from matplotlib.colors import is_color_like

    for name in ("image_size", "search_size"):
        size = getattr(config, name)
        if len(size) != 2 or any(int(x) != x or x < 16 for x in size):
            raise ValueError(f"{name} must be two integers >= 16 (width, height), got {size!r}")
    if config.dpi <= 0:
        raise ValueError("dpi must be > 0")
    if config.tess_tolerance_rel <= 0:
        raise ValueError("tess_tolerance_rel must be > 0")
    if len(config.default_view) != 2:
        raise ValueError("default_view must be (azimuth, elevation)")
    if len(config.azimuths) == 0 or len(config.elevations) == 0:
        raise ValueError("azimuths and elevations must not be empty")
    for el in (*config.elevations, config.default_view[1]):
        if not 0 <= el <= 90:
            raise ValueError(f"Elevations must lie in [0, 90] deg (the base is never viewed from below), got {el}")
    for name in ("vis_min_end", "vis_min_start", "xray_alpha", "ghost_alpha"):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    if config.min_trajectory_px_frac < 0:
        raise ValueError("min_trajectory_px_frac must be >= 0")
    if config.perturb_step_deg <= 0 or config.perturb_max_deg < 0:
        raise ValueError("perturb_step_deg must be > 0 and perturb_max_deg >= 0")
    for name in ("placed_color", "new_color", "stander_color"):
        if not is_color_like(getattr(config, name)):
            raise ValueError(f"{name} is not a colour: {getattr(config, name)!r}")
    if config.language not in STRINGS:
        raise ValueError(f"language must be one of {sorted(STRINGS)}, got {config.language!r}")


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------


@dataclass
class Mesh:
    vertices: np.ndarray  # (V, 3) mm, global coordinates
    triangles: np.ndarray  # (T, 3) vertex indices, outward winding when closed
    normals: np.ndarray  # (T, 3) unit triangle normals
    centroid: np.ndarray  # (3,) area-weighted
    closed: bool  # closed and consistently oriented: back faces may be culled
    key: str = ""  # cache key (geometry hash + tessellation tolerance)


def _is_closed(V: np.ndarray, F: np.ndarray) -> bool:
    """True if every directed edge occurs once and its reverse exists (closed, consistently oriented)."""
    if not len(F):
        return False
    scale = float(np.ptp(V, axis=0).max()) or 1.0
    _, weld = np.unique(np.round(V / (1e-7 * scale)), axis=0, return_inverse=True)
    W = weld.reshape(-1)[F]
    W = W[(W[:, 0] != W[:, 1]) & (W[:, 1] != W[:, 2]) & (W[:, 2] != W[:, 0])]
    n = int(W.max()) + 1 if len(W) else 1
    a = np.concatenate([W[:, 0], W[:, 1], W[:, 2]]).astype(np.int64)
    b = np.concatenate([W[:, 1], W[:, 2], W[:, 0]]).astype(np.int64)
    fwd, rev = a * n + b, b * n + a
    return len(np.unique(fwd)) == len(fwd) and bool(np.isin(rev, fwd).all())


def mesh_from_arrays(vertices, triangles, key: str = "") -> Mesh:
    """Mesh with normals, centroid and closedness; a closed mesh with inward winding is flipped."""
    V = np.asarray(vertices, dtype=float).reshape(-1, 3)
    F = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    keep = area2 > 1e-12 * max(float(np.ptp(V, axis=0).max()) if len(V) else 1.0, 1e-9) ** 2
    F, a, b, c, cr, area2 = F[keep], a[keep], b[keep], c[keep], cr[keep], area2[keep]
    if not len(F):
        raise ValueError("Mesh has no non-degenerate triangle")
    closed = _is_closed(V, F)
    if closed and np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        F, cr = F[:, ::-1].copy(), -cr
    centroid = (area2[:, None] * (a + b + c) / 3.0).sum(axis=0) / area2.sum()
    return Mesh(V, F, cr / area2[:, None], centroid, closed, key)


def _diagonal(meshes) -> float:
    lo = np.min([m.vertices.min(axis=0) for m in meshes], axis=0)
    hi = np.max([m.vertices.max(axis=0) for m in meshes], axis=0)
    return float(np.linalg.norm(hi - lo)) or 1.0


# ---------------------------------------------------------------------------
# Camera, framing and rasterization
# ---------------------------------------------------------------------------


def view_basis(az: float, el: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r, u, v): screen right, screen up, viewing direction (depth = v . p grows away from the viewer).

    Azimuth about +z, elevation above the xy plane, up vector +z (spec 4).
    """
    a, e = np.radians(az), np.radians(el)
    toward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    r = np.array([-np.sin(a), np.cos(a), 0.0])
    return r, np.cross(toward, r), -toward


def view_angle(v1, v2) -> float:
    """Angle in degrees between the viewing directions of two (azimuth, elevation) views."""
    c = float(view_basis(*v1)[2] @ view_basis(*v2)[2])
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


@dataclass
class Layer:
    """One component rasterized in one view, cropped to its coverage; inf depth = not covered."""

    x0: int
    y0: int
    depth: np.ndarray  # (h, w) float32
    shade: np.ndarray  # (h, w) float32

    @property
    def mask(self) -> np.ndarray:
        return np.isfinite(self.depth)


def rasterize(xy: np.ndarray, z: np.ndarray, tris: np.ndarray, shade: np.ndarray, W: int, H: int) -> Layer:
    """Z-buffer of triangles (pixel coordinates xy (V, 2) = (col, row), pixel centres at +0.5).

    Every triangle covers the pixels of its bounding box whose centre passes the barycentric test; the
    boxes of many triangles are expanded together (vectorized), and the nearest sample wins per pixel.
    """
    depth = np.full(H * W, np.inf, dtype=np.float32)
    shd = np.zeros(H * W, dtype=np.float32)
    A, B, C = xy[tris[:, 0]], xy[tris[:, 1]], xy[tris[:, 2]]
    e1, e2 = B - A, C - A
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    xs = np.stack([A[:, 0], B[:, 0], C[:, 0]])
    ys = np.stack([A[:, 1], B[:, 1], C[:, 1]])
    x0 = np.clip(np.ceil(xs.min(0) - 0.5), 0, W).astype(np.int64)
    x1 = np.clip(np.floor(xs.max(0) - 0.5), -1, W - 1).astype(np.int64)
    y0 = np.clip(np.ceil(ys.min(0) - 0.5), 0, H).astype(np.int64)
    y1 = np.clip(np.floor(ys.max(0) - 0.5), -1, H - 1).astype(np.int64)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    idx = np.flatnonzero((np.abs(det) > 1e-12) & (w > 0) & (h > 0))
    cnt = (w * h)[idx]
    csum = np.cumsum(cnt)
    za, zb, zc = z[tris[:, 0]], z[tris[:, 1]], z[tris[:, 2]]
    start, done = 0, 0
    while start < len(idx):
        end = max(int(np.searchsorted(csum, done + CHUNK, side="right")), start + 1)
        sel, n = idx[start:end], cnt[start:end]
        rep = np.repeat(np.arange(len(sel)), n)
        local = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        t = sel[rep]
        ww = w[t]
        px, py = x0[t] + local % ww, y0[t] + local // ww
        dx, dy = px + 0.5 - A[t, 0], py + 0.5 - A[t, 1]
        l1 = (dx * e2[t, 1] - dy * e2[t, 0]) / det[t]
        l2 = (e1[t, 0] * dy - e1[t, 1] * dx) / det[t]
        ok = (l1 >= -BARY_TOL) & (l2 >= -BARY_TOL) & (l1 + l2 <= 1 + BARY_TOL)
        t, l1, l2 = t[ok], l1[ok], l2[ok]
        zz = za[t] + l1 * (zb[t] - za[t]) + l2 * (zc[t] - za[t])
        lin = py[ok] * W + px[ok]
        order = np.lexsort((zz, lin))
        first = order[np.r_[True, lin[order][1:] != lin[order][:-1]]] if len(order) else order
        lu, zu = lin[first], zz[first].astype(np.float32)
        better = zu < depth[lu]
        depth[lu[better]] = zu[better]
        shd[lu[better]] = shade[t[first][better]]
        done = int(csum[end - 1])
        start = end
    depth, shd = depth.reshape(H, W), shd.reshape(H, W)
    rows, cols = np.flatnonzero(np.isfinite(depth).any(1)), np.flatnonzero(np.isfinite(depth).any(0))
    if not len(rows):
        return Layer(0, 0, np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return Layer(int(c0), int(r0), depth[r0:r1, c0:c1].copy(), shd[r0:r1, c0:c1].copy())


def _clip(layer: Layer, ix: int, iy: int, W: int, H: int):
    """Image and layer slices of a layer shifted by (ix, iy) pixels, or None if it is outside the image."""
    h, w = layer.depth.shape
    gx, gy = layer.x0 + ix, layer.y0 + iy
    cx0, cy0, cx1, cy1 = max(gx, 0), max(gy, 0), min(gx + w, W), min(gy + h, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return (slice(cy0, cy1), slice(cx0, cx1)), (slice(cy0 - gy, cy1 - gy), slice(cx0 - gx, cx1 - gx))


def zmerge(W: int, H: int, items) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth-min merge of layers. items: (layer, ix, iy, dz, id). Returns depth, id (-1 = empty), shade."""
    depth = np.full((H, W), np.inf, dtype=np.float32)
    ids = np.full((H, W), -1, dtype=np.int32)
    shade = np.zeros((H, W), dtype=np.float32)
    for layer, ix, iy, dz, k in items:
        cl = _clip(layer, ix, iy, W, H)
        if cl is None:
            continue
        img, sub = cl
        d = layer.depth[sub] + np.float32(dz)
        better = d < depth[img]
        depth[img][better] = d[better]
        ids[img][better] = k
        shade[img][better] = layer.shade[sub][better]
    return depth, ids, shade


class ViewContext:
    """One view at one raster size: basis, constant framing over all steps, and its layers.

    The framing box is the union of the projected extents of every step's component at its final and at
    its start position, plus a margin (spec 5.3), so scale and offset are the same for every frame of the view.
    """

    def __init__(self, view, size, meshes: list[Mesh], offsets: np.ndarray, layer_dir: Path | None = None,
                 stats: dict | None = None):
        self.view = (float(view[0]), float(view[1]))
        self.r, self.u, self.v = view_basis(*self.view)
        self.W = int(size[0])
        self.H = int(round(size[1] * (1 - FOOTER_FRAC)))
        self.meshes, self.offsets = meshes, np.asarray(offsets, dtype=float).reshape(-1, 3)
        self.layer_dir, self.stats = layer_dir, stats if stats is not None else {}
        self.diag = _diagonal(meshes)
        self.eps = DEPTH_EPS_REL * self.diag
        R = np.stack([self.r, self.u], axis=1)
        self._extent = []
        lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
        for m, o in zip(meshes, self.offsets):
            p = m.vertices @ R
            a, b = p.min(axis=0), p.max(axis=0)
            self._extent.append((a, b))
            lo = np.minimum(lo, np.minimum(a, a + o @ R))
            hi = np.maximum(hi, np.maximum(b, b + o @ R))
        span = hi - lo
        margin = FRAME_MARGIN * max(float(span.max()), 1e-9)
        lo, hi = lo - margin, hi + margin
        self.center = (lo + hi) / 2
        self.scale = float(min(self.W / (hi[0] - lo[0]), self.H / (hi[1] - lo[1])))
        self.layers: dict[int, Layer] = {}

    def to_pixel(self, p) -> np.ndarray:
        """(col, row) of world points p (..., 3)."""
        p = np.asarray(p, dtype=float)
        x = (p @ self.r - self.center[0]) * self.scale + self.W / 2
        y = self.H / 2 - (p @ self.u - self.center[1]) * self.scale
        return np.stack([x, y], axis=-1)

    def shift(self, offset) -> tuple[float, float, float]:
        """Screen shift (pixels) and depth shift of a translation; exact under orthographic projection."""
        o = np.asarray(offset, dtype=float)
        return float(o @ self.r * self.scale), float(-(o @ self.u) * self.scale), float(o @ self.v)

    def in_frame(self, k: int, with_offset: bool) -> bool:
        a, b = self._extent[k]
        o = self.offsets[k] @ np.stack([self.r, self.u], axis=1) if with_offset else np.zeros(2)
        lo = np.array([(a[0] + o[0] - self.center[0]) * self.scale + self.W / 2,
                       self.H / 2 - (b[1] + o[1] - self.center[1]) * self.scale])
        hi = np.array([(b[0] + o[0] - self.center[0]) * self.scale + self.W / 2,
                       self.H / 2 - (a[1] + o[1] - self.center[1]) * self.scale])
        return bool(lo.min() >= -0.5 and hi[0] <= self.W + 0.5 and hi[1] <= self.H + 0.5)

    def _layer_key(self, k: int) -> str:
        m = self.meshes[k]
        payload = [LAYER_VERSION, m.key, self.W, self.H, *(round(x, 9) for x in
                   (*self.view, *self.center, self.scale))]
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    def layer(self, k: int) -> Layer:
        if k in self.layers:
            return self.layers[k]
        path = None
        if self.layer_dir is not None and self.meshes[k].key:
            path = self.layer_dir / f"{self._layer_key(k)}.npz"
            if path.exists():
                try:
                    d = np.load(path)
                    self.layers[k] = Layer(int(d["x0"]), int(d["y0"]), d["depth"], d["shade"])
                    self.stats["layers_loaded"] = self.stats.get("layers_loaded", 0) + 1
                    return self.layers[k]
                except Exception as exc:  # noqa: BLE001  (corrupt cache: recompute)
                    log.warning("Could not read layer %s (%s); re-rasterizing", path.name, exc)
        layer = self.rasterize_mesh(self.meshes[k])
        self.layers[k] = layer
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, x0=layer.x0, y0=layer.y0, depth=layer.depth, shade=layer.shade)
            os.replace(tmp, path)
        return layer

    def rasterize_mesh(self, mesh: Mesh) -> Layer:
        t0 = time.perf_counter()
        nv = mesh.normals @ self.v
        keep = nv < 0 if mesh.closed else np.ones(len(nv), dtype=bool)  # cull back faces of solids only
        shade = (AMBIENT + (1 - AMBIENT) * np.abs(nv)).astype(np.float32)
        layer = rasterize(self.to_pixel(mesh.vertices), mesh.vertices @ self.v, mesh.triangles[keep], shade[keep],
                          self.W, self.H)
        self.stats["layers_rasterized"] = self.stats.get("layers_rasterized", 0) + 1
        self.stats["raster_s"] = self.stats.get("raster_s", 0.0) + time.perf_counter() - t0
        return layer


# ---------------------------------------------------------------------------
# Visibility and view selection (spec 6)
# ---------------------------------------------------------------------------


def _step_visibility(ctx: ViewContext, k: int, placed: np.ndarray) -> dict:
    """Visible fractions of step k's part (end) and ghost (start) against the placed depth buffer."""
    layer = ctx.layer(k)
    dx, dy, dz = ctx.shift(ctx.offsets[k])
    out = {"traj": float(np.hypot(dx, dy)) / min(ctx.W, ctx.H),
           "in_frame": ctx.in_frame(k, False) and ctx.in_frame(k, True)}
    m = layer.mask
    n = int(m.sum())
    if n == 0:
        return {**out, "end": 0.0, "start": 0.0, "in_frame": False}
    for name, ix, iy, d in (("end", 0, 0, 0.0), ("start", int(round(dx)), int(round(dy)), dz)):
        cl = _clip(layer, ix, iy, ctx.W, ctx.H)
        if cl is None:
            out[name] = 0.0
            continue
        img, sub = cl
        dep = layer.depth[sub] + np.float32(d)
        out[name] = float((np.isfinite(dep) & (dep <= placed[img] + ctx.eps)).sum()) / n
    return out


def evaluate_view(ctx: ViewContext, only: int | None = None) -> dict[str, np.ndarray]:
    """Visibility figures of every step (or only step index `only`, NaN elsewhere) in one view."""
    N = len(ctx.meshes)
    res = {name: np.full(N, np.nan) for name in ("end", "start", "traj")}
    res["in_frame"] = np.zeros(N, dtype=bool)
    placed = np.full((ctx.H, ctx.W), np.inf, dtype=np.float32)
    last = N - 1 if only is None else only
    for s in range(last + 1):
        if only is None or s == only:
            v = _step_visibility(ctx, s, placed)
            for name in ("end", "start", "traj", "in_frame"):
                res[name][s] = v[name]
        if s < last:
            layer = ctx.layer(s)
            cl = _clip(layer, 0, 0, ctx.W, ctx.H)
            if cl is not None:
                np.minimum(placed[cl[0]], layer.depth[cl[1]], out=placed[cl[0]])
    return res


def _ratio(x, thr):
    return np.ones_like(x) if thr <= 0 else x / thr


def visible_mask(ev: dict, config: IllustrationConfig, occlusion: bool = True) -> np.ndarray:
    """Spec 6.1 rule; with occlusion=False (x-ray) only the trajectory and framing conditions count."""
    ok = (ev["traj"] >= config.min_trajectory_px_frac) & ev["in_frame"]
    if occlusion:
        ok &= (ev["end"] >= config.vis_min_end) & (ev["start"] >= config.vis_min_start)
    return ok


def step_score(ev: dict, config: IllustrationConfig) -> np.ndarray:
    """Bottleneck of the visibility conditions (>= 1 means visible), used to pick the view to perturb."""
    s = np.minimum(np.minimum(_ratio(ev["end"], config.vis_min_end), _ratio(ev["start"], config.vis_min_start)),
                   _ratio(ev["traj"], config.min_trajectory_px_frac))
    return np.where(ev["in_frame"], np.nan_to_num(s, nan=-1.0), -1.0)


def candidate_views(config: IllustrationConfig) -> list[tuple[float, float]]:
    """default_view first, then the (azimuth, elevation) grid; duplicates removed."""
    out, seen = [], set()
    for az, el in [tuple(config.default_view)] + [(a, e) for e in config.elevations for a in config.azimuths]:
        key = (round(float(az) % 360.0, 6), round(float(el), 6))
        if key not in seen:
            seen.add(key)
            out.append((float(az), float(el)))
    return out


def perturbations(config: IllustrationConfig) -> list[tuple[float, float]]:
    """(d_azimuth, d_elevation) offsets, nearest ring first."""
    n = int(config.perturb_max_deg // config.perturb_step_deg)
    st = config.perturb_step_deg
    offs = [(i, j) for i in range(-n, n + 1) for j in range(-n, n + 1) if (i, j) != (0, 0)]
    offs.sort(key=lambda ij: (max(abs(ij[0]), abs(ij[1])), np.hypot(*ij), ij))
    return [(i * st, j * st) for i, j in offs]


def _figures(ev: dict, s: int, occluded: bool = True) -> dict:
    return {"vis_end": float(ev["end"][s]) if occluded else 1.0,
            "vis_start": float(ev["start"][s]) if occluded else 1.0,
            "vis_end_occluded": float(ev["end"][s]), "vis_start_occluded": float(ev["start"][s]),
            "trajectory_px_frac": float(ev["traj"][s]), "in_frame": bool(ev["in_frame"][s])}


def select_views(meshes: list[Mesh], offsets: np.ndarray, config: IllustrationConfig,
                 stats: dict | None = None) -> tuple[tuple[float, float], list[dict]]:
    """Spec 6.2 cascade at search resolution. Returns (global view, per-step choice)."""
    stats = stats if stats is not None else {}
    N = len(meshes)
    cands = candidate_views(config)
    evals = []
    for view in cands:
        ctx = ViewContext(view, config.search_size, meshes, offsets, stats=stats)
        ev = evaluate_view(ctx)
        ev["visible"] = visible_mask(ev, config)
        evals.append(ev)
    stats["views_evaluated"] = stats.get("views_evaluated", 0) + len(cands)

    def global_rank(i):
        ev = evals[i]
        return (-int(ev["visible"].sum()), -round(float(np.nansum(ev["end"])), 9),
                round(view_angle(cands[i], config.default_view), 9), i)

    g = min(range(len(cands)), key=global_rank)
    glob = cands[g]
    by_distance = sorted((i for i in range(len(cands)) if i != g),
                         key=lambda i: (round(view_angle(cands[i], glob), 9), i))
    log.info("Global view az %.1f el %.1f: %d / %d steps visible", *glob, int(evals[g]["visible"].sum()), N)

    choices = []
    for s in range(N):
        base = {"step_index": s, "warnings": []}
        if evals[g]["visible"][s]:
            choices.append({**base, "view": glob, "is_global": True, "kind": "global", "visible": True,
                            "xray": False, **_figures(evals[g], s)})
            continue
        hit = next((i for i in by_distance if evals[i]["visible"][s]), None)
        if hit is not None:
            choices.append({**base, "view": cands[hit], "is_global": False, "kind": "candidate", "visible": True,
                            "xray": False, **_figures(evals[hit], s)})
            continue
        # Perturb around the candidate that comes closest (spec 6.2 step 3).
        scores = [step_score(evals[i], config)[s] for i in range(len(cands))]
        best = max(range(len(cands)), key=lambda i: (round(float(scores[i]), 9), -view_angle(cands[i], glob), -i))
        tried = []  # (view, evaluation)
        found = None
        for daz, dele in perturbations(config):
            view = (cands[best][0] + daz, cands[best][1] + dele)
            if not 0 <= view[1] <= 90:
                continue
            ev = evaluate_view(ViewContext(view, config.search_size, meshes, offsets, stats=stats), only=s)
            stats["views_evaluated"] = stats.get("views_evaluated", 0) + 1
            tried.append((view, ev))
            if visible_mask(ev, config)[s]:
                found = (view, ev)
                break
        if found is not None:
            choices.append({**base, "view": found[0], "is_global": False, "kind": "perturbed", "visible": True,
                            "xray": False, **_figures(found[1], s)})
            continue
        # X-ray fallback (spec 6.2 step 4): visibility judged without occlusion, cascade order again.
        order = [(cands[i], evals[i]) for i in [g] + by_distance] + tried
        xr = next(((v, ev) for v, ev in order if visible_mask(ev, config, occlusion=False)[s]), None)
        if xr is not None:
            msg = (f"step {s + 1}: no view shows the part unoccluded; x-ray fallback at az {xr[0][0]:.1f} "
                   f"el {xr[0][1]:.1f}")
            log.warning(msg)
            choices.append({**base, "view": xr[0], "is_global": xr[0] == glob, "kind": "xray", "visible": True,
                            "xray": True, **_figures(xr[1], s, occluded=False), "warnings": [msg]})
            continue
        msg = (f"step {s + 1}: not visible in any view (degenerate trajectory or out of frame); using the "
               f"best-scoring view az {cands[best][0]:.1f} el {cands[best][1]:.1f}")
        log.warning(msg)
        occluded = bool(evals[best]["end"][s] < config.vis_min_end or evals[best]["start"][s] < config.vis_min_start)
        choices.append({**base, "view": cands[best], "is_global": best == g, "kind": "none", "visible": False,
                        "xray": occluded, **_figures(evals[best], s), "warnings": [msg]})
    return glob, choices


# ---------------------------------------------------------------------------
# Frame composition (spec 5.3) and footnote (spec 7)
# ---------------------------------------------------------------------------


def _rgb(color) -> np.ndarray:
    from matplotlib.colors import to_rgb

    return np.array(to_rgb(color), dtype=np.float32)


def _edges(ids: np.ndarray, depth: np.ndarray, jump: float) -> np.ndarray:
    """Silhouette pixels: the component ID changes, or the depth jumps by more than `jump` inside one part."""
    e = np.zeros(ids.shape, dtype=bool)
    for axis in (0, 1):
        a = [slice(None), slice(None)]
        b = [slice(None), slice(None)]
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        i1, i0 = ids[tuple(a)], ids[tuple(b)]
        with np.errstate(invalid="ignore"):
            jumpy = (i1 == i0) & (i1 >= 0) & (np.abs(depth[tuple(a)] - depth[tuple(b)]) > jump)
        mark = (i1 != i0) | jumpy
        e[tuple(a)] |= mark
        e[tuple(b)] |= mark
    return e


def _paint(ids: np.ndarray, shade: np.ndarray, colors: np.ndarray, rgb: np.ndarray) -> None:
    m = ids >= 0
    rgb[m] = colors[ids[m]] * shade[m, None]


def _ghost_overlay(ctx: ViewContext, s: int, rgb: np.ndarray, scene_depth: np.ndarray, color: np.ndarray,
                   alpha: float) -> None:
    """Start-position copy: shifted layer, alpha-blended where it is not behind the scene, dashed outline."""
    layer = ctx.layer(s)
    dx, dy, dz = ctx.shift(ctx.offsets[s])
    gd, gid, gsh = zmerge(ctx.W, ctx.H, [(layer, int(round(dx)), int(round(dy)), dz, 0)])
    vis = (gid >= 0) & (gd <= scene_depth + ctx.eps)
    rgb[vis] = (1 - alpha) * rgb[vis] + alpha * color * gsh[vis, None]
    yy, xx = np.indices(gid.shape)
    dash = max(3, ctx.W // 160)
    line = _edges(gid, gd, np.inf) & vis & (((xx + yy) // dash) % 2 == 0)
    rgb[line] = 0.55 * color


def compose_frame(ctx: ViewContext, s: int | None, config: IllustrationConfig, is_stander: bool = False,
                  xray: bool = False) -> np.ndarray:
    """RGB image (H, W, 3) of step index s: placed parts 0..s-1, part s, its ghost. s=None: all parts placed."""
    W, H = ctx.W, ctx.H
    N = len(ctx.meshes)
    placed_rgb = _rgb(config.placed_color)
    new_rgb = _rgb(config.stander_color if is_stander else config.new_color)
    jump = EDGE_JUMP_REL * ctx.diag
    n_placed = N if s is None else s
    colors = np.array([placed_rgb] * n_placed + [new_rgb] * (N - n_placed))
    placed_items = [(ctx.layer(k), 0, 0, 0.0, k) for k in range(n_placed)]
    new_items = [] if s is None else [(ctx.layer(s), 0, 0, 0.0, s)]
    rgb = np.ones((H, W, 3), dtype=np.float32)

    if not xray:
        depth, ids, shade = zmerge(W, H, placed_items + new_items)
        _paint(ids, shade, colors, rgb)
        if s is not None:
            _ghost_overlay(ctx, s, rgb, depth, new_rgb, config.ghost_alpha)
        rgb[_edges(ids, depth, jump)] = OUTLINE_RGB
        return rgb

    # X-ray: part and ghost drawn alone, the placed parts blended over them at xray_alpha.
    depth, ids, shade = zmerge(W, H, new_items)
    _paint(ids, shade, colors, rgb)
    _ghost_overlay(ctx, s, rgb, depth, new_rgb, config.ghost_alpha)
    rgb[_edges(ids, depth, jump)] = OUTLINE_RGB
    pd, pid, psh = zmerge(W, H, placed_items)
    prgb = np.ones_like(rgb)
    _paint(pid, psh, colors, prgb)
    prgb[_edges(pid, pd, jump)] = OUTLINE_RGB
    m = pid >= 0
    rgb[m] = config.xray_alpha * prgb[m] + (1 - config.xray_alpha) * rgb[m]
    return rgb


def _fmt_vec(v) -> str:
    return "(" + ", ".join(f"{round(float(c), 2) + 0.0:g}" for c in v) + ")"


def tool_text(step: dict, bom: dict, language: str = "en") -> str:
    """BOM tool by Label; a blank or missing tool prints as 'by hand'."""
    tool = (bom.get(step["component_label"], {}).get("tool") or "").strip()
    return tool or STRINGS[language]["by_hand"]


def footnote_lines(steps: list[dict], s: int, bom: dict, config: IllustrationConfig) -> tuple[list[str], dict]:
    """Footnote of step index s (spec 7) and its flags {reorient, hold}."""
    T = STRINGS[config.language]
    st = steps[s]
    first = (f"{T['step']} {st['step']} / {len(steps)}   {st['component_label']}   "
             f"{T['tool']}: {tool_text(st, bom, config.language)}")
    if config.footnote_show_vector:
        first += f"   {T['move']}: {_fmt_vec(np.asarray(st['direction'], float) * float(st['travel']))} mm"
    lines = [first]
    g, g_prev = st.get("gravity"), steps[s - 1].get("gravity") if s > 0 else None
    reorient = g is not None and g_prev is not None and not np.allclose(g, g_prev, atol=1e-9)
    if reorient:
        lines.append(T["reorient"].format(g=_fmt_vec(g)))
    hold = bool(config.mark_unstable and st.get("stability_class") == 1)
    if hold:
        lines.append(T["hold"])
    return lines, {"reorient": reorient, "hold": hold}


def _snap(point: np.ndarray, other: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`point` if it lies on the mask, else the nearest mask pixel on the segment point-other (ring parts)."""
    H, W = mask.shape
    c, r = int(np.floor(point[0])), int(np.floor(point[1]))
    if 0 <= r < H and 0 <= c < W and mask[r, c]:
        return point
    n = max(int(np.ceil(np.linalg.norm(other - point) * 2)), 2)
    pts = point + np.linspace(0, 1, n)[:, None] * (other - point)
    cc, rr = np.floor(pts[:, 0]).astype(int), np.floor(pts[:, 1]).astype(int)
    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    ok[ok] = mask[rr[ok], cc[ok]]
    return pts[np.flatnonzero(ok)[0]] if ok.any() else point


def trajectory_points(ctx: ViewContext, s: int) -> tuple[np.ndarray, np.ndarray]:
    """(start, end) of the arrow in pixel coordinates: ghost centroid -> part centroid, snapped to the silhouettes."""
    end = ctx.to_pixel(ctx.meshes[s].centroid)
    dx, dy, _ = ctx.shift(ctx.offsets[s])
    start = end + np.array([dx, dy])
    layer = ctx.layer(s)
    _, own, _ = zmerge(ctx.W, ctx.H, [(layer, 0, 0, 0.0, 0)])
    _, gho, _ = zmerge(ctx.W, ctx.H, [(layer, int(round(dx)), int(round(dy)), 0.0, 0)])
    return _snap(start, end, gho >= 0), _snap(end, start, own >= 0)


def _draw_gravity(ax, ctx: ViewContext, g) -> None:
    from matplotlib.patches import Circle

    R = 0.055 * min(ctx.W, ctx.H)
    cx, cy = ctx.W - 1.7 * R, ctx.H - 1.7 * R
    ax.add_patch(Circle((cx, cy), R, facecolor="white", edgecolor="#555555", lw=1.0, zorder=5))
    g = np.asarray(g, dtype=float)
    vec = np.array([g @ ctx.r, -(g @ ctx.u)])
    n = float(np.linalg.norm(vec))
    if n < 0.25:  # gravity nearly along the view: dot = towards the viewer, cross = away
        if g @ ctx.v < 0:
            ax.add_patch(Circle((cx, cy), 0.15 * R, color="#222222", zorder=6))
        else:
            ax.plot([cx - 0.45 * R, cx + 0.45 * R], [cy - 0.45 * R, cy + 0.45 * R], color="#222222", lw=1.6, zorder=6)
            ax.plot([cx - 0.45 * R, cx + 0.45 * R], [cy + 0.45 * R, cy - 0.45 * R], color="#222222", lw=1.6, zorder=6)
    else:
        d = vec / n * 0.7 * R * min(1.0, n)
        ax.annotate("", xy=(cx + d[0], cy + d[1]), xytext=(cx - d[0], cy - d[1]), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.6, mutation_scale=12,
                                    shrinkA=0, shrinkB=0))
    ax.text(cx - 1.25 * R, cy - 0.85 * R, "g", fontsize=10, style="italic", ha="center", va="center", zorder=6)


def _figure(config: IllustrationConfig):
    from matplotlib.figure import Figure

    W, H = config.image_size
    return Figure(figsize=(W / config.dpi, H / config.dpi), dpi=config.dpi, facecolor="white")


def _image_axes(fig, ctx: ViewContext, rgb: np.ndarray):
    ax = fig.add_axes([0, FOOTER_FRAC, 1, 1 - FOOTER_FRAC])
    ax.imshow(rgb, extent=(0, ctx.W, ctx.H, 0), interpolation="none")
    ax.set_xlim(0, ctx.W)
    ax.set_ylim(ctx.H, 0)
    ax.axis("off")
    return ax


def draw_step(ctx: ViewContext, steps: list[dict], s: int, choice: dict, lines: list[str],
              config: IllustrationConfig):
    """Matplotlib figure of one step: frame, trajectory arrow, gravity pictogram, footnote."""
    st = steps[s]
    fig = _figure(config)
    ax = _image_axes(fig, ctx, compose_frame(ctx, s, config, bool(st["is_stander"]), choice["xray"]))
    a, b = trajectory_points(ctx, s)
    ax.plot([a[0], b[0]], [a[1], b[1]], linestyle=(0, (5, 4)), color=TRAJECTORY_COLOR, lw=1.5, zorder=4)
    length = float(np.linalg.norm(b - a))
    if length > 1e-6:
        head = min(length, 0.03 * min(ctx.W, ctx.H))
        ax.annotate("", xy=tuple(b), xytext=tuple(b - (b - a) / length * head), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=TRAJECTORY_COLOR, lw=1.5, mutation_scale=16,
                                    shrinkA=0, shrinkB=0))
    if config.show_gravity and st.get("gravity") is not None:
        _draw_gravity(ax, ctx, st["gravity"])
    if choice["xray"]:
        ax.text(0.01 * ctx.W, 0.02 * ctx.H, "x-ray", fontsize=8, color="#666666", va="top")
    for i, line in enumerate(lines):
        fig.text(0.02, FOOTER_FRAC * (0.75 - 0.28 * i), line, fontsize=11 if i == 0 else 9,
                 weight="bold" if i == 0 else "normal", va="center", color="#111111" if i == 0 else "#a33")
    return fig


def draw_cover(ctx: ViewContext, steps: list[dict], bom: dict, title: str, config: IllustrationConfig):
    """Cover page: title, number of steps, parts list, final assembly in the global view."""
    T = STRINGS[config.language]
    fig = _figure(config)
    fig.text(0.03, 0.95, title, fontsize=16, weight="bold", va="top")
    fig.text(0.03, 0.89, T["steps"].format(n=len(steps)), fontsize=10, va="top", color="#444444")
    rows = []
    for st in steps:
        row = bom.get(st["component_label"], {})
        rows.append([st["component_label"], T["yes"] if st["is_stander"] else T["no"],
                     tool_text(st, bom, config.language) if st["is_stander"] or row.get("tool") else "",
                     row.get("group") or ""])
    tax = fig.add_axes([0.02, 0.06, 0.46, 0.78])
    tax.axis("off")
    table = tax.table(cellText=rows, colLabels=list(T["columns"]), loc="upper left", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7 if len(rows) > 10 else 8)
    table.auto_set_column_width(list(range(len(T["columns"]))))
    table.scale(1.0, 1.25)
    rgb = compose_frame(ctx, None, config)
    # The manual framing leaves room for every start position; the cover crops to the assembly itself.
    filled = np.flatnonzero((rgb < 0.999).any(axis=(1, 2))), np.flatnonzero((rgb < 0.999).any(axis=(0, 2)))
    pad = int(0.04 * min(ctx.W, ctx.H))
    r0, r1 = max(filled[0][0] - pad, 0), min(filled[0][-1] + pad + 1, ctx.H)
    c0, c1 = max(filled[1][0] - pad, 0), min(filled[1][-1] + pad + 1, ctx.W)
    iax = fig.add_axes([0.53, 0.06, 0.45, 0.78])
    iax.imshow(rgb[r0:r1, c0:c1], interpolation="antialiased")
    iax.axis("off")
    iax.set_title(T["final"], fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# Result, caching and entry points
# ---------------------------------------------------------------------------


@dataclass
class IllustrationResult:
    frames: list[dict]  # per step (assembly order): png, view, visibility figures, flags, warnings
    manual_path: str
    global_view: tuple[float, float]
    config_hash: str
    stats: dict = field(default_factory=dict)


def _steps_of(sequence) -> list[dict]:
    steps = list(getattr(sequence, "steps", sequence))
    ids = [s["component_id"] for s in steps]
    if len(set(ids)) != len(ids):
        raise ValueError("A component appears more than once in the sequence")
    if not steps:
        raise ValueError("The sequence has no steps")
    return steps


def compute_config_hash(steps: list[dict], meshes: list[Mesh], bom: dict, config: IllustrationConfig) -> str:
    cfg = asdict(config)
    cfg.pop("force")
    seq = [[s["component_id"], s["component_label"], bool(s["is_stander"]),
            [round(float(x), 9) for x in s["direction"]], round(float(s["travel"]), 9),
            [round(float(x), 9) for x in s["start_offset"]],
            None if s.get("gravity") is None else [round(float(x), 9) for x in s["gravity"]],
            s.get("stability_class")] for s in steps]
    parts = [[bom.get(s["component_label"], {}).get(c, "") for c in ("tool", "group", "is_stander")] for s in steps]
    h = hashlib.sha256(f"v{FRAMES_VERSION};".encode())
    h.update(json.dumps({"sequence": seq, "meshes": [m.key for m in meshes], "bom": parts, "config": cfg},
                        sort_keys=True, default=list).encode())
    return h.hexdigest()


def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "part"


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_cached(output_dir: Path, key: str) -> IllustrationResult | None:
    path = output_dir / REPORT_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read %s (%s); re-rendering", path, exc)
        return None
    if data.get("config_hash") != key:
        return None
    files = [f["png"] for f in data["frames"]] + [data["manual_path"]]
    if not all(Path(f).exists() for f in files):
        return None
    return IllustrationResult(data["frames"], data["manual_path"], tuple(data["global_view"]), key, data["stats"])


def _cached_views(mesh_list, offsets, config, output_dir: Path, force: bool, stats: dict):
    """select_views, cached on disk keyed by meshes, start offsets and the search fields of the config."""
    h = hashlib.sha256(f"v{FRAMES_VERSION};".encode())
    h.update(json.dumps({"meshes": [m.key for m in mesh_list], "offsets": np.round(offsets, 9).tolist(),
                         "search": {f: getattr(config, f) for f in SEARCH_FIELDS}}, default=list).encode())
    key = h.hexdigest()
    path = output_dir / VIEWS_NAME
    if not force and all(m.key for m in mesh_list) and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("key") == key:
                for c in data["choices"]:
                    c["view"] = tuple(c["view"])
                stats["views_cached"] = True
                return tuple(data["global_view"]), data["choices"]
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read %s (%s); searching views again", path, exc)
    glob, choices = select_views(mesh_list, offsets, config, stats)
    _write_json_atomic(path, {"key": key, "global_view": glob, "choices": choices})
    return glob, choices


def render_illustrations(sequence, meshes: dict[str, Mesh], bom: dict, output_dir: str | Path,
                         config: IllustrationConfig | None = None, title: str = "Assembly",
                         force: bool | None = None) -> IllustrationResult:
    """PNG per step + manual PDF from meshes keyed by component_id (no FreeCAD). Cached on disk (spec 8).

    bom: Label -> row dict (columns tool, group, is_stander; missing columns read as blank).
    """
    from matplotlib.backends.backend_pdf import PdfPages

    config = config or IllustrationConfig()
    force = config.force if force is None else force
    validate_config(config)
    steps = _steps_of(sequence)
    missing = [s["component_id"] for s in steps if s["component_id"] not in meshes]
    if missing:
        raise ValueError(f"No mesh for component(s) {missing}")
    mesh_list = [meshes[s["component_id"]] for s in steps]
    offsets = np.array([s["start_offset"] for s in steps], dtype=float).reshape(-1, 3)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key = compute_config_hash(steps, mesh_list, bom, config)
    if not force:
        cached = _load_cached(output_dir, key)
        if cached is not None:
            log.info("Loaded cached illustrations from %s", output_dir)
            return cached

    t0 = time.perf_counter()
    stats: dict = {}
    glob, choices = _cached_views(mesh_list, offsets, config, output_dir, force, stats)
    stats["view_search_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    for old in output_dir.glob("step_*.png"):
        old.unlink()
    contexts: dict[tuple[float, float], ViewContext] = {}

    def context(view) -> ViewContext:
        if view not in contexts:
            contexts[view] = ViewContext(view, config.image_size, mesh_list, offsets,
                                         layer_dir=output_dir / "layers", stats=stats)
        return contexts[view]

    width = max(2, len(str(len(steps))))
    frames = []
    manual = output_dir / MANUAL_NAME
    tmp_manual = manual.with_suffix(".tmp.pdf")
    with PdfPages(tmp_manual, metadata={"Title": title, "CreationDate": None, "ModDate": None}) as pdf:
        pdf.savefig(draw_cover(context(glob), steps, bom, title, config))
        for s, choice in enumerate(choices):
            st = steps[s]
            lines, flags = footnote_lines(steps, s, bom, config)
            fig = draw_step(context(choice["view"]), steps, s, choice, lines, config)
            png = output_dir / f"step_{st['step']:0{width}d}_{_safe(st['component_label'])}.png"
            fig.savefig(png, dpi=config.dpi, metadata={"Software": None})
            pdf.savefig(fig)
            frames.append({
                "step": st["step"],
                "component_id": st["component_id"],
                "component_label": st["component_label"],
                "is_stander": bool(st["is_stander"]),
                "png": str(png),
                "view": {"azimuth": choice["view"][0], "elevation": choice["view"][1],
                         "is_global": choice["is_global"]},
                "view_kind": choice["kind"],
                "visible": choice["visible"],
                "xray": choice["xray"],
                **{k: choice[k] for k in ("vis_end", "vis_start", "vis_end_occluded", "vis_start_occluded",
                                          "trajectory_px_frac", "in_frame")},
                "footnote": lines,
                **flags,
                "warnings": choice["warnings"],
            })
            # Free the layers of views no later step uses (bounds memory at final resolution).
            if all(c["view"] != choice["view"] for c in choices[s + 1:]) and choice["view"] != glob:
                contexts.pop(choice["view"], None)
    os.replace(tmp_manual, manual)
    stats["render_s"] = time.perf_counter() - t1
    stats["total_s"] = time.perf_counter() - t0

    result = IllustrationResult(frames, str(manual), glob, key, stats)
    _write_json_atomic(output_dir / REPORT_NAME, {**asdict(result), "config": {
        k: v for k, v in asdict(config).items() if k != "force"}})
    n_vis = sum(f["visible"] and not f["xray"] for f in frames)
    log.info("Illustrations: %d frames (%d visible, %d x-ray, %d not visible), global view az %.1f el %.1f, "
             "search %.1f s, render %.1f s", len(frames), n_vis, sum(f["xray"] for f in frames),
             sum(not f["visible"] for f in frames), *glob, stats["view_search_s"], stats["render_s"])
    return result


# ---------------------------------------------------------------------------
# FreeCAD side
# ---------------------------------------------------------------------------


def read_bom(bom_path) -> dict[str, dict]:
    """Label -> row (stripped strings). Missing columns (for example `tool`) simply read as blank."""
    rows = {}
    with open(bom_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if row.get("label"):
                rows[row["label"]] = row
    return rows


def shape_geometry_hash(shape) -> str:
    """Session-independent hash of a FreeCAD shape: global B-rep vertices and face/edge geometry types.

    Volume, Area and BoundBox are left out on purpose: once a shape carries a triangulation (after
    tessellate), OCC evaluates them on the mesh and they change in the last digits.
    """
    pts = np.array([[v.X, v.Y, v.Z] for v in shape.Vertexes], dtype=float).reshape(-1, 3).round(6) + 0.0
    pts = pts[np.lexsort(pts.T[::-1])] if len(pts) else pts
    head = [len(shape.Solids), len(shape.Edges), sorted(f.Surface.TypeId for f in shape.Faces)]
    h = hashlib.sha256(json.dumps(head).encode())
    h.update(pts.tobytes())
    return h.hexdigest()


def tessellate_shape(shape, tol: float, mesh_dir: Path | None = None, force: bool = False) -> Mesh:
    """Mesh of a global-placement shape, cached as .npz keyed by geometry hash and tolerance (spec 5.1)."""
    key = hashlib.sha256(f"v{MESH_VERSION};{shape_geometry_hash(shape)};{tol:.9g}".encode()).hexdigest()
    path = None if mesh_dir is None else Path(mesh_dir) / f"{key}.npz"
    if path is not None and path.exists() and not force:
        try:
            d = np.load(path)
            return mesh_from_arrays(d["vertices"], d["triangles"], key)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read mesh %s (%s); re-tessellating", path.name, exc)
    pts, facets = shape.tessellate(tol)
    if not facets:
        raise ValueError("Tessellation produced no triangles")
    mesh = mesh_from_arrays([[p.x, p.y, p.z] for p in pts], facets, key)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, vertices=mesh.vertices, triangles=mesh.triangles)
        os.replace(tmp, path)
    return mesh


def generate_illustrations(sequence, prep, output_dir: str | Path, config: IllustrationConfig | None = None,
                           force: bool | None = None) -> IllustrationResult:
    """Pipeline entry: shapes from the Preprocessing documents -> meshes -> frames and manual."""
    from shapes import component_shapes

    config = config or IllustrationConfig()
    force = config.force if force is None else force
    validate_config(config)
    steps = _steps_of(sequence)
    output_dir = Path(output_dir)
    shapes = component_shapes([s["component_id"] for s in steps], [s["is_stander"] for s in steps], prep)
    if prep.assembly_bbox is not None:
        lo, hi = prep.assembly_bbox
        diag = float(np.linalg.norm([hi.x - lo.x, hi.y - lo.y, hi.z - lo.z]))
    else:
        bb = shapes[0].BoundBox
        for sh in shapes[1:]:
            bb = bb.united(sh.BoundBox)
        diag = float(bb.DiagonalLength)
    tol = config.tess_tolerance_rel * diag
    t0 = time.perf_counter()
    meshes = {s["component_id"]: tessellate_shape(sh, tol, output_dir / "meshes", force)
              for s, sh in zip(steps, shapes)}
    t_mesh = time.perf_counter() - t0
    log.info("Meshes: %d components, %d triangles, tolerance %.3f mm, %.1f s", len(meshes),
             sum(len(m.triangles) for m in meshes.values()), tol, t_mesh)
    bom = read_bom(prep.bom_path) if prep.bom_path is not None else {}
    title = config.title or (Path(prep.step_path).stem if prep.step_path is not None else prep.reduced_assembly.Name)
    result = render_illustrations(steps, meshes, bom, output_dir, config, title=title, force=force)
    result.stats["tessellation_s"] = t_mesh
    return result
