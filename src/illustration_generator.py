"""Illustration Generator: one action diagram per assembly step plus a PDF manual.

Inputs
    - The assembly sequence (`OptimizationResult.steps`, or a list of step dicts): per step the component,
      insertion direction, travel, start offset, gravity, stability class and screw turns.
    - The `PreprocessingResult` (to get each component's global FreeCAD shape and the assembly box).
    - The BOM CSV (`tool`, `group`, `is_stander` columns, matched on Label).
    - An `IllustrationConfig` (image size, view search, colours, footnote options).
Outputs
    - In the output folder (default `output/<assembly>/illustrations`): one PNG per frame
      (`step_<n>_<label>.png`, or `step_<a>-<b>_<group>.png` for a group frame), `manual.pdf` (cover page +
      one page per frame), `illustration_report.json` (views, visibility figures, warnings) and cache
      folders (`meshes/`, `layers/`, `view_selection.json`).
    - An `IllustrationResult` describing the frames.

What this module does
    Every component is tessellated once (FreeCAD), rasterised per view into a layer (depth + shade, numpy
    z-buffer, orthographic), and frames are depth-min merges of layers: the placed parts (steps 1..t-1),
    the new part at its final position and its ghost at the start position (the same layer shifted on
    screen, exact under an orthographic projection). The view is one global camera for the whole manual,
    replaced for a step only when that step fails the visibility rule; if no view works the part is
    drawn in x-ray. Consecutive steps of the same BOM group share one frame.

Layout:
  IllustrationConfig, validate_config      configuration
  Mesh, mesh_from_arrays                   triangle meshes
  view_basis, ViewContext, rasterize       camera, framing, layers
  evaluate_view, select_views              visibility rule and view cascade
  compose_frame, footnote_lines            frame content
  render_illustrations                     pure numpy/matplotlib entry, with caching
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

from shapes import shape_geometry_hash

log = logging.getLogger("aap.illustration")

FRAME_MARGIN = 0.06  # framing box margin, fraction of its larger side
FOOTER_FRAC = 0.1  # bottom band of each frame reserved for the footnote
AMBIENT = 0.35  # Lambert shading: ambient share, the rest is one headlight
DEPTH_EPS_REL = 1e-4  # depth tolerance (fraction of the scene diagonal) for touching surfaces
EDGE_JUMP_REL = 0.02  # depth jump (fraction of the scene diagonal) drawn as a line inside one part
BARY_TOL = 1e-7  # barycentric tolerance, closes cracks between neighbouring triangles
CHUNK = 1_000_000  # candidate pixels per rasterizer batch (memory bound)
OUTLINE_RGB = np.array([0.12, 0.12, 0.12])  # colour of the silhouette lines
TRAJECTORY_COLOR = "#1a1a1a"  # colour of the dashed trajectory arrow
MESH_VERSION = "1"  # bump when the mesh cache contents change
LAYER_VERSION = "1"  # bump when the layer cache contents change
FRAMES_VERSION = "3"  # 2: group frames; 3: no view perturbation, stander fallback

REPORT_NAME = "illustration_report.json"
VIEWS_NAME = "view_selection.json"
# Config fields the view search depends on (the view cache ignores colours, footnote and page options).
SEARCH_FIELDS = ("search_size", "default_view", "azimuths", "elevations", "vis_min_end", "vis_min_start",
                 "min_trajectory_px_frac")
MANUAL_NAME = "manual.pdf"

# Texts of the footnote, cover page and table, per language (only English exists).
STRINGS = {
    "en": {
        "step": "Step",
        "steps_range": "Steps",
        "parts": "Parts",
        "tool": "Tool",
        "by_hand": "by hand",
        "move": "Move",
        "reorient": "Re-orient assembly: gravity along {g}",
        "hold": "Hold in place",
        "steps": "{n} steps",
        "final": "Final assembly",
        "columns": ("Part", "Stander", "Tool", "Group", "Turns"),
        "yes": "yes",
        "no": "no",
        "turns": "Turns",
        "clockwise": "clockwise",
    },
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IllustrationConfig:
    """Settings of the illustration generation (config input of `render_illustrations` / `generate_illustrations`)."""

    image_size: tuple[int, int] = (1600, 1200)  # (width, height) px of the final frame. Recommended: (1600, 1200).
    search_size: tuple[int, int] = (480, 360)  # (width, height) px used for the view search (lower = faster).
    dpi: int = 200  # int. Resolution of the saved PNG and PDF page.
    tess_tolerance_rel: float = 0.002  # float. Tessellation deflection as a fraction of the assembly bbox diagonal.
    default_view: tuple[float, float] = (-35.0, 30.0)  # (azimuth, elevation) deg, start of the search order.
    azimuths: tuple = tuple(range(0, 360, 15))  # candidate azimuths (deg) of the view search.
    elevations: tuple = (20, 35, 50)  # candidate elevations (deg, 0-90) of the view search.
    vis_min_end: float = 0.35  # float in [0, 1]. Minimum visible fraction of the part at its final position.
    vis_min_start: float = 0.35  # float in [0, 1]. Minimum visible fraction of the ghost at the start position.
    min_trajectory_px_frac: float = 0.10  # float. Minimum on-screen length of the trajectory, fraction of the shorter image side.
    xray_alpha: float = 0.25  # float in [0, 1]. Opacity of the placed parts drawn over the new part in an x-ray frame.
    placed_color: str = "#bdbdbd"  # matplotlib colour of already placed parts.
    new_color: str = "#f28e2b"  # colour of the part being placed.
    stander_color: str = "#3b75af"  # colour of a stander being placed.
    ghost_alpha: float = 0.35  # float in [0, 1]. Opacity of the ghost (start position).
    show_gravity: bool = True  # bool. Draw the gravity pictogram (needs the stability data in the steps).
    mark_unstable: bool = True  # bool. Add "Hold in place" to the footnote of class-1 (unsupported) steps.
    footnote_show_vector: bool = False  # bool. Add the move vector (direction x travel) to the footnote.
    # bool. Consecutive steps whose BOM `group` matches (non-blank) share one frame; only the first movement
    # of the block must pass the visibility rule.
    group_frames: bool = True
    title: str | None = None  # str or None. Manual title; None = assembly name.
    language: str = "en"  # str. Key of STRINGS.
    force: bool = False  # bool. Ignore cached results and re-render.


def validate_config(config: IllustrationConfig) -> None:
    """[AUX] Validation of an `IllustrationConfig`; raises ValueError."""
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
    """Triangle mesh of one component in global coordinates (input of the rasteriser; built by `mesh_from_arrays`)."""

    vertices: np.ndarray  # (V, 3) float, mm, global coordinates.
    triangles: np.ndarray  # (T, 3) int, vertex indices, outward winding when closed.
    normals: np.ndarray  # (T, 3) float, unit triangle normals (used for shading and back-face culling).
    centroid: np.ndarray  # (3,) float, area-weighted centroid (anchor of the trajectory arrow).
    closed: bool  # bool. Closed and consistently oriented: back faces may be culled.
    key: str = ""  # str. Cache key (geometry hash + tessellation tolerance); "" disables caching.


def _is_closed(V: np.ndarray, F: np.ndarray) -> bool:
    """[AUX] True if every directed edge occurs once and its reverse exists (closed, consistently oriented)."""
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
    """[AUX] Build a `Mesh` with normals, centroid and closedness; degenerate triangles are dropped.

    A closed mesh with inward winding is flipped so that the normals point outward.
    """
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
    """[AUX] Length of the diagonal of the box enclosing all meshes (the scene scale for tolerances)."""
    lo = np.min([m.vertices.min(axis=0) for m in meshes], axis=0)
    hi = np.max([m.vertices.max(axis=0) for m in meshes], axis=0)
    return float(np.linalg.norm(hi - lo)) or 1.0


# ---------------------------------------------------------------------------
# Camera, framing and rasterization
# ---------------------------------------------------------------------------


def view_basis(az: float, el: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[AUX] Camera basis (r, u, v): screen right, screen up, viewing direction (depth = v . p grows away from the viewer).

    Azimuth about +z, elevation above the xy plane, up vector +z.
    """
    a, e = np.radians(az), np.radians(el)
    toward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    r = np.array([-np.sin(a), np.cos(a), 0.0])
    return r, np.cross(toward, r), -toward


def view_angle(v1, v2) -> float:
    """[AUX] Angle in degrees between the viewing directions of two (azimuth, elevation) views."""
    c = float(view_basis(*v1)[2] @ view_basis(*v2)[2])
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


@dataclass
class Layer:
    """One component rasterised in one view, cropped to its coverage; inf depth = not covered."""

    x0: int  # int. Column of the layer's left edge in the image.
    y0: int  # int. Row of the layer's top edge in the image.
    depth: np.ndarray  # (h, w) float32. Depth of the nearest surface per pixel (inf = empty).
    shade: np.ndarray  # (h, w) float32. Lambert shade per pixel.

    @property
    def mask(self) -> np.ndarray:
        """[AUX] Boolean mask of the pixels this layer covers."""
        return np.isfinite(self.depth)


def rasterize(xy: np.ndarray, z: np.ndarray, tris: np.ndarray, shade: np.ndarray, W: int, H: int) -> Layer:
    """[MAIN] Z-buffer rasteriser: draw triangles into a depth and shade image of W x H pixels.

    Inputs: projected vertices xy (V, 2) = (col, row) with pixel centres at +0.5, their depths z (V,), the
    triangles (T, 3), the flat shade of each triangle (T,) and the image size. Output: a `Layer` cropped to
    the covered area.

    Every triangle covers the pixels of its bounding box whose centre passes the barycentric test; the boxes
    of many triangles are expanded together (vectorised, in batches of CHUNK candidate pixels to bound the
    memory), and the nearest sample wins per pixel.
    """
    depth = np.full(H * W, np.inf, dtype=np.float32)
    shd = np.zeros(H * W, dtype=np.float32)
    # Per-triangle setup: edge vectors, determinant and the pixel bounding box.
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
        # One batch of triangles whose bounding boxes hold about CHUNK candidate pixels in total.
        end = max(int(np.searchsorted(csum, done + CHUNK, side="right")), start + 1)
        sel, n = idx[start:end], cnt[start:end]
        rep = np.repeat(np.arange(len(sel)), n)
        local = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        t = sel[rep]
        ww = w[t]
        px, py = x0[t] + local % ww, y0[t] + local // ww
        # Barycentric coordinates of each candidate pixel centre; keep those inside the triangle.
        dx, dy = px + 0.5 - A[t, 0], py + 0.5 - A[t, 1]
        l1 = (dx * e2[t, 1] - dy * e2[t, 0]) / det[t]
        l2 = (e1[t, 0] * dy - e1[t, 1] * dx) / det[t]
        ok = (l1 >= -BARY_TOL) & (l2 >= -BARY_TOL) & (l1 + l2 <= 1 + BARY_TOL)
        t, l1, l2 = t[ok], l1[ok], l2[ok]
        # Interpolated depth; per pixel keep the nearest sample, then write it if nearer than the buffer.
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
    # Crop to the covered rows and columns.
    depth, shd = depth.reshape(H, W), shd.reshape(H, W)
    rows, cols = np.flatnonzero(np.isfinite(depth).any(1)), np.flatnonzero(np.isfinite(depth).any(0))
    if not len(rows):
        return Layer(0, 0, np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return Layer(int(c0), int(r0), depth[r0:r1, c0:c1].copy(), shd[r0:r1, c0:c1].copy())


def _clip(layer: Layer, ix: int, iy: int, W: int, H: int):
    """[AUX] Image and layer slices of a layer shifted by (ix, iy) pixels, or None if it is outside the image."""
    h, w = layer.depth.shape
    gx, gy = layer.x0 + ix, layer.y0 + iy
    cx0, cy0, cx1, cy1 = max(gx, 0), max(gy, 0), min(gx + w, W), min(gy + h, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return (slice(cy0, cy1), slice(cx0, cx1)), (slice(cy0 - gy, cy1 - gy), slice(cx0 - gx, cx1 - gx))


def zmerge(W: int, H: int, items) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[MAIN] Depth-min merge of layers into one image: the nearest surface wins at every pixel.

    items: iterable of (layer, ix, iy, dz, id), a layer with its screen shift (pixels), depth shift and an
    id (the component index). Returns (depth, id image with -1 = empty, shade). This is how a frame is
    composed from placed parts, the new part and its ghost.
    """
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
    its start position, plus a margin, so scale and offset are the same for every frame of the view.
    Layers are rasterised lazily and cached in memory and (if `layer_dir` is given) on disk.
    """

    def __init__(self, view, size, meshes: list[Mesh], offsets: np.ndarray, layer_dir: Path | None = None,
                 stats: dict | None = None):
        """[MAIN] Set up the camera and framing of one view.

        Inputs: the (azimuth, elevation) view, the raster size (W, H), the meshes of the steps in assembly
        order, their start offsets (N, 3), an optional folder for the layer cache and a stats dict. The
        height used for drawing excludes the footnote band. The scale and centre are fixed here so every
        frame of this view has the same framing.
        """
        self.view = (float(view[0]), float(view[1]))
        self.r, self.u, self.v = view_basis(*self.view)
        self.W = int(size[0])
        self.H = int(round(size[1] * (1 - FOOTER_FRAC)))
        self.meshes, self.offsets = meshes, np.asarray(offsets, dtype=float).reshape(-1, 3)
        self.layer_dir, self.stats = layer_dir, stats if stats is not None else {}
        self.diag = _diagonal(meshes)
        self.eps = DEPTH_EPS_REL * self.diag
        # Framing: union of the projected extents of every part at its final and start position.
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
        """[AUX] (col, row) pixel coordinates of world points p (..., 3)."""
        p = np.asarray(p, dtype=float)
        x = (p @ self.r - self.center[0]) * self.scale + self.W / 2
        y = self.H / 2 - (p @ self.u - self.center[1]) * self.scale
        return np.stack([x, y], axis=-1)

    def shift(self, offset) -> tuple[float, float, float]:
        """[AUX] Screen shift (pixels) and depth shift of a translation; exact under orthographic projection."""
        o = np.asarray(offset, dtype=float)
        return float(o @ self.r * self.scale), float(-(o @ self.u) * self.scale), float(o @ self.v)

    def in_frame(self, k: int, with_offset: bool) -> bool:
        """[AUX] True if part k (at its final position, or shifted by its start offset) lies inside the image."""
        a, b = self._extent[k]
        o = self.offsets[k] @ np.stack([self.r, self.u], axis=1) if with_offset else np.zeros(2)
        lo = np.array([(a[0] + o[0] - self.center[0]) * self.scale + self.W / 2,
                       self.H / 2 - (b[1] + o[1] - self.center[1]) * self.scale])
        hi = np.array([(b[0] + o[0] - self.center[0]) * self.scale + self.W / 2,
                       self.H / 2 - (a[1] + o[1] - self.center[1]) * self.scale])
        return bool(lo.min() >= -0.5 and hi[0] <= self.W + 0.5 and hi[1] <= self.H + 0.5)

    def _layer_key(self, k: int) -> str:
        """[SUPP] Cache key of part k's layer in this view (mesh key, raster size, view and framing)."""
        m = self.meshes[k]
        payload = [LAYER_VERSION, m.key, self.W, self.H, *(round(x, 9) for x in
                   (*self.view, *self.center, self.scale))]
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    def layer(self, k: int) -> Layer:
        """[AUX] Layer of part k in this view: from memory, else from the disk cache, else rasterised (and saved)."""
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
        """[AUX] Rasterise one mesh in this view: back faces culled for closed solids, Lambert shade from the normals."""
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
# Visibility and view selection
# ---------------------------------------------------------------------------


def step_group(step: dict, bom: dict) -> str:
    """[AUX] BOM `group` of a step by Label (a `group` key on the step itself wins); blank = no group."""
    return (step.get("group") or bom.get(step["component_label"], {}).get("group") or "").strip()


def group_blocks(steps: list[dict], bom: dict, config: IllustrationConfig) -> list[tuple[int, int]]:
    """[AUX] (first, last) step indices of each frame: runs of consecutive steps sharing a non-blank group."""
    blocks: list[tuple[int, int]] = []
    for s, st in enumerate(steps):
        g = step_group(st, bom)
        if config.group_frames and g and blocks and step_group(steps[blocks[-1][1]], bom) == g:
            blocks[-1] = (blocks[-1][0], s)
        else:
            blocks.append((s, s))
    return blocks


def _merge_into(buf: np.ndarray, layer: Layer) -> None:
    """[AUX] Depth-min merge of a layer into a depth buffer, in place."""
    cl = _clip(layer, 0, 0, buf.shape[1], buf.shape[0])
    if cl is not None:
        np.minimum(buf[cl[0]], layer.depth[cl[1]], out=buf[cl[0]])


def _step_visibility(ctx: ViewContext, k: int, placed: np.ndarray) -> dict:
    """[AUX] Visible fractions of step k's part (end) and ghost (start) against the placed depth buffer.

    Also returns the on-screen trajectory length (fraction of the shorter image side) and whether the part
    and its ghost lie inside the image. These are the figures the visibility rule judges.
    """
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


def evaluate_view(ctx: ViewContext, blocks: list[tuple[int, int]] | None = None) -> dict[str, np.ndarray]:
    """[MAIN] Visibility figures of every frame in one view (input of the view selection).

    Output: dict of arrays over the frames: `end` and `start` (visible fraction of the part and of its
    ghost), `traj` (trajectory length fraction) and `in_frame`. Frames are judged in assembly order against
    the depth buffer of the parts placed before them.

    Without `blocks` every step is its own frame. For a block (a, b) only step a is judged, against the
    placed parts 0..a-1 and the other members a+1..b, which the same frame draws at their final positions.
    """
    blocks = blocks or [(s, s) for s in range(len(ctx.meshes))]
    B = len(blocks)
    res = {name: np.full(B, np.nan) for name in ("end", "start", "traj")}
    res["in_frame"] = np.zeros(B, dtype=bool)
    placed = np.full((ctx.H, ctx.W), np.inf, dtype=np.float32)
    for i, (a, b) in enumerate(blocks):
        occluders = placed
        if b > a:
            occluders = placed.copy()
            for k in range(a + 1, b + 1):
                _merge_into(occluders, ctx.layer(k))
        v = _step_visibility(ctx, a, occluders)
        for name in ("end", "start", "traj", "in_frame"):
            res[name][i] = v[name]
        for k in range(a, b + 1):
            _merge_into(placed, ctx.layer(k))
    return res


def _ratio(x, thr):
    """[AUX] x / thr, or 1 when the threshold is 0 (condition switched off)."""
    return np.ones_like(x) if thr <= 0 else x / thr


def visible_mask(ev: dict, config: IllustrationConfig, occlusion: bool = True) -> np.ndarray:
    """[AUX] Visibility rule per frame (bool array); with occlusion=False (x-ray) only the trajectory and framing conditions count."""
    ok = (ev["traj"] >= config.min_trajectory_px_frac) & ev["in_frame"]
    if occlusion:
        ok &= (ev["end"] >= config.vis_min_end) & (ev["start"] >= config.vis_min_start)
    return ok


def step_score(ev: dict, config: IllustrationConfig) -> np.ndarray:
    """[AUX] Bottleneck of the visibility conditions per frame (>= 1 means visible), used to pick the best failing view."""
    s = np.minimum(np.minimum(_ratio(ev["end"], config.vis_min_end), _ratio(ev["start"], config.vis_min_start)),
                   _ratio(ev["traj"], config.min_trajectory_px_frac))
    return np.where(ev["in_frame"], np.nan_to_num(s, nan=-1.0), -1.0)


def candidate_views(config: IllustrationConfig) -> list[tuple[float, float]]:
    """[AUX] Candidate (azimuth, elevation) views: default_view first, then the grid; duplicates removed."""
    out, seen = [], set()
    for az, el in [tuple(config.default_view)] + [(a, e) for e in config.elevations for a in config.azimuths]:
        key = (round(float(az) % 360.0, 6), round(float(el), 6))
        if key not in seen:
            seen.add(key)
            out.append((float(az), float(el)))
    return out


def _figures(ev: dict, s: int, occluded: bool = True) -> dict:
    """[AUX] The visibility figures of frame s as a plain dict for the report (unoccluded values for x-ray)."""
    return {"vis_end": float(ev["end"][s]) if occluded else 1.0,
            "vis_start": float(ev["start"][s]) if occluded else 1.0,
            "vis_end_occluded": float(ev["end"][s]), "vis_start_occluded": float(ev["start"][s]),
            "trajectory_px_frac": float(ev["traj"][s]), "in_frame": bool(ev["in_frame"][s])}


def select_views(meshes: list[Mesh], offsets: np.ndarray, config: IllustrationConfig,
                 stats: dict | None = None, blocks: list[tuple[int, int]] | None = None,
                 is_stander: list[bool] | None = None) -> tuple[tuple[float, float], list[dict]]:
    """[MAIN] View cascade at search resolution: the global view and the view of every frame.

    Inputs: the meshes and start offsets in assembly order, the config, an optional stats dict, the frame
    `blocks` ((first, last) step indices per frame, default one frame per step) and the is_stander flag of
    every step. Output: (global view, list of per-frame choices). Each choice holds the view, its kind
    ("global", "candidate", "stander", "xray" or "none"), the visibility figures and warnings.

    Cascade: evaluate every candidate view; the global view is the one that makes most frames visible. Per
    frame: use the global view if it passes the rule; else the nearest candidate view that passes; else, for
    a stander, draw normally in the best failing view; else the x-ray fallback; else the best-scoring view
    with a warning. Only the first movement of a block is judged, so that movement's is_stander
    flag (per step, default False) picks the fallback when no main view passes.
    """
    stats = stats if stats is not None else {}
    blocks = blocks or [(s, s) for s in range(len(meshes))]
    N = len(blocks)
    cands = candidate_views(config)
    evals = []
    for view in cands:
        ctx = ViewContext(view, config.search_size, meshes, offsets, stats=stats)
        ev = evaluate_view(ctx, blocks=blocks)
        ev["visible"] = visible_mask(ev, config)
        evals.append(ev)
    stats["views_evaluated"] = stats.get("views_evaluated", 0) + len(cands)

    # Global view: most visible frames, then largest total visible end fraction, then closest to default_view.
    def global_rank(i):
        ev = evals[i]
        return (-int(ev["visible"].sum()), -round(float(np.nansum(ev["end"])), 9),
                round(view_angle(cands[i], config.default_view), 9), i)

    g = min(range(len(cands)), key=global_rank)
    glob = cands[g]
    by_distance = sorted((i for i in range(len(cands)) if i != g),
                         key=lambda i: (round(view_angle(cands[i], glob), 9), i))
    log.info("Global view az %.1f el %.1f: %d / %d frames visible", *glob, int(evals[g]["visible"].sum()), N)

    choices = []
    for s in range(N):
        first = blocks[s][0]
        base = {"step_index": first, "warnings": []}
        # 1. The global view works.
        if evals[g]["visible"][s]:
            choices.append({**base, "view": glob, "is_global": True, "kind": "global", "visible": True,
                            "xray": False, **_figures(evals[g], s)})
            continue
        # 2. The nearest other candidate view that works.
        hit = next((i for i in by_distance if evals[i]["visible"][s]), None)
        if hit is not None:
            choices.append({**base, "view": cands[hit], "is_global": False, "kind": "candidate", "visible": True,
                            "xray": False, **_figures(evals[hit], s)})
            continue
        # No main view passes. Best failing view: clear trajectory first, then the bottleneck score, then near glob.
        scores = [step_score(evals[i], config)[s] for i in range(len(cands))]
        best = max(range(len(cands)), key=lambda i: (bool(visible_mask(evals[i], config, occlusion=False)[s]),
                                                     round(float(scores[i]), 9), -view_angle(cands[i], glob), -i))
        # 3. A stander is drawn normally in the best failing view (never x-ray).
        if is_stander is not None and is_stander[first]:
            # Standers are easy to assemble: drawn normally in the best view, never in x-ray.
            msg = (f"step {first + 1}: stander not fully visible in any main view; drawn without x-ray at "
                   f"az {cands[best][0]:.1f} el {cands[best][1]:.1f}")
            log.info(msg)
            choices.append({**base, "view": cands[best], "is_global": best == g, "kind": "stander", "visible": False,
                            "xray": False, **_figures(evals[best], s), "warnings": [msg]})
            continue
        # 4. X-ray fallback: visibility judged without occlusion, cascade order again.
        order = [(cands[i], evals[i]) for i in [g] + by_distance]
        xr = next(((v, ev) for v, ev in order if visible_mask(ev, config, occlusion=False)[s]), None)
        if xr is not None:
            msg = (f"step {first + 1}: no main view shows the part unoccluded; x-ray fallback at az {xr[0][0]:.1f} "
                   f"el {xr[0][1]:.1f}")
            log.warning(msg)
            choices.append({**base, "view": xr[0], "is_global": xr[0] == glob, "kind": "xray", "visible": True,
                            "xray": True, **_figures(xr[1], s, occluded=False), "warnings": [msg]})
            continue
        # 5. Nothing works (degenerate trajectory or out of frame): best-scoring view, with a warning.
        msg = (f"step {first + 1}: not visible in any view (degenerate trajectory or out of frame); using the "
               f"best-scoring view az {cands[best][0]:.1f} el {cands[best][1]:.1f}")
        log.warning(msg)
        occluded = bool(evals[best]["end"][s] < config.vis_min_end or evals[best]["start"][s] < config.vis_min_start)
        choices.append({**base, "view": cands[best], "is_global": best == g, "kind": "none", "visible": False,
                        "xray": occluded, **_figures(evals[best], s), "warnings": [msg]})
    return glob, choices


# ---------------------------------------------------------------------------
# Frame composition and footnote
# ---------------------------------------------------------------------------


def _rgb(color) -> np.ndarray:
    """[AUX] Matplotlib colour as a float32 RGB array."""
    from matplotlib.colors import to_rgb

    return np.array(to_rgb(color), dtype=np.float32)


def _edges(ids: np.ndarray, depth: np.ndarray, jump: float) -> np.ndarray:
    """[AUX] Silhouette pixels: the component id changes, or the depth jumps by more than `jump` inside one part."""
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
    """[AUX] Colour the covered pixels of rgb (in place) with the part colour times the shade."""
    m = ids >= 0
    rgb[m] = colors[ids[m]] * shade[m, None]


def _ghost_overlay(ctx: ViewContext, s: int, rgb: np.ndarray, scene_depth: np.ndarray, color: np.ndarray,
                   alpha: float) -> None:
    """[AUX] Draw the ghost (start-position copy) of part s onto rgb, in place.

    The layer is shifted by the start offset, alpha-blended where it is not behind the scene, with a
    dashed outline.
    """
    layer = ctx.layer(s)
    dx, dy, dz = ctx.shift(ctx.offsets[s])
    gd, gid, gsh = zmerge(ctx.W, ctx.H, [(layer, int(round(dx)), int(round(dy)), dz, 0)])
    vis = (gid >= 0) & (gd <= scene_depth + ctx.eps)
    rgb[vis] = (1 - alpha) * rgb[vis] + alpha * color * gsh[vis, None]
    yy, xx = np.indices(gid.shape)
    dash = max(3, ctx.W // 160)
    line = _edges(gid, gd, np.inf) & vis & (((xx + yy) // dash) % 2 == 0)
    rgb[line] = 0.55 * color


def compose_frame(ctx: ViewContext, s: int | None, config: IllustrationConfig, is_stander=False,
                  xray: bool = False, last: int | None = None) -> np.ndarray:
    """[MAIN] Compose the RGB image (H, W, 3) of one frame.

    Inputs: the view context, the step index s (None = final assembly, all parts placed), the config, the
    stander flag (one value, or one per part s..last), `xray` and `last` (the last step of a group frame).
    Output: the image with parts 0..s-1 in the placed colour and parts s..last in the new (or stander)
    colour, each with its ghost at the start position, plus silhouette lines.

    Normal frame: all layers are depth-merged and painted. X-ray frame: the new parts and ghosts are drawn
    alone and the placed parts are blended over them at `xray_alpha`, so a hidden part stays visible.
    """
    W, H = ctx.W, ctx.H
    N = len(ctx.meshes)
    placed_rgb = _rgb(config.placed_color)
    jump = EDGE_JUMP_REL * ctx.diag
    n_placed = N if s is None else s
    last = n_placed - 1 if s is None else (s if last is None else last)
    members = list(range(n_placed, last + 1))
    flags = list(is_stander) if isinstance(is_stander, (list, tuple)) else [bool(is_stander)] * len(members)
    member_rgb = {k: _rgb(config.stander_color if f else config.new_color) for k, f in zip(members, flags)}
    colors = np.array([placed_rgb] * n_placed + [member_rgb.get(k, placed_rgb) for k in range(n_placed, N)])
    placed_items = [(ctx.layer(k), 0, 0, 0.0, k) for k in range(n_placed)]
    new_items = [(ctx.layer(k), 0, 0, 0.0, k) for k in members]
    rgb = np.ones((H, W, 3), dtype=np.float32)

    if not xray:
        depth, ids, shade = zmerge(W, H, placed_items + new_items)
        _paint(ids, shade, colors, rgb)
        for k in members:
            _ghost_overlay(ctx, k, rgb, depth, member_rgb[k], config.ghost_alpha)
        rgb[_edges(ids, depth, jump)] = OUTLINE_RGB
        return rgb

    # X-ray: parts and ghosts drawn alone, the placed parts blended over them at xray_alpha.
    depth, ids, shade = zmerge(W, H, new_items)
    _paint(ids, shade, colors, rgb)
    for k in members:
        _ghost_overlay(ctx, k, rgb, depth, member_rgb[k], config.ghost_alpha)
    rgb[_edges(ids, depth, jump)] = OUTLINE_RGB
    pd, pid, psh = zmerge(W, H, placed_items)
    prgb = np.ones_like(rgb)
    _paint(pid, psh, colors, prgb)
    prgb[_edges(pid, pd, jump)] = OUTLINE_RGB
    m = pid >= 0
    rgb[m] = config.xray_alpha * prgb[m] + (1 - config.xray_alpha) * rgb[m]
    return rgb


def _fmt_vec(v) -> str:
    """[SUPP] Format a vector with up to 2 decimals for the footnote."""
    return "(" + ", ".join(f"{round(float(c), 2) + 0.0:g}" for c in v) + ")"


def tool_text(step: dict, bom: dict, language: str = "en") -> str:
    """[AUX] BOM tool of a step by Label; a blank or missing tool prints as 'by hand'."""
    tool = (bom.get(step["component_label"], {}).get("tool") or "").strip()
    return tool or STRINGS[language]["by_hand"]


def footnote_lines(steps: list[dict], s: int, bom: dict, config: IllustrationConfig) -> tuple[list[str], dict]:
    """[AUX] Footnote of step index s: step number, part, tool, turns, re-orient and hold notes.

    Returns (lines, {"reorient", "hold"}): "reorient" is set when the gravity differs from the previous
    step, "hold" when the part is class 1 (unsupported) and `mark_unstable` is on.
    """
    T = STRINGS[config.language]
    st = steps[s]
    first = (f"{T['step']} {st['step']} / {len(steps)}   {st['component_label']}   "
             f"{T['tool']}: {tool_text(st, bom, config.language)}")
    turns = st.get("turns")
    if turns is not None:
        first += f"   {T['turns']}: {turns:.1f} ({T['clockwise']})"
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


def block_footnote_lines(steps: list[dict], first: int, last: int, bom: dict,
                         config: IllustrationConfig) -> tuple[list[str], dict]:
    """[AUX] Footnote of a frame covering steps first..last; a single step uses `footnote_lines` unchanged.

    A group frame lists the step range, the group, the parts (with their tool or turns when they differ),
    and the re-orient and hold notes of the whole block.
    """
    if first == last:
        return footnote_lines(steps, first, bom, config)
    T = STRINGS[config.language]
    block = steps[first:last + 1]
    labels = [st["component_label"] for st in block]
    tools = [tool_text(st, bom, config.language) for st in block]
    turns_vals = [st.get("turns") for st in block]
    head = f"{T['steps_range']} {block[0]['step']}-{block[-1]['step']} / {len(steps)}   {step_group(block[0], bom)}"
    same_tool = len(set(tools)) == 1
    if same_tool:
        head += f"   {T['tool']}: {tools[0]}"
    has_turns = any(t is not None for t in turns_vals)
    if same_tool and not has_turns:
        parts_line = ", ".join(labels)
    else:
        parts = []
        for lab, tool, turns in zip(labels, tools, turns_vals):
            extras = [] if same_tool else [tool]
            if turns is not None:
                extras.append(f"{turns:.1f} {T['turns'].lower()}")
            parts.append(f"{lab} ({', '.join(extras)})" if extras else lab)
        parts_line = ", ".join(parts)
    lines = [head, f"{T['parts']}: " + parts_line]
    if config.footnote_show_vector:
        lines.append(f"{T['move']}: " + "; ".join(
            f"{st['component_label']} {_fmt_vec(np.asarray(st['direction'], float) * float(st['travel']))} mm"
            for st in block))
    changed = [s for s in range(first, last + 1) if s > 0 and steps[s].get("gravity") is not None
               and steps[s - 1].get("gravity") is not None
               and not np.allclose(steps[s]["gravity"], steps[s - 1]["gravity"], atol=1e-9)]
    if changed:
        lines.append(T["reorient"].format(g=_fmt_vec(steps[changed[-1]]["gravity"])))
    unstable = [st["component_label"] for st in block if st.get("stability_class") == 1]
    hold = bool(config.mark_unstable and unstable)
    if hold:
        lines.append(f"{T['hold']}: {', '.join(unstable)}")
    return lines, {"reorient": bool(changed), "hold": hold}


def _snap(point: np.ndarray, other: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[AUX] `point` if it lies on the mask, else the nearest mask pixel on the segment point-other (ring parts)."""
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
    """[AUX] (start, end) of the trajectory arrow in pixel coordinates: ghost centroid -> part centroid, snapped to the silhouettes."""
    end = ctx.to_pixel(ctx.meshes[s].centroid)
    dx, dy, _ = ctx.shift(ctx.offsets[s])
    start = end + np.array([dx, dy])
    layer = ctx.layer(s)
    _, own, _ = zmerge(ctx.W, ctx.H, [(layer, 0, 0, 0.0, 0)])
    _, gho, _ = zmerge(ctx.W, ctx.H, [(layer, int(round(dx)), int(round(dy)), 0.0, 0)])
    return _snap(start, end, gho >= 0), _snap(end, start, own >= 0)


def _draw_gravity(ax, ctx: ViewContext, g) -> None:
    """[AUX] Draw the gravity pictogram (a circle with an arrow, or a dot / cross if g points along the view) in a corner."""
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
    """[AUX] Empty matplotlib figure of the configured image size and dpi."""
    from matplotlib.figure import Figure

    W, H = config.image_size
    return Figure(figsize=(W / config.dpi, H / config.dpi), dpi=config.dpi, facecolor="white")


def _image_axes(fig, ctx: ViewContext, rgb: np.ndarray):
    """[AUX] Add the image axes (all but the footer band) to the figure and show the frame image in pixel coordinates."""
    ax = fig.add_axes([0, FOOTER_FRAC, 1, 1 - FOOTER_FRAC])
    ax.imshow(rgb, extent=(0, ctx.W, ctx.H, 0), interpolation="none")
    ax.set_xlim(0, ctx.W)
    ax.set_ylim(ctx.H, 0)
    ax.axis("off")
    return ax


def draw_step(ctx: ViewContext, steps: list[dict], s: int, choice: dict, lines: list[str],
              config: IllustrationConfig, last: int | None = None):
    """[MAIN] Matplotlib figure of one frame (steps s..last, default s): image, trajectory arrows, gravity, footnote.

    Inputs: the view context of the chosen view, the steps, the first step index s, the view choice (for the
    x-ray flag), the footnote lines and the config. Output: a figure ready to save as PNG and add to the PDF.
    It composes the frame, draws one dashed trajectory arrow per part, the gravity pictogram of the last
    step, an "x-ray" tag when needed, and the footnote lines (shrunk to fit the page width).
    """
    last = s if last is None else last
    block = range(s, last + 1)
    fig = _figure(config)
    rgb = compose_frame(ctx, s, config, [bool(steps[k]["is_stander"]) for k in block], choice["xray"], last)
    ax = _image_axes(fig, ctx, rgb)
    # One dashed trajectory (ghost -> part) with an arrowhead per part of the frame.
    for k in block:
        a, b = trajectory_points(ctx, k)
        ax.plot([a[0], b[0]], [a[1], b[1]], linestyle=(0, (5, 4)), color=TRAJECTORY_COLOR, lw=1.5, zorder=4)
        length = float(np.linalg.norm(b - a))
        if length > 1e-6:
            head = min(length, 0.03 * min(ctx.W, ctx.H))
            ax.annotate("", xy=tuple(b), xytext=tuple(b - (b - a) / length * head), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=TRAJECTORY_COLOR, lw=1.5, mutation_scale=16,
                                        shrinkA=0, shrinkB=0))
    if config.show_gravity and steps[last].get("gravity") is not None:
        _draw_gravity(ax, ctx, steps[last]["gravity"])
    if choice["xray"]:
        ax.text(0.01 * ctx.W, 0.02 * ctx.H, "x-ray", fontsize=8, color="#666666", va="top")
    # Footnote text in the bottom band; warning lines (hold / re-orient) are red.
    T = STRINGS[config.language]
    gap = 0.28 if len(lines) <= 3 else 0.84 / len(lines)  # more lines (group frames) are packed tighter
    small = 9 if len(lines) <= 3 else 7
    width_pt = 0.96 * config.image_size[0] / config.dpi * 72
    for i, line in enumerate(lines):
        size = 11 if i == 0 else small
        size = min(size, width_pt / (0.62 * max(len(line), 1)))  # shrink a line that would leave the page
        warn = line.startswith(T["hold"]) or line.startswith(T["reorient"].split("{")[0])
        fig.text(0.02, FOOTER_FRAC * (0.75 - gap * i), line, fontsize=size, weight="bold" if i == 0 else "normal",
                 va="center", color="#a33" if warn else "#111111")
    return fig


def draw_cover(ctx: ViewContext, steps: list[dict], bom: dict, title: str, config: IllustrationConfig):
    """[AUX] Cover page figure: title, number of steps, parts table (part, stander, tool, group, turns) and the final assembly."""
    T = STRINGS[config.language]
    fig = _figure(config)
    fig.text(0.03, 0.95, title, fontsize=16, weight="bold", va="top")
    fig.text(0.03, 0.89, T["steps"].format(n=len(steps)), fontsize=10, va="top", color="#444444")
    rows = []
    for st in steps:
        row = bom.get(st["component_label"], {})
        turns = st.get("turns")
        rows.append([st["component_label"], T["yes"] if st["is_stander"] else T["no"],
                     tool_text(st, bom, config.language) if st["is_stander"] or row.get("tool") else "",
                     row.get("group") or "", f"{turns:.1f}" if turns is not None else ""])
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
    """What the illustration stage produced (output of `render_illustrations` / `generate_illustrations`)."""

    frames: list[dict]  # list of dicts, one per frame in assembly order: png path, steps, view, visibility figures, flags, footnote, warnings.
    manual_path: str  # str. Path of manual.pdf.
    global_view: tuple[float, float]  # (azimuth, elevation) deg of the global camera.
    config_hash: str  # str. Hash of the sequence, meshes, BOM columns and config; identifies the cached result.
    stats: dict = field(default_factory=dict)  # timings and counters (view search, raster, render, ...).


def _steps_of(sequence) -> list[dict]:
    """[AUX] The step dicts of a sequence (an OptimizationResult or a list); ValueError if empty or a component repeats."""
    steps = list(getattr(sequence, "steps", sequence))
    ids = [s["component_id"] for s in steps]
    if len(set(ids)) != len(ids):
        raise ValueError("A component appears more than once in the sequence")
    if not steps:
        raise ValueError("The sequence has no steps")
    return steps


def compute_config_hash(steps: list[dict], meshes: list[Mesh], bom: dict, config: IllustrationConfig) -> str:
    """[SUPP] Cache key of the illustrations: hash of the sequence, mesh keys, BOM tool/group/is_stander and config (without `force`)."""
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
    """[SUPP] File-name safe version of a label."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "part"


def _write_json_atomic(path: Path, data) -> None:
    """[SUPP] Write JSON to a temporary file, then rename it over the target."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_cached(output_dir: Path, key: str) -> IllustrationResult | None:
    """[SUPP] Cache function: the saved result if its hash matches and all its PNGs and the PDF still exist, else None."""
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


def _cached_views(mesh_list, offsets, config, output_dir: Path, force: bool, stats: dict,
                  blocks: list[tuple[int, int]], is_stander: list[bool]):
    """[SUPP] Cache function: `select_views`, cached on disk keyed by meshes, start offsets, frame blocks, stander flags and search fields."""
    h = hashlib.sha256(f"v{FRAMES_VERSION};".encode())
    h.update(json.dumps({"meshes": [m.key for m in mesh_list], "offsets": np.round(offsets, 9).tolist(),
                         "blocks": [list(b) for b in blocks], "is_stander": is_stander,
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
    glob, choices = select_views(mesh_list, offsets, config, stats, blocks, is_stander)
    _write_json_atomic(path, {"key": key, "global_view": glob, "choices": choices})
    return glob, choices


def render_illustrations(sequence, meshes: dict[str, Mesh], bom: dict, output_dir: str | Path,
                         config: IllustrationConfig | None = None, title: str = "Assembly",
                         force: bool | None = None) -> IllustrationResult:
    """[MAIN] Render the PNG of every frame and the manual PDF from meshes keyed by component_id (no FreeCAD).

    Inputs: the sequence (steps), the meshes {component_id: Mesh}, the BOM {Label: row dict} (columns tool,
    group, is_stander; missing columns read as blank), the output folder, the config and the manual title.
    Output: an `IllustrationResult`; files are written to `output_dir`. Cached on disk: a matching
    result is returned without rendering.

    One frame per step, except that consecutive steps of the same BOM group share one frame.
    Order of work: group the steps into frames, choose the views (global view + per-frame cascade),
    then for every frame compose the image, draw the arrows, gravity and footnote, save the PNG and add the
    page to the PDF (the cover page first), and finally write the JSON report.
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

    # View selection (cached separately, it is the slow part).
    t0 = time.perf_counter()
    stats: dict = {}
    blocks = group_blocks(steps, bom, config)
    glob, choices = _cached_views(mesh_list, offsets, config, output_dir, force, stats, blocks,
                                  [bool(st["is_stander"]) for st in steps])
    stats["view_search_s"] = time.perf_counter() - t0

    # Rendering at the final resolution: one ViewContext per view actually used.
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
        for i, ((s, last), choice) in enumerate(zip(blocks, choices)):
            st, block = steps[s], steps[s:last + 1]
            lines, flags = block_footnote_lines(steps, s, last, bom, config)
            fig = draw_step(context(choice["view"]), steps, s, choice, lines, config, last)
            if last == s:
                name = f"step_{st['step']:0{width}d}_{_safe(st['component_label'])}.png"
            else:
                name = (f"step_{st['step']:0{width}d}-{steps[last]['step']:0{width}d}_"
                        f"{_safe(step_group(st, bom))}.png")
            png = output_dir / name
            fig.savefig(png, dpi=config.dpi, metadata={"Software": None})
            pdf.savefig(fig)
            frames.append({
                "step": st["step"],  # first step of the frame; its movement is the one judged visible
                "steps": [b["step"] for b in block],
                "component_id": st["component_id"],
                "component_label": st["component_label"],
                "component_ids": [b["component_id"] for b in block],
                "component_labels": [b["component_label"] for b in block],
                "group": step_group(st, bom) if last > s else None,
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
            if all(c["view"] != choice["view"] for c in choices[i + 1:]) and choice["view"] != glob:
                contexts.pop(choice["view"], None)
    os.replace(tmp_manual, manual)
    stats["render_s"] = time.perf_counter() - t1
    stats["total_s"] = time.perf_counter() - t0

    # Save the report and log the summary.
    result = IllustrationResult(frames, str(manual), glob, key, stats)
    _write_json_atomic(output_dir / REPORT_NAME, {**asdict(result), "config": {
        k: v for k, v in asdict(config).items() if k != "force"}})
    n_vis = sum(f["visible"] and not f["xray"] for f in frames)
    log.info("Illustrations: %d frames for %d steps (%d visible, %d x-ray, %d not visible), global view az %.1f "
             "el %.1f, search %.1f s, render %.1f s", len(frames), len(steps), n_vis, sum(f["xray"] for f in frames),
             sum(not f["visible"] for f in frames), *glob, stats["view_search_s"], stats["render_s"])
    return result


# ---------------------------------------------------------------------------
# FreeCAD side
# ---------------------------------------------------------------------------


def read_bom(bom_path) -> dict[str, dict]:
    """[AUX] Read the BOM as {Label: row dict} (stripped strings). Missing columns (for example `tool`) simply read as blank."""
    rows = {}
    with open(bom_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if row.get("label"):
                rows[row["label"]] = row
    return rows


def tessellate_shape(shape, tol: float, mesh_dir: Path | None = None, force: bool = False) -> Mesh:
    """[AUX] Triangle mesh of a global-placement shape, cached as .npz keyed by geometry hash and tolerance."""
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
    """[MAIN] Pipeline entry point: shapes from the Preprocessing documents -> meshes -> frames and manual.

    Inputs: the sequence (an `OptimizationResult` or step list), the `PreprocessingResult` (shapes, assembly
    box, BOM path, names), the output folder, the config. Output: an `IllustrationResult`. Gets each
    component's global shape (`shapes.component_shapes`), tessellates it with a tolerance relative to the
    assembly diagonal, reads the BOM and calls `render_illustrations`.
    """
    from shapes import component_shapes

    config = config or IllustrationConfig()
    force = config.force if force is None else force
    validate_config(config)
    steps = _steps_of(sequence)
    output_dir = Path(output_dir)
    shapes = component_shapes([s["component_id"] for s in steps], [s["is_stander"] for s in steps], prep)
    # Tessellation tolerance from the assembly diagonal (the preprocessing box, or the shapes' own box).
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
