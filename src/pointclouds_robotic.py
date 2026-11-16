"""Per-component point clouds for the grasp engine.

Inputs
    - The global-placement FreeCAD shape of each component (from `shapes.component_shapes`).
    - Sampling settings (`RobotConfig.pcd_spacing_mm`, `pcd_max_points`, `pcd_seed`) and an optional cache folder.
Outputs
    - One cloud per component, `(points (N, 3) mm, normals (N, 3))`, plus a cache key per component. These
      are the point clouds that go to the grasp engine (target = one component, obstacles = merged others).

What this module does
    Area-weighted random surface samples with the outward normal of the tessellation triangle they fall
    on (logic of gripper_test/make_pcds_freecad.py, numpy only). Requirements:

      - fixed seed and spacing (identical input -> identical cloud). Unlike the reference script, which
        draws every component from one shared RNG (so a cloud depended on the component order), each
        component gets its own RNG seeded by (seed, geometry hash): the cloud depends only on the shape,
        its placement and the config, which is also what makes it cacheable per component.
      - at least `MIN_POINTS` points (the reference minimum; the engine drops projected obstacle regions
        with <= 500 points, so obstacle clouds must stay well above that).
      - cached on disk per component, keyed by geometry hash (global vertices, so placement included),
        spacing, max points, seed and `CLOUD_VERSION`.
    `merge_clouds` and `rotate_cloud` build the obstacle cloud and the rotated obstacle cloud of a nut-partner step.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from shapes import shape_geometry_hash

log = logging.getLogger("aap.robot.pointclouds")

CLOUD_VERSION = "1"  # str. Part of the cloud cache key; bump when the sampling changes.
MIN_POINTS = 1000  # int. Minimum points per cloud, as gripper_test/make_pcds_freecad.py


def cloud_key(geometry_hash: str, spacing: float, max_points: int, seed: int) -> str:
    """[SUPP] Cache key of a cloud (also keys the grasp-engine results): geometry hash, spacing, max points, seed and version."""
    return hashlib.sha256(f"v{CLOUD_VERSION};geom:{geometry_hash};spacing:{float(spacing)!r};"
                          f"max:{int(max_points)};seed:{int(seed)}".encode()).hexdigest()


def sample_shape(shape, spacing: float, max_points: int, rng: np.random.Generator):
    """[MAIN] Area-weighted random surface samples with outward normals (from the tessellation winding).

    Inputs: a global-placement shape, the target spacing in mm (the point count is about area / spacing^2),
    the maximum number of points and the random generator. Output: (points (n, 3), normals (n, 3)), the
    point cloud the engine expects.

    The shape is tessellated finely (0.1 * spacing, at least 0.01 mm), triangles are drawn with probability
    proportional to their area, and a uniform random point is taken inside each drawn triangle.
    """
    # Tessellate and get the triangle normals (from the winding) and areas.
    deflection = max(0.1 * spacing, 0.01)
    verts, tris = shape.tessellate(deflection)
    V = np.array([[v.x, v.y, v.z] for v in verts], dtype=float)
    T = np.array(tris, dtype=int).reshape(-1, 3)
    if not len(T):
        raise ValueError("Shape has no triangles to sample")
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    cr = np.cross(b - a, c - a)
    norm = np.linalg.norm(cr, axis=1)
    area = 0.5 * norm
    n = cr / np.maximum(norm[:, None], 1e-12)
    total = float(area.sum())
    # Number of points: area / spacing^2, at least MIN_POINTS, at most max_points.
    count = int(min(max_points, max(MIN_POINTS, total / (spacing ** 2))))
    # Pick triangles by area, then a uniform point inside each (square-root barycentric sampling).
    idx = rng.choice(len(T), size=count, p=area / total)
    r1, r2 = rng.random(count), rng.random(count)
    s = np.sqrt(r1)
    P = (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]
    return P, n[idx]


def component_cloud(shape, spacing: float, max_points: int, seed: int, geometry_hash: str | None = None):
    """[AUX] (points, normals) of one global-placement shape; deterministic in (shape, spacing, max_points, seed).

    The RNG is seeded by (seed, first 15 hex digits of the geometry hash), so a component's cloud does not
    depend on the other components or on their order.
    """
    gh = geometry_hash or shape_geometry_hash(shape)
    rng = np.random.default_rng([int(seed), int(gh[:15], 16)])
    return sample_shape(shape, spacing, max_points, rng)


def component_clouds(component_ids: list[str], shapes: list, spacing: float = 0.3, max_points: int = 60000,
                     seed: int = 1, cache_dir: str | Path | None = None, labels: list[str] | None = None):
    """[MAIN] Clouds for every component, in the given order. Returns (clouds, keys).

    Inputs: component ids and their shapes (same order), spacing (default 0.3 mm), max points (60000),
    seed (1), an optional cache folder and labels for the log. Output: clouds[i] = (P, N) and keys[i] =
    its cache key (also used to key grasp-engine results). Each cloud is read from the cache if present,
    else sampled and saved; a cloud with <= 500 points is warned about, because the engine ignores
    obstacle regions that small.
    """
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    clouds, keys = [], []
    for i, (cid, shape) in enumerate(zip(component_ids, shapes)):
        gh = shape_geometry_hash(shape)
        key = cloud_key(gh, spacing, max_points, seed)
        path = None if cache is None else cache / f"{key[:32]}.npz"
        # Cache hit, else sample and save (temporary file, then rename).
        if path is not None and path.exists():
            d = np.load(path)
            P, N = d["points"], d["normals"]
        else:
            P, N = component_cloud(shape, spacing, max_points, seed, geometry_hash=gh)
            if path is not None:
                tmp = path.with_suffix(".tmp.npz")
                np.savez(tmp, points=P, normals=N)
                tmp.replace(path)
        name = labels[i] if labels else cid
        if len(P) <= 500:
            log.warning("Cloud of %r has only %d points; the engine ignores obstacle regions with <= 500 "
                        "points", name, len(P))
        log.info("Point cloud %r: %d points", name, len(P))
        clouds.append((P, N))
        keys.append(key)
    return clouds, keys


def merge_clouds(clouds: list[tuple[np.ndarray, np.ndarray]]):
    """[AUX] Concatenate clouds (the obstacle cloud of a step); (empty, empty) for no clouds."""
    if not clouds:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.vstack([c[0] for c in clouds]), np.vstack([c[1] for c in clouds])


def rotate_cloud(cloud: tuple[np.ndarray, np.ndarray], axis_point, axis_dir, angle_deg: float):
    """[AUX] Rotate points (about the line axis_point + t * axis_dir) and normals by angle_deg (right hand).

    Used for the nut-partner rotation check: the part turns by +angle about the nut axis, so in its own
    frame the obstacles are rotated by -angle (Rodrigues formula).
    """
    P, N = cloud
    k = np.asarray(axis_dir, float)
    k = k / np.linalg.norm(k)
    th = np.radians(angle_deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    p0 = np.asarray(axis_point, float)
    return (P - p0) @ R.T + p0, N @ R.T
