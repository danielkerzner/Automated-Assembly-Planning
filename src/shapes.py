"""Shared FreeCAD shape access for the stages that run after Preprocessing.

Used by: Optimization, Illustration Generator and Levels (per-level orchestration).

Inputs
    - A `PreprocessingResult` (`prep`): its `reduced_assembly` is the reduced FreeCAD document that holds the
      movable parts at their global placement; the original document must still be open for standards.
    - A list of component ids and a parallel list of standard flags.
Outputs
    - `component_shapes`: one FreeCAD `Part.Shape` per component id, in the same order, all at global placement.
    - `shape_geometry_hash`: a hex string that identifies a shape's geometry across sessions (cache keys).

What this module does
    The later stages need the final geometry of every component, but movable parts and standards live in
    different documents (standards are removed from the reduced document by Preprocessing). This module hides
    that split and provides a session-independent geometry hash for caches.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def shape_geometry_hash(shape) -> str:
    """[AUX] Session-independent hash of a FreeCAD shape, used as (part of) a cache key.

    It hashes the solid/edge counts, the sorted face surface types and the rounded, sorted global B-rep
    vertices. Volume, Area and BoundBox are left out on purpose: once a shape carries a triangulation
    (after tessellate), OCC evaluates them on the mesh and they change in the last digits, which would
    make the key unstable.
    """
    # Vertices rounded to 6 decimals and sorted, so ordering and float noise do not change the hash.
    pts = np.array([[v.X, v.Y, v.Z] for v in shape.Vertexes], dtype=float).reshape(-1, 3).round(6) + 0.0
    pts = pts[np.lexsort(pts.T[::-1])] if len(pts) else pts
    # Topology summary: number of solids, number of edges, and the sorted face surface types.
    head = [len(shape.Solids), len(shape.Edges), sorted(f.Surface.TypeId for f in shape.Faces)]
    h = hashlib.sha256(json.dumps(head).encode())
    h.update(pts.tobytes())
    return h.hexdigest()


def component_shapes(component_ids, is_standard, prep) -> list:
    """[AUX] Return the global shape of each component, in the order given.

    Movable parts are read from the reduced document (already stored at global placement). Standards were
    removed from the reduced document by Preprocessing, so they are read from the original document
    (run_preprocessing names the reduced doc '<doc>_reduced'; the original must still be open) and placed
    with their global placement. Raises ValueError if a component cannot be found.
    """
    import FreeCAD

    reduced = prep.reduced_assembly
    # Find the original document from the '<doc>_reduced' naming convention; None if it is not open.
    original = None
    if reduced.Name.endswith("_reduced"):
        try:
            original = FreeCAD.getDocument(reduced.Name[: -len("_reduced")])
        except Exception:  # noqa: BLE001
            original = None
    shapes = []
    for cid, standard in zip(component_ids, is_standard):
        if not standard:
            obj = reduced.getObject(cid)
            shape = None if obj is None else obj.Shape  # stored at global placement
        else:
            obj = None if original is None else original.getObject(cid)
            if obj is not None:
                # Copy before setting the placement so the original document is not modified.
                shape = obj.Shape.copy()
                shape.Placement = obj.getGlobalPlacement()
        if obj is None:
            raise ValueError(f"Component {cid} not found (standard shapes need the original document open)")
        shapes.append(shape)
    return shapes
