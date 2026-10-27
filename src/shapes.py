"""Shared FreeCAD shape access for the modules after Preprocessing (Optimization, Illustration Generator)."""

from __future__ import annotations


def component_shapes(component_ids, is_stander, prep) -> list:
    """Global shapes in the given order: movable parts from the reduced document, standers from the
    original document (run_preprocessing names the reduced doc '<doc>_reduced', the original must still be open)."""
    import FreeCAD

    reduced = prep.reduced_assembly
    original = None
    if reduced.Name.endswith("_reduced"):
        try:
            original = FreeCAD.getDocument(reduced.Name[: -len("_reduced")])
        except Exception:  # noqa: BLE001
            original = None
    shapes = []
    for cid, stander in zip(component_ids, is_stander):
        if not stander:
            obj = reduced.getObject(cid)
            shape = None if obj is None else obj.Shape  # stored at global placement
        else:
            obj = None if original is None else original.getObject(cid)
            if obj is not None:
                shape = obj.Shape.copy()
                shape.Placement = obj.getGlobalPlacement()
        if obj is None:
            raise ValueError(f"Component {cid} not found (stander shapes need the original document open)")
        shapes.append(shape)
    return shapes
