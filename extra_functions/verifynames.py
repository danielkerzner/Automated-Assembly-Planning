import sys; sys.path.insert(0, "src")
import freecad  # bootstrap for the conda build
from preprocessing import load_assembly

def walk(obj, depth=0):
    linked = obj.getLinkedObject() if hasattr(obj, "getLinkedObject") else obj
    note = f" -> link to {linked.Name!r}" if linked is not obj else ""
    print("  " * depth + f"{obj.Name!r} [{obj.TypeId}] label={obj.Label!r}{note}")
    for child in getattr(obj, "Group", []):
        walk(child, depth + 1)

doc = load_assembly(r"data\lunar_bodylager\0002-A-0900-A06-bodylager_1.stp")
for obj in doc.RootObjects:
    walk(obj)

    