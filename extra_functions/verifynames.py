"""Diagnostic: print the whole FreeCAD object tree of one STEP file (Name, TypeId, Label, and the target of an App::Link).

Used to see how a STEP is structured (sub-assemblies, links, transparent groups) before writing its BOM or when
build_hierarchy rejects an App::Link. The STEP path is hard-coded below (lunar_motor_tire_complete); edit the
`load_assembly(...)` line to inspect another file. Run from the repo root in environmentAAP:

    python extra_functions/verifynames.py

Expected result: one indented line per object, children below their parent, e.g.
`'Part' [App::Part] label='Assembly'` followed by its leaves as `'Part__Feature' [Part::Feature] label='...'`;
a link object additionally prints `-> link to '<name>'`. Nothing is written to disk.
"""
import sys; sys.path.insert(0, "src")
import freecad  # bootstrap for the conda build
from preprocessing import load_assembly

def walk(obj, depth=0):
    linked = obj.getLinkedObject() if hasattr(obj, "getLinkedObject") else obj
    note = f" -> link to {linked.Name!r}" if linked is not obj else ""
    print("  " * depth + f"{obj.Name!r} [{obj.TypeId}] label={obj.Label!r}{note}")
    for child in getattr(obj, "Group", []):
        walk(child, depth + 1)

doc = load_assembly(r"data\lunar_motor_tire_complete\0013-A-0100-A01_3_complete.stp")
for obj in doc.RootObjects:
    walk(obj)

    