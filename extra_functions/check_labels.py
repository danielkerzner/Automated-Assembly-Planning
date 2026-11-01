"""Diagnostic: print the exact FreeCAD Label for every leaf component in a STEP
assembly, so you can confirm what belongs in bom.csv's `label` column --
in particular the auto-disambiguated label FreeCAD assigns to a duplicate
instance (e.g. the second occurrence of industrial_fastner_part4).

Uses load_assembly() / _leaf_components() from src/preprocessing.py directly,
so it imports the STEP file exactly the same way the real pipeline does.

Run with FreeCAD's own Python (the environmentAPP conda env), not a plain
Python -- FreeCAD/Part/Import must be importable:

    python extra_functions/check_labels.py data/industrial_fastner/industrial_fastner.stp

or, from the FreeCAD GUI's Python console:

    exec(open("extra_functions/check_labels.py").read().replace(
        "if __name__", "if True or __name__"))  # then call main("path/to.stp")
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import load_assembly, _leaf_components  # noqa: E402


def main(step_path: str) -> None:
    doc = load_assembly(step_path)
    components = _leaf_components(doc)
    print(f"{len(components)} leaf component(s) found in {step_path!r}:\n")
    for obj in components:
        print(f"  Name={obj.Name!r:20s}  Label={obj.Label!r}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python check_labels.py <step_file>")
        raise SystemExit(1)
    main(sys.argv[1])
