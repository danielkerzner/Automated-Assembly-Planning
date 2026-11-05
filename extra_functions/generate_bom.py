"""Generate a starter bom.csv listing every leaf component's Label, plus every subassembly.

Writes `label,is_stander,screw,nut,pitch,tool,group,processed,subassembly` (one row per leaf
Part::Feature, same filter Preprocessing uses, plus one row per App::Part subassembly).
`is_stander`, `screw`, `nut`, `pitch` and `tool` are filled by hand (`screw`/`nut` pick a
stander's category, docs/preprocessing-spec.md P1; `pitch` is mm of axial travel per full turn,
read only for screw/nut rows, docs/stander_processing-spec.md section 12 -- blank means no turn
count is shown for that stander). See data/industrial_fastner/generated_bom.csv for the columns
the pipeline actually reads.

The assembly tree may contain nested subassemblies (App::Part). The very first one (the root
object representing the whole assembly) is always expanded. Every other App::Part encountered
while walking the tree prompts the user: the answer is recorded in `processed`
(true/false), and if declined, none of the parts inside that subassembly are listed (its own
row still is, so the decision is visible in the BOM). `subassembly` traces each row back to
the Label of its nearest enclosing App::Part (blank at the document root).

Usage: python extra_functions/generate_bom.py <step_path> <output_csv>
"""

import csv
import sys

sys.path.insert(0, "src")
import freecad  # bootstrap for the conda build
from preprocessing import load_assembly

FIELDNAMES = ["label", "is_stander", "screw", "nut", "pitch", "tool", "group", "processed", "subassembly"]


def is_leaf_component(obj) -> bool:
    """True for a leaf Part::Feature with a non-null shape and at least one solid."""
    return (
        obj.TypeId == "Part::Feature"
        and getattr(obj, "Shape", None) is not None
        and not obj.Shape.isNull()
        and len(obj.Shape.Solids) > 0
    )


def ask_include_subassembly(obj) -> bool:
    """Prompt whether to list the parts inside a nested subassembly."""
    while True:
        answer = input(
            f"Subassembly {obj.Label!r} [{obj.Name}] found - list its parts in the BOM? [y/n]: "
        ).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def _row(label: str, subassembly: str, processed: str = "") -> dict[str, str]:
    return {
        "label": label,
        "is_stander": False,
        "screw": False,
        "nut": False,
        "pitch": "",
        "tool": "By_hand",
        "group": label,
        "processed": processed,
        "subassembly": subassembly,
    }


def collect_rows(obj, depth: int = 0, parent_label: str = "") -> list[dict[str, str]]:
    """Walk the tree, collecting one BOM row per leaf component and per subassembly.

    `parent_label` is the Label of the nearest enclosing App::Part (empty at the document
    root). Asks before descending into a nested App::Part; the root App::Part is exempt.
    """
    if is_leaf_component(obj):
        return [_row(obj.Label, parent_label)]

    if obj.TypeId != "App::Part":
        rows: list[dict[str, str]] = []
        for child in getattr(obj, "Group", []):
            rows.extend(collect_rows(child, depth, parent_label))
        return rows

    # obj is a subassembly (App::Part).
    included = depth == 0 or ask_include_subassembly(obj)
    rows = [_row(obj.Label, parent_label, processed="true" if included else "false")]
    if not included:
        return rows
    for child in getattr(obj, "Group", []):
        rows.extend(collect_rows(child, depth + 1, obj.Label))
    return rows


def generate_bom_rows(doc) -> list[dict[str, str]]:
    """BOM rows for every leaf component and subassembly reachable from the document root."""
    rows: list[dict[str, str]] = []
    for obj in doc.RootObjects:
        rows.extend(collect_rows(obj, depth=0))
    return rows


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python extra_functions/generate_bom.py <step_path> <output_csv>")
        raise SystemExit(1)
    step_path, output_csv = sys.argv[1], sys.argv[2]

    doc = load_assembly(step_path)
    rows = generate_bom_rows(doc)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
