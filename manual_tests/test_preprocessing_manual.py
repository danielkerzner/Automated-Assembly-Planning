"""Manual smoke test for Preprocessing (src/preprocessing.py).

Not a pytest suite -- there's no hand-verified expected output for this
assembly yet, so this doesn't assert anything. It just runs
run_preprocessing() end to end and dumps everything it produced somewhere
you can inspect it:

  output/industrial_fastner/   (the default; the folder given as third argument otherwise)
    reduced_assembly.step   the standard-reduced assembly as its own STEP
                             file -- open it in FreeCAD/any STEP viewer and
                             confirm part4 (both instances) and part7 are
                             actually gone, nothing else is
    report.json              per-standard record (category, options[] each with
                             direction/exit_point/blockers, holds,
                             contact_neighbors, bbox, partner, nut_rule) plus
                             the full candidate direction list and the
                             whole-assembly bbox -- everything else
                             run_preprocessing() returns

Run (from repo root, environmentAAP active):

    python manual_tests/test_preprocessing_manual.py

or point it at a different STEP/BOM pair and output folder:

    python manual_tests/test_preprocessing_manual.py path/to.stp path/to/bom.csv tests/out_dir

Expected result: no error; the console lists every standard with its category (screw / nut / other), its removal
option(s) with blockers, holds, number of contact neighbours, partner and nut rule; then the size of the reduced
assembly, the number of unique candidate directions and the assembly bounding box. reduced_assembly.step contains
only the movable parts (standards removed) and report.json holds the same numbers as the console. Nothing is
compared with a reference: you check by eye that the standards found are the ones flagged in the BOM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import FreeCAD
import Import

from preprocessing import run_preprocessing

REPO_ROOT = Path(__file__).resolve().parent.parent

#DEFAULT_STEP = REPO_ROOT / "data" / "lunar_motor_tire" / "0013-A-0100-A01_3.stp"
#DEFAULT_BOM = REPO_ROOT / "data" / "lunar_motor_tire" / "generated_bom.csv"
#DEFAULT_OUT = REPO_ROOT / "output" / "lunar_motor_tire"

#DEFAULT_STEP = REPO_ROOT / "data" / "lunar_motor_tire" / "0013-A-0100-A01_3.stp"
#DEFAULT_BOM = REPO_ROOT / "data" / "lunar_motor_tire" / "generated_bom.csv"
#DEFAULT_OUT = REPO_ROOT / "output" / "lunar_motor_tire"
DEFAULT_STEP = REPO_ROOT / "data" / "industrial_fastner" / "industrial_fastner.stp"
DEFAULT_BOM = REPO_ROOT / "data" / "industrial_fastner" / "generated_bom.csv"
DEFAULT_OUT = REPO_ROOT / "output" / "industrial_fastner"



def _vec(v: FreeCAD.Vector) -> list[float]:
    return [v.x, v.y, v.z]


def main(step_path: Path, bom_path: Path, out_dir: Path) -> None:
    """Pipeline: run_preprocessing (STEP + BOM -> standards classified and removed from the assembly, candidate
    directions, bounding box) -> export the reduced assembly as STEP -> dump report.json -> print a summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_preprocessing(step_path, bom_path)
    reduced_objs = list(result.reduced_assembly.Objects)

    # 1. Export the reduced assembly so it can be opened and eyeballed directly.
    reduced_step = out_dir / "reduced_assembly.step"
    Import.export(reduced_objs, str(reduced_step))

    # 2. Dump everything else to a readable JSON report.
    report = {
        "reduced_assembly_component_count": len(reduced_objs),
        "reduced_assembly_components": [obj.Label for obj in reduced_objs],
        "standard_count": len(result.standard_records),
        "standard_records": {
            name: {
                "category": rec.category,
                "options": [
                    {"direction": _vec(opt.direction), "exit_point": _vec(opt.exit_point),
                     "blockers": sorted(opt.blockers)}
                    for opt in rec.options
                ],
                "holds": sorted(rec.holds),
                "contact_neighbors": sorted(rec.contact_neighbors),
                "bbox": [_vec(v) for v in rec.bbox] if rec.bbox else None,
                "partner": rec.partner,
                "nut_rule": rec.nut_rule,
            }
            for name, rec in result.standard_records.items()
        },
        "candidate_direction_count": len(result.candidate_directions),
        "candidate_directions": [_vec(d) for d in result.candidate_directions],
        "assembly_bbox": (
            [_vec(v) for v in result.assembly_bbox] if result.assembly_bbox else None
        ),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # 3. Short console summary.
    print(f"Standards processed: {len(result.standard_records)}")
    cat_name = {1: "screw", 2: "nut", 3: "other"}
    for name, rec in result.standard_records.items():
        bbox_str = f"{_vec(rec.bbox[0])} -> {_vec(rec.bbox[1])}" if rec.bbox else None
        opts_str = "; ".join(f"dir={_vec(opt.direction)} blockers={sorted(opt.blockers)}" for opt in rec.options)
        print(
            f"  {name} ({cat_name.get(rec.category, rec.category)}): {opts_str}, "
            f"holds={sorted(rec.holds)}, {len(rec.contact_neighbors)} contact neighbor(s), "
            f"partner={rec.partner}, nut_rule={rec.nut_rule}, bbox={bbox_str}"
        )
    print(
        f"Reduced assembly: {len(reduced_objs)} component(s) "
        f"-> {reduced_step}"
    )
    print(f"Candidate directions: {len(result.candidate_directions)} unique")
    if result.assembly_bbox:
        lo, hi = result.assembly_bbox
        print(f"Assembly bbox: {_vec(lo)} -> {_vec(hi)}")
    print(f"Full report written to {report_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        step_path, bom_path, out_dir = DEFAULT_STEP, DEFAULT_BOM, DEFAULT_OUT
    elif len(args) == 3:
        step_path, bom_path, out_dir = Path(args[0]), Path(args[1]), Path(args[2])
    else:
        print("usage: python test_preprocessing.py [step_path bom_path out_dir]")
        raise SystemExit(1)
    main(step_path, bom_path, out_dir)
