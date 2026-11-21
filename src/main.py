"""AAP pipeline entry point (command line).

Spec: CLAUDE.md (pipeline).

Inputs
    - STEP file of the assembly (may contain nested sub-assemblies, i.e. App::Part inside App::Part).
    - BOM CSV, one row per part or sub-assembly, matched on `label` (columns used: `is_standard`, category,
      `tool`, `group`, `processed`, `subassembly`).
    - An output directory.
Outputs
    - `<output_dir>/plan.json`: the combined plan: per level the
      status ("complete" or "locked"), the ordered steps with their direction, travel, cost and kind
      ("part", "standard", "subassembly"), and the nested plan of every expanded sub-assembly.
    - Per-level folders with the stage outputs: interference matrix, AoG, optimization result and, in
      manual mode, the illustrations and PDF manual (and in robot mode the gripper report under
      plan.json -> "robot").
    - Exit code 0 on success, 1 if the root level ended "locked" (manual mode).

What this module does
    It loads the STEP and BOM, builds the sub-assembly hierarchy (hierarchy.py), checks the level sizes
    and calls `levels.solve_level` for the root. That runs, once per level and depth-first:
    Preprocessing -> Interference Identification -> AoG -> Optimization (ACO) -> Illustration Generator.
    A flat assembly is just one level (the root). `main.py` then writes plan.json.
    Robot mode instead calls `robot_loop_robotic.run_robot_pipeline` (see robot_*_robotic.py).

How to run
    From the repository root, in the conda environment `environmentAAP` (it provides FreeCAD):

        python src/main.py <step_path> <bom_path> <output_dir> [options]

    Positional arguments (all required)
        step_path    STEP file of the assembly.
        bom_path     BOM CSV (see Inputs).
        output_dir   Folder for plan.json and the per-level outputs (created if missing).

    Options
        --step H                 Collision step size h in mm. Default: `CollisionConfig.step` (5.0,
                                 intentional). Calibrate per assembly with collision test E6.
        --force-interference     Recompute the interference matrix even if a matching cache exists.
        --gapskipping            Enable the interference sweep's distance-jump shortcut (gap_skip).
                                 Default: off (fixed-step common()-only sweep).
        --max-level-components N Size guard: abort before any geometry work if a processed level has more
                                 than N components. Default 20 (the AoG target size). Fix: set
                                 `processed = false` in the BOM for that sub-assembly.
        --mode {manual,robot}    manual = illustrated instructions for a human; robot = gripper and tool
                                 feasibility loop. Default: none; then you
                                 are asked on a terminal (Enter = manual), and without a terminal it is
                                 manual.
        --check-subassembly-grasp  Robot mode only (ignored with a warning in manual mode): an edge is
                                 feasible only if both sub-assemblies it joins (before and after the
                                 insertion) are graspable as one object. One extra grasp-engine call per
                                 distinct sub-assembly.

    Example
        python src/main.py data/assembly.step data/assembly_bom.csv output/assembly --mode manual --step 5

    Other settings (PreprocessingConfig, OptimizationConfig, IllustrationConfig, ...) are not on the
    command line; change them in code, or pass the config objects to `run_pipeline` from your own script.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from collision import CollisionConfig
from hierarchy import Hierarchy, build_hierarchy, check_level_sizes
from illustration_generator import IllustrationConfig
from levels import LevelPlan, solve_level
from optimization import OptimizationConfig
from preprocessing import PreprocessingConfig, load_assembly, load_bom, load_standard_categories

log = logging.getLogger("aap.main")
DEFAULT_MAX_LEVEL_COMPONENTS = 20  # default of --max-level-components (AoG target size)

def _read_bom_rows(bom_path: Path) -> list[dict]:
    """[AUX] Read the BOM CSV as a list of row dicts (utf-8 with optional BOM), for `build_hierarchy`."""
    with open(bom_path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _step_kind(hier: Hierarchy, container_id: str, step: dict) -> str:
    """[AUX] Kind of a step for plan.json: "standard", "subassembly" or "part".

    Why this order: a standard-flagged sub-assembly is structurally a container but is fused into one
    rigid component and treated like any other standard downstream, so it is never reported as
    "subassembly".
    """
    # is_standard wins over "is a child container": a standard-flagged sub-assembly is fused into one
    # rigid component and treated exactly like any other standard downstream -- it is never recursed
    # into, so it should not be reported as
    # "subassembly" even though it is structurally a container.
    if step["is_standard"]:
        return "standard"
    if step["component_id"] in hier.containers[container_id].child_ids:
        return "subassembly"
    return "part"


def _level_to_json(hier: Hierarchy, plan: LevelPlan) -> dict:
    """[MAIN] Convert one `LevelPlan` (and, recursively, its children) into the JSON dict of plan.json.

    Input: the hierarchy (for labels and members) and the level's plan. Output: a dict with the level's
    status, locked parts, total cost, inseparable pairs and steps. Each step carries its kind and, for a
    sub-assembly step, its members and its own nested plan under "sub_plan" (None for parts, and for
    fused standards and unprocessed containers).
    """
    if plan.status == "locked":
        # Phase 2: no closed group could be found for every
        # blocking dead end, or every remaining dead end is rule-induced -- no sequence, no
        # illustrations for this level.
        return {
            "container_id": plan.container_id,
            "label": plan.label,
            "status": "locked",
            "locked_parts": list(plan.locked_parts),
            "total_cost": None,
            "inseparable_pairs": [list(pair) for pair in plan.inseparable_pairs],
            "steps": [],
        }

    steps = []
    for step in plan.sequence.steps:
        cid = step["component_id"]
        kind = _step_kind(hier, plan.container_id, step)
        entry = dict(step)
        entry["kind"] = kind
        # A standard-flagged sub-assembly is still a real container: report its members/processed/auto
        # like "subassembly" would, just under kind "standard" and with no sub_plan (it is fused, never
        # recursed into).
        if cid in hier.containers[plan.container_id].child_ids:
            child = hier.containers[cid]
            entry["members"] = {
                "leaf_ids": list(child.all_leaf_ids),
                "labels": [hier.leaf_label[leaf_id] for leaf_id in child.all_leaf_ids],
            }
            entry["processed"] = child.processed
            entry["auto"] = child.auto
            entry["sub_plan"] = _level_to_json(hier, plan.children[cid]) if cid in plan.children else None
        else:
            entry["sub_plan"] = None
        steps.append(entry)
    return {
        "container_id": plan.container_id,
        "label": plan.label,
        "status": "complete",
        "locked_parts": [],
        "total_cost": plan.sequence.total_cost,
        "inseparable_pairs": [list(pair) for pair in plan.inseparable_pairs],
        "steps": steps,
    }


def build_plan(hier: Hierarchy, plan: LevelPlan, assembly_name: str) -> dict:
    """[AUX] plan.json contents: assembly name (root label, else the STEP stem) and the root level."""
    root_label = hier.containers[hier.root].label or assembly_name
    return {
        "assembly": root_label,
        "levels_independent": True,
        "level": _level_to_json(hier, plan),
    }


def run_pipeline(
    step_path: str | Path,
    bom_path: str | Path,
    output_dir: str | Path,
    preprocessing_config: PreprocessingConfig | None = None,
    collision_config: CollisionConfig | None = None,
    illustration_config: IllustrationConfig | None = None,
    optimization_config: OptimizationConfig | None = None,
    force_interference: bool = False,
    max_level_components: int = DEFAULT_MAX_LEVEL_COMPONENTS,
) -> dict:
    """[MAIN] Run the full manual-mode pipeline on one assembly and write plan.json.

    Inputs: STEP, BOM and output folder; optional config objects (None = each stage's defaults, which is
    what the command line uses except for the collision step); `force_interference` to bypass the
    interference cache; `max_level_components` for the size guard. Output: the plan.json dict (also
    written to `output_dir/plan.json`). A "locked" root level is a normal result, not an exception.

    Steps: load STEP and BOM, build the hierarchy and check sizes, solve the root level (which recurses
    into every processed sub-assembly), then convert the result to JSON.
    """
    step_path, bom_path, output_dir = Path(step_path), Path(bom_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the inputs: FreeCAD document, BOM labels, standard categories (screw / nut / other) and raw BOM rows.
    doc = load_assembly(step_path)
    bom_labels = load_bom(bom_path)
    bom_categories = load_standard_categories(bom_path)
    bom_rows = _read_bom_rows(bom_path)

    # Sub-assembly tree from the STEP, decided by the BOM `processed` column; fail early on oversized levels.
    hier = build_hierarchy(doc, bom_rows=bom_rows)
    check_level_sizes(hier, max_components=max_level_components)

    # Plan the root level; levels.solve_level runs the whole per-level pipeline and recurses depth-first.
    plan = solve_level(
        hier, hier.root, doc,
        step_path=step_path, bom_path=bom_path, bom_labels=bom_labels,
        output_dir=output_dir, bom_categories=bom_categories,
        preprocessing_config=preprocessing_config,
        collision_config=collision_config,
        illustration_config=illustration_config,
        optimization_config=optimization_config,
        force_interference=force_interference,
    )

    # Write the combined result.
    plan_dict = build_plan(hier, plan, assembly_name=step_path.stem)
    (output_dir / "plan.json").write_text(json.dumps(plan_dict, indent=1), encoding="utf-8")
    if plan.status == "locked":
        log.warning("Wrote plan.json to %s (root level LOCKED, stuck parts: %s)", output_dir,
                   plan.locked_parts)
    else:
        log.info("Wrote plan.json to %s (total cost %.4f)", output_dir, plan.sequence.total_cost)
    return plan_dict


def _parse_args() -> argparse.Namespace:
    """[AUX] Command-line arguments; the full list with defaults is in the module docstring ("How to run")."""
    parser = argparse.ArgumentParser(
        description="Run the AAP assembly sequence planning pipeline on a STEP assembly."
    )
    parser.add_argument("step_path", type=Path, help="STEP file for the assembly.")
    parser.add_argument("bom_path", type=Path, help="BOM CSV flagging standard components.")
    parser.add_argument(
        "output_dir", type=Path, help="Directory to write the plan and per-level output into.")
    parser.add_argument(
        "--step",
        type=float,
        default=CollisionConfig.step,
        help="Collision step size h in mm (calibrate per assembly with collision test E6).",
    )
    parser.add_argument(
        "--force-interference",
        action="store_true",
        help="Recompute the interference matrix even if a cached one matches.",
    )
    parser.add_argument(
        "--gapskipping",
        action="store_true",
        help="Enable the interference sweep's distance-jump shortcut (CollisionConfig.gap_skip). "
             "Default: off -- a fixed-step common()-only sweep measured faster overall (see CLAUDE.md).",
    )
    parser.add_argument(
        "--max-level-components",
        type=int,
        default=DEFAULT_MAX_LEVEL_COMPONENTS,
        help="Size guard: raise before any geometry work if a processed level has more components "
             "than this (AoG target size; default 20).",
    )
    parser.add_argument(
        "--mode",
        choices=("manual", "robot"),
        default=None,
        help="manual (illustrated human instructions) or robot (gripper/tool feasibility loop). "
             "If omitted you are asked; without a terminal, manual.",
    )
    parser.add_argument(
        "--check-subassembly-grasp",
        action="store_true",
        help="robot mode only: an edge is feasible only if both sub-assemblies it joins (before and after the "
             "insertion) are graspable as one object (one extra grasp-engine call per distinct sub-assembly).",
    )
    return parser.parse_args()


def resolve_mode(flag: str | None, ask=input, interactive: bool | None = None) -> str:
    """[AUX] Choose manual or robot mode: the --mode flag wins; else ask on a terminal (Enter = manual); no terminal = manual.

    `ask` and `interactive` are injectable so the prompt can be tested without a terminal.
    """
    if flag is not None:
        if flag not in ("manual", "robot"):
            raise ValueError(f"mode must be 'manual' or 'robot', got {flag!r}")
        return flag
    if interactive is None:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    if not interactive:
        log.info("No --mode given and no terminal to ask; using manual mode")
        return "manual"
    while True:
        answer = ask("Mode [manual/robot] (Enter = manual): ").strip().lower()
        if answer in ("", "manual", "m"):
            return "manual"
        if answer in ("robot", "r"):
            return "robot"
        print(f"Please answer 'manual' or 'robot' (got {answer!r}).", file=sys.stderr)


def _main_robot(args) -> int:
    """[MAIN] Robot-mode run: `run_robot_pipeline` plans, checks gripper/tool feasibility per step and re-plans around failures.

    The robot modules are imported here, so manual mode never loads them (they need the grasp engine
    set-up). If the chosen sequence still has infeasible steps they are
    listed on stderr (details in plan.json -> "robot"); the exit code stays 0 because the plan is written.
    """
    from robot_config_robotic import RobotConfig
    from robot_loop_robotic import run_robot_pipeline  # robot modules only load in robot mode

    plan_dict = run_robot_pipeline(
        args.step_path, args.bom_path, args.output_dir,
        robot_config=RobotConfig(check_subassembly_grasp=args.check_subassembly_grasp),
        collision_config=CollisionConfig(step=args.step, gap_skip=args.gapskipping),
        force_interference=args.force_interference,
        max_level_components=args.max_level_components,
    )
    robot = plan_dict["robot"]
    if not robot["feasible"]:
        print(f"Robot plan: the chosen sequence still has {len(robot['failing_steps'])} infeasible step(s): "
              + ", ".join(f"step {f['step']} {f['label']} ({f['reason']})" for f in robot["failing_steps"])
              + ". See plan.json -> robot.", file=sys.stderr)
    return 0


def main() -> int:
    """[MAIN] Command-line entry: parse arguments, pick the mode, run it and return the exit code (1 if the root level is locked)."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    if resolve_mode(args.mode) == "robot":
        return _main_robot(args)
    if args.check_subassembly_grasp:
        log.warning("--check-subassembly-grasp only applies to robot mode; ignored in manual mode")
    plan_dict = run_pipeline(
        args.step_path,
        args.bom_path,
        args.output_dir,
        collision_config=CollisionConfig(step=args.step, gap_skip=args.gapskipping),
        force_interference=args.force_interference,
        max_level_components=args.max_level_components,
    )
    if plan_dict["level"]["status"] == "locked":
        print(f"LOCKED: the root level could not be fully disassembled -- Phase 2 found no closed "
             f"group for the remaining dead end(s). Stuck parts: "
             f"{plan_dict['level']['locked_parts']}. See plan.json for details.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
