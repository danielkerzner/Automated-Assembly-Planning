"""AAP pipeline entry point.

Wires the full pipeline from CLAUDE.md, now sub-assembly aware (docs/subassembly-spec.md):

    Preprocessing -> Interference Identification -> AoG generator -> Optimization (ACO)
        -> Illustration Generator

The input STEP may contain nested sub-assemblies. `hierarchy.build_hierarchy` reads the BOM's
`processed` column to decide, per sub-assembly, whether to treat it as one rigid component or to
recurse into it; `levels.solve_level` then runs the pipeline above once per level (the whole
assembly, then every processed sub-assembly, depth-first), writing `plan.json` as the combined
result. A flat assembly (no nested sub-assemblies) is just one level, the root, so its output
layout and cache keys are unchanged from before this feature.

Written Instructions was dropped: the optimization JSON (steps + displacement vectors) already is
the textual result.
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
from preprocessing import PreprocessingConfig, load_assembly, load_bom, load_stander_categories

log = logging.getLogger("aap.main")
DEFAULT_MAX_LEVEL_COMPONENTS = 20

def _read_bom_rows(bom_path: Path) -> list[dict]:
    with open(bom_path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _step_kind(hier: Hierarchy, container_id: str, step: dict) -> str:
    # is_stander wins over "is a child container": a stander-flagged sub-assembly (docs/
    # subassembly-spec.md section 3) is fused into one rigid component and treated exactly like
    # any other stander downstream -- it is never recursed into, so it should not be reported as
    # "subassembly" even though it is structurally a container.
    if step["is_stander"]:
        return "stander"
    if step["component_id"] in hier.containers[container_id].child_ids:
        return "subassembly"
    return "part"


def _level_to_json(hier: Hierarchy, plan: LevelPlan) -> dict:
    if plan.status == "locked":
        # Phase 2 (docs/subassembly-spec.md section 10): no closed group could be found for every
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
        # A stander-flagged sub-assembly is still a real container (docs/subassembly-spec.md
        # section 3): report its members/processed/auto like "subassembly" would, just under
        # kind "stander" and with no sub_plan (it is fused, never recursed into).
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
    """plan.json contents (docs/subassembly-spec.md section 7)."""
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
    step_path, bom_path, output_dir = Path(step_path), Path(bom_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = load_assembly(step_path)
    bom_labels = load_bom(bom_path)
    bom_categories = load_stander_categories(bom_path)
    bom_rows = _read_bom_rows(bom_path)

    hier = build_hierarchy(doc, bom_rows=bom_rows)
    check_level_sizes(hier, max_components=max_level_components)

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

    plan_dict = build_plan(hier, plan, assembly_name=step_path.stem)
    (output_dir / "plan.json").write_text(json.dumps(plan_dict, indent=1), encoding="utf-8")
    if plan.status == "locked":
        log.warning("Wrote plan.json to %s (root level LOCKED, stuck parts: %s)", output_dir,
                   plan.locked_parts)
    else:
        log.info("Wrote plan.json to %s (total cost %.4f)", output_dir, plan.sequence.total_cost)
    return plan_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AAP assembly sequence planning pipeline on a STEP assembly."
    )
    parser.add_argument("step_path", type=Path, help="STEP file for the assembly.")
    parser.add_argument("bom_path", type=Path, help="BOM CSV flagging stander components.")
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
        help="manual (illustrated human instructions) or robot (gripper/tool feasibility loop, "
             "docs/robotic-gripper_spec.md). If omitted you are asked; without a terminal, manual.",
    )
    parser.add_argument(
        "--check-subassembly-grasp",
        action="store_true",
        help="robot mode only: an edge is feasible only if both sub-assemblies it joins (before and after the "
             "insertion) are graspable as one object (one extra grasp-engine call per distinct sub-assembly).",
    )
    return parser.parse_args()


def resolve_mode(flag: str | None, ask=input, interactive: bool | None = None) -> str:
    """The --mode flag wins; without it ask on a terminal (empty answer = manual); no terminal = manual."""
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
    from robot_config_robotic import RobotConfig
    from robot_loop_robotic import run_robot_pipeline  # robot modules only load in robot mode

    plan_dict = run_robot_pipeline(
        args.step_path, args.bom_path, args.output_dir,
        robot_config=RobotConfig(check_subassembly_grasp=args.check_subassembly_grasp),
        collision_config=CollisionConfig(step=args.step),
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
        collision_config=CollisionConfig(step=args.step),
        force_interference=args.force_interference,
        max_level_components=args.max_level_components,
    )
    if plan_dict["level"]["status"] == "locked":
        print(f"LOCKED: the root level could not be fully disassembled -- Phase 2 found no closed "
             f"group for the remaining dead end(s). Stuck parts: "
             f"{plan_dict['level']['locked_parts']}. See plan.json for details "
             f"(docs/subassembly-spec.md section 10).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
