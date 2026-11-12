"""Robot-mode output: the usual plan.json plus a `robot` block (docs/robotic-gripper_spec.md section 8).

The plan itself is written by `main.build_plan` exactly as in manual mode (same keys, same step records),
for the sequence the robot loop chose. The `robot` block is added next to it:

  robot.mode                 "robot"
  robot.feasible             the chosen sequence has no infeasible step
  robot.failing_steps        [{step, label, end_effector, reason, score_relative | contact_with}]
  robot.steps                per step of the chosen sequence (same `step` numbers as level.steps):
                               end_effector ("gripper" or the tool name / "tool (none)"), feasible, and
                               gripper: tcp, axes (X edge normal, Y edge direction, Z plane normal),
                                 grasp_width, contact_area, score, score_alone, score_relative,
                                 rotation_checked, angles, rotation (nut axis), reason,
                                 intrinsically_infeasible
                               tool: tool, checked, sense (direction index), sense_direction, contact,
                                 senses [{direction_index, direction, contact, contact_with}]
  robot.loop                 iterations, stop_reason, chosen_iteration, blocked_edges (with the iteration
                             that pruned each and why), sequences (every recorded sequence with its
                             per-step results and metrics; `chosen` marks the chosen one)
  robot.alone_scores         label -> S_alone (non-stander components)
  robot.intrinsically_infeasible  labels that fail even alone (never pruned; check by hand)
  robot.thresholds, robot.config, robot.engine
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("aap.robot.output")


def _state_labels(aog, state: int) -> list[str]:
    return [aog.component_labels[c] for c in range(aog.n_components) if (state >> c) & 1]


def step_json(s, aog) -> dict:
    """One StepEval as robot-block JSON."""
    base = {"step": s.step, "component_id": aog.component_ids[s.component], "component_label": s.label,
            "edge": s.edge, "state": s.state, "feasible": bool(s.feasible)}
    if s.kind == "gripper":
        g = s.grasp
        best = g.best
        base.update({
            "end_effector": "gripper",
            "tcp": None if best is None else [float(v) for v in best.tcp],
            "axes": None if best is None else [[float(v) for v in r] for r in best.axes],
            "grasp_width": None if best is None else float(best.grasp_width),
            "contact_area": None if best is None else float(best.contact_area),
            "score": g.score_context,
            "score_alone": g.score_alone,
            "score_relative": g.score_relative,
            "rotation_checked": g.rotation_checked,
            "angles": list(g.angles),
            "rotation": None if g.rotation is None else g.rotation.to_json(),
            "reason": g.reason,
            "intrinsically_infeasible": g.intrinsic,
        })
    else:
        t = s.tool
        tj = t.to_json(aog.directions)
        base.update({
            "end_effector": t.tool if t.tool else "tool (none)",
            "tool": t.tool,
            "checked": t.checked,
            "sense": t.sense,
            "sense_direction": tj["sense_direction"],
            "contact": None if not t.checked else (not t.feasible if t.sense is None else bool(t.senses.get(t.sense))),
            "senses": tj["senses"],
            "reason": None if t.feasible else "tool contact in every free sense",
        })
    return base


def failing_json(s) -> dict:
    out = {"step": s.step, "label": s.label, "end_effector": "gripper" if s.kind == "gripper" else (s.tool.tool or "tool")}
    if s.kind == "gripper":
        out.update({"reason": s.grasp.reason, "score": s.grasp.score_context,
                    "score_relative": s.grasp.score_relative, "intrinsically_infeasible": s.grasp.intrinsic})
    else:
        out.update({"reason": "tool contact in every free sense", "contact_with": s.tool.contact_with})
    return out


def robot_block(loop, aog, analysis, config) -> dict:
    chosen = loop.records[loop.chosen]
    steps = chosen["steps"]
    alone = analysis.alone_scores()
    intrinsic = sorted({s.label for r in loop.records for s in r["steps"] if s.intrinsic})
    return {
        "mode": "robot",
        "feasible": all(s.feasible for s in steps),
        "failing_steps": [failing_json(s) for s in steps if not s.feasible],
        "steps": [step_json(s, aog) for s in steps],
        "loop": {
            "iterations": loop.iterations,
            "stop_reason": loop.stop_reason,
            "chosen_iteration": chosen["iteration"],
            "blocked_edges": [{"state": st, "state_components": _state_labels(aog, st), "component": c,
                               "component_label": aog.component_labels[c], **info}
                              for (st, c), info in sorted(loop.blocked.items(), key=lambda kv: kv[1]["iteration"])],
            "sequences": [{"iteration": r["iteration"], "chosen": i == loop.chosen, "total_cost": r["total_cost"],
                           "n_infeasible": r["n_infeasible"], "min_relative": r["min_relative"],
                           "sum_relative": r["sum_relative"], "edge_path": r["edge_path"],
                           "order": [s.label for s in r["steps"]],
                           "pruned": [{"state": st, "component_label": aog.component_labels[c]} for st, c in r["pruned"]],
                           "steps": [step_json(s, aog) for s in r["steps"]]}
                          for i, r in enumerate(loop.records)],
            "wall_time_s": loop.wall_time_s,
        },
        "alone_scores": {aog.component_labels[c]: v for c, v in sorted(alone.items())},
        "intrinsically_infeasible": intrinsic,
        "thresholds": config.thresholds(),
        "config": config.to_json(),
        "engine": loop.extra.get("engine", {}),
    }


def write_robot_plan(hier, plan, loop, aog, analysis, config, output_dir: Path, assembly_name: str) -> dict:
    from main import build_plan

    plan_dict = build_plan(hier, plan, assembly_name=assembly_name)
    plan_dict["robot"] = robot_block(loop, aog, analysis, config)
    (Path(output_dir) / "plan.json").write_text(json.dumps(plan_dict, indent=1), encoding="utf-8")
    return plan_dict
