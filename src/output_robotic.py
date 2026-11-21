"""Robot-mode output: the usual plan.json plus a `robot` block.

Inputs
    - The hierarchy and `LevelPlan` of the chosen sequence, the `RobotLoopResult` (every sequence tried,
      blocked edges), the AoG, the `GripperAnalysis` (scores) and the `RobotConfig`.
Outputs
    - `<output_dir>/plan.json`: the manual-mode plan plus the `robot` block described below. Pure
      formatting: no decision is made here.

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
  robot.alone_scores         label -> S_alone (non-standard components)
  robot.subassembly_scores   optional sub-assembly check (RobotConfig.check_subassembly_grasp), else null:
                             state mask -> {score, feasible, n_components, components, best}. Each step then
                             also has part_feasible, subassembly_after, subassembly_before, and `feasible`
                             is the edge's (part and both nodes)
  robot.intrinsically_infeasible  labels that fail even alone (never pruned; check by hand)
  robot.thresholds, robot.config, robot.engine
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("aap.robot.output")


def _state_labels(aog, state: int) -> list[str]:
    """[AUX] Labels of the components present in an AoG state bitmask (for readable blocked edges and node scores)."""
    return [aog.component_labels[c] for c in range(aog.n_components) if (state >> c) & 1]


def step_json(s, aog) -> dict:
    """[MAIN] One `StepEval` as robot-block JSON (the per-step record of plan.json -> robot).

    A gripper step reports the best grasp (TCP, frame, width, contact area), the scores and the rotation
    check; a tool step reports the tool, the sense used and the contact in every free sense. The optional
    sub-assembly check adds its two nodes and, if only they fail, their reason.
    """
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
    if s.node_after is not None:  # optional sub-assembly check
        base["part_feasible"] = s.part_feasible
        base["subassembly_after"] = s.node_after.to_json()
        base["subassembly_before"] = s.node_before.to_json()
        if s.part_feasible and s.failing_nodes:
            base["reason"] = "; ".join(n.reason for n in s.failing_nodes)
    return base


def failing_json(s) -> dict:
    """[AUX] Short record of an infeasible step (end effector, reason, score or contact) for `robot.failing_steps`."""
    out = {"step": s.step, "label": s.label, "end_effector": "gripper" if s.kind == "gripper" else (s.tool.tool or "tool")}
    if s.kind == "gripper":
        out.update({"reason": s.grasp.reason, "score": s.grasp.score_context,
                    "score_relative": s.grasp.score_relative, "intrinsically_infeasible": s.grasp.intrinsic})
    else:
        out.update({"reason": None if s.tool.feasible else "tool contact in every free sense",
                    "contact_with": s.tool.contact_with})
    nodes = s.failing_nodes
    if nodes:  # optional sub-assembly check
        node_reason = "; ".join(n.reason for n in nodes)
        out["reason"] = node_reason if s.part_feasible else f"{out['reason']}; {node_reason}"
        out["subassembly_not_graspable"] = [{"state": n.state, "n_components": n.n_components, "score": n.score,
                                             "full_assembly": n.state == s.root_state} for n in nodes]
    return out


def robot_block(loop, aog, analysis, config) -> dict:
    """[MAIN] Build the whole `robot` block from the loop result: the chosen sequence's steps and failures, the loop history, scores, thresholds, config and engine info."""
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
        # optional sub-assembly check: every evaluated node, keyed by its state mask (as a string, for JSON)
        "subassembly_scores": ({str(st): {**n.to_json(), "components": _state_labels(aog, st)}
                                for st, n in sorted(analysis.node_scores().items())}
                               if config.check_subassembly_grasp else None),
        "intrinsically_infeasible": intrinsic,
        "thresholds": config.thresholds(),
        "config": config.to_json(),
        "engine": loop.extra.get("engine", {}),
    }


def write_robot_plan(hier, plan, loop, aog, analysis, config, output_dir: Path, assembly_name: str) -> dict:
    """[AUX] Write plan.json: the manual-mode plan (`main.build_plan`) of the chosen sequence plus the `robot` block; returns the dict."""
    from main import build_plan

    plan_dict = build_plan(hier, plan, assembly_name=assembly_name)
    plan_dict["robot"] = robot_block(loop, aog, analysis, config)
    (Path(output_dir) / "plan.json").write_text(json.dumps(plan_dict, indent=1), encoding="utf-8")
    return plan_dict
