"""Gripper analysis for robot mode. Pure numpy, no FreeCAD.

Inputs
    - The per-component point clouds and their cache keys (`pointclouds_robotic.component_clouds`).
    - A grasp engine object with `evaluate(target_cloud, obstacle_cloud) -> EngineResult`
      (`gripper_engine_robotic.GraspEngine`; any engine with that signature works, see the ENGINE SWAP
      POINT block in that module).
    - A `RobotConfig` (thresholds, rotation step) and an optional cache folder.
Outputs
    - `StepGrasp` per (AoG state, component): scores, feasible flag, reason, best grasp.
    - `NodeGrasp` per AoG state for the optional sub-assembly check.

What this module does
    For a step "remove component i from state S" (S = bitmask of present components): target = cloud of i
    in place, obstacles = merged clouds of every other component present in S (standards included: they are
    physical obstacles for the gripper). The check is static (grasp with i still inserted). One score per
    (state, component), shared by every free direction of the edge.

      S_alone    engine score of i with no obstacles, once per component.
      S_context  engine score of i in state S.
      S_rel      min(1, S_context / S_alone); S_alone = 0 -> infeasible.
      feasible   S_rel >= grasp_rel_threshold and S_context >= grasp_abs_floor (boundary per config).

    Intrinsically infeasible components (implementation decision): if
    S_alone itself already fails the rule (S_alone = 0, or S_alone below the absolute floor), no state can
    pass (obstacles only remove candidates), so no engine call is made for any state and every
    step of that component is reported infeasible with `intrinsic = True`. The loop never prunes such an
    edge and leaves it out of the quarter: the component is infeasible in every state, so pruning it would
    only disconnect the AoG without improving anything. They are reported for the user to check by hand.

    Rotation: when the removed component is the partner of a nut that is still present, the
    component (and the gripper with it) turns about the nut axis. The target and grasp frame stay fixed and
    the obstacle cloud is rotated by the opposite angle, at every angle of 0..360 in steps of
    `grasp_rotation_step_deg`. A grasp survives if a matching grasp (TCP within `grasp_match_tol_mm`, each
    axis within `grasp_match_angle_deg`, signed) is feasible at every angle; its score is the lowest of its
    matched scores (conservative), and the step score is the best surviving score (0 if none). Angle 0 is
    the static call and is shared with the non-rotating cache entry.

    Caching: every engine call is memoised in memory, and optionally on disk under `cache_dir`,
    keyed by the cloud keys of target and obstacles, the rotation, and an engine tag (gripper YAML, engine
    copy version and source hash, seed, runner mode).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gripper_engine_robotic import EngineResult, Grasp
from pointclouds_robotic import merge_clouds, rotate_cloud
from robot_config_robotic import RobotConfig

log = logging.getLogger("aap.robot.gripper")


# ---------------------------------------------------------------------------
# Scores and rule
# ---------------------------------------------------------------------------


def relative_score(score_context: float, score_alone: float) -> float:
    """[AUX] S_rel = min(1, S_context / S_alone); 0 when the component has no grasp at all (S_alone <= 0)."""
    if score_alone <= 0:
        return 0.0
    return float(min(1.0, score_context / score_alone))


def _ge(a: float, b: float, inclusive: bool) -> bool:
    """[AUX] a >= b if `inclusive` (RobotConfig.grasp_boundary_inclusive), else a > b."""
    return a >= b if inclusive else a > b


def feasibility(score_context: float, score_alone: float, config: RobotConfig) -> tuple[bool, str | None]:
    """[MAIN] The feasibility rule: (feasible, reason if not).

    Inputs: S_context, S_alone and the config thresholds. Feasible iff S_rel >= `grasp_rel_threshold` AND
    S_context >= `grasp_abs_floor`. The relative test says the surroundings cost too much of the free-space
    grasp quality; the absolute floor says the grasp is poor even when little is lost. The reason string
    is what plan.json reports for a failing step.
    """
    if score_alone <= 0:
        return False, "no grasp even alone (S_alone = 0)"
    rel = relative_score(score_context, score_alone)
    if not _ge(rel, config.grasp_rel_threshold, config.grasp_boundary_inclusive):
        return False, f"relative score {rel:.3f} below {config.grasp_rel_threshold}"
    if not _ge(score_context, config.grasp_abs_floor, config.grasp_boundary_inclusive):
        return False, f"score {score_context:.3f} below absolute floor {config.grasp_abs_floor}"
    return True, None


def intrinsically_infeasible(score_alone: float, config: RobotConfig) -> bool:
    """[AUX] S_alone already fails the rule, so every state fails (S_context <= S_alone); no engine call is needed for it."""
    return score_alone <= 0 or not _ge(score_alone, config.grasp_abs_floor, config.grasp_boundary_inclusive)


# ---------------------------------------------------------------------------
# Rotation matching
# ---------------------------------------------------------------------------


def _match(g: Grasp, candidates: list[Grasp], tol_mm: float, cos_tol: float) -> Grasp | None:
    """[AUX] Best-scoring grasp in candidates with the same TCP (within tol) and the same three axes (each within the angle)."""
    best = None
    for h in candidates:
        if np.linalg.norm(h.tcp - g.tcp) > tol_mm:
            continue
        if np.any(np.einsum("ij,ij->i", g.axes, h.axes) < cos_tol):
            continue
        if best is None or h.score > best.score:
            best = h
    return best


def surviving_grasps(per_angle: list[EngineResult], tol_mm: float, angle_deg: float) -> list[Grasp]:
    """[MAIN] Grasps that exist at every rotation angle, each scored by its weakest angle.

    Input: one `EngineResult` per angle (the first is angle 0) and the matching tolerances (TCP distance in
    mm, axis angle in degrees). Output: the grasps of the first angle that have a match at every other
    angle, with score = the lowest matched score (conservative, so a grasp is only as good as its worst
    angle). The caller takes the best of them as the step score.
    """
    if not per_angle:
        return []
    cos_tol = float(np.cos(np.radians(angle_deg)))
    out = []
    for g in per_angle[0].grasps:
        score = g.score
        ok = True
        for res in per_angle[1:]:
            h = _match(g, res.grasps, tol_mm, cos_tol)
            if h is None:
                ok = False
                break
            score = min(score, h.score)
        if ok:
            out.append(Grasp(tcp=g.tcp, axes=g.axes, contact_area=g.contact_area, grasp_width=g.grasp_width,
                             score=score, pair=g.pair, edge=g.edge, point=g.point))
    return out


# ---------------------------------------------------------------------------
# Step results
# ---------------------------------------------------------------------------


@dataclass
class Rotation:
    """The nut axis a rotating partner turns about. Input of `GripperAnalysis.step`; built by `robot_loop_robotic.rotation_for`."""

    axis_point: tuple[float, float, float]  # (3,) mm. A point on the nut axis (the nut's bbox centre).
    axis_dir: tuple[float, float, float]  # (3,) unit. Direction of the nut axis (the nut's removal direction).
    nut_id: str  # str. Component id of the nut (for the report).
    nut_label: str  # str. BOM label of the nut (for the report and warnings).

    def key(self) -> str:
        """[SUPP] Cache key part of the axis (rounded to 6 decimals)."""
        return json.dumps([np.round(self.axis_point, 6).tolist(), np.round(self.axis_dir, 6).tolist()])

    def to_json(self) -> dict:
        """[SUPP] JSON-ready dict for plan.json."""
        return {"axis_point": [float(v) for v in self.axis_point], "axis_dir": [float(v) for v in self.axis_dir],
                "nut_id": self.nut_id, "nut_label": self.nut_label}


@dataclass
class StepGrasp:
    """Grasp result of one step (output of `GripperAnalysis.step`, held by `robot_loop_robotic.StepEval`)."""

    state: int  # int bitmask. Components present before the removal (AoG state).
    component: int  # int. AoG component index of the removed part.
    label: str  # str. Its BOM label.
    score_alone: float  # float 0..1. S_alone: engine score of the part with no obstacles.
    score_context: float | None  # float 0..1, None when not evaluated (intrinsically infeasible component). S_context.
    score_relative: float | None  # float 0..1 or None. S_rel = min(1, S_context / S_alone).
    feasible: bool  # bool. The feasibility rule's verdict.
    reason: str | None  # str or None. Why it is infeasible.
    intrinsic: bool  # bool. Fails even alone; never pruned (see module docstring).
    best: Grasp | None  # Grasp or None. The best surviving grasp (its TCP and frame go into plan.json).
    rotation_checked: bool = False  # bool. True if the rotation check was applied.
    angles: list[float] = field(default_factory=list)  # list of deg. The angles evaluated (empty without rotation).
    rotation: Rotation | None = None  # Rotation or None. The nut axis used.
    n_grasps: int = 0  # int. Number of (surviving) grasps found.
    engine_calls: int = 0  # int. New engine calls this evaluation cost (0 = all cached).

    def to_json(self) -> dict:
        """[SUPP] JSON-ready dict for plan.json."""
        return {"state": self.state, "component": self.component, "label": self.label,
                "score_alone": self.score_alone, "score": self.score_context,
                "score_relative": self.score_relative, "feasible": self.feasible, "reason": self.reason,
                "intrinsically_infeasible": self.intrinsic,
                "best": None if self.best is None else self.best.to_json(),
                "rotation_checked": self.rotation_checked, "angles": self.angles,
                "rotation": None if self.rotation is None else self.rotation.to_json(), "n_grasps": self.n_grasps}


@dataclass
class NodeGrasp:
    """Optional sub-assembly check: grasp score of a whole AoG state grasped as one object, nothing around."""

    state: int  # int bitmask. The AoG state (the sub-assembly) grasped as one object.
    n_components: int  # int. Number of parts in it.
    score: float | None  # float 0..1, None for the empty state. Engine score of the merged cloud.
    feasible: bool  # bool. score >= `RobotConfig.subassembly_floor()` (the empty state is trivially feasible).
    best: Grasp | None  # Grasp or None. Best grasp found.
    reason: str | None  # str or None. Why it is not graspable.

    def to_json(self) -> dict:
        """[SUPP] JSON-ready dict for plan.json."""
        return {"state": self.state, "n_components": self.n_components, "score": self.score,
                "feasible": self.feasible, "reason": self.reason,
                "best": None if self.best is None else self.best.to_json()}


class GripperAnalysis:
    """Scores per (state, component) with caching. `clouds[c]`/`cloud_keys[c]` by AoG component index.

    The only object that talks to the grasp engine: it builds the target and obstacle point clouds, calls
    `engine.evaluate` and turns the scores into `StepGrasp` / `NodeGrasp` results.
    """

    def __init__(self, engine, clouds: list, cloud_keys: list[str], labels: list[str], config: RobotConfig,
                 cache_dir: str | Path | None = None, engine_tag: str = ""):
        """[AUX] Store the engine, clouds (by AoG component index), their cache keys, labels and config; `cache_dir` None = memory cache only.

        `engine_tag` identifies the engine setup (YAML, engine version, seed, runner) inside the disk cache key.
        """
        self.engine, self.clouds, self.cloud_keys, self.labels = engine, clouds, cloud_keys, labels
        self.config, self.engine_tag = config, engine_tag
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._calls: dict[str, EngineResult] = {}
        self._alone: dict[int, float] = {}
        self._steps: dict[tuple[int, int, bool], StepGrasp] = {}
        self._nodes: dict[int, NodeGrasp] = {}
        self.n_engine_calls = 0  # calls actually sent to the engine (not served from a cache)
        self.n_disk_hits = 0

    # -- engine calls with cache --
    # `target` is one component index, or a tuple of indices for a sub-assembly grasped as one object (node check);
    # a one-element tuple is the same call as that component alone (same key, shared cache entry).
    def _target_key(self, target) -> str:
        """[SUPP] Cache key of a target (a component, or a merged sub-assembly)."""
        ids = (target,) if isinstance(target, int) else tuple(sorted(target))
        if len(ids) == 1:
            return self.cloud_keys[ids[0]]
        return "node:" + hashlib.sha256(",".join(sorted(self.cloud_keys[i] for i in ids)).encode()).hexdigest()

    def _target_cloud(self, target):
        """[AUX] Point cloud of a target: the component's cloud, or the merged cloud of a sub-assembly."""
        if isinstance(target, int):
            return self.clouds[target]
        return merge_clouds([self.clouds[i] for i in target])

    def _target_label(self, target) -> str:
        """[SUPP] Label of a target for the log."""
        if isinstance(target, int):
            return self.labels[target]
        return self.labels[target[0]] if len(target) == 1 else f"sub-assembly of {len(target)} parts"

    def _call_key(self, target, obstacles: tuple[int, ...], rotation: Rotation | None, angle: float) -> str:
        """[SUPP] Cache key of one engine call: engine tag, target, obstacle set and rotation (angle 0 = no rotation)."""
        rot = "none" if rotation is None or angle % 360 == 0 else f"{rotation.key()}@{angle % 360:.6f}"
        h = hashlib.sha256(f"engine:{self.engine_tag};target:{self._target_key(target)};".encode())
        h.update(("obstacles:" + ",".join(sorted(self.cloud_keys[o] for o in obstacles)) + f";rot:{rot}").encode())
        return h.hexdigest()

    def _call(self, target, obstacles: tuple[int, ...], rotation: Rotation | None = None,
              angle: float = 0.0) -> EngineResult:
        """[MAIN] One cached engine evaluation: build the clouds, call `engine.evaluate`, return the `EngineResult`.

        Inputs: the target (component index or tuple), the obstacle component indices, and optionally a
        rotation and angle. The memory cache, then the disk cache, are tried first. On a miss the obstacle
        clouds are merged (and rotated by -angle about the nut axis, since the part turns by +angle) and the
        engine is called; the result is stored in both caches. Every engine call of the robot mode passes here.
        """
        key = self._call_key(target, obstacles, rotation, angle)
        if key in self._calls:
            return self._calls[key]
        path = None if self.cache_dir is None else self.cache_dir / f"{key[:40]}.json"
        if path is not None and path.exists():
            try:
                res = EngineResult.from_json(json.loads(path.read_text(encoding="utf-8")))
                self._calls[key] = res
                self.n_disk_hits += 1
                return res
            except Exception as exc:  # noqa: BLE001
                log.warning("Unreadable engine cache file %s (%s); recomputing", path.name, exc)
        # Cache miss: obstacle point cloud (rotated if needed) and the engine call.
        obst = merge_clouds([self.clouds[o] for o in obstacles]) if obstacles else None
        if obst is not None and rotation is not None and angle % 360 != 0:
            # The part turns by +angle about the nut axis; in its own frame the rest turns by -angle.
            obst = rotate_cloud(obst, rotation.axis_point, rotation.axis_dir, -angle)
        res = self.engine.evaluate(self._target_cloud(target), obst)
        self.n_engine_calls += 1
        log.info("Engine: %s with %d obstacle(s)%s -> %d grasps, best %.3f (%.1f s)", self._target_label(target),
                 len(obstacles), "" if rotation is None or angle % 360 == 0 else f", rotated {angle:g} deg",
                 len(res.grasps), res.score, res.wall_time_s)
        self._calls[key] = res
        if path is not None:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(res.to_json()), encoding="utf-8")
            tmp.replace(path)
        return res

    # -- public --
    def alone(self, c: int) -> float:
        """[AUX] S_alone of component c (engine score with no obstacles), computed once."""
        if c not in self._alone:
            self._alone[c] = self._call(c, ()).score
        return self._alone[c]

    def alone_scores(self) -> dict[int, float]:
        """[SUPP] All S_alone values computed so far (component index -> score), for the report."""
        return dict(self._alone)

    def node(self, state: int) -> "NodeGrasp":
        """[MAIN] Is the sub-assembly `state` (all its components as one object, nothing around) graspable?

        Optional sub-assembly check (RobotConfig.check_subassembly_grasp). Input: an AoG state bitmask.
        Output: a `NodeGrasp`; the merged cloud of its components is the target with no obstacles, and it
        passes if the score reaches `RobotConfig.subassembly_floor()`. The empty state is trivially fine.
        """
        state = int(state)
        if state in self._nodes:
            return self._nodes[state]
        ids = tuple(i for i in range(len(self.clouds)) if (state >> i) & 1)
        floor = self.config.subassembly_floor()
        if not ids:
            out = NodeGrasp(state=state, n_components=0, score=None, feasible=True, best=None, reason=None)
        else:
            res = self._call(ids, ())
            s = res.score
            ok = _ge(s, floor, self.config.grasp_boundary_inclusive)
            out = NodeGrasp(state=state, n_components=len(ids), score=s, feasible=ok, best=res.best,
                            reason=None if ok else f"sub-assembly score {s:.3f} below {floor}")
        self._nodes[state] = out
        return out

    def node_scores(self) -> dict[int, "NodeGrasp"]:
        """[SUPP] All node checks computed so far (state -> NodeGrasp), for the report."""
        return dict(self._nodes)

    def step(self, state: int, c: int, rotation: Rotation | None = None) -> StepGrasp:
        """[MAIN] Grasp feasibility of removing component c from `state`: the core of the gripper analysis.

        Inputs: the AoG state bitmask, the component index and, if c is the partner of a present nut, the
        `Rotation` to check. Output: a `StepGrasp`. Results are cached per (state, c, rotating?), so a step
        repeated in a later loop iteration costs no engine call.

        Steps: get S_alone; if it already fails the rule the step is intrinsically infeasible and no further
        call is made. Otherwise the obstacles are every other component present in `state`, the engine is
        called (once, or at every rotation angle with `surviving_grasps`), the best score is S_context, and
        `feasibility` decides.
        """
        key = (int(state), int(c), rotation is not None)
        if key in self._steps:
            cached = self._steps[key]
            return StepGrasp(**{**cached.__dict__, "engine_calls": 0})
        calls_before = self.n_engine_calls
        s_alone = self.alone(c)
        obstacles = tuple(o for o in range(len(self.clouds)) if (state >> o) & 1 and o != c)
        angles: list[float] = []
        if intrinsically_infeasible(s_alone, self.config):
            reason = ("no grasp even alone (S_alone = 0)" if s_alone <= 0 else
                      f"fails even alone (S_alone {s_alone:.3f} below absolute floor {self.config.grasp_abs_floor})")
            out = StepGrasp(state=key[0], component=key[1], label=self.labels[c], score_alone=s_alone,
                            score_context=None, score_relative=None, feasible=False, reason=reason, intrinsic=True,
                            best=None, rotation_checked=False, rotation=rotation)
        else:
            # Static call, or one call per rotation angle with only the grasps that survive all angles.
            if rotation is None:
                res = self._call(c, obstacles)
                grasps = res.grasps
            else:
                step_deg = self.config.grasp_rotation_step_deg
                angles = [float(a) for a in np.arange(0.0, 360.0, step_deg)]
                per_angle = [self._call(c, obstacles, rotation, a) for a in angles]
                grasps = surviving_grasps(per_angle, self.config.grasp_match_tol_mm,
                                          self.config.grasp_match_angle_deg)
            # S_context = best score; apply the feasibility rule.
            best = max(grasps, key=lambda g: g.score) if grasps else None
            s_ctx = 0.0 if best is None else float(best.score)
            feasible, reason = feasibility(s_ctx, s_alone, self.config)
            out = StepGrasp(state=key[0], component=key[1], label=self.labels[c], score_alone=s_alone,
                            score_context=s_ctx, score_relative=relative_score(s_ctx, s_alone), feasible=feasible,
                            reason=reason, intrinsic=False, best=best, rotation_checked=rotation is not None,
                            angles=angles, rotation=rotation, n_grasps=len(grasps))
        out.engine_calls = self.n_engine_calls - calls_before
        self._steps[key] = out
        return out
