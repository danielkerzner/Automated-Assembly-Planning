"""Gripper analysis for robot mode (docs/robotic-gripper_spec.md section 5). Pure numpy, no FreeCAD.

For a step "remove component i from state S" (S = bitmask of present components): target = cloud of i
in place, obstacles = merged clouds of every other component present in S (standers included: they are
physical obstacles for the gripper). The check is static (grasp with i still inserted). One score per
(state, component), shared by every free direction of the edge.

  S_alone    engine score of i with no obstacles, once per component.
  S_context  engine score of i in state S.
  S_rel      min(1, S_context / S_alone); S_alone = 0 -> infeasible.
  feasible   S_rel >= grasp_rel_threshold and S_context >= grasp_abs_floor (boundary per config).

Intrinsically infeasible components (implementation decision, flagged in the spec's status line): if
S_alone itself already fails the rule (S_alone = 0, or S_alone below the absolute floor), no state can
pass (obstacles only remove candidates, spec 5.2), so no engine call is made for any state and every
step of that component is reported infeasible with `intrinsic = True`. The loop never prunes such an
edge and leaves it out of the quarter: the component is infeasible in every state, so pruning it would
only disconnect the AoG without improving anything. They are reported for the user to check by hand.

Rotation (spec 5.3): when the removed component is the partner of a nut that is still present, the
component (and the gripper with it) turns about the nut axis. The target and grasp frame stay fixed and
the obstacle cloud is rotated by the opposite angle, at every angle of 0..360 in steps of
`grasp_rotation_step_deg`. A grasp survives if a matching grasp (TCP within `grasp_match_tol_mm`, each
axis within `grasp_match_angle_deg`, signed) is feasible at every angle; its score is the lowest of its
matched scores (conservative), and the step score is the best surviving score (0 if none). Angle 0 is
the static call and is shared with the non-rotating cache entry.

Caching (spec 5.4): every engine call is memoised in memory, and optionally on disk under `cache_dir`,
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
    """S_rel = min(1, S_context / S_alone); 0 when the component has no grasp at all."""
    if score_alone <= 0:
        return 0.0
    return float(min(1.0, score_context / score_alone))


def _ge(a: float, b: float, inclusive: bool) -> bool:
    return a >= b if inclusive else a > b


def feasibility(score_context: float, score_alone: float, config: RobotConfig) -> tuple[bool, str | None]:
    """(feasible, reason if not)."""
    if score_alone <= 0:
        return False, "no grasp even alone (S_alone = 0)"
    rel = relative_score(score_context, score_alone)
    if not _ge(rel, config.grasp_rel_threshold, config.grasp_boundary_inclusive):
        return False, f"relative score {rel:.3f} below {config.grasp_rel_threshold}"
    if not _ge(score_context, config.grasp_abs_floor, config.grasp_boundary_inclusive):
        return False, f"score {score_context:.3f} below absolute floor {config.grasp_abs_floor}"
    return True, None


def intrinsically_infeasible(score_alone: float, config: RobotConfig) -> bool:
    """S_alone already fails the rule, so every state fails (S_context <= S_alone, spec 5.2)."""
    return score_alone <= 0 or not _ge(score_alone, config.grasp_abs_floor, config.grasp_boundary_inclusive)


# ---------------------------------------------------------------------------
# Rotation matching
# ---------------------------------------------------------------------------


def _match(g: Grasp, candidates: list[Grasp], tol_mm: float, cos_tol: float) -> Grasp | None:
    """Best-scoring grasp in candidates with the same TCP (within tol) and the same three axes."""
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
    """Grasps of the first angle with a match at every other angle; score = lowest matched score."""
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
    """The nut axis a rotating partner turns about (spec 5.3)."""
    axis_point: tuple[float, float, float]
    axis_dir: tuple[float, float, float]
    nut_id: str
    nut_label: str

    def key(self) -> str:
        return json.dumps([np.round(self.axis_point, 6).tolist(), np.round(self.axis_dir, 6).tolist()])

    def to_json(self) -> dict:
        return {"axis_point": [float(v) for v in self.axis_point], "axis_dir": [float(v) for v in self.axis_dir],
                "nut_id": self.nut_id, "nut_label": self.nut_label}


@dataclass
class StepGrasp:
    state: int
    component: int
    label: str
    score_alone: float
    score_context: float | None  # None when not evaluated (intrinsically infeasible component)
    score_relative: float | None
    feasible: bool
    reason: str | None
    intrinsic: bool  # fails even alone; never pruned (see module docstring)
    best: Grasp | None
    rotation_checked: bool = False
    angles: list[float] = field(default_factory=list)
    rotation: Rotation | None = None
    n_grasps: int = 0
    engine_calls: int = 0  # new engine calls this evaluation cost (0 = all cached)

    def to_json(self) -> dict:
        return {"state": self.state, "component": self.component, "label": self.label,
                "score_alone": self.score_alone, "score": self.score_context,
                "score_relative": self.score_relative, "feasible": self.feasible, "reason": self.reason,
                "intrinsically_infeasible": self.intrinsic,
                "best": None if self.best is None else self.best.to_json(),
                "rotation_checked": self.rotation_checked, "angles": self.angles,
                "rotation": None if self.rotation is None else self.rotation.to_json(), "n_grasps": self.n_grasps}


class GripperAnalysis:
    """Scores per (state, component) with caching. `clouds[c]`/`cloud_keys[c]` by AoG component index."""

    def __init__(self, engine, clouds: list, cloud_keys: list[str], labels: list[str], config: RobotConfig,
                 cache_dir: str | Path | None = None, engine_tag: str = ""):
        self.engine, self.clouds, self.cloud_keys, self.labels = engine, clouds, cloud_keys, labels
        self.config, self.engine_tag = config, engine_tag
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._calls: dict[str, EngineResult] = {}
        self._alone: dict[int, float] = {}
        self._steps: dict[tuple[int, int, bool], StepGrasp] = {}
        self.n_engine_calls = 0  # calls actually sent to the engine (not served from a cache)
        self.n_disk_hits = 0

    # -- engine calls with cache --
    def _call_key(self, target: int, obstacles: tuple[int, ...], rotation: Rotation | None, angle: float) -> str:
        rot = "none" if rotation is None or angle % 360 == 0 else f"{rotation.key()}@{angle % 360:.6f}"
        h = hashlib.sha256(f"engine:{self.engine_tag};target:{self.cloud_keys[target]};".encode())
        h.update(("obstacles:" + ",".join(sorted(self.cloud_keys[o] for o in obstacles)) + f";rot:{rot}").encode())
        return h.hexdigest()

    def _call(self, target: int, obstacles: tuple[int, ...], rotation: Rotation | None = None,
              angle: float = 0.0) -> EngineResult:
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
        obst = merge_clouds([self.clouds[o] for o in obstacles]) if obstacles else None
        if obst is not None and rotation is not None and angle % 360 != 0:
            # The part turns by +angle about the nut axis; in its own frame the rest turns by -angle.
            obst = rotate_cloud(obst, rotation.axis_point, rotation.axis_dir, -angle)
        res = self.engine.evaluate(self.clouds[target], obst)
        self.n_engine_calls += 1
        log.info("Engine: %s with %d obstacle(s)%s -> %d grasps, best %.3f (%.1f s)", self.labels[target],
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
        if c not in self._alone:
            self._alone[c] = self._call(c, ()).score
        return self._alone[c]

    def alone_scores(self) -> dict[int, float]:
        return dict(self._alone)

    def step(self, state: int, c: int, rotation: Rotation | None = None) -> StepGrasp:
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
            if rotation is None:
                res = self._call(c, obstacles)
                grasps = res.grasps
            else:
                step_deg = self.config.grasp_rotation_step_deg
                angles = [float(a) for a in np.arange(0.0, 360.0, step_deg)]
                per_angle = [self._call(c, obstacles, rotation, a) for a in angles]
                grasps = surviving_grasps(per_angle, self.config.grasp_match_tol_mm,
                                          self.config.grasp_match_angle_deg)
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
