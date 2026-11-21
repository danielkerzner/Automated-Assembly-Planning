"""RobotConfig: every robot-mode parameter.

Inputs
    - Nothing is read from disk: a `RobotConfig` is built in code (`main._main_robot` builds one with
      `check_subassembly_grasp` from the command line, everything else at its default).
Outputs
    - One `RobotConfig` object passed to the point-cloud, engine, gripper-analysis, tool-check, loop and
      output modules. Its `to_json` / `thresholds` are written into plan.json -> robot.

Values decided by the thesis author on 2026-11-12 are marked "decided"; the rest are
defaults or implementation choices, marked "to confirm" where still open.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RobotConfig:
    """Settings of robot mode (config input of `run_robot_pipeline`). Defaults are the recommended values."""

    # Loop
    robot_max_iterations: int = 10  # int >= 1. Maximum verify-prune-reoptimize iterations. Recommended: 10.
    # float in (0, 1]. Share of the worst gripper steps pruned per iteration, prune max(1, floor(N * f)).
    # Decided: counted over gripper steps only. Recommended: 0.25.
    grasp_quarter_fraction: float = 0.25

    # Feasibility rule: feasible iff S_rel >= grasp_rel_threshold and S_context >= grasp_abs_floor
    # float in [0, 1]. Minimum S_context / S_alone (how much of the free-space grasp quality must survive the
    # obstacles). Decided 2026-11-12: 0.5.
    grasp_rel_threshold: float = 0.5
    # float in [0, 1]. Minimum absolute engine score. Decided 2026-11-12: "adjustable, give it 50% and let's see".
    grasp_abs_floor: float = 0.5
    grasp_boundary_inclusive: bool = True  # bool. True: a score exactly on a threshold passes (>=). To confirm.

    # Optional sub-assembly check (added 2026-11-13): an edge is feasible only if, besides the part's
    # own grasp/tool check, both its nodes (the sub-assembly before and after the insertion) are graspable as one
    # object with nothing around. Off by default (one extra engine call per distinct sub-assembly).
    check_subassembly_grasp: bool = False  # bool. Command-line flag --check-subassembly-grasp.
    subassembly_grasp_floor: float | None = None  # float or None. Node graspable iff score >= this; None = grasp_abs_floor.

    # Rotation of nut partners
    # float, deg, must divide 360. Angle step of the rotation check. Decided 2026-11-12: 90 (0, 90, 180, 270).
    grasp_rotation_step_deg: float = 90.0
    grasp_match_tol_mm: float = 0.5  # float, mm. Same grasp at two angles: TCP within this distance ...
    grasp_match_angle_deg: float = 5.0  # float, deg. ... and each of the three axes within this angle (signed).

    # Point clouds
    pcd_spacing_mm: float = 0.3  # float, mm. Sampling spacing of the clouds (points ~ area / spacing^2).
    pcd_max_points: int = 60000  # int. Upper bound of points per component cloud.
    pcd_seed: int = 1  # int. Seed of the cloud sampling.

    # Engine
    engine_python: str | None = None  # str or None. Engine interpreter; None: $AAP_GRIPPER_PYTHON, else the sibling env 'parallelgrippercode'.
    engine_repo: str | None = None  # str or None. The engine's JMS folder (read only); None: ../gripperparallel/...
    gripper_yaml: str | None = None  # str or None. Gripper description; None: <engine_repo>/gripper_parameter/Franka.yaml.
    engine_timeout_s: float = 1800.0  # float, s. Timeout of one engine call (and worker start).
    engine_mode: str = "worker"  # str. "worker" (function copy, long-lived; recommended) or "script" (slow fallback runner).
    engine_seed: int = 1  # int. open3d RANSAC seed, set before every call.

    # Tool check (interface rule of 2026-11-12). Standard category -> tool STEP file in
    # tool_dir; None = no tool (the step is reported unchecked and treated as feasible, the stub behaviour).
    tool_dir: str = str(REPO_ROOT / "tools" / "gripper")  # str. Folder of the tool STEP files.
    # dict category -> file name or None. 1 = screw, 2 = nut, 3 = other (bushing).
    tools: dict = field(default_factory=lambda: {1: "screwing_effector.step", 2: "nut_effector.stp",
                                                 3: "bushing_effector.stp"})
    # Standard category -> fraction of the standard's extent along its removal axis that sits inside the tool,
    # past the tool's interface plane (decided 2026-11-12: screws 0 = touching, nuts and bushings half).
    tool_overlap_fraction: dict = field(default_factory=lambda: {1: 0.0, 2: 0.5, 3: 0.5})
    tool_contact_volume_tol: float = 1e-3  # float, mm^3. A common volume above this is a contact (collision).

    disk_cache: bool = True  # bool. Engine results cached under <output_dir>/robot/engine_cache/.

    def __post_init__(self):
        """[AUX] Validate the settings (ValueError): engine_mode, iterations >= 1, quarter fraction in (0, 1], rotation step dividing 360."""
        if self.engine_mode not in ("worker", "script"):
            raise ValueError(f"engine_mode must be 'worker' or 'script', got {self.engine_mode!r}")
        if self.robot_max_iterations < 1:
            raise ValueError("robot_max_iterations must be >= 1")
        if not 0 < self.grasp_quarter_fraction <= 1:
            raise ValueError("grasp_quarter_fraction must be in (0, 1]")
        if self.grasp_rotation_step_deg <= 0 or 360 % self.grasp_rotation_step_deg > 1e-9:
            raise ValueError("grasp_rotation_step_deg must divide 360")

    def subassembly_floor(self) -> float:
        """[AUX] Score floor of the sub-assembly check: `subassembly_grasp_floor`, or `grasp_abs_floor` when None."""
        return self.grasp_abs_floor if self.subassembly_grasp_floor is None else self.subassembly_grasp_floor

    def thresholds(self) -> dict:
        """[SUPP] The decision thresholds as a dict, for plan.json -> robot.thresholds."""
        return {"grasp_rel_threshold": self.grasp_rel_threshold, "grasp_abs_floor": self.grasp_abs_floor,
                "check_subassembly_grasp": self.check_subassembly_grasp,
                "subassembly_grasp_floor": self.subassembly_floor() if self.check_subassembly_grasp else None,
                "boundary_inclusive": self.grasp_boundary_inclusive,
                "grasp_rotation_step_deg": self.grasp_rotation_step_deg,
                "grasp_quarter_fraction": self.grasp_quarter_fraction,
                "tool_contact_volume_tol_mm3": self.tool_contact_volume_tol}

    def to_json(self) -> dict:
        """[SUPP] The whole config as a JSON-ready dict (category keys of `tools` / `tool_overlap_fraction` as strings), for plan.json -> robot.config."""
        d = asdict(self)
        d["tools"] = {str(k): v for k, v in self.tools.items()}
        d["tool_overlap_fraction"] = {str(k): v for k, v in self.tool_overlap_fraction.items()}
        return d
