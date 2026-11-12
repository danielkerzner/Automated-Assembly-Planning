"""RobotConfig: every robot-mode parameter (docs/robotic-gripper_spec.md section 9).

Values decided by the thesis author on 2026-11-12 are marked "decided"; the rest are the spec's
defaults or implementation choices, marked "to confirm" where the spec leaves them open.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RobotConfig:
    # Loop (spec 6)
    robot_max_iterations: int = 10
    grasp_quarter_fraction: float = 0.25  # decided: counted over gripper steps only; prune max(1, floor(N * f))

    # Feasibility rule (spec 5.2): feasible iff S_rel >= grasp_rel_threshold and S_context >= grasp_abs_floor
    grasp_rel_threshold: float = 0.5  # decided 2026-11-12: "guess a value" -> the spec's suggested 0.5
    grasp_abs_floor: float = 0.5  # decided 2026-11-12: "adjustable, give it 50% and let's see"
    grasp_boundary_inclusive: bool = True  # to confirm (spec 12)

    # Rotation of nut partners (spec 5.3)
    grasp_rotation_step_deg: float = 90.0  # decided 2026-11-12: 0, 90, 180, 270 degrees
    grasp_match_tol_mm: float = 0.5  # same grasp at two angles: TCP within this distance ...
    grasp_match_angle_deg: float = 5.0  # ... and each of the three axes within this angle (signed)

    # Point clouds (spec 3)
    pcd_spacing_mm: float = 0.3
    pcd_max_points: int = 60000
    pcd_seed: int = 1

    # Engine (spec 4)
    engine_python: str | None = None  # None: $AAP_GRIPPER_PYTHON, else the sibling env 'parallelgrippercode'
    engine_repo: str | None = None  # the engine's JMS folder (read only); None: ../gripperparallel/...
    gripper_yaml: str | None = None  # None: <engine_repo>/gripper_parameter/Franka.yaml
    engine_timeout_s: float = 1800.0
    engine_mode: str = "worker"  # "worker" (function copy, long-lived) or "script" (fallback runner)
    engine_seed: int = 1  # open3d RANSAC seed, set before every call

    # Tool check (spec 7). Stander category -> tool STEP file in tool_dir; None = no tool yet
    # (the step is reported as unchecked and treated as feasible, the spec's stub behaviour).
    tool_dir: str = str(REPO_ROOT / "tools" / "gripper")
    tools: dict = field(default_factory=lambda: {1: "screwing_effector.step", 2: None, 3: None})
    tool_engage_depth_mm: float = 0.0  # to confirm: how far the bit tip sits inside the screw head
    tool_contact_volume_tol: float = 1e-3  # mm^3; a common volume above this is a contact (collision)

    disk_cache: bool = True  # engine results cached under <output_dir>/robot/engine_cache/

    def __post_init__(self):
        if self.engine_mode not in ("worker", "script"):
            raise ValueError(f"engine_mode must be 'worker' or 'script', got {self.engine_mode!r}")
        if self.robot_max_iterations < 1:
            raise ValueError("robot_max_iterations must be >= 1")
        if not 0 < self.grasp_quarter_fraction <= 1:
            raise ValueError("grasp_quarter_fraction must be in (0, 1]")
        if self.grasp_rotation_step_deg <= 0 or 360 % self.grasp_rotation_step_deg > 1e-9:
            raise ValueError("grasp_rotation_step_deg must divide 360")

    def thresholds(self) -> dict:
        return {"grasp_rel_threshold": self.grasp_rel_threshold, "grasp_abs_floor": self.grasp_abs_floor,
                "boundary_inclusive": self.grasp_boundary_inclusive,
                "grasp_rotation_step_deg": self.grasp_rotation_step_deg,
                "grasp_quarter_fraction": self.grasp_quarter_fraction,
                "tool_contact_volume_tol_mm3": self.tool_contact_volume_tol}

    def to_json(self) -> dict:
        d = asdict(self)
        d["tools"] = {str(k): v for k, v in self.tools.items()}
        return d
