"""Grasp engine wrapper for robot mode (docs/robotic-gripper_spec.md section 4).

The JMS parallel-gripper engine needs open3d/shapely/opencv/scikit-learn and runs in its own conda env
(`parallelgrippercode`), separate from `environmentAAP` (FreeCAD). This module is imported on both sides:

  client (environmentAAP)   `GraspEngine`: starts one long-lived worker subprocess and sends it
                            evaluations (file exchange: npz clouds in, json grasps out).
  worker (parallelgrippercode)  `python gripper_engine_robotic.py --worker`: imports the function copy
                            `third_party/jms_grasp/jms_engine.py` once, then serves requests from stdin.

Two runners (`RobotConfig.engine_mode`):
  "worker"  (default) the function copy in the long-lived worker.
  "script"  the fallback runner: a copy of the UNCHANGED engine script in a scratch folder per call
            (the reference harness of gripper_test/run_engine_tests.py). Slow; kept for the equivalence
            test and as a fallback.

Both seed open3d's RNG before each evaluation (`o3d.utility.random.seed`): the engine's plane detection
is RANSAC (`segment_plane`), so without a seed identical input does not give identical output. This is
the only addition around the engine; the algorithm is unchanged.

Clouds are handed over exactly as the engine would read them from the ASCII PCD files of the reference
pipeline (values rounded to 5 decimals, parsed as double), so worker, script runner and the reference
measurements of spec 5.2 all see the same numbers. RANSAC is sensitive to differences as small as 1e-6 mm.

No FreeCAD import here; no open3d import on the client side.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("aap.robot.engine")

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_COPY_DIR = REPO_ROOT / "third_party" / "jms_grasp"
DEFAULT_ENGINE_REPO = (REPO_ROOT.parent / "gripperparallel" / "Adaptive-In-Orbit-Servicing-of-Altered-Satellite-Components"
                       / "JMS")
SCRIPT_NAME = "main_script_parallel_grasp_valid_assembly.py"
PROTOCOL_PREFIX = "@@AAP@@ "


class EngineError(RuntimeError):
    """The engine crashed, timed out or the worker died. Never turned into a score of 0 (spec 4)."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class Grasp:
    tcp: np.ndarray  # (3,) mm, assembly frame
    axes: np.ndarray  # (3, 3) rows: X = edge normal, Y = edge direction, Z = plane normal
    contact_area: float
    grasp_width: float
    score: float  # engine score 0.5 * center + 0.5 * area, in 0..1
    pair: int | None = None
    edge: int | None = None
    point: int | None = None

    def to_json(self) -> dict:
        return {"tcp": [float(v) for v in self.tcp], "axes": [[float(v) for v in r] for r in self.axes],
                "contact_area": float(self.contact_area), "grasp_width": float(self.grasp_width),
                "score": float(self.score), "pair": self.pair, "edge": self.edge, "point": self.point}

    @staticmethod
    def from_json(d: dict) -> "Grasp":
        return Grasp(tcp=np.asarray(d["tcp"], float), axes=np.asarray(d["axes"], float),
                     contact_area=float(d["contact_area"]), grasp_width=float(d["grasp_width"]),
                     score=float(d["score"]), pair=d.get("pair"), edge=d.get("edge"), point=d.get("point"))


@dataclass
class EngineResult:
    grasps: list[Grasp] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def best(self) -> Grasp | None:
        return max(self.grasps, key=lambda g: g.score) if self.grasps else None

    @property
    def score(self) -> float:
        """Score of a call = best score in the store, 0 if the store is empty (spec 4)."""
        b = self.best
        return 0.0 if b is None else float(b.score)

    def to_json(self) -> dict:
        return {"grasps": [g.to_json() for g in self.grasps], "wall_time_s": self.wall_time_s}

    @staticmethod
    def from_json(d: dict) -> "EngineResult":
        return EngineResult(grasps=[Grasp.from_json(g) for g in d["grasps"]],
                            wall_time_s=float(d.get("wall_time_s", 0.0)))


# ---------------------------------------------------------------------------
# Cloud exchange (shared by both sides)
# ---------------------------------------------------------------------------


TRANSFER_VERSION = "2"  # 1 cast to float32 (wrong: open3d parses ASCII PCD as double); part of the cache key


def as_pcd_precision(a: np.ndarray) -> np.ndarray:
    """What open3d reads back from an ASCII PCD written with '%.5f': the values rounded to 5 decimals,
    parsed as double (checked: no float32 step, despite the header's TYPE F)."""
    return np.round(np.asarray(a, dtype=float), 5)


def save_cloud_npz(path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    np.savez(path, points=np.asarray(points, float).reshape(-1, 3), normals=np.asarray(normals, float).reshape(-1, 3))


def write_pcd(path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    """ASCII PCD with normals, same format as gripper_test/make_pcds_freecad.py."""
    P, N = np.asarray(points, float).reshape(-1, 3), np.asarray(normals, float).reshape(-1, 3)
    n = len(P)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
                "FIELDS x y z normal_x normal_y normal_z\nSIZE 4 4 4 4 4 4\nTYPE F F F F F F\n"
                "COUNT 1 1 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n" % (n, n))
        if n:
            np.savetxt(f, np.hstack([P, N]), fmt="%.5f")


def _check_target(points: np.ndarray) -> None:
    if len(points) == 0:
        raise ValueError("Empty target cloud")
    extent = float((points.max(0) - points.min(0)).max())
    if extent <= 1.0:
        # The engine treats such a cloud as metres and scales it by 1000 (spec 3).
        raise ValueError(f"Target cloud's longest bbox side is {extent:.3f} mm (<= 1); the engine would "
                         "treat it as metres")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def default_engine_python() -> Path:
    """The `parallelgrippercode` interpreter: $AAP_GRIPPER_PYTHON, else the sibling of this conda env."""
    env = os.environ.get("AAP_GRIPPER_PYTHON")
    if env:
        return Path(env)
    prefix = Path(sys.prefix)
    base = prefix.parent if prefix.parent.name == "envs" else prefix / "envs"
    exe = "python.exe" if os.name == "nt" else "bin/python"
    return base / "parallelgrippercode" / exe


def _subprocess_env(engine_python: Path) -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", MPLBACKEND="Agg")
    env.pop("PYTHONPATH", None)  # environmentAAP's PYTHONPATH points at FreeCAD's Library\bin
    prefix = engine_python.parent
    extra = [prefix, prefix / "Library" / "mingw-w64" / "bin", prefix / "Library" / "usr" / "bin",
             prefix / "Library" / "bin", prefix / "Scripts"]
    env["PATH"] = os.pathsep.join([str(p) for p in extra] + [env.get("PATH", "")])
    return env


class GraspEngine:
    """Evaluate (target cloud, obstacle cloud) with the JMS engine. Use as a context manager."""

    def __init__(self, engine_python: str | Path | None = None, gripper_yaml: str | Path | None = None,
                 timeout_s: float = 1800.0, seed: int = 1, mode: str = "worker",
                 work_dir: str | Path | None = None, engine_repo: str | Path | None = None):
        if mode not in ("worker", "script"):
            raise ValueError(f"engine mode must be 'worker' or 'script', got {mode!r}")
        self.engine_python = Path(engine_python) if engine_python else default_engine_python()
        if not self.engine_python.exists():
            raise FileNotFoundError(f"Grasp engine interpreter not found: {self.engine_python} (set "
                                    "RobotConfig.engine_python or $AAP_GRIPPER_PYTHON)")
        self.engine_repo = Path(engine_repo) if engine_repo else DEFAULT_ENGINE_REPO
        self.gripper_yaml = Path(gripper_yaml) if gripper_yaml else self.engine_repo / "gripper_parameter" / "Franka.yaml"
        if not self.gripper_yaml.exists():
            raise FileNotFoundError(f"Gripper YAML not found: {self.gripper_yaml}")
        self.timeout_s, self.seed, self.mode = float(timeout_s), int(seed), mode
        self._own_tmp = work_dir is None
        # Absolute: the worker runs with cwd = work_dir, so relative request paths would resolve twice.
        self.work_dir = (Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="aap_grasp_"))).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.n_calls = 0
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue | None = None
        self._req = 0

    # -- lifecycle --
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.stdin.write("quit\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc = None
        if self._own_tmp:
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def _start_worker(self) -> None:
        log_path = self.work_dir / "worker_engine_output.log"
        cmd = [str(self.engine_python), "-u", str(Path(__file__).resolve()), "--worker", "--log", str(log_path)]
        log.info("Starting grasp engine worker: %s", self.engine_python)
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, encoding="utf-8", errors="replace",
                                      env=_subprocess_env(self.engine_python), cwd=str(self.work_dir))
        self._lines = queue.Queue()
        self._stderr_tail: list[str] = []

        def pump_out(stream, q):
            for line in stream:
                q.put(line)
            q.put(None)

        def pump_err(stream, tail):
            for line in stream:
                tail.append(line.rstrip())
                del tail[:-40]

        threading.Thread(target=pump_out, args=(self._proc.stdout, self._lines), daemon=True).start()
        threading.Thread(target=pump_err, args=(self._proc.stderr, self._stderr_tail), daemon=True).start()
        reply = self._read_reply(timeout=300.0)
        if not reply.get("ready"):
            raise EngineError(f"Grasp engine worker did not start: {reply}")
        log.info("Grasp engine worker ready (function copy %s, source sha %s...)", reply.get("copy_version"),
                 str(reply.get("source_sha256"))[:12])

    def _read_reply(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                self._kill()
                raise EngineError(f"Grasp engine worker timed out after {timeout:.0f} s")
            try:
                line = self._lines.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                code = self._proc.poll() if self._proc else None
                tail = "\n".join(self._stderr_tail[-15:])
                self._proc = None
                raise EngineError(f"Grasp engine worker exited (code {code}). stderr tail:\n{tail}")
            if line.startswith(PROTOCOL_PREFIX):
                return json.loads(line[len(PROTOCOL_PREFIX):])

    def _kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
        self._proc = None

    # -- evaluation --
    def evaluate(self, target: tuple[np.ndarray, np.ndarray],
                 obstacles: tuple[np.ndarray, np.ndarray] | None) -> EngineResult:
        """One engine call. `obstacles` None or empty = target alone (handled without an empty file)."""
        tp, tn = (np.asarray(a, float).reshape(-1, 3) for a in target)
        _check_target(tp)
        if obstacles is not None and len(obstacles[0]) == 0:
            obstacles = None
        self.n_calls += 1
        t0 = time.perf_counter()
        if self.mode == "worker":
            res = self._evaluate_worker(tp, tn, obstacles)
        else:
            res = self._evaluate_script(tp, tn, obstacles)
        res.wall_time_s = time.perf_counter() - t0
        return res

    def _evaluate_worker(self, tp, tn, obstacles) -> EngineResult:
        if self._proc is None or self._proc.poll() is not None:
            self._start_worker()
        self._req += 1
        rid = self._req
        tdir = self.work_dir / "exchange"
        tdir.mkdir(exist_ok=True)
        target_path, obst_path, out_path = tdir / f"t{rid}.npz", tdir / f"o{rid}.npz", tdir / f"r{rid}.json"
        save_cloud_npz(target_path, tp, tn)
        req = {"id": rid, "target": str(target_path), "obstacles": None, "out": str(out_path),
               "seed": self.seed, "gripper_yaml": str(self.gripper_yaml)}
        if obstacles is not None:
            save_cloud_npz(obst_path, *obstacles)
            req["obstacles"] = str(obst_path)
        try:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
        except OSError as exc:
            self._proc = None
            raise EngineError(f"Could not send a request to the grasp engine worker: {exc}") from exc
        reply = self._read_reply(self.timeout_s)
        try:
            if reply.get("id") != rid:
                raise EngineError(f"Grasp engine worker answered request {reply.get('id')}, expected {rid}")
            if not reply.get("ok"):
                raise EngineError(f"Grasp engine crashed:\n{reply.get('error')}")
            data = json.loads(out_path.read_text(encoding="utf-8"))
            return EngineResult.from_json(data)
        finally:
            for p in (target_path, obst_path, out_path):
                p.unlink(missing_ok=True)

    def _evaluate_script(self, tp, tn, obstacles) -> EngineResult:
        """Fallback runner: the unchanged script in a scratch folder (as gripper_test/run_engine_tests.py)."""
        import csv

        self._req += 1
        run_dir = self.work_dir / f"script_run_{self._req}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        (run_dir / "Object").mkdir(parents=True)
        (run_dir / "gripper_parameter").mkdir()
        shutil.copy(self.engine_repo / SCRIPT_NAME, run_dir / SCRIPT_NAME)
        shutil.copy(self.engine_repo / "grasp_store.py", run_dir / "grasp_store.py")
        shutil.copy(self.gripper_yaml, run_dir / "gripper_parameter" / "Franka.yaml")
        write_pcd(run_dir / "Object" / "unknown_3.pcd", tp, tn)
        if obstacles is None:
            write_pcd(run_dir / "Object" / "unknown_0.pcd", np.zeros((0, 3)), np.zeros((0, 3)))
        else:
            write_pcd(run_dir / "Object" / "unknown_0.pcd", *obstacles)
        launcher = (f"import open3d as o3d, random, numpy as np; o3d.utility.random.seed({self.seed}); "
                    f"random.seed({self.seed}); np.random.seed({self.seed}); import runpy; "
                    f"runpy.run_path({SCRIPT_NAME!r}, run_name='__main__')")
        try:
            proc = subprocess.run([str(self.engine_python), "-c", launcher], cwd=run_dir, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=self.timeout_s,
                                  env=_subprocess_env(self.engine_python))
        except subprocess.TimeoutExpired as exc:
            raise EngineError(f"Grasp engine script timed out after {self.timeout_s:.0f} s") from exc
        (run_dir / "engine_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (run_dir / "engine_stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
            raise EngineError(f"Grasp engine script crashed (code {proc.returncode}):\n{tail}")
        grasps = []
        for csv_path in (run_dir / "results").glob("*/grasp_data/*.csv"):
            with open(csv_path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    grasps.append(Grasp(
                        tcp=np.array([float(r["x"]), float(r["y"]), float(r["z"])]),
                        axes=np.asarray(json.loads(r["pose_json"]), float),
                        contact_area=float(r["contact_area"]), grasp_width=float(r["grasp_width"]),
                        score=float(r["score"]),
                        pair=int(r["pair_num"]) if r["pair_num"] else None,
                        edge=int(r["edge_num"]) if r["edge_num"] else None,
                        point=int(r["point_num"]) if r["point_num"] else None))
        shutil.rmtree(run_dir, ignore_errors=True)
        return EngineResult(grasps=grasps)


# ---------------------------------------------------------------------------
# Worker (runs in the parallelgrippercode env)
# ---------------------------------------------------------------------------


def _worker_cloud(o3d, path: str | None):
    pc = o3d.geometry.PointCloud()
    if path is None:
        return pc
    d = np.load(path)
    pts, nrm = as_pcd_precision(d["points"]), as_pcd_precision(d["normals"])
    if len(pts):
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.normals = o3d.utility.Vector3dVector(nrm)
    return pc


def _serve(log_path: str) -> int:
    # Protocol goes to a private copy of the real stdout; everything else (the engine's prints and
    # open3d's C++ messages, which write to fd 1 directly) goes to the log file.
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    os.dup2(log_f.fileno(), 1)
    sys.stdout = log_f

    def reply(obj):
        proto.write(PROTOCOL_PREFIX + json.dumps(obj) + "\n")
        proto.flush()

    import random

    import open3d as o3d
    import yaml

    sys.path.insert(0, str(ENGINE_COPY_DIR))
    import jms_engine  # noqa: E402

    reply({"ready": True, "copy_version": jms_engine.COPY_VERSION, "source_sha256": jms_engine.SOURCE_SHA256})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "quit":
            break
        req = json.loads(line)
        try:
            seed = int(req["seed"])
            o3d.utility.random.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            pcd = _worker_cloud(o3d, req["target"])
            sur = _worker_cloud(o3d, req.get("obstacles"))
            with open(req["gripper_yaml"], "r", encoding="utf-8") as f:
                params = yaml.safe_load(f)
            print(f"\n===== AAP request {req['id']}: target {len(pcd.points)} pts, obstacles {len(sur.points)} pts =====")
            store = jms_engine.evaluate(pcd, sur, params)
            grasps = []
            for r in store.all():
                grasps.append({"tcp": [float(v) for v in np.asarray(r.TCP, float).reshape(3)],
                               "axes": [[float(v) for v in vec] for vec in r.pose],
                               "contact_area": float(r.contact_area), "grasp_width": float(r.grasp_width),
                               "score": float(r.score), "pair": r.pair_num, "edge": r.edge_num,
                               "point": r.point_num})
            Path(req["out"]).write_text(json.dumps({"grasps": grasps}), encoding="utf-8")
            reply({"id": req["id"], "ok": True, "n": len(grasps)})
        except Exception:  # noqa: BLE001
            reply({"id": req.get("id"), "ok": False, "error": traceback.format_exc()})
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--log", default="worker_engine_output.log")
    a = ap.parse_args()
    if not a.worker:
        ap.error("this module is a library; run with --worker only from GraspEngine")
    raise SystemExit(_serve(a.log))
