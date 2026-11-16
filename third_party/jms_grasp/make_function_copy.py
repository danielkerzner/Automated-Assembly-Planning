"""Generate `jms_engine.py`, the callable function copy of the JMS parallel-gripper script.

[SUPP] Support script: it is not imported by the pipeline. Run it only to refresh the copy.

Inputs
    - The JMS engine repository (read only), file `main_script_parallel_grasp_valid_assembly.py` and
      `grasp_store.py`. Default location: `../gripperparallel/Adaptive-In-Orbit-Servicing-of-Altered-Satellite-Components/JMS`
      next to the AAP repository, or `--engine <path>`.
Outputs
    - `jms_engine.py` (the generated function copy, with `SOURCE_SHA256` and `COPY_VERSION` constants) and a
      verbatim copy of `grasp_store.py`, both written to this folder (or `--out`).
    - `jms_engine.py` must never be edited by hand: change the edit list below and regenerate.

What this module does
    The engine repository is read-only and its script is not a library. This generator reads the UNCHANGED script and
    applies a short, fixed list of text edits, each asserted to match exactly as often as expected, so the copy is
    reproducible and every deviation from the original is listed here (and repeated in the generated file's header):

  E1  The whole script body (everything after the imports) becomes the body of
      `evaluate(pcd, pcd_surround, params)`; it returns the `GraspStore`.
  E2  Path catalog: the hard-coded `Object/unknown_3.pcd` / `Object/unknown_0.pcd` /
      `gripper_parameter/Franka.yaml` paths are replaced by dummy names (still used in strings).
  E3  `save_image = False` (was True): the matplotlib figures/SVGs are not drawn. The guarded
      blocks only plot (verified by reading them; `_p2_inner_rings` is only created and used inside
      them) -- the equivalence test in automatic_tests/test_robotic.py checks this.
  E4  The results/ folder creation (`mkdir`) is removed.
  E5  The gripper YAML is passed in as `params` (dict) instead of being read from disk.
  E6  `logging.basicConfig(...)` is removed (would reconfigure the worker's root logger).
  E7  The target / obstacle clouds are passed in as open3d PointClouds instead of being read
      from `Object/*.pcd`.
  E8  `o3d.io.write_point_cloud("overlap_pcd.pcd" / "proj_pcd_p3/4/5.pcd", ...)` cwd side effects
      are removed.
  E9  The final `store.save_npz/save_csv` + `np.save(candidate...)` writes are removed.
  E10 The trailing best/top-5/worst console report is kept (it only prints).

Not changed (default = unchanged): `remove_pcd_outlier_dbscan` stub (drops any
projected obstacle region with <= 500 points), every algorithm parameter, the score
(0.5 * center + 0.5 * area). Determinism (open3d RANSAC seeding) is handled by the caller
(`src/gripper_engine_robotic.py`), not by an edit here.

How to run (any Python, no third-party imports; from the repository root)
    python third_party/jms_grasp/make_function_copy.py [--engine <path to JMS folder>] [--out <folder>]
    Expected result: it prints "wrote .../jms_engine.py". It fails with a RuntimeError if the source script changed so
    that an edit no longer matches exactly the expected number of times (a deliberate stop, see EDITS).
    After regenerating, run `python automatic_tests/test_robotic.py --groups engine` (worker equals the unchanged script).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent  # this folder: default output of the copy
DEFAULT_ENGINE = (HERE.parents[2] / "gripperparallel" / "Adaptive-In-Orbit-Servicing-of-Altered-Satellite-Components"
                  / "JMS")
SCRIPT = "main_script_parallel_grasp_valid_assembly.py"  # the unchanged engine script that is copied
COPY_VERSION = "1"  # bump when the edit list changes (part of the grasp disk-cache key)

# The edit list E2..E9 described in the module docstring (E1, the function wrapper, is built in `generate`).
# Each entry is (id, old text, new text, expected number of matches) and is applied in order to the script body
# (everything after the import block). A different count aborts the generation, so a changed source never
# produces a silently different copy.
EDITS = [
    ("E2", 'pcd_path = Path("Object/unknown_3.pcd")\npcd_path_surround= Path("Object/unknown_0.pcd")\n\n'
           'gripper_path = Path("gripper_parameter/Franka.yaml")\n',
     'pcd_path = Path("target.pcd")  # E2: dummy name, clouds are passed in\n'
     'pcd_path_surround = Path("obstacles.pcd")  # E2\n\ngripper_path = Path("gripper.yaml")  # E2\n', 1),
    ("E3", "save_image = True # running with image compute and save image\n",
     "save_image = False  # E3: was True; no figures are drawn or saved\n", 1),
    ("E4", "graspdata_basepath.mkdir(parents=True, exist_ok=True)\nimageout_basepath.mkdir(parents=True, exist_ok=True)\n",
     "# E4: results folders are not created\n", 1),
    ("E5", 'with open(gripper_path, "r", encoding="utf-8") as f:\n    params = yaml.safe_load(f)\n',
     "# E5: `params` (the gripper YAML as a dict) is an argument\n", 1),
    ("E6", 'logging.basicConfig(level=logging.INFO, format="%(message)s")\n', "# E6: logging.basicConfig removed\n", 1),
    ("E7", "path = pathlib.Path(pcd_path)\npath_surround = pathlib.Path(pcd_path_surround)\n"
           "pcd = o3d.io.read_point_cloud(str(path))\npcd_surround = o3d.io.read_point_cloud(str(path_surround))\n",
     "# E7: `pcd` / `pcd_surround` are arguments (open3d PointClouds)\n", 1),
    ("E8", 'o3d.io.write_point_cloud("overlap_pcd.pcd", overlap_pcd)', "pass  # E8", 1),
    ("E8", 'o3d.io.write_point_cloud("proj_pcd_p3.pcd", proj_pcd_p3)', "pass  # E8", 1),
    ("E8", 'o3d.io.write_point_cloud("proj_pcd_p4.pcd", proj_pcd_p4)', "pass  # E8", 1),
    ("E8", 'o3d.io.write_point_cloud("proj_pcd_p5.pcd", proj_pcd_p5)', "pass  # E8", 1),
    ("E9", "store.save_npz(npz_outpath)\nstore.save_csv(csv_outpath)\n\n"
           "np.save(npz_outpath2, np.array(candidate_TCP_all_pairs, dtype=object))\n",
     "# E9: result files are not written; the store is returned\n", 1),
]
# Line of the source script that ends its import block: what comes before stays at module level, what comes after
# becomes the body of `evaluate`. It must occur exactly once.
IMPORT_END_MARKER = "#**************** path catalog **************\n"


def _git_head(path: Path) -> str:
    """[AUX] Commit hash (`git rev-parse HEAD`) of the engine repository, or "unknown" if git or the repo is missing.

    Why: only written into the header of the copy for traceability; it is not part of any cache key.
    """
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def generate(engine: Path, out_dir: Path) -> Path:
    """[MAIN] Write `jms_engine.py` and `grasp_store.py` into `out_dir`; return the path of `jms_engine.py`.

    Input: `engine` is the JMS folder of the engine repository (read only), `out_dir` the target folder (created if
    missing). Steps: hash the source script; split it at `IMPORT_END_MARKER`; apply every edit of `EDITS` (each
    must match the expected number of times); indent the body into `evaluate(pcd, pcd_surround, params)`; prepend
    the header with source path, commit, SHA-256 and copy version; syntax-check the result with `compile`; write it.
    Raises RuntimeError if the marker or an edit does not match, or if the source no longer has
    `no_image = True` (input() prompts would then be active in the copy).
    """
    src_path = engine / SCRIPT
    src = src_path.read_text(encoding="utf-8")
    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    if src.count(IMPORT_END_MARKER) != 1:
        raise RuntimeError("import block end marker not found exactly once; the script changed")
    head, body = src.split(IMPORT_END_MARKER, 1)
    body = IMPORT_END_MARKER + body
    for eid, old, new, count in EDITS:
        n = body.count(old)
        if n != count:
            raise RuntimeError(f"{eid}: expected {count} match(es), found {n}: {old[:60]!r}")
        body = body.replace(old, new)
    # no_image = True in the source switches off its interactive input() prompts; the worker has no terminal.
    if "no_image = True" not in body:
        raise RuntimeError("no_image is not True in the source; input() prompts would be active")

    # E1: the whole body goes one level deeper, blank lines stay empty.
    indented = "".join(("    " + ln) if ln.strip() else ln for ln in body.splitlines(keepends=True))
    header = f'''"""GENERATED FILE -- do not edit; regenerate with make_function_copy.py.

Function copy of the JMS parallel-gripper grasp engine (made by third_party/jms_grasp/make_function_copy.py).
Source (read only, never edited): {src_path.as_posix()}
Source repository commit: {_git_head(engine)}
Source file SHA-256: {sha}
Copy version: {COPY_VERSION}
Original license: MIT, Copyright (c) 2025 Justus Rein (see the source repository's LICENSE).

Edits relative to the source (see make_function_copy.py for the exact text of each):
  E1 body wrapped into evaluate(pcd, pcd_surround, params) -> GraspStore
  E2 hard-coded paths -> dummy names     E3 save_image = False (no figures)
  E4 no results/ folders                 E5 gripper params passed in
  E6 no logging.basicConfig              E7 clouds passed in as open3d PointClouds
  E8 no overlap/proj_pcd_*.pcd writes    E9 no npz/csv/npy result writes
Algorithm, parameters and the DBSCAN stub are unchanged.
"""
SOURCE_SHA256 = "{sha}"
COPY_VERSION = "{COPY_VERSION}"

'''
    # E1: the script's own result object is returned instead of being written to disk.
    footer = "\n    return store  # E1\n"
    code = header + head + "\n\ndef evaluate(pcd, pcd_surround, params):  # E1\n" + indented + footer
    compile(code, "jms_engine.py", "exec")  # syntax check
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "jms_engine.py"
    out.write_text(code, encoding="utf-8")
    shutil.copy(engine / "grasp_store.py", out_dir / "grasp_store.py")  # data classes used by evaluate(); copied verbatim
    return out


def main() -> int:
    """[MAIN] Command line entry: parse `--engine` / `--out`, call `generate`, print the path written; return 0."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help="the engine repo's JMS folder (read only)")
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()
    out = generate(args.engine, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
