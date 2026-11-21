"""Manual smoke test for the Illustration Generator (src/illustration_generator.py).

Counterpart of manual_tests/test_preprocessing_manual.py, manual_tests/test_interference_manual.py,
manual_tests/test_aog_manual.py and manual_tests/test_optimization_manual.py. Not an acceptance suite --
those are the L1-L14 tests in automatic_tests/test_illustration.py. This script runs the illustration module end to end
on one assembly and leaves the result where you can look at it:

  <out_dir>/illustrations/
    step_01_<label>.png ...     one action diagram per assembly step
    manual.pdf                  cover page (parts list, final assembly) + one page per step
    illustration_report.json    the module's own report (views, visibility figures, warnings)
    view_selection.json         cached view search; meshes/ and layers/ are the geometry caches
    script_report.json          this script's readable report (see below)

What it prints and writes:
  - the configuration actually used (image size, views searched, visibility thresholds)
  - one line per step, in assembly order: part, tool, camera (azimuth/elevation, global or not, how
    it was found), visible fraction of the part at its end and start position, projected trajectory
    length, and flags (x-ray, re-orient, hold in place, warnings)
  - run statistics: meshes and triangles, view search time, render time, cache hits
  - consistency checks that do not use the module's own rendering code:
      V1 files        one PNG per step, manual.pdf is a PDF, report files exist
      V2 order        frames follow the sequence order, one per part, component ids match
      V3 visibility   every frame passes the visibility rule, or carries an x-ray flag / a warning
      V4 geometry     the sequence itself: insertion direction is a unit vector, start offset =
                      -travel * direction (the ghost position the module draws)
      V5 views        the global view is used by every frame that is marked global; the views are
                      inside the elevation range; all frames of the global view share one camera
      V6 footnote     step counter, component label and tool (BOM tool, "by hand" when blank) on
                      every frame; re-orient / hold lines exactly where gravity changes / class 1
      V7 images       every PNG opens, is not blank, and all have the same size
      V8 cache        (--check-cache) a second call returns the cached result with the same hash

Two kinds of run:

  Real assembly (default). Runs the pipeline through Optimization (Interference and AoG are cached,
  --force-interference and --force-aog recompute them; the optimization result is cached too),
  then renders the optimized sequence from the FreeCAD shapes. Needs FreeCAD (environmentAAP).

      python manual_tests/test_illustration_manual.py

  Synthetic assembly (no FreeCAD, no STEP file): a hand-built assembly from test_aog_manual.py,
  optimized with the same made-up tools and groups as test_optimization_manual.py, drawn as boxes
  from its bounding boxes.

      python manual_tests/test_illustration_manual.py --synthetic demo
      python manual_tests/test_illustration_manual.py --synthetic lattice --n 6
      python manual_tests/test_illustration_manual.py --synthetic random --n 6 --m 2 --asm-seed 3

Tune the rendering from the command line, for example:

      python manual_tests/test_illustration_manual.py --size 1000x750 --dpi 120
      python manual_tests/test_illustration_manual.py --view -60 25 --show-vector
      python manual_tests/test_illustration_manual.py --vis-min 0.7 --no-gravity --no-hold-mark
      python manual_tests/test_illustration_manual.py --force            # re-render, ignore the illustration caches

The exit code is 0 when every check passes, 1 otherwise.

Expected result: checks V1-V7 (and V8 with --check-cache) print PASS and the exit code is 0. In <out_dir>/illustrations
there is one step_NN_<label>.png per step (one per group block when a BOM group is used), manual.pdf and the report
files; opening a PNG shows the assembled parts in grey, the new part in colour with its ghost and an arrow along the
insertion, and the footnote with step counter, part and tool. A frame that could not be shown plainly carries a warning
or the x-ray flag in the console table instead of failing. The second run with --check-cache returns the cached result
almost instantly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the synthetic builders in test_aog_manual.py

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402

import illustration_generator as il  # noqa: E402
from illustration_generator import IllustrationConfig  # noqa: E402
DEFAULT_STEP = REPO_ROOT / "data" / "industrial_fastner" / "industrial_fastner.stp"
DEFAULT_BOM = REPO_ROOT / "data" / "industrial_fastner" / "generated_bom.csv"
DEFAULT_OUT = REPO_ROOT / "output" / "industrial_fastner"
DEFAULT_H = 5.0

TOL = 1e-6


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _fmt_vec(v) -> str:
    return "(" + ", ".join(f"{round(float(c), 2) + 0.0:g}" for c in v) + ")"


def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise SystemExit(f"--size expects WIDTHxHEIGHT, e.g. 1600x1200, got {text!r}") from None


def build_config(args: argparse.Namespace) -> IllustrationConfig:
    kwargs: dict = {}
    if args.size:
        kwargs["image_size"] = _parse_size(args.size)
    if args.dpi is not None:
        kwargs["dpi"] = args.dpi
    if args.view:
        kwargs["default_view"] = (float(args.view[0]), float(args.view[1]))
    if args.vis_min is not None:
        kwargs["vis_min_end"] = kwargs["vis_min_start"] = args.vis_min
    if args.tess_tol is not None:
        kwargs["tess_tolerance_rel"] = args.tess_tol
    if args.no_gravity:
        kwargs["show_gravity"] = False
    if args.no_hold_mark:
        kwargs["mark_unstable"] = False
    if args.show_vector:
        kwargs["footnote_show_vector"] = True
    if args.title:
        kwargs["title"] = args.title
    if args.force:
        kwargs["force"] = True
    return IllustrationConfig(**kwargs)


# ---------------------------------------------------------------------------
# Synthetic assembly: boxes from the bounding boxes
# ---------------------------------------------------------------------------

_BOX_FACES = np.array([  # outward winding for a box with vertices numbered by bit pattern (x=1, y=2, z=4)
    [0, 2, 3], [0, 3, 1],  # -z
    [4, 5, 7], [4, 7, 6],  # +z
    [0, 1, 5], [0, 5, 4],  # -y
    [2, 6, 7], [2, 7, 3],  # +y
    [0, 4, 6], [0, 6, 2],  # -x
    [1, 3, 7], [1, 7, 5],  # +x
])


def box_mesh(lo, hi, key: str) -> il.Mesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    verts = np.array([[(hi if b & 1 else lo)[0], (hi if b & 2 else lo)[1], (hi if b & 4 else lo)[2]]
                      for b in range(8)])
    return il.mesh_from_arrays(verts, _BOX_FACES, key)


# ---------------------------------------------------------------------------
# Consistency checks (independent of the module's rendering code)
# ---------------------------------------------------------------------------


def _check(cid: str, name: str, problems: list[str], ok_detail: str = "") -> dict:
    return {"id": cid, "name": name, "passed": not problems,
            "detail": ok_detail if not problems else "; ".join(problems[:6]) + (
                f" (+{len(problems) - 6} more)" if len(problems) > 6 else "")}


def check_files(result, steps: list[dict]) -> dict:
    p: list[str] = []
    pngs = [Path(f["png"]) for f in result.frames]
    missing = [x.name for x in pngs if not x.is_file() or x.stat().st_size == 0]
    if missing:
        p.append(f"missing or empty PNG: {missing}")
    if len(result.frames) != len(steps):
        p.append(f"{len(result.frames)} frames for {len(steps)} steps")
    manual = Path(result.manual_path)
    if not manual.is_file():
        p.append("manual.pdf missing")
    elif manual.read_bytes()[:5] != b"%PDF-":
        p.append("manual.pdf does not start with %PDF-")
    out_dir = manual.parent
    for name in (il.REPORT_NAME, il.VIEWS_NAME):
        if not (out_dir / name).is_file():
            p.append(f"{name} missing")
    return _check("V1", "files", p, f"{len(pngs)} PNG + {manual.name} ({manual.stat().st_size // 1024} kB)"
                  if manual.is_file() else "")


def check_order(result, steps: list[dict]) -> dict:
    p: list[str] = []
    for s, (st, fr) in enumerate(zip(steps, result.frames)):
        if fr["step"] != st["step"] or fr["step"] != s + 1:
            p.append(f"frame {s} is step {fr['step']}, expected {st['step']}")
        if fr["component_id"] != st["component_id"]:
            p.append(f"step {s + 1}: frame shows {fr['component_id']}, sequence has {st['component_id']}")
    ids = [f["component_id"] for f in result.frames]
    if len(set(ids)) != len(ids):
        p.append("a part appears in more than one frame")
    return _check("V2", "order", p, "frames follow the sequence, each part once")


def check_visibility(result, config: IllustrationConfig) -> dict:
    p: list[str] = []
    n_xray = n_off = 0
    for f in result.frames:
        if f["xray"]:
            n_xray += 1
            if not f["warnings"]:
                p.append(f"step {f['step']}: x-ray frame without a warning")
        if not f["visible"]:
            n_off += 1
            if not f["warnings"]:
                p.append(f"step {f['step']}: not visible and no warning")
        elif not f["xray"]:
            if f["vis_end"] < config.vis_min_end - TOL:
                p.append(f"step {f['step']}: end visible {f['vis_end']:.2f} < {config.vis_min_end}")
            if f["vis_start"] < config.vis_min_start - TOL:
                p.append(f"step {f['step']}: start visible {f['vis_start']:.2f} < {config.vis_min_start}")
            if f["trajectory_px_frac"] < config.min_trajectory_px_frac - TOL:
                p.append(f"step {f['step']}: trajectory {f['trajectory_px_frac']:.2f} of the image is too short")
    n_ok = len(result.frames) - n_xray - n_off
    return _check("V3", "visibility", p, f"{n_ok} visible, {n_xray} x-ray, {n_off} not visible")


def check_geometry(steps: list[dict]) -> dict:
    p: list[str] = []
    for st in steps:
        d = np.asarray(st["direction"], float)
        off = np.asarray(st["start_offset"], float)
        travel = float(st["travel"])
        if abs(np.linalg.norm(d) - 1.0) > 1e-6:
            p.append(f"step {st['step']}: direction is not a unit vector")
        if np.abs(off + travel * d).max() > 1e-6 * max(1.0, travel):
            p.append(f"step {st['step']}: start_offset {_fmt_vec(off)} != -travel*direction {_fmt_vec(-travel * d)}")
        if travel <= 0:
            p.append(f"step {st['step']}: travel {travel} <= 0")
    return _check("V4", "geometry", p, "start offset = -travel * direction on every step")


def check_views(result, config: IllustrationConfig) -> dict:
    p: list[str] = []
    gv = tuple(result.global_view)
    for f in result.frames:
        v = (f["view"]["azimuth"], f["view"]["elevation"])
        if not 0 <= v[1] <= 90:
            p.append(f"step {f['step']}: elevation {v[1]} outside [0, 90]")
        same = abs(v[0] - gv[0]) < 1e-9 and abs(v[1] - gv[1]) < 1e-9
        if f["view"]["is_global"] and not same:
            p.append(f"step {f['step']}: marked global but uses az {v[0]:g} el {v[1]:g}")
        if same and not f["view"]["is_global"]:
            p.append(f"step {f['step']}: uses the global camera but is not marked global")
    n_g = sum(f["view"]["is_global"] for f in result.frames)
    return _check("V5", "views", p, f"global view az {gv[0]:g} el {gv[1]:g}, {n_g}/{len(result.frames)} frames use it")


def check_footnote(result, steps: list[dict], bom: dict, config: IllustrationConfig) -> dict:
    p: list[str] = []
    N = len(steps)
    for s, (st, f) in enumerate(zip(steps, result.frames)):
        text = " ".join(f["footnote"])
        tool = (bom.get(st["component_label"], {}).get("tool") or "").strip() or "by hand"
        for part in (f"{st['step']} / {N}", st["component_label"], tool):
            if part not in text:
                p.append(f"step {st['step']}: footnote lacks {part!r}")
        g, g_prev = st.get("gravity"), steps[s - 1].get("gravity") if s else None
        want_re = g is not None and g_prev is not None and not np.allclose(g, g_prev, atol=1e-9)
        if bool(f.get("reorient")) != want_re:
            p.append(f"step {st['step']}: re-orient flag {f.get('reorient')} but gravity "
                     f"{'changes' if want_re else 'does not change'}")
        want_hold = bool(config.mark_unstable and st.get("stability_class") == 1)
        if bool(f.get("hold")) != want_hold:
            p.append(f"step {st['step']}: hold flag {f.get('hold')} but stability class is {st.get('stability_class')}")
    n_re = sum(bool(f.get("reorient")) for f in result.frames)
    n_hold = sum(bool(f.get("hold")) for f in result.frames)
    return _check("V6", "footnote", p, f"label, tool and counter on all frames; {n_re} re-orient, {n_hold} hold")


def check_images(result) -> dict:
    import matplotlib.image as mpimg

    p: list[str] = []
    shapes = set()
    for f in result.frames:
        try:
            img = mpimg.imread(f["png"])
        except Exception as exc:  # noqa: BLE001
            p.append(f"step {f['step']}: PNG does not open ({exc})")
            continue
        shapes.add(img.shape[:2])
        if float(np.asarray(img[..., :3]).std()) < 1e-3:
            p.append(f"step {f['step']}: PNG is blank")
    if len(shapes) > 1:
        p.append(f"frames have different sizes: {sorted(shapes)}")
    size = next(iter(shapes)) if len(shapes) == 1 else None
    return _check("V7", "images", p, f"all open, none blank, {size[1]} x {size[0]} px" if size else "")


def check_cache(result, again, seconds: float) -> dict:
    p: list[str] = []
    if again.config_hash != result.config_hash:
        p.append("config hash changed between two identical calls")
    if [f["png"] for f in again.frames] != [f["png"] for f in result.frames]:
        p.append("the cached result lists different frames")
    return _check("V8", "cache", p, f"second call returned the same hash in {seconds:.2f} s")


def run_checks(result, steps: list[dict], bom: dict, config: IllustrationConfig, again=None,
               again_s: float = 0.0) -> list[dict]:
    checks = [check_files(result, steps), check_order(result, steps), check_visibility(result, config),
              check_geometry(steps), check_views(result, config), check_footnote(result, steps, bom, config),
              check_images(result)]
    if again is not None:
        checks.append(check_cache(result, again, again_s))
    return checks


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def run_real(args: argparse.Namespace, config: IllustrationConfig):
    import freecad  # noqa: F401  (sets up the FreeCAD lib path in the conda env; must come first)

    import optimization as op
    from aog_generator import AoGConfig, build_aog
    from collision import CollisionConfig
    from interference_identification import identify_interferences
    from preprocessing import run_preprocessing

    step, bom_path, out = args.paths if args.paths else (DEFAULT_STEP, DEFAULT_BOM, DEFAULT_OUT)
    out = Path(out)
    print(f"Preprocessing {Path(step).name} ...", flush=True)
    prep = run_preprocessing(step, bom_path)
    print(f"Interference Identification (h = {args.step}; cached unless --force-interference) ...", flush=True)
    interference = identify_interferences(prep, CollisionConfig(step=args.step), out / "interference",
                                          force=args.force_interference)
    print("AoG generation (cached unless --force-aog) ...", flush=True)
    aog = build_aog(interference, prep, AoGConfig(), cache_dir=out / "aog", force=args.force_aog)
    if not aog.complete:
        raise SystemExit("The AoG root is not live: there is no complete disassembly sequence to illustrate.")
    print("Optimization (cached unless --force-optimization) ...", flush=True)
    sequence = op.optimize_sequence(aog, interference, prep, op.OptimizationConfig(force=args.force_optimization),
                                    cache_dir=out / "optimization")
    bom = il.read_bom(prep.bom_path) if prep.bom_path is not None else {}
    illus_dir = out / "illustrations"

    def render(cfg):
        print("Illustration Generator ...", flush=True)
        return il.generate_illustrations(sequence, prep, illus_dir, cfg)

    return "real", sequence, bom, illus_dir, render


def run_synthetic(args: argparse.Namespace, config: IllustrationConfig):
    import optimization as op
    from aog_generator import AoGConfig, generate_aog
    from test_aog import synthetic_demo, synthetic_lattice, synthetic_random, synthetic_stack
    from test_optimization import synthetic_tables

    builders = {
        "stack": synthetic_stack,
        "demo": synthetic_demo,
        "lattice": lambda: synthetic_lattice(args.n),
        "random": lambda: synthetic_random(args.n, args.m, args.asm_seed),
    }
    inputs = builders[args.synthetic]()
    print(f"Synthetic assembly '{args.synthetic}': {len(inputs.component_ids)} components ...", flush=True)
    aog = generate_aog(inputs, AoGConfig())
    if not aog.complete:
        raise SystemExit("The synthetic AoG root is not live (dead ends): try another --asm-seed or --n.")
    tables = synthetic_tables(aog, inputs)
    sequence = op.optimize(aog, tables, op.OptimizationConfig())
    meshes = {cid: box_mesh(lo, hi, cid) for cid, lo, hi in zip(inputs.component_ids, inputs.bbox_lo, inputs.bbox_hi)}
    bom = {label: {"label": label, "is_standard": str(bool(st)).lower(), "tool": tool or "", "group": group or ""}
           for label, st, tool, group in zip(aog.component_labels, aog.is_standard, tables.tool, tables.group)}
    illus_dir = (args.out or REPO_ROOT / "tests" / "synthetic" / args.synthetic) / "illustrations"

    def render(cfg):
        return il.render_illustrations(sequence, meshes, bom, illus_dir, cfg, title=f"synthetic {args.synthetic}")

    return f"synthetic:{args.synthetic}", sequence, bom, illus_dir, render


# ---------------------------------------------------------------------------
# Report and summary
# ---------------------------------------------------------------------------


def short_names(labels: list[str]) -> dict[str, str]:
    import os

    prefix = ""
    if len(labels) > 1:
        common = os.path.commonprefix(list(labels))
        cut = max((common.rfind(ch) for ch in "_- ."), default=-1)
        prefix = common[: cut + 1] if cut >= 0 else ""
    out = {}
    for label in labels:
        name = label[len(prefix):] or label
        out[label] = name if len(name) <= 24 else "..." + name[-21:]
    return out


def build_report(mode: str, steps: list[dict], result, config: IllustrationConfig, checks: list[dict]) -> dict:
    return {
        "mode": mode,
        "config": {k: v for k, v in asdict(config).items() if k != "force"},
        "config_hash": result.config_hash,
        "global_view": list(result.global_view),
        "manual": result.manual_path,
        "stats": result.stats,
        "steps": [{
            "step": f["step"], "component": f["component_label"], "is_standard": f["is_standard"],
            "png": f["png"], "view": f["view"], "view_kind": f["view_kind"], "visible": f["visible"],
            "xray": f["xray"], "vis_end": f["vis_end"], "vis_start": f["vis_start"],
            "trajectory_px_frac": f["trajectory_px_frac"], "footnote": f["footnote"],
            "warnings": f["warnings"],
        } for f in result.frames],
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }


def print_summary(mode: str, steps: list[dict], result, config: IllustrationConfig, checks: list[dict],
                  bom: dict, elapsed: float, report_path: Path) -> None:
    short = short_names([f["component_label"] for f in result.frames])
    print()
    print(f"Illustration Generator ({mode})")
    print(f"  image {config.image_size[0]} x {config.image_size[1]} px at {config.dpi} dpi; search "
          f"{len(config.azimuths)} azimuths x {len(config.elevations)} elevations, default view "
          f"az {config.default_view[0]:g} el {config.default_view[1]:g}")
    print(f"  visibility: part visible at end >= {config.vis_min_end:g}, at start >= {config.vis_min_start:g}, "
          f"trajectory >= {config.min_trajectory_px_frac:g} of the shorter side")
    print(f"  global view: az {result.global_view[0]:g}, el {result.global_view[1]:g}")
    print()
    print(f"  {'step':>4}  {'part':<24} {'tool':<10} {'view':>11} {'how':<9} {'end':>5} {'start':>5} {'traj':>5}  flags")
    for st, f in zip(steps, result.frames):
        tool = (bom.get(f["component_label"], {}).get("tool") or "").strip() or "by hand"
        view = f"{f['view']['azimuth']:g}/{f['view']['elevation']:g}"
        flags = []
        if f["xray"]:
            flags.append("x-ray")
        if not f["visible"]:
            flags.append("NOT VISIBLE")
        if f.get("reorient"):
            flags.append("re-orient")
        if f.get("hold"):
            flags.append("hold")
        if f["is_standard"]:
            flags.append("standard")
        print(f"  {f['step']:>4}  {short[f['component_label']]:<24} {tool:<10} {view:>11} "
              f"{f['view_kind']:<9} {f['vis_end']:>5.2f} {f['vis_start']:>5.2f} {f['trajectory_px_frac']:>5.2f}  "
              f"{', '.join(flags)}")
        for w in f["warnings"]:
            print(f"        warning: {w}")
    print()
    s = result.stats
    parts = [f"{k[:-2].replace('_', ' ')} {v:.1f} s" for k, v in s.items() if k.endswith("_s") and isinstance(v, (int, float))]
    print(f"  stats: {', '.join(parts)}" if parts else "  stats: (loaded from cache)")
    print()
    print("  checks:")
    for c in checks:
        print(f"    {c['id']} {c['name']:<11} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    print()
    print(f"  manual:  {result.manual_path}")
    print(f"  frames:  {Path(result.frames[0]['png']).parent}")
    print(f"  report:  {report_path}")
    print(f"  total {elapsed:.1f} s; all checks passed: {all(c['passed'] for c in checks)}")


def main(args: argparse.Namespace) -> int:
    """Pipeline: build the optimized sequence (real: preprocessing -> interference -> AoG -> optimization, all
    cached; synthetic: hand-built assembly drawn as boxes) -> validate the settings -> generate_illustrations
    (view search, rasterised frames, manual.pdf) -> optional second call to check the cache -> checks V1-V8 ->
    report and summary."""
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    if args.quiet:
        logging.getLogger("aap.illustration").setLevel(logging.WARNING)
    config = build_config(args)
    try:
        il.validate_config(config)  # fail before any pipeline work
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2
    t0 = time.perf_counter()

    if args.synthetic:
        mode, sequence, bom, illus_dir, render = run_synthetic(args, config)
    else:
        mode, sequence, bom, illus_dir, render = run_real(args, config)
    steps = il._steps_of(sequence)

    result = render(config)
    again, again_s = None, 0.0
    if args.check_cache:
        t1 = time.perf_counter()
        again = render(replace(config, force=False))
        again_s = time.perf_counter() - t1
    checks = run_checks(result, steps, bom, config, again, again_s)

    illus_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(mode, steps, result, config, checks)
    report_path = illus_dir / "script_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print_summary(mode, steps, result, config, checks, bom, time.perf_counter() - t0, report_path)
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Illustration Generator and check its output.")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="[step_path bom_path out_dir] (default: industrial_fastner)")
    parser.add_argument("--synthetic", choices=["stack", "demo", "lattice", "random"],
                        help="use a hand-built assembly drawn as boxes instead of the STEP pipeline (no FreeCAD needed)")
    parser.add_argument("--n", type=int, default=5, help="movable parts for lattice/random (default 5)")
    parser.add_argument("--m", type=int, default=2, help="standards for random (default 2)")
    parser.add_argument("--asm-seed", type=int, default=1, help="seed of the random synthetic assembly (default 1)")
    parser.add_argument("--out", type=Path, help="output folder for synthetic runs (default tests/synthetic/<name>)")
    parser.add_argument("--step", type=float, default=DEFAULT_H,
                        help=f"collision step h in mm for the real run (default {DEFAULT_H}, "
                             "calibrated for industrial_fastner only)")
    parser.add_argument("--force-interference", action="store_true", help="recompute the interference matrix")
    parser.add_argument("--force-aog", action="store_true", help="regenerate the AoG")
    parser.add_argument("--force-optimization", action="store_true", help="recompute the optimization result")
    parser.add_argument("--force", action="store_true", help="re-render, ignoring the illustration caches")
    parser.add_argument("--size", metavar="WxH", help="frame size in px (default 1600x1200)")
    parser.add_argument("--dpi", type=int, help="PNG and PDF resolution (default 200)")
    parser.add_argument("--view", nargs=2, type=float, metavar=("AZ", "EL"),
                        help="default view (azimuth, elevation in degrees) the global-view search starts from")
    parser.add_argument("--vis-min", type=float, metavar="F",
                        help="minimum visible fraction of the part at its start and end position (default 0.5)")
    parser.add_argument("--tess-tol", type=float, metavar="REL",
                        help="tessellation tolerance as a fraction of the assembly diagonal (default 0.002)")
    parser.add_argument("--no-gravity", action="store_true", help="do not draw the gravity pictogram")
    parser.add_argument("--no-hold-mark", action="store_true", help="no 'hold in place' mark on class 1 steps")
    parser.add_argument("--show-vector", action="store_true", help="add the displacement vector to the footnote")
    parser.add_argument("--title", help="manual title (default: the assembly name)")
    parser.add_argument("--check-cache", action="store_true",
                        help="call the module a second time and check that it returns the cached result")
    parser.add_argument("--quiet", action="store_true", help="silence the module's own INFO logging")
    sys.exit(main(parser.parse_args()))
