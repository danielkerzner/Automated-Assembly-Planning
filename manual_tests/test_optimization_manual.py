"""Manual smoke test for Optimization (src/optimization.py).

Counterpart of manual_tests/test_preprocessing_manual.py, manual_tests/test_interference_manual.py and
manual_tests/test_aog_manual.py. Not an acceptance suite -- those are the O1-O15 tests in
automatic_tests/test_optimization.py. This
script runs the optimizer end to end on one assembly and dumps the result in a form you can read by
eye:

  <out_dir>/optimization/
    optimization_report.json    this script's readable report (see below)
    optimization_result.json    the module's own result (written by optimize_sequence, real runs only)
    assembly_sequence.txt       the sequence as text (written by the module, real runs only)
    term_tables.json, contact_patches.json, stability_classes.json
                                cached geometry / stability, written by the module (real runs only)

What it prints and writes:
  - the configuration: which criteria are on, and the weights actually used (renormalized over the
    enabled criteria, ratios kept)
  - the assembly sequence, in assembly order: part, insertion direction, travel, gravity direction,
    stability class, and the BOM tool / group where those criteria are on
  - the cost breakdown per criterion: raw value, normalized value in [0, 1], weight, weighted value
  - sequence statistics: direction changes, gravity changes, class-1 insertions, group switches,
    tool changes between standers (information only, not scored here)
  - run statistics: greedy path cost vs the ACO result, iterations, stop reason, distinct paths
  - consistency checks that do not use the optimizer's own cost code:
      K1 weights        enabled criteria only, sum to 1, default ratios kept (unless overridden)
      K2 path           a valid root-to-terminal walk over live nodes; every part exactly once; each
                        chosen removal direction is one of the edge's free directions
      K3 order          steps are the reverse of the walk, insertion direction = -removal direction,
                        assembly step t = number of parts present
      K4 cost           total = sum of weighted terms; every normalized term in [0, 1]
      K5 direction      the direction-change cost recomputed from the step directions
      K6 travel         the raw travel value recomputed from the step travel distances
      K7 stability      class 1/2 and a gravity direction on every step; the first part is class 2
  - optionally (--ablation) a leave-one-out table: the sequence found with each criterion switched off

Two kinds of run:

  Real assembly (default). Reuses the cached Interference matrix and the cached AoG (--force-interference
  and --force-aog recompute them), reads the BOM `tool` and `group` columns, extracts contact patches
  with FreeCAD (cached) and caches the optimization result. Needs FreeCAD (environmentAAP).

      python manual_tests/test_optimization_manual.py

  Synthetic assembly (no FreeCAD, no STEP file, nothing cached): a hand-built assembly from
  test_aog_manual.py, with geometry derived from its bounding boxes (touching faces become contact
  patches; interpenetrating parts such as screws rest on the overlap), made-up tools (standers alternate
  tool1/tool2) and groups (movable parts alone, all standers together).

      python manual_tests/test_optimization_manual.py --synthetic demo
      python manual_tests/test_optimization_manual.py --synthetic lattice --n 6
      python manual_tests/test_optimization_manual.py --synthetic random --n 6 --m 2 --asm-seed 3

Switch criteria off and tune the run from the command line, for example:

      python manual_tests/test_optimization_manual.py --no-tool-change --no-grouping
      python manual_tests/test_optimization_manual.py --weights stability=0.5 travel=0.05
      python manual_tests/test_optimization_manual.py --no-stability --no-gravity-change --seed 5 --beta 2
      python manual_tests/test_optimization_manual.py --synthetic demo --ablation

Disabling --no-stability also requires --no-gravity-change.
The exit code is 0 when every check passes, 1 otherwise.

Expected result: checks K1-K7 print PASS and the exit code is 0 (2 for an invalid configuration, before any work).
The console shows the weights used, the assembly sequence (every part exactly once, the first one class 2), the
cost breakdown per criterion with total = sum of the weighted terms, and the greedy cost against the ACO cost.
The same seed gives the same sequence. With --ablation a table
shows how the sequence and cost change when each criterion is switched off in turn.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the synthetic builders in test_aog_manual.py

import numpy as np  # noqa: E402

import optimization as op  # noqa: E402
from optimization import OptimizationConfig, TermTables  # noqa: E402

DEFAULT_STEP = REPO_ROOT / "data" / "industrial_fastner" / "industrial_fastner.stp"
DEFAULT_BOM = REPO_ROOT / "data" / "industrial_fastner" / "generated_bom.csv"
DEFAULT_OUT = REPO_ROOT / "output" / "industrial_fastner"
DEFAULT_H = 5.0

TOL = 1e-6
SYNTHETIC_CONTACT_TOL = 0.1  # mm, same as the AoG contact threshold

CLI_FLAGS = {  # --no-<name> switches the criterion off
    "travel": "--no-travel",
    "direction_change": "--no-direction-change",
    "tool_change": "--no-tool-change",
    "size_contacts": "--no-size-contacts",
    "grouping": "--no-grouping",
    "stability": "--no-stability",
    "gravity_change": "--no-gravity-change",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _vec(v) -> list[float]:
    return [round(float(c), 6) + 0.0 for c in v]


def _fmt_vec(v) -> str:
    return op._fmt_vec(v)


def short_names(labels: list[str]) -> dict[str, str]:
    """label -> shorter display name: the prefix shared by all labels is dropped (back to a separator)."""
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


# ---------------------------------------------------------------------------
# Synthetic assembly: tables from the bounding boxes
# ---------------------------------------------------------------------------


def _box_patch(lo, hi, cog, i: int, j: int, tol: float):
    """Contact triangles between boxes i < j with normals pointing from j into i, or None."""
    olo = np.maximum(lo[i], lo[j])
    ohi = np.minimum(hi[i], hi[j])
    ext = ohi - olo
    a = int(np.argmin(ext))
    if ext[a] > 0:  # interpenetrating (a screw in a part): rest on the overlap, along z
        a = 2
        plane = olo[2]
        normal = np.array([0.0, 0.0, 1.0 if cog[i][2] >= cog[j][2] else -1.0])
    elif hi[i][a] <= lo[j][a] + tol:  # i below j, touching j's lower face
        plane = hi[i][a]
        normal = np.zeros(3)
        normal[a] = -1.0
    else:  # i above j
        plane = lo[i][a]
        normal = np.zeros(3)
        normal[a] = 1.0
    b, c = [x for x in range(3) if x != a]
    if ext[b] <= 0 or ext[c] <= 0:
        return None
    o = np.zeros(3)
    o[a], o[b], o[c] = plane, olo[b], olo[c]
    u = np.zeros(3)
    u[b] = ext[b]
    v = np.zeros(3)
    v[c] = ext[c]
    tri = np.array([[o, o + u, o + u + v], [o, o + u + v, o + v]])
    return tri, np.tile(normal, (2, 1))


def synthetic_tables(aog, inputs) -> TermTables:
    lo, hi = np.asarray(inputs.bbox_lo, float), np.asarray(inputs.bbox_hi, float)
    n = len(lo)
    size = hi - lo
    cog = (lo + hi) / 2
    contact = np.zeros((n, n), dtype=bool)
    patches = {}
    for i in range(n):
        for j in range(i + 1, n):
            near = (lo[i] - SYNTHETIC_CONTACT_TOL <= hi[j]) & (lo[j] - SYNTHETIC_CONTACT_TOL <= hi[i])
            if not near.all():
                continue
            contact[i, j] = contact[j, i] = True
            patch = _box_patch(lo, hi, cog, i, j, SYNTHETIC_CONTACT_TOL)
            if patch is not None:
                patches[(i, j)] = patch
    stander_idx = [c for c in range(n) if aog.is_stander[c]]
    tool = [None] * n
    for q, c in enumerate(stander_idx):
        tool[c] = "tool1" if q % 2 == 0 else "tool2"
    group = [f"{aog.component_labels[c]}" for c in range(n)]
    if len(stander_idx) > 1:
        for c in stander_idx:
            group[c] = "fasteners"
    return TermTables(
        component_ids=list(aog.component_ids),
        volume=size.prod(axis=1),
        area=2 * (size[:, 0] * size[:, 1] + size[:, 1] * size[:, 2] + size[:, 0] * size[:, 2]),
        contact=contact,
        cog=cog,
        tool=tool,
        group=group,
        pitch=[None] * n,
        has_tool_column=True,
        has_group_column=True,
        patches=patches,
        cache_key=str(inputs.cache_key),
    )


# ---------------------------------------------------------------------------
# Configuration from the command line
# ---------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> OptimizationConfig:
    kwargs: dict = {}
    for term, flag in CLI_FLAGS.items():
        if getattr(args, flag[2:].replace("-", "_")):
            kwargs[op.FLAG_OF[term]] = False
    if args.weights:
        weights = {}
        for item in args.weights:
            name, _, value = item.partition("=")
            if name not in op.TERMS or not value:
                raise SystemExit(f"--weights expects term=value with term in {list(op.TERMS)}, got {item!r}")
            weights[name] = float(value)
        kwargs["weights"] = weights
    for name in ("seed", "alpha", "beta", "rho", "n_iterations", "stagnation_limit"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    if args.ants is not None:
        kwargs["n_ants"] = args.ants
    if args.cost_to_go:
        kwargs["use_cost_to_go"] = True
    if args.support_region:
        kwargs["support_region"] = args.support_region
    if args.no_fall_test:
        kwargs["stability_fall_test"] = False
    if args.force:
        kwargs["force"] = True
    return OptimizationConfig(**kwargs)


# ---------------------------------------------------------------------------
# Consistency checks (independent of the optimizer's cost code)
# ---------------------------------------------------------------------------


def _check(cid: str, name: str, problems: list[str], ok_detail: str = "") -> dict:
    return {"id": cid, "name": name, "passed": not problems,
            "detail": ok_detail if not problems else "; ".join(problems[:6])}


def check_weights(config: OptimizationConfig, result) -> dict:
    problems = []
    enabled = [t for t in op.TERMS if getattr(config, op.FLAG_OF[t])]
    weights = {t: b["weight"] for t, b in result.cost_breakdown.items()}
    if sorted(weights) != sorted(enabled):
        problems.append(f"breakdown terms {sorted(weights)} differ from the enabled criteria {sorted(enabled)}")
    if abs(sum(weights.values()) - 1.0) > TOL:
        problems.append(f"weights sum to {sum(weights.values()):.9f}, not 1")
    if not config.weights:
        base = {t: op.DEFAULT_WEIGHTS[t] for t in enabled}
        scale = sum(base.values())
        for t in enabled:
            if t in weights and abs(weights[t] - base[t] / scale) > TOL:
                problems.append(f"{t}: weight {weights[t]:.6f} is not the default ratio {base[t] / scale:.6f}")
    return _check("K1", "weights", problems, f"{len(enabled)} criteria on, weights sum to 1")


def check_path(aog, result) -> dict:
    problems = []
    N = aog.n_components
    edges = result.edge_path
    if len(edges) != N:
        problems.append(f"walk has {len(edges)} edges, expected {N}")
        return _check("K2", "path", problems)
    v = aog.root
    seen = set()
    if not aog.live[v]:
        problems.append("root is not live")
    for pos, e in enumerate(edges):
        if int(aog.edge_src[e]) != v:
            problems.append(f"edge {e} at walk position {pos} does not start at the current node {v}")
            break
        c = int(aog.edge_comp[e])
        if c in seen:
            problems.append(f"component {c} removed twice")
        seen.add(c)
        k = result.removal_directions[pos]
        if k not in set(int(x) for x in aog.edge_direction_indices(e)):
            problems.append(f"walk position {pos}: direction {k} is not free on edge {e}")
        v = int(aog.edge_dst[e])
        if not aog.live[v]:
            problems.append(f"walk position {pos} enters the dead node {v}")
    if aog.terminal is not None and v != aog.terminal:
        problems.append(f"walk ends at node {v}, not the terminal {aog.terminal}")
    if len(seen) != N:
        problems.append(f"{len(seen)} distinct components removed, expected {N}")
    return _check("K2", "path", problems, f"{N} edges, live nodes only, every part once, directions free")


def check_order(aog, result) -> dict:
    problems = []
    N = aog.n_components
    steps = result.steps
    if len(steps) != N:
        return _check("K3", "order", [f"{len(steps)} steps, expected {N}"])
    for s, step in enumerate(steps):
        e = result.edge_path[N - 1 - s]
        k = result.removal_directions[N - 1 - s]
        if step["edge"] != e or step["component"] != int(aog.edge_comp[e]):
            problems.append(f"step {s + 1} is not walk edge {N - 1 - s} reversed")
        if step["removal_direction_index"] != k or step["direction_index"] != int(aog.neg[k]):
            problems.append(f"step {s + 1}: insertion direction is not the negative of the removal direction")
        present = int(aog.node_level[int(aog.edge_src[e])])  # components removed before this edge
        if step["step"] != N - present or step["step"] != s + 1:
            problems.append(f"step {s + 1}: expected t = {N - present}")
    return _check("K3", "order", problems, "reverse of the walk, insertion = -removal, t = parts present")


def check_cost(result) -> dict:
    problems = []
    total = sum(b["weighted"] for b in result.cost_breakdown.values())
    if abs(total - result.total_cost) > TOL:
        problems.append(f"sum of weighted terms {total:.9f} != total cost {result.total_cost:.9f}")
    for t, b in result.cost_breakdown.items():
        if not (-TOL <= b["normalized"] <= 1 + TOL):
            problems.append(f"{t}: normalized value {b['normalized']:.6f} outside [0, 1]")
        if abs(b["weight"] * b["normalized"] - b["weighted"]) > TOL:
            problems.append(f"{t}: weighted != weight x normalized")
    return _check("K4", "cost", problems, f"total {result.total_cost:.4f} = sum of weighted terms, terms in [0, 1]")


def check_direction(aog, config, result) -> dict:
    if "direction_change" not in result.cost_breakdown:
        return _check("K5", "direction", [], "criterion off, skipped")
    idx = [s["direction_index"] for s in result.steps]
    total = 0.0
    for prev, cur in zip(idx[:-1], idx[1:]):
        if cur == prev:
            continue
        total += config.dir_change_axis_ratio if cur == int(aog.neg[prev]) else 1.0
    expected = total / max(len(idx) - 1, 1)
    got = result.cost_breakdown["direction_change"]["normalized"]
    problems = [] if abs(expected - got) <= TOL else [f"recomputed {expected:.6f}, module says {got:.6f}"]
    return _check("K5", "direction", problems, f"normalized direction-change cost {got:.4f} recomputed")


def check_travel(result) -> dict:
    if "travel" not in result.cost_breakdown:
        return _check("K6", "travel", [], "criterion off, skipped")
    expected = sum(s["travel"] for s in result.steps)
    got = result.cost_breakdown["travel"]["raw"]
    problems = [] if abs(expected - got) <= 1e-6 * max(1.0, abs(expected)) else [
        f"sum of step travel {expected:.6f}, module raw value {got:.6f}"]
    return _check("K6", "travel", problems, f"raw travel {got:.2f} mm = sum of the step distances")


def check_stability(config, result) -> dict:
    if "stability" not in result.cost_breakdown:
        return _check("K7", "stability", [], "criterion off, skipped")
    problems = []
    G = len(config.gravity_dirs)
    for s in result.steps:
        if s["stability_class"] not in (1, 2):
            problems.append(f"step {s['step']}: class {s['stability_class']}")
        if s["gravity_index"] is None or not 0 <= s["gravity_index"] < G:
            problems.append(f"step {s['step']}: no valid gravity direction")
    if result.steps and result.steps[0]["stability_class"] != 2:
        problems.append("the first inserted part must be class 2 (nothing to support)")
    return _check("K7", "stability", problems, "class 1/2 and a gravity direction on every step, first part class 2")


def run_checks(aog, config: OptimizationConfig, result) -> list[dict]:
    return [check_weights(config, result), check_path(aog, result), check_order(aog, result),
            check_cost(result), check_direction(aog, config, result), check_travel(result),
            check_stability(config, result)]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def sequence_stats(aog, config: OptimizationConfig, result, tool: list, group: list) -> dict:
    steps = result.steps
    idx = [s["direction_index"] for s in steps]
    same_axis = sum(1 for a, b in zip(idx[:-1], idx[1:]) if b != a and b == int(aog.neg[a]))
    other_axis = sum(1 for a, b in zip(idx[:-1], idx[1:]) if b != a and b != int(aog.neg[a]))
    gravity = [s["gravity_index"] for s in steps if s["gravity_index"] is not None]
    comps = [s["component"] for s in steps]
    stander_tools = [tool[c] for c in comps if aog.is_stander[c]]
    # group switches: consecutive steps in different groups (a lower bound is the number of groups - 1)
    groups = [group[c] for c in comps]
    return {
        "direction_changes_same_axis": same_axis,
        "direction_changes_other_axis": other_axis,
        "gravity_changes": sum(1 for a, b in zip(gravity[:-1], gravity[1:]) if a != b) if gravity else None,
        "class1_insertions": sum(1 for s in steps if s["stability_class"] == 1) if gravity else None,
        "group_switches": sum(1 for a, b in zip(groups[:-1], groups[1:]) if a != b),
        "number_of_groups": len(set(groups)),
        "tool_changes_between_standers": sum(1 for a, b in zip(stander_tools[:-1], stander_tools[1:]) if a != b),
        "total_travel_mm": round(sum(s["travel"] for s in steps), 3),
    }


def build_report(mode: str, aog, config: OptimizationConfig, result, checks: list[dict], tool: list,
                 group: list, ablation: list[dict] | None) -> dict:
    history = {k: v for k, v in result.history.items() if k != "stability_cache"}
    cfg = asdict(config)
    cfg["gravity_dirs"] = [list(map(float, g)) for g in config.gravity_dirs]
    return {
        "mode": mode,
        "component_count": aog.n_components,
        "enabled_criteria": [t for t in op.TERMS if getattr(config, op.FLAG_OF[t])],
        "weights": {t: b["weight"] for t, b in result.cost_breakdown.items()},
        "config": cfg,
        "total_cost": result.total_cost,
        "cost_breakdown": result.cost_breakdown,
        "steps": [
            {**s, "tool": tool[s["component"]], "group": group[s["component"]]} for s in result.steps
        ],
        "sequence_stats": sequence_stats(aog, config, result, tool, group),
        "run": history,
        "checks": checks,
        "ablation": ablation,
        "config_hash": result.config_hash,
    }


def print_summary(aog, config: OptimizationConfig, result, report: dict, checks: list[dict],
                  elapsed: float, report_path: Path, ablation: list[dict] | None) -> None:
    names = short_names(list(aog.component_labels))
    width = max(len(names[s["component_label"]]) for s in result.steps)
    on = report["enabled_criteria"]
    off = [t for t in op.TERMS if t not in on]
    print()
    print(f"Components: {aog.n_components}   AoG: {aog.n_nodes} states, {aog.n_edges} moves")
    print("Criteria on:  " + ", ".join(f"{t} ({report['weights'][t]:.3f})" for t in on))
    if off:
        print("Criteria off: " + ", ".join(off))
    print(f"ACO: alpha {config.alpha}, beta {config.beta}, rho {config.rho}, seed {config.seed}, "
          f"cost-to-go {'on' if config.use_cost_to_go else 'off'}, support region {config.support_region}")

    print()
    print("Assembly sequence (insert in this order):")
    show_gravity = "stability" in on
    show_tool = "tool_change" in on
    show_group = "grouping" in on
    head = f"  {'step':>4}  {'part':<{width}}  {'insertion dir':<16} {'travel mm':>9}"
    if show_gravity:
        head += f"  {'gravity':<14} {'class':>5}"
    if show_tool:
        head += f"  {'tool':<10}"
    if show_group:
        head += f"  {'group':<14}"
    print(head)
    for s, row in zip(result.steps, report["steps"]):
        line = (f"  {s['step']:>4}  {names[s['component_label']]:<{width}}  {_fmt_vec(s['direction']):<16} "
                f"{s['travel']:>9.2f}")
        if show_gravity:
            line += f"  {_fmt_vec(s['gravity']):<14} {s['stability_class']:>5}"
        if show_tool:
            line += f"  {(row['tool'] or '-'):<10}"
        if show_group:
            line += f"  {(row['group'] or '-'):<14}"
        print(line + ("   [stander]" if s["is_stander"] else ""))

    print()
    print("Cost breakdown (normalized in [0, 1], times weight):")
    for t, b in result.cost_breakdown.items():
        print(f"  {t:<17} raw {b['raw']:>10.3f}   normalized {b['normalized']:.4f}   weight {b['weight']:.4f}   "
              f"weighted {b['weighted']:.4f}")
    print(f"  {'TOTAL':<17} {'':>10}                                                       {result.total_cost:.4f}")

    st = report["sequence_stats"]
    print()
    print("Sequence statistics (information only):")
    print(f"  direction changes: {st['direction_changes_same_axis']} same axis, "
          f"{st['direction_changes_other_axis']} other axis   total travel {st['total_travel_mm']} mm")
    if st["gravity_changes"] is not None:
        print(f"  gravity changes: {st['gravity_changes']}   class-1 insertions: {st['class1_insertions']}")
    print(f"  groups: {st['number_of_groups']}, group switches {st['group_switches']} "
          f"(at least {st['number_of_groups'] - 1} are unavoidable)   "
          f"tool changes between standers: {st['tool_changes_between_standers']}")

    h = result.history
    print()
    print(f"Run: greedy path cost {h['greedy_cost']:.4f} -> ACO {result.total_cost:.4f}; "
          f"{h['iterations']} iterations x {h['n_ants']} ants, stopped by {h['stopped_by']}, "
          f"{h['distinct_paths_evaluated']} distinct paths, {h['wall_time_s']:.2f} s")

    print()
    print("Checks:")
    for c in checks:
        print(f"  {c['id']} {'PASS' if c['passed'] else 'FAIL'}  {c['name']:<10} {c['detail']}")

    if ablation:
        print()
        print("Leave-one-out (total cost is on each run's own scale, so compare the sequences, not the costs):")
        for row in ablation:
            if row.get("error"):
                print(f"  {row['name']:<26} {row['error']}")
            else:
                order = " > ".join(names[x] for x in row["order"])
                print(f"  {row['name']:<26} cost {row['total_cost']:.4f}   {order}")

    print()
    print(f"Time: {elapsed:.1f} s")
    print(f"Report written to {report_path}")


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def leave_one_out(run, config: OptimizationConfig, labels: list[str]) -> list[dict]:
    rows = []
    on = [t for t in op.TERMS if getattr(config, op.FLAG_OF[t])]
    for t in on:
        extra = {op.FLAG_OF[t]: False}
        if t == "stability":
            extra[op.FLAG_OF["gravity_change"]] = False  # gravity change needs stability
        cfg = replace(config, **extra)
        name = "without " + t + (" + gravity_change" if t == "stability" and "gravity_change" in on else "")
        if not any(getattr(cfg, op.FLAG_OF[x]) for x in op.TERMS):
            rows.append({"name": name, "error": "no criterion left, skipped"})
            continue
        try:
            res = run(cfg)
        except ValueError as exc:
            rows.append({"name": name, "error": f"skipped: {exc}"})
            continue
        rows.append({"name": name, "total_cost": res.total_cost,
                     "order": [s["component_label"] for s in res.steps]})
    return rows


def run_real(args: argparse.Namespace, config: OptimizationConfig):
    import freecad  # noqa: F401  (sets up the FreeCAD lib path in the conda env; must come first)

    from aog_generator import AoGConfig, build_aog
    from collision import CollisionConfig
    from interference_identification import identify_interferences
    from preprocessing import run_preprocessing

    step, bom, out = args.paths if args.paths else (DEFAULT_STEP, DEFAULT_BOM, DEFAULT_OUT)
    out = Path(out)
    cache_dir = out / "optimization"
    print(f"Preprocessing {Path(step).name} ...", flush=True)
    prep = run_preprocessing(step, bom)
    print(f"Interference Identification (h = {args.step}; cached unless --force-interference) ...", flush=True)
    interference = identify_interferences(prep, CollisionConfig(step=args.step), out / "interference",
                                          force=args.force_interference)
    print("AoG generation (cached unless --force-aog) ...", flush=True)
    aog = build_aog(interference, prep, AoGConfig(), cache_dir=out / "aog", force=args.force_aog)
    if not aog.complete:
        raise SystemExit("The AoG root is not live: there is no complete disassembly sequence to optimize.")

    columns, rows, _ = op._read_bom(prep.bom_path)
    tool = [rows.get(label, {}).get("tool") or None for label in aog.component_labels]
    group = [rows.get(label, {}).get("group") or None for label in aog.component_labels]

    def run(cfg):
        return op.optimize_sequence(aog, interference, prep, cfg, cache_dir=cache_dir)

    return aog, run, tool, group, cache_dir, ("tool" in columns, "group" in columns)


def run_synthetic(args: argparse.Namespace):
    from aog_generator import AoGConfig, generate_aog
    from test_aog import synthetic_demo, synthetic_lattice, synthetic_random, synthetic_stack

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
        raise SystemExit("The synthetic AoG root is not live (dead ends): try another --seed or --n.")
    tables = synthetic_tables(aog, inputs)

    def run(cfg):
        return op.optimize(aog, tables, cfg)

    cache_dir = (args.out or REPO_ROOT / "tests" / "synthetic" / args.synthetic) / "optimization"
    return aog, run, tables.tool, tables.group, cache_dir, (tables.has_tool_column, tables.has_group_column)


def main(args: argparse.Namespace) -> int:
    """Pipeline: get the AoG and the BOM tool / group data (real: preprocessing -> interference -> AoG ->
    contact patches, all cached; synthetic: hand-built assembly) -> validate the criteria -> optional leave-one-out
    ablation -> run the ACO optimizer (optimize_sequence) -> checks K1-K7 -> report and summary."""
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    if args.quiet:
        logging.getLogger("aap.optimization").setLevel(logging.WARNING)
    config = build_config(args)
    t0 = time.perf_counter()

    if args.synthetic:
        aog, run, tool, group, cache_dir, bom_columns = run_synthetic(args)
        mode = f"synthetic:{args.synthetic}"
    else:
        aog, run, tool, group, cache_dir, bom_columns = run_real(args, config)
        mode = "real"
    try:
        op.validate_config(config, *bom_columns)  # fail before any optimization work
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2

    # The ablation runs first so that, for a real run, the cached result on disk is the main configuration's.
    ablation = leave_one_out(run, config, aog.component_labels) if args.ablation else None
    result = run(config)
    checks = run_checks(aog, config, result)

    cache_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(mode, aog, config, result, checks, tool, group, ablation)
    report_path = cache_dir / "optimization_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print_summary(aog, config, result, report, checks, time.perf_counter() - t0, report_path, ablation)
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the optimizer and dump a readable report.")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="[step_path bom_path out_dir] (default: industrial_fastner)")
    parser.add_argument("--synthetic", choices=["stack", "demo", "lattice", "random"],
                        help="use a hand-built assembly instead of the STEP pipeline (no FreeCAD needed)")
    parser.add_argument("--n", type=int, default=5, help="movable parts for lattice/random (default 5)")
    parser.add_argument("--m", type=int, default=2, help="standers for random (default 2)")
    parser.add_argument("--asm-seed", type=int, default=1, help="seed of the random synthetic assembly (default 1)")
    parser.add_argument("--out", type=Path, help="output folder for synthetic runs (default tests/synthetic/<name>)")
    parser.add_argument("--step", type=float, default=DEFAULT_H,
                        help=f"collision step h in mm for the real run (default {DEFAULT_H}, "
                             "calibrated for industrial_fastner only)")
    parser.add_argument("--force-interference", action="store_true", help="recompute the interference matrix")
    parser.add_argument("--force-aog", action="store_true", help="regenerate the AoG")
    parser.add_argument("--force", action="store_true", help="recompute cached optimization tables and result")
    for term, flag in CLI_FLAGS.items():
        parser.add_argument(flag, action="store_true", help=f"switch the {term} criterion off")
    parser.add_argument("--weights", nargs="+", metavar="TERM=VALUE",
                        help=f"override weights, e.g. stability=0.5 (terms: {', '.join(op.TERMS)}); "
                             "the enabled weights are renormalized")
    parser.add_argument("--seed", type=int, help="ACO random seed (default: the config default)")
    parser.add_argument("--ants", type=int, help="number of ants (default max(2N, 10))")
    parser.add_argument("--iterations", dest="n_iterations", type=int, help="maximum iterations")
    parser.add_argument("--stagnation", dest="stagnation_limit", type=int,
                        help="stop after this many iterations without improvement")
    parser.add_argument("--alpha", type=float, help="pheromone exponent")
    parser.add_argument("--beta", type=float, help="heuristic exponent")
    parser.add_argument("--rho", type=float, help="evaporation rate in (0, 1)")
    parser.add_argument("--cost-to-go", action="store_true",
                        help="add the exact remaining static cost to the heuristic (off by default)")
    parser.add_argument("--support-region", choices=["hull", "union"], help="stability support region")
    parser.add_argument("--no-fall-test", action="store_true", help="skip the 'it could fall' part of class 1")
    parser.add_argument("--ablation", action="store_true",
                        help="also run the optimizer with each enabled criterion switched off in turn")
    parser.add_argument("--quiet", action="store_true", help="silence the module's own INFO logging")
    sys.exit(main(parser.parse_args()))
