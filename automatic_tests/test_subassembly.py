"""Acceptance tests for sub-assemblies: Phase 1 (H1-H13, user-defined containers) and
Phase 2 (R2-R8, automatic sub-assemblies auto_<n> and locked levels). Modules under test: src/hierarchy.py (tree -> containers,
"processed" / "is_standard" BOM columns, size guard), src/levels.py (per-level orchestration, fusing, merge loop, locked levels),
src/main.py build_plan (plan.json), and merge_groups / sub_interference in src/interference_identification.py.

Standalone runner (pytest is not installed in environmentAAP). From repo root:

    python automatic_tests/test_subassembly.py                    # core group (seconds, no FreeCAD run)
    python automatic_tests/test_subassembly.py --tests H1 H5      # individual tests
    python automatic_tests/test_subassembly.py --groups core real # + FreeCAD runs on industrial_fastner /
                                                                  # lunar_motor_tire

Groups:
  core  hierarchy.py and levels.py on fake tree objects and stubbed pipeline modules; no FreeCAD run
        (H1, H4, H4b, H5, H6, H8, H9, H13).
  real  hierarchy/fusing/collision checks against the real STEP files: industrial_fastner (H1b, H10,
        H12) and lunar_bodylager (H1b, H2, H2b, H2c, H3, H6b, H7, H11). H6b and H12 stub the pipeline
        modules (fast orchestration/regression check only); H10 runs the real Optimization BOM-row
        lookup against industrial_fastner's cached interference/AoG (never writes to
        output/industrial_fastner/); the others run real fusing or collision.

Scratch output goes to automatic_tests/output/subassembly_test/; results to
automatic_tests/output/subassembly_results.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import freecad  # noqa: E402,F401  (bootstrap for the conda build)

import hierarchy as hierarchy_mod  # noqa: E402
import levels as levels_mod  # noqa: E402
import preprocessing as preprocessing_mod  # noqa: E402
import interference_identification as ii_mod  # noqa: E402
import aog_generator as ag_mod  # noqa: E402
import optimization as opt_mod  # noqa: E402
import illustration_generator as ig_mod  # noqa: E402
from hierarchy import LevelSizeError, build_hierarchy, check_level_sizes  # noqa: E402
from main import build_plan  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
SCRATCH = OUT_DIR / "subassembly_test"
FASTENER_STEP = REPO_ROOT / "data" / "industrial_fastner" / "industrial_fastner.stp"
FASTENER_BOM = REPO_ROOT / "data" / "industrial_fastner" / "generated_bom.csv"
LUNAR_STEP = REPO_ROOT / "data" / "lunar_bodylager" / "0002-A-0900-A06-bodylager_1.stp"
LUNAR_BOM = REPO_ROOT / "data" / "lunar_bodylager" / "generated_bom.csv"
NONLINEAR_STEP = REPO_ROOT / "data" / "nonlinearsimple" / "Assembly.stp"
NONLINEAR_BOM = REPO_ROOT / "data" / "nonlinearsimple" / "generated_bom.csv"


# ---------------------------------------------------------------------------
# Tiny test harness (same shape as test_aog.py)
# ---------------------------------------------------------------------------


class Recorder:
    """Collects PASS/FAIL checks, notes and errors per test id for the summary table and the JSON file."""

    def __init__(self):
        self.results: dict[str, dict] = {}
        self._current: str | None = None

    def start(self, test_id: str):
        self._current = test_id
        self.results[test_id] = {"passed": True, "checks": [], "notes": []}

    def check(self, ok: bool, label: str):
        entry = self.results[self._current]
        entry["checks"].append({"label": label, "passed": bool(ok)})
        if not ok:
            entry["passed"] = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}", flush=True)

    def note(self, text: str):
        self.results[self._current]["notes"].append(text)
        print(f"  note: {text}", flush=True)

    def error(self, exc: BaseException):
        entry = self.results[self._current]
        entry["passed"] = False
        entry["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"  [ERROR] {entry['error']}", flush=True)
        traceback.print_exc()


class LogCapture(logging.Handler):
    """[SUPP] Context manager that records the log records of one named logger (to assert that a warning was emitted)."""

    def __init__(self, logger_name: str):
        super().__init__(level=logging.INFO)
        self.logger_name = logger_name
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def __enter__(self):
        logging.getLogger(self.logger_name).addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger(self.logger_name).removeHandler(self)


def fresh_dir(name: str) -> Path:
    # [SUPP] Empty scratch folder under output/subassembly_test/.
    path = SCRATCH / name
    shutil.rmtree(path, ignore_errors=True)
    return path


# ---------------------------------------------------------------------------
# Fake tree objects (duck-typed: TypeId, Name, Label, Group)
# ---------------------------------------------------------------------------


class FakeObj:
    """[SUPP] Stand-in for a FreeCAD tree object (Name, Label, TypeId, Group)."""

    def __init__(self, name, label, type_id, group=None):
        self.Name = name
        self.Label = label
        self.TypeId = type_id
        self.Group = group or []


class FakeDoc:
    """[SUPP] Stand-in for a FreeCAD document: RootObjects plus the flat Objects list."""

    def __init__(self, roots):
        self.RootObjects = roots
        self.Objects = []
        seen: set[str] = set()
        stack = list(roots)
        while stack:
            obj = stack.pop()
            if obj.Name in seen:
                continue
            seen.add(obj.Name)
            self.Objects.append(obj)
            stack.extend(obj.Group)


def leaf_is(obj) -> bool:
    # [SUPP] Leaf predicate for the fake trees: a Part::Feature is a component.
    return obj.TypeId == "Part::Feature"


def flat_tree(n: int = 8) -> FakeDoc:
    # [SUPP] One root assembly with n leaves.
    leaves = [FakeObj(f"leaf{i}", f"Leaf{i}", "Part::Feature") for i in range(n)]
    root = FakeObj("root", "Assembly", "App::Part", group=leaves)
    return FakeDoc([root])


def nested_tree() -> FakeDoc:
    # [SUPP] Root with a leaf, a transparent group, and two containers (X: 2 leaves, Y: 1 leaf); drawn in the comment below.
    # root -> leafA, groupT(transparent) -> leafB, containerX -> leafC, leafD, containerY -> leafE
    leaf_a = FakeObj("leafA", "LeafA", "Part::Feature")
    leaf_b = FakeObj("leafB", "LeafB", "Part::Feature")
    group_t = FakeObj("groupT", "GroupT", "App::DocumentObjectGroup", group=[leaf_b])
    leaf_c = FakeObj("leafC", "LeafC", "Part::Feature")
    leaf_d = FakeObj("leafD", "LeafD", "Part::Feature")
    container_x = FakeObj("containerX", "ContainerX", "App::Part", group=[leaf_c, leaf_d])
    leaf_e = FakeObj("leafE", "LeafE", "Part::Feature")
    container_y = FakeObj("containerY", "ContainerY", "App::Part", group=[leaf_e])
    root = FakeObj("root", "Assembly", "App::Part", group=[leaf_a, group_t, container_x, container_y])
    return FakeDoc([root])


def virtual_root_tree() -> FakeDoc:
    # [SUPP] Two top-level leaves and no single App::Part root.
    return FakeDoc([FakeObj("leafA", "LeafA", "Part::Feature"), FakeObj("leafB", "LeafB", "Part::Feature")])


def link_tree() -> FakeDoc:
    # [SUPP] A tree containing an App::Link (unsupported).
    link = FakeObj("link1", "Link1", "App::Link")
    root = FakeObj("root", "Assembly", "App::Part", group=[link])
    return FakeDoc([root])


def two_same_product_tree() -> FakeDoc:
    # [SUPP] Two instances of the same sub-assembly product (labels Widget and Widget001).
    def widget(name, label):
        leaves = [FakeObj(f"{name}_l1", "L1", "Part::Feature"), FakeObj(f"{name}_l2", "L2", "Part::Feature")]
        return FakeObj(name, label, "App::Part", group=leaves)

    root = FakeObj("root", "Assembly", "App::Part", group=[widget("widgetA", "Widget"),
                                                            widget("widgetB", "Widget001")])
    return FakeDoc([root])


# ---------------------------------------------------------------------------
# Stubbed pipeline (H6, H8, H9, H13): isolates levels.solve_level's orchestration
# from the real geometry modules.
# ---------------------------------------------------------------------------


@contextmanager
def stub_pipeline(hier):
    """[SUPP] Patch every module solve_level calls into with FreeCAD-free fakes; yields the ordered list
    of level ids passed to the fake optimize_sequence (one entry per solved level, call order)."""
    calls: list[str] = []

    def fake_build_level_document(h, container_id, original_doc, cache_dir):
        return SimpleNamespace(Name=f"doc_{container_id}"), frozenset(), {}

    def fake_run_preprocessing_on_document(doc_L, bom_labels, config, *, step_path, bom_path,
                                           level_id, level_label, compound_ids, members,
                                           collision_config=None, bom_categories=None):
        c = hier.containers[level_id]
        return SimpleNamespace(level_id=level_id, level_label=level_label,
                               leaf_ids=list(c.leaf_ids), child_ids=list(c.child_ids),
                               step_path=step_path, bom_path=bom_path)

    def fake_identify_interferences(prep, collision_config, cache_dir, force=False):
        return SimpleNamespace(component_ids=[], component_labels=[],
                               free=np.zeros((0, 0, 0), dtype=bool), cache_key="stub")

    def fake_build_aog(interf, prep, config=None, cache_dir=None):
        return SimpleNamespace(complete=True, dead_ends=[])

    def fake_optimize_sequence(aog, interf, prep, config=None, cache_dir=None):
        calls.append(prep.level_id)
        steps = [{"component_id": c, "is_standard": False} for c in prep.leaf_ids + prep.child_ids]
        return SimpleNamespace(steps=steps, total_cost=1.0)

    def fake_generate_illustrations(*a, **k):
        return None

    patches = [
        (levels_mod, "build_level_document", fake_build_level_document),
        (preprocessing_mod, "run_preprocessing_on_document", fake_run_preprocessing_on_document),
        (ii_mod, "identify_interferences", fake_identify_interferences),
        (ag_mod, "build_aog", fake_build_aog),
        (opt_mod, "optimize_sequence", fake_optimize_sequence),
        (ig_mod, "generate_illustrations", fake_generate_illustrations),
    ]
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in patches]
    for mod, name, fn in patches:
        setattr(mod, name, fn)
    try:
        yield calls
    finally:
        for mod, name, fn in saved:
            setattr(mod, name, fn)


def solve(hier, container_id="__auto__", **kwargs):
    # [SUPP] levels.solve_level on the root (default) with dummy paths; the pipeline must be stubbed by the caller.
    if container_id == "__auto__":
        container_id = hier.root
    scratch = kwargs.pop("output_dir")
    return levels_mod.solve_level(
        hier, container_id, SimpleNamespace(Name="fakedoc"),
        step_path=Path("fake.stp"), bom_path=Path("fake_bom.csv"), bom_labels=set(),
        output_dir=scratch, **kwargs,
    )


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------


def test_h1(rec, args):
    """Tests: build_hierarchy on fake trees (src/hierarchy.py).
    What: a flat assembly gives a root with all leaves and no containers; a nested tree gives the root's direct leaves
          (a transparent group is flattened into its parent) and its containers, with the right leaves per container; a tree with
          no single top-level App::Part gets a virtual "__root__"; an App::Link raises NotImplementedError.
    """
    hier = build_hierarchy(flat_tree(), is_leaf=leaf_is)
    root = hier.containers[hier.root]
    rec.check(hier.root == "root" and len(root.leaf_ids) == 8 and not root.child_ids,
              f"flat: root + 8 leaves (got {len(root.leaf_ids)} leaves, {len(root.child_ids)} containers)")

    hier = build_hierarchy(nested_tree(), is_leaf=leaf_is)
    root = hier.containers[hier.root]
    rec.check(hier.root == "root" and set(root.leaf_ids) == {"leafA", "leafB"}
              and set(root.child_ids) == {"containerX", "containerY"},
              f"nested: root direct leaves {sorted(root.leaf_ids)} (transparent group flattened), "
              f"containers {sorted(root.child_ids)}")
    rec.check(hier.containers["containerX"].leaf_ids == ["leafC", "leafD"]
              and hier.containers["containerY"].leaf_ids == ["leafE"], "nested: container children correct")
    rec.check(hier.leaf_parent["leafB"] == "root", "transparent group type does not change the parent id")

    hier = build_hierarchy(virtual_root_tree(), is_leaf=leaf_is)
    rec.check(hier.root == "__root__" and set(hier.containers["__root__"].leaf_ids) == {"leafA", "leafB"},
              "no single top-level App::Part: virtual __root__ holds the top-level objects")

    try:
        build_hierarchy(link_tree(), is_leaf=leaf_is)
        rec.check(False, "App::Link raises NotImplementedError")
    except NotImplementedError:
        rec.check(True, "App::Link raises NotImplementedError")


def test_h4(rec, args):
    """Tests: the BOM column "processed" (build_hierarchy, bom_rows).
    What: blank defaults to true; explicit false marks the container as not processed (it is then fused into one part of its
          parent and gets no level of its own); false on a leaf row is ignored with a warning; the root is always processed
          (overriding it warns); an invalid text raises ValueError.
    """
    hier = build_hierarchy(nested_tree(), bom_rows=[{"label": "ContainerX", "processed": ""}], is_leaf=leaf_is)
    rec.check(hier.containers["containerX"].processed is True, "blank processed defaults to true")

    hier = build_hierarchy(nested_tree(), bom_rows=[{"label": "ContainerX", "processed": "false"}], is_leaf=leaf_is)
    rec.check(hier.containers["containerX"].processed is False, "explicit false (2 leaves) -> not processed, no child level")

    with LogCapture("aap.hierarchy") as cap:
        build_hierarchy(nested_tree(), bom_rows=[{"label": "LeafA", "processed": "false"}], is_leaf=leaf_is)
    rec.check(any("leaf row" in r.getMessage() for r in cap.records), "false on a leaf row is ignored with a warning")

    with LogCapture("aap.hierarchy") as cap:
        hier = build_hierarchy(nested_tree(), bom_rows=[{"label": "Assembly", "processed": "false"}], is_leaf=leaf_is)
    rec.check(hier.containers[hier.root].processed is True, "root is processed whatever its BOM value")
    rec.check(any("root" in r.getMessage() for r in cap.records), "overriding root processed warns")

    try:
        build_hierarchy(nested_tree(), bom_rows=[{"label": "ContainerX", "processed": "maybe"}], is_leaf=leaf_is)
        rec.check(False, "invalid processed text raises")
    except ValueError:
        rec.check(True, "invalid processed text raises")


def test_h4b(rec, args):
    """Tests: the BOM column "subassembly" and unmatched BOM labels (build_hierarchy).
    What: a "subassembly" value that matches the real parent passes, a mismatching one raises and names the row, a blank cell is
          not checked; a BOM label that matches no component only warns, unless it carries a non-blank "processed", which raises.
    """
    build_hierarchy(nested_tree(), bom_rows=[{"label": "ContainerX", "subassembly": "Assembly"}], is_leaf=leaf_is)
    rec.check(True, "matching subassembly column does not raise")

    try:
        build_hierarchy(nested_tree(), bom_rows=[{"label": "ContainerX", "subassembly": "WrongParent"}], is_leaf=leaf_is)
        rec.check(False, "mismatching subassembly column raises")
    except ValueError as exc:
        rec.check("ContainerX" in str(exc), f"mismatching subassembly column raises and names the row ({exc})")

    build_hierarchy(nested_tree(), bom_rows=[{"label": "LeafA", "subassembly": ""}], is_leaf=leaf_is)
    rec.check(True, "blank subassembly cell is not checked")

    with LogCapture("aap.hierarchy") as cap:
        build_hierarchy(nested_tree(), bom_rows=[{"label": "NoSuchLabel"}], is_leaf=leaf_is)
    rec.check(any("matches no component" in r.getMessage() for r in cap.records), "unmatched BOM label warns")

    try:
        build_hierarchy(nested_tree(), bom_rows=[{"label": "NoSuchLabel", "processed": "false"}], is_leaf=leaf_is)
        rec.check(False, "unmatched BOM label with a non-blank 'processed' raises")
    except ValueError as exc:
        rec.check("NoSuchLabel" in str(exc),
                  f"unmatched BOM label with a non-blank 'processed' raises and names the label ({exc})")


def test_h4c(rec, args):
    """A container CAN be a standard (added 2026-11-11, for
    lunar_motor_tire_complete's motor sub-assembly): is_standard=true forces processed=false (blank
    defaults to false here, not true); is_standard=true with an explicit processed=true is a
    ValueError (a standard subassembly is fused as one rigid component, it cannot also be
    recursively processed as its own level); is_standard=true on the root is also a ValueError."""
    hier = build_hierarchy(nested_tree(), bom_rows=[{"label": "ContainerX", "is_standard": "true"}], is_leaf=leaf_is)
    rec.check(hier.containers["containerX"].processed is False,
              "is_standard=true with 'processed' blank defaults processed to false")

    hier = build_hierarchy(nested_tree(),
                           bom_rows=[{"label": "ContainerX", "is_standard": "true", "processed": "false"}],
                           is_leaf=leaf_is)
    rec.check(hier.containers["containerX"].processed is False,
              "is_standard=true with an explicit processed=false does not raise")

    try:
        build_hierarchy(nested_tree(),
                        bom_rows=[{"label": "ContainerX", "is_standard": "true", "processed": "true"}],
                        is_leaf=leaf_is)
        rec.check(False, "is_standard=true with an explicit processed=true raises")
    except ValueError as exc:
        rec.check("ContainerX" in str(exc) and "is_standard" in str(exc) and "processed" in str(exc),
                  f"is_standard=true + processed=true raises and names the row ({exc})")

    try:
        build_hierarchy(nested_tree(), bom_rows=[{"label": "Assembly", "is_standard": "true"}], is_leaf=leaf_is)
        rec.check(False, "is_standard=true on the root raises")
    except ValueError as exc:
        rec.check("Assembly" in str(exc) and "is_standard" in str(exc),
                  f"is_standard=true on the root raises and names it ({exc})")

    build_hierarchy(nested_tree(), bom_rows=[{"label": "ContainerX", "is_standard": "false"}], is_leaf=leaf_is)
    rec.check(True, "is_standard=false on a container row does not raise")

    build_hierarchy(nested_tree(), bom_rows=[{"label": "ContainerX", "is_standard": ""}], is_leaf=leaf_is)
    rec.check(True, "blank is_standard on a container row does not raise")

    build_hierarchy(nested_tree(), bom_rows=[{"label": "LeafA", "is_standard": "true"}], is_leaf=leaf_is)
    rec.check(True, "is_standard=true on a leaf row is unaffected (unchanged leaf-standard path)")


def test_h5(rec, args):
    """Tests: check_level_sizes (size guard, max components per level).
    What: a root with 25 components and a processed container with 25 leaves both raise LevelSizeError naming the level; setting
          processed=false on the container removes it from the guard.
    """
    hier = build_hierarchy(flat_tree(25), is_leaf=leaf_is)
    try:
        check_level_sizes(hier, max_components=20)
        rec.check(False, "oversized root raises LevelSizeError")
    except LevelSizeError as exc:
        rec.check("root" in str(exc), f"oversized root raises and names it ({exc})")

    big_leaves = [FakeObj(f"c{i}", f"C{i}", "Part::Feature") for i in range(25)]
    big = FakeObj("big", "Big", "App::Part", group=big_leaves)
    other = FakeObj("other", "Other", "Part::Feature")
    doc2 = FakeDoc([FakeObj("root2", "Root2", "App::Part", group=[big, other])])

    hier2 = build_hierarchy(doc2, is_leaf=leaf_is)
    try:
        check_level_sizes(hier2, max_components=20)
        rec.check(False, "oversized processed container raises")
    except LevelSizeError as exc:
        rec.check("big" in str(exc), f"oversized container named ({exc})")

    hier3 = build_hierarchy(doc2, bom_rows=[{"label": "Big", "processed": "false"}], is_leaf=leaf_is)
    check_level_sizes(hier3, max_components=20)  # must not raise
    rec.check(True, "processed=false removes the oversized container from the guard")


def test_h6(rec, args):
    """Tests: levels.solve_level orchestration on a fake nested tree (stubbed pipeline).
    What: levels are solved depth-first (parent before its processed children, in tree order); the root level directory is the
          output dir and each child is flat under output_dir/levels/<id>; only processed children get a LevelPlan; the root sequence
          has one step per direct leaf and container.
    """
    hier = build_hierarchy(nested_tree(), is_leaf=leaf_is)
    scratch = fresh_dir("h6")
    with stub_pipeline(hier) as calls:
        plan = solve(hier, output_dir=scratch)

    def expected_dfs(cid):
        order = [cid]
        for child_id in hier.containers[cid].child_ids:
            if hier.containers[child_id].processed:
                order += expected_dfs(child_id)
        return order

    rec.check(calls == expected_dfs(hier.root), f"depth-first solve order: {calls} == {expected_dfs(hier.root)}")
    rec.check(plan.level_dir == scratch, f"root level_dir == output_dir ({plan.level_dir})")
    rec.check(set(plan.children) == {cid for cid in hier.containers[hier.root].child_ids
                                     if hier.containers[cid].processed},
              f"only processed children get a LevelPlan ({sorted(plan.children)})")
    for cid, child_plan in plan.children.items():
        rec.check(child_plan.level_dir == scratch / "levels" / cid,
                  f"child {cid} level_dir flat under output_dir/levels/{cid} ({child_plan.level_dir})")
    root = hier.containers[hier.root]
    rec.check(len(plan.sequence.steps) == len(root.leaf_ids) + len(root.child_ids),
              "root step count == direct leaves + containers")


def test_h6b(rec, args):
    """Orchestration on the real lunar_bodylager hierarchy (stub pipeline modules, no collision
    run): root resolves to 30 components (28 leaves + 2 containers); both child containers are
    processed=true here (unlike lunar_motor_tire's unprocessed motor), so both get solved and
    both get their own level directory (the real-geometry analog of H8's fake-tree check)."""
    doc, hier = lunar_doc_and_hierarchy()
    root = hier.containers[hier.root]
    rec.check(len(root.child_ids) == 2 and all(hier.containers[c].processed for c in root.child_ids),
              f"both root-level containers are processed=true (got {[(c, hier.containers[c].processed) for c in root.child_ids]})")

    scratch = fresh_dir("h6b")
    with stub_pipeline(hier) as calls:
        plan = solve(hier, output_dir=scratch)

    n_root_components = len(root.leaf_ids) + len(root.child_ids)
    rec.check(len(plan.sequence.steps) == n_root_components,
              f"root level resolves to {n_root_components} components (got {len(plan.sequence.steps)})")
    rec.check(set(plan.children) == set(root.child_ids),
              f"both processed containers recurse into their own level (got {sorted(plan.children)})")
    rec.check(set(root.child_ids) <= set(calls), f"both containers solved as their own level (calls: {calls})")

    for cid in root.child_ids:
        level_dir = scratch / "levels" / cid
        rec.check(level_dir.exists(), f"container {cid} got a level directory ({level_dir})")


def test_h8(rec, args):
    """Tests: two instances of the same sub-assembly product (stubbed pipeline + main.build_plan).
    What: both instances are solved as separate levels and appear in plan.json as distinct "subassembly" steps, each with its own
          sub_plan.
    """
    hier = build_hierarchy(two_same_product_tree(), is_leaf=leaf_is)
    scratch = fresh_dir("h8")
    with stub_pipeline(hier):
        plan = solve(hier, output_dir=scratch)
    rec.check(set(plan.children) == {"widgetA", "widgetB"}, f"both instances solved: {sorted(plan.children)}")

    plan_dict = build_plan(hier, plan, assembly_name="assy")
    sub_steps = [s for s in plan_dict["level"]["steps"] if s["kind"] == "subassembly"]
    rec.check({s["component_id"] for s in sub_steps} == {"widgetA", "widgetB"},
              "plan.json: both instances appear as distinct subassembly steps")
    rec.check(all(s["sub_plan"] is not None for s in sub_steps), "both instances carry a sub_plan")


def test_h9(rec, args):
    """Tests: plan.json structure (main.build_plan).
    What: the plan survives a json round trip; steps have kind "subassembly" or "part"; a processed container nests its own sub_plan
          with its container_id, an unprocessed one has processed=false and no sub_plan.
    """
    hier = build_hierarchy(nested_tree(), is_leaf=leaf_is)
    scratch = fresh_dir("h9")
    with stub_pipeline(hier):
        plan = solve(hier, output_dir=scratch)
    plan_dict = build_plan(hier, plan, assembly_name="assy")

    round_tripped = json.loads(json.dumps(plan_dict))
    rec.check(round_tripped == plan_dict, "plan.json round-trips through json.dumps/loads")

    kinds = {s["component_id"]: s["kind"] for s in plan_dict["level"]["steps"]}
    rec.check(kinds.get("containerX") == "subassembly" and kinds.get("leafA") == "part",
              f"kind values: containerX=subassembly, leafA=part (got {kinds})")

    cx = next(s for s in plan_dict["level"]["steps"] if s["component_id"] == "containerX")
    rec.check(cx["sub_plan"] is not None and cx["sub_plan"]["container_id"] == "containerX",
              "processed subassembly (containerX, 2 leaves) nests its own sub_plan")

    cy = next(s for s in plan_dict["level"]["steps"] if s["component_id"] == "containerY")
    rec.check(cy["processed"] is False and cy["sub_plan"] is None,
              "unprocessed subassembly (containerY, 1 leaf, trivial rule) has no sub_plan")


def test_h13(rec, args):
    """Tests: reporting of inseparable pairs in levels.solve_level.
    What: with a fake interference result in which two parts have no free direction against each other, the level plan records
          that pair in inseparable_pairs and logs an "Inseparable" warning.
    """
    leaves = [FakeObj(f"p{i}", f"Part{i}", "Part::Feature") for i in range(1, 4)]
    hier = build_hierarchy(FakeDoc([FakeObj("root", "Assembly", "App::Part", group=leaves)]), is_leaf=leaf_is)
    scratch = fresh_dir("h13")

    def fake_identify_interferences(prep, collision_config, cache_dir, force=False):
        ids, labels = ["p1", "p2", "p3"], ["Part1", "Part2", "Part3"]
        free = np.ones((3, 3, 6), dtype=bool)
        free[0, 1, :] = free[1, 0, :] = False  # Part1 / Part2: no free direction (rule-induced or blocking)
        return SimpleNamespace(component_ids=ids, component_labels=labels, free=free, cache_key="stub")

    with stub_pipeline(hier):
        ii_mod.identify_interferences = fake_identify_interferences
        with LogCapture("aap.levels") as cap:
            plan = solve(hier, output_dir=scratch)

    rec.check(("Part1", "Part2") in plan.inseparable_pairs, f"inseparable pair recorded: {plan.inseparable_pairs}")
    rec.check(any("Inseparable" in r.getMessage() for r in cap.records), "inseparable pair logs a warning")


# ---------------------------------------------------------------------------
# Real-FreeCAD tests
# ---------------------------------------------------------------------------


def lunar_doc_and_hierarchy():
    # [SUPP] (FreeCAD document, hierarchy) of lunar_bodylager with its BOM.
    import csv

    from preprocessing import load_assembly

    doc = load_assembly(LUNAR_STEP)
    with open(LUNAR_BOM, newline="", encoding="utf-8-sig") as f:
        bom_rows = list(csv.DictReader(f))
    return doc, build_hierarchy(doc, bom_rows=bom_rows)


def test_h2(rec, args):
    """Tests: fusing a container into one solid (levels._fuse_container) on lunar_bodylager.
    What: for each root-level container, the fused bounding box equals the union of its leaf boxes and the fused volume does not
          exceed the sum of the leaf volumes (small tolerance); the fuse time is reported.
    """
    from preprocessing import _global_shape
    from levels import _fuse_container

    doc, hier = lunar_doc_and_hierarchy()
    scratch = fresh_dir("h2")
    for cid in hier.containers[hier.root].child_ids:
        container = hier.containers[cid]
        shapes = [_global_shape(doc.getObject(lid)) for lid in container.all_leaf_ids]
        leaf_volume = sum(s.Volume for s in shapes)
        bbox = None
        for s in shapes:
            bbox = s.BoundBox if bbox is None else bbox.united(s.BoundBox)
        t0 = time.perf_counter()
        fused = _fuse_container(container, doc, scratch)
        rec.note(f"{container.label!r}: fused {len(shapes)} leaves in {time.perf_counter() - t0:.2f} s")
        tol = 1e-3 * max(bbox.DiagonalLength, 1.0)
        fb = fused.BoundBox
        bbox_match = (abs(fb.XMin - bbox.XMin) < tol and abs(fb.XMax - bbox.XMax) < tol
                      and abs(fb.YMin - bbox.YMin) < tol and abs(fb.YMax - bbox.YMax) < tol
                      and abs(fb.ZMin - bbox.ZMin) < tol and abs(fb.ZMax - bbox.ZMax) < tol)
        rec.check(bbox_match, f"{container.label!r}: fused bbox == union of {len(shapes)} leaf bboxes "
                  f"(tol {tol:.3g} mm)")
        rec.check(fused.Volume <= leaf_volume * 1.0005,
                  f"{container.label!r}: fused volume {fused.Volume:.1f} <= leaf sum {leaf_volume:.1f} mm^3 "
                  f"(ratio {fused.Volume / leaf_volume:.5f})")


def test_h2b(rec, args):
    """Boolean sanity (priority test): a fused multi-leaf container vs. a neighbour gives the same
    free/blocked answer as AND-ing the leaf-by-leaf tests, on every axis direction, with no
    exception cells."""
    import FreeCAD

    from collision import CollisionConfig, Stats, is_free_detail
    from preprocessing import _global_shape
    from levels import _fuse_container

    doc, hier = lunar_doc_and_hierarchy()
    root = hier.containers[hier.root]
    container_id = root.child_ids[0]
    container = hier.containers[container_id]
    neighbor_id = root.leaf_ids[0]
    neighbor_label = hier.leaf_label[neighbor_id]
    neighbor = _global_shape(doc.getObject(neighbor_id))
    leaves = [_global_shape(doc.getObject(lid)) for lid in container.all_leaf_ids]

    t0 = time.perf_counter()
    fused = _fuse_container(container, doc, fresh_dir("h2b"))
    rec.note(f"fused {container.label!r} ({len(leaves)} leaves) in {time.perf_counter() - t0:.2f} s, "
             f"volume {fused.Volume:.1f} mm^3 vs leaf sum {sum(s.Volume for s in leaves):.1f} mm^3")

    config = CollisionConfig(step=2.0)
    axes = [FreeCAD.Vector(1, 0, 0), FreeCAD.Vector(-1, 0, 0), FreeCAD.Vector(0, 1, 0),
            FreeCAD.Vector(0, -1, 0), FreeCAD.Vector(0, 0, 1), FreeCAD.Vector(0, 0, -1)]
    axis_names = ["+x", "-x", "+y", "-y", "+z", "-z"]

    mismatches, exceptions, ground_truth = [], [], {}
    t0 = time.perf_counter()
    for name, d in zip(axis_names, axes):
        stats = Stats()
        leaf_free = []
        for leaf in leaves:
            out = is_free_detail(neighbor, leaf, d, config, stats)
            leaf_free.append(out.free)
            if out.exception_cell:
                exceptions.append(f"leaf-by-leaf {name}")
        truth = all(leaf_free)
        ground_truth[name] = truth
        fused_out = is_free_detail(neighbor, fused, d, config, Stats())
        if fused_out.exception_cell:
            exceptions.append(f"fused {name}")
        if fused_out.free != truth:
            mismatches.append((name, truth, fused_out.free))
        rec.note(f"{name}: leaf-by-leaf ({len(leaves)} leaves) free={truth}, fused free={fused_out.free}")
    rec.note(f"neighbour = {neighbor_label!r}; {len(axes) * (len(leaves) + 1)} collision tests in "
             f"{time.perf_counter() - t0:.1f} s")

    rec.check(not mismatches, f"fused result matches the leaf-by-leaf ground truth on all 6 axes "
              f"(mismatches: {mismatches})")
    rec.check(not exceptions, f"no exception cells on either side (common()/distToShape() never failed "
              f"twice): {exceptions or 'none'}")
    free_axes = [n for n, v in ground_truth.items() if v]
    blocked_axes = [n for n, v in ground_truth.items() if not v]
    rec.check(bool(free_axes) and bool(blocked_axes),
              f"covers both a known-free axis {free_axes} and a known-blocked axis {blocked_axes}")


def test_h2c(rec, args):
    """Tests: that building level documents does not modify the original document (levels.build_level_document).
    What: the geometry hash of every leaf of the original document is the same before and after building the root level document;
          a child level document is built from the same original leaves.
    """
    from preprocessing import _global_shape
    from levels import build_level_document
    from shapes import shape_geometry_hash

    doc, hier = lunar_doc_and_hierarchy()
    root = hier.containers[hier.root]
    before = {lid: shape_geometry_hash(_global_shape(doc.getObject(lid))) for lid in root.all_leaf_ids}

    build_level_document(hier, hier.root, doc, fresh_dir("h2c_root"))

    after = {lid: shape_geometry_hash(_global_shape(doc.getObject(lid))) for lid in root.all_leaf_ids}
    rec.check(before == after, f"{len(before)} leaf shapes in the original document unchanged after "
              "build_level_document (geometry hash identical)")

    child_id = next(c for c in root.child_ids if hier.containers[c].processed)
    child = hier.containers[child_id]
    doc_child, _, _ = build_level_document(hier, child_id, doc, fresh_dir("h2c_child"))
    child_leaf_names = {o.Name for o in doc_child.Objects if o.TypeId == "Part::Feature"}
    rec.check(child_leaf_names == set(child.leaf_ids),
              f"child level document builds from the same (unchanged) original leaves ({sorted(child_leaf_names)})")


def test_h3(rec, args):
    """Tests: placements of nested parts (FreeCAD global placement) on lunar_bodylager.
    What: the global placement of a leaf one level deep equals parent placement x leaf placement; the global bounding box differs
          from the local one (a real transform is applied); a shape placed by manual composition has the same box as _global_shape.
    """
    doc, hier = lunar_doc_and_hierarchy()
    from preprocessing import _global_shape

    root = hier.containers[hier.root]
    child_id = next(c for c in root.child_ids if hier.containers[c].processed)
    child = hier.containers[child_id]
    leaf_obj = doc.getObject(child.leaf_ids[0])
    container_obj = doc.getObject(child_id)

    manual = container_obj.Placement.multiply(leaf_obj.Placement)
    rec.check(leaf_obj.getGlobalPlacement().Base.distanceToPoint(manual.Base) < 1e-6,
              "getGlobalPlacement() of a leaf nested one level deep == parent.Placement * leaf.Placement")

    global_shape = _global_shape(leaf_obj)
    local_bbox = leaf_obj.Shape.BoundBox
    rec.check(global_shape.BoundBox.Center.distanceToPoint(local_bbox.Center) > 1e-3,
              "global bbox differs from the leaf's own local-placement bbox (a real transform is applied)")

    manual_shape = leaf_obj.Shape.copy()
    manual_shape.Placement = manual
    gb = global_shape.BoundBox
    mb = manual_shape.BoundBox
    rec.check(abs(mb.XMin - gb.XMin) < 1e-6 and abs(mb.YMax - gb.YMax) < 1e-6 and abs(mb.ZMin - gb.ZMin) < 1e-6,
              "shape placed by manual composition has the same bbox as _global_shape's result "
              "(lunar part positions independently confirmed)")


def test_h7(rec, args):
    """Passes bom_categories (reconciled 2026-11-15): without it every standard on lunar_bodylager
    counts as category 3 ("other") instead of its real screw/nut category, which is both unfaithful
    to the real pipeline (main.py always passes it) and much slower (~11 min vs. faster with real
    categories, since screw/nut get a single-direction P4 probe instead of category 3's blocker
    sweep in both directions)."""
    from preprocessing import run_preprocessing_on_document, load_bom, load_standard_categories
    from levels import build_level_document

    doc, hier = lunar_doc_and_hierarchy()
    bom_labels = load_bom(LUNAR_BOM)
    bom_categories = load_standard_categories(LUNAR_BOM)
    root = hier.containers[hier.root]
    child_id = next(c for c in root.child_ids if hier.containers[c].processed)
    child = hier.containers[child_id]

    doc_L, compound_ids, members = build_level_document(hier, child_id, doc, fresh_dir("h7"))
    prep = run_preprocessing_on_document(doc_L, bom_labels, None, step_path=LUNAR_STEP, bom_path=LUNAR_BOM,
                                         level_id=child_id, level_label=child.label,
                                         compound_ids=frozenset(compound_ids), members=members,
                                         bom_categories=bom_categories)
    lo, hi = prep.assembly_bbox
    union = None
    for leaf_id in child.leaf_ids:
        bb = doc_L.getObject(leaf_id).Shape.BoundBox
        union = bb if union is None else union.united(bb)
    rec.check(abs(lo.x - union.XMin) < 1e-6 and abs(lo.y - union.YMin) < 1e-6 and abs(lo.z - union.ZMin) < 1e-6
              and abs(hi.x - union.XMax) < 1e-6 and abs(hi.y - union.YMax) < 1e-6 and abs(hi.z - union.ZMax) < 1e-6,
              f"child level ({child.label!r}) assembly_bbox == bounding box of its own "
              f"{len(child.leaf_ids)} components")

    root_doc_L, root_compound_ids, root_members = build_level_document(hier, hier.root, doc, fresh_dir("h7_root"))
    root_prep = run_preprocessing_on_document(root_doc_L, bom_labels, None, step_path=LUNAR_STEP, bom_path=LUNAR_BOM,
                                              level_id=hier.root, level_label=root.label,
                                              compound_ids=frozenset(root_compound_ids), members=root_members,
                                              bom_categories=bom_categories)
    rec.check(prep.assembly_bbox != root_prep.assembly_bbox,
              "child level's box is genuinely its own, not reused from the root level")


def test_h11(rec, args):
    """'none' vs 'exterior' candidate-direction counts at the lunar root level. Uses <=, not <
    (reconciled 2026-11-15): both give 10 directions there, so the strict inequality does not hold.

    The old second half of this test tried to trigger the "none" -> "exterior" fallback retry
    (run_preprocessing_on_document) with a synthetic scene of two compound-only boxes, expecting an
    empty candidate list. That branch can never fire: discover_candidate_directions always adds the
    six main axes first, unconditionally, before skip_ids is even applied, so the list is never
    empty regardless of what is skipped -- confirmed by reading src/preprocessing.py directly. A
    synthetic scene cannot reach "empty" either, for the same reason, so that check is dropped
    rather than kept failing on an unreachable code path (docs say so; the code is unchanged in the
    closed implementation phase). Also passes bom_categories now, for the same reason as H7 (real
    screw/nut categories instead of every standard counting as category 3)."""
    from preprocessing import PreprocessingConfig, load_bom, load_standard_categories, run_preprocessing_on_document
    from levels import build_level_document

    doc, hier = lunar_doc_and_hierarchy()
    bom_labels = load_bom(LUNAR_BOM)
    bom_categories = load_standard_categories(LUNAR_BOM)
    doc_L, compound_ids, members = build_level_document(hier, hier.root, doc, fresh_dir("h11"))

    with LogCapture("aap.preprocessing") as cap:
        none_prep = run_preprocessing_on_document(doc_L, bom_labels, PreprocessingConfig(compound_directions="none"),
                                                  step_path=LUNAR_STEP, bom_path=LUNAR_BOM, level_id=hier.root,
                                                  compound_ids=frozenset(compound_ids), members=members,
                                                  bom_categories=bom_categories)
    ext_prep = run_preprocessing_on_document(doc_L, bom_labels, PreprocessingConfig(compound_directions="exterior"),
                                             step_path=LUNAR_STEP, bom_path=LUNAR_BOM, level_id=hier.root,
                                             compound_ids=frozenset(compound_ids), members=members,
                                             bom_categories=bom_categories)
    rec.check(len(none_prep.candidate_directions) <= len(ext_prep.candidate_directions),
              f"'none' ({len(none_prep.candidate_directions)}) <= 'exterior' ({len(ext_prep.candidate_directions)}) "
              f"candidate directions at the lunar root level")
    rec.check(len(none_prep.candidate_directions) > 0
              and not any("retrying with 'exterior'" in r.getMessage() for r in cap.records),
              "root level's 'none' result is already non-empty, so no fallback retry/warning fires "
              f"(got {len(none_prep.candidate_directions)} directions)")


def test_h10(rec, args):
    """Optimization (M7 grouping) on a BOM row that's missing entirely or has a blank `group`
    field -- the scenario a fused or Phase 2 container hits if its BOM row is dropped or left
    blank. 2026-11-01: collect_tables now defaults a component with no BOM group to a singleton
    group of its own (its label) -- the same convention extra_functions/generate_bom.py uses when
    seeding a fresh BOM -- instead of raising; use_grouping=True must now succeed on both a blank
    and a missing row, resolving the group to the component's own label. Uses industrial_fastner's
    real pipeline objects (cached interference, fresh AoG build; never writes to
    output/industrial_fastner/) to avoid the hours-long collision stage -- the BOM-row-lookup code
    path under test (optimization.py's `rows.get(label, {})` in collect_tables) does not depend on
    which assembly's cache it runs against."""
    import csv as csv_mod
    import shutil as shutil_mod
    from dataclasses import replace

    from collision import CollisionConfig
    from interference_identification import identify_interferences
    from preprocessing import run_preprocessing

    fastener_dir = REPO_ROOT / "output" / "industrial_fastner"
    scratch = fresh_dir("h10")
    if (fastener_dir / "interference").exists():
        shutil_mod.copytree(fastener_dir / "interference", scratch / "interference")
    else:
        rec.note("output/industrial_fastner/interference is missing (pre-existing, unrelated to "
                "this test); recomputing instead of reusing the committed cache")

    prep = run_preprocessing(FASTENER_STEP, FASTENER_BOM)
    ir = identify_interferences(prep, CollisionConfig(step=2.0), scratch / "interference")
    aog = ag_mod.build_aog(ir, prep, cache_dir=scratch / "aog")

    non_standards = [c for c in range(aog.n_components) if not aog.is_standard[c]]
    target = aog.component_labels[non_standards[0]]

    with open(FASTENER_BOM, newline="", encoding="utf-8-sig") as f:
        reader = csv_mod.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    def write_bom(path, rows_out):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)

    blank_rows = [dict(r, group="") if r["label"] == target else r for r in rows]
    blank_bom = scratch / "bom_blank_group.csv"
    write_bom(blank_bom, blank_rows)

    missing_rows = [r for r in rows if r["label"] != target]
    missing_bom = scratch / "bom_missing_row.csv"
    write_bom(missing_bom, missing_rows)

    import optimization as op_mod

    on = op_mod.OptimizationConfig(use_grouping=True, use_stability=False, use_gravity_change=False)

    for name, bom_path in [("blank group field", blank_bom), ("missing row", missing_bom)]:
        prep2 = replace(prep, bom_path=bom_path)
        tables = op_mod.collect_tables(aog, ir, prep2, on, scratch / f"opt_{bom_path.stem}")
        rec.check(tables.group[non_standards[0]] == target,
                  f"{name}: collect_tables defaults {target!r}'s group to its own label "
                  f"(got {tables.group[non_standards[0]]!r})")
        try:
            op_mod.CostModel(aog, tables, on)
            rec.check(True, f"{name}: use_grouping=True now succeeds (no raise) with the defaulted group")
        except ValueError as exc:
            rec.check(False, f"{name}: use_grouping=True should not raise any more ({exc})")


def test_h1b(rec, args):
    """Tests: build_hierarchy on the real STEP files.
    What: industrial_fastner is flat (root + 8 leaves, no containers); lunar_bodylager has 28 leaves and 2 containers at the root,
          each processed=true with 2 leaves.
    """
    from preprocessing import load_assembly

    doc = load_assembly(FASTENER_STEP)
    hier = build_hierarchy(doc)
    root = hier.containers[hier.root]
    rec.check(not root.child_ids and len(root.leaf_ids) == 8,
              f"industrial_fastner: flat, root + {len(root.leaf_ids)} leaves (expected 8, no containers)")

    doc = load_assembly(LUNAR_STEP)
    import csv

    with open(LUNAR_BOM, newline="", encoding="utf-8-sig") as f:
        bom_rows = list(csv.DictReader(f))
    hier = build_hierarchy(doc, bom_rows=bom_rows)
    root = hier.containers[hier.root]
    rec.note(f"lunar root: {len(root.leaf_ids)} leaves, {len(root.child_ids)} containers "
             f"{[hier.containers[c].label for c in root.child_ids]}")
    rec.check(len(root.leaf_ids) == 28 and len(root.child_ids) == 2,
              f"lunar_bodylager root: 28 leaves + 2 containers (got {len(root.leaf_ids)} leaves, "
              f"{len(root.child_ids)} containers)")
    for cid in root.child_ids:
        c = hier.containers[cid]
        rec.check(c.processed is True and len(c.leaf_ids) == 2,
                  f"container {c.label!r}: processed=true, 2 leaves (got processed={c.processed}, "
                  f"{len(c.leaf_ids)} leaves)")


def test_h12(rec, args):
    """Flat regression: the new per-document entry point reproduces the old one on industrial_fastner
    at the Preprocessing level (same cache key, component ids/shapes, candidate directions). Does not
    re-run the (multi-minute) interference/AoG/optimization stages."""
    from preprocessing import load_assembly, load_bom, run_preprocessing, run_preprocessing_on_document
    from interference_identification import _components, _directions, compute_cache_key

    old = run_preprocessing(FASTENER_STEP, FASTENER_BOM)

    doc = load_assembly(FASTENER_STEP)
    bom_labels = load_bom(FASTENER_BOM)
    new = run_preprocessing_on_document(doc, bom_labels, None,
                                        step_path=FASTENER_STEP, bom_path=FASTENER_BOM)

    old_ids, old_labels, old_shapes = _components(old.reduced_assembly)
    new_ids, new_labels, new_shapes = _components(new.reduced_assembly)
    rec.check(old_ids == new_ids and old_labels == new_labels,
              f"same component ids/labels ({len(old_ids)} components)")
    volumes_match = all(abs(a.Volume - b.Volume) < 1e-9 for a, b in zip(old_shapes, new_shapes))
    rec.check(volumes_match, "same component shapes (volumes match within 1e-9)")

    old_dirs, old_neg = _directions(old.candidate_directions)
    new_dirs, new_neg = _directions(new.candidate_directions)
    rec.check(len(old_dirs) == len(new_dirs)
              and all((a - b).Length < 1e-9 for a, b in zip(old_dirs, new_dirs)),
              f"same candidate directions ({len(old_dirs)})")

    from collision import CollisionConfig
    key_old = compute_cache_key(old, old_ids, old_shapes, old_dirs, CollisionConfig())
    key_new = compute_cache_key(new, new_ids, new_shapes, new_dirs, CollisionConfig())
    rec.check(key_old == key_new, f"same interference cache_key ({key_old == key_new})")
    rec.check(old.standard_records.keys() == new.standard_records.keys(), "same standard ids")

    # The actual regression risk: going through build_hierarchy + build_level_document (the real
    # nested-aware entry point for the root level) must reproduce the *committed* cache (generated
    # by `main.py ... --step 5`, i.e. CollisionConfig(step=5.0, gap_skip=False) -- compute_cache_key
    # hashes via json.dumps, so the float from argparse's `type=float` is not interchangeable with a
    # plain int 5 here), so an existing cache from before this feature still hits.
    cached_key = json.loads(
        (REPO_ROOT / "output" / "industrial_fastner" / "interference" / "interference.json")
        .read_text(encoding="utf-8")
    )["cache_key"]
    hier = build_hierarchy(doc)
    doc_L, compound_ids, members = levels_mod.build_level_document(hier, hier.root, doc, fresh_dir("h12_level_doc"))
    via_level_doc = preprocessing_mod.run_preprocessing_on_document(
        doc_L, bom_labels, None, step_path=FASTENER_STEP, bom_path=FASTENER_BOM,
        level_id=hier.root, compound_ids=frozenset(compound_ids), members=members,
    )
    ld_ids, ld_labels, ld_shapes = _components(via_level_doc.reduced_assembly)
    ld_dirs, ld_neg = _directions(via_level_doc.candidate_directions)
    key_via_level_doc = compute_cache_key(via_level_doc, ld_ids, ld_shapes, ld_dirs,
                                          CollisionConfig(step=5.0, gap_skip=False))
    rec.check(key_via_level_doc == cached_key,
              f"build_level_document on the root level reproduces the committed cache_key "
              f"({key_via_level_doc == cached_key})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 2: merge_groups / sub_interference, no FreeCAD.
# ---------------------------------------------------------------------------


def _random_interf(rng, n: int, D: int = 4):
    """[SUPP] A random synthetic InterferenceResult (no geometry): antisymmetric free, symmetric contact."""
    from collision import CollisionConfig
    from interference_identification import InterferenceResult

    ids = [f"p{i}" for i in range(n)]
    neg = np.array([(k + D // 2) % D for k in range(D)])
    free = np.zeros((n, n, D), dtype=bool)
    for i in range(n):
        free[i, i, :] = True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(D):
                val = bool(rng.integers(0, 2))
                free[i, j, k] = val
                free[j, i, int(neg[k])] = val
    contact = rng.integers(0, 2, size=(n, n)).astype(bool)
    contact = contact | contact.T
    np.fill_diagonal(contact, False)
    gaps = np.where(contact, 0.0, rng.uniform(1, 10, size=(n, n)))
    gaps = (gaps + gaps.T) / 2
    return InterferenceResult(
        component_ids=ids, component_labels=ids, directions=np.zeros((D, 3)), neg=neg, free=free,
        pair_info={}, exception_cells=[], caught_exceptions=[], outcome_detail={},
        config=CollisionConfig(), stats={}, cache_key=f"rng:{rng.bit_generator.state['state']['state']}",
        contact=contact, contact_threshold=0.1, gaps=gaps,
    )


def test_r4(rec, args):
    """merge_groups correctness + merge-loop termination: the derived free/
    contact/gaps equal an independent brute-force AND/OR/min over 20 random synthetic tensors;
    repeated merging down to 1 component always terminates (each merge strictly shrinks the
    component count by |group| - 1 >= 1, so the loop terminates)."""
    from interference_identification import merge_groups

    rng = np.random.default_rng(4)
    mismatches = []
    for _trial in range(20):
        n = int(rng.integers(4, 9))
        interf = _random_interf(rng, n)
        group_size = int(rng.integers(2, n))
        group = sorted(rng.choice(n, size=group_size, replace=False).tolist())
        group_ids = [interf.component_ids[i] for i in group]
        merged = merge_groups(interf, group_ids, "auto_x", "Auto X")
        i_auto = merged.index_of("auto_x")
        keep = [i for i in range(n) if i not in group]
        for i in keep:
            b = merged.index_of(interf.component_ids[i])
            expect_free = np.ones(len(interf.directions), dtype=bool)
            for g in group:
                expect_free &= interf.free[g, i, :]
            if not np.array_equal(merged.free[i_auto, b, :], expect_free):
                mismatches.append(("free", group_ids, i))
            if bool(merged.contact[i_auto, b]) != any(bool(interf.contact[g, i]) for g in group):
                mismatches.append(("contact", group_ids, i))
            expect_gap = min(float(interf.gaps[g, i]) for g in group)
            if abs(float(merged.gaps[i_auto, b]) - expect_gap) > 1e-9:
                mismatches.append(("gaps", group_ids, i))
    rec.check(not mismatches, f"merge_groups' free/contact/gaps match an independent brute-force "
             f"AND/OR/min over 20 random trials (mismatches: {mismatches[:3] or 'none'})")

    rng = np.random.default_rng(5)
    n0 = 10
    interf = _random_interf(rng, n0)
    steps = 0
    while len(interf.component_ids) > 1 and steps < n0:
        ids = interf.component_ids
        pair = rng.choice(len(ids), size=2, replace=False)
        group_ids = [ids[int(pair[0])], ids[int(pair[1])]]
        interf = merge_groups(interf, group_ids, f"auto_{steps}", f"Auto {steps}")
        steps += 1
    rec.check(len(interf.component_ids) == 1 and steps < n0,
             f"repeated merging terminates (down to 1 component in {steps} merges)")


def test_r8(rec, args):
    """Phase 2 cache-key stability (plus the AoG cache-key confirmation): merge_
    groups/sub_interference are deterministic and change with the parent cache_key, the group, or
    the new id; the committed AoG's own cache_key (aog_generator.compute_cache_key) is confirmed by
    source inspection to fold in both AoGInputs.cache_key (itself built from `interference_result.
    cache_key`, per collect_inputs) and `inputs.component_ids` directly -- so a Phase 2 merge's
    distinct interference cache_key, and its distinct resulting component_ids, each independently
    give the committed AoG a distinct cache_key too."""
    import inspect

    import aog_generator as ag_mod
    from dataclasses import replace as _replace

    from interference_identification import merge_groups, sub_interference

    rng = np.random.default_rng(8)
    interf = _random_interf(rng, 5)
    m1 = merge_groups(interf, [interf.component_ids[0], interf.component_ids[1]], "auto_0", "Auto 0")
    m2 = merge_groups(interf, [interf.component_ids[0], interf.component_ids[1]], "auto_0", "Auto 0")
    rec.check(m1.cache_key == m2.cache_key, "merge_groups is deterministic: same inputs -> same cache_key")

    m3 = merge_groups(interf, [interf.component_ids[0], interf.component_ids[2]], "auto_0", "Auto 0")
    rec.check(m1.cache_key != m3.cache_key, "a different group -> a different cache_key")

    m4 = merge_groups(interf, [interf.component_ids[0], interf.component_ids[1]], "auto_1", "Auto 1")
    rec.check(m1.cache_key != m4.cache_key, "a different new_id -> a different cache_key")

    interf2 = _replace(interf, cache_key=interf.cache_key + "x")
    m5 = merge_groups(interf2, [interf2.component_ids[0], interf2.component_ids[1]], "auto_0", "Auto 0")
    rec.check(m1.cache_key != m5.cache_key, "a different parent cache_key -> a different merged cache_key")

    s1 = sub_interference(interf, [interf.component_ids[0], interf.component_ids[1]])
    s2 = sub_interference(interf, [interf.component_ids[0], interf.component_ids[2]])
    rec.check(s1.cache_key != s2.cache_key, "sub_interference: a different member set -> a different cache_key")

    key_src = inspect.getsource(ag_mod.compute_cache_key)
    rec.check('inputs.cache_key.encode()' in key_src,
             "aog_generator.compute_cache_key folds in AoGInputs.cache_key (confirmed by source "
             "inspection, src/aog_generator.py)")
    rec.check('"ids:" + "|".join(inputs.component_ids)' in key_src,
             "aog_generator.compute_cache_key also folds in inputs.component_ids directly "
             "(confirmed by source inspection) -- so a merge's distinct component_ids alone "
             "already changes the committed AoG's cache_key, independent of the interference key")
    inputs_src = inspect.getsource(ag_mod.collect_inputs)
    rec.check('interference_result.cache_key' in inputs_src,
             "aog_generator.collect_inputs derives AoGInputs.cache_key from interference_result."
             "cache_key (confirmed by source inspection)")


def test_r2(rec, args):
    """industrial_fastner's AoG is complete with no dead end at all (reconciled 2026-11-15): the
    {part3, part6, part7} dead end this test used to check for is obsolete -- since the three-class
    standard model (2026-11-04), the old gate = prerequisites | contact_neighbors is gone and every
    standard's gate is derived from per-option blocker masks instead, so the AoG is now complete on
    its own (V=61, E=141). Phase 2 therefore has nothing to merge here; confirmed directly against
    the real cached pipeline objects (never writes to output/industrial_fastner/). Rule-induced
    classification itself is covered by the synthetic R5."""
    import shutil as shutil_mod

    from aog_generator import classify_dead_ends
    from collision import CollisionConfig
    from preprocessing import run_preprocessing

    fastener_dir = REPO_ROOT / "output" / "industrial_fastner"
    scratch = fresh_dir("r2")
    if (fastener_dir / "interference").exists():
        shutil_mod.copytree(fastener_dir / "interference", scratch / "interference")
    else:
        rec.note("output/industrial_fastner/interference is missing (pre-existing, unrelated to "
                "this test); recomputing instead of reusing the committed cache")

    prep = run_preprocessing(FASTENER_STEP, FASTENER_BOM)
    ir = ii_mod.identify_interferences(prep, CollisionConfig(step=2.0), scratch / "interference")
    aog = ag_mod.build_aog(ir, prep, cache_dir=scratch / "aog")
    rec.check(aog.complete, "industrial_fastner's AoG is complete")
    rec.check(len(aog.dead_ends) == 0, f"no dead end at all (got {len(aog.dead_ends)})")

    kinds = classify_dead_ends(aog, ir)
    rec.check(kinds == {}, f"nothing to classify, so Phase 2 finds nothing to merge (got {kinds})")


def test_r3(rec, args):
    """Locked level handling (core -- pure numpy, no FreeCAD): a fully mutually-
    blocking 3-part state (every pair blocked in the one direction) has no valid 2-part closure --
    any closure immediately expands to all 3 members, which fails the `< n_present` check -- so
    Phase 2 finds no group to merge; the level must report status="locked" naming the stuck parts,
    with no sequence, and must never call optimize_sequence or generate_illustrations."""
    from dataclasses import replace as dc_replace

    from aog_generator import AoGConfig, AoGInputs, generate_aog
    from collision import CollisionConfig
    from interference_identification import InterferenceResult
    from preprocessing import PreprocessingResult

    ids = ["p0", "p1", "p2"]
    D = 1
    neg = np.array([0])
    free = np.zeros((3, 3, D), dtype=bool)
    for i in range(3):
        free[i, i, :] = True  # every other pair fully blocked (left False)
    contact = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    interf = InterferenceResult(
        component_ids=ids, component_labels=ids, directions=np.zeros((D, 3)), neg=neg, free=free,
        pair_info={}, exception_cells=[], caught_exceptions=[], outcome_detail={},
        config=CollisionConfig(), stats={}, cache_key="r3",
        contact=contact, contact_threshold=0.1, gaps=np.zeros((3, 3)),
    )
    inputs = AoGInputs(
        component_ids=ids, component_labels=ids, n_movable=3, free=free, directions=np.zeros((D, 3)),
        neg=neg, standard_options={}, standard_holds={}, nut_links={}, bbox_lo=np.zeros((3, 3)), bbox_hi=np.ones((3, 3)),
        assembly_lo=np.zeros(3), assembly_hi=np.ones(3), cache_key="r3-inputs", contact=contact, group=None,
    )
    aog = generate_aog(inputs, AoGConfig())
    rec.check(not aog.complete, f"setup: the fully-entangled 3-part state is genuinely incomplete "
             f"(V={aog.n_nodes}, E={aog.n_edges})")

    prep = PreprocessingResult(reduced_assembly=None, standard_records={}, candidate_directions=[],
                               level_id="root", level_label="Assembly")

    calls = {"optimize": 0, "illustrate": 0}
    real_build_aog, real_optimize, real_illustrate = ag_mod.build_aog, opt_mod.optimize_sequence, ig_mod.generate_illustrations

    def fake_build_aog(*a, **k):
        return aog

    def fake_optimize_sequence(*a, **k):
        calls["optimize"] += 1
        return SimpleNamespace(steps=[], total_cost=0.0)

    def fake_generate_illustrations(*a, **k):
        calls["illustrate"] += 1

    ag_mod.build_aog, opt_mod.optimize_sequence, ig_mod.generate_illustrations = (
        fake_build_aog, fake_optimize_sequence, fake_generate_illustrations)
    try:
        hier = build_hierarchy(flat_tree(3), is_leaf=leaf_is)
        scratch = fresh_dir("r3")
        plan = levels_mod._solve_pipeline(hier, hier.root, "Assembly", prep, interf, scratch)
    finally:
        ag_mod.build_aog, opt_mod.optimize_sequence, ig_mod.generate_illustrations = (
            real_build_aog, real_optimize, real_illustrate)

    rec.check(plan.status == "locked", f"locked level reports status='locked' (got {plan.status!r})")
    rec.check(set(plan.locked_parts) == {"p0", "p1", "p2"},
             f"locked_parts names the stuck parts (got {plan.locked_parts})")
    rec.check(plan.sequence is None, "locked level has no sequence")
    rec.check(calls == {"optimize": 0, "illustrate": 0},
             f"optimize_sequence/generate_illustrations are never called for a locked level (got {calls})")


def test_r5(rec, args):
    """D12/D13 interaction (confirmed answer: D12/D13 deadlocks are never merged,
    logged as such): an AoG incomplete solely because a BOM group block cannot be assembled
    consecutively (D12) classifies rule_induced, exactly like a standard-gate or D11 conflict --
    classify_dead_ends never inspects group state at all, so this holds by construction. Confirmed
    on the same "group {A, C} but B must leave between them" case as test_aog.py's A13."""
    from aog_generator import classify_dead_ends, generate_aog
    from collision import CollisionConfig
    from interference_identification import InterferenceResult
    from test_aog import AXES, AXES_NEG, make_inputs

    ids = ["A", "B", "C"]
    free3 = np.ones((3, 3, 6), dtype=bool)
    free3[0, 1, :] = False  # A blocked by B
    free3[1, 2, :] = False  # B blocked by C
    inputs = make_inputs(free3, ids=ids, group=["G", None, "G"])
    aog = generate_aog(inputs)
    rec.check(not aog.complete, "setup: the group-consecutive conflict is genuinely incomplete (matches A13)")

    interf = InterferenceResult(
        component_ids=ids, component_labels=ids, directions=np.asarray(AXES, dtype=float),
        neg=np.asarray(AXES_NEG), free=free3,
        pair_info={}, exception_cells=[], caught_exceptions=[], outcome_detail={},
        config=CollisionConfig(), stats={}, cache_key="r5",
        contact=~np.eye(3, dtype=bool), contact_threshold=0.1, gaps=np.zeros((3, 3)),
    )
    kinds = classify_dead_ends(aog, interf)
    rec.check(bool(kinds) and all(k == "rule_induced" for k in kinds.values()),
             f"every dead end in the D12-conflict AoG classifies rule_induced, never blocking "
             f"(got {kinds}) -- Phase 2 would never attempt to merge this")


def test_r6(rec, args):
    """Phase 2 M7 group inheritance (amended 2026-11-01): a virtual auto_<n>
    container inherits its direct members' shared BOM group if they agree, else falls back to its
    own id and logs a warning; neither case ever raises with use_grouping=True any more (H10's
    default-group fix applies identically to auto ids). Exercised on nonlinearsimple's real
    geometry via levels._commit_merge directly (no need to run the full merge loop)."""
    import csv as csv_mod
    from dataclasses import replace as dc_replace

    from collision import CollisionConfig
    from hierarchy import build_hierarchy as build_hierarchy_real
    from interference_identification import identify_interferences
    from levels import _commit_merge, build_level_document
    from preprocessing import load_assembly, load_bom, run_preprocessing_on_document

    doc = load_assembly(NONLINEAR_STEP)
    bom_labels = load_bom(NONLINEAR_BOM)
    with open(NONLINEAR_BOM, newline="", encoding="utf-8-sig") as f:
        reader = csv_mod.DictReader(f)
        fieldnames = reader.fieldnames
        bom_rows = list(reader)
    hier = build_hierarchy_real(doc, bom_rows=bom_rows)
    scratch = fresh_dir("r6")
    doc_L, compound_ids, members = build_level_document(hier, hier.root, doc, scratch)
    prep = run_preprocessing_on_document(doc_L, bom_labels, None, step_path=NONLINEAR_STEP,
                                         bom_path=NONLINEAR_BOM, level_id=hier.root,
                                         compound_ids=frozenset(compound_ids), members=members)
    interf = identify_interferences(prep, CollisionConfig(), scratch / "interference")
    group_labels = ["component", "component001"]  # BOM labels, for matching BOM rows
    group_ids = [next(o.Name for o in doc_L.Objects if o.Label == lbl) for lbl in group_labels]

    # Case (b): the real BOM gives these two different default groups (their own labels) -> no
    # shared group -> auto_0 falls back to its own id and warns.
    with LogCapture("aap.levels") as cap:
        _, prep_b, _, _ = _commit_merge(doc_L, prep, interf, group_ids, "auto_0", "Auto 0 test", scratch)
    rec.check(prep_b.auto_group_override.get("auto_0") == "auto_0",
             f"no shared group: auto_0 falls back to its own id (got {prep_b.auto_group_override.get('auto_0')!r})")
    rec.check(any("do not share one BOM group" in r.getMessage() for r in cap.records),
             "no shared group: logs a warning")

    # Case (a): a BOM where both members explicitly share one group -> auto_1 inherits it, no warning.
    shared_rows = [dict(r, group="Shared") if r.get("label") in group_labels else r for r in bom_rows]
    shared_bom = scratch / "bom_shared_group.csv"
    with open(shared_bom, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shared_rows)
    prep_shared = dc_replace(prep, bom_path=shared_bom)
    with LogCapture("aap.levels") as cap:
        _, prep_a, _, _ = _commit_merge(doc_L, prep_shared, interf, group_ids, "auto_1", "Auto 1 test", scratch)
    rec.check(prep_a.auto_group_override.get("auto_1") == "Shared",
             f"shared group 'Shared': auto_1 inherits it (got {prep_a.auto_group_override.get('auto_1')!r})")
    rec.check(not any("do not share one BOM group" in r.getMessage() for r in cap.records),
             "shared group: no warning logged")


def test_r7(rec, args):
    """Nested Phase 2 (confirmed: allowed) -- and a real edge case this test run
    surfaced along the way, worth recording. Two disjoint mutually-blocked pairs {p0,p1}/{p2,p3}:
    find_candidate_groups' closures are a pure function of the static, pairwise interference
    tensor, so a group that is genuinely minimal where it was found is, by the same math, also
    irreducible within its own restricted (sub_interference) view -- there is no smaller sub-group
    to split out further. Concretely here: after {p0,p1} merges into auto_0 and is removed, the
    remaining state is exactly {p2, p3} -- and the candidate rule requires `2 <= |A| <
    |S|`, a *strict* less-than, so a dead end that is exactly a mutually-blocked pair
    with nothing else present can never be merged (|A| would have to equal |S|). This is not a
    missing feature: merging "everything left" isn't a sub-assembly extraction, so the rule
    correctly excludes it -- the level reports locked. This test confirms _solve_pipeline handles
    exactly this sequence correctly: a first merge commits for real (auto_0 is fused, logged), the
    AoG is rebuilt, a second dead end is found with no valid candidate, and the level reports
    status="locked" with the right stuck parts -- not a crash, not a silently wrong "complete"."""
    import FreeCAD
    import Part

    from preprocessing import PreprocessingResult

    doc = FreeCAD.newDocument("r7")
    names = ["p0", "p1", "p2", "p3"]
    for i, name in enumerate(names):
        f = doc.addObject("Part::Feature", name)
        f.Shape = Part.makeBox(10, 10, 10, FreeCAD.Vector(i * 20, 0, 0))
        f.Label = name
    doc.recompute()

    # Two disjoint mutually-blocked pairs ({p0,p1}, {p2,p3}) in both directions -- every part has
    # zero free directions (blocked by its own pair-partner regardless of the others), so this is
    # a genuine dead end with two valid 2-part candidate groups.
    D = 2
    neg = np.array([1, 0])
    free = np.ones((4, 4, D), dtype=bool)
    free[0, 1, :] = free[1, 0, :] = False
    free[2, 3, :] = free[3, 2, :] = False
    contact = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=bool)
    from interference_identification import InterferenceResult
    from collision import CollisionConfig

    interf = InterferenceResult(
        component_ids=names, component_labels=names, directions=np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
        neg=neg, free=free,
        pair_info={}, exception_cells=[], caught_exceptions=[], outcome_detail={},
        config=CollisionConfig(), stats={}, cache_key="r7",
        contact=contact, contact_threshold=0.1, gaps=np.zeros((4, 4)),
    )
    prep = PreprocessingResult(reduced_assembly=doc, standard_records={}, candidate_directions=[],
                               assembly_bbox=(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(60, 10, 10)),
                               step_path=None, bom_path=None, level_id="child_level", level_label="Child")

    calls = {"optimize": 0, "illustrate": 0}
    real_build_aog, real_optimize, real_illustrate = ag_mod.build_aog, opt_mod.optimize_sequence, ig_mod.generate_illustrations

    def fake_build_aog(interference_result, p, config=None, cache_dir=None, force=None):
        from aog_generator import AoGInputs, AoGConfig, generate_aog
        n = len(interference_result.component_ids)
        inputs = AoGInputs(
            component_ids=list(interference_result.component_ids),
            component_labels=list(interference_result.component_labels),
            n_movable=n, free=interference_result.free, directions=interference_result.directions,
            neg=interference_result.neg, standard_options={}, standard_holds={}, nut_links={},
            bbox_lo=np.zeros((n, 3)), bbox_hi=np.ones((n, 3)) * 10,
            assembly_lo=np.zeros(3), assembly_hi=np.array([60.0, 10.0, 10.0]),
            cache_key=interference_result.cache_key, contact=interference_result.contact, group=None,
        )
        return generate_aog(inputs, config or AoGConfig())

    def fake_optimize_sequence(*a, **k):
        calls["optimize"] += 1
        return SimpleNamespace(steps=[], total_cost=0.0)

    def fake_generate_illustrations(*a, **k):
        calls["illustrate"] += 1

    ag_mod.build_aog, opt_mod.optimize_sequence, ig_mod.generate_illustrations = (
        fake_build_aog, fake_optimize_sequence, fake_generate_illustrations)
    try:
        fake_root = FakeObj("child_level", "Child", "App::Part",
                            group=[FakeObj(n, n, "Part::Feature") for n in names])
        hier = build_hierarchy(FakeDoc([fake_root]), is_leaf=leaf_is)
        hier.root = "child_level"
        scratch = fresh_dir("r7")
        plan = levels_mod._solve_pipeline(hier, "child_level", "Child", prep, interf, scratch)
    finally:
        ag_mod.build_aog, opt_mod.optimize_sequence, ig_mod.generate_illustrations = (
            real_build_aog, real_optimize, real_illustrate)

    rec.check(plan.status == "locked",
             f"the second dead end ({{p2, p3}} alone, nothing else present) has no valid candidate "
             f"-- 2 <= |A| < |S| is strict, so the whole remaining state can never be the group -- "
             f"and the level correctly reports locked rather than silently 'complete' or crashing "
             f"(got status={plan.status!r})")
    rec.check(set(plan.locked_parts) == {"p2", "p3"},
             f"locked_parts names exactly the still-stuck pair (got {plan.locked_parts})")
    rec.check(plan.sequence is None, "a locked level has no sequence")
    rec.check("auto_0" in hier.containers and "auto_0" in hier.containers["child_level"].child_ids,
             "the first merge still committed for real (auto_0 fused and registered) even though "
             "the overall level ends up locked on what's left")


TESTS = [
    ("H1", test_h1, "core"), ("H4", test_h4, "core"), ("H4b", test_h4b, "core"), ("H4c", test_h4c, "core"),
    ("H5", test_h5, "core"),
    ("H6", test_h6, "core"), ("H6b", test_h6b, "real"), ("H8", test_h8, "core"), ("H9", test_h9, "core"),
    ("H13", test_h13, "core"),
    ("H2", test_h2, "real"), ("H2b", test_h2b, "real"), ("H2c", test_h2c, "real"), ("H3", test_h3, "real"),
    ("H7", test_h7, "real"), ("H10", test_h10, "real"), ("H11", test_h11, "real"), ("H1b", test_h1b, "real"),
    ("H12", test_h12, "real"),
    ("R2", test_r2, "real"), ("R3", test_r3, "core"), ("R4", test_r4, "core"), ("R5", test_r5, "core"),
    ("R6", test_r6, "real"), ("R7", test_r7, "real"), ("R8", test_r8, "core"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="*", help="individual test ids, e.g. H1 H5")
    ap.add_argument("--groups", nargs="*", default=["core"], choices=["core", "real"],
                    help="core (default), real (FreeCAD runs on industrial_fastner / lunar_bodylager)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="    %(name)s %(levelname)s: %(message)s")

    rec = Recorder()
    for test_id, fn, group in TESTS:
        if args.tests and test_id not in args.tests:
            continue
        if not args.tests and group not in args.groups:
            continue
        print(f"\n=== {test_id} ===", flush=True)
        rec.start(test_id)
        t0 = time.perf_counter()
        try:
            fn(rec, args)
        except Exception as exc:  # noqa: BLE001
            rec.error(exc)
        rec.results[test_id]["seconds"] = round(time.perf_counter() - t0, 3)

    print("\n| Test | Result | Notes |\n|---|---|---|")
    for test_id, r in rec.results.items():
        status = "PASS" if r["passed"] else ("ERROR" if "error" in r else "FAIL")
        failed = [c["label"] for c in r["checks"] if not c["passed"]]
        notes = "; ".join(failed[:2] + r["notes"][:1] + ([r["error"]] if "error" in r else []))
        print(f"| {test_id} | {status} | {notes} |")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "subassembly_results.json"
    out.write_text(json.dumps({"groups": args.groups, "results": rec.results}, indent=2))
    print(f"\nDetails written to {out}")
    return 0 if all(r["passed"] for r in rec.results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
