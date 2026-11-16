"""Sub-assembly hierarchy extraction: Containers (App::Part) and the Hierarchy tree.

Inputs
    - A FreeCAD document (anything exposing `RootObjects`, and per object `TypeId`, `Name`, `Label`,
      `Group`; tests use fakes, so no FreeCAD is needed here).
    - The BOM rows as plain dicts (the caller reads the CSV): columns `label`, `processed`,
      `subassembly`, `is_stander` are used here.
Outputs
    - A `Hierarchy`: the tree of `Container`s (one per App::Part, plus a virtual root when the document
      has no single top-level part), with the leaves (solid components) of every container and each
      container's `processed` decision.
    - `check_level_sizes` raises `LevelSizeError` before any geometry work if a level is too large.

What this module does
    It walks the document tree once. A leaf is a `Part::Feature` with solids (`_default_is_leaf`); an
    `App::Part` becomes a `Container`; any other group type (for example `App::DocumentObjectGroup`) is
    transparent, like in `generate_bom.collect_rows`. Then it resolves the BOM `processed` / `subassembly`
    columns against the tree (BOM and STEP must agree), applies the "fewer than 2 components" rule, and
    leaves the per-level work to `levels.solve_level`.
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass, field
from typing import Callable, Iterable

log = logging.getLogger("aap.hierarchy")

_TRUE = {"1", "true", "yes", "y", "t"}  # accepted spellings of a true BOM flag
_FALSE = {"0", "false", "no", "n", "f"}  # accepted spellings of a false BOM flag


class LevelSizeError(RuntimeError):
    """A processed level has more components than the configured limit."""


@dataclass
class Container:
    """One App::Part of the assembly tree (or the virtual root): a node of the `Hierarchy`.

    A container is one level of the planning: its direct leaves plus its direct child containers are the
    components that `levels.solve_level` orders. Child containers are fused into rigid solids at the
    parent level.
    """

    id: str  # str. obj.Name (unique in the document); "__root__" for the virtual root.
    label: str  # str. obj.Label, matched against the BOM `label` column; "" for the virtual root.
    parent: str | None  # str or None. Id of the parent container; None for the root.
    leaf_ids: list[str]  # list[str]. Direct leaf children (obj.Name) in document order: the parts of this level.
    child_ids: list[str]  # list[str]. Direct child containers: sub-assemblies, fused as one component at this level.
    processed: bool  # bool. True = expanded into its own level; False = one rigid component. Set from the BOM.
    # list[str]. Every descendant leaf (nested too); needed to fuse the container into one solid.
    all_leaf_ids: list[str] = field(default_factory=list)
    # bool. Phase 2: a virtual container (auto_<n>) created by levels.py from a closed group of parts,
    # not a real STEP/BOM sub-assembly.
    auto: bool = False


@dataclass
class Hierarchy:
    """The assembly tree: output of `build_hierarchy`, input of `check_level_sizes` and `levels.solve_level`."""

    root: str  # str. Id of the root container.
    containers: dict[str, Container]  # dict id -> Container. Every container of the tree, root included.
    leaf_label: dict[str, str]  # dict leaf id -> Label (needed to match BOM rows and to name results).
    leaf_parent: dict[str, str]  # dict leaf id -> id of its immediate parent container.


def _default_is_leaf(obj) -> bool:
    """[AUX] True if obj is a solid component; the same filter as `preprocessing._leaf_components`, for one object."""
    return (
        getattr(obj, "TypeId", None) == "Part::Feature"
        and getattr(obj, "Shape", None) is not None
        and not obj.Shape.isNull()
        and len(obj.Shape.Solids) > 0
    )


def _leaf_component_names(doc, is_leaf: Callable) -> set[str]:
    """[AUX] Names of every leaf in the document (used to check that the tree walk reached them all)."""
    return {obj.Name for obj in getattr(doc, "Objects", []) if is_leaf(obj)}


def _walk(obj, container: Container, containers: dict[str, Container], leaf_label: dict[str, str],
          leaf_parent: dict[str, str], is_leaf: Callable) -> None:
    """[MAIN] Recursive tree walk: fill `container`'s leaf_ids / child_ids from obj's direct `Group`.

    Leaves are recorded in the current container, each App::Part becomes a child `Container` and is
    walked in turn, other group types are walked as if they were the same container (no depth change),
    and an App::Link (repeated sub-assembly) is rejected as not supported yet.
    """
    for child in getattr(obj, "Group", None) or []:
        type_id = getattr(child, "TypeId", None)
        if type_id == "App::Link":
            raise NotImplementedError(
                f"Repeated sub-assembly (App::Link) {getattr(child, 'Label', child)!r} is not "
                "supported yet."
            )
        if is_leaf(child):
            container.leaf_ids.append(child.Name)
            leaf_label[child.Name] = child.Label
            leaf_parent[child.Name] = container.id
            continue
        if type_id == "App::Part":
            child_rec = Container(id=child.Name, label=child.Label, parent=container.id,
                                   leaf_ids=[], child_ids=[], processed=True)
            containers[child.Name] = child_rec
            container.child_ids.append(child.Name)
            _walk(child, child_rec, containers, leaf_label, leaf_parent, is_leaf)
            continue
        # Transparent group type: same container, no depth change.
        _walk(child, container, containers, leaf_label, leaf_parent, is_leaf)


def _fill_all_leaf_ids(containers: dict[str, Container], root_id: str) -> None:
    """[AUX] Set `all_leaf_ids` of every container to its direct leaves plus those of all descendants (memoised recursion)."""
    memo: dict[str, list[str]] = {}

    def collect(cid: str) -> list[str]:
        if cid in memo:
            return memo[cid]
        c = containers[cid]
        leaves = list(c.leaf_ids)
        for child_id in c.child_ids:
            leaves.extend(collect(child_id))
        memo[cid] = leaves
        c.all_leaf_ids = leaves
        return leaves

    collect(root_id)


def _parse_processed(value: str | None, context: str) -> bool:
    """[AUX] Parse a BOM `processed` cell: blank = True, else one of _TRUE / _FALSE; ValueError otherwise (context names the row)."""
    v = (value or "").strip().lower()
    if v == "":
        return True
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"Invalid 'processed' value {value!r} for {context}")


def _index_bom_rows(bom_rows: Iterable[dict] | None) -> dict[str, dict]:
    """[AUX] BOM rows indexed by their stripped `label` (rows without a label are dropped)."""
    index: dict[str, dict] = {}
    for row in bom_rows or []:
        label = (row.get("label") or "").strip()
        if label:
            index[label] = row
    return index


def _resolve_bom(containers: dict[str, Container], leaf_label: dict[str, str], leaf_parent: dict[str, str],
                 bom_rows: Iterable[dict] | None, root_id: str) -> None:
    """[MAIN] Resolve the BOM columns against the tree: sets `Container.processed` and checks BOM/STEP consistency.

    Why: the BOM is where the user decides which sub-assemblies are expanded (`processed`) or fused
    (stander), and the BOM must describe the same tree as the STEP, otherwise a decision would be
    silently lost, so inconsistencies raise ValueError instead of being guessed around.
    """
    by_label = _index_bom_rows(bom_rows)
    used_labels: set[str] = set()
    mismatches: list[str] = []

    # Containers: processed flag from the BOM row, plus the stander and subassembly consistency checks.
    for cid, c in containers.items():
        row = by_label.get(c.label) if c.label else None
        if row is not None:
            used_labels.add(c.label)
            is_stander = (row.get("is_stander") or "").strip().lower() in _TRUE
            if is_stander and cid == root_id:
                raise ValueError(
                    f"BOM row for the root container {c.label!r} [{cid}] sets is_stander=true, but "
                    "the root cannot be a stander"
                )
            raw_processed = (row.get("processed") or "").strip()
            if is_stander and raw_processed and _parse_processed(raw_processed, f"container {c.label!r} [{cid}]"):
                raise ValueError(
                    f"BOM row for container {c.label!r} [{cid}] sets is_stander=true and "
                    "processed=true; a stander subassembly is fused into one rigid component "
                    "('other' category) and cannot also be "
                    "recursively processed as its own level -- set processed=false or leave it blank"
                )
            if is_stander:
                # A stander subassembly is always fused as a single rigid component: unlike a plain
                # container, blank 'processed' here defaults to false, not true.
                if not raw_processed:
                    log.info("Container %r [%s] sets is_stander=true with 'processed' blank; "
                             "defaulting to processed=false", c.label, cid)
                processed = False
            else:
                processed = _parse_processed(row.get("processed"), f"container {c.label!r} [{cid}]")
            sub = (row.get("subassembly") or "").strip()
            if sub:
                parent_label = containers[c.parent].label if c.parent else ""
                if sub != parent_label:
                    mismatches.append(
                        f"container {c.label!r} [{cid}]: BOM subassembly={sub!r}, tree parent={parent_label!r}"
                    )
        else:
            processed = True  # no row (or a stale/renamed label): default true, size guard catches it
        if cid == root_id:
            if row is not None and not processed:
                log.warning("BOM sets processed=false on the root container; ignoring (the root is "
                            "always processed)")
            processed = True
        c.processed = processed

    # Leaves: `processed` has no meaning for a part (warn and ignore); `subassembly` must match the tree parent.
    for leaf_id, label in leaf_label.items():
        row = by_label.get(label)
        if row is None:
            continue
        used_labels.add(label)
        raw = (row.get("processed") or "").strip()
        if raw and not _parse_processed(raw, f"leaf {label!r}"):
            log.warning("BOM sets processed=false on leaf row %r; ignored (processed only applies "
                        "to subassemblies)", label)
        sub = (row.get("subassembly") or "").strip()
        if sub:
            parent_label = containers[leaf_parent[leaf_id]].label
            if sub != parent_label:
                mismatches.append(
                    f"leaf {label!r} [{leaf_id}]: BOM subassembly={sub!r}, tree parent={parent_label!r}"
                )

    if mismatches:
        raise ValueError(
            "BOM 'subassembly' column does not match the STEP tree (BOM and STEP are out of sync):\n"
            + "\n".join(mismatches)
        )

    # BOM labels that match nothing: an error if they carry a `processed` decision, else only a warning.
    unmatched_errors: list[str] = []
    for label in sorted(set(by_label) - used_labels):
        if (by_label[label].get("processed") or "").strip():
            unmatched_errors.append(
                f"BOM label {label!r} sets 'processed' but matches no component in the assembly tree "
                "(stale or misspelled label; a processed/not-processed decision on it would be silently lost)"
            )
        else:
            log.warning("BOM label %r matches no component in the assembly tree", label)
    if unmatched_errors:
        raise ValueError(
            "BOM has unmatched labels with a 'processed' value (BOM and STEP are out of sync):\n"
            + "\n".join(unmatched_errors)
        )


def _apply_trivial_rule(containers: dict[str, Container], root_id: str) -> None:
    """[AUX] A processed container with fewer than 2 components has nothing to order: mark it unprocessed (fused).

    The root is the exception: fewer than 2 components there means nothing to plan, so it raises ValueError.
    """
    root = containers[root_id]
    if len(root.leaf_ids) + len(root.child_ids) < 2:
        raise ValueError("Root container has fewer than 2 components; nothing to plan.")
    for cid, c in containers.items():
        if cid == root_id:
            continue
        if c.processed and len(c.leaf_ids) + len(c.child_ids) < 2:
            log.info("Container %r [%s] has fewer than 2 components; treating as unprocessed "
                     "(no child level)", c.label, cid)
            c.processed = False


def build_hierarchy(doc, bom_rows: Iterable[dict] | None = None, is_leaf: Callable | None = None) -> Hierarchy:
    """[MAIN] Build the Container tree from `doc.RootObjects`, resolving BOM `processed`/`subassembly`.

    Inputs: the document, the BOM rows (list of dicts, or None) and `is_leaf(obj) -> bool`, injectable
    for tests (the default replicates `preprocessing._leaf_components`). Output: the `Hierarchy`, which
    `check_level_sizes` and `levels.solve_level` consume. Raises `NotImplementedError` on an `App::Link`
    (not supported yet), `ValueError` on a BOM/tree mismatch, an invalid `processed` value or too few root
    components, and `RuntimeError` if the walk missed a leaf the filter accepts.

    A single top-level App::Part is the root; otherwise a virtual root `__root__` collects all top-level
    objects. After the walk the leaf set is cross-checked, descendants are collected, the BOM is applied
    and the "fewer than 2 components" rule is enforced.
    """
    is_leaf = is_leaf or _default_is_leaf
    containers: dict[str, Container] = {}
    leaf_label: dict[str, str] = {}
    leaf_parent: dict[str, str] = {}

    # Root: the single top-level App::Part if there is one, else a virtual root over all top-level objects.
    roots = list(getattr(doc, "RootObjects", []))
    top_parts = [o for o in roots if getattr(o, "TypeId", None) == "App::Part"]
    if len(roots) == 1 and top_parts:
        root_obj = roots[0]
        root_id = root_obj.Name
        root = Container(id=root_id, label=root_obj.Label, parent=None, leaf_ids=[], child_ids=[],
                          processed=True)
        containers[root_id] = root
        _walk(root_obj, root, containers, leaf_label, leaf_parent, is_leaf)
    else:
        root_id = "__root__"
        root = Container(id=root_id, label="", parent=None, leaf_ids=[], child_ids=[], processed=True)
        containers[root_id] = root
        _walk(types.SimpleNamespace(Group=roots), root, containers, leaf_label, leaf_parent, is_leaf)

    # Safety check: the walk must reach exactly the leaves the filter finds in the document.
    reached = set(leaf_label)
    expected = _leaf_component_names(doc, is_leaf)
    if reached != expected:
        raise RuntimeError(
            f"Hierarchy walk reached a different leaf set than the leaf filter "
            f"(missing {sorted(expected - reached)}, extra {sorted(reached - expected)}); check for "
            "unsupported group/link types in the assembly tree."
        )

    _fill_all_leaf_ids(containers, root_id)
    _resolve_bom(containers, leaf_label, leaf_parent, bom_rows, root_id)
    _apply_trivial_rule(containers, root_id)
    return Hierarchy(root=root_id, containers=containers, leaf_label=leaf_label, leaf_parent=leaf_parent)


def check_level_sizes(hier: Hierarchy, max_components: int = 20) -> None:
    """[AUX] Raise `LevelSizeError` if any processed level has more than `max_components` direct components.

    A level is the root, or a container reached through processed containers; its size is leaves plus
    child containers. Run before any geometry or collision work, so an oversized level fails fast (the
    fix is `processed = false` in the BOM). Default 20 is the AoG target size (`--max-level-components`).
    """

    def visit(cid: str) -> None:
        c = hier.containers[cid]
        if not c.processed:
            return  # never expanded into its own level; its own size is irrelevant
        count = len(c.leaf_ids) + len(c.child_ids)
        if count > max_components:
            raise LevelSizeError(
                f"Container {c.label!r} [{cid}] has {count} components at its level, over the limit "
                f"{max_components}; set 'processed = false' in the BOM for it (or an ancestor)."
            )
        for child_id in c.child_ids:
            visit(child_id)

    visit(hier.root)
