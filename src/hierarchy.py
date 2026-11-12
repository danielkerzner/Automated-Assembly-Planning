"""Sub-assembly hierarchy extraction: Containers (App::Part) and the Hierarchy tree.

Implements docs/subassembly-spec.md section 4. `build_hierarchy` walks a FreeCAD document's
`RootObjects` through `Group`, duck-typed over objects exposing `TypeId`, `Name`, `Label` and
`Group` (so tests need no FreeCAD). A `Container` is one App::Part; a leaf is whatever `is_leaf`
accepts (default: the `_leaf_components` filter). Other group types (e.g.
`App::DocumentObjectGroup`) are transparent, same convention as `generate_bom.collect_rows`.

The BOM's `processed`/`subassembly` columns are resolved here against rows passed in as plain
dicts (caller reads the CSV), and `check_level_sizes` guards against an oversized level before
any geometry work runs.
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass, field
from typing import Callable, Iterable

log = logging.getLogger("aap.hierarchy")

_TRUE = {"1", "true", "yes", "y", "t"}
_FALSE = {"0", "false", "no", "n", "f"}


class LevelSizeError(RuntimeError):
    """A processed level has more components than the configured limit."""


@dataclass
class Container:
    id: str  # obj.Name; "__root__" for a virtual root
    label: str
    parent: str | None
    leaf_ids: list[str]  # direct leaf children (obj.Name), document order
    child_ids: list[str]  # direct child containers
    processed: bool
    all_leaf_ids: list[str] = field(default_factory=list)  # every descendant leaf, for fusing
    auto: bool = False  # Phase 2: a virtual container (auto_<n>) merged by levels.py, not a real
                         # STEP/BOM sub-assembly (docs/subassembly-spec.md section 10)


@dataclass
class Hierarchy:
    root: str
    containers: dict[str, Container]
    leaf_label: dict[str, str]
    leaf_parent: dict[str, str]  # leaf id -> immediate parent container id


def _default_is_leaf(obj) -> bool:
    """Same filter as `preprocessing._leaf_components`, applied to one object."""
    return (
        getattr(obj, "TypeId", None) == "Part::Feature"
        and getattr(obj, "Shape", None) is not None
        and not obj.Shape.isNull()
        and len(obj.Shape.Solids) > 0
    )


def _leaf_component_names(doc, is_leaf: Callable) -> set[str]:
    return {obj.Name for obj in getattr(doc, "Objects", []) if is_leaf(obj)}


def _walk(obj, container: Container, containers: dict[str, Container], leaf_label: dict[str, str],
          leaf_parent: dict[str, str], is_leaf: Callable) -> None:
    """Fill `container`'s leaf_ids/child_ids from obj's direct Group, recursing into nested
    containers and transparently through other group types."""
    for child in getattr(obj, "Group", None) or []:
        type_id = getattr(child, "TypeId", None)
        if type_id == "App::Link":
            raise NotImplementedError(
                f"Repeated sub-assembly (App::Link) {getattr(child, 'Label', child)!r} is not "
                "supported yet; see docs/subassembly-spec.md open item 4."
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
    v = (value or "").strip().lower()
    if v == "":
        return True
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"Invalid 'processed' value {value!r} for {context}")


def _index_bom_rows(bom_rows: Iterable[dict] | None) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in bom_rows or []:
        label = (row.get("label") or "").strip()
        if label:
            index[label] = row
    return index


def _resolve_bom(containers: dict[str, Container], leaf_label: dict[str, str], leaf_parent: dict[str, str],
                 bom_rows: Iterable[dict] | None, root_id: str) -> None:
    by_label = _index_bom_rows(bom_rows)
    used_labels: set[str] = set()
    mismatches: list[str] = []

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
                    "(docs/preprocessing-spec.md P1/P2, 'other' category) and cannot also be "
                    "recursively processed as its own level -- set processed=false or leave it blank"
                )
            if is_stander:
                # A stander subassembly is always fused as a single rigid component: unlike a plain
                # container, blank 'processed' here defaults to false, not true (docs/subassembly-
                # spec.md section 3).
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
    """Build the Container tree from `doc.RootObjects`, resolving BOM `processed`/`subassembly`.

    `is_leaf(obj) -> bool` is injectable for tests; the default replicates
    `preprocessing._leaf_components`. Raises `NotImplementedError` on an `App::Link` (open item 4),
    `ValueError` on a BOM/tree mismatch, an invalid `processed` value, or too few root components.
    """
    is_leaf = is_leaf or _default_is_leaf
    containers: dict[str, Container] = {}
    leaf_label: dict[str, str] = {}
    leaf_parent: dict[str, str] = {}

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
    """Raise `LevelSizeError` if any processed level (root, or reached through processed containers)
    has more than `max_components` direct components. Run before any geometry or collision work."""

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
