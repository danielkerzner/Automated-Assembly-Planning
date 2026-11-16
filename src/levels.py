"""Level documents, fusing, and the recursive per-level pipeline run.

Covers levels, fusing and Phase 2 (automatic sub-assemblies).

Inputs
    - The `Hierarchy` (hierarchy.py) and the original FreeCAD document of the STEP.
    - STEP path, BOM path, the BOM labels and stander categories, an output directory.
    - Optional stage configs (Preprocessing, Collision, Optimization, Illustration).
Outputs
    - A `LevelPlan` tree: for each level its preprocessing result, interference tensor, AoG, optimised
      sequence (or "locked"), inseparable pairs and the plans of its child levels. `main.py` turns it into
      plan.json.
    - On disk, per level: `<level_dir>/` with the stage caches (interference, aog, optimization,
      illustrations) and `fused/` BREP caches. The root level lives in `output_dir`, others in
      `output_dir/levels/<container_id>` (Phase 2 groups nest under their parent level).

What this module does
    A "level" is one `Container` plus its direct children (leaves and child containers). It is rendered as
    a flat FreeCAD document in which every child container is ONE `Part::Feature` whose shape is all its
    descendant leaves fused into one solid, so every module downstream of Preprocessing sees a flat
    assembly and needs no change. `solve_level` runs the per-level pipeline on that document and recurses
    into every child container marked `processed` in the BOM.

    Phase 2: if the AoG of a level is not complete, `_solve_pipeline` finds groups of parts that block
    each other (`aog_generator.find_candidate_groups`), merges the best group into a virtual `auto_<n>`
    container (`_pick_winner`, `_commit_merge`), rebuilds the AoG and repeats. Each `auto_<n>` group is
    then planned on its own as a child level. If no group can be found the level ends "locked".
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from hierarchy import Container, Hierarchy

log = logging.getLogger("aap.levels")

FUSED_VERSION = "2"  # str. Part of the fused-shape cache key; bump when the fuse procedure changes.

# A small fuzzy boolean tolerance rescues real-world STEP geometry (curve-on-surface defects,
# near-but-not-quite-touching leaves) that a zero-tolerance fuse() leaves invalid or unorientable;
# same order of magnitude as the other contact tolerances in this codebase (see CLAUDE.md).
_FUSE_FUZZY_TOLERANCE = 0.01  # mm. Fuzzy tolerance of the second fuse attempt.


@dataclass
class LevelPlan:
    """Result of planning one level (output of `solve_level` / `_solve_pipeline`, input of `main._level_to_json`).

    One `LevelPlan` per container that was expanded into its own level, including Phase 2 groups; child
    levels hang from `children`.
    """

    container_id: str  # str. Id of the container this level belongs to (hier.root for the root level).
    label: str  # str. Container label, used in plan.json and to title the illustrations.
    level_dir: Path  # Path. Folder holding this level's stage outputs.
    prep: "object"  # PreprocessingResult. The level's flat document and standers (after Phase 2 merges).
    interference: "object"  # InterferenceResult. The level's interference tensor (after Phase 2 merges).
    aog: "object"  # AoG. The level's assembly graph (complete, or the last incomplete one when locked).
    sequence: "object"  # OptimizationResult | None. The ordered steps; None when status == "locked".
    # dict child container id -> LevelPlan. Plans of the child levels (real sub-assemblies and auto_<n> groups).
    children: dict[str, "LevelPlan"] = field(default_factory=dict)
    # list of (label, label). Part pairs with no free direction at all (only warned, the plan can still be complete).
    inseparable_pairs: list[tuple[str, str]] = field(default_factory=list)
    status: str = "complete"  # str. "complete" or "locked" (Phase 2 found no way out).
    locked_parts: list[str] = field(default_factory=list)  # list[str]. Labels of the stuck parts; only set when locked.


def _fuse_container(container: Container, original_doc, cache_dir: Path):
    """[MAIN] Fuse every leaf under `container` into one solid; returns a FreeCAD shape.

    Input: the container (its `all_leaf_ids` are fused), the document holding the leaves and a cache
    folder. Output: the fused shape, so a sub-assembly can be handled as one rigid component of its
    parent level. Cached as BREP in `<cache_dir>/fused/`, keyed by the leaves' geometry hashes.

    Strategy: exact fuse, then fuzzy fuse (`_FUSE_FUZZY_TOLERANCE`) if the exact one raised or is
    invalid; the result is then cleaned with `removeSplitter`. A result missing or under half of the
    leaves' summed volume is distrusted and replaced by a plain compound of the leaves.
    """
    import Part

    from preprocessing import _global_shape
    from shapes import shape_geometry_hash

    # Collect the leaves' global shapes.
    leaves = []
    for leaf_id in container.all_leaf_ids:
        obj = original_doc.getObject(leaf_id)
        if obj is None:
            raise ValueError(f"Leaf component {leaf_id} (under container {container.label!r}) not "
                             "found in the original document")
        leaves.append(obj)
    shapes = [_global_shape(obj) for obj in leaves]

    # Cache lookup: key = version + geometry hash of every leaf.
    h = hashlib.sha256(f"v{FUSED_VERSION};".encode())
    for shape in shapes:
        h.update(shape_geometry_hash(shape).encode())
    cache_key = h.hexdigest()
    cache_path = Path(cache_dir) / "fused" / f"{container.id}_{cache_key}.brep"

    if cache_path.exists():
        fused = Part.Shape()
        fused.read(str(cache_path))
        log.info("Loaded cached fused shape for container %r [%s] from %s", container.label,
                 container.id, cache_path)
        return fused

    # Fuse: exact first, then fuzzy; accept the first valid, non-null result.
    t0 = time.perf_counter()
    if len(shapes) == 1:
        fused = shapes[0]
    else:
        fused = None
        for attempt, fuzzy in (("exact", 0.0), ("fuzzy", _FUSE_FUZZY_TOLERANCE)):
            try:
                candidate = shapes[0].multiFuse(shapes[1:], fuzzy) if fuzzy else shapes[0].fuse(shapes[1:])
            except Exception as exc:  # noqa: BLE001  (OCC boolean failure on real-world touching geometry)
                log.warning("%s fuse raised for container %r [%s] (%s)", attempt, container.label,
                            container.id, exc)
                continue
            if candidate.isNull() or not candidate.isValid():
                log.warning("%s fuse for container %r [%s] is %s; trying the next strategy",
                            attempt, container.label, container.id,
                            "null" if candidate.isNull() else "invalid")
                continue
            try:
                candidate = candidate.removeSplitter()
            except Exception as exc:  # noqa: BLE001  (removeSplitter is a cosmetic cleanup, not required)
                log.info("removeSplitter() raised for container %r [%s] (%s); keeping the "
                         "un-refined %s fuse result", container.label, container.id, exc, attempt)
            fused = candidate
            break
    # Sanity check on the volume; fall back to a compound if the fuse failed or lost material.
    leaf_volume = sum(s.Volume for s in shapes)
    if fused is None or fused.Volume < 0.5 * leaf_volume:
        if fused is not None:
            log.warning("Fuse suspiciously small for container %r [%s] (fused volume %.3f vs leaf "
                        "sum %.3f mm^3); falling back to a compound", container.label, container.id,
                        fused.Volume, leaf_volume)
        else:
            log.warning("Every fuse strategy failed for container %r [%s]; falling back to a compound",
                        container.label, container.id)
        fused = Part.makeCompound(shapes)
    log.info("Fused container %r [%s]: %d leaves, volume %.3f mm^3 (leaf sum %.3f mm^3), %.2f s",
             container.label, container.id, len(shapes), fused.Volume, leaf_volume,
             time.perf_counter() - t0)

    # Save to the cache (temporary file, then rename).
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.brep")
    fused.exportBrep(str(tmp))
    os.replace(tmp, cache_path)
    return fused


def build_level_document(hier: Hierarchy, container_id: str, original_doc, cache_dir: Path):
    """[MAIN] Build the flat FreeCAD document of one level.

    Inputs: the hierarchy, the container to render, the original document and a cache folder. Output:
    (doc_L, compound_ids, members): the new document, the ids of the components that are fused child
    containers, and {container component id: descendant leaf ids}. The last two go to Preprocessing so it
    knows which components are fused sub-assemblies and what they contain.

    Direct leaves are copied as they are (global placement); each child container becomes one
    `Part::Feature` holding its fused solid (`_fuse_container`).
    """
    import FreeCAD

    from preprocessing import _global_shape

    container = hier.containers[container_id]
    doc_L = FreeCAD.newDocument(f"{original_doc.Name}__{container_id}")
    compound_ids: set[str] = set()
    members: dict[str, list[str]] = {}

    # Direct leaves: copied with their global shape and label.
    for leaf_id in container.leaf_ids:
        leaf = original_doc.getObject(leaf_id)
        if leaf is None:
            raise ValueError(f"Leaf component {leaf_id} not found in the original document")
        feature = doc_L.addObject("Part::Feature", leaf.Name)
        feature.Shape = _global_shape(leaf)
        feature.Label = leaf.Label

    # Child containers: one fused feature each, remembering which leaves it contains.
    for child_id in container.child_ids:
        child = hier.containers[child_id]
        fused = _fuse_container(child, original_doc, cache_dir)
        feature = doc_L.addObject("Part::Feature", child_id)
        feature.Shape = fused
        feature.Label = child.label
        compound_ids.add(child_id)
        members[child_id] = list(child.all_leaf_ids)

    doc_L.recompute()
    return doc_L, compound_ids, members


# ---------------------------------------------------------------------------
# Phase 2: automatic sub-assemblies.
# ---------------------------------------------------------------------------


def _trial_box_feature(doc, name: str, shapes: list):
    """[AUX] Add a bounding-box placeholder `Part::Feature` named `name` for one Phase 2 trial.

    No real boolean fuse (cheap: a candidate may be discarded by the tie-break); it only gives
    `aog_generator.collect_inputs` a bounding box for the travel model. The caller removes it with
    `doc.removeObject(name)` once done; never used for the committed merge.
    """
    import FreeCAD
    import Part

    bbox = None
    for s in shapes:
        bbox = s.BoundBox if bbox is None else bbox.united(s.BoundBox)
    box = Part.makeBox(max(bbox.XLength, 1e-3), max(bbox.YLength, 1e-3), max(bbox.ZLength, 1e-3),
                       FreeCAD.Vector(bbox.XMin, bbox.YMin, bbox.ZMin))
    feature = doc.addObject("Part::Feature", name)
    feature.Shape = box
    doc.recompute()
    return feature


def _trial_aog(doc_L, prep_L, interf, group_ids: list[str], trial_name: str, aog_config):
    """[MAIN] One Phase 2 trial: what would the AoG look like if `group_ids` were merged into one component?

    Input: the level's document, preprocessing result and interference, the member ids, a temporary
    component name and the AoG config. Output: (trial_aog, trial_interf), used only to rank candidates.

    It calls `merge_groups` (tensor only), adds a bbox placeholder (no real fuse) and calls
    `generate_aog` directly, bypassing `build_aog`'s disk cache, since a trial is thrown away at once and
    never persisted. The placeholder is always removed again.
    """
    from aog_generator import collect_inputs, generate_aog
    from interference_identification import merge_groups

    shapes = [doc_L.getObject(gid).Shape for gid in group_ids]
    _trial_box_feature(doc_L, trial_name, shapes)
    try:
        trial_interf = merge_groups(interf, group_ids, trial_name, "(trial)")
        trial_prep = replace(prep_L, reduced_assembly=doc_L)
        inputs = collect_inputs(trial_interf, trial_prep, aog_config)
        trial_aog = generate_aog(inputs, aog_config)
    finally:
        doc_L.removeObject(trial_name)
    return trial_aog, trial_interf


def _collect_candidates(aog, interf, blocking_dead_ends: list[int]) -> list[dict]:
    """[AUX] Every candidate group over the given blocking dead ends, deduplicated.

    Returns dicts {"members": sorted component indices, "occurrences": number of those dead ends the
    group is a candidate for}; `occurrences` is the second tie-break criterion.
    """
    from aog_generator import find_candidate_groups

    tally: dict[frozenset, dict] = {}
    for v in blocking_dead_ends:
        seen_this_dead_end = set()
        for g in find_candidate_groups(aog, interf, v):
            key = frozenset(g["members"])
            if key in seen_this_dead_end:
                continue
            seen_this_dead_end.add(key)
            entry = tally.setdefault(key, {"members": sorted(key), "occurrences": 0})
            entry["occurrences"] += 1
    return list(tally.values())


def _simulate_to_completion(doc_L, prep_L, interf, aog, aog_config, max_depth: int,
                            trial_counter: list[int]) -> tuple[bool, int]:
    """[AUX] Greedy estimate of how many further merges a trial state needs (amended tie-break).

    Repeatedly merges the best candidate by (size, occurrences) only, with no further nested trial
    ranking to keep it bounded, until the AoG completes, no blocking dead end has a candidate (locked) or
    `max_depth` merges were made. Returns (complete, depth).
    """
    from aog_generator import classify_dead_ends

    depth = 0
    while not aog.complete and depth < max_depth:
        kinds = classify_dead_ends(aog, interf)
        blocking = [v for v, k in kinds.items() if k == "blocking"]
        if not blocking:
            return False, depth
        candidates = _collect_candidates(aog, interf, blocking)
        if not candidates:
            return False, depth
        best = min(candidates, key=lambda c: (len(c["members"]), -c["occurrences"], c["members"]))
        group_ids = [interf.component_ids[c] for c in best["members"]]
        trial_name = f"__sim_{trial_counter[0]}"
        trial_counter[0] += 1
        aog, interf = _trial_aog(doc_L, prep_L, interf, group_ids, trial_name, aog_config)
        depth += 1
    return bool(aog.complete), depth


def _pick_winner(doc_L, prep_L, interf, aog_config, candidates: list[dict]) -> list[int] | None:
    """[MAIN] Choose which candidate group to merge (amended tie-break).

    Input: the level state and the candidate groups from `_collect_candidates`. Output: the winning group
    as movable component indices, or None if there are no candidates (the level is then locked).

    Ranking: smallest group first, then most dead-end occurrences. A single survivor is taken without
    trials. Otherwise every tied candidate is trial-merged (no real fuse) and the winner is the one whose
    AoG completes, then the one needing the fewest further merges (greedy simulation), and only then the
    one with the smallest sorted member ids. Every trial is logged.
    """
    if not candidates:
        return None
    # Tier 1 and 2: smallest size, then most occurrences.
    best_size = min(len(c["members"]) for c in candidates)
    tier1 = [c for c in candidates if len(c["members"]) == best_size]
    best_occ = max(c["occurrences"] for c in tier1)
    tier2 = sorted((c for c in tier1 if c["occurrences"] == best_occ), key=lambda c: c["members"])
    if len(tier2) == 1:
        log.info("Phase 2: single candidate group %s (size %d, %d occurrence(s)), no trial needed",
                 [interf.component_labels[c] for c in tier2[0]["members"]], best_size, best_occ)
        return tier2[0]["members"]

    # Tie: trial-merge every tied candidate and measure how far it gets.
    trial_counter = [0]
    results = []
    for cand in tier2:
        group_ids = [interf.component_ids[c] for c in cand["members"]]
        labels = [interf.component_labels[c] for c in cand["members"]]
        trial_name = f"__trial_{trial_counter[0]}"
        trial_counter[0] += 1
        trial_aog, trial_interf = _trial_aog(doc_L, prep_L, interf, group_ids, trial_name, aog_config)
        if trial_aog.complete:
            complete, depth = True, 0
        else:
            complete, sim_depth = _simulate_to_completion(
                doc_L, prep_L, trial_interf, trial_aog, aog_config,
                max_depth=len(interf.component_ids), trial_counter=trial_counter)
            complete, depth = complete, sim_depth + 1
        results.append((cand, complete, depth))
        log.info("Phase 2 trial: merging %s -> AoG complete=%s (merges to completion=%s)",
                 labels, complete, depth if complete else "locked")

    # Best = completes, then fewest merges, then sorted member ids.
    results.sort(key=lambda r: (not r[1], r[2] if r[1] else float("inf"), r[0]["members"]))
    winner = results[0][0]
    log.info("Phase 2: chose group %s (tie-break: %s)",
             [interf.component_labels[c] for c in winner["members"]],
             "completes" if results[0][1] else "fewest/sorted (none of the tied candidates complete alone)")
    return winner["members"]


def _commit_merge(doc_L, prep_L, interf, group_ids: list[str], new_id: str, new_label: str,
                  level_dir: Path):
    """[MAIN] Commit one Phase 2 merge for real: replace the group by one fused component in every level structure.

    Input: the level's document, preprocessing result and interference, the member ids and the new
    virtual component's id and label. Output: (new_interf, new_prep, new_doc, group_shapes), the level
    after the merge. `group_shapes` (original member id -> global Shape, captured before removal) is what
    the group's own child level is built from later.

    Steps: fuse the members (`_fuse_container` with a synthetic `Container`; `doc_L`'s members are already
    globally placed, so it plays the `original_doc` role unchanged); build a new flat document with the
    group replaced by that one feature; merge the tensor (`merge_groups`); remap the stander gates;
    update `compound_ids` / `members`; and give the new component its M7 group.
    """
    import FreeCAD

    from interference_identification import merge_groups

    # Fuse the members into one solid.
    group_shapes = {gid: doc_L.getObject(gid).Shape.copy() for gid in group_ids}

    synth = Container(id=new_id, label=new_label, parent=prep_L.level_id, leaf_ids=list(group_ids),
                      child_ids=[], processed=True, all_leaf_ids=list(group_ids), auto=True)
    fused = _fuse_container(synth, doc_L, level_dir)

    # New flat document: every non-member as it was, plus the fused group.
    new_doc = FreeCAD.newDocument(f"{doc_L.Name}__{new_id}")
    group_set = set(group_ids)
    for obj in doc_L.Objects:
        if obj.Name in group_set:
            continue
        feature = new_doc.addObject("Part::Feature", obj.Name)
        feature.Shape = obj.Shape.copy()
        feature.Label = obj.Label
    feature = new_doc.addObject("Part::Feature", new_id)
    feature.Shape = fused
    feature.Label = new_label
    new_doc.recompute()

    # Interference tensor with the group collapsed into new_id.
    new_interf = merge_groups(interf, group_ids, new_id, new_label)

    # Stander gates: every reference to a member now points to the new component.
    new_stander_records = {}
    for sid, rec in prep_L.stander_records.items():
        new_options = [
            replace(opt, blockers={new_id if m in group_set else m for m in opt.blockers})
            for opt in rec.options
        ]
        new_stander_records[sid] = replace(
            rec,
            options=new_options,
            holds={new_id if m in group_set else m for m in rec.holds},
            contact_neighbors={new_id if m in group_set else m for m in rec.contact_neighbors},
            partner=(new_id if rec.partner in group_set else rec.partner),
        )

    # Fused-component bookkeeping: new_id is a fused component made of the members' leaves.
    new_compound_ids = (set(prep_L.compound_ids) - group_set) | {new_id}
    new_members = {k: v for k, v in prep_L.members.items() if k not in group_set}
    new_members[new_id] = sorted({leaf for gid in group_ids for leaf in prep_L.members.get(gid, [gid])})

    # M7 group inheritance (optimization.py's collect_tables):
    # new_id's own BOM `group` -- the direct members' shared group if they agree (including a
    # member that's itself an earlier auto_<n>, via its own override), else new_id itself, logged.
    bom_rows = {}
    if prep_L.bom_path is not None:
        from optimization import _read_bom
        _, bom_rows, _ = _read_bom(prep_L.bom_path)
    resolved_groups = {}
    for gid in group_ids:
        glabel = interf.component_labels[interf.index_of(gid)]
        resolved_groups[gid] = prep_L.auto_group_override.get(gid) or bom_rows.get(glabel, {}).get("group") or glabel
    if len(set(resolved_groups.values())) == 1:
        auto_group_value = next(iter(resolved_groups.values()))
    else:
        auto_group_value = new_id
        log.warning("Phase 2 group %r's direct members do not share one BOM group (%s); %r will "
                   "use its own id as its M7 group", new_id, resolved_groups, new_id)
    new_auto_group_override = {**prep_L.auto_group_override, new_id: auto_group_value}

    new_prep = replace(prep_L, reduced_assembly=new_doc, stander_records=new_stander_records,
                       compound_ids=frozenset(new_compound_ids), members=new_members,
                       auto_group_override=new_auto_group_override)

    return new_interf, new_prep, new_doc, group_shapes


def _group_child_prep(prep_L, group_ids: list[str], group_shapes: dict, new_id: str,
                      new_label: str):
    """[AUX] PreprocessingResult for a Phase 2 group's own child level.

    The document is built from the captured member shapes (same global placement as every other level,
    not re-derived). The parent's candidate-direction list is reused (not rediscovered) and
    `assembly_bbox` is the union of the members' boxes. Only the stander records whose whole gate lies
    inside the group are kept; cross-group gates are dropped, the same rule as a real container's
    cross-container gates (not supported yet).
    """
    import FreeCAD

    group_set = set(group_ids)
    doc_child = FreeCAD.newDocument(f"{prep_L.reduced_assembly.Name}__{new_id}")
    for gid in group_ids:
        feature = doc_child.addObject("Part::Feature", gid)
        feature.Shape = group_shapes[gid].copy()
        feature.Label = gid
    doc_child.recompute()

    child_stander_records = {}
    for sid, rec in prep_L.stander_records.items():
        refs = set(rec.contact_neighbors) | set(rec.holds)
        if rec.partner:
            refs.add(rec.partner)
        for opt in rec.options:
            refs |= opt.blockers
        if sid in group_set and refs <= group_set:
            child_stander_records[sid] = rec
        elif sid in group_set:
            log.warning("Stander %r's gate reaches outside its Phase 2 group %s (%s); dropped at "
                       "the group's own level", sid, new_id, sorted(refs - group_set))

    bbox = None
    for gid in group_ids:
        b = group_shapes[gid].BoundBox
        bbox = b if bbox is None else bbox.united(b)
    assembly_bbox = (FreeCAD.Vector(bbox.XMin, bbox.YMin, bbox.ZMin),
                     FreeCAD.Vector(bbox.XMax, bbox.YMax, bbox.ZMax))

    child_members = {k: v for k, v in prep_L.members.items() if k in group_set}

    return replace(
        prep_L, reduced_assembly=doc_child, stander_records=child_stander_records,
        assembly_bbox=assembly_bbox, level_id=new_id, level_label=new_label,
        compound_ids=frozenset(prep_L.compound_ids & group_set), members=child_members,
    )


def _level_output_dir(output_dir: Path, hier: Hierarchy, container_id: str) -> Path:
    """[AUX] Output folder of a level: `output_dir` for the root, `output_dir/levels/<container_id>` otherwise."""
    return output_dir if container_id == hier.root else output_dir / "levels" / container_id


def _solve_pipeline(hier: Hierarchy, container_id: str, label: str, prep_L, interf, level_dir: Path,
                    illustration_config=None, optimization_config=None) -> LevelPlan:
    """[MAIN] AoG -> Phase 2 merge loop (if needed) -> Optimization -> Illustrations for one level.

    Input: a level that already has its Preprocessing and Interference results (a real container or a
    Phase 2 group), plus the output folder and the optional Optimization / Illustration configs. Output:
    the level's `LevelPlan`, with the plans of every `auto_<n>` group created here in `children`
    (nested Phase 2 is allowed: a group's own level can trigger another merge loop through the same
    recursive call). A level that cannot be completed returns status "locked" without a sequence.
    """
    from aog_generator import AoGConfig, build_aog, classify_dead_ends
    from illustration_generator import IllustrationConfig, generate_illustrations
    from interference_identification import sub_interference
    from optimization import optimize_sequence

    aog_config = AoGConfig()
    level_dir.mkdir(parents=True, exist_ok=True)
    # Only used inside the merge loop below; a stubbed PreprocessingResult (no FreeCAD, used by
    # tests that fake build_aog to always be complete) never needs it.
    doc_L = getattr(prep_L, "reduced_assembly", None)
    aog = build_aog(interf, prep_L, config=aog_config, cache_dir=level_dir / "aog")

    status = "complete"
    locked_parts: list[str] = []
    # (new_id, new_label, group_ids, group_shapes, child_interf, prep_before_this_merge)
    auto_groups: list[tuple] = []
    n_auto = 0

    # Phase 2 loop: while the AoG cannot reach the full assembly, merge one group of mutually blocking
    # parts into an auto_<n> component and rebuild the AoG. Stops "locked" if no group can help.
    while not aog.complete:
        kinds = classify_dead_ends(aog, interf)
        blocking = [v for v, k in kinds.items() if k == "blocking"]
        if not blocking:
            status = "locked"
            locked_parts = sorted({aog.component_labels[c] for v in kinds for c in aog.state_components(v)})
            log.warning("Level %r [%s] is locked: every remaining dead end is rule-induced (a "
                       "stander gate, D11, or D12/D13 conflict, not an interference block) -- "
                       "Phase 2 cannot help. Stuck parts: %s", label, container_id, locked_parts)
            break
        candidates = _collect_candidates(aog, interf, blocking)
        winner = _pick_winner(doc_L, prep_L, interf, aog_config, candidates)
        if winner is None:
            status = "locked"
            locked_parts = sorted({aog.component_labels[c] for v in blocking
                                   for c in aog.state_components(v)})
            log.warning("Level %r [%s] is locked: no closed group found for any blocking dead "
                       "end. Stuck parts: %s", label, container_id, locked_parts)
            break

        group_ids = [interf.component_ids[c] for c in winner]
        group_labels = [interf.component_labels[c] for c in winner]
        new_id = f"auto_{n_auto}"
        n_auto += 1
        new_label = f"{new_id} ({' + '.join(sorted(group_labels))})"
        log.info("Level %r [%s]: merging %s into %r", label, container_id, group_labels, new_id)

        # Commit the merge; keep what the group's own child level will need.
        child_interf = sub_interference(interf, group_ids)  # before the group's ids are merged away
        pre_merge_prep = prep_L
        interf, prep_L, doc_L, group_shapes = _commit_merge(
            doc_L, prep_L, interf, group_ids, new_id, new_label, level_dir)
        auto_groups.append((new_id, new_label, group_ids, group_shapes, child_interf, pre_merge_prep))

        # A group member may itself be a real (or an earlier auto_<n>) container rather than a
        # plain leaf -- expand to real leaves so all_leaf_ids keeps its usual "every descendant
        # leaf" meaning (used by main.py's plan.json and by _group_child_prep's bbox union).
        all_leaf_ids = []
        for gid in group_ids:
            c = hier.containers.get(gid)
            all_leaf_ids.extend(c.all_leaf_ids if c is not None else [gid])
        hier.containers[new_id] = Container(
            id=new_id, label=new_label, parent=container_id, leaf_ids=list(group_ids),
            child_ids=[], processed=True, all_leaf_ids=all_leaf_ids, auto=True,
        )
        hier.containers[container_id].child_ids.append(new_id)

        aog = build_aog(interf, prep_L, config=aog_config, cache_dir=level_dir / "aog")

    if status == "locked":
        return LevelPlan(
            container_id=container_id, label=label, level_dir=level_dir,
            prep=prep_L, interference=interf, aog=aog, sequence=None,
            status="locked", locked_parts=locked_parts,
        )

    # Sequence optimisation (ACO) and illustrations for the complete AoG.
    sequence = optimize_sequence(aog, interf, prep_L, optimization_config, cache_dir=level_dir / "optimization")

    base_ill_config = illustration_config or IllustrationConfig()
    title = label or base_ill_config.title
    generate_illustrations(sequence, prep_L, level_dir / "illustrations",
                           config=replace(base_ill_config, title=title))

    # Report part pairs with no free direction at all (warning only).
    inseparable_pairs: list[tuple[str, str]] = []
    n = len(interf.component_ids)
    for i in range(n):
        for j in range(i + 1, n):
            if not interf.free[i, j, :].any():
                pair = (interf.component_labels[i], interf.component_labels[j])
                inseparable_pairs.append(pair)
                log.warning("Inseparable by translation (or missing candidate direction): %s / %s", *pair)

    plan = LevelPlan(
        container_id=container_id, label=label, level_dir=level_dir,
        prep=prep_L, interference=interf, aog=aog, sequence=sequence,
        inseparable_pairs=inseparable_pairs,
    )
    # Every auto_<n> group is planned as its own child level (this may trigger further Phase 2 inside it).
    for new_id, new_label, group_ids, group_shapes, child_interf, pre_merge_prep in auto_groups:
        child_prep = _group_child_prep(pre_merge_prep, group_ids, group_shapes, new_id, new_label)
        plan.children[new_id] = _solve_pipeline(
            hier, new_id, new_label, child_prep, child_interf,
            level_dir / "levels" / new_id, illustration_config=illustration_config,
            optimization_config=optimization_config,
        )
    return plan


def solve_level(
    hier: Hierarchy,
    container_id: str,
    original_doc,
    *,
    step_path,
    bom_path,
    bom_labels: set[str],
    output_dir: Path,
    bom_categories: dict | None = None,
    preprocessing_config=None,
    collision_config=None,
    illustration_config=None,
    optimization_config=None,
    force_interference: bool = False,
) -> LevelPlan:
    """[MAIN] Plan one level and, depth-first, every child container marked `processed`.

    Inputs: the hierarchy, the container to plan, the original document, STEP and BOM paths, the BOM
    labels and stander categories, the root output folder, optional stage configs and
    `force_interference`. Output: the level's `LevelPlan`, with child plans in `children`.

    Steps: build the level's flat document, run Preprocessing and Interference Identification on it, then
    `_solve_pipeline` (AoG, Phase 2, Optimization, Illustrations). Only a complete level recurses, and
    only into real child containers: Phase 2 `auto_<n>` groups are handled inside `_solve_pipeline`.
    """
    from interference_identification import identify_interferences
    from preprocessing import run_preprocessing_on_document

    container = hier.containers[container_id]
    level_dir = _level_output_dir(Path(output_dir), hier, container_id)
    level_dir.mkdir(parents=True, exist_ok=True)

    # Flat document of this level, then Preprocessing and the interference tensor on it.
    doc_L, compound_ids, members = build_level_document(hier, container_id, original_doc, level_dir)
    prep_L = run_preprocessing_on_document(
        doc_L, bom_labels, preprocessing_config,
        collision_config=collision_config, bom_categories=bom_categories,
        step_path=step_path, bom_path=bom_path,
        level_id=container_id, level_label=container.label,
        compound_ids=frozenset(compound_ids), members=members,
    )
    interf = identify_interferences(prep_L, collision_config, level_dir / "interference",
                                    force=force_interference)

    plan = _solve_pipeline(hier, container_id, container.label, prep_L, interf, level_dir,
                           illustration_config=illustration_config,
                           optimization_config=optimization_config)

    # Recurse into the real child containers the BOM marked processed (not Phase 2 groups).
    if plan.status == "complete":
        for child_id in container.child_ids:
            if hier.containers[child_id].processed and not hier.containers[child_id].auto:
                plan.children[child_id] = solve_level(
                    hier, child_id, original_doc,
                    step_path=step_path, bom_path=bom_path, bom_labels=bom_labels,
                    output_dir=output_dir, bom_categories=bom_categories,
                    preprocessing_config=preprocessing_config, collision_config=collision_config,
                    illustration_config=illustration_config, optimization_config=optimization_config,
                    force_interference=force_interference,
                )
    return plan
