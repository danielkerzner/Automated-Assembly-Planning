"""Level documents, fusing, and the recursive per-level pipeline run.

Implements docs/subassembly-spec.md section 5. A "level" is one Container and its direct
children (leaves and child containers); `build_level_document` renders it as a flat FreeCAD
document (every direct child container appears as one `Part::Feature` whose shape is its
descendant leaves fused into one solid), so every existing module downstream of Preprocessing
sees a flat assembly and needs no change. `solve_level` runs the existing pipeline on that
document and recurses into every child container marked `processed` in the BOM.
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

FUSED_VERSION = "2"

# A small fuzzy boolean tolerance rescues real-world STEP geometry (curve-on-surface defects,
# near-but-not-quite-touching leaves) that a zero-tolerance fuse() leaves invalid or unorientable;
# same order of magnitude as the other contact tolerances in this codebase (see CLAUDE.md).
_FUSE_FUZZY_TOLERANCE = 0.01


@dataclass
class LevelPlan:
    container_id: str
    label: str
    level_dir: Path
    prep: "object"  # PreprocessingResult
    interference: "object"  # InterferenceResult
    aog: "object"  # AoG
    sequence: "object"  # OptimizationResult | None (None when status == "locked")
    children: dict[str, "LevelPlan"] = field(default_factory=dict)
    inseparable_pairs: list[tuple[str, str]] = field(default_factory=list)
    status: str = "complete"  # "complete" | "locked" (Phase 2, spec section 10)
    locked_parts: list[str] = field(default_factory=list)  # labels, only set when locked


def _fuse_container(container: Container, original_doc, cache_dir: Path):
    """Fused solid of every leaf under `container` (spec 5). Cached as BREP, keyed by the leaves'
    geometry hashes; falls back to a compound if the fuse fails or looks suspiciously small."""
    import Part

    from preprocessing import _global_shape
    from shapes import shape_geometry_hash

    leaves = []
    for leaf_id in container.all_leaf_ids:
        obj = original_doc.getObject(leaf_id)
        if obj is None:
            raise ValueError(f"Leaf component {leaf_id} (under container {container.label!r}) not "
                             "found in the original document")
        leaves.append(obj)
    shapes = [_global_shape(obj) for obj in leaves]

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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.brep")
    fused.exportBrep(str(tmp))
    os.replace(tmp, cache_path)
    return fused


def build_level_document(hier: Hierarchy, container_id: str, original_doc, cache_dir: Path):
    """Flat FreeCAD document for one level: leaves copied as-is, child containers fused into one
    `Part::Feature` each. Returns (doc_L, compound_ids, members): the ids of the fused-container
    components and {container component id: descendant leaf ids}."""
    import FreeCAD

    from preprocessing import _global_shape

    container = hier.containers[container_id]
    doc_L = FreeCAD.newDocument(f"{original_doc.Name}__{container_id}")
    compound_ids: set[str] = set()
    members: dict[str, list[str]] = {}

    for leaf_id in container.leaf_ids:
        leaf = original_doc.getObject(leaf_id)
        if leaf is None:
            raise ValueError(f"Leaf component {leaf_id} not found in the original document")
        feature = doc_L.addObject("Part::Feature", leaf.Name)
        feature.Shape = _global_shape(leaf)
        feature.Label = leaf.Label

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
# Phase 2 (docs/subassembly-spec.md section 10): automatic sub-assemblies.
# ---------------------------------------------------------------------------


def _trial_box_feature(doc, name: str, shapes: list):
    """A bbox-matching placeholder `Part::Feature` for one Phase 2 trial -- no real boolean fuse
    (cheap: a candidate may be discarded by the tie-break), just enough for `aog_generator.
    collect_inputs` to read a bounding box for the travel model. Caller removes it with
    `doc.removeObject(name)` once done; never used for the committed merge."""
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
    """One Phase 2 trial: `merge_groups` (tensor only) + a bbox placeholder (no real fuse) +
    `generate_aog` called directly, bypassing `build_aog`'s disk cache entirely -- a trial is
    thrown away immediately after, never persisted. Returns (trial_aog, trial_interf)."""
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
    """Every candidate group across the given blocking dead ends, deduplicated, each with how many
    of those dead ends it is a candidate for (spec section 10's tie-break)."""
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
    """Greedy estimate of how many further merges a trial state needs (amended tie-break): repeat
    picking the best-ranked candidate by (size, occurrences) only -- no further nested trial-
    ranking, to keep this bounded -- until the AoG completes or no blocking dead end has a
    candidate group (locked). Returns (complete, depth)."""
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
    """Rank Phase 2 merge candidates (spec section 10, amended tie-break): smallest size, then
    most dead-end occurrences; within that tier, trial-merge each (no real fuse) and prefer one
    whose AoG completes, then the one needing the fewest further merges (greedy simulation), and
    only then sorted component ids. Logs every trial. Returns the winning group as movable
    component indices, or None if there are no candidates at all."""
    if not candidates:
        return None
    best_size = min(len(c["members"]) for c in candidates)
    tier1 = [c for c in candidates if len(c["members"]) == best_size]
    best_occ = max(c["occurrences"] for c in tier1)
    tier2 = sorted((c for c in tier1 if c["occurrences"] == best_occ), key=lambda c: c["members"])
    if len(tier2) == 1:
        log.info("Phase 2: single candidate group %s (size %d, %d occurrence(s)), no trial needed",
                 [interf.component_labels[c] for c in tier2[0]["members"]], best_size, best_occ)
        return tier2[0]["members"]

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

    results.sort(key=lambda r: (not r[1], r[2] if r[1] else float("inf"), r[0]["members"]))
    winner = results[0][0]
    log.info("Phase 2: chose group %s (tie-break: %s)",
             [interf.component_labels[c] for c in winner["members"]],
             "completes" if results[0][1] else "fewest/sorted (none of the tied candidates complete alone)")
    return winner["members"]


def _commit_merge(doc_L, prep_L, interf, group_ids: list[str], new_id: str, new_label: str,
                  level_dir: Path):
    """Commit one Phase 2 merge for real: a fused solid (`_fuse_container`, reusing a synthetic
    `Container` -- `doc_L`'s members are already globally placed, same as any other level, so it
    plays the `original_doc` role unchanged), a new flat level document with the group replaced by
    that one feature, `merge_groups` for the tensor, and stander gate remapping. Returns
    (new_interf, new_prep, new_doc, group_shapes) -- `group_shapes` (original member id -> global
    Shape, captured before removal) is what the group's own child level is built from."""
    import FreeCAD

    from interference_identification import merge_groups

    group_shapes = {gid: doc_L.getObject(gid).Shape.copy() for gid in group_ids}

    synth = Container(id=new_id, label=new_label, parent=prep_L.level_id, leaf_ids=list(group_ids),
                      child_ids=[], processed=True, all_leaf_ids=list(group_ids), auto=True)
    fused = _fuse_container(synth, doc_L, level_dir)

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

    new_interf = merge_groups(interf, group_ids, new_id, new_label)

    new_stander_records = {}
    for sid, rec in prep_L.stander_records.items():
        new_stander_records[sid] = replace(
            rec,
            prerequisites={new_id if m in group_set else m for m in rec.prerequisites},
            contact_neighbors={new_id if m in group_set else m for m in rec.contact_neighbors},
        )

    new_compound_ids = (set(prep_L.compound_ids) - group_set) | {new_id}
    new_members = {k: v for k, v in prep_L.members.items() if k not in group_set}
    new_members[new_id] = sorted({leaf for gid in group_ids for leaf in prep_L.members.get(gid, [gid])})

    # M7 group inheritance (optimization.py's collect_tables, docs/subassembly-spec.md section
    # 10): new_id's own BOM `group` -- the direct members' shared group if they agree (including a
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
    """PreprocessingResult for a Phase 2 group's own child level: a flat document built from the
    captured member shapes (same global placement as every other level, not re-derived), the
    parent's candidate-direction list and assembly_bbox (not rediscovered, per spec), and only the
    stander records whose whole gate lies inside the group (cross-group gates dropped, same rule
    as a real container's cross-container gates -- open item 5)."""
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
        gate = rec.prerequisites | rec.contact_neighbors
        if sid in group_set and gate <= group_set:
            child_stander_records[sid] = rec
        elif sid in group_set:
            log.warning("Stander %r's gate reaches outside its Phase 2 group %s (%s); dropped at "
                       "the group's own level", sid, new_id, sorted(gate - group_set))

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
    return output_dir if container_id == hier.root else output_dir / "levels" / container_id


def _solve_pipeline(hier: Hierarchy, container_id: str, label: str, prep_L, interf, level_dir: Path,
                    illustration_config=None, optimization_config=None) -> LevelPlan:
    """AoG -> (Phase 2 merge loop, if needed) -> Optimization -> Illustrations for one level that
    already has its Preprocessing + Interference result (a real container or a Phase 2 group), then
    recurses into every `auto_<n>` group created at this level (nested Phase 2 is allowed: a
    group's own child level can trigger another merge loop, via the same recursive call)."""
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

    sequence = optimize_sequence(aog, interf, prep_L, optimization_config, cache_dir=level_dir / "optimization")

    base_ill_config = illustration_config or IllustrationConfig()
    title = label or base_ill_config.title
    generate_illustrations(sequence, prep_L, level_dir / "illustrations",
                           config=replace(base_ill_config, title=title))

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
    preprocessing_config=None,
    collision_config=None,
    illustration_config=None,
    optimization_config=None,
    force_interference: bool = False,
) -> LevelPlan:
    """Run Preprocessing -> Interference -> AoG (incl. Phase 2) -> Optimization -> Illustrations
    for one level, then recurse depth-first into every child container marked `processed`."""
    from interference_identification import identify_interferences
    from preprocessing import run_preprocessing_on_document

    container = hier.containers[container_id]
    level_dir = _level_output_dir(Path(output_dir), hier, container_id)
    level_dir.mkdir(parents=True, exist_ok=True)

    doc_L, compound_ids, members = build_level_document(hier, container_id, original_doc, level_dir)
    prep_L = run_preprocessing_on_document(
        doc_L, bom_labels, preprocessing_config,
        step_path=step_path, bom_path=bom_path,
        level_id=container_id, level_label=container.label,
        compound_ids=frozenset(compound_ids), members=members,
    )
    interf = identify_interferences(prep_L, collision_config, level_dir / "interference",
                                    force=force_interference)

    plan = _solve_pipeline(hier, container_id, container.label, prep_L, interf, level_dir,
                           illustration_config=illustration_config,
                           optimization_config=optimization_config)

    if plan.status == "complete":
        for child_id in container.child_ids:
            if hier.containers[child_id].processed and not hier.containers[child_id].auto:
                plan.children[child_id] = solve_level(
                    hier, child_id, original_doc,
                    step_path=step_path, bom_path=bom_path, bom_labels=bom_labels,
                    output_dir=output_dir,
                    preprocessing_config=preprocessing_config, collision_config=collision_config,
                    illustration_config=illustration_config, optimization_config=optimization_config,
                    force_interference=force_interference,
                )
    return plan
