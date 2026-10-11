# Preprocessing Module — Implementation Spec

Status: reviewed and design-complete as of 2026-09-21. This is the module to implement first.

This spec supersedes the "Preprocessing" bullet points in `architecture-overview.md` — read this file for
the actual algorithm; the overview doc keeps the one-paragraph summary and points here for detail.

## Correction to the workflow diagram

`Individual_Workflows.png` draws Preprocessing's two halves (stander handling, candidate-direction discovery)
as a fork/join — i.e. as if they ran in parallel and independently. **They do not.** Candidate-direction
discovery must run on the assembly *after* standers have been removed (see rationale below), so the real
control flow is sequential:

```
1. Stander handling  →  produces the stander-reduced assembly
2. Candidate-direction discovery  →  runs on the stander-reduced assembly, produces the unique direction list
```

Do not implement these as concurrent branches.

## Part 1 — Stander handling

**Input:** the full assembly (STEP) + an external BOM that flags which components are "stander" parts
(screws, nuts, sealing rings — small fasteners/seals that are not part of the main disassembly sequence).

For each stander component:

1. **Identify it** via the BOM flag (not geometric inference).
2. **Extract its cylindrical axis** using the CAD kernel (the shaft/bore cylindrical feature). STEP has no
   thread data, so the axis is the only geometric handle available — threads are explicitly out of scope.
3. **Determine dismantle direction by trial in both directions along the axis:**
   - For each of the two directions, walk outward along the axis and collect every *distinct* component
     that falls within a radial distance threshold of the stander's shaft (a tunable config parameter —
     start with a small value, calibrate against the Vulkan case study).
   - Count the number of distinct components collided with in each direction.
   - The direction with the **fewer** collided components is the dismantle direction.
   - **Tie-break:** if both directions collide with the same number of components, pick the direction with
     the smaller **total swept volume**.
   - Rationale: this is a precedence/contact check, not a real removal-direction search — e.g. for a screw
     joining two plates, it tells you those plates must already be present before the screw is placed. It
     naturally converges on "exit through the head side" without explicitly detecting which end is the head,
     since the head side almost always has fewer obstructions.
4. **Store, per stander:**
   - `prerequisites`: the set of distinct components collided with along the *chosen* (winning) direction
     only. Per `docs/aog-spec.md` (D4), the stander is assembled *after* these components, so in disassembly
     it must be removed *before* them, while they are still present — not "removed once they're gone."
     Together with `contact_neighbors`, this forms the stander's gate set the AoG generator uses for its
     precedence rule (`docs/aog-spec.md`, "Model").
   - `exit_point`: the point where the stander's trajectory crosses the assembly's outer bounding box along
     the chosen direction. Computed once here so the Illustration Generator doesn't need to recompute it.
   - `contact_neighbors`: the set of *all* non-stander components physically touching the stander (minimum
     surface-to-surface distance below a tunable `contact_threshold`), independent of dismantle direction.
     This is a separate, non-directional analysis from `prerequisites` above — it answers "what must already
     be assembled before this stander is added," for assembly-order purposes, not "what blocks this
     stander's removal." A component can be a contact neighbor without being a (disassembly) prerequisite,
     and vice versa.
5. **Stander-stander contact is explicitly ignored for ordering.** Two standers may physically touch (e.g. a
   bolt through a nut), but this must never be used to infer precedence between them — they are always
   treated as independent. This applies to both `prerequisites` and `contact_neighbors`.
6. Once all standers are processed, **generate the stander-reduced assembly** (a new version of the assembly
   with all stander components removed). This is what Part 2 and downstream Interference Identification run
   against, to keep collision analysis computationally lighter.

## Part 2 — Candidate direction discovery

**Input:** the stander-reduced assembly from Part 1 (not the original — see correction above; cylindrical
features on stander parts would otherwise just re-derive directions Part 1 already committed to a different
way).

1. **Scan the whole (reduced) assembly** for two feature types: planar faces and cylindrical faces. (Scope
   for now — other feature types, e.g. conical/spherical, are out of scope.) A component may contribute
   planar features, cylindrical features, or both.
2. **For each feature found, derive a direction:**
   - Cylindrical → its axis
   - Planar → its normal
   - Every feature yields **two** opposite candidate directions (one per side), before filtering.
3. **Deduplicate** the full pool of candidate directions by **exact signed-vector equality** (no angular
   tolerance — the CAD kernel returns exact analytic values, not tessellated/approximate ones, so exact
   comparison is expected to be reliable here, unlike the radial-threshold case above).
4. **Output:** the unique list of candidate disassembly directions. This feeds Interference Identification
   (which will test each remaining component against each candidate direction).

## Data structures / output contract (what Preprocessing hands downstream)

- Stander-reduced assembly (geometry, ready for Interference Identification and Part 2's feature scan).
- Per-stander record: `{ prerequisites: set[component_id], exit_point: Point3D, dismantle_direction: Vec3,
  contact_neighbors: set[component_id] }`.
- Unique candidate direction list: `list[Vec3]`.

## Open / tunable parameters (not design blockers, just need calibration)

- Radial distance threshold used in stander contact detection (Part 1, step 3). Start with a small default,
  tune against the Vulkan validation case.
- Contact threshold (minimum gap) used for `contact_neighbors` in Part 1 step 4. Independent from the radial
  distance threshold above — this one governs literal touching-contact detection, not the directional walk.

## Implementation environment

- Repo: this folder (`AAP`), git-tracked, conda env `environmentAPP`.
- `test.py` is a throwaway conda-env smoke test, not part of the real implementation — ignore/replace it.
- Validation case study: the "Vulkan" part assembly (STEP files under `../vulkan parts/step/`).
