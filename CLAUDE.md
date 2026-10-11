# AAP — Assembly Sequence Planning (Master's thesis)

Automated assembly/disassembly sequence planning from STEP files: preprocess an assembly, identify
interferences, build an assembly-oriented graph (AoG / AND-OR graph) of feasible disassembly sequences,
optimize it with ACO, then generate illustrated + written assembly instructions.

Conda env: `environmentAAP` (see `environment.yml`). `test.py` is a leftover conda-env smoke test — ignore it,
not part of the real implementation. Validation case study: the "Vulkan" part assembly, STEP files under
`../vulkan parts/step/` (one level up from this repo). "Vulkan" and `industrial_fastner` are the same
assembly (the file was renamed to avoid a proprietary name). Design target: assemblies of up to 20 components.

## Pipeline

```
Preprocessing ─┐
                ├─► Interference Identification ─► AoG generator ─► Optimization (ACO) ─┬─► Illustration Generator ─┐
                │                                                                          ├─► Written Instructions ─┤
                └──────────────────────────────────────────────────────────────────────────                          └─► done
```

Two grouped stages: **ASP/APP** (Preprocessing, Interference Identification, AoG) produces an optimized
assembly sequence; **AIG** (Illustration + Written instructions) turns that sequence into output for a human
assembler.

Modules are being implemented one at a time, each preceded by a conceptual design review. Detailed specs for
reviewed modules live in `docs/`; read the relevant one before touching that module's code.

## Current status

- **Preprocessing** — design-reviewed, spec'd and implemented. Spec: `docs/preprocessing-spec.md` (corrects
  a sequencing error in the workflow diagram: the two Preprocessing sub-steps are NOT parallel, despite how
  `Individual_Workflows.png` draws them; also pins down dismantle-direction selection, contact thresholds
  and the direction dedup rule).
- **Interference Identification** — design-reviewed, spec'd and implemented (`src/interference_identification.py`
  on top of the collision core in `src/collision.py`). Spec: `docs/interference-spec.md`; acceptance tests
  I1–I11 in `tests_scripts/test_interference.py`. The collision test it uses (stepping with gap-skipping,
  contact counts as free, 1 mm³ static-overlap rule) is verified by `docs/collisiontest.md` /
  `tests_scripts/test_collision.py`, which imports the production core.
- **AoG generator** — design-reviewed 2026-09-29, spec'd and implemented (`src/aog_generator.py`). Spec:
  `docs/aog-spec.md` (iterative level-by-level generation over bitmask states; standers are ordinary
  components with a precedence rule; all free directions kept on each edge; no graph reduction; dead ends
  reported only). Travel distance is measured to the constant whole-assembly bounding box
  (`PreprocessingResult.assembly_bbox`); a stander's gate set is `prerequisites ∪ contact_neighbors`.
  States whose parts do not touch as one group are dropped (spec D11; movable contacts come from
  `InterferenceResult.contact`, gap ≤ 0.1 mm).
  Acceptance tests A1–A12 in `tests_scripts/test_aog.py` (`--groups core real scale`).
- Optimization, Illustration Generator, Written Instructions — not yet design-reviewed in detail. Rough
  sketch only (see below); do not implement ahead of a dedicated review — bring design questions back to
  conversation before coding, the way the other modules were handled.

## Rough sketch of not-yet-reviewed modules (do not treat as final)

- **Optimization**: ACO over the AoG; edges start evenly weighted, penalize (reduce weight of) consecutive
  steps that reuse the same removal direction; run ACO, extract shortest path as the final sequence. The
  AoG hands it every free direction per edge and chooses none; the ACO picks the direction (a longer move
  is acceptable if it continues the previous direction). The "penalize reuse" wording is probably inverted
  (continuity should be favoured); settle it in the Optimization design pass (see `docs/aog-spec.md`).
- **Illustration Generator**: per step, get the trajectory of the added component, render it; if start/end
  points aren't both visible, slightly rotate the assembly view and retry; highlight path, footnote
  component+tool, save frame.
- **Written Instructions Generator**: per step, same trajectory computation, write out component name +
  displacement vector as text.

## Working conventions

- Test scripts are plain Python under `tests_scripts/` (pytest is not installed in `environmentAAP`), run
  from the repo root; outputs go to `tests_scripts/output/`.

- This thesis's Claude Projects workspace (claude.ai) holds `architecture-overview.md`, a living doc mirroring
  this file's pipeline section at a higher level, plus the original workflow diagrams. This `CLAUDE.md` +
  `docs/` are the source of truth for implementation-level detail; keep them in sync if either changes.
