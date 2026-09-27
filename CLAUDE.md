# AAP — Assembly Sequence Planning (Master's thesis)

Automated assembly/disassembly sequence planning from STEP files: preprocess an assembly, identify
interferences, build an assembly-oriented graph (AoG / AND-OR graph) of feasible disassembly sequences,
optimize it with ACO, then generate illustrated + written assembly instructions.

Conda env: `environmentAAP` (see `environment.yml`). `test.py` is a leftover conda-env smoke test — ignore it,
not part of the real implementation. Validation case study: the "Vulkan" part assembly, STEP files under
`../vulkan parts/step/` (one level up from this repo).

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
- AoG generator, Optimization, Illustration Generator, Written Instructions — not yet design-reviewed in
  detail. Rough sketch only (see below); do not implement ahead of a dedicated review — bring design
  questions back to conversation before coding, the way Preprocessing and Interference Identification were
  handled.

## Rough sketch of not-yet-reviewed modules (do not treat as final)

- **AoG generator** (AoG = AND/OR graph, same thing, no need to distinguish them in the writeup): recursive —
  given a component with a free removal vector (or search the matrix for one on first call), pick its
  smallest-separation-distance direction if multiple exist, remove it from the matrix, dedupe/create AoG
  nodes for the resulting sub-assembly state, check whether removing it frees any stander component (query
  that stander's `prerequisites` from `docs/preprocessing-spec.md`) and release it if so, recurse
  until the matrix is empty. Input is the `InterferenceResult` from `docs/interference-spec.md` (free
  directions of a part in a sub-assembly are pure set intersections over its rows; standers are not in the
  matrix and are released via their `StanderRecord`s). Flagged as the highest-uncertainty module — needs its own dedicated design pass,
  especially the sub-assembly "already analyzed" state representation.
- **Optimization**: ACO over the AoG; edges start evenly weighted, penalize (reduce weight of) consecutive
  steps that reuse the same removal direction; run ACO, extract shortest path as the final sequence.
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
