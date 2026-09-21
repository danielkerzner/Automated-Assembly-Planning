# AAP — Assembly Sequence Planning (Master's thesis)

Automated assembly/disassembly sequence planning from STEP files: preprocess an assembly, identify
interferences, build an assembly-oriented graph (AoG / AND-OR graph) of feasible disassembly sequences,
optimize it with ACO, then generate illustrated + written assembly instructions.

Conda env: `environmentAPP` (see `environment.yml`). `test.py` is a leftover conda-env smoke test — ignore it,
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

- **Preprocessing** — design-reviewed and spec'd, this is the module to implement now. Full spec:
  `docs/preprocessing-spec.md`. Read it before writing any Preprocessing code — it corrects a sequencing
  error in the original workflow diagram (the two Preprocessing sub-steps are NOT parallel, despite how
  `Individual_Workflows.png` draws them) and pins down algorithm details (dismantle-direction selection,
  contact-threshold tie-breaking, direction dedup rule) that aren't in the diagrams at all.
- Interference Identification, AoG generator, Optimization, Illustration Generator, Written Instructions —
  not yet design-reviewed in detail. Rough sketch only (see below); do not implement ahead of a dedicated
  review — bring design questions back to conversation before coding, the way Preprocessing was handled.

## Rough sketch of not-yet-reviewed modules (do not treat as final)

- **Interference Identification**: loop over every remaining component × every candidate direction from
  Preprocessing; run a collision analysis per (component, direction) pair; store results in an interference
  matrix.
- **AoG generator** (AoG = AND/OR graph, same thing, no need to distinguish them in the writeup): recursive —
  given a component with a free removal vector (or search the matrix for one on first call), pick its
  smallest-separation-distance direction if multiple exist, remove it from the matrix, dedupe/create AoG
  nodes for the resulting sub-assembly state, check whether removing it frees any stander component (query
  that stander's `prerequisites` from `docs/preprocessing-spec.md`) and re-add it to the matrix if so, recurse
  until the matrix is empty. Flagged as the highest-uncertainty module — needs its own dedicated design pass,
  especially the sub-assembly "already analyzed" state representation.
- **Optimization**: ACO over the AoG; edges start evenly weighted, penalize (reduce weight of) consecutive
  steps that reuse the same removal direction; run ACO, extract shortest path as the final sequence.
- **Illustration Generator**: per step, get the trajectory of the added component, render it; if start/end
  points aren't both visible, slightly rotate the assembly view and retry; highlight path, footnote
  component+tool, save frame.
- **Written Instructions Generator**: per step, same trajectory computation, write out component name +
  displacement vector as text.

## Working conventions

- This thesis's Claude Projects workspace (claude.ai) holds `architecture-overview.md`, a living doc mirroring
  this file's pipeline section at a higher level, plus the original workflow diagrams. This `CLAUDE.md` +
  `docs/` are the source of truth for implementation-level detail; keep them in sync if either changes.
