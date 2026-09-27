"""AoG generator (AND/OR graph): not yet design-reviewed — sketch only, per CLAUDE.md.

Recursive: given a component with a free removal vector (or search the matrix for one on
first call), pick its smallest-separation-distance direction if multiple exist, remove it
from the matrix, dedupe/create AoG nodes for the resulting sub-assembly state, check whether
removing it frees any stander (query that stander's `prerequisites`) and re-add it to the
matrix if so, recurse until the matrix is empty.

Flagged as the highest-uncertainty module — needs its own dedicated design pass, especially
the sub-assembly "already analyzed" state representation.
"""

from __future__ import annotations

from interference_identification import InterferenceResult
from preprocessing import StanderRecord


def build_aog(interference_result: InterferenceResult, stander_records: dict[str, StanderRecord]):
    """Return the assembly-oriented graph (AND/OR graph) of feasible disassembly sequences.

    Free removal directions come from `interference_result` queries (free_directions,
    removable, blockers) with no geometry calls; stander records are keyed by the same
    obj.Name IDs as `interference_result.component_ids`.

    Not yet implemented — the AoG generator has not been design-reviewed yet; see
    CLAUDE.md's "Current status" section.
    """
    raise NotImplementedError(
        "AoG generator has not been design-reviewed yet; see CLAUDE.md."
    )
