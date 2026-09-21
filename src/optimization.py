"""Optimization (ACO): not yet design-reviewed — sketch only, per CLAUDE.md.

ACO over the AoG; edges start evenly weighted, penalize (reduce weight of) consecutive steps
that reuse the same removal direction; run ACO, extract shortest path as the final sequence.
"""

from __future__ import annotations


def optimize_sequence(aog):
    """Return the optimized disassembly sequence (shortest ACO path through the AoG).

    Not yet implemented — Optimization has not been design-reviewed yet; see CLAUDE.md's
    "Current status" section.
    """
    raise NotImplementedError(
        "Optimization (ACO) has not been design-reviewed yet; see CLAUDE.md."
    )
