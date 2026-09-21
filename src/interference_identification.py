"""Interference Identification: not yet design-reviewed — sketch only, per CLAUDE.md.

Loop over every remaining component x every candidate direction from Preprocessing; run a
collision analysis per (component, direction) pair; store results in an interference matrix.
"""

from __future__ import annotations

import FreeCAD


def identify_interferences(
    reduced_assembly: "FreeCAD.Document",
    candidate_directions: list["FreeCAD.Vector"],
) -> dict:
    """Return the interference matrix keyed by (component, direction).

    Not yet implemented — Interference Identification has not had its design review;
    see CLAUDE.md's "Current status" section.
    """
    raise NotImplementedError(
        "Interference Identification has not been design-reviewed yet; see CLAUDE.md."
    )
