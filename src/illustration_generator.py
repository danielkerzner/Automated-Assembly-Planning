"""Illustration Generator: not yet design-reviewed — sketch only, per CLAUDE.md.

Per step, get the trajectory of the added component, render it; if start/end points aren't
both visible, slightly rotate the assembly view and retry; highlight path, footnote
component+tool, save frame.
"""

from __future__ import annotations

from pathlib import Path


def generate_illustrations(sequence, output_dir: Path) -> None:
    """Render one illustration frame per assembly step into `output_dir`.

    Not yet implemented — the Illustration Generator has not been design-reviewed yet;
    see CLAUDE.md's "Current status" section.
    """
    raise NotImplementedError(
        "Illustration Generator has not been design-reviewed yet; see CLAUDE.md."
    )
