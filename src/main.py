"""AAP pipeline entry point.

Wires the full pipeline from CLAUDE.md:

    Preprocessing -> Interference Identification -> AoG generator -> Optimization (ACO)
        -> Illustration Generator
        -> Written Instructions

The first group (through Optimization) is ASP/APP: it produces an optimized disassembly
sequence. The second group (AIG) turns that sequence into output for a human assembler;
Illustration Generator and Written Instructions both consume the same optimized sequence
independently, not sequentially.

Preprocessing (docs/preprocessing-spec.md) and Interference Identification
(docs/interference-spec.md) are implemented. The remaining stages are wired in through their
stub modules, each of which raises NotImplementedError until it gets its own design review —
see CLAUDE.md's "Current status" section.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aog_generator import build_aog
from collision import CollisionConfig
from illustration_generator import generate_illustrations
from interference_identification import identify_interferences
from optimization import optimize_sequence
from preprocessing import PreprocessingConfig, run_preprocessing
from written_instructions import generate_written_instructions


def run_pipeline(
    step_path: str | Path,
    bom_path: str | Path,
    output_dir: str | Path,
    preprocessing_config: PreprocessingConfig | None = None,
    collision_config: CollisionConfig | None = None,
    force_interference: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ASP/APP: preprocess, identify interferences, build the AoG, optimize -> sequence.
    preprocessing_result = run_preprocessing(step_path, bom_path, preprocessing_config)
    interference_result = identify_interferences(
        preprocessing_result,
        collision_config,
        output_dir / "interference",
        force=force_interference,
    )
    aog = build_aog(interference_result, preprocessing_result.stander_records)
    sequence = optimize_sequence(aog)

    # AIG: turn the optimized sequence into output for a human assembler.
    generate_illustrations(sequence, output_dir / "illustrations")
    generate_written_instructions(sequence, output_dir / "instructions")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AAP assembly sequence planning pipeline on a STEP assembly."
    )
    parser.add_argument("step_path", type=Path, help="STEP file for the assembly.")
    parser.add_argument("bom_path", type=Path, help="BOM CSV flagging stander components.")
    parser.add_argument(
        "output_dir", type=Path, help="Directory to write illustrations/instructions into."
    )
    parser.add_argument(
        "--step",
        type=float,
        default=CollisionConfig.step,
        help="Collision step size h in mm (calibrate per assembly with collision test E6).",
    )
    parser.add_argument(
        "--force-interference",
        action="store_true",
        help="Recompute the interference matrix even if a cached one matches.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    run_pipeline(
        args.step_path,
        args.bom_path,
        args.output_dir,
        collision_config=CollisionConfig(step=args.step),
        force_interference=args.force_interference,
    )


if __name__ == "__main__":
    main()
