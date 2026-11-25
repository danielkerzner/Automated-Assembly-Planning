# AAP: Automated Assembly Planning for Manual and Robot-Based Assembly Processes

## Introduction

This repository is the software implementation of the master's thesis **"Automated Assembly Planning for Manual and Robot Based
Assembly Processes"** by Daniel Kerzner. The written thesis is the reference for the motivation, the state of the art, the
methods, the case studies and the results; this README only explains how to install, run and navigate the code.
<!-- Add the link or citation of the written thesis here once it is published. -->

Given a **STEP file** of an assembly and a **BOM** (a CSV table with one row per part), the program computes an optimized
**assembly sequence** (found as a disassembly sequence, one component removed at a time along a straight line) and turns it into
one of two results:

- **Manual mode**: an illustrated instruction manual for a human assembler (one PNG per step plus a PDF).
- **Robot mode**: a sequence checked for feasibility with a parallel gripper and with the robot tools for screws, nuts and
  washers/bushings, written to JSON with the gripper poses.

### Pipeline

```
STEP + BOM
   |
   v
Preprocessing  ->  Interference Identification  ->  AoG (assembly graph)  ->  Optimization (ACO)  ->  Illustration Generator   (manual mode)
 (standards,          (collision tests, free           (all feasible states,     (best sequence)         or Gripper/tool feasibility loop (robot mode)
  candidate           directions per pair)             level by level)
  directions)
```

| Stage | File | What it does |
|---|---|---|
| Hierarchy | `src/hierarchy.py` | Reads the nested sub-assembly structure of the STEP file and the BOM flags (`processed`, `subassembly`). |
| Levels | `src/levels.py` | Runs the whole pipeline once per level (one container and its direct children), depth-first. Creates automatic sub-assemblies when a group of parts blocks itself. |
| Preprocessing | `src/preprocessing.py` | Loads STEP and BOM, classifies the standards (screws, nuts, others), finds candidate assembly directions. |
| Collision core | `src/collision.py` | Stepped translation test with gap skipping, used by the interference stage. |
| Interference Identification | `src/interference_identification.py` | Builds the boolean tensor `free[i, j, k]`: component *i* can leave along direction *k* past component *j*. Cached on disk. |
| AoG generator | `src/aog_generator.py` | Builds the graph of states (bitmasks) level by level, with the standard, connectivity and BOM-group rules. |
| Optimization | `src/optimization.py` | Ant Colony Optimization over the AoG; cost terms are travel, direction change, tool change, part score, grouping and gravity stability. |
| Illustration Generator | `src/illustration_generator.py` | Own z-buffer renderer plus matplotlib annotation; PNG per step and `manual.pdf`. |
| Robot mode | `src/robot_*_robotic.py`, `src/gripper_*_robotic.py`, `src/pointclouds_robotic.py`, `src/output_robotic.py`, `src/tool_check.py` | Optimize, check grasp and tool feasibility, prune the infeasible edge, re-optimize, repeat. |
| Entry point | `src/main.py` | Command line interface; writes `plan.json`. |

## Requirements

- **Windows** with [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html). The code was developed and tested on Windows.
- **Main environment `environmentAAP`** (defined in `environment.yml`):

  | Package | Used for |
  |---|---|
  | Python 3.11 | everything |
  | `freecad` 1.1 (conda-forge) | reading STEP files, shapes, boolean and distance operations (`FreeCAD`, `Part`) |
  | `numpy` | all numerical work, the AoG arrays and the renderer |
  | `matplotlib` | illustrations, the manual PDF and the visualization scripts |
  | `pyyaml` | read by `extra_functions/visualize_grasps.py` |
  | `networkx` (optional, **not installed by default**) | only for `AoG.to_networkx()` (graph figures); `pip install networkx` if you need it |

  `numpy`, `matplotlib` and `pyyaml` are installed automatically as dependencies of the FreeCAD conda package (checked on the
  development environment: numpy 2.4, matplotlib 3.11, pyyaml 6.0). `environment.yml` lists `defaults` as the channel;
  if `freecad` is not found there, add `conda-forge`.
- **Robot mode only: second environment `parallelgrippercode`** for the JMS parallel-gripper grasp engine. It is a plain Python 3.11
  environment with the engine packages installed through `pip`: `open3d`, `shapely`, `opencv-python`, `scikit-learn`, `pyyaml`,
  `numpy` and `matplotlib`. Development versions: open3d 0.19, shapely 2.1, opencv-python 4.11, scikit-learn 1.7, numpy 2.3,
  matplotlib 3.10. AAP starts this environment itself as a worker
  subprocess; **never activate it to run AAP**. It is found as the sibling environment of `environmentAAP`, or through the
  variable `AAP_GRIPPER_PYTHON`, or through `RobotConfig.engine_python`.
- **Robot mode only: the JMS engine repository** (read only, not part of this repository), expected next to this one at
  `../gripperparallel/Adaptive-In-Orbit-Servicing-of-Altered-Satellite-Components/JMS`. Change `DEFAULT_ENGINE_REPO` in
  `src/gripper_engine_robotic.py` or pass `RobotConfig.engine_repo` if it lives elsewhere.
- No pytest: all tests are plain Python scripts.

## Installation

```bat
conda env create -f environment.yml
conda activate environmentAAP
```

`environment.yml` has a machine-specific `PYTHONPATH` (`...\envs\environmentAAP\Library\bin`) and `prefix`. Edit both to your own
Miniconda path before creating the environment; the `PYTHONPATH` is what lets FreeCAD find its native libraries.

Check the installation:

```bat
python -c "import freecad; import FreeCAD, Part; print(FreeCAD.Version())"
```

If `import FreeCAD` or `import Part` fails with the conda build, import `freecad` (lowercase) first, as above.
All commands below are run **from the repository root**, with `environmentAAP` active (or prefixed with
`conda run -n environmentAAP`; calling the environment's `python.exe` directly fails to load numpy's DLLs).

## Inputs

### STEP file

One STEP file of the assembly. It may contain nested sub-assemblies (a product structure with parts inside parts).
Units are millimetres.

### BOM (CSV, UTF-8)

One row per part or sub-assembly, matched to the STEP model on the FreeCAD **Label**. Only `label` is required.

| Column | Meaning |
|---|---|
| `label` | Label of the part or sub-assembly in the STEP model. |
| `is_standard` | `true` for standards (screws, nuts, washers, seals...). Standards are not part of the interference matrix; they get their own removal rule. |
| `screw`, `nut` | Pick the standard category. A standard with neither is category 3 ("other": washers, bearings, bushings). Both true on one row is an error. |
| `pitch` | Axial travel in mm per full turn, for screw and nut rows (used only to show thread turns). |
| `tool` | Tool needed by a standard; used by the tool-change cost in manual mode. |
| `group` | Parts with the same group are assembled consecutively. |
| `processed` | Container rows only (blank = `true`). `false` treats a sub-assembly as one rigid component and does not analyze its inside. |
| `subassembly` | Label of the parent container (blank for the root); checked against the STEP tree when filled. |

`extra_functions/generate_bom.py` writes a starter BOM from a STEP file (see below). Ready-made examples are in `data/`.

## How to run the main program

```bat
python src/main.py <step_path> <bom_path> <output_dir> [options]
```

Manual mode (illustrated instructions) on the reference assembly:

```bat
python src/main.py data/industrial_fastner/industrial_fastner.stp data/industrial_fastner/generated_bom.csv output/industrial_fastner --mode manual
```

Robot mode (gripper and tool feasibility loop) on the same assembly:

```bat
python src/main.py data/industrial_fastner/industrial_fastner.stp data/industrial_fastner/generated_bom.csv output/industrial_fastner_robot --mode robot
```

| Option | Meaning |
|---|---|
| `--mode {manual,robot}` | Output type. Without it, you are asked in the terminal (Enter = manual); without a terminal, manual. |
| `--step H` | Collision step size *h* in mm (default 5.0). Reduce it for assemblies with thin features. |
| `--force-interference` | Recompute the interference matrix even if a matching cache exists. |
| `--max-level-components N` | Abort before any geometry work if a processed level has more than *N* direct components (default 20). Fix by setting `processed = false` for that sub-assembly in the BOM. |
| `--check-subassembly-grasp` | Robot mode only: an edge is feasible only if both sub-assemblies it joins are graspable as one object. |

Other settings (`PreprocessingConfig`, `OptimizationConfig`, `IllustrationConfig`, `RobotConfig`...) are not on the command line.
Change them in code, or pass the config objects to `run_pipeline` from your own script.

### What you get

Inside `<output_dir>`:

- `plan.json`: the combined plan. Per level: status (`complete` or `locked`), the ordered steps with component, direction,
  travel, cost and kind (`part`, `standard`, `subassembly`), and the nested plan of every expanded sub-assembly. In robot mode it
  also has a `robot` block with the grasp scores, poses and every sequence tried.
- One folder per stage with its cache and report: `interference/`, `aog/`, `optimization/` and, in manual mode,
  `illustrations/` (one PNG per step and `manual.pdf`). Sub-assemblies write to `levels/<id>/`.
- Exit code 0 on success; 1 if the root level ended `locked` (some parts need a motion the system does not model, for example
  a rotation). The stuck parts are printed to stderr and listed in `plan.json`.

Expensive results are cached, keyed by a hash of the inputs and the configuration, so a repeated run with the same inputs is fast.
The interference stage also checkpoints per pair and resumes after an interruption.

### Test assemblies in `data/`

| Folder | Content |
|---|---|
| `industrial_fastner/` | Flat assembly, 5 movable parts + 3 standards. The reference case for all tests. |
| `lunar_bodylager/` | Nested assembly used by the sub-assembly tests. |
| `lunar_motor_tire/`, `lunar_motor_tire_complete/` | Larger nested assemblies with a motor sub-assembly (processed or flagged as a standard). |
| `nonlinearsimple/` | Four mutually blocked parts: reference for the automatic sub-assembly creation. |
| `obliquesimple/` | Small fixture with an oblique assembly direction. |

Each folder has the STEP file and a `generated_bom.csv`.

## Repository layout

```
AAP/
  src/                 the pipeline (modules in the table above)
  data/                test assemblies (STEP + BOM)
  output/              run results; output/industrial_fastner/ also holds the cached reference results
  automatic_tests/     acceptance test suites
  manual_tests/        scripts that run one module and print readable results
  extra_functions/     utilities and viewers (not part of the pipeline)
  third_party/jms_grasp/   generated copy of the JMS grasp engine functions (MIT license, source commit in its header)
  tools/gripper/       robot tool STEP files (screw, nut and bushing end effectors)
  environment.yml      conda environment
```

## Tests

Run from the repository root. There is no pytest; each script is a standalone runner.

### Automatic tests (`automatic_tests/`)

Each suite has a `core` group (seconds, no FreeCAD run) and heavier groups. Common options: `--tests <ids>` and `--groups <names>`.

| Script | Tests | Groups |
|---|---|---|
| `test_aog.py` | A1-A13: AoG generation, rules, dead ends | `core`, `real`, `scale` |
| `test_optimization.py` | O1-O15: costs, ACO, determinism | `core`, `real` (about 4 min), `scale` |
| `test_illustration.py` | L1-L14: renderer, visibility, PDF | `core`, `real`, `scale` |
| `test_subassembly.py` | H-tests and R-tests: sub-assemblies, automatic groups | `core`, `real` |
| `test_robotic.py` | R1-R13: robot mode | `core`, `freecad`, `engine` (real grasp engine, a few minutes) |

```bat
python automatic_tests/test_aog.py
python automatic_tests/test_optimization.py --tests O1 O5
python automatic_tests/test_subassembly.py --groups core real
python automatic_tests/test_robotic.py --groups core freecad engine
```

The `real` groups read the cached results in `output/industrial_fastner/` and never write to it. Run
`manual_tests/test_aog_manual.py` or the main program on `industrial_fastner` first if that folder is missing.

### Manual tests (`manual_tests/`)

One script per module. They run the module and print a readable report, with the expected result described at the top of each
file. Without arguments they use `industrial_fastner`, its `generated_bom.csv` and `output/industrial_fastner`.

```bat
python manual_tests/test_preprocessing_manual.py
python manual_tests/test_interference_manual.py [step.stp bom.csv out_dir] [--step 1.0]
python manual_tests/test_aog_manual.py          # also writes aog_graph.html
python manual_tests/test_optimization_manual.py
python manual_tests/test_illustration_manual.py
```

Note: with the default step of 5.0 a default run writes into `output/industrial_fastner/` and may recompute the caches there.

### Extra functions (`extra_functions/`)

| Script | Purpose |
|---|---|
| `generate_bom.py <step> <csv>` | Write a starter BOM from a STEP file, asking about each nested sub-assembly. |
| `check_labels.py` | Print the FreeCAD Name and Label of each part (to write the BOM correctly). |
| `verifynames.py` | Dump the raw FreeCAD Name/Label/TypeId tree. |
| `zhang_matrices.py` | Derive the P/E/C/G task-constraint matrices (Zhang et al.) from a cached AoG. |
| `visualize_grasps.py <output_dir> <step>` | Interactive 3D viewer of a robot-mode `plan.json` (arrow keys step through the sequence). |
| `grasp_tree.py <output_dir> <step>` | Draw the explored robot-loop steps as an AND/OR disassembly tree. |
| `visualize_tool.py <output_dir> <step> <standard label>` | Show how a robot tool is placed on a standard and what it touches. |
| `weight_sensitivity.py <output_dir> <seeds> <output_dir>` | Pefrorm a sensitivity analyses for the industrial fastener on the weights of the optimization criteria. |

## Conventions and limits

- Units are millimetres. Motion is a pure translation of one component along a straight line to infinity; no rotations, no
  non-linear paths, threads are only displayed (turn counts), not simulated.
- Components are identified by the FreeCAD `Name`; the `Label` is used for display and for the BOM.
- Contact (sliding along touching faces) counts as a free motion.
- The system targets up to 20 components per level (hard limit 64). Larger assemblies are handled through sub-assemblies.
- Parts that block each other in every direction cannot be solved with translations; the level is reported `locked`.

## Authorship and use of AI

The thesis, the methods and the design decisions are by the author. **Claude (Anthropic's AI assistant) was used during the
implementation** of this code, as a coding assistant. **The author reviewed the work and is responsible for every part of what
is implemented here.**

## Disclaimer

This software was developed for academic research. **It must not be used for financial purposes**, and it is provided "as is",
without warranty of any kind. Sequences, grasps and tool paths it produces are research results and must be verified before any
use on real equipment.

## Contact

Daniel Kerzner: daniel.kerzner@gmail.com
