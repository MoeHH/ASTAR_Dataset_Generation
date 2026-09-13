<!-- PrePrint available @ <ADD_PREPRINT_URL_HERE> -->

# ASTAR Dataset Generation: A\* Grid Pathfinding Datasets for Transformer Path Planning

**Abstract:** Learning-based path planners require large, well-structured datasets of grids with
known-optimal solutions. This repository is a GPU-accelerated generator that produces such
datasets for 2D grid navigation using the A\* search algorithm. For each grid it records the
context (per-cell features and status), the found route(s), the action masking, and the target
move sequence (Up, Down, Right, Left), and exports every sample in three interchangeable formats
(PKL, JSON, and PyTorch `.pt` tensors) so the same data can drive both classical tooling and a
Transformer training pipeline. Two generation modes are provided: a *templates* mode that places
obstacles from a bank of pre-defined patterns under a configurable open-space schedule, and a
*legacy* mode that scatters obstacles at random. Alongside solvable (`sol`) grids, the generator
deliberately constructs unsolvable (`nosol`) grids — via disconnected components and isolated
goals — to train a route-availability gate. A staged pipeline generates the master datasets,
extracts a balanced gate ("RAGate") dataset, optionally renders ImageNet-style JPEGs of the
grids, and validates everything with a two-level inspection sweep. The datasets produced here are
consumed directly by the companion [`PSTAR`](../PSTAR) path-planning model.

**Keywords:** A\*; Pathfinding; Dataset Generation; Grid Navigation; CuPy; GPU; Solvable and
Unsolvable Grids; Transformer Training Data

---

## Overview

This is the **data** half of the pipeline. It generates the grids, solves them with A\*, and
exports the datasets that the companion [`PSTAR`](../PSTAR) repository trains on. The work is
organised as a four-stage pipeline driven by a single, fully commented configuration file
(`config.py`):

* **Stage 1 — Base datasets.** Generate the solvable (`sol`) and unsolvable (`nosol`) master
  datasets and split them into train / val / test.
* **Stage 2 — RAGate dataset.** Extract a balanced route-availability classifier dataset
  (`label=1` solvable, `label=0` unsolvable) into `Datasets/RAGate/`.
* **Stage 3 — Images (optional).** Render ImageNet-style JPEGs of the master grids
  (start / goal / obstacles / path) at one or more resolutions.
* **Stage 4 — Inspection.** Validate the datasets with Quick and/or Detailed integrity sweeps.

## Requirements

* Python 3.10+
* An NVIDIA GPU with CUDA is strongly recommended (A\* search runs on the GPU via CuPy).

Install the dependencies:

```bash
pip install -r requirements.txt
```

The core dependencies are `numpy`, `matplotlib`, `torch`, `pillow`, and **CuPy**. CuPy must match
your CUDA runtime:

```bash
# CUDA 12.x
pip install cupy-cuda12x
# CUDA 11.x
pip install cupy-cuda11x
```

`main.py` detects your CUDA version at startup and installs the correct CuPy variant
automatically if it is missing.

## Instructions

Choose which stages run and configure their behaviour with the switches near the bottom of
`config.py`, then run:

```bash
python main.py
```

### Pipeline switches (in `config.py`)

```python
"run_stage1_generation": True,    # Stage 1 — base sol/nosol datasets
"run_RAGate_generation": True,    # Stage 2 — RAGate classifier dataset
"generate_images":       True,    # Stage 3 — ImageNet-style JPEGs
"run_validation":        True,    # Stage 4 — dataset inspection
```

### Generation settings (in `config.py`)

```python
"grid_size":       "10,10",       # grid dimensions (rows, cols)
"device":          "cuda",
"generation_mode": "templates",   # "templates" | "legacy"

# Templates mode
"open_obstacles_percent": 0.50,
"open_percent_schedule":  "0.0,0.1,0.2,0.3,0.4,0.5,0.7",
"min_start_goal_distance": 4,
"chunk_size":              250,

# Legacy mode
"max_obstacles": 50,

# Stage 1 target counts (loop until reached; 0 = single pass)
"sol_target_count":   250000,
"nosol_target_count": 250000,

# Train / val / test split
"train_ratio": 0.80,
"val_ratio":   0.10,
"test_ratio":  0.10,
"split_seed":  42,
```

### RAGate dataset settings (in `config.py`)

```python
"ragate_sol_count":   200000,     # solvable samples in the gate dataset
"ragate_nosol_count": 100000,     # unsolvable samples in the gate dataset
"ragate_nosol_isolated_goal_fraction": 0.3,   # share built via isolated-goal method
```

### Image rendering settings (in `config.py`)

```python
"generate_images": True,
"image_sizes":     "64",          # comma-separated, e.g. "64,128,224"
"image_modes":     "color",       # "color" and/or "gray"
"images_dir":      "Datasets/Images",
```

### Inspection settings (in `config.py`)

```python
"inspect_quick":    True,          # fast sanity check (< 60s)
"inspect_detailed": True,          # full integrity sweep (5–15 min)
```

## Outputs

All artifacts are written under `Datasets/`, laid out by group:

```
Datasets/
├── templates.pkl / templates.json      # cached obstacle templates
├── Masters/                            # sol_dataset / nosol_dataset (.pkl, .json, .pt)
├── PSTAR/                              # transformer training data
│   ├── train/  val/  test/            # {train,val,test}_{sol,nosol}.{pkl,json,pt}
├── RAGate/                             # route-availability classifier data
│   ├── RAGate_{sol,nosol}.{pkl,json,pt}
│   └── train/  val/  test/            # RAGate_{train,val,test}.{pkl,json,pt}
└── Images/                            # optional ImageNet-style JPEGs
    └── <mode>/<size>/<split>/<class>/<id>.jpg
```

Each exported sample carries a consistent set of fields — `Context`, `Found_Routes`, `Masking`,
`Targets`, `TourCost`, the corresponding `input_tensor` / `masking_tensor` / `target_tensor`, and
a `label` (`1` = solvable, `0` = unsolvable) — so all three formats stay in sync.

## Results

Route plots and rendered grid images are written under `Datasets/`. Enable `save_route_plots`,
`show_animation`, or `show_finalimage` in `config.py` to visualize individual A\* solutions.

<!-- Add example grid / route images here, e.g.:
![Solvable grid with A* route](docs/sol_example.png)
![Unsolvable grid (isolated goal)](docs/nosol_example.png)
-->

## Project structure

```
ASTAR_Dataset_Generation/
├── main.py                     # entry point; runs the stages enabled in config.py
├── config.py                   # central, fully-commented configuration
├── requirements.txt
├── astar/
│   ├── astar_cupy.py           # GPU-accelerated A* search (CuPy)
│   └── grid_templates.py       # bank of pre-defined 10x10 obstacle templates
├── dataset/
│   ├── dataset_utils.py        # sample building, splitting, export (pkl/json/pt)
│   ├── generate_ragate.py      # Stage 2 — RAGate classifier dataset
│   └── render_images.py        # Stage 3 — ImageNet-style JPEG rendering
└── tools/
    ├── inspect_datasets.py     # Stage 4 — Quick / Detailed integrity sweeps
    └── visualize_routes.py     # plot individual routes
```

## Related repository

* [`PSTAR`](../PSTAR) — the Transformer path-planning model and RAGate gate that train on the
  datasets produced here.

## License

This project is released under the [MIT License](LICENSE).
