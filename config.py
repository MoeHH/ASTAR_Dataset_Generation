"""
config.py: Unified configuration for dataset generation, experiment management,
and export.

Dataset folder structure:
  datasets/
    masters/          ← sol_dataset and nosol_dataset (PKL, JSON, .pt)
    PSTAR/       ← train/val/test splits for sol and nosol (PKL, JSON, .pt)
    RAGate/        ← RAGate classifier dataset (PKL, JSON, .pt)

Rename history (RAGate → RAGate):
  - All confidence_* config keys renamed to ragate_*.
  - run_confidence_generation renamed to run_RAGate_generation.
  - File paths moved to datasets/RAGate/ subfolder.
  - Master datasets similarly moved to named subfolders.

Master datasets now include .pt files and a label field (label=1 for sol,
label=0 for nosol), consistent with all other dataset groups.

  Context, Found_Routes, Masking, Targets, TourCost,
  input_tensor, masking_tensor, target_tensor, label.

Inspection controlled via inspect_datasets.py — two levels:
  Quick Datasets Inspection  (inspect_quick=True)
  Detailed Datasets Inspection (inspect_detailed=True)
Both are independently toggled. If both are True, Quick runs first.
"""

import argparse



# --- make the astar/ dataset/ tools/ packages importable with flat names ---
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _sub in ("astar", "dataset", "tools"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _ROOT, _sub, _p


def get_dataset_config():
    parser = argparse.ArgumentParser(
        description="Dataset Generation and Export Configuration"
    )

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    parser.add_argument("--algorithm_used",   default="A*",                type=str)
    parser.add_argument("--heuristic_method", default="Manhattan Distance", type=str)
    parser.add_argument("--grid_size",        default="10,10",             type=str)
    parser.add_argument("--device",           default="cuda",              type=str)

    parser.add_argument(
        "--generation_mode", default="templates",
        choices=["legacy", "templates"], type=str,
        help="Dataset generation mode: 'templates' uses pre-defined obstacle "
             "patterns; 'legacy' places obstacles randomly.",
    )

    # -------------------------------------------------------------------------
    # Shared run controls
    # -------------------------------------------------------------------------
    parser.add_argument("--max_attempts",                default=150,    type=int)
    parser.add_argument("--grid_reuse_count",            default=50,     type=int)
    parser.add_argument("--max_finding_shortest_routes", default=6,      type=int)
    parser.add_argument("--max_open_set_size",           default=500000, type=int)

    # -------------------------------------------------------------------------
    # Legacy Mode
    # -------------------------------------------------------------------------
    parser.add_argument("--max_obstacles", default=50, type=int)

    # -------------------------------------------------------------------------
    # Templates Mode
    # -------------------------------------------------------------------------
    parser.add_argument("--open_obstacles_percent", default=0.50, type=float)
    parser.add_argument(
        "--open_percent_schedule",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.7",
        type=str,
    )
    parser.add_argument("--min_start_goal_distance", default=4,   type=int)
    parser.add_argument("--chunk_size",              default=250, type=int)

    # =========================================================================
    # FILE PATHS — organised by dataset group / subfolder
    # =========================================================================

    # ── Templates cache ───────────────────────────────────────────────────────
    parser.add_argument("--templates_pkl_file",  default="Datasets/templates.pkl",  type=str)
    parser.add_argument("--templates_json_file", default="Datasets/templates.json", type=str)

    # ── Masters group  (Datasets/Masters/) ───────────────────────────────────
    parser.add_argument("--sol_dataset_pkl_file",
                        default="Datasets/Masters/sol_dataset.pkl",  type=str)
    parser.add_argument("--sol_dataset_json_file",
                        default="Datasets/Masters/sol_dataset.json", type=str)
    parser.add_argument("--sol_dataset_pt_file",
                        default="Datasets/Masters/sol_dataset.pt",   type=str,
                        help="Master sol dataset .pt — label=1 for every entry.")

    parser.add_argument("--nosol_dataset_pkl_file",
                        default="Datasets/Masters/nosol_dataset.pkl",  type=str)
    parser.add_argument("--nosol_dataset_json_file",
                        default="Datasets/Masters/nosol_dataset.json", type=str)
    parser.add_argument("--nosol_dataset_pt_file",
                        default="Datasets/Masters/nosol_dataset.pt",   type=str,
                        help="Master nosol dataset .pt — label=0 for every entry.")

    # ── Navigation group  (datasets/PSTAR/) ─────────────────────────────
    # Sol splits — PKL
    parser.add_argument("--dataset_train_sol_pkl_file",
                        default="Datasets/PSTAR/train/train_sol.pkl", type=str)
    parser.add_argument("--dataset_val_sol_pkl_file",
                        default="Datasets/PSTAR/val/val_sol.pkl",   type=str)
    parser.add_argument("--dataset_test_sol_pkl_file",
                        default="Datasets/PSTAR/test/test_sol.pkl",  type=str)

    # Sol splits — JSON
    parser.add_argument("--dataset_train_sol_json_file",
                        default="Datasets/PSTAR/train/train_sol.json", type=str)
    parser.add_argument("--dataset_val_sol_json_file",
                        default="Datasets/PSTAR/val/val_sol.json",   type=str)
    parser.add_argument("--dataset_test_sol_json_file",
                        default="Datasets/PSTAR/test/test_sol.json",  type=str)

    # Sol splits — .pt
    parser.add_argument("--dataset_train_sol_pt_file",
                        default="Datasets/PSTAR/train/train_sol.pt",   type=str)
    parser.add_argument("--dataset_val_sol_pt_file",
                        default="Datasets/PSTAR/val/val_sol.pt",     type=str)
    parser.add_argument("--dataset_test_sol_pt_file",
                        default="Datasets/PSTAR/test/test_sol.pt",    type=str)

    # Nosol splits — PKL
    parser.add_argument("--dataset_train_nosol_pkl_file",
                        default="Datasets/PSTAR/train/train_nosol.pkl", type=str)
    parser.add_argument("--dataset_val_nosol_pkl_file",
                        default="Datasets/PSTAR/val/val_nosol.pkl",   type=str)
    parser.add_argument("--dataset_test_nosol_pkl_file",
                        default="Datasets/PSTAR/test/test_nosol.pkl",  type=str)

    # Nosol splits — JSON
    parser.add_argument("--dataset_train_nosol_json_file",
                        default="Datasets/PSTAR/train/train_nosol.json", type=str)
    parser.add_argument("--dataset_val_nosol_json_file",
                        default="Datasets/PSTAR/val/val_nosol.json",   type=str)
    parser.add_argument("--dataset_test_nosol_json_file",
                        default="Datasets/PSTAR/test/test_nosol.json",  type=str)

    # Nosol splits — .pt
    parser.add_argument("--dataset_train_nosol_pt_file",
                        default="Datasets/PSTAR/train/train_nosol.pt",   type=str)
    parser.add_argument("--dataset_val_nosol_pt_file",
                        default="Datasets/PSTAR/val/val_nosol.pt",     type=str)
    parser.add_argument("--dataset_test_nosol_pt_file",
                        default="Datasets/PSTAR/test/test_nosol.pt",    type=str)

    # ── RAGate group  (datasets/RAGate/) ────────────────────────────────
    # Per-class files
    parser.add_argument("--ragate_sol_json",
                        default="Datasets/RAGate/RAGate_sol.json",   type=str)
    parser.add_argument("--ragate_sol_pkl",
                        default="Datasets/RAGate/RAGate_sol.pkl",    type=str)
    parser.add_argument("--ragate_sol_pt",
                        default="Datasets/RAGate/RAGate_sol.pt",     type=str)
    parser.add_argument("--ragate_nosol_json",
                        default="Datasets/RAGate/RAGate_nosol.json", type=str)
    parser.add_argument("--ragate_nosol_pkl",
                        default="Datasets/RAGate/RAGate_nosol.pkl",  type=str)
    parser.add_argument("--ragate_nosol_pt",
                        default="Datasets/RAGate/RAGate_nosol.pt",   type=str)

    # Split files — JSON
    parser.add_argument("--ragate_train_json",
                        default="Datasets/RAGate/train/RAGate_train.json", type=str)
    parser.add_argument("--ragate_val_json",
                        default="Datasets/RAGate/val/RAGate_val.json",   type=str)
    parser.add_argument("--ragate_test_json",
                        default="Datasets/RAGate/test/RAGate_test.json",  type=str)

    # Split files — PKL
    parser.add_argument("--ragate_train_pkl",
                        default="Datasets/RAGate/train/RAGate_train.pkl",  type=str)
    parser.add_argument("--ragate_val_pkl",
                        default="Datasets/RAGate/val/RAGate_val.pkl",    type=str)
    parser.add_argument("--ragate_test_pkl",
                        default="Datasets/RAGate/test/RAGate_test.pkl",   type=str)

    # Split files — .pt
    parser.add_argument("--ragate_train_pt",
                        default="Datasets/RAGate/train/RAGate_train.pt",   type=str)
    parser.add_argument("--ragate_val_pt",
                        default="Datasets/RAGate/val/RAGate_val.pt",     type=str)
    parser.add_argument("--ragate_test_pt",
                        default="Datasets/RAGate/test/RAGate_test.pt",    type=str)

    # ── RAGate extraction / generation controls ────────────────────────────

    # Sol: extract-vs-regenerate threshold
    parser.add_argument(
        "--ragate_sol_min_dataset_size", default=200000, type=int,
        help="Minimum samples in sol_dataset master before RAGate extracts. "
             "Triggers Stage 1 re-run if below.",
    )
    parser.add_argument("--ragate_sol_count",
                        default=200000, type=int,
                        help="Number of sol samples in the RAGate dataset.")
    parser.add_argument("--ragate_sol_template_range_start",   default=36,        type=int)
    parser.add_argument("--ragate_sol_template_range_end",     default=150,       type=int)
    parser.add_argument("--ragate_sol_schedule_steps",         default="0.2,0.3", type=str)
    parser.add_argument("--ragate_sol_min_path_length",        default=5,         type=int)
    parser.add_argument("--ragate_sol_min_template_diversity", default=10,        type=int)

    # Nosol: extract-vs-regenerate threshold
    parser.add_argument(
        "--ragate_nosol_min_dataset_size", default=200000, type=int,
        help="Minimum nosol samples in nosol_dataset master before RAGate "
             "extracts. Triggers Stage 1 re-run if below.",
    )
    parser.add_argument("--ragate_nosol_count",
                        default=100000, type=int,
                        help="Number of nosol samples in the RAGate dataset.")
    parser.add_argument("--ragate_nosol_disconnected_template_range_start",
                        default=1,  type=int)
    parser.add_argument("--ragate_nosol_disconnected_template_range_end",
                        default=35, type=int)
    parser.add_argument(
        "--ragate_nosol_disconnected_schedule_steps", default="0.0,0.1", type=str,
        help="Open-percent schedule for disconnected-component nosol samples.",
    )
    parser.add_argument(
        "--ragate_nosol_isolated_schedule_steps", default="0.2,0.3", type=str,
        help="Open-percent schedule for isolated-goal nosol samples.",
    )
    parser.add_argument(
        "--ragate_nosol_isolated_goal_fraction", default=0.3, type=float,
        help="Fraction of nosol samples using the isolated-goal method.",
    )

    # =========================================================================
    # INSPECTION SETTINGS
    # inspect_datasets.py reads these to control depth and scope.
    # =========================================================================

    # Which inspection level(s) to run.
    # If both are True, Quick runs first then Detailed follows.
    parser.add_argument("--inspect_quick",    default=True,  type=bool,
                        help="Run Quick Datasets Inspection — fast sanity check: "
                             "file existence, sample counts, field name presence "
                             "(first 5 samples), tensor shapes. "
                             "Completes in under 60 seconds.")
    parser.add_argument("--inspect_detailed", default=True, type=bool,
                        help="Run Detailed Datasets Inspection — full quality "
                             "sweep: every PKL sample checked for route integrity, "
                             "field values, nosol structure, cross-format "
                             "consistency. Slower (5-15 minutes for full dataset).")

    # Dataset group toggles (apply to both inspection levels).
    parser.add_argument("--inspect_masters",    default=True, type=bool,
                        help="Inspect masters group (sol_dataset, nosol_dataset).")
    parser.add_argument("--inspect_navigation", default=True, type=bool,
                        help="Inspect PSTAR group (train/val/test splits).")
    parser.add_argument("--inspect_gridcheck",  default=True, type=bool,
                        help="Inspect RAGate group (classifier dataset).")

    # Detailed-mode depth controls.
    parser.add_argument("--inspect_tensor_spot_size", default=200, type=int,
                        help="Number of .pt entries spot-checked for tensor/"
                             "raw-field consistency in Detailed mode. "
                             "Full sweep used for RAGate.")

    # =========================================================================
    # PIPELINE ORCHESTRATION SWITCHES
    # =========================================================================

    # Stage 1 — base dataset generation.
    # False = skip entirely (overrides all target checks).
    # True  = run normal skip / top-up / scratch logic.
    parser.add_argument("--run_stage1_generation",   default=True, type=bool,
                        help="Run Stage 1 base dataset generation.")
    parser.add_argument("--run_RAGate_generation", default=True, type=bool,
                        help="Run Stage 2 RAGate classifier dataset generation.")
    parser.add_argument("--run_validation",           default=True, type=bool,
                        help="Run Stage 4 dataset validation (inspect_datasets.py).")

    # -------------------------------------------------------------------------
    # Visualization / Logging
    # -------------------------------------------------------------------------
    parser.add_argument("--show_animation",       default=False, type=bool)
    parser.add_argument("--show_finalimage",      default=False, type=bool)
    parser.add_argument("--verbose",              default=False, type=bool,
                        help="Print every route addition and skip. Keep False "
                             "for large runs — progress is shown via chunk "
                             "flush messages instead.")
    parser.add_argument("--animation_pause_time", default=0.1,   type=float)
    parser.add_argument("--save_route_plots",     default=True, type=bool)

    # -------------------------------------------------------------------------
    # Dataset Saving (legacy outputs — kept for backwards compatibility)
    # -------------------------------------------------------------------------
    parser.add_argument("--output_dir",        default="Datasets/",             type=str)
    parser.add_argument("--dataset_pkl_file",  default="Datasets/dataset.pkl",  type=str)
    parser.add_argument("--dataset_json_file", default="Datasets/dataset.json", type=str)

    # -------------------------------------------------------------------------
    # Pathfinding Settings
    # -------------------------------------------------------------------------
    parser.add_argument("--max_depth_shortest_route",     default=74, type=int)
    parser.add_argument("--min_depth_shortest_route",     default=0,  type=int)
    parser.add_argument("--max_route_length_for_dataset", default=75, type=int)
    parser.add_argument("--min_route_length_for_dataset", default=0,  type=int)

    # Target counts for Stage 1 generation.
    # 0 = single pass (original behaviour).
    parser.add_argument("--nosol_target_count", default=250000, type=int,
                        help="Target nosol sample count for Stage 1. "
                             "Loop repeats until reached. 0 = single pass only.")
    parser.add_argument("--sol_target_count",   default=250000, type=int,
                        help="Target sol sample count for Stage 1. "
                             "Stops early if reached; repeats if first pass falls "
                             "short. 0 = single pass.")

    # -------------------------------------------------------------------------
    # Dataset Structure & Padding
    # -------------------------------------------------------------------------
    parser.add_argument("--context_matrix_length",    default=100,  type=int)
    parser.add_argument("--foundroute_matrix_length", default=75,   type=int)
    parser.add_argument("--mask_matrix_length",       default=75,   type=int)
    parser.add_argument("--target_matrix_length",     default=75,   type=int)
    parser.add_argument("--feature_size",             default=4,    type=int)
    parser.add_argument("--padding_routes",           default=True, type=bool)
    parser.add_argument("--padding_value",            default=-1,   type=int)
    parser.add_argument("--masking",                  default=True, type=bool)
    parser.add_argument("--masking_value",            default=-1,   type=int)

    # -------------------------------------------------------------------------
    # State Status Codes
    # -------------------------------------------------------------------------
    parser.add_argument("--status_start",    default=1, type=int)
    parser.add_argument("--status_end",      default=2, type=int)
    parser.add_argument("--status_obstacle", default=3, type=int)
    parser.add_argument("--status_free",     default=4, type=int)
    parser.add_argument("--status_path",     default=5, type=int)

    # -------------------------------------------------------------------------
    # Train / Val / Test Split
    # -------------------------------------------------------------------------
    parser.add_argument("--split_ratio",     default=0.80, type=float)
    parser.add_argument("--train_ratio",     default=0.80, type=float)
    parser.add_argument("--val_ratio",       default=0.10, type=float)
    parser.add_argument("--test_ratio",      default=0.10, type=float)
    parser.add_argument("--split_seed",      default=42,   type=int)
    parser.add_argument("--export_test_set", default=True, type=bool)

    # -------------------------------------------------------------------------
    # Parse and post-process
    # -------------------------------------------------------------------------
    # ── Image rendering (Stage 3) — CONFIG-DRIVEN (edit the defaults here) ────
    #     Renders ImageNet-style JPEGs of the sol/nosol MASTER grids
    #     (start/goal/obstacles/path) into
    #       Datasets/Images/<mode>/<size>/<split>/<class>/<id>.jpg
    #     Set generate_images=True to enable; off by default (large output).
    parser.add_argument("--generate_images", default=True, type=bool,
                        help="Set True to render the ImageNet-style master JPEGs (Stage 3).")
    parser.add_argument("--image_sizes",  default="64", type=str, # 64,128,224
                        help="Comma-separated square resolutions to render, e.g. 64,128,224.")
    parser.add_argument("--image_modes",  default="color", type=str, # color,gray
                        help="Comma-separated colour modes to render: color and/or gray.")
    parser.add_argument("--image_format", default="jpg", type=str)
    parser.add_argument("--images_dir",   default="Datasets/Images", type=str)

    args, _ = parser.parse_known_args()
    args.grid_size         = tuple(map(int, args.grid_size.split(",")))
    args.image_sizes       = [int(s) for s in str(args.image_sizes).split(",") if str(s).strip()]
    args.image_modes       = [m.strip() for m in str(args.image_modes).split(",") if m.strip()]
    args.num_unique_tokens = args.grid_size[0] * args.grid_size[1]

    cfg = vars(args)
    cfg.update({
        # Raw field names — used by ML pipeline dataloader and all dataset scripts.
        # Case-sensitive. Must match exactly across all formats (JSON, PKL, .pt).
        "context_field":             "Context",
        "found_routes_field":        "Found_Routes",
        "masking_field":             "Masking",
        "targets_field":             "Targets",
        "tourcost_field":            "TourCost",
        # Direction mappings
        "direction_map":             {"U": 0, "D": 1, "R": 2, "L": 3},
        "reverse_direction_mapping": {0: "U", 1: "D", 2: "R", 3: "L"},
        # Number of movement directions (U/D/L/R = 4).
        # Distinct from num_unique_tokens (grid vocabulary size = rows × cols).
        "num_directions":            4,
    })

    return cfg


dataset_config = get_dataset_config()
