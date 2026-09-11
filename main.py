import sys
import os
import subprocess
import importlib
import importlib.util


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _get_cuda_major():
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True,
                           text=True, timeout=10)
        if r.returncode == 0:
            import re
            m = re.search(r"release (\d+)\.\d+", r.stdout)
            if m:
                return int(m.group(1))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True,
                           text=True, timeout=10)
        if r.returncode == 0:
            import re
            m = re.search(r"CUDA Version:\s*(\d+)", r.stdout)
            if m:
                return int(m.group(1))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _install(pkg):
    print(f"[deps] Installing: {pkg} ...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
        capture_output=False,
    )
    if r.returncode != 0:
        print(f"[deps] WARNING: failed to install {pkg}. Install manually.")


def _ensure_dependencies():
    if importlib.util.find_spec("numpy") is None:
        _install("numpy>=1.24.0")
    if importlib.util.find_spec("matplotlib") is None:
        _install("matplotlib>=3.7.0")
    if importlib.util.find_spec("cupy") is None:
        v   = _get_cuda_major()
        pkg = ("cupy-cuda12x" if v is None or v >= 12
               else "cupy-cuda11x" if v == 11 else None)
        if pkg is None:
            print(f"[deps] CUDA {v} — install cupy manually.")
        else:
            _install(pkg)
    print("[deps] All dependencies satisfied.")


_ensure_dependencies()


# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------

import pickle
import random
import torch

from config import dataset_config as config
from dataset_utils import (
    extract_obstacles,
    context_matrix,
    foundroutes_matrix,
    mask_matrix,
    target_matrix,
    save_to_pickle,
    convert_pkl_to_json,
    split_data,
    export_splits_json,
    export_splits_pickle,
    validate_route,
    truncate_route_to_goal,
    load_templates,
    template_to_obstacles,
    open_obstacles,
    sample_start_goal,
    build_nosol_sample,
    is_goal_reachable,
    find_k_shortest_paths_on_grid,
    finalise_dataset_streaming,
    count_pkl_samples,
)
import dataset_utils as _utils

from astar_cupy import (
    is_accessible,
    reset_grid,
    set_start_and_goal,
    generate_random_obstacles,
    grid_size,
    map_grid,
    set_obstacles,
    find_route,
)

_VIZ_ENABLED = (
    config.get("show_animation", False) or config.get("show_finalimage", False)
)
if _VIZ_ENABLED:
    import matplotlib.pyplot as plt
    from visualize_routes import plot_route_matplotlib
else:
    plt                   = None
    plot_route_matplotlib = None


# ---------------------------------------------------------------------------
# Streaming chunk writer
# ---------------------------------------------------------------------------

class _ChunkWriter:
    """Streams generated samples to disk in fixed-size chunk files."""

    def __init__(self, base_name, chunk_size=500):
        self.base_name  = base_name
        self.chunk_size = chunk_size
        self._buffer    = []
        self._chunk_idx = 0
        self.paths      = []
        self._total     = 0
        chunk_dir = os.path.join(config["output_dir"], "_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        self._chunk_dir = chunk_dir

    def append(self, sample):
        self._buffer.append(sample)
        self._total += 1
        if len(self._buffer) >= self.chunk_size:
            self._flush()

    def flush(self):
        if self._buffer:
            self._flush()

    def _flush(self):
        path = os.path.join(
            self._chunk_dir,
            f"{self.base_name}_chunk_{self._chunk_idx:05d}.pkl",
        )
        with open(path, "wb") as f:
            pickle.dump(self._buffer, f)
        print(f"  [chunk] {self.base_name}: wrote {len(self._buffer)} samples "
              f"→ {os.path.basename(path)}  (running total: {self._total})")
        self.paths.append(path)
        self._buffer    = []
        self._chunk_idx += 1

    @property
    def total(self):
        return self._total


class _Sink:
    """Absorbs samples silently — used when a class does not need generation."""
    total = 0
    paths = []
    def append(self, _): pass
    def flush(self):     pass


# ---------------------------------------------------------------------------
# Progress counter (sol routes)
# ---------------------------------------------------------------------------

_sol_routes_added  = 0
_PROGRESS_INTERVAL = 10_000


# ---------------------------------------------------------------------------
# Sample assembly helpers
# ---------------------------------------------------------------------------

def _append_sol_sample(route, start, goal, obstacles, experiment_idx, writer):
    global _sol_routes_added

    if not route:
        return False

    is_valid, reason = validate_route(route, grid_size, start, goal, obstacles)
    if not is_valid:
        if config.get("verbose", False):
            print(f"  Skipping invalid route: {reason}")
        return False

    route = truncate_route_to_goal(route, grid_size, goal)
    if not route:
        return False

    tour_cost = len(route)
    if (tour_cost < config["min_route_length_for_dataset"] or
            tour_cost > config["max_route_length_for_dataset"]):
        return False

    writer.append({
        "Experiment_number": str(experiment_idx + 1),
        "Context":      context_matrix(grid_size, obstacles, start, goal),
        "Found_Routes": foundroutes_matrix(route, grid_size, start, goal),
        "Masking":      mask_matrix(route, grid_size, obstacles, start, goal),
        "Targets":      target_matrix(route, grid_size, goal),
        "TourCost":     tour_cost,
    })

    _sol_routes_added += 1
    if _sol_routes_added % _PROGRESS_INTERVAL == 0:
        print(f"  [progress] sol routes added: {_sol_routes_added:,}  "
              f"(experiment_idx={experiment_idx + 1})")

    if config.get("save_route_plots", False) and plot_route_matplotlib is not None:
        plot_route_matplotlib(
            start=start, goal=goal, route=route, obstacles=obstacles,
            sample_idx=experiment_idx,
            save_folder=os.path.join(config["output_dir"], "plotRoutes"),
        )
    if _VIZ_ENABLED:
        _maybe_visualize_with_astar(start, goal, obstacles)

    return True


def _process_pair(start, goal, obstacles, experiment_idx,
                  sol_writer, nosol_writer, sol_needed, nosol_needed):
    k      = config["max_finding_shortest_routes"]
    routes = find_k_shortest_paths_on_grid(
        start, goal, obstacles, k=k, grid_shape=grid_size
    )

    sol_added = nosol_added = 0

    if not routes:
        if nosol_needed:
            nosol_writer.append(
                build_nosol_sample(start, goal, obstacles, experiment_idx)
            )
            experiment_idx += 1
            nosol_added     = 1
        return experiment_idx, sol_added, nosol_added

    if sol_needed:
        for route in routes:
            ok = _append_sol_sample(
                route, start, goal, obstacles, experiment_idx, sol_writer
            )
            if ok:
                experiment_idx += 1
                sol_added      += 1

    return experiment_idx, sol_added, nosol_added


# ---------------------------------------------------------------------------
# Shared generation loop  (used by both templates and legacy modes)
# ---------------------------------------------------------------------------

def _run_generation_loop(pair_source, generate_sol, generate_nosol,
                         sol_target, nosol_target):
    chunk_size   = int(config.get("chunk_size", 500))
    sol_writer   = _ChunkWriter("sol",   chunk_size=chunk_size) if generate_sol   else _Sink()
    nosol_writer = _ChunkWriter("nosol", chunk_size=chunk_size) if generate_nosol else _Sink()

    experiment_idx = 0
    pass_num       = 0

    while True:
        pass_num   += 1
        done        = False

        for start, goal, obstacles in pair_source():

            sol_needed   = (generate_sol   and
                            (sol_target   == 0 or sol_writer.total   < sol_target))
            nosol_needed = (generate_nosol and
                            (nosol_target == 0 or nosol_writer.total < nosol_target))

            if not sol_needed and not nosol_needed:
                done = True
                break

            try:
                if _VIZ_ENABLED:
                    set_start_and_goal(start, goal)

                # Forward: start → goal
                experiment_idx, s_fwd, _ = _process_pair(
                    start, goal, obstacles, experiment_idx,
                    sol_writer, nosol_writer, sol_needed, nosol_needed,
                )

                # Re-check after forward
                sol_needed   = (generate_sol   and
                                (sol_target   == 0 or sol_writer.total   < sol_target))
                nosol_needed = (generate_nosol and
                                (nosol_target == 0 or nosol_writer.total < nosol_target))

                if not sol_needed and not nosol_needed:
                    done = True
                    break

                # Reverse: goal → start (sol only — avoids double-counting nosol)
                if sol_needed and s_fwd > 0:
                    if _VIZ_ENABLED:
                        reset_grid()
                        set_start_and_goal(goal, start)

                    experiment_idx, _, _ = _process_pair(
                        goal, start, obstacles, experiment_idx,
                        sol_writer, nosol_writer,
                        sol_needed=sol_needed,
                        nosol_needed=False,
                    )

            except Exception as e:
                print(f"  Error ({start}→{goal}): {e}")
                if nosol_needed:
                    nosol_writer.append(
                        build_nosol_sample(start, goal, obstacles, experiment_idx)
                    )
                    experiment_idx += 1
            finally:
                if _VIZ_ENABLED and plt is not None:
                    plt.close("all")

            if done:
                break

        # End of pass
        sol_done   = (not generate_sol   or sol_target   == 0
                      or sol_writer.total   >= sol_target)
        nosol_done = (not generate_nosol or nosol_target == 0
                      or nosol_writer.total >= nosol_target)

        if sol_done and nosol_done:
            if sol_target > 0 or nosol_target > 0:
                print(f"  Targets reached after {pass_num} pass(es).  "
                      f"sol={sol_writer.total:,}  nosol={nosol_writer.total:,}")
            break

        if sol_target == 0 and nosol_target == 0:
            break   # single-pass mode

        print(f"  Pass {pass_num} complete — "
              f"sol={sol_writer.total:,}/{sol_target:,}  "
              f"nosol={nosol_writer.total:,}/{nosol_target:,}")

    if generate_sol and not isinstance(sol_writer, _Sink):
        sol_writer.flush()
        print(f"  Sol    : {sol_writer.total:,} new samples")

    if generate_nosol and not isinstance(nosol_writer, _Sink):
        nosol_writer.flush()
        print(f"  Nosol  : {nosol_writer.total:,} new samples")

    return sol_writer, nosol_writer


# ---------------------------------------------------------------------------
# Templates mode pair source
# ---------------------------------------------------------------------------

def _templates_pair_source():
    templates    = load_templates()
    template_ids = list(templates.keys())
    if not template_ids:
        raise RuntimeError("No templates found. Cannot generate dataset.")

    schedule_str = str(config.get("open_percent_schedule", "")).strip()
    open_percent = config["open_obstacles_percent"]
    schedule     = ([float(x.strip()) for x in schedule_str.split(",") if x.strip()]
                    if schedule_str else [open_percent])

    rng = random.Random(config["split_seed"])

    print(f"  Open-percent schedule : {schedule}")
    print(f"  Templates             : {len(template_ids)}")
    print(f"  Chunk size            : {config.get('chunk_size', 500)}")

    def _source():
        shuffled = template_ids[:]
        rng.shuffle(shuffled)
        for tid in shuffled:
            base_obs = template_to_obstacles(templates[tid])
            for open_pct in schedule:
                reset_grid()
                obstacles = open_obstacles(base_obs, open_pct, rng)
                set_obstacles(obstacles)
                for _ in range(config["grid_reuse_count"]):
                    start, goal = sample_start_goal(obstacles, rng)
                    if start is None:
                        continue
                    yield start, goal, obstacles

    return _source


# ---------------------------------------------------------------------------
# Legacy mode pair source
# ---------------------------------------------------------------------------

def _legacy_pair_source():
    rng = random.Random(config["split_seed"])

    print(f"  Max obstacles : {config['max_obstacles']}")
    print(f"  Max attempts  : {config['max_attempts']}")
    print(f"  Grid reuse    : {config['grid_reuse_count']}")

    def _source():
        for _ in range(config["max_attempts"]):
            reset_grid()
            generate_random_obstacles(config["max_obstacles"])
            obstacles      = extract_obstacles()
            obstacle_set   = set(obstacles)

            for _ in range(config["grid_reuse_count"]):
                # Random start/goal avoiding obstacles
                tries = 0
                while tries < 200:
                    tries += 1
                    start = (random.randint(0, grid_size[0] - 1),
                             random.randint(0, grid_size[1] - 1))
                    goal  = (random.randint(0, grid_size[0] - 1),
                             random.randint(0, grid_size[1] - 1))
                    if (start != goal
                            and start not in obstacle_set
                            and goal  not in obstacle_set):
                        break
                else:
                    continue

                set_start_and_goal(start, goal)

                if (not is_accessible(start, map_grid.get(), grid_size) or
                        not is_accessible(goal,  map_grid.get(), grid_size)):
                    continue

                yield start, goal, obstacles

    return _source


# ---------------------------------------------------------------------------
# Finalise helpers
# ---------------------------------------------------------------------------

def _finalise_dataset(base_name, chunk_paths, total_samples):
    if not chunk_paths:
        print(f"No new {base_name} samples; skipping finalisation.")
        return

    label = 1 if base_name == "sol" else 0

    finalise_dataset_streaming(
        base_name       = base_name,
        chunk_paths     = chunk_paths,
        total_samples   = total_samples,
        master_pkl      = config[f"{base_name}_dataset_pkl_file"],
        master_json     = config[f"{base_name}_dataset_json_file"],
        train_pkl       = config[f"dataset_train_{base_name}_pkl_file"],
        val_pkl         = config[f"dataset_val_{base_name}_pkl_file"],
        test_pkl        = config[f"dataset_test_{base_name}_pkl_file"],
        train_json      = config[f"dataset_train_{base_name}_json_file"],
        val_json        = config[f"dataset_val_{base_name}_json_file"],
        test_json       = config[f"dataset_test_{base_name}_json_file"],
        train_ratio     = config["train_ratio"],
        val_ratio       = config["val_ratio"],
        split_seed      = config["split_seed"],
        export_test_set = config["export_test_set"],
        master_pt       = config.get(f"{base_name}_dataset_pt_file"),
        train_pt        = config.get(f"dataset_train_{base_name}_pt_file"),
        val_pt          = config.get(f"dataset_val_{base_name}_pt_file"),
        test_pt         = config.get(f"dataset_test_{base_name}_pt_file"),
        label           = label,
    )


def _append_to_master(base_name, chunk_paths):
    if not chunk_paths:
        return

    master_pkl = config[f"{base_name}_dataset_pkl_file"]
    print(f"\n── Appending {base_name} top-up samples to {master_pkl} ──")

    new_samples = []
    for path in chunk_paths:
        with open(path, "rb") as f:
            new_samples.extend(pickle.load(f))
        try:
            os.remove(path)
        except OSError:
            pass

    print(f"  New samples to append: {len(new_samples):,}")

    with open(master_pkl, "ab") as f:
        for sample in new_samples:
            pickle.dump(sample, f)

    total = count_pkl_samples(master_pkl)
    print(f"  Master PKL total after append: {total:,}")

    print(f"  Rebuilding {base_name} JSON and splits from updated master...")
    _rebuild_splits_from_master(base_name, master_pkl, total)


def _rebuild_splits_from_master(base_name, master_pkl, total_samples):
    rng     = random.Random(config["split_seed"])
    indices = list(range(total_samples))
    rng.shuffle(indices)

    train_r = config["train_ratio"]
    val_r   = config["val_ratio"]
    n_train = int(train_r * total_samples)
    n_val   = int(val_r   * total_samples)
    if val_r > 0 and total_samples >= 2 and n_val == 0:
        n_train = max(0, n_train - 1)
        n_val   = 1

    assignment = [""] * total_samples
    for rank, idx in enumerate(indices):
        if   rank < n_train:             assignment[idx] = "train"
        elif rank < n_train + n_val:     assignment[idx] = "val"
        else:                            assignment[idx] = "test"

    # label: 1=sol, 0=nosol — written into every .pt entry
    label = 1 if base_name == "sol" else 0

    master_json = config[f"{base_name}_dataset_json_file"]
    master_pt   = config.get(f"{base_name}_dataset_pt_file")
    train_pkl   = config[f"dataset_train_{base_name}_pkl_file"]
    val_pkl     = config[f"dataset_val_{base_name}_pkl_file"]
    test_pkl    = config[f"dataset_test_{base_name}_pkl_file"]
    train_json  = config[f"dataset_train_{base_name}_json_file"]
    val_json    = config[f"dataset_val_{base_name}_json_file"]
    test_json   = config[f"dataset_test_{base_name}_json_file"]
    train_pt    = config.get(f"dataset_train_{base_name}_pt_file")
    val_pt      = config.get(f"dataset_val_{base_name}_pt_file")
    test_pt     = config.get(f"dataset_test_{base_name}_pt_file")
    export_test = config["export_test_set"]

    master_json_w = _utils._StreamingJsonWriter(master_json)
    train_pkl_w   = _utils._StreamingPickleWriter(train_pkl)
    val_pkl_w     = _utils._StreamingPickleWriter(val_pkl)
    test_pkl_w    = _utils._StreamingPickleWriter(test_pkl) if export_test else None
    train_json_w  = _utils._StreamingJsonWriter(train_json)
    val_json_w    = _utils._StreamingJsonWriter(val_json)
    test_json_w   = _utils._StreamingJsonWriter(test_json) if export_test else None

    split_pkl_w  = {"train": train_pkl_w,  "val": val_pkl_w,  "test": test_pkl_w}
    split_json_w = {"train": train_json_w, "val": val_json_w, "test": test_json_w}

    write_split_pt  = any(p for p in (train_pt, val_pt, test_pt))
    write_master_pt = master_pt is not None
    master_pt_buffer = []
    pt_buffers  = {"train": [], "val": [], "test": []}
    pt_paths    = {"train": train_pt, "val": val_pt, "test": test_pt}

    for idx, sample in enumerate(_utils._streaming_pickle_load(master_pkl)):
        split = assignment[idx]
        master_json_w.write(sample)
        split_pkl_w[split].write(sample)
        split_json_w[split].write(sample)
        if write_master_pt:
            master_pt_buffer.append(_utils._sample_to_pt_entry(sample, label=label))
        if write_split_pt and pt_paths.get(split):
            pt_buffers[split].append(_utils._sample_to_pt_entry(sample, label=label))

    master_json_w.close()
    train_pkl_w.close()
    val_pkl_w.close()
    if test_pkl_w:  test_pkl_w.close()
    train_json_w.close()
    val_json_w.close()
    if test_json_w: test_json_w.close()

    # Write master .pt
    if write_master_pt and master_pt_buffer:
        os.makedirs(
            os.path.dirname(master_pt) if os.path.dirname(master_pt) else ".",
            exist_ok=True,
        )
        torch.save(master_pt_buffer, master_pt)
        print(f"  Written: {master_pt} ({len(master_pt_buffer)} samples)")

    # Write split .pt files
    for split_name in ("train", "val", "test"):
        pt_path = pt_paths.get(split_name)
        if pt_path and pt_buffers[split_name]:
            os.makedirs(
                os.path.dirname(pt_path) if os.path.dirname(pt_path) else ".",
                exist_ok=True,
            )
            torch.save(pt_buffers[split_name], pt_path)
            print(f"  Written: {pt_path} ({len(pt_buffers[split_name])} samples)")

    print(f"  {base_name} splits rebuilt — "
          f"train={train_pkl_w.count}  "
          f"val={val_pkl_w.count}  "
          f"test={test_pkl_w.count if test_pkl_w else 0}")


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def _maybe_visualize_with_astar(start, goal, obstacles):
    reset_grid()
    set_obstacles(obstacles)
    set_start_and_goal(start, goal)
    _ = find_route(start, goal)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def _count_v3_nosol_samples(pkl_path):
    count = 0
    try:
        with open(pkl_path, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)
                except EOFError:
                    break
                samples = obj if isinstance(obj, list) else [obj]
                for s in samples:
                    if (isinstance(s, dict)
                            and isinstance(s.get("Found_Routes"), list)
                            and len(s.get("Found_Routes", [1])) == 0
                            and s.get("TourCost") == 0):
                        count += 1
    except Exception:
        pass
    return count


def _check_class_status(base_name):
    pkl_path = config[f"{base_name}_dataset_pkl_file"]
    target   = int(config.get(f"{base_name}_target_count", 0))
    exists   = os.path.exists(pkl_path)

    if not exists:
        return False, 0, target, (target > 0), 0

    if base_name == "nosol":
        current_count = _count_v3_nosol_samples(pkl_path)
    else:
        current_count = count_pkl_samples(pkl_path)

    if target == 0 or current_count >= target:
        return True, current_count, target, False, 0

    topup = target - current_count
    return True, current_count, target, True, topup


def _splits_complete(base_name):
    # Master .pt
    master_pt = config.get(f"{base_name}_dataset_pt_file")
    if master_pt and not os.path.exists(master_pt):
        return False
    # Split files
    for split in ("train", "val", "test"):
        for fmt in ("pkl", "json", "pt"):
            key  = f"dataset_{split}_{base_name}_{fmt}_file"
            path = config.get(key)
            if not path or not os.path.exists(path):
                return False
    return True


def _ensure_splits(base_name):

    master_pkl = config[f"{base_name}_dataset_pkl_file"]
    total      = count_pkl_samples(master_pkl)
    print(f"  {base_name}: master has {total:,} samples — "
          f"rebuilding missing split files...")
    _rebuild_splits_from_master(base_name, master_pkl, total)


def _run_stage1():

    global _sol_routes_added
    _sol_routes_added = 0

    mode = config.get("generation_mode", "templates").lower()

    sol_exists,   sol_count,   sol_target,   sol_needed,   sol_topup   = _check_class_status("sol")
    nosol_exists, nosol_count, nosol_target, nosol_needed, nosol_topup = _check_class_status("nosol")

    # Status report
    for name, exists, count, target, needed, topup in [
        ("sol",   sol_exists,   sol_count,   sol_target,   sol_needed,   sol_topup),
        ("nosol", nosol_exists, nosol_count, nosol_target, nosol_needed, nosol_topup),
    ]:
        if not exists:
            print(f"  {name:5s}: not found — "
                  f"generating {target:,} samples from scratch  [{mode} mode]")
        elif not needed:
            print(f"  {name:5s}: {count:,} samples >= target {target:,} — skipping")
        else:
            print(f"  {name:5s}: {count:,} samples < target {target:,} "
                  f"— topping up {topup:,} more samples  [{mode} mode]")

    if not sol_needed and sol_exists and not _splits_complete("sol"):
        print("  sol  : master complete but split files incomplete — rebuilding.")
        _ensure_splits("sol")

    if not nosol_needed and nosol_exists and not _splits_complete("nosol"):
        print("  nosol: master complete but split files incomplete — rebuilding.")
        _ensure_splits("nosol")

    if not sol_needed and not nosol_needed:
        print("\n[STAGE 1] Both classes at target — skipping generation.")
        return

    # Select pair source
    if mode == "templates":
        pair_source = _templates_pair_source()
    else:
        pair_source = _legacy_pair_source()

    # NEW samples needed this run (top-up amount, or full target if from scratch)
    sol_new   = sol_topup   if sol_topup   > 0 else sol_target
    nosol_new = nosol_topup if nosol_topup > 0 else nosol_target

    sol_writer, nosol_writer = _run_generation_loop(
        pair_source    = pair_source,
        generate_sol   = sol_needed,
        generate_nosol = nosol_needed,
        sol_target     = sol_new,
        nosol_target   = nosol_new,
    )

    # Finalise each class that was generated
    if sol_needed:
        chunks = sol_writer.paths if not isinstance(sol_writer, _Sink) else []
        total  = sol_writer.total if not isinstance(sol_writer, _Sink) else 0
        if sol_topup > 0:
            _append_to_master("sol", chunks)
        else:
            _finalise_dataset("sol", chunks, total)

    if nosol_needed:
        chunks = nosol_writer.paths if not isinstance(nosol_writer, _Sink) else []
        total  = nosol_writer.total if not isinstance(nosol_writer, _Sink) else 0
        if nosol_topup > 0:
            _append_to_master("nosol", chunks)
        else:
            _finalise_dataset("nosol", chunks, total)


def _ragate_dataset_complete():

    return all(os.path.exists(config[k]) for k in (
        "ragate_train_pkl", "ragate_val_pkl",  "ragate_test_pkl",
        "ragate_train_pt",  "ragate_val_pt",   "ragate_test_pt",
        "ragate_sol_pkl",   "ragate_nosol_pkl",
        "ragate_sol_pt",    "ragate_nosol_pt",
    ))


def _run_RAGate_generation():
    print("\n" + "=" * 70)
    print("  STAGE 2 — RAGate classifier dataset generation")
    print("=" * 70)
    import generate_ragate
    generate_ragate.main()


def _run_validation():
    print("\n" + "=" * 70)
    print("  STAGE 4 — Dataset inspection")
    print("=" * 70)
    import inspect_datasets
    try:
        inspect_datasets.main()
    except SystemExit as e:
        if e.code != 0:
            print("\nInspection reported failures — see report above.")
            raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n=== Pathfinding Dataset Generator V3 ===")
    print(f"  Generation mode: {config.get('generation_mode', 'templates')}")

    # ------------------------------------------------------------------
    # STAGE 1 — Base dataset
    # ------------------------------------------------------------------
    if not config.get("run_stage1_generation", True):
        print("\n[STAGE 1] Disabled in config — skipping.")
    else:
        print("\n[STAGE 1] Checking dataset status...")
        _run_stage1()

    # ------------------------------------------------------------------
    # STAGE 2 — RAGate classifier dataset
    # ------------------------------------------------------------------
    if not config.get("run_RAGate_generation", True):
        print("\n[STAGE 2] Disabled in config — skipping.")
    elif _ragate_dataset_complete():
        print("\n[STAGE 2] RAGate dataset files already exist — skipping.")
    else:
        try:
            _run_RAGate_generation()
        except Exception as e:
            from generate_ragate import DatasetTooSmallError
            if isinstance(e, DatasetTooSmallError):
                print(
                    f"\n[STAGE 2] {e}\n"
                    f"[STAGE 1] Topping up {e.dataset_label} to {e.required:,} samples "
                    f"(currently {e.found:,})..."
                )
                config[f"{e.dataset_label}_target_count"] = e.required
                _run_stage1()
                print("\n[STAGE 2] Retrying RAGate dataset generation...")
                _run_RAGate_generation()  # propagates if still fails
            else:
                raise

    # ------------------------------------------------------------------
    # STAGE 3 — ImageNet-style JPEG rendering of the master grids (optional)
    # ------------------------------------------------------------------
    if not config.get("generate_images", False):
        print("\n[STAGE 3] Image rendering disabled (--generate_images off) — skipping.")
    else:
        print("\n[STAGE 3] Rendering ImageNet-style master images...")
        from render_images import render_master_images
        render_master_images(config)

    # ------------------------------------------------------------------
    # STAGE 4 — Dataset inspection
    # ------------------------------------------------------------------
    if not config.get("run_validation", True):
        print("\n[STAGE 4] Disabled in config — skipping.")
    else:
        _run_validation()

    print("\n=== Pipeline complete. ===")


if __name__ == "__main__":
    main()
