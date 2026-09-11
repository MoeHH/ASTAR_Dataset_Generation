import os
import json
import pickle
import random
import math
import heapq
import torch
from collections import deque
from config import dataset_config as config
from astar_cupy import map_grid, grid_size, WHITE, BLACK

_BAKED_FEATURE_SIZE = 8   # f0..f7, fixed layout

def _find_goal_xy(context_cells, status_end, padding_value):
    """Goal (x, y) = context cell with State_Status == status_end.
    Fallbacks match the ML pipeline: first non-padded cell, then (1, 1)."""
    for cell in context_cells:
        cn = cell.get("Cell_Number", padding_value)
        x  = cell.get("x", padding_value)
        y  = cell.get("y", padding_value)
        if cn == padding_value or x == padding_value or y == padding_value:
            continue
        if cell.get("State_Status") == status_end:
            return float(x), float(y)
    for cell in context_cells:
        cn = cell.get("Cell_Number", padding_value)
        x  = cell.get("x", padding_value)
        y  = cell.get("y", padding_value)
        if cn == padding_value or x == padding_value or y == padding_value:
            continue
        return float(x), float(y)
    return 1.0, 1.0


def _bake_context_features(context_cells):
    padding_value   = int(config["padding_value"])
    status_end      = int(config["status_end"])
    status_obstacle = int(config["status_obstacle"])
    n_rows, n_cols  = config["grid_size"]
    max_neighbours  = float(config.get("max_neighbours", 4))

    goal_x, goal_y = _find_goal_xy(context_cells, status_end, padding_value)

    obstacle_cells = set()
    for cell in context_cells:
        cn = cell.get("Cell_Number", padding_value)
        if cn != padding_value and cell.get("State_Status") == status_obstacle:
            obstacle_cells.add(int(cn))

    def _extras(cell_num, x, y):
        dx = goal_x - float(x)
        dy = goal_y - float(y)
        row = (int(cell_num) - 1) // n_cols
        col = (int(cell_num) - 1) % n_cols
        free_count = 0
        has_oob    = False
        for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            r, c = row + dr, col + dc
            if not (0 <= r < n_rows and 0 <= c < n_cols):
                has_oob = True
            elif (r * n_cols + c + 1) not in obstacle_cells:
                free_count += 1
        free_nbrs   = free_count / max_neighbours
        is_interior = 0.0 if has_oob else 1.0
        return dx, dy, free_nbrs, is_interior

    baked = []
    extras_by_cellnum = {}
    for cell in context_cells:
        cn = cell.get("Cell_Number", padding_value)
        x  = cell.get("x", padding_value)
        y  = cell.get("y", padding_value)
        c2 = dict(cell)
        if cn == padding_value or x == padding_value or y == padding_value:
            baked.append(c2)
            continue
        dx, dy, fn, it = _extras(cn, x, y)
        c2["dx_to_goal"]      = dx
        c2["dy_to_goal"]      = dy
        c2["free_neighbours"] = fn
        c2["is_interior"]     = it
        baked.append(c2)
        extras_by_cellnum[int(cn)] = (dx, dy, fn, it)
    return baked, extras_by_cellnum


def _sample_to_pt_entry(sample, label=None):
    padding      = float(config["padding_value"])
    F            = _BAKED_FEATURE_SIZE                        # 8
    context_len  = int(config["context_matrix_length"])       # 100
    latent_len   = int(config["foundroute_matrix_length"])    # 75
    num_dirs     = int(config.get("num_directions", 4))
    seq_len      = context_len + latent_len                   # 175

    inp = torch.full((seq_len,    F),        padding, dtype=torch.float32)
    msk = torch.full((latent_len, num_dirs), padding, dtype=torch.float32)
    tgt = torch.full((latent_len, num_dirs), padding, dtype=torch.float32)

    raw_context      = sample.get("Context",      [])
    foundroute_cells = sample.get("Found_Routes", [])
    masking_data     = sample.get("Masking",      [])
    target_data      = sample.get("Targets",      [])

    context_cells, extras_by_cellnum = _bake_context_features(raw_context)

    pv = config["padding_value"]
    for i, cell in enumerate(context_cells[:context_len]):
        cn = cell.get("Cell_Number", pv)
        inp[i, 0] = float(cn)
        inp[i, 1] = float(cell.get("x", pv))
        inp[i, 2] = float(cell.get("y", pv))
        inp[i, 3] = float(cell.get("State_Status", pv))
        if cn != pv and int(cn) in extras_by_cellnum:
            dx, dy, fn, it = extras_by_cellnum[int(cn)]
            inp[i, 4] = dx; inp[i, 5] = dy; inp[i, 6] = fn; inp[i, 7] = it

    for i, cell in enumerate(foundroute_cells[:latent_len]):
        row = context_len + i
        cn  = cell.get("Cell_Number", pv)
        inp[row, 0] = float(cn)
        inp[row, 1] = float(cell.get("x", pv))
        inp[row, 2] = float(cell.get("y", pv))
        inp[row, 3] = float(cell.get("State_Status", pv))
        if cn != pv and int(cn) in extras_by_cellnum:
            dx, dy, fn, it = extras_by_cellnum[int(cn)]
            inp[row, 4] = dx; inp[row, 5] = dy; inp[row, 6] = fn; inp[row, 7] = it

    for i, entry in enumerate(masking_data[:latent_len]):
        vec = entry.get("mask_vec", [pv] * num_dirs)
        msk[i] = torch.tensor(vec[:num_dirs], dtype=torch.float32)

    for i, entry in enumerate(target_data[:latent_len]):
        vec = entry.get("target_vec", [pv] * num_dirs)
        tgt[i] = torch.tensor(vec[:num_dirs], dtype=torch.float32)

    entry = {
        "Context":      context_cells,
        "Found_Routes": sample.get("Found_Routes", []),
        "Masking":      sample.get("Masking",      []),
        "Targets":      sample.get("Targets",      []),
        "TourCost":     sample.get("TourCost",     0),
        "input_tensor":   inp,
        "masking_tensor": msk,
        "target_tensor":  tgt,
    }
    if label is not None:
        entry["label"] = int(label)
    return entry


def _save_split_pt(samples, path, label=None):

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    tensor_list = [_sample_to_pt_entry(s, label=label) for s in samples]
    torch.save(tensor_list, path)
    print(f"  Written: {path} ({len(tensor_list)} samples)")


def count_pkl_samples(path):

    if not os.path.exists(path):
        return 0

    count = 0
    try:
        with open(path, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)
                except EOFError:
                    break
                # Legacy format: the whole dataset is one pickled list
                if isinstance(obj, list):
                    count += len(obj)
                # Streaming format: each object is one sample dict
                elif isinstance(obj, dict):
                    count += 1
        return count
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Basic I/O
# ---------------------------------------------------------------------------

def stringify_cell_values(cell_values):
    if cell_values is None:
        return None
    return {f"{k[0]},{k[1]}": v for k, v in cell_values.items()}


def save_to_pickle(data, filename=None):
    if filename is None:
        filename = config["dataset_pkl_file"]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_from_pickle(filename=None):
    if filename is None:
        filename = config["dataset_pkl_file"]
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_dataset_file(dataset, filename=None):
    if filename is None:
        filename = config["dataset_json_file"]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=4)


# ---------------------------------------------------------------------------
# Streaming JSON + .idx writer
# ---------------------------------------------------------------------------

class _StreamingJsonWriter:
    def __init__(self, json_path):
        os.makedirs(
            os.path.dirname(json_path) if os.path.dirname(json_path) else ".",
            exist_ok=True,
        )
        self._json_path = json_path
        self._idx_path  = json_path + ".idx"
        self._file      = open(json_path, "w", encoding="utf-8")
        self._offsets   = []
        self._count     = 0
        self._file.write("[\n")

    def write(self, sample):
        if self._count > 0:
            self._file.write(",\n")
        self._offsets.append(self._file.tell())
        self._file.write(json.dumps(sample, indent=2))
        self._count += 1

    def close(self):
        self._file.write("\n]\n")
        self._file.close()
        with open(self._idx_path, "wb") as f:
            pickle.dump(self._offsets, f)
        print(f"  Written: {self._json_path} ({self._count} samples)")
        print(f"  Index  : {self._idx_path} ({len(self._offsets)} offsets)")

    @property
    def count(self):
        return self._count


# ---------------------------------------------------------------------------
# Streaming PKL writer
# ---------------------------------------------------------------------------

class _StreamingPickleWriter:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._path  = path
        self._file  = open(path, "wb")
        self._count = 0

    def write(self, sample):
        pickle.dump(sample, self._file)
        self._count += 1

    def close(self):
        self._file.close()

    @property
    def count(self):
        return self._count


def _streaming_pickle_load(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


# ---------------------------------------------------------------------------
# Streaming finalisation  (O(chunk_size) peak RAM)
# ---------------------------------------------------------------------------

def finalise_dataset_streaming(base_name, chunk_paths, total_samples,
                                master_pkl, master_json,
                                train_pkl, val_pkl, test_pkl,
                                train_json, val_json, test_json,
                                train_ratio, val_ratio, split_seed,
                                export_test_set=True,
                                master_pt=None,
                                train_pt=None, val_pt=None, test_pt=None,
                                label=None):
    if not chunk_paths:
        print(f"No {base_name} samples generated; skipping.")
        return

    print(f"\n── Finalising {base_name} ({total_samples:,} samples, "
          f"{len(chunk_paths)} chunks) ──────────────────────────────────────")

    rng     = random.Random(split_seed)
    indices = list(range(total_samples))
    rng.shuffle(indices)

    n_train = int(train_ratio * total_samples)
    n_val   = int(val_ratio   * total_samples)
    if val_ratio > 0 and total_samples >= 2 and n_val == 0:
        n_train = max(0, n_train - 1)
        n_val   = 1

    assignment = [""] * total_samples
    for rank, idx in enumerate(indices):
        if   rank < n_train:             assignment[idx] = "train"
        elif rank < n_train + n_val:     assignment[idx] = "val"
        else:                            assignment[idx] = "test"

    master_pkl_w  = _StreamingPickleWriter(master_pkl)
    master_json_w = _StreamingJsonWriter(master_json)
    train_pkl_w   = _StreamingPickleWriter(train_pkl)
    val_pkl_w     = _StreamingPickleWriter(val_pkl)
    test_pkl_w    = _StreamingPickleWriter(test_pkl) if export_test_set else None
    train_json_w  = _StreamingJsonWriter(train_json)
    val_json_w    = _StreamingJsonWriter(val_json)
    test_json_w   = _StreamingJsonWriter(test_json) if export_test_set else None

    split_pkl  = {"train": train_pkl_w,  "val": val_pkl_w,  "test": test_pkl_w}
    split_json = {"train": train_json_w, "val": val_json_w, "test": test_json_w}

    write_split_pt = any(p is not None for p in (train_pt, val_pt, test_pt))
    write_master_pt = master_pt is not None
    pt_buffers  = {"train": [], "val": [], "test": []}
    master_pt_buffer = []
    pt_paths    = {"train": train_pt, "val": val_pt, "test": test_pt}

    global_idx = 0
    for chunk_no, chunk_path in enumerate(chunk_paths, 1):
        with open(chunk_path, "rb") as f:
            chunk = pickle.load(f)

        for sample in chunk:
            split = assignment[global_idx]

            master_pkl_w.write(sample)
            master_json_w.write(sample)
            split_pkl[split].write(sample)
            split_json[split].write(sample)

            if write_master_pt:
                master_pt_buffer.append(_sample_to_pt_entry(sample, label=label))

            if write_split_pt and pt_paths.get(split):
                pt_buffers[split].append(_sample_to_pt_entry(sample, label=label))

            global_idx += 1

        del chunk

        if chunk_no % 100 == 0 or chunk_no == len(chunk_paths):
            print(f"  Processed chunk {chunk_no}/{len(chunk_paths)}  "
                  f"({global_idx:,}/{total_samples:,} samples)")

    master_pkl_w.close()
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

    print(f"\n  {base_name} splits — "
          f"train={train_pkl_w.count}  "
          f"val={val_pkl_w.count}  "
          f"test={test_pkl_w.count if test_pkl_w else 0}")

    for p in chunk_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    print(f"  Deleted {len(chunk_paths)} chunk file(s).")


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------

def merge_chunk_pkls(chunk_paths, master_pkl):
    dataset = []
    for path in chunk_paths:
        with open(path, "rb") as f:
            dataset.extend(pickle.load(f))
    os.makedirs(os.path.dirname(master_pkl) if os.path.dirname(master_pkl) else ".", exist_ok=True)
    with open(master_pkl, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def convert_pkl_to_json(pkl_file=None, json_file=None):
    if pkl_file is None:
        pkl_file = config["dataset_pkl_file"]
    if json_file is None:
        json_file = config["dataset_json_file"]

    def fix_case(obj):
        if isinstance(obj, dict):
            return {("x" if k == "X" else "y" if k == "Y" else k): fix_case(v)
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [fix_case(i) for i in obj]
        return obj

    writer = _StreamingJsonWriter(json_file)
    try:
        count = 0
        for sample in _streaming_pickle_load(pkl_file):
            fixed = {k: fix_case(v) for k, v in sample.items()
                     if k in ("Context", "Found_Routes", "Masking", "Targets", "TourCost")}
            writer.write(fixed)
            count += 1
        if count == 0:
            raise ValueError("Empty streaming PKL — trying legacy load.")
    except Exception:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        for sample in data:
            fixed = {k: fix_case(v) for k, v in sample.items()
                     if k in ("Context", "Found_Routes", "Masking", "Targets", "TourCost")}
            writer.write(fixed)

    writer.close()
    print(f"Converted {pkl_file} → {json_file}")


def split_data(data, train_ratio, val_ratio, split_seed):
    rng     = random.Random(split_seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    total   = len(data)
    n_train = int(train_ratio * total)
    n_val   = int(val_ratio   * total)

    if val_ratio > 0 and total >= 2 and n_val == 0:
        n_train = max(0, n_train - 1)
        n_val   = 1

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    return ([data[i] for i in train_idx],
            [data[i] for i in val_idx],
            [data[i] for i in test_idx])


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

def is_goal_reachable(start, goal, obstacles, grid_shape=None):
    if grid_shape is None:
        grid_shape = grid_size
    if start is None or goal is None:
        return False
    if start == goal:
        return True

    obstacle_set = set(obstacles)
    if start in obstacle_set or goal in obstacle_set:
        return False

    rows, cols = grid_shape
    q       = deque([start])
    visited = {start}

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            nxt    = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if nxt in obstacle_set or nxt in visited:
                continue
            if nxt == goal:
                return True
            visited.add(nxt)
            q.append(nxt)

    return False


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def export_templates(templates, templates_pkl_file=None, templates_json_file=None):
    if templates_pkl_file is None:
        templates_pkl_file = config["templates_pkl_file"]
    if templates_json_file is None:
        templates_json_file = config["templates_json_file"]

    os.makedirs(os.path.dirname(templates_pkl_file), exist_ok=True)
    with open(templates_pkl_file, "wb") as f:
        pickle.dump(templates, f)

    os.makedirs(os.path.dirname(templates_json_file), exist_ok=True)
    with open(templates_json_file, "w") as f:
        json.dump(templates, f, indent=2)


def load_templates(templates_pkl_file=None, templates_json_file=None):
    if templates_pkl_file is None:
        templates_pkl_file = config["templates_pkl_file"]
    if templates_json_file is None:
        templates_json_file = config["templates_json_file"]

    if os.path.exists(templates_pkl_file):
        with open(templates_pkl_file, "rb") as f:
            return pickle.load(f)
    if os.path.exists(templates_json_file):
        with open(templates_json_file, "r") as f:
            return json.load(f)

    from grid_templates import GRID_TEMPLATES
    export_templates(GRID_TEMPLATES,
                     templates_pkl_file=templates_pkl_file,
                     templates_json_file=templates_json_file)
    return GRID_TEMPLATES


def template_to_obstacles(template_grid):
    return [(r, c)
            for r, row in enumerate(template_grid)
            for c, v in enumerate(row)
            if int(v) == 1]


def open_obstacles(obstacles, open_percent, rng):
    if not obstacles:
        return []
    if open_percent <= 0:
        return list(obstacles)
    k = int(math.floor(open_percent * len(obstacles)))
    if k <= 0:
        return list(obstacles)
    if k >= len(obstacles):
        return []
    to_open = set(rng.sample(obstacles, k))
    return [o for o in obstacles if o not in to_open]


def sample_start_goal(obstacles, rng):
    obstacle_set = set(obstacles)
    free_cells = [(r, c)
                  for r in range(grid_size[0])
                  for c in range(grid_size[1])
                  if (r, c) not in obstacle_set]
    if len(free_cells) < 2:
        return None, None

    min_dist  = int(config.get("min_start_goal_distance", 0))
    max_tries = int(config.get("max_attempts", 50))

    for _ in range(max_tries):
        start = rng.choice(free_cells)
        goal  = rng.choice(free_cells)
        if goal == start:
            continue
        if min_dist > 0 and (abs(start[0] - goal[0]) + abs(start[1] - goal[1])) < min_dist:
            continue
        return start, goal

    for _ in range(max_tries):
        start = rng.choice(free_cells)
        goal  = rng.choice(free_cells)
        if goal != start:
            return start, goal

    return None, None


# ---------------------------------------------------------------------------
# No-solution sample builder  (V3 spec: empty lists, TourCost=0)
# ---------------------------------------------------------------------------

def build_nosol_sample(start, goal, obstacles, experiment_idx):
    """
    Build a no-solution sample dict.

    V3 spec: Found_Routes, Masking, and Targets are empty lists — no path
    exists. TourCost is 0. Context is fully populated so the transformer
    can still read the grid layout.
    """
    return {
        "Experiment_number": str(experiment_idx + 1),
        "Context":      context_matrix(grid_size, obstacles, start, goal),
        "Found_Routes": [],
        "Masking":      [],
        "Targets":      [],
        "TourCost":     0,
    }


# ---------------------------------------------------------------------------
# Obstacle extraction
# ---------------------------------------------------------------------------

def extract_obstacles():
    return [(r, c)
            for r in range(grid_size[0])
            for c in range(grid_size[1])
            if map_grid[r, c] == BLACK]


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------

def context_matrix(grid_size, obstacles, start, goal, context_length=None):
    if context_length is None:
        context_length = config["context_matrix_length"]

    rows, columns = grid_size
    obstacle_set  = set(obstacles)
    context       = []

    for idx in range(rows * columns):
        y, x  = divmod(idx, columns)
        cell  = (y, x)
        if   cell == start:        state = config["status_start"]
        elif cell == goal:         state = config["status_end"]
        elif cell in obstacle_set: state = config["status_obstacle"]
        else:                      state = config["status_free"]

        context.append({
            "Cell_Number":  idx + 1,
            "x":            x + 1,
            "y":            y + 1,
            "State_Status": state,
        })

    while len(context) < context_length:
        context.append({
            "Cell_Number":  config["padding_value"],
            "x":            config["padding_value"],
            "y":            config["padding_value"],
            "State_Status": config["padding_value"],
        })
    return context


def truncate_route_to_goal(route, grid_size, goal):
    columns  = grid_size[1]
    goal_idx = goal[0] * columns + goal[1]
    if not route:
        return []
    try:
        return route[:route.index(goal_idx) + 1]
    except ValueError:
        return []


def validate_route(route, grid_size, start, goal, obstacles):
    if not route:
        return False, "Route is empty."
    columns   = grid_size[1]
    start_idx = start[0] * columns + start[1]
    goal_idx  = goal[0]  * columns + goal[1]

    if route[0] != start_idx:
        return False, f"Route does not start at start. route[0]={route[0]} start_idx={start_idx}"
    if goal_idx not in route:
        return False, "Route does not contain the goal."
    if route[-1] != goal_idx:
        return False, "Route does not end at the goal."
    if len(route) != len(set(route)):
        return False, "Route contains revisits."

    obstacle_set = set(obstacles)
    for idx in route:
        y, x = divmod(idx, columns)
        if (y, x) in obstacle_set:
            return False, "Route passes through an obstacle."
    for a, b in zip(route, route[1:]):
        ay, ax = divmod(a, columns)
        by, bx = divmod(b, columns)
        if abs(ay - by) + abs(ax - bx) != 1:
            return False, f"Non-adjacent move: {a} → {b}"
    return True, "OK"


def foundroutes_matrix(route, grid_size, start, goal, foundroute_length=None):
    if foundroute_length is None:
        foundroute_length = config["foundroute_matrix_length"]

    rows, columns = grid_size
    foundroutes   = []

    if not route:
        while len(foundroutes) < foundroute_length:
            foundroutes.append({k: config["padding_value"]
                                for k in ("Cell_Number", "x", "y", "State_Status")})
        return foundroutes

    goal_idx = route[-1]
    for i, idx in enumerate(route):
        y, x  = divmod(idx, columns)
        state = (config["status_start"] if i == 0
                 else config["status_end"] if idx == goal_idx
                 else config["status_path"])
        foundroutes.append({
            "Cell_Number":  idx + 1,
            "x":            x + 1,
            "y":            y + 1,
            "State_Status": state,
        })

    while len(foundroutes) < foundroute_length:
        foundroutes.append({k: config["padding_value"]
                            for k in ("Cell_Number", "x", "y", "State_Status")})
    return foundroutes


def mask_matrix(route, grid_size, obstacles, start, goal, mask_length=None):
    if mask_length is None:
        mask_length = config["mask_matrix_length"]

    direction_map = config["direction_map"]
    columns       = grid_size[1]

    def _pad():
        e = {d: {"cell": config["padding_value"], "dir": i}
             for d, i in direction_map.items()}
        e["mask_vec"] = [-1, -1, -1, -1]
        return e

    if not route:
        return [_pad() for _ in range(mask_length)]

    obstacle_set = set(obstacles)
    goal_idx     = goal[0] * columns + goal[1]
    visited      = {route[0]}
    mask         = []

    for t, current in enumerate(route):
        if current == goal_idx:
            break
        m = {}; mask_vec = []
        cy, cx = divmod(current, columns)
        for dir_key, dir_idx in direction_map.items():
            if   dir_key == "U": ny, nx = cy - 1, cx
            elif dir_key == "D": ny, nx = cy + 1, cx
            elif dir_key == "R": ny, nx = cy, cx + 1
            else:                ny, nx = cy, cx - 1

            valid = True; cell_val = config["padding_value"]
            if not (0 <= ny < grid_size[0] and 0 <= nx < grid_size[1]):
                valid = False
            elif (ny, nx) in obstacle_set:
                valid = False
            else:
                candidate = ny * columns + nx
                if candidate in visited:
                    valid = False
                else:
                    cell_val = candidate + 1
            m[dir_key] = {"cell": cell_val, "dir": dir_idx}
            mask_vec.append(1 if valid else 0)
        m["mask_vec"] = mask_vec
        mask.append(m)
        if (t + 1) < len(route):
            visited.add(route[t + 1])

    while len(mask) < mask_length:
        mask.append(_pad())
    return mask


def target_matrix(route, grid_size, goal, target_length=None):
    if target_length is None:
        target_length = config["target_matrix_length"]

    direction_map = config["direction_map"]
    columns       = grid_size[1]

    def _pad():
        e = {d: {"cell": config["padding_value"], "dir": i}
             for d, i in direction_map.items()}
        e["target_vec"] = [-1, -1, -1, -1]
        return e

    if len(route) < 2:
        return [_pad() for _ in range(target_length)]

    goal_idx = goal[0] * columns + goal[1]
    target   = []

    for i in range(len(route) - 1):
        from_idx = route[i]
        to_idx   = route[i + 1]
        if from_idx == goal_idx:
            break

        from_y, from_x = divmod(from_idx, columns)
        to_y,   to_x   = divmod(to_idx,   columns)

        t = {}
        for dir_key, dir_idx in direction_map.items():
            cell_val = config["padding_value"]
            if   dir_key == "U":
                c = from_idx - columns
                if c >= 0: cell_val = c + 1
            elif dir_key == "D":
                c = from_idx + columns
                if c < grid_size[0] * grid_size[1]: cell_val = c + 1
            elif dir_key == "R":
                if (from_idx + 1) % columns != 0: cell_val = from_idx + 2
            else:
                if from_idx % columns != 0: cell_val = from_idx
            t[dir_key] = {"cell": cell_val, "dir": dir_idx}

        dx = to_x - from_x; dy = to_y - from_y
        if   dx == 0 and dy == -1: di = 0
        elif dx == 0 and dy ==  1: di = 1
        elif dx == 1 and dy ==  0: di = 2
        elif dx == -1 and dy == 0: di = 3
        else:                      di = -1

        t["target_vec"] = ([1 if j == di else 0 for j in range(4)]
                           if di != -1 else [-1, -1, -1, -1])
        target.append(t)

    while len(target) < target_length:
        target.append(_pad())
    return target


# ---------------------------------------------------------------------------
# Export helpers  (used by regenerate_pkl.py / regenerate_json.py)
# ---------------------------------------------------------------------------

def _write_json_with_idx(data, json_path):
    writer = _StreamingJsonWriter(json_path)
    for sample in data:
        writer.write(sample)
    writer.close()


def export_splits_json(train_data, val_data, test_data,
                       base_name, output_dir, export_test_set=True):
    os.makedirs(output_dir, exist_ok=True)
    if base_name in ("sol", "nosol"):
        _write_json_with_idx(train_data, config[f"dataset_train_{base_name}_json_file"])
        _write_json_with_idx(val_data,   config[f"dataset_val_{base_name}_json_file"])
        if export_test_set:
            _write_json_with_idx(test_data, config[f"dataset_test_{base_name}_json_file"])
        return
    _write_json_with_idx(train_data, os.path.join(output_dir, f"dataset_train_{base_name}.json"))
    _write_json_with_idx(val_data,   os.path.join(output_dir, f"dataset_val_{base_name}.json"))
    if export_test_set:
        _write_json_with_idx(test_data, os.path.join(output_dir, f"dataset_test_{base_name}.json"))


def export_splits_pickle(train_data, val_data, test_data,
                         base_name, output_dir, export_test_set=True):
    """Write PKL splits and, if .pt config keys exist, also write .pt files."""
    os.makedirs(output_dir, exist_ok=True)

    def _write_pkl(data, path):
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Written: {path} ({len(data)} samples)")

    if base_name in ("sol", "nosol"):
        train_pkl = config[f"dataset_train_{base_name}_pkl_file"]
        val_pkl   = config[f"dataset_val_{base_name}_pkl_file"]
        test_pkl  = config[f"dataset_test_{base_name}_pkl_file"]

        _write_pkl(train_data, train_pkl)
        _write_pkl(val_data,   val_pkl)
        if export_test_set:
            _write_pkl(test_data, test_pkl)

        train_pt = config.get(f"dataset_train_{base_name}_pt_file")
        val_pt   = config.get(f"dataset_val_{base_name}_pt_file")
        test_pt  = config.get(f"dataset_test_{base_name}_pt_file")

        if train_pt:
            _save_split_pt(train_data, train_pt)
        if val_pt:
            _save_split_pt(val_data, val_pt)
        if export_test_set and test_pt:
            _save_split_pt(test_data, test_pt)
        return

    _write_pkl(train_data, os.path.join(output_dir, f"dataset_train_{base_name}.pkl"))
    _write_pkl(val_data,   os.path.join(output_dir, f"dataset_val_{base_name}.pkl"))
    if export_test_set:
        _write_pkl(test_data, os.path.join(output_dir, f"dataset_test_{base_name}.pkl"))


# ---------------------------------------------------------------------------
# k-shortest paths  (pure Python, no CuPy dependency)
# ---------------------------------------------------------------------------

def find_k_shortest_paths_on_grid(start, goal, obstacles, k, grid_shape=None):
    if grid_shape is None:
        grid_shape = grid_size
    if k <= 0:
        return []

    rows, cols   = grid_shape
    obstacle_set = set(obstacles)

    def to_idx(cell): return cell[0] * cols + cell[1]
    def h(cell):      return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    if start == goal:         return [[to_idx(start)]]
    if start in obstacle_set: return []
    if goal  in obstacle_set: return []

    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    open_set     = [(h(start), 0, start)]
    g_best       = {start: 0}
    shortest_len = None

    while open_set:
        f, g, cur = heapq.heappop(open_set)
        if cur == goal:
            shortest_len = g + 1
            break
        if g > g_best.get(cur, float("inf")):
            continue
        r, c = cur
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            nxt    = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols): continue
            if nxt in obstacle_set:                      continue
            ng = g + 1
            if ng < g_best.get(nxt, float("inf")):
                g_best[nxt] = ng
                heapq.heappush(open_set, (ng + h(nxt), ng, nxt))

    if shortest_len is None:
        return []

    results       = []
    seen_complete = set()
    enum_heap     = [(h(start), 0, (start,))]

    while enum_heap and len(results) < k:
        f, g, path = heapq.heappop(enum_heap)
        cur        = path[-1]

        if cur == goal:
            if path not in seen_complete:
                seen_complete.add(path)
                results.append([to_idx(c) for c in path])
            continue

        if len(path) >= shortest_len:
            continue

        path_set = set(path)
        r, c     = cur
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            nxt    = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols): continue
            if nxt in obstacle_set or nxt in path_set:  continue
            ng = g + 1
            if ng + h(nxt) > shortest_len - 1:          continue
            heapq.heappush(enum_heap, (ng + h(nxt), ng, path + (nxt,)))

    return results
