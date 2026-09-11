import os
import sys
import json
import pickle
import random
import hashlib
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import dataset_config as config
from dataset_utils import (
    load_templates,
    template_to_obstacles,
    open_obstacles,
    build_nosol_sample,
)


# ---------------------------------------------------------------------------
# Constants — all derived from config, nothing hardcoded
# ---------------------------------------------------------------------------

GRID_SIZE       = config["grid_size"]
PADDING         = config["padding_value"]
FEATURE_SIZE    = config["feature_size"]              # 4
CONTEXT_LEN     = config["context_matrix_length"]     # 100
LATENT_LEN      = config["foundroute_matrix_length"]  # 75  <- must NOT be context_matrix_length
NUM_DIRS        = config.get("num_directions", 4)  # 4 directions: U/D/L/R — NOT num_unique_tokens (which is grid vocab size=100)
STATUS_END      = config["status_end"]
STATUS_OBSTACLE = config["status_obstacle"]

assert LATENT_LEN == config["foundroute_matrix_length"] and LATENT_LEN != CONTEXT_LEN, (
    f"LATENT_LEN={LATENT_LEN} is wrong. Must equal foundroute_matrix_length="
    f"{config['foundroute_matrix_length']} (not context_matrix_length={CONTEXT_LEN}). "
    "Replace this file with the latest version from the output folder."
)

SOL_COUNT       = config["ragate_sol_count"]       # 100
NOSOL_COUNT     = config["ragate_nosol_count"]     # 100
MIN_PATH_LENGTH = config["ragate_sol_min_path_length"]
MIN_DIVERSITY   = config["ragate_sol_min_template_diversity"]
ISOLATED_FRAC   = config["ragate_nosol_isolated_goal_fraction"]
SPLIT_SEED      = config["split_seed"]

SOL_TEMPLATE_START = config["ragate_sol_template_range_start"]
SOL_TEMPLATE_END   = config["ragate_sol_template_range_end"]
DISC_TMPL_START    = config["ragate_nosol_disconnected_template_range_start"]
DISC_TMPL_END      = config["ragate_nosol_disconnected_template_range_end"]

SOL_MIN_DATASET_SIZE   = config["ragate_sol_min_dataset_size"]
NOSOL_MIN_DATASET_SIZE = config["ragate_nosol_min_dataset_size"]

SOL_SCHEDULE = [
    float(x.strip())
    for x in str(config["ragate_sol_schedule_steps"]).split(",")
    if x.strip()
]
DISC_SCHEDULE = [
    float(x.strip())
    for x in str(config["ragate_nosol_disconnected_schedule_steps"]).split(",")
    if x.strip()
]
ISO_SCHEDULE = [
    float(x.strip())
    for x in str(config["ragate_nosol_isolated_schedule_steps"]).split(",")
    if x.strip()
]


# ---------------------------------------------------------------------------
# Custom exception — signals main.py to re-run Stage 1
# ---------------------------------------------------------------------------

class DatasetTooSmallError(RuntimeError):

    def __init__(self, dataset_label, found, required):
        self.dataset_label = dataset_label
        self.found         = found
        self.required      = required
        super().__init__(
            f"{dataset_label} master dataset has {found} qualifying samples "
            f"but {required} are required (ragate_*_min_dataset_size). "
            "Stage 1 must regenerate the dataset before Stage 2 (RAGate) can proceed."
        )


# ---------------------------------------------------------------------------
# Tensor builders
# ---------------------------------------------------------------------------

def _build_input_tensor(context_cells, foundroute_cells):
    """Shape: [CONTEXT_LEN + LATENT_LEN, FEATURE_SIZE] = [175, 4]."""
    pad     = float(PADDING)
    seq_len = CONTEXT_LEN + LATENT_LEN
    t       = torch.full((seq_len, FEATURE_SIZE), pad, dtype=torch.float32)

    for i, cell in enumerate(context_cells[:CONTEXT_LEN]):
        t[i, 0] = float(cell.get("Cell_Number",  PADDING))
        t[i, 1] = float(cell.get("x",            PADDING))
        t[i, 2] = float(cell.get("y",            PADDING))
        t[i, 3] = float(cell.get("State_Status", PADDING))

    for i, cell in enumerate(foundroute_cells[:LATENT_LEN]):
        row       = CONTEXT_LEN + i
        t[row, 0] = float(cell.get("Cell_Number",  PADDING))
        t[row, 1] = float(cell.get("x",            PADDING))
        t[row, 2] = float(cell.get("y",            PADDING))
        t[row, 3] = float(cell.get("State_Status", PADDING))

    return t


def _build_masking_tensor(masking_data):
    pad = float(PADDING)
    t   = torch.full((LATENT_LEN, NUM_DIRS), pad, dtype=torch.float32)
    for i, entry in enumerate(masking_data[:LATENT_LEN]):
        vec  = entry.get("mask_vec", [PADDING] * NUM_DIRS)
        t[i] = torch.tensor(vec[:NUM_DIRS], dtype=torch.float32)
    return t


def _build_target_tensor(target_data):
    pad = float(PADDING)
    t   = torch.full((LATENT_LEN, NUM_DIRS), pad, dtype=torch.float32)
    for i, entry in enumerate(target_data[:LATENT_LEN]):
        vec  = entry.get("target_vec", [PADDING] * NUM_DIRS)
        t[i] = torch.tensor(vec[:NUM_DIRS], dtype=torch.float32)
    return t


def _all_padding_tensor(rows, cols):
    return torch.full((rows, cols), float(PADDING), dtype=torch.float32)


def _sample_to_tensors(sample):
    context_cells    = sample.get("Context",      [])
    foundroute_cells = sample.get("Found_Routes", [])
    masking_data     = sample.get("Masking",      [])
    target_data      = sample.get("Targets",      [])

    inp = _build_input_tensor(context_cells, foundroute_cells)
    msk = (_build_masking_tensor(masking_data) if masking_data
           else _all_padding_tensor(LATENT_LEN, NUM_DIRS))
    tgt = (_build_target_tensor(target_data) if target_data
           else _all_padding_tensor(LATENT_LEN, NUM_DIRS))

    return inp, msk, tgt


# ---------------------------------------------------------------------------
# V3 format checks
# ---------------------------------------------------------------------------

def _is_v3_nosol_sample(sample):
    fr        = sample.get("Found_Routes", None)
    tour_cost = sample.get("TourCost")
    return isinstance(fr, list) and len(fr) == 0 and tour_cost == 0


def _is_sol_sample(sample):
    fr = sample.get("Found_Routes", [])
    return any(
        isinstance(cell, dict)
        and cell.get("Cell_Number", PADDING) != PADDING
        for cell in fr
    )


# ---------------------------------------------------------------------------
# Obstacle-pattern hash  (template-diversity proxy)
# ---------------------------------------------------------------------------

def _obstacle_hash(context_cells):
    obs = tuple(sorted(
        int(c["Cell_Number"])
        for c in context_cells
        if isinstance(c, dict)
        and c.get("Cell_Number", PADDING) != PADDING
        and int(c.get("State_Status", 0)) == STATUS_OBSTACLE
    ))
    return hashlib.md5(str(obs).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved JSON  ({len(data):4d} samples): {path}")


def _save_pkl(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved PKL   ({len(data):4d} samples): {path}")


def _save_pt(samples, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    tensor_list = []
    for s in samples:
        inp, msk, tgt = _sample_to_tensors(s)
        entry = {
            "Context":      s.get("Context",      []),
            "Found_Routes": s.get("Found_Routes", []),
            "Masking":      s.get("Masking",      []),
            "Targets":      s.get("Targets",      []),
            "TourCost":     s.get("TourCost",     0),
            "input_tensor":   inp,
            "masking_tensor": msk,
            "target_tensor":  tgt,
            # ── Metadata ──────────────────────────────────────────────────
            "label":          s.get("label", -1),
        }
        if "split" in s:
            entry["split"] = s["split"]
        tensor_list.append(entry)
    torch.save(tensor_list, path)
    print(f"  Saved .pt   ({len(samples):4d} samples): {path}")


def _load_pkl_streaming_or_legacy(pkl_path):
    samples = []
    with open(pkl_path, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if isinstance(obj, list):
                samples.extend(obj)   # legacy: one big list
            elif isinstance(obj, dict):
                samples.append(obj)   # streaming: one sample per dump
    return samples


def _load_master(pkl_key, json_key, label):
    pkl_path  = config[pkl_key]
    json_path = config[json_key]

    if os.path.exists(pkl_path):
        print(f"  Loading {label} master: {pkl_path}")
        data = _load_pkl_streaming_or_legacy(pkl_path)
        return data, pkl_path

    if os.path.exists(json_path):
        print(f"  Loading {label} master: {json_path}")
        with open(json_path, "r") as f:
            return json.load(f), json_path

    return None, None


# ---------------------------------------------------------------------------
# Reachability helpers
# ---------------------------------------------------------------------------

def _bfs_reachable(start_rc, goal_rc, obs_set, grid_size):
    from collections import deque
    n_rows, n_cols = grid_size
    visited = {start_rc}
    queue   = deque([start_rc])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal_rc:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < n_rows and 0 <= nc < n_cols
                    and (nr, nc) not in obs_set
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def _free_cells(obs_set, grid_size):
    n_rows, n_cols = grid_size
    return [(r, c) for r in range(n_rows) for c in range(n_cols)
            if (r, c) not in obs_set]


# ---------------------------------------------------------------------------
# True route length helper
# ---------------------------------------------------------------------------

def _true_route_length(found_routes):
    return sum(
        1 for cell in found_routes
        if isinstance(cell, dict)
        and cell.get("Cell_Number", PADDING) != PADDING
    )


# ---------------------------------------------------------------------------
# Diversity-aware sampler (shared by sol and nosol extraction)
# ---------------------------------------------------------------------------

def _round_robin(keys, groups):
    pointers  = {k: 0 for k in keys}
    exhausted = set()
    while len(exhausted) < len(keys):
        for k in keys:
            if k in exhausted:
                continue
            idx = pointers[k]
            if idx >= len(groups[k]):
                exhausted.add(k)
                continue
            yield groups[k][idx]
            pointers[k] += 1


def _diversity_sample(candidates, count, rng, hash_fn):
    groups = defaultdict(list)
    for s in candidates:
        groups[hash_fn(s)].append(s)

    for g in groups.values():
        rng.shuffle(g)

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    selected = []
    for s in _round_robin(group_keys, groups):
        selected.append(s)
        if len(selected) >= count:
            break

    return selected


# ---------------------------------------------------------------------------
# Sol extraction
# ---------------------------------------------------------------------------

def extract_sol_samples():
    print("\n=== Sol samples ===")

    raw, source = _load_master(
        "sol_dataset_pkl_file", "sol_dataset_json_file", "sol"
    )

    if raw is None:
        raise DatasetTooSmallError("sol", 0, SOL_MIN_DATASET_SIZE)

    print(f"  Master dataset size: {len(raw):,}")

    if len(raw) < SOL_MIN_DATASET_SIZE:
        raise DatasetTooSmallError("sol", len(raw), SOL_MIN_DATASET_SIZE)

    print(f"  Size {len(raw):,} >= threshold {SOL_MIN_DATASET_SIZE} — extracting.")

    # Build blacklist of obstacle hashes for disconnected templates (1–35).
    templates = load_templates()
    rng       = random.Random(SPLIT_SEED)

    blacklist_hashes = set()
    n_cols = GRID_SIZE[1]
    for tid in range(1, SOL_TEMPLATE_START):
        key = str(tid)
        if key not in templates:
            continue
        grid = templates[key]
        obs  = tuple(sorted(
            r * n_cols + c + 1
            for r, row in enumerate(grid)
            for c, v in enumerate(row)
            if int(v) == 1
        ))
        blacklist_hashes.add(hashlib.md5(str(obs).encode()).hexdigest()[:12])

    filtered = [
        s for s in raw
        if _is_sol_sample(s)
        and _true_route_length(s.get("Found_Routes", [])) >= MIN_PATH_LENGTH
        and _obstacle_hash(s.get("Context", [])) not in blacklist_hashes
    ]

    print(f"  After filters (sol, path≥{MIN_PATH_LENGTH}, connected template): "
          f"{len(filtered):,}")

    if len(filtered) < SOL_COUNT:
        raise RuntimeError(
            f"Only {len(filtered)} samples pass filters, need {SOL_COUNT}. "
            "Regenerate the dataset with more samples."
        )

    selected = _diversity_sample(
        filtered, SOL_COUNT, rng,
        hash_fn=lambda s: _obstacle_hash(s.get("Context", [])),
    )

    if len(selected) < SOL_COUNT:
        raise RuntimeError(
            f"Could only select {len(selected)} diverse sol samples; "
            f"{SOL_COUNT} required."
        )

    diversity = len({_obstacle_hash(s.get("Context", [])) for s in selected})
    print(f"  Diversity groups: {diversity} (requirement ≥{MIN_DIVERSITY})")
    if diversity < MIN_DIVERSITY:
        print(f"  WARNING: diversity {diversity} < {MIN_DIVERSITY}.")

    for s in selected:
        s["label"] = 1

    print(f"  Sol samples selected: {len(selected)}")
    return selected


# ---------------------------------------------------------------------------
# Nosol extraction
# ---------------------------------------------------------------------------

def extract_nosol_samples(raw, rng):
    print("  Qualifying count meets threshold — extracting from master.")

    v3_samples = [s for s in raw if _is_v3_nosol_sample(s)]
    v2_count   = len(raw) - len(v3_samples)

    if v2_count > 0:
        print(f"  V2-format nosol samples skipped: {v2_count:,}")
    print(f"  V3-format nosol samples available: {len(v3_samples):,}")

    selected = _diversity_sample(
        v3_samples, NOSOL_COUNT, rng,
        hash_fn=lambda s: _obstacle_hash(s.get("Context", [])),
    )

    for s in selected:
        s["label"] = 0

    print(f"  Nosol samples extracted: {len(selected)}")
    return selected


# ---------------------------------------------------------------------------
# Nosol generation — disconnected-component
# ---------------------------------------------------------------------------

def _generate_disconnected_nosol(n_needed, templates, rng, experiment_offset):
    disc_tids = [
        str(tid) for tid in range(DISC_TMPL_START, DISC_TMPL_END + 1)
        if str(tid) in templates
    ]
    if not disc_tids:
        raise RuntimeError(
            f"No templates in disconnected range {DISC_TMPL_START}–{DISC_TMPL_END}."
        )

    samples      = []
    attempts     = 0
    max_attempts = max(5000, n_needed * 30)
    min_dist     = config.get("min_start_goal_distance", 4)

    while len(samples) < n_needed and attempts < max_attempts:
        attempts += 1
        tid      = rng.choice(disc_tids)
        grid     = templates[tid]
        open_pct = rng.choice(DISC_SCHEDULE)
        obs_list = open_obstacles(template_to_obstacles(grid), open_pct, rng)
        obs_set  = set(obs_list)

        free = _free_cells(obs_set, GRID_SIZE)
        if len(free) < 2:
            continue

        rng.shuffle(free)
        start_rc = goal_rc = None

        for i in range(len(free)):
            for j in range(i + 1, min(i + 40, len(free))):
                a, b = free[i], free[j]
                if (abs(a[0] - b[0]) + abs(a[1] - b[1])) < min_dist:
                    continue
                if not _bfs_reachable(a, b, obs_set, GRID_SIZE):
                    start_rc, goal_rc = a, b
                    break
            if start_rc is not None:
                break

        if start_rc is None:
            continue

        s = build_nosol_sample(start_rc, goal_rc, obs_list,
                               experiment_offset + len(samples))
        s["label"] = 0
        samples.append(s)

    if len(samples) < n_needed:
        print(f"  WARNING: disconnected-component produced "
              f"{len(samples)}/{n_needed} after {max_attempts} attempts.")
    return samples


# ---------------------------------------------------------------------------
# Nosol generation — isolated-goal
# ---------------------------------------------------------------------------

def _generate_isolated_goal_nosol(n_needed, templates, rng, experiment_offset):
    all_tids     = [str(tid) for tid in range(1, 151) if str(tid) in templates]
    n_rows, n_cols = GRID_SIZE
    samples      = []
    attempts     = 0
    max_attempts = max(5000, n_needed * 80)
    min_dist     = config.get("min_start_goal_distance", 4)

    while len(samples) < n_needed and attempts < max_attempts:
        attempts += 1
        tid      = rng.choice(all_tids)
        grid     = templates[tid]
        open_pct = rng.choice(ISO_SCHEDULE)
        obs_list = open_obstacles(template_to_obstacles(grid), open_pct, rng)
        obs_set  = set(obs_list)

        free = _free_cells(obs_set, GRID_SIZE)
        if len(free) < 2:
            continue

        candidates = []
        for cell_rc in free:
            r, c = cell_rc
            free_neighbours = [
                (r + dr, c + dc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= r + dr < n_rows
                and 0 <= c + dc < n_cols
                and (r + dr, c + dc) not in obs_set
            ]
            candidates.append((len(free_neighbours), cell_rc, free_neighbours))

        candidates.sort()
        if not candidates:
            continue

        _, goal_rc, extra_to_block = candidates[0]
        new_obs_set  = (obs_set | set(extra_to_block)) - {goal_rc}
        new_obs_list = list(new_obs_set)

        remaining_free = [
            rc for rc in free
            if rc not in new_obs_set and rc != goal_rc
        ]
        if not remaining_free:
            continue

        remaining_free.sort(
            key=lambda rc: -(abs(rc[0] - goal_rc[0]) + abs(rc[1] - goal_rc[1]))
        )
        start_rc = remaining_free[0]

        if (abs(start_rc[0] - goal_rc[0]) + abs(start_rc[1] - goal_rc[1])) < min_dist:
            continue
        if _bfs_reachable(start_rc, goal_rc, new_obs_set, GRID_SIZE):
            continue

        s = build_nosol_sample(start_rc, goal_rc, new_obs_list,
                               experiment_offset + len(samples))
        s["label"] = 0
        samples.append(s)

    if len(samples) < n_needed:
        print(f"  WARNING: isolated-goal produced "
              f"{len(samples)}/{n_needed} after {max_attempts} attempts.")
    return samples


# ---------------------------------------------------------------------------
# Nosol top-level orchestrator (extract or raise for Stage 1 regeneration)
# ---------------------------------------------------------------------------

def generate_nosol_samples():
    print("\n=== Nosol samples ===")

    rng = random.Random(SPLIT_SEED + 1)

    raw, source = _load_master(
        "nosol_dataset_pkl_file", "nosol_dataset_json_file", "nosol"
    )

    if raw is None:
        qualifying_count = 0
    else:
        qualifying_count = sum(1 for s in raw if _is_v3_nosol_sample(s))
        v2_count         = len(raw) - qualifying_count
        print(f"  Master dataset size: {len(raw):,}  "
              f"(V3-format: {qualifying_count:,}, V2-format skipped: {v2_count:,})")

    print(f"  V3 qualifying count: {qualifying_count:,}  "
          f"(threshold: {NOSOL_MIN_DATASET_SIZE})")

    if qualifying_count < NOSOL_MIN_DATASET_SIZE:
        raise DatasetTooSmallError("nosol", qualifying_count, NOSOL_MIN_DATASET_SIZE)

    return extract_nosol_samples(raw, rng)


# ---------------------------------------------------------------------------
# Split and combine
# ---------------------------------------------------------------------------

def split_and_combine(sol_samples, nosol_samples):
    """
    Split sol and nosol independently at 80/10/10, then interleave so that
    the 1:1 ratio is preserved in every split. Attaches 'split' field.
    """
    print("\n=== Combining and splitting ===")

    rng = random.Random(SPLIT_SEED)

    def _split_class(samples):
        s     = samples[:]
        rng.shuffle(s)
        n     = len(s)
        n_tr  = round(n * config["train_ratio"])
        n_val = round(n * config["val_ratio"])
        return s[:n_tr], s[n_tr:n_tr + n_val], s[n_tr + n_val:]

    sol_tr,   sol_val,   sol_te   = _split_class(sol_samples)
    nosol_tr, nosol_val, nosol_te = _split_class(nosol_samples)

    def _merge(a, b, split_name):
        merged = a + b
        for s in merged:
            s["split"] = split_name
        rng.shuffle(merged)
        return merged

    train = _merge(sol_tr,  nosol_tr,  "train")
    val   = _merge(sol_val, nosol_val, "val")
    test  = _merge(sol_te,  nosol_te,  "test")

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        n_sol   = sum(1 for s in split_data if s["label"] == 1)
        n_nosol = sum(1 for s in split_data if s["label"] == 0)
        print(f"  {split_name:5s}: {len(split_data):3d} samples  "
              f"({n_sol} sol, {n_nosol} nosol)")

    return train, val, test


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _per_class_files_complete(class_name):
    expected = int(config[f"ragate_{class_name}_count"])
    for key in (f"ragate_{class_name}_json",
                f"ragate_{class_name}_pkl",
                f"ragate_{class_name}_pt"):
        if not os.path.exists(config[key]):
            return False
    # Check sample count in the PKL to guard against partial writes
    actual = _count_pkl(config[f"ragate_{class_name}_pkl"])
    return actual >= expected


def _count_pkl(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def main():
    print("=" * 70)
    print("  generate_ragate.py  —  RAGate Dataset Generation")
    print("=" * 70)

    # ── Sol ──────────────────────────────────────────────────────────────────
    if _per_class_files_complete("sol"):
        print("\n=== Sol samples ===")
        print("  ragate_sol files exist with correct count — loading from PKL.")
        with open(config["ragate_sol_pkl"], "rb") as f:
            sol_samples = pickle.load(f)
        print(f"  Sol samples loaded: {len(sol_samples):,}")
    else:
        sol_samples = extract_sol_samples()
        _save_json(sol_samples, config["ragate_sol_json"])
        _save_pkl (sol_samples, config["ragate_sol_pkl"])
        _save_pt  (sol_samples, config["ragate_sol_pt"])

    # ── Nosol ─────────────────────────────────────────────────────────────────
    if _per_class_files_complete("nosol"):
        print("\n=== Nosol samples ===")
        print("  ragate_nosol files exist with correct count — loading from PKL.")
        with open(config["ragate_nosol_pkl"], "rb") as f:
            nosol_samples = pickle.load(f)
        print(f"  Nosol samples loaded: {len(nosol_samples):,}")
    else:
        nosol_samples = generate_nosol_samples()
        _save_json(nosol_samples, config["ragate_nosol_json"])
        _save_pkl (nosol_samples, config["ragate_nosol_pkl"])
        _save_pt  (nosol_samples, config["ragate_nosol_pt"])

    # ── Splits ────────────────────────────────────────────────────────────────
    train, val, test = split_and_combine(sol_samples, nosol_samples)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        _save_json(split_data, config[f"ragate_{split_name}_json"])
        _save_pkl (split_data, config[f"ragate_{split_name}_pkl"])
        _save_pt  (split_data, config[f"ragate_{split_name}_pt"])

    print("\n=== Done. All RAGate dataset files written. ===")


if __name__ == "__main__":
    main()
