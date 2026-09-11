import os
import pickle
import random

import numpy as np

try:
    from PIL import Image
except ImportError:                                   # pragma: no cover
    Image = None

_COLORS = {
    "free":     (255, 255, 255),
    "obstacle": (110, 110, 110),
    "start":    (0, 200, 0),
    "goal":     (220, 0, 0),
    "path":     (40, 90, 230),
}


def _iter_pkl(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def _load_samples(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    for obj in _iter_pkl(path):
        if isinstance(obj, list):
            out.extend(obj)
        else:
            out.append(obj)
    return out


def _split_indices(n, train_ratio, val_ratio, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    return {
        "train": idx[:n_tr],
        "val":   idx[n_tr:n_tr + n_va],
        "test":  idx[n_tr + n_va:],
    }


def _sample_to_rgb(sample, n_rows, n_cols, status):
    grid = np.empty((n_rows, n_cols, 3), dtype=np.uint8)
    grid[:] = _COLORS["free"]

    def rc(cell_num):
        i = int(cell_num) - 1
        return i // n_cols, i % n_cols

    # obstacles
    for cell in sample.get("Context", []):
        cn = cell.get("Cell_Number")
        if cn is None:
            continue
        r, c = rc(cn)
        if 0 <= r < n_rows and 0 <= c < n_cols and cell.get("State_Status") == status["obstacle"]:
            grid[r, c] = _COLORS["obstacle"]

    # path (sol only; nosol has an empty Found_Routes)
    for cell in sample.get("Found_Routes", []):
        cn = cell.get("Cell_Number")
        if cn is None:
            continue
        r, c = rc(cn)
        if 0 <= r < n_rows and 0 <= c < n_cols:
            grid[r, c] = _COLORS["path"]

    # start / goal drawn last so they sit on top of the path
    for cell in sample.get("Context", []):
        cn = cell.get("Cell_Number")
        if cn is None:
            continue
        r, c = rc(cn)
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            continue
        st = cell.get("State_Status")
        if st == status["start"]:
            grid[r, c] = _COLORS["start"]
        elif st == status["end"]:
            grid[r, c] = _COLORS["goal"]

    return grid


def _save(rgb, size, mode, out_path):
    img = Image.fromarray(rgb, mode="RGB").resize((size, size), Image.NEAREST)
    if mode == "gray":
        img = img.convert("L")
    img.save(out_path, quality=95)


def render_master_images(config):
    if Image is None:
        print("[IMAGES] Pillow not installed (`pip install pillow`) — skipping.")
        return

    n_rows, n_cols = config["grid_size"]
    status = {
        "start":    int(config["status_start"]),
        "end":      int(config["status_end"]),
        "obstacle": int(config["status_obstacle"]),
        "free":     int(config["status_free"]),
    }
    sizes = [int(s) for s in config.get("image_sizes", [64, 128, 224])]
    modes = [str(m) for m in config.get("image_modes", ["color", "gray"])]
    base  = config.get("images_dir", "Datasets/Images")
    fmt   = str(config.get("image_format", "jpg"))
    tr    = float(config.get("train_ratio", 0.80))
    va    = float(config.get("val_ratio", 0.10))
    seed  = int(config.get("split_seed", 42))

    classes = {
        "sol":   config.get("sol_dataset_pkl_file"),
        "nosol": config.get("nosol_dataset_pkl_file"),
    }

    print(f"[IMAGES] Rendering ImageNet-style JPEGs -> {base}  "
          f"(sizes={sizes}, modes={modes})")

    for cls, pkl_path in classes.items():
        samples = _load_samples(pkl_path)
        if not samples:
            print(f"[IMAGES] {cls}: no master samples at {pkl_path!r} — skipping.")
            continue
        splits = _split_indices(len(samples), tr, va, seed)
        print(f"[IMAGES] {cls}: {len(samples):,} samples "
              f"(train={len(splits['train'])}, val={len(splits['val'])}, "
              f"test={len(splits['test'])})")

        # pre-create the output directories once
        for mode in modes:
            for size in sizes:
                for split in splits:
                    os.makedirs(os.path.join(base, mode, str(size), split, cls),
                                exist_ok=True)

        for split, idxs in splits.items():
            for i in idxs:
                rgb = _sample_to_rgb(samples[i], n_rows, n_cols, status)
                for mode in modes:
                    for size in sizes:
                        out_path = os.path.join(
                            base, mode, str(size), split, cls, f"{i:07d}.{fmt}"
                        )
                        _save(rgb, size, mode, out_path)

    print("[IMAGES] Done.")
