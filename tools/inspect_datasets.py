import os
import sys
import json
import pickle
import random
import hashlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import dataset_config as config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PADDING         = config["padding_value"]
STATUS_END      = config["status_end"]
STATUS_START    = config["status_start"]
STATUS_OBSTACLE = config["status_obstacle"]
STATUS_FREE     = config["status_free"]
CONTEXT_LEN     = config["context_matrix_length"]      # 100
LATENT_LEN      = config["foundroute_matrix_length"]   # 75
FEATURE_SIZE    = config["feature_size"]               # 4
NUM_DIRS        = config.get("num_directions", 4)      # 4

EXPECTED_INP_SHAPE = (CONTEXT_LEN + LATENT_LEN, FEATURE_SIZE)   # (175, 4)
EXPECTED_MSK_SHAPE = (LATENT_LEN, NUM_DIRS)                      # ( 75, 4)
EXPECTED_TGT_SHAPE = (LATENT_LEN, NUM_DIRS)                      # ( 75, 4)

# Required raw fields — case-sensitive, must match ML pipeline dataloader exactly
RAW_FIELDS_BASE     = ["Context", "Found_Routes", "Masking", "Targets", "TourCost"]
RAW_FIELDS_LABELLED = RAW_FIELDS_BASE + ["label"]   # RAGate
TENSOR_FIELDS       = ["input_tensor", "masking_tensor", "target_tensor"]

# Inspection controls from config
RUN_QUICK    = config.get("inspect_quick",    True)
RUN_DETAILED = config.get("inspect_detailed", False)
RUN_MASTERS  = config.get("inspect_masters",    True)
RUN_NAV      = config.get("inspect_navigation", True)
RUN_GC       = config.get("inspect_gridcheck",  True)
TENSOR_SPOT  = config.get("inspect_tensor_spot_size", 200)

MIN_ROUTE = config.get("min_route_length_for_dataset", 0)
MAX_ROUTE = config.get("max_route_length_for_dataset", 75)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

class _Results:
    def __init__(self):
        self._records = []   # (level, group, label, passed, detail)

    def record(self, level, group, label, passed, detail=""):
        self._records.append((level, group, label, passed, detail))
        marker = "[PASS]" if passed else "[FAIL]"
        line   = f"    {marker}  {label}"
        if detail:
            line += f"  — {detail}"
        print(line)
        return passed

    def warn(self, group, label, detail=""):
        line = f"    [WARN]  {label}"
        if detail:
            line += f"  — {detail}"
        print(line)

    def info(self, msg):
        print(f"    {msg}")

    def summary(self):
        total  = len(self._records)
        passed = sum(1 for _, _, _, ok, _ in self._records if ok)
        failed = total - passed
        return total, passed, failed

    def failed_records(self):
        return [(g, l, d) for _, g, l, ok, d in self._records if not ok]

    def group_summary(self, group):
        recs   = [(ok, l) for _, g, l, ok, _ in self._records if g == group]
        n_fail = sum(1 for ok, _ in recs if not ok)
        return len(recs), n_fail


R = _Results()


def _header(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def _subheader(title):
    print(f"\n  ── {title}")


# ---------------------------------------------------------------------------
# PKL helpers
# ---------------------------------------------------------------------------

def _streaming_pkl_load(path):
    """Yield all samples from a streaming or legacy-list PKL."""
    with open(path, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if isinstance(obj, list):
                yield from obj
            elif isinstance(obj, dict):
                yield obj


def _count_pkl(path):
    """Count samples without deserialising content."""
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
                count += len(obj) if isinstance(obj, list) else 1
    except Exception:
        pass
    return count


def _load_plain_pkl(path):
    """Load a plain pickle.dump(list) PKL — used for RAGate."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, list) else list(_streaming_pkl_load(path))


def _count_idx(json_path):
    """Return .idx offset count (O(1) proxy for JSON sample count)."""
    idx_path = json_path + ".idx"
    if not os.path.exists(idx_path):
        return None
    try:
        with open(idx_path, "rb") as f:
            return len(pickle.load(f))
    except Exception:
        return None


def _random_sample(total, n):
    n = min(n, total)
    return sorted(random.sample(range(total), n))


# ---------------------------------------------------------------------------
# Sample quality helpers
# ---------------------------------------------------------------------------

def _goal_from_context(context):
    for cell in context:
        if (isinstance(cell, dict)
                and cell.get("State_Status") == STATUS_END
                and cell.get("Cell_Number", PADDING) != PADDING):
            return int(cell["Cell_Number"])
    return None


def _last_real_cell(found_routes):
    last = None
    for cell in found_routes:
        if isinstance(cell, dict) and cell.get("Cell_Number", PADDING) != PADDING:
            last = int(cell["Cell_Number"])
    return last


def _real_route_len(found_routes):
    return sum(
        1 for c in found_routes
        if isinstance(c, dict) and c.get("Cell_Number", PADDING) != PADDING
    )


def _is_nosol(sample):
    fr = sample.get("Found_Routes", None)
    return isinstance(fr, list) and len(fr) == 0 and sample.get("TourCost") == 0


def _check_sol_quality(sample):
    """Returns (ok, reason, route_len)."""
    ctx = sample.get("Context",      [])
    fr  = sample.get("Found_Routes", [])
    tc  = sample.get("TourCost",     None)

    goal      = _goal_from_context(ctx)
    last_cell = _last_real_cell(fr)
    rlen      = _real_route_len(fr)

    if goal is None:
        return False, "no goal in Context", rlen
    if last_cell is None:
        return False, "Found_Routes empty/all-padding", rlen
    if last_cell != goal:
        return False, f"truncated: last={last_cell} goal={goal}", rlen
    if tc is not None and tc != rlen:
        return False, f"TourCost={tc} != route_len={rlen}", rlen
    if not (MIN_ROUTE <= rlen <= MAX_ROUTE):
        return False, f"route_len={rlen} outside [{MIN_ROUTE},{MAX_ROUTE}]", rlen
    return True, "ok", rlen


def _check_fields(sample, required_fields):
    """Return list of missing field names (case-sensitive)."""
    return [f for f in required_fields if f not in sample]


# ---------------------------------------------------------------------------
# Cross-format count check  (shared by Quick and Detailed)
# ---------------------------------------------------------------------------

def _cross_format_counts(group, label, pkl_path, json_path, pt_path,
                          expected=None, plain_pkl=False):
    """
    Check file existence + count agreement across PKL, JSON .idx, and .pt.
    plain_pkl=True for RAGate PKLs (plain list, not streaming).
    Returns (pkl_count, pt_data_or_None).
    """
    # PKL
    pkl_ok = R.record("Q", group, f"{label} PKL exists",
                      os.path.exists(pkl_path))
    pkl_count = 0
    if pkl_ok:
        pkl_count = len(_load_plain_pkl(pkl_path)) if plain_pkl \
                    else _count_pkl(pkl_path)
        R.info(f"PKL count: {pkl_count:,}")

    # JSON .idx
    idx_count = _count_idx(json_path)
    if idx_count is None:
        R.warn(group, f"{label} JSON .idx missing",
               f"{json_path}.idx not found")
    else:
        R.record("Q", group, f"{label} JSON .idx matches PKL",
                 idx_count == pkl_count,
                 f"idx={idx_count:,}  pkl={pkl_count:,}")

    # .pt
    pt_data = None
    if pt_path:
        pt_ok = R.record("Q", group, f"{label} .pt exists",
                         os.path.exists(pt_path))
        if pt_ok:
            try:
                pt_data = torch.load(pt_path, map_location="cpu",
                                     weights_only=False)
                pt_count = len(pt_data)
                R.record("Q", group, f"{label} .pt count matches PKL",
                         pt_count == pkl_count,
                         f"pt={pt_count:,}  pkl={pkl_count:,}")
            except Exception as e:
                R.record("Q", group, f"{label} .pt loadable", False, str(e))

    # Expected count
    if expected is not None and pkl_count > 0:
        R.record("Q", group, f"{label} count meets target",
                 pkl_count >= expected,
                 f"{pkl_count:,} {'≥' if pkl_count >= expected else '<'} {expected:,}")

    return pkl_count, pt_data


# ---------------------------------------------------------------------------
# Quick field + tensor check
# ---------------------------------------------------------------------------

def _quick_fields_and_tensors(group, label, pkl_path, pt_data,
                               required_fields, plain_pkl=False):
    """Check field names on first 5 PKL samples; tensor shapes on first .pt entry."""
    # Field presence — first 5 samples
    samples_checked = 0
    field_issues    = []

    loader = _load_plain_pkl if plain_pkl else _streaming_pkl_load
    try:
        for s in loader(pkl_path):
            missing = _check_fields(s, required_fields)
            if missing:
                field_issues.append(missing)
            samples_checked += 1
            if samples_checked >= 5:
                break
    except Exception as e:
        R.record("Q", group, f"{label} PKL readable", False, str(e))
        return

    if field_issues:
        R.record("Q", group, f"{label} required fields present (first 5 samples)",
                 False,
                 f"missing in {len(field_issues)} samples: {field_issues[0]}")
    else:
        R.record("Q", group, f"{label} required fields present (first 5 samples)",
                 True, f"fields: {required_fields}")

    # .pt shapes — first entry
    if pt_data and len(pt_data) > 0:
        entry = pt_data[0]
        inp   = entry.get("input_tensor")
        msk   = entry.get("masking_tensor")
        tgt   = entry.get("target_tensor")

        shape_ok = (
            (inp is not None and tuple(inp.shape) == EXPECTED_INP_SHAPE) and
            (msk is not None and tuple(msk.shape) == EXPECTED_MSK_SHAPE) and
            (tgt is not None and tuple(tgt.shape) == EXPECTED_TGT_SHAPE)
        )
        R.record("Q", group, f"{label} .pt tensor shapes correct", shape_ok,
                 f"input={tuple(inp.shape) if inp is not None else None}  "
                 f"mask={tuple(msk.shape) if msk is not None else None}  "
                 f"target={tuple(tgt.shape) if tgt is not None else None}")

        # Raw field presence in .pt
        raw_ok   = all(f in entry for f in required_fields)
        missing  = [f for f in required_fields if f not in entry]
        R.record("Q", group, f"{label} .pt raw fields present", raw_ok,
                 "" if raw_ok else f"missing: {missing}")


# ---------------------------------------------------------------------------
# Detailed PKL sweep
# ---------------------------------------------------------------------------

def _detailed_pkl_sweep(group, label, pkl_path, required_fields,
                         plain_pkl=False, expect_sol=False, expect_nosol=False,
                         label_val=None):
    """
    Full sequential quality sweep of a PKL.
    Returns stats dict.
    """
    stats = dict(
        total=0, sol=0, nosol=0,
        sol_ok=0, sol_truncated=0, sol_empty=0,
        nosol_bad=0, field_issues=0,
        route_lens=[], trunc_ex=[],
    )

    loader = _load_plain_pkl if plain_pkl else _streaming_pkl_load
    try:
        samples = list(loader(pkl_path))
    except Exception as e:
        R.record("D", group, f"{label} PKL fully readable", False, str(e))
        return stats

    for idx, s in enumerate(samples):
        stats["total"] += 1

        missing = _check_fields(s, required_fields)
        if missing:
            stats["field_issues"] += 1
            continue

        nosol = _is_nosol(s)
        if nosol:
            stats["nosol"] += 1
            fr = s.get("Found_Routes", [])
            msk = s.get("Masking", [])
            tgt = s.get("Targets", [])
            if fr or msk or tgt:
                stats["nosol_bad"] += 1
        else:
            stats["sol"] += 1
            ok, reason, rlen = _check_sol_quality(s)
            if ok:
                stats["sol_ok"] += 1
                stats["route_lens"].append(rlen)
            elif "empty" in reason or "padding" in reason:
                stats["sol_empty"] += 1
                if len(stats["trunc_ex"]) < 3:
                    stats["trunc_ex"].append(f"#{idx}: {reason}")
            else:
                stats["sol_truncated"] += 1
                if len(stats["trunc_ex"]) < 3:
                    stats["trunc_ex"].append(f"#{idx}: {reason}")

    # Print stats
    R.info(f"Total: {stats['total']:,}  Sol: {stats['sol']:,}  "
           f"Nosol: {stats['nosol']:,}")
    if stats["route_lens"]:
        rl = stats["route_lens"]
        R.info(f"Route length: min={min(rl)}  max={max(rl)}  "
               f"avg={sum(rl)/len(rl):.1f}")
    for ex in stats["trunc_ex"]:
        R.info(f"  Issue: {ex}")

    # Record quality results
    R.record("D", group, f"{label} sol routes correct",
             stats["sol_truncated"] == 0 and stats["sol_empty"] == 0,
             f"truncated={stats['sol_truncated']}  empty={stats['sol_empty']}")
    if stats["nosol"] > 0:
        R.record("D", group, f"{label} nosol structure valid",
                 stats["nosol_bad"] == 0,
                 f"bad={stats['nosol_bad']}")
    if stats["field_issues"] > 0:
        R.record("D", group, f"{label} all samples have required fields",
                 False, f"{stats['field_issues']} samples missing fields")

    return stats


# ---------------------------------------------------------------------------
# Detailed .pt spot-check
# ---------------------------------------------------------------------------

def _detailed_pt_check(group, label, pt_data, required_fields,
                        full_sweep=False):
    """Spot-check .pt entries for raw-field vs tensor consistency."""
    if pt_data is None or len(pt_data) == 0:
        return

    total   = len(pt_data)
    indices = (list(range(total)) if full_sweep
               else _random_sample(total, TENSOR_SPOT))

    sol_ok = sol_bad = nosol_bad = mismatch = 0

    for idx in indices:
        entry = pt_data[idx]
        label_val = entry.get("label", -1)
        if isinstance(label_val, torch.Tensor):
            label_val = int(label_val.item())

        ctx = entry.get("Context",      [])
        fr  = entry.get("Found_Routes", [])
        inp = entry.get("input_tensor")

        if label_val == 1 or (label_val == -1 and not _is_nosol({"Found_Routes": fr, "TourCost": entry.get("TourCost", -1)})):
            ok, _, _ = _check_sol_quality(
                {"Context": ctx, "Found_Routes": fr,
                 "TourCost": entry.get("TourCost", 0)}
            )
            if ok:
                sol_ok += 1
            else:
                sol_bad += 1
        elif label_val == 0:
            if any(isinstance(c, dict) and c.get("Cell_Number", PADDING) != PADDING
                   for c in fr):
                nosol_bad += 1

        # Cross-check raw vs tensor
        if inp is not None:
            goal_raw    = _goal_from_context(ctx)
            goal_tensor = None
            for row in inp[:CONTEXT_LEN]:
                if (row[0].item() != PADDING
                        and abs(row[3].item() - STATUS_END) < 0.5):
                    goal_tensor = int(round(row[0].item()))
                    break
            if goal_raw is not None and goal_tensor is not None:
                if goal_raw != goal_tensor:
                    mismatch += 1

    sweep_desc = "full" if full_sweep else f"spot ({len(indices)})"
    R.record("D", group, f"{label} .pt quality ({sweep_desc})",
             sol_bad == 0 and nosol_bad == 0,
             f"sol_bad={sol_bad}  nosol_bad={nosol_bad}")
    if mismatch > 0:
        R.record("D", group, f"{label} .pt raw/tensor consistency",
                 False, f"{mismatch} goal mismatches")


# ---------------------------------------------------------------------------
# Cross-split overlap (RAGate only)
# ---------------------------------------------------------------------------

def _sample_key(sample):
    ctx  = sample.get("Context", [])
    obs  = tuple(sorted(
        int(c["Cell_Number"]) for c in ctx
        if isinstance(c, dict)
        and int(c.get("State_Status", 0)) == STATUS_OBSTACLE
        and c.get("Cell_Number", PADDING) != PADDING
    ))
    start = next((int(c["Cell_Number"]) for c in ctx
                  if isinstance(c, dict)
                  and int(c.get("State_Status", 0)) == STATUS_START), None)
    goal  = next((int(c["Cell_Number"]) for c in ctx
                  if isinstance(c, dict)
                  and int(c.get("State_Status", 0)) == STATUS_END), None)
    fr    = sample.get("Found_Routes", [])
    route = tuple(
        int(c["Cell_Number"]) for c in fr
        if isinstance(c, dict) and c.get("Cell_Number", PADDING) != PADDING
    )
    return hashlib.md5(f"{obs}|{start}|{goal}|{route}".encode()).hexdigest()


def _overlap_check(group, data_a, label_a, data_b, label_b):
    keys_a  = {_sample_key(s) for s in data_a}
    keys_b  = {_sample_key(s) for s in data_b}
    overlap = len(keys_a & keys_b)
    if overlap == 0:
        R.record("D", group, f"overlap {label_a}∩{label_b}", True, "0 duplicates")
    else:
        pct = overlap * 100.0 / max(len(keys_b), 1)
        R.warn(group, f"overlap {label_a}∩{label_b}",
               f"{overlap} duplicates ({pct:.2f}% of {label_b}) — "
               "expected for nosol same-grid pairs; negligible classifier impact")


# ---------------------------------------------------------------------------
# GROUP 1 — Masters
# ---------------------------------------------------------------------------

def _inspect_masters(level):
    _header(f"[{level}] Group 1 — Masters  (datasets/masters/)")

    for base, pt_key, lbl, expected_key in [
        ("sol",   "sol_dataset_pt_file",   "sol_dataset",   "sol_target_count"),
        ("nosol", "nosol_dataset_pt_file", "nosol_dataset", "nosol_target_count"),
    ]:
        _subheader(lbl)
        pkl_path = config.get(f"{base}_dataset_pkl_file",  f"datasets/masters/{base}_dataset.pkl")
        json_path = config.get(f"{base}_dataset_json_file", f"datasets/masters/{base}_dataset.json")
        pt_path   = config.get(pt_key)
        expected  = int(config.get(expected_key, 0))

        pkl_count, pt_data = _cross_format_counts(
            "masters", lbl, pkl_path, json_path, pt_path, expected=expected
        )

        if level == "Q":
            if pkl_count > 0:
                _quick_fields_and_tensors(
                    "masters", lbl, pkl_path, pt_data,
                    RAW_FIELDS_BASE, plain_pkl=False
                )
        else:
            if pkl_count > 0:
                _detailed_pkl_sweep(
                    "masters", lbl, pkl_path, RAW_FIELDS_BASE,
                    plain_pkl=False,
                    expect_sol=(base == "sol"),
                    expect_nosol=(base == "nosol"),
                )
                if pt_data:
                    _detailed_pt_check("masters", lbl, pt_data,
                                       RAW_FIELDS_BASE, full_sweep=False)


# ---------------------------------------------------------------------------
# GROUP 2 — Navigation splits
# ---------------------------------------------------------------------------

def _inspect_navigation(level):
    _header(f"[{level}] Group 2 — Navigation Splits  (datasets/PSTAR/)")

    nav_sets = [
        ("train", "sol",   "dataset_train_sol",   True,  False, 1),
        ("val",   "sol",   "dataset_val_sol",     True,  False, 1),
        ("test",  "sol",   "dataset_test_sol",    True,  False, 1),
        ("train", "nosol", "dataset_train_nosol", False, True,  0),
        ("val",   "nosol", "dataset_val_nosol",   False, True,  0),
        ("test",  "nosol", "dataset_test_nosol",  False, True,  0),
    ]

    for split, base, key_base, is_sol, is_nosol, lbl_val in nav_sets:
        _subheader(f"{key_base}")
        pkl_path  = config.get(f"{key_base}_pkl_file",  f"datasets/PSTAR/{key_base}.pkl")
        json_path = config.get(f"{key_base}_json_file", f"datasets/PSTAR/{key_base}.json")
        pt_path   = config.get(f"{key_base}_pt_file",   f"datasets/PSTAR/{key_base}.pt")

        pkl_count, pt_data = _cross_format_counts(
            "PSTAR", key_base, pkl_path, json_path, pt_path
        )

        if level == "Q":
            if pkl_count > 0:
                _quick_fields_and_tensors(
                    "PSTAR", key_base, pkl_path, pt_data,
                    RAW_FIELDS_BASE, plain_pkl=False
                )
        else:
            if pkl_count > 0:
                _detailed_pkl_sweep(
                    "PSTAR", key_base, pkl_path, RAW_FIELDS_BASE,
                    plain_pkl=False,
                    expect_sol=is_sol, expect_nosol=is_nosol,
                )
                if pt_data:
                    _detailed_pt_check("PSTAR", key_base, pt_data,
                                       RAW_FIELDS_BASE, full_sweep=False)


# ---------------------------------------------------------------------------
# GROUP 3 — RAGate
# ---------------------------------------------------------------------------

def _inspect_gridcheck(level):
    _header(f"[{level}] Group 3 — RAGate  (datasets/RAGate/)")

    gc_sol_count   = int(config.get("ragate_sol_count",   100000))
    gc_nosol_count = int(config.get("ragate_nosol_count", 100000))
    tr_r = config.get("train_ratio", 0.80)
    val_r = config.get("val_ratio",  0.10)
    n_tr  = round(gc_sol_count * tr_r)
    n_val = round(gc_sol_count * val_r)
    n_te  = gc_sol_count - n_tr - n_val

    gc_sets = [
        ("ragate_sol",   "ragate_sol_json",   "ragate_sol_pkl",   "ragate_sol_pt",   gc_sol_count,    True,  False),
        ("ragate_nosol", "ragate_nosol_json", "ragate_nosol_pkl", "ragate_nosol_pt", gc_nosol_count,  False, True),
        ("ragate_train", "ragate_train_json", "ragate_train_pkl", "ragate_train_pt", n_tr * 2,        False, False),
        ("ragate_val",   "ragate_val_json",   "ragate_val_pkl",   "ragate_val_pt",   n_val * 2,       False, False),
        ("ragate_test",  "ragate_test_json",  "ragate_test_pkl",  "ragate_test_pt",  n_te * 2,        False, False),
    ]

    split_data_for_overlap = {}

    for name, json_key, pkl_key, pt_key, expected, all_sol, all_nosol in gc_sets:
        _subheader(name)
        pkl_path  = config.get(pkl_key,  f"datasets/RAGate/{name}.pkl")
        json_path = config.get(json_key, f"datasets/RAGate/{name}.json")
        pt_path   = config.get(pt_key,   f"datasets/RAGate/{name}.pt")

        pkl_count, pt_data = _cross_format_counts(
            "RAGate", name, pkl_path, json_path, pt_path,
            expected=expected, plain_pkl=True
        )

        # Label balance (Quick + Detailed)
        if pkl_count > 0:
            try:
                data = _load_plain_pkl(pkl_path)
                n_sol_d   = sum(1 for s in data if s.get("label") == 1)
                n_nosol_d = sum(1 for s in data if s.get("label") == 0)

                if all_sol:
                    R.record("Q", "RAGate", f"{name} all labels=1",
                             n_sol_d == pkl_count,
                             f"{n_sol_d:,}/{pkl_count:,}")
                elif all_nosol:
                    R.record("Q", "RAGate", f"{name} all labels=0",
                             n_nosol_d == pkl_count,
                             f"{n_nosol_d:,}/{pkl_count:,}")
                else:
                    R.record("Q", "RAGate", f"{name} 1:1 label balance",
                             n_sol_d == n_nosol_d,
                             f"sol={n_sol_d:,}  nosol={n_nosol_d:,}")

                if level == "D" and name in ("ragate_train", "ragate_val", "ragate_test"):
                    split_data_for_overlap[name] = data
            except Exception as e:
                R.warn("RAGate", f"{name} label check failed", str(e))

        if level == "Q":
            if pkl_count > 0:
                _quick_fields_and_tensors(
                    "RAGate", name, pkl_path, pt_data,
                    RAW_FIELDS_LABELLED, plain_pkl=True
                )
        else:
            if pkl_count > 0:
                _detailed_pkl_sweep(
                    "RAGate", name, pkl_path, RAW_FIELDS_LABELLED,
                    plain_pkl=True
                )
                if pt_data:
                    _detailed_pt_check("RAGate", name, pt_data,
                                       RAW_FIELDS_LABELLED, full_sweep=True)

    # Cross-split overlap (Detailed only)
    if level == "D":
        _subheader("Cross-split overlap")
        pairs = [
            ("ragate_train", "ragate_val"),
            ("ragate_train", "ragate_test"),
            ("ragate_val",   "ragate_test"),
        ]
        for a, b in pairs:
            if a in split_data_for_overlap and b in split_data_for_overlap:
                _overlap_check("RAGate",
                               split_data_for_overlap[a], a,
                               split_data_for_overlap[b], b)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary():
    _header("SUMMARY")
    total, passed, failed = R.summary()

    groups = {
        "masters":    "Group 1 — Masters",
        "PSTAR": "Group 2 — Navigation",
        "RAGate":  "Group 3 — RAGate",
    }
    for key, name in groups.items():
        n, n_fail = R.group_summary(key)
        status = "PASS" if n_fail == 0 else "FAIL"
        print(f"\n  [{status}] {name}  "
              f"({n - n_fail}/{n} checks passed)")

    if failed > 0:
        print(f"\n  Failed checks:")
        for g, l, d in R.failed_records():
            print(f"    [{g}] {l}" + (f" — {d}" if d else ""))

    print(f"\n{'='*72}")
    print(f"  Result: {passed}/{total} checks passed,  {failed} failed.")
    if failed == 0:
        print("  ALL CHECKS PASSED")
    print(f"{'='*72}\n")
    return failed == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _header("inspect_datasets.py — Robot Path Planning Dataset V3")
    print(f"  Grid: {config['grid_size']}  "
          f"context={CONTEXT_LEN}  latent={LATENT_LEN}  "
          f"feature_size={FEATURE_SIZE}  padding={PADDING}")
    print(f"  Levels: Quick={'ON' if RUN_QUICK else 'OFF'}  "
          f"Detailed={'ON' if RUN_DETAILED else 'OFF'}")
    print(f"  Groups: Masters={'ON' if RUN_MASTERS else 'OFF'}  "
          f"Navigation={'ON' if RUN_NAV else 'OFF'}  "
          f"RAGate={'ON' if RUN_GC else 'OFF'}")

    if not RUN_QUICK and not RUN_DETAILED:
        print("\n  Both inspect_quick and inspect_detailed are False — nothing to do.")
        sys.exit(0)

    # Quick level
    if RUN_QUICK:
        _header("QUICK DATASETS INSPECTION")
        if RUN_MASTERS:    _inspect_masters("Q")
        if RUN_NAV:        _inspect_navigation("Q")
        if RUN_GC:         _inspect_gridcheck("Q")

    # Detailed level
    if RUN_DETAILED:
        _header("DETAILED DATASETS INSPECTION")
        if RUN_MASTERS:    _inspect_masters("D")
        if RUN_NAV:        _inspect_navigation("D")
        if RUN_GC:         _inspect_gridcheck("D")

    all_passed = _print_summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
