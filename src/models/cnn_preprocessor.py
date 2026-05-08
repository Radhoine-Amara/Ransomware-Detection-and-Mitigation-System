# src/models/cnn_preprocessor.py
# =============================================================================
# Layer 2 — Dynamic Behavioral Monitor
# Time-Series Preprocessing: raw VM telemetry → 3D tensor (N, timesteps, features)
# =============================================================================

import os
import glob
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing   import RobustScaler
from sklearn.model_selection import GroupKFold


# ── Base feature columns (raw telemetry) ──────────────────────────────────────
CNN_FEATURE_COLS_BASE = [
    "cpu_percent",
    "memory_rss_mb",
    "memory_vms_mb",
    "io_read_bytes_delta",
    "io_write_bytes_delta",
    "net_bytes_sent_delta",
    "num_open_files",
]

# ── Derived ratio features (encryption-behaviour proxy) ───────────────────────
# These capture the ransomware encryption signature without raw file bytes:
#   write_read_ratio  : ransomware writes as much as it reads (or more)
#   cpu_x_write       : CPU AND write bytes both spike during AES encryption
#   io_write_intensity: write rate relative to process memory footprint
CNN_FEATURE_COLS_DERIVED = CNN_FEATURE_COLS_BASE + [
    "write_read_ratio",
    "cpu_x_write",
    "io_write_intensity",
]

# Default uses derived features (better discrimination)
CNN_FEATURE_COLS = CNN_FEATURE_COLS_DERIVED

CNN_LABEL_COL = "label"
CNN_SORT_COL  = "step_number"
CNN_CACHE_VERSION = 4
CNN_LABEL_POLICY = "v7_auto_evidence_dilated_active_labels"


# ─────────────────────────────────────────────────────────────────────────────
def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 3 ratio features that proxy file-write entropy without raw bytes.

    write_read_ratio:   io_write / (io_read + 1)
        Ransomware reads a plaintext file, encrypts it, writes back.
        During encryption the write rate equals or exceeds the read rate.
        Normal processes read far more than they write.

    cpu_x_write:        cpu_percent × io_write_bytes_delta
        AES encryption is CPU-intensive AND produces high write I/O.
        Both must be elevated simultaneously (the key discriminator).
        Normal CPU-heavy processes (compiling) don't produce high I/O.

    io_write_intensity: io_write_bytes_delta / (memory_rss_mb × 1024 + 1)
        Normalises write rate by process size.
        Ransomware is disproportionately I/O-heavy relative to its footprint.
    """
    df = df.copy()
    df["write_read_ratio"]   = (df["io_write_bytes_delta"]
                                / (df["io_read_bytes_delta"] + 1.0))
    df["cpu_x_write"]        = (df["cpu_percent"]
                                * df["io_write_bytes_delta"])
    df["io_write_intensity"] = (df["io_write_bytes_delta"]
                                / (df["memory_rss_mb"] * 1024.0 + 1.0))
    return df


# ─────────────────────────────────────────────────────────────────────────────
def _validate_split(y_train, y_val, y_test, verbose=True):
    n_ran_tr = int((y_train == 1).sum())
    n_ben_tr = int((y_train == 0).sum())
    n_ran_va = int((y_val   == 1).sum())
    n_ran_te = int((y_test  == 1).sum())
    n_te     = len(y_test)

    if n_ran_tr == 0:
        return False, "Training split has 0 ransomware windows."
    if n_ran_va == 0:
        return False, "Validation split has 0 ransomware windows."
    if n_ran_te == 0:
        return False, "Test split has 0 ransomware windows."
    weight = n_ben_tr / max(n_ran_tr, 1)
    if weight > 200:
        return False, (f"Class imbalance too extreme: ratio={weight:.0f}:1. "
                       "Cache likely has broken labels.")
    if verbose:
        print(f"  Split OK: train ran={n_ran_tr} ben={n_ben_tr}  "
              f"val ran={n_ran_va}  test ran={n_ran_te}/{n_te}")
    return True, "OK"


# ─────────────────────────────────────────────────────────────────────────────
def discover_raw_csvs(behavioral_raw_dir: str) -> dict:
    """
    Discover CSVs under data/behavioral_raw/.
    Ransomware: cerber/, ryuk/, wannacry/
    Benign:     data/  (folder literally named 'data')
    """
    ransomware_csvs, benign_csvs = [], []
    for d in ["cerber", "ryuk", "wannacry"]:
        folder = os.path.join(behavioral_raw_dir, d)
        if os.path.isdir(folder):
            ransomware_csvs.extend(sorted(glob.glob(os.path.join(folder, "*.csv"))))
    for d in ["data"]:
        folder = os.path.join(behavioral_raw_dir, d)
        if os.path.isdir(folder):
            benign_csvs.extend(sorted(glob.glob(os.path.join(folder, "*.csv"))))
    return {"ransomware": ransomware_csvs, "benign": benign_csvs,
            "all": ransomware_csvs + benign_csvs}


# ─────────────────────────────────────────────────────────────────────────────
def _build_windows(feature_array, label_array, window_size, step_size,
                   majority_pos_fraction=0.25):
    """
    Slide a fixed window across one process lifecycle.

    Label strategy — positive-fraction (default, fraction=0.25):
        Window label = 1 if ≥ majority_pos_fraction of its ticks are ransomware.
        Better than any-positive:  avoids labeling quiet pre-encryption ticks as 1.
        Better than last-step:     catches windows that are MOSTLY encryption even
                                   if the final tick happens to be quiet.
        At fraction=0.25 with window_size=8: two active-encryption ticks are enough.
        This is a V7 recall-safety fix: ransomware bursts are often shorter than a
        full window, so a strict 50% majority fragmented attack labels too much.
    """
    n_feat = feature_array.shape[1]
    T      = len(feature_array)
    if T < window_size:
        return (np.empty((0, window_size, n_feat), dtype=np.float32),
                np.empty((0,), dtype=np.int32))
    X_w, y_w = [], []
    for start in range(0, T - window_size + 1, step_size):
        end   = start + window_size
        chunk = label_array[start:end]
        X_w.append(feature_array[start:end])
        y_w.append(1 if chunk.mean() >= majority_pos_fraction else 0)
    return np.array(X_w, dtype=np.float32), np.array(y_w, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
def _normalise_label_series(labels: pd.Series, forced_label: int) -> pd.Series:
    """
    Convert per-tick labels to clean {0, 1} integers.

    IMPORTANT: this preserves existing per-tick labels whenever the CSV already
    contains them. Folder labels are only used for missing/invalid values. This
    prevents the CNN from learning "this whole ransomware session is positive"
    when early ticks are actually pre-encryption/quiet.
    """
    if labels is None:
        return pd.Series(dtype=np.int32)

    if labels.dtype == object:
        mapped = labels.astype(str).str.lower().str.strip().map({
            "1": 1, "true": 1, "ransomware": 1, "malicious": 1, "attack": 1,
            "0": 0, "false": 0, "benign": 0, "normal": 0, "clean": 0,
        })
    else:
        mapped = pd.to_numeric(labels, errors="coerce")

    mapped = mapped.fillna(forced_label).astype(int).clip(0, 1)
    return mapped.astype(np.int32)


# Evidence-label thresholds are intentionally more sensitive than the runtime
# mitigation gate. They are used only to identify active-encryption ticks inside
# ransomware sessions so the CNN learns behavior, not folder/session identity.
DEFAULT_LABEL_EVIDENCE_THRESHOLDS = {
    "write_min_bytes": 64 * 1024,
    "read_min_bytes": 16 * 1024,
    "ratio_min": 0.25,
    "cpu_write_min": 1_000_000.0,
    "write_intensity_min": 128.0,
    "cpu_min": 8.0,
    "open_files_min": 2.0,
    "min_hits": 2,
    # V7 temporal dilation: one tick before and two ticks after active evidence.
    # This turns isolated active ticks into short bursts without relabeling the
    # whole ransomware session as malicious.
    "dilate_before_ticks": 1,
    "dilate_after_ticks": 2,
    "smooth_ticks": 0,
}


def _derive_active_encryption_labels(df: pd.DataFrame, thresholds: dict = None) -> pd.Series:
    """
    Build pseudo per-tick labels for active encryption behavior.

    This is used when the raw ransomware CSV labels are all-positive. In that
    situation, preserving labels is technically correct but behaviorally wrong:
    quiet pre-encryption ticks teach the CNN that ransomware-session membership
    is the target. This helper relabels only ticks that show encryption-like
    evidence as positive.
    """
    t = dict(DEFAULT_LABEL_EVIDENCE_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    tmp = _add_derived_features(df)
    read_b = pd.to_numeric(tmp.get("io_read_bytes_delta", 0.0), errors="coerce").fillna(0.0)
    write_b = pd.to_numeric(tmp.get("io_write_bytes_delta", 0.0), errors="coerce").fillna(0.0)
    ratio = pd.to_numeric(tmp.get("write_read_ratio", 0.0), errors="coerce").fillna(0.0)
    cpu_x_write = pd.to_numeric(tmp.get("cpu_x_write", 0.0), errors="coerce").fillna(0.0)
    intensity = pd.to_numeric(tmp.get("io_write_intensity", 0.0), errors="coerce").fillna(0.0)
    cpu = pd.to_numeric(tmp.get("cpu_percent", 0.0), errors="coerce").fillna(0.0)
    open_files = pd.to_numeric(tmp.get("num_open_files", 0.0), errors="coerce").fillna(0.0)

    checks = pd.DataFrame({
        "write_volume": write_b >= t["write_min_bytes"],
        "read_volume": read_b >= t["read_min_bytes"],
        "read_write_ratio": ratio >= t["ratio_min"],
        "cpu_write_coupling": cpu_x_write >= t["cpu_write_min"],
        "write_intensity": intensity >= t["write_intensity_min"],
        "cpu_activity": cpu >= t["cpu_min"],
        "open_file_pressure": open_files >= t["open_files_min"],
    })
    active = (checks.sum(axis=1) >= int(t["min_hits"])).astype(np.int32)

    # V7: temporally dilate active labels. Encryption behavior is bursty and
    # sliding windows may straddle the transition into active encryption. A small
    # dilation labels the local burst while still avoiding the old bug where the
    # entire ransomware session was forced positive.
    before = int(t.get("dilate_before_ticks", 0))
    after = int(t.get("dilate_after_ticks", 0))
    smooth = int(t.get("smooth_ticks", 0))
    if (before > 0 or after > 0) and len(active):
        arr = active.to_numpy(dtype=np.int32)
        dilated = arr.copy()
        active_idx = np.where(arr == 1)[0]
        for i in active_idx:
            lo = max(0, i - before)
            hi = min(len(arr), i + after + 1)
            dilated[lo:hi] = 1
        active = pd.Series(dilated, index=active.index, dtype=np.int32)
    elif smooth > 0 and len(active):
        active = (active.rolling(window=2 * smooth + 1, center=True, min_periods=1).max()
                  .fillna(0).astype(np.int32))
    return active.astype(np.int32)


def _maybe_apply_evidence_relabeling(
    df: pd.DataFrame,
    forced_label: int,
    relabel_policy: str,
    evidence_thresholds: dict = None,
) -> tuple[pd.DataFrame, str]:
    """Apply optional active-encryption relabeling for ransomware sessions.

    relabel_policy:
      - preserve: keep CSV/folder labels exactly as loaded.
      - evidence: always derive labels for forced ransomware sessions.
      - auto_evidence: derive labels only when a forced ransomware session is
        all-positive, which is the common label-noise pattern in this dataset.
    """
    df = df.copy()
    policy = (relabel_policy or "auto_evidence").lower()
    if forced_label != 1 or policy == "preserve":
        return df, "preserved"

    labels = df[CNN_LABEL_COL].astype(int)
    is_all_positive = bool(len(labels) > 0 and labels.min() == 1 and labels.max() == 1)
    should_relabel = policy == "evidence" or (policy == "auto_evidence" and is_all_positive)
    if not should_relabel:
        return df, "preserved_mixed_csv_labels"

    active = _derive_active_encryption_labels(df, thresholds=evidence_thresholds)
    # Safety: if thresholds produce zero positives for a ransomware file, keep the
    # strongest I/O tick positive so the session is not lost entirely. This is a
    # fallback, not the normal path.
    if int(active.sum()) == 0 and len(active):
        write = pd.to_numeric(df.get("io_write_bytes_delta", 0.0), errors="coerce").fillna(0.0)
        read = pd.to_numeric(df.get("io_read_bytes_delta", 0.0), errors="coerce").fillna(0.0)
        score = write + 0.5 * read
        active.iloc[int(score.to_numpy().argmax())] = 1
    df[CNN_LABEL_COL] = active.astype(np.int32)
    return df, "evidence_active_encryption"


def _load_csvs(csv_paths, forced_label, feature_cols_base, verbose,
               relabel_policy="auto_evidence", label_evidence_thresholds=None):
    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            if verbose:
                print(f"    WARNING skip {os.path.basename(path)}: {e}")
            continue

        # Preserve per-tick labels if they exist; use folder label only as fallback.
        if CNN_LABEL_COL in df.columns:
            original = df[CNN_LABEL_COL].copy()
            df[CNN_LABEL_COL] = _normalise_label_series(original, forced_label)
            if verbose:
                n_pos = int((df[CNN_LABEL_COL] == 1).sum())
                n_tot = len(df)
                print(f"    labels preserved: {os.path.basename(path)} "
                      f"pos={n_pos}/{n_tot} ({n_pos/max(n_tot,1):.1%})")
        else:
            df[CNN_LABEL_COL] = int(forced_label)
            if verbose:
                print(f"    labels fallback : {os.path.basename(path)} "
                      f"folder_label={forced_label}")

        if "session_id" not in df.columns:
            df["session_id"] = os.path.splitext(os.path.basename(path))[0]
        if "pid"        not in df.columns: df["pid"]        = 0
        if CNN_SORT_COL not in df.columns: df[CNN_SORT_COL] = np.arange(len(df))
        for col in feature_cols_base:
            if col not in df.columns: df[col] = 0.0

        before_pos = int((df[CNN_LABEL_COL] == 1).sum())
        df, relabel_source = _maybe_apply_evidence_relabeling(
            df, forced_label=forced_label, relabel_policy=relabel_policy,
            evidence_thresholds=label_evidence_thresholds,
        )
        if verbose and relabel_source != "preserved":
            after_pos = int((df[CNN_LABEL_COL] == 1).sum())
            n_tot = len(df)
            print(f"    labels relabeled: {os.path.basename(path)} "
                  f"{before_pos}/{n_tot} → {after_pos}/{n_tot} "
                  f"({after_pos/max(n_tot,1):.1%}) via {relabel_source}")

        keep = feature_cols_base + ["session_id", "pid", CNN_SORT_COL, CNN_LABEL_COL]
        frames.append(df[[c for c in keep if c in df.columns]].copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _print_label_audit(df: pd.DataFrame, stage: str = "raw"):
    """Print tick/session label distribution to catch broken dynamic labels."""
    if df.empty or CNN_LABEL_COL not in df.columns:
        print(f"  [Label audit:{stage}] no rows")
        return
    n = len(df)
    pos = int((df[CNN_LABEL_COL] == 1).sum())
    neg = int((df[CNN_LABEL_COL] == 0).sum())
    print(f"  [Label audit:{stage}] ticks benign={neg:,} ransomware={pos:,} "
          f"ran%={pos/max(n,1):.1%}")
    if "_gkey" in df.columns:
        sess = df.groupby("_gkey")[CNN_LABEL_COL].agg(["count", "sum", "mean"])
        n_all_pos = int((sess["mean"] == 1.0).sum())
        n_mixed   = int(((sess["mean"] > 0.0) & (sess["mean"] < 1.0)).sum())
        n_all_neg = int((sess["mean"] == 0.0).sum())
        print(f"  [Label audit:{stage}] sessions benign={n_all_neg:,} "
              f"mixed={n_mixed:,} all-ransomware={n_all_pos:,}")


# ─────────────────────────────────────────────────────────────────────────────
def preprocess_dynamic_telemetry(
    behavioral_raw_dir:    str,
    window_size:           int   = 8,
    step_size:             int   = 1,
    n_folds:               int   = 5,
    scaler_path:           str   = None,
    npz_cache:             str   = None,
    feature_cols:          list  = None,
    majority_pos_fraction: float = 0.25,
    relabel_policy:        str   = "auto_evidence",
    label_evidence_thresholds: dict = None,
    force_reprocess:       bool  = False,
    verbose:               bool  = True,
) -> dict:
    """
    Full preprocessing pipeline for dynamic VM telemetry.

    Key parameters
    ──────────────
    window_size     : default 8 (was 5/20). 8 ticks gives ~7 windows per
                      15-tick session with enough temporal context for the CNN.
    majority_pos_fraction : fraction of ticks in a window that must be
                      ransomware for the window to be labeled 1 (default 0.25 in V7).
    feature_cols    : defaults to CNN_FEATURE_COLS_DERIVED (10 features:
                      7 raw + 3 ratio features as encryption-entropy proxy).
    relabel_policy  : auto_evidence is recommended for this project. It preserves
                      mixed CSV labels but converts all-positive ransomware
                      sessions into active-encryption tick labels.

    Fast path: loads .npz if it exists, passes validation, and was built
               with the same window_size and last_step_label settings.
               force_reprocess=True bypasses this.
    """
    if feature_cols is None:
        feature_cols = CNN_FEATURE_COLS
    feature_cols_base = CNN_FEATURE_COLS_BASE

    # ── Fast path ─────────────────────────────────────────────────────────────
    required = {"X_train","y_train","X_val","y_val","X_test","y_test"}
    if not force_reprocess and npz_cache and os.path.exists(npz_cache):
        try:
            loaded = np.load(npz_cache, allow_pickle=True)
            if required.issubset(set(loaded.files)):
                cache_version = int(loaded["cache_version"][0]) if "cache_version" in loaded.files else 0
                if cache_version < CNN_CACHE_VERSION:
                    if verbose:
                        print(f"  Cache version {cache_version} < {CNN_CACHE_VERSION} — reprocessing.")
                    raise ValueError("cache version mismatch")
                X_tr = loaded["X_train"].astype(np.float32)
                y_tr = loaded["y_train"].astype(np.int32)
                X_va = loaded["X_val"].astype(np.float32)
                y_va = loaded["y_val"].astype(np.int32)
                X_te = loaded["X_test"].astype(np.float32)
                y_te = loaded["y_test"].astype(np.int32)
                # Check shape and label policy metadata match requested V7 settings.
                expected_feats = len(feature_cols)
                if X_tr.shape[1] != window_size or X_tr.shape[2] != expected_feats:
                    if verbose:
                        print(f"  Cache shape {X_tr.shape} ≠ requested "
                              f"({window_size},{expected_feats}) — reprocessing.")
                    raise ValueError("shape mismatch")
                if "majority_pos_fraction_meta" in loaded.files:
                    cached_frac = float(loaded["majority_pos_fraction_meta"][0])
                    if abs(cached_frac - float(majority_pos_fraction)) > 1e-9:
                        if verbose:
                            print(f"  Cache label fraction {cached_frac:.2f} ≠ requested "
                                  f"{majority_pos_fraction:.2f} — reprocessing.")
                        raise ValueError("label fraction mismatch")
                if "relabel_policy_meta" in loaded.files:
                    cached_policy = str(loaded["relabel_policy_meta"][0])
                    if cached_policy != str(relabel_policy):
                        if verbose:
                            print(f"  Cache relabel policy {cached_policy} ≠ requested "
                                  f"{relabel_policy} — reprocessing.")
                        raise ValueError("relabel policy mismatch")
                valid, reason = _validate_split(y_tr, y_va, y_te, verbose=verbose)
                if not valid:
                    if verbose:
                        print(f"  Cache rejected — {reason}. Reprocessing.")
                    raise ValueError(reason)
                if verbose:
                    print(f"  ✓ Cache valid. Loading from {os.path.basename(npz_cache)}")
                companion = npz_cache.replace(".npz", "_scaler.pkl")
                scaler = joblib.load(companion) if os.path.exists(companion) else None
                if scaler and scaler_path:
                    os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
                    joblib.dump(scaler, scaler_path)
                n_ben = int((y_tr==0).sum()); n_ran = int((y_tr==1).sum())
                raw_w = n_ben / max(n_ran, 1)
                cw = {0: 1.0, 1: round(min(8.0, raw_w), 4)}
                if verbose:
                    print(f"  X_train:{X_tr.shape} X_val:{X_va.shape} X_test:{X_te.shape}")
                    print(f"  class_weight={cw}")
                return {"X_train":X_tr,"y_train":y_tr,"X_val":X_va,"y_val":y_va,
                        "X_test":X_te,"y_test":y_te,"scaler":scaler,
                        "feature_cols":feature_cols,"input_shape":X_tr.shape[1:],
                        "class_weight":cw,"session_ids_test":set()}
        except Exception as e:
            if verbose and "shape mismatch" not in str(e) and "rejected" not in str(e):
                print(f"  Cache load failed ({e}), reprocessing.")
    elif force_reprocess and verbose:
        print("  force_reprocess=True — bypassing cache.")

    # ── Full path ─────────────────────────────────────────────────────────────
    if not os.path.isdir(behavioral_raw_dir):
        raise FileNotFoundError(
            f"behavioral_raw_dir not found: {behavioral_raw_dir}")

    csv_map = discover_raw_csvs(behavioral_raw_dir)
    if verbose:
        print(f"  Ransomware CSVs: {len(csv_map['ransomware'])} | "
              f"Benign CSVs: {len(csv_map['benign'])}")

    df_ran = _load_csvs(
        csv_map["ransomware"], 1, feature_cols_base, verbose,
        relabel_policy=relabel_policy,
        label_evidence_thresholds=label_evidence_thresholds,
    )
    df_ben = _load_csvs(
        csv_map["benign"], 0, feature_cols_base, verbose,
        relabel_policy="preserve",
        label_evidence_thresholds=label_evidence_thresholds,
    )

    if df_ran.empty and df_ben.empty:
        raise ValueError("No CSVs loaded.")

    df_all = pd.concat([df_ran, df_ben], ignore_index=True)

    # ── Add derived ratio features (encryption proxy) ──────────────────────────
    df_all = _add_derived_features(df_all)

    df_all["_gkey"] = (df_all["session_id"].astype(str)
                       + "__" + df_all["pid"].astype(str))
    if verbose:
        _print_label_audit(df_all, stage="after_load")
    grp_label  = df_all.groupby("_gkey")[CNN_LABEL_COL].max().to_dict()
    all_groups = df_all["_gkey"].unique()
    dummy_y    = np.array([grp_label[g] for g in all_groups])

    n_ran_sess = int((dummy_y == 1).sum())
    actual_folds = min(n_folds, max(2, n_ran_sess // 2))
    if actual_folds < n_folds and verbose:
        print(f"  Adjusting n_folds {n_folds}→{actual_folds} "
              f"({n_ran_sess} ransomware sessions)")

    if verbose:
        print(f"  Total sessions: {len(all_groups)} "
              f"(ransomware={n_ran_sess}, benign={int((dummy_y==0).sum())})")

    gkf    = GroupKFold(n_splits=actual_folds)
    folds  = list(gkf.split(np.zeros(len(all_groups)), dummy_y, all_groups))
    _, test_idx  = folds[0]
    test_groups  = set(all_groups[test_idx])
    remaining    = [g for g in all_groups if g not in test_groups]
    rem_y        = np.array([grp_label[g] for g in remaining])
    n_inner      = min(max(2, actual_folds-1), max(2, int((rem_y==1).sum())))
    gkf2         = GroupKFold(n_splits=n_inner)
    inner        = list(gkf2.split(np.zeros(len(remaining)), rem_y, remaining))
    _, val_i     = inner[0]
    val_groups   = set(np.array(remaining)[val_i])
    train_groups = set(remaining) - val_groups

    df_tr = df_all[df_all["_gkey"].isin(train_groups)].copy()
    df_va = df_all[df_all["_gkey"].isin(val_groups)].copy()
    df_te = df_all[df_all["_gkey"].isin(test_groups)].copy()

    if verbose:
        print(f"  Sessions — train:{len(train_groups)} "
              f"val:{len(val_groups)} test:{len(test_groups)}")

    # Fit scaler on train only — ALL feature columns including derived
    scaler = RobustScaler()
    df_tr[feature_cols] = scaler.fit_transform(df_tr[feature_cols])
    df_va[feature_cols] = scaler.transform(df_va[feature_cols])
    df_te[feature_cols] = scaler.transform(df_te[feature_cols])

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
        joblib.dump(scaler, scaler_path)
        if verbose: print(f"  RobustScaler saved → {scaler_path}")

    def _to_windows(df):
        all_X, all_y = [], []
        for gkey, grp in df.groupby("_gkey"):
            grp = grp.sort_values(CNN_SORT_COL)
            Xw, yw = _build_windows(
                grp[feature_cols].values.astype(np.float32),
                grp[CNN_LABEL_COL].values.astype(np.int32),
                window_size, step_size,
                majority_pos_fraction=majority_pos_fraction)
            if len(Xw):
                all_X.append(Xw); all_y.append(yw)
        if not all_X:
            return (np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
                    np.empty((0,), dtype=np.int32))
        return np.concatenate(all_X), np.concatenate(all_y)

    if verbose:
        label_strategy = f"positive-fraction (≥{majority_pos_fraction:.0%})"
        print(f"  Building windows (ws={window_size}, step={step_size}, label={label_strategy}) ...")
    X_train, y_train = _to_windows(df_tr)
    X_val,   y_val   = _to_windows(df_va)
    X_test,  y_test  = _to_windows(df_te)

    valid, reason = _validate_split(y_train, y_val, y_test, verbose=verbose)
    if not valid:
        raise ValueError(f"Split failed validation: {reason}")

    if npz_cache:
        os.makedirs(os.path.dirname(npz_cache) or ".", exist_ok=True)
        np.savez_compressed(npz_cache,
                            X_train=X_train, y_train=y_train,
                            X_val=X_val,     y_val=y_val,
                            X_test=X_test,   y_test=y_test,
                            cache_version=np.array([CNN_CACHE_VERSION], dtype=np.int32),
                            label_policy=np.array([f"{CNN_LABEL_POLICY}:{relabel_policy}"]),
                            window_size_meta=np.array([window_size], dtype=np.int32),
                            majority_pos_fraction_meta=np.array([majority_pos_fraction], dtype=np.float32),
                            relabel_policy_meta=np.array([str(relabel_policy)], dtype=object),
                            feature_cols_meta=np.array(feature_cols, dtype=object))
        joblib.dump(scaler, npz_cache.replace(".npz", "_scaler.pkl"))
        if verbose: print(f"  ✓ Cache saved → {npz_cache}")

    n_ben = int((y_train==0).sum())
    n_ran = int((y_train==1).sum())
    if n_ran == 0:
        raise ValueError("No ransomware windows in training split.")

    # Class weight: sqrt-softened to prevent model from over-penalising
    # false negatives to the point of flagging everything.
    # sqrt(n_ben/n_ran) gives a gentler weight than the raw ratio.
    # Capped at 10 to prevent extreme weights from small splits.
    raw_weight = n_ben / n_ran
    # Balanced, capped class weight. The previous very aggressive weighting could
    # push the CNN toward over-predicting ransomware. A lower cap keeps recall
    # emphasis without turning Layer 2 into an almost-always-positive model.
    cw = {0: 1.0, 1: round(min(5.0, raw_weight), 4)}

    if verbose:
        print(f"\n  Features: {len(feature_cols)} "
              f"({len(CNN_FEATURE_COLS_BASE)} base + "
              f"{len(feature_cols)-len(CNN_FEATURE_COLS_BASE)} derived)")
        print(f"  X_train:{X_train.shape}  ran%={y_train.mean()*100:.1f}%")
        print(f"  X_val  :{X_val.shape}  ran%={y_val.mean()*100:.1f}%")
        print(f"  X_test :{X_test.shape}  ran%={y_test.mean()*100:.1f}%")
        print(f"  class_weight={cw}  "
              f"(raw ratio {raw_weight:.2f}:1 → capped at 5.0)")
        print(f"  input_shape={X_train.shape[1:]}")

    return {"X_train":X_train,"y_train":y_train,
            "X_val":X_val,    "y_val":y_val,
            "X_test":X_test,  "y_test":y_test,
            "scaler":scaler,"feature_cols":feature_cols,
            "input_shape":X_train.shape[1:],
            "class_weight":cw,"session_ids_test":test_groups}
