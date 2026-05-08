# src/evaluation/counterfactual_tests.py
# =============================================================================
# Counterfactual helpers for testing whether CNN decisions depend on encryption
# behavior features rather than lab/session artifacts.
# =============================================================================

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


BASE_FEATURES = {
    "cpu_percent", "memory_rss_mb", "memory_vms_mb",
    "io_read_bytes_delta", "io_write_bytes_delta",
    "net_bytes_sent_delta", "num_open_files",
}


def _idx(feature_cols: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(feature_cols)}


def recompute_derived_features(window_raw: np.ndarray, feature_cols: List[str]) -> np.ndarray:
    """
    Recompute derived CNN features from the raw/base features.

    This is critical for counterfactual tests. If we reduce io_write_bytes_delta
    but leave cpu_x_write unchanged, the counterfactual becomes internally
    inconsistent and the explanation can invert. This helper enforces the exact
    same formulas used by cnn_preprocessor and edr_orchestrator.
    """
    x = np.asarray(window_raw, dtype=np.float32).copy()
    idx = _idx(feature_cols)

    def has(*names: str) -> bool:
        return all(n in idx for n in names)

    if has("write_read_ratio", "io_write_bytes_delta", "io_read_bytes_delta"):
        write_b = x[:, idx["io_write_bytes_delta"]]
        read_b = x[:, idx["io_read_bytes_delta"]]
        x[:, idx["write_read_ratio"]] = write_b / (read_b + 1.0)

    if has("cpu_x_write", "cpu_percent", "io_write_bytes_delta"):
        x[:, idx["cpu_x_write"]] = (
            x[:, idx["cpu_percent"]] * x[:, idx["io_write_bytes_delta"]]
        )

    if has("io_write_intensity", "io_write_bytes_delta", "memory_rss_mb"):
        x[:, idx["io_write_intensity"]] = (
            x[:, idx["io_write_bytes_delta"]]
            / (x[:, idx["memory_rss_mb"]] * 1024.0 + 1.0)
        )

    return x


def suppress_encryption_features(
    window_raw: np.ndarray,
    feature_cols: List[str],
    factor: float = 0.05,
    cpu_floor: float = 5.0,
) -> np.ndarray:
    """
    Return a coherent counterfactual copy where active-encryption behavior is
    suppressed.

    A well-behaved CNN should lower ransomware probability on this version.
    """
    x = np.asarray(window_raw, dtype=np.float32).copy()
    idx = _idx(feature_cols)

    for name in ["io_read_bytes_delta", "io_write_bytes_delta", "num_open_files"]:
        if name in idx:
            x[:, idx[name]] = np.maximum(0.0, x[:, idx[name]] * factor)

    if "cpu_percent" in idx:
        x[:, idx["cpu_percent"]] = np.minimum(x[:, idx["cpu_percent"]] * factor, cpu_floor)

    if "net_bytes_sent_delta" in idx:
        x[:, idx["net_bytes_sent_delta"]] = np.maximum(0.0, x[:, idx["net_bytes_sent_delta"]] * factor)

    return recompute_derived_features(x, feature_cols)


def inject_encryption_features(
    window_raw: np.ndarray,
    feature_cols: List[str],
    multiplier: float = 8.0,
    read_floor: float = 512 * 1024,
    write_floor: float = 1024 * 1024,
    cpu_floor: float = 65.0,
    open_files_floor: float = 50.0,
) -> np.ndarray:
    """
    Return a coherent counterfactual copy where encryption-like behavior is
    injected.

    Multiplying quiet benign I/O by 10 often keeps it near zero. Therefore this
    function uses both a multiplier and realistic floors for read/write/CPU/open
    files. A well-behaved CNN should increase ransomware probability.
    """
    x = np.asarray(window_raw, dtype=np.float32).copy()
    idx = _idx(feature_cols)

    if "io_read_bytes_delta" in idx:
        x[:, idx["io_read_bytes_delta"]] = np.maximum(
            x[:, idx["io_read_bytes_delta"]] * multiplier, read_floor
        )
    if "io_write_bytes_delta" in idx:
        x[:, idx["io_write_bytes_delta"]] = np.maximum(
            x[:, idx["io_write_bytes_delta"]] * multiplier, write_floor
        )
    if "cpu_percent" in idx:
        x[:, idx["cpu_percent"]] = np.maximum(x[:, idx["cpu_percent"]], cpu_floor)
    if "num_open_files" in idx:
        x[:, idx["num_open_files"]] = np.maximum(
            x[:, idx["num_open_files"]] * max(1.0, multiplier / 2.0), open_files_floor
        )

    return recompute_derived_features(x, feature_cols)


def predict_raw_window(cnn_model, window_raw: np.ndarray, scaler=None) -> float:
    """Predict one raw unscaled telemetry window with the CNN's scaler."""
    if scaler is None:
        scaler = getattr(cnn_model, "scaler", None)
    if scaler is None:
        raise ValueError("A CNN scaler is required.")
    model = getattr(cnn_model, "model", cnn_model)
    raw = recompute_derived_features(np.asarray(window_raw, dtype=np.float32), getattr(cnn_model, "feature_cols", None) or []) \
        if getattr(cnn_model, "feature_cols", None) else np.asarray(window_raw, dtype=np.float32)
    scaled = scaler.transform(raw)
    return float(model.predict(scaled[np.newaxis, ...], verbose=0)[0, 0])


def run_counterfactual_pair(
    cnn_model,
    window_raw: np.ndarray,
    feature_cols: List[str],
    scaler=None,
) -> Dict[str, float]:
    """
    Run both counterfactual directions on one window.

    Expected for ransomware windows:
        original_probability > suppressed_probability

    Expected for benign windows:
        injected_probability > original_probability
    """
    coherent_original = recompute_derived_features(window_raw, feature_cols)
    original = predict_raw_window(cnn_model, coherent_original, scaler)
    suppressed_window = suppress_encryption_features(coherent_original, feature_cols)
    injected_window = inject_encryption_features(coherent_original, feature_cols)
    suppressed = predict_raw_window(cnn_model, suppressed_window, scaler)
    injected = predict_raw_window(cnn_model, injected_window, scaler)
    return {
        "original_probability": original,
        "suppressed_probability": suppressed,
        "suppression_drop": original - suppressed,
        "injected_probability": injected,
        "injection_increase": injected - original,
        "valid_suppression_direction": bool(original > suppressed),
        "valid_injection_direction": bool(injected > original),
    }


def is_counterfactual_window_useful(
    probability: float,
    low: float = 0.05,
    high: float = 0.98,
) -> bool:
    """
    Saturated windows near 0 or 1 are poor explanation examples because changing
    one mechanism may not move the probability visibly. Prefer mid-confidence
    true positives/false positives for counterfactual reporting.
    """
    return low <= float(probability) <= high
