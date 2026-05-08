# src/explainability/edr_explain.py
# =============================================================================
# Explainability helpers: TreeSHAP for Layer 1 when available, occlusion for CNN
# =============================================================================

from __future__ import annotations

from typing import Dict, List
import numpy as np


def _recompute_derived_features(window_raw: np.ndarray, feature_cols: List[str]) -> np.ndarray:
    """Recompute derived features after coherent occlusion of base features."""
    x = np.asarray(window_raw, dtype=np.float32).copy()
    idx = {c: i for i, c in enumerate(feature_cols)}

    if {"write_read_ratio", "io_write_bytes_delta", "io_read_bytes_delta"}.issubset(idx):
        x[:, idx["write_read_ratio"]] = (
            x[:, idx["io_write_bytes_delta"]]
            / (x[:, idx["io_read_bytes_delta"]] + 1.0)
        )
    if {"cpu_x_write", "cpu_percent", "io_write_bytes_delta"}.issubset(idx):
        x[:, idx["cpu_x_write"]] = (
            x[:, idx["cpu_percent"]] * x[:, idx["io_write_bytes_delta"]]
        )
    if {"io_write_intensity", "io_write_bytes_delta", "memory_rss_mb"}.issubset(idx):
        x[:, idx["io_write_intensity"]] = (
            x[:, idx["io_write_bytes_delta"]]
            / (x[:, idx["memory_rss_mb"]] * 1024.0 + 1.0)
        )
    return x


def _baseline_vector(window_raw: np.ndarray, scaler, mode: str) -> np.ndarray:
    if mode == "scaler_center":
        center = getattr(scaler, "center_", None)
        if center is not None and len(center) == window_raw.shape[1]:
            return np.asarray(center, dtype=np.float32)
        # Fall back gracefully if scaler has no center_.
        return np.median(window_raw, axis=0).astype(np.float32)
    if mode == "median":
        return np.median(window_raw, axis=0).astype(np.float32)
    if mode == "zero":
        return np.zeros(window_raw.shape[1], dtype=np.float32)
    raise ValueError("baseline must be 'scaler_center', 'median', or 'zero'.")


def cnn_occlusion_explanation(
    cnn_model,
    window_raw: np.ndarray,
    feature_cols: List[str],
    scaler=None,
    baseline: str = "scaler_center",
    top_k: int = 5,
    coherent: bool = True,
) -> List[Dict[str, float]]:
    """
    Explain a CNN alert by feature occlusion.

    V8.1 fixes two explanation issues:
      1) use the same calibrated CNN prediction path as runtime inference;
      2) do not recompute a derived feature after it is directly occluded,
         otherwise the occlusion is silently undone and the drop becomes zero.

    Base-feature occlusion remains coherent: when a raw feature such as
    io_write_bytes_delta is changed, dependent derived features are recomputed.
    """
    if scaler is None:
        scaler = getattr(cnn_model, "scaler", None)
    if scaler is None:
        raise ValueError("A CNN scaler is required for occlusion explanation.")

    window_raw = np.asarray(window_raw, dtype=np.float32)
    if window_raw.ndim != 2:
        raise ValueError("window_raw must have shape (timesteps, features).")

    base_names = {
        "cpu_percent", "memory_rss_mb", "memory_vms_mb",
        "io_read_bytes_delta", "io_write_bytes_delta",
        "net_bytes_sent_delta", "num_open_files",
    }
    derived_names = {"write_read_ratio", "cpu_x_write", "io_write_intensity"}

    def _predict(raw_window: np.ndarray, recompute: bool = True) -> float:
        raw_window = _recompute_derived_features(raw_window, feature_cols) if (coherent and recompute) else raw_window
        if hasattr(cnn_model, "predict_proba_raw_window"):
            return float(cnn_model.predict_proba_raw_window(raw_window, scaler=scaler))
        model = getattr(cnn_model, "model", cnn_model)
        scaled = scaler.transform(raw_window)
        return float(model.predict(scaled[np.newaxis, ...], verbose=0)[0, 0])

    original_window = _recompute_derived_features(window_raw, feature_cols) if coherent else window_raw.copy()
    original_prob = _predict(original_window, recompute=False)
    replacement = _baseline_vector(original_window, scaler, baseline)

    rows = []
    for j, name in enumerate(feature_cols):
        occluded = original_window.copy()
        occluded[:, j] = replacement[j]
        # Recompute derived features only when a base feature was changed. If the
        # derived feature itself is occluded, recomputing would erase the test.
        recompute_after = bool(coherent and name in base_names)
        if recompute_after:
            occluded = _recompute_derived_features(occluded, feature_cols)
        prob = _predict(occluded, recompute=False)
        rows.append({
            "feature": name,
            "feature_type": "derived" if name in derived_names else "base",
            "original_probability": original_prob,
            "occluded_probability": prob,
            "probability_drop": original_prob - prob,
            "used_calibrated_prediction": bool(hasattr(cnn_model, "predict_proba_raw_window")),
        })

    rows.sort(key=lambda r: r["probability_drop"], reverse=True)
    return rows[:top_k]


def select_explainable_windows(cnn_model, X_raw: np.ndarray, scaler=None, low: float = 0.70, high: float = 0.95, max_items: int = 10):
    """Return indices of non-saturated windows suitable for occlusion examples."""
    if scaler is None:
        scaler = getattr(cnn_model, "scaler", None)
    if scaler is None:
        raise ValueError("A CNN scaler is required.")
    rows = []
    for i, w in enumerate(np.asarray(X_raw, dtype=np.float32)):
        if hasattr(cnn_model, "predict_proba_raw_window"):
            p = float(cnn_model.predict_proba_raw_window(w, scaler=scaler))
        else:
            model = getattr(cnn_model, "model", cnn_model)
            p = float(model.predict(scaler.transform(w)[np.newaxis, ...], verbose=0)[0, 0])
        if low <= p <= high:
            rows.append((i, p))
            if len(rows) >= max_items:
                break
    return rows


def static_shap_explanation(static_model, features: Dict[str, float], top_k: int = 10):
    """
    Optional Layer 1 SHAP explanation.

    This function uses SHAP if it is installed. If SHAP is not available, it
    falls back to model feature importances, which still gives a useful report
    explanation for RF/XGBoost/LightGBM.
    """
    feature_names = list(getattr(static_model, "feature_names", features.keys()))
    x = np.array([features.get(f, 0.0) for f in feature_names], dtype=np.float32).reshape(1, -1)

    scaler = getattr(static_model, "scaler", None)
    x_model = scaler.transform(x) if scaler is not None else x
    base_model = getattr(static_model, "model", static_model)

    try:
        import shap  # type: ignore
        explainer = shap.TreeExplainer(base_model)
        values = explainer.shap_values(x_model)
        if isinstance(values, list):
            values = values[-1]
        scores = np.asarray(values).reshape(-1)
        rows = [
            {"feature": f, "value": float(features.get(f, 0.0)), "shap_value": float(v)}
            for f, v in zip(feature_names, scores)
        ]
        rows.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
        return rows[:top_k]
    except Exception:
        importances = getattr(base_model, "feature_importances_", None)
        if importances is None:
            return []
        rows = [
            {"feature": f, "value": float(features.get(f, 0.0)), "importance": float(v)}
            for f, v in zip(feature_names, importances)
        ]
        rows.sort(key=lambda r: r["importance"], reverse=True)
        return rows[:top_k]
