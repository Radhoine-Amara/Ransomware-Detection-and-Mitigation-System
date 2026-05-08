# src/models/cnn_model.py
# =============================================================================
# Layer 2 — RansomwareCNN v8
# 1D-CNN behavioral ransomware detector trained on sliding-window telemetry.
# =============================================================================

import os
import json
import math
import numpy as np
import joblib
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve, roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

_tf = None

def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


class RansomwareCNN:
    """
    1D-CNN behavioral ransomware detector (Layer 2).

    Architecture
    ────────────
    Conv1D(64, k=3, relu, same, L2) → BN → MaxPool(2) → Dropout(0.25)
    Conv1D(128, k=3, relu, same, L2) → BN → Dropout(0.4)
    GlobalAveragePooling1D
    Dense(64, relu, L2) → Dropout(0.4) → Dense(1, sigmoid)

    L2 regularization (kernel_regularizer=l2(1e-4)) on Conv1D and Dense
    prevents the model from collapsing to a constant output when training
    data is small or noisy — this was the root cause of the FPR=100% bug.

    Threshold is tuned on the validation set with a precision-recall sweep.
    Among thresholds that meet the recall target, the model selects the one
    with the best F1 balance. This avoids recall-only training selecting an
    almost-always-positive model.
    """

    def __init__(
        self,
        window_size:   int   = 8,
        n_features:    int   = 10,
        filters_1:     int   = 64,
        filters_2:     int   = 128,
        dropout_rate:  float = 0.4,
        learning_rate: float = 5e-4,
        l2_reg:        float = 1e-4,
        recall_target: float = 0.85,
        threshold_floor: float = 0.35,
        calibration_method: str = "platt",
        strict_fpr_target: float = 0.20,
        balanced_fpr_target: float = 0.35,
        sensitive_min_threshold: float = 0.05,
        balanced_min_threshold: float = 0.12,
        strict_min_threshold: float = 0.20,
        epochs:        int   = 80,
        batch_size:    int   = 64,
        random_state:  int   = 42,
    ):
        self.window_size     = window_size
        self.n_features      = n_features
        self.filters_1       = filters_1
        self.filters_2       = filters_2
        self.dropout_rate    = dropout_rate
        self.learning_rate   = learning_rate
        self.l2_reg          = l2_reg
        self.recall_target   = recall_target
        self.threshold_floor = threshold_floor
        self.calibration_method = calibration_method
        self.strict_fpr_target = strict_fpr_target
        self.balanced_fpr_target = balanced_fpr_target
        self.sensitive_min_threshold = float(sensitive_min_threshold)
        self.balanced_min_threshold = float(balanced_min_threshold)
        self.strict_min_threshold = float(strict_min_threshold)
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.random_state    = random_state

        self.model        = None
        self.scaler       = None
        self.threshold    = 0.5
        self.threshold_modes = {}
        self.calibrator = None
        self.feature_cols = None
        self._history     = None
        self._y_test      = None
        self._y_prob_test = None
        self.is_trained   = False

    # ── Build ─────────────────────────────────────────────────────────────────
    def _build(self, input_shape):
        tf = _get_tf()
        tf.random.set_seed(self.random_state)
        keras = tf.keras
        reg   = keras.regularizers.l2(self.l2_reg)

        model = keras.Sequential(name="RansomwareCNN_Layer2", layers=[
            keras.Input(shape=input_shape, name="telemetry_window"),

            # Block A — short-range pattern extraction
            keras.layers.Conv1D(
                self.filters_1, kernel_size=3,
                activation="relu", padding="same",
                kernel_regularizer=reg, name="conv_a"),
            keras.layers.BatchNormalization(name="bn_a"),
            keras.layers.MaxPooling1D(pool_size=2, padding="same", name="pool_a"),
            keras.layers.Dropout(self.dropout_rate / 2, name="drop_a"),

            # Block B — compound pattern over pooled sequence
            keras.layers.Conv1D(
                self.filters_2, kernel_size=3,
                activation="relu", padding="same",
                kernel_regularizer=reg, name="conv_b"),
            keras.layers.BatchNormalization(name="bn_b"),
            keras.layers.Dropout(self.dropout_rate, name="drop_b"),

            # GlobalAveragePooling — NOT Flatten
            keras.layers.GlobalAveragePooling1D(name="gap"),

            # Classification head with L2
            keras.layers.Dense(64, activation="relu",
                               kernel_regularizer=reg, name="dense_head"),
            keras.layers.Dropout(self.dropout_rate, name="drop_head"),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                "accuracy",
            ],
        )
        return model

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, data: dict, model_dir: str = None) -> dict:
        tf = _get_tf()
        keras = tf.keras

        X_tr, y_tr   = data["X_train"], data["y_train"]
        X_va, y_va   = data["X_val"],   data["y_val"]
        X_te, y_te   = data["X_test"],  data["y_test"]
        class_weight = data["class_weight"]
        input_shape  = data["input_shape"]

        self.window_size  = input_shape[0]
        self.n_features   = input_shape[1]
        self.feature_cols = data.get("feature_cols")
        self.model        = self._build(input_shape)

        # Monitor PR-AUC instead of recall alone. Recall-only training can select
        # an almost-always-positive model, which is unsafe for hard EDR actions.
        monitor_metric = "val_pr_auc"
        cbs = [
            keras.callbacks.EarlyStopping(
                monitor=monitor_metric, mode="max",
                patience=12, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric, mode="max",
                factor=0.5, patience=6, min_lr=1e-6, verbose=0),
        ]
        ckpt_path = None
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            ckpt_path = os.path.join(model_dir, "model.keras")
            cbs.append(keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path, monitor=monitor_metric,
                mode="max", save_best_only=True, verbose=0))

        hist = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=cbs,
            verbose=1,
        )
        self._history = hist.history

        if ckpt_path and os.path.exists(ckpt_path):
            self.model = keras.models.load_model(ckpt_path)

        # Validation probabilities, optional calibration, and V8 threshold modes.
        raw_val_prob = self.model.predict(X_va, verbose=0).squeeze()
        self._fit_calibrator(raw_val_prob, y_va)
        val_prob = self._apply_calibration(raw_val_prob)
        self.threshold_modes = self._build_threshold_modes(val_prob, y_va)
        self.threshold = float(self.threshold_modes.get("balanced", {}).get("threshold", self._tune_threshold_from_probs(val_prob, y_va)))

        # Test evaluation with the same calibration used for validation tuning.
        raw_test_prob = self.model.predict(X_te, verbose=0).squeeze()
        y_prob = self._apply_calibration(raw_test_prob)
        y_pred = (y_prob >= self.threshold).astype(int)

        rec  = recall_score(y_te, y_pred, zero_division=0)
        prec = precision_score(y_te, y_pred, zero_division=0)
        f1   = f1_score(y_te, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_te, y_prob)
        except Exception:
            auc = 0.0
        try:
            ap = average_precision_score(y_te, y_prob)
        except Exception:
            ap = 0.0
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        fpr = FP / max(FP + TN, 1)
        specificity = TN / max(TN + FP, 1)
        bal_acc = balanced_accuracy_score(y_te, y_pred) if len(np.unique(y_te)) > 1 else 0.0

        self._y_test      = y_te
        self._y_prob_test = y_prob
        self.is_trained   = True

        return {
            "recall":           round(float(rec),  4),
            "precision":        round(float(prec), 4),
            "f1":               round(float(f1),   4),
            "roc_auc":          round(float(auc),  4),
            "pr_auc":           round(float(ap),   4),
            "fpr":              round(float(fpr),  4),
            "specificity":      round(float(specificity), 4),
            "balanced_accuracy":round(float(bal_acc), 4),
            "miss_rate":        round(float(1-rec), 4),
            "threshold":        round(self.threshold, 6),
            "threshold_modes":  self.threshold_modes,
            "calibration_method": self.calibration_method,
            "confusion_matrix": cm.tolist(),
            "y_test":           y_te,
            "test_proba":       y_prob,
            "history":          self._history,
        }

    # ── Threshold tuning and calibration ───────────────────────────────────────
    def _fit_calibrator(self, val_prob: np.ndarray, y_val: np.ndarray):
        """Fit an optional probability calibrator on validation probabilities.

        Calibration is intentionally one-dimensional: the CNN already produces a
        probability-like score, and the calibrator only reshapes that score so a
        benign window is less likely to sit near the alert threshold. If fitting
        fails, the model safely falls back to raw CNN probabilities.
        """
        method = (self.calibration_method or "none").lower()
        self.calibrator = None
        val_prob = np.asarray(val_prob, dtype=float).reshape(-1)
        y_val = np.asarray(y_val).astype(int).reshape(-1)
        if method in ("none", "off", "false") or len(np.unique(y_val)) < 2:
            return
        try:
            if method == "platt":
                lr = LogisticRegression(solver="lbfgs", max_iter=1000)
                lr.fit(val_prob.reshape(-1, 1), y_val)
                self.calibrator = lr
            elif method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(val_prob, y_val)
                self.calibrator = iso
            else:
                warnings.warn(f"Unknown calibration_method={method!r}; using raw probabilities.")
                self.calibration_method = "none"
                return
            print(f"  [Calibration] fitted {method} calibrator on validation scores")
        except Exception as e:
            warnings.warn(f"CNN calibration skipped: {e}")
            self.calibrator = None
            self.calibration_method = "none"

    def _apply_calibration(self, proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba, dtype=float).reshape(-1)
        if self.calibrator is None:
            return proba
        if isinstance(self.calibrator, LogisticRegression):
            return self.calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
        # IsotonicRegression exposes predict().
        return np.asarray(self.calibrator.predict(proba), dtype=float).reshape(-1)

    def _threshold_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
        pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        return {
            "threshold": float(threshold),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "fpr": float(fp / max(fp + tn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
        }

    def _build_threshold_modes(self, val_prob: np.ndarray, y_val: np.ndarray) -> dict:
        """Create genuinely distinct sensitive/balanced/strict operating points.

        V8.1 fixes the previous issue where Platt-calibrated probabilities could
        make sensitive, balanced, and strict thresholds nearly identical. Each
        mode now has a minimum operating threshold and an explicit selection goal:

          • sensitive: high recall, used only when Layer 1 is HIGH/CRITICAL.
          • balanced : default training/evaluation threshold.
          • strict   : lower-FPR mode for SAFE/WATCH contexts and hard-benign tests.

        The final EDR still uses the orchestrator evidence gate, so window-level
        threshold tuning should reduce noise without creating hard-kill risk.
        """
        y_val = np.asarray(y_val).astype(int)
        val_prob = np.asarray(val_prob, dtype=float).reshape(-1)
        if len(np.unique(y_val)) < 2:
            return {"balanced": self._threshold_metrics(y_val, val_prob, 0.5)}

        precs, recs, thrs_pr = precision_recall_curve(y_val, val_prob)
        candidates = set(float(t) for t in thrs_pr)
        candidates.update(np.linspace(0.001, 0.999, 999))
        candidates = sorted(t for t in candidates if 0.0 < t < 1.0)
        rows_all = [self._threshold_metrics(y_val, val_prob, t) for t in candidates]

        mode_floors = {
            "sensitive": max(0.001, float(self.sensitive_min_threshold), float(self.threshold_floor)),
            "balanced": max(0.001, float(self.balanced_min_threshold), float(self.threshold_floor)),
            "strict": max(0.001, float(self.strict_min_threshold), float(self.threshold_floor)),
        }

        def rows_for(mode):
            floor = mode_floors[mode]
            r = [x for x in rows_all if x["threshold"] >= floor]
            return r or rows_all

        def best_where(rows, predicate, key):
            valid = [r for r in rows if predicate(r)]
            if not valid:
                return None
            return max(valid, key=key)

        sens_rows = rows_for("sensitive")
        bal_rows = rows_for("balanced")
        strict_rows = rows_for("strict")

        # Sensitive: recall-first, but keep the best precision among valid choices.
        sensitive = best_where(
            sens_rows,
            lambda r: r["recall"] >= self.recall_target,
            lambda r: (r["precision"], r["f1"], -r["fpr"], r["threshold"]),
        )
        if sensitive is None:
            sensitive = max(sens_rows, key=lambda r: (r["recall"], r["f1"], r["precision"]))

        # Balanced: the default score. Prefer recall >= target-5%, but constrain FPR.
        balanced = best_where(
            bal_rows,
            lambda r: r["recall"] >= max(0.80, self.recall_target - 0.05) and r["fpr"] <= self.balanced_fpr_target,
            lambda r: (r["f1"], r["precision"], r["recall"], -r["fpr"]),
        )
        if balanced is None:
            balanced = best_where(
                bal_rows,
                lambda r: r["recall"] >= max(0.75, self.recall_target - 0.10),
                lambda r: (r["f1"], -r["fpr"], r["precision"], r["recall"]),
            ) or max(bal_rows, key=lambda r: (r["f1"], -r["fpr"], r["recall"]))

        # Strict: lower-FPR mode. Keep useful recall, but make FPR the priority.
        strict = best_where(
            strict_rows,
            lambda r: r["fpr"] <= self.strict_fpr_target and r["recall"] >= 0.50,
            lambda r: (r["f1"], r["precision"], r["recall"], -r["fpr"]),
        )
        if strict is None:
            strict = min(strict_rows, key=lambda r: (abs(r["fpr"] - self.strict_fpr_target), -r["recall"], -r["precision"]))

        modes = {"sensitive": sensitive, "balanced": balanced, "strict": strict}

        # Ensure interpretability: sensitive <= balanced <= strict when possible.
        # If validation data is pathological, keep metrics but mark the issue.
        order_ok = modes["sensitive"]["threshold"] <= modes["balanced"]["threshold"] <= modes["strict"]["threshold"]
        for name, r in modes.items():
            r["mode_floor"] = float(mode_floors[name])
            r["order_ok"] = bool(order_ok)

        print("  [Threshold modes — validation]")
        for name, r in modes.items():
            print(f"    {name:<9}: thr={r['threshold']:.4f} floor={r['mode_floor']:.3f} "
                  f"recall={r['recall']:.3f} precision={r['precision']:.3f} "
                  f"fpr={r['fpr']:.3f} f1={r['f1']:.3f}")
        if not order_ok:
            print("    ⚠ threshold order is not monotonic; validation distribution is unusual.")
        return modes

    def probability_diagnostics(self, y_true: np.ndarray = None, y_prob: np.ndarray = None) -> dict:
        """Return compact calibrated-score diagnostics for notebook/reporting."""
        if y_true is None:
            y_true = self._y_test
        if y_prob is None:
            y_prob = self._y_prob_test
        if y_true is None or y_prob is None:
            return {}
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
        out = {}
        for label, name in [(0, "benign"), (1, "ransomware")]:
            vals = y_prob[y_true == label]
            if len(vals):
                out[name] = {
                    "count": int(len(vals)),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "p90": float(np.quantile(vals, 0.90)),
                    "p95": float(np.quantile(vals, 0.95)),
                    "max": float(np.max(vals)),
                }
        return out

    def _tune_threshold_from_probs(self, y_prob: np.ndarray, y_val: np.ndarray) -> float:
        modes = self._build_threshold_modes(y_prob, y_val)
        return float(modes.get("balanced", {}).get("threshold", 0.5))

    def _tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Backwards-compatible API: tune a balanced threshold from validation X."""
        raw = self.model.predict(X_val, verbose=0).squeeze()
        self._fit_calibrator(raw, y_val)
        prob = self._apply_calibration(raw)
        self.threshold_modes = self._build_threshold_modes(prob, y_val)
        return float(self.threshold_modes.get("balanced", {}).get("threshold", 0.5))

    # ── Predict → SystemEvent ─────────────────────────────────────────────────
    def predict_proba_windows(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predict probabilities for already-scaled CNN windows."""
        raw = self.model.predict(X_scaled, verbose=0).squeeze()
        return self._apply_calibration(raw)

    def predict_proba_raw_window(self, window_raw: np.ndarray, scaler=None) -> float:
        """Predict one raw unscaled telemetry window using the attached scaler."""
        if scaler is None:
            scaler = self.scaler
        if scaler is None:
            raise ValueError("CNN scaler is required for raw-window inference.")
        scaled = scaler.transform(window_raw)
        return float(self.predict_proba_windows(scaled[np.newaxis, ...])[0])

    def predict(self, window_array: np.ndarray,
                pid: int = 0, process_name: str = "unknown"):
        from src.engine.system_event import (
            SystemEvent, SEVERITY_HIGH, SEVERITY_NONE,
            ACTION_SEND_ALERT, ACTION_LOG_EVENT,
        )
        prob = float(self.predict_proba_windows(window_array[np.newaxis, ...])[0])
        alert = prob >= self.threshold
        return SystemEvent(
            alert               = alert,
            severity            = SEVERITY_HIGH if alert else SEVERITY_NONE,
            model_source        = "Layer2_CNN",
            confidence          = round(prob, 4),
            pid                 = pid,
            process_name        = process_name,
            # Layer 2 alone is a behavioral suspicion signal; the orchestrator
            # decides soft/hard mitigation after persistence + evidence gating.
            recommended_actions = ([ACTION_SEND_ALERT, ACTION_LOG_EVENT]
                                   if alert else [ACTION_LOG_EVENT]),
            description         = (
                f"CNN behavioral score={prob:.4f} "
                f"{'≥' if alert else '<'} threshold={self.threshold:.4f}"
            ),
        )

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, model_dir: str, scaler_path: str = None):
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(os.path.join(model_dir, "model.keras"))
        meta = {
            "window_size":    self.window_size,
            "n_features":     self.n_features,
            "threshold":      self.threshold,
            "threshold_modes": self.threshold_modes,
            "calibration_method": self.calibration_method,
            "strict_fpr_target": self.strict_fpr_target,
            "balanced_fpr_target": self.balanced_fpr_target,
            "sensitive_min_threshold": self.sensitive_min_threshold,
            "balanced_min_threshold": self.balanced_min_threshold,
            "strict_min_threshold": self.strict_min_threshold,
            "filters_1":      self.filters_1,
            "filters_2":      self.filters_2,
            "dropout_rate":   self.dropout_rate,
            "l2_reg":         self.l2_reg,
            "recall_target":  self.recall_target,
            "threshold_floor":self.threshold_floor,
            "feature_cols":   self.feature_cols,
        }
        with open(os.path.join(model_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if self.calibrator is not None:
            joblib.dump(self.calibrator, os.path.join(model_dir, "calibrator.pkl"))
        if scaler_path and self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)
        print(f"  CNN saved → {model_dir}")

    @classmethod
    def load(cls, model_dir: str, scaler_path: str = None):
        tf = _get_tf()
        keras = tf.keras
        with open(os.path.join(model_dir, "meta.json")) as f:
            meta = json.load(f)
        obj = cls(
            window_size     = meta["window_size"],
            n_features      = meta["n_features"],
            filters_1       = meta.get("filters_1", 64),
            filters_2       = meta.get("filters_2", 128),
            dropout_rate    = meta.get("dropout_rate", 0.4),
            l2_reg          = meta.get("l2_reg", 1e-4),
            recall_target   = meta.get("recall_target", 0.85),
            threshold_floor = meta.get("threshold_floor", 0.35),
            calibration_method = meta.get("calibration_method", "none"),
            strict_fpr_target = meta.get("strict_fpr_target", 0.20),
            balanced_fpr_target = meta.get("balanced_fpr_target", 0.35),
            sensitive_min_threshold = meta.get("sensitive_min_threshold", 0.05),
            balanced_min_threshold = meta.get("balanced_min_threshold", 0.12),
            strict_min_threshold = meta.get("strict_min_threshold", 0.20),
        )
        obj.threshold    = meta["threshold"]
        obj.threshold_modes = meta.get("threshold_modes", {})
        obj.feature_cols = meta.get("feature_cols")
        obj.model        = keras.models.load_model(
            os.path.join(model_dir, "model.keras"))
        cal_path = os.path.join(model_dir, "calibrator.pkl")
        if os.path.exists(cal_path):
            obj.calibrator = joblib.load(cal_path)
        obj.is_trained   = True
        if scaler_path and os.path.exists(scaler_path):
            obj.scaler = joblib.load(scaler_path)
        return obj

    # ── Plots ─────────────────────────────────────────────────────────────────
    def plot_training_history(self, save_path: str):
        if not self._history:
            print("No history — run .train() first."); return
        hist = self._history
        er   = range(1, len(hist["loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(er, hist["loss"],     label="Train")
        axes[0].plot(er, hist["val_loss"], label="Val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
        if "pr_auc" in hist:
            axes[1].plot(er, hist["pr_auc"],     label="Train PR-AUC")
            axes[1].plot(er, hist["val_pr_auc"], label="Val PR-AUC")
            axes[1].set_title("PR-AUC (EarlyStopping metric)")
        else:
            axes[1].plot(er, hist["recall"],     label="Train recall")
            axes[1].plot(er, hist["val_recall"], label="Val recall")
            axes[1].axhline(self.recall_target, color="red", linestyle="--",
                            lw=1.2, label=f"{self.recall_target:.0%} target")
            axes[1].set_title("Recall")
        axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[2].plot(er, hist["precision"],     label="Train prec")
        axes[2].plot(er, hist["val_precision"], label="Val prec")
        axes[2].set_title("Precision")
        axes[2].set_ylim(0, 1.05); axes[2].legend(); axes[2].grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_confusion_matrix(self, save_path: str):
        if self._y_test is None:
            print("No test data — run .train() first."); return
        y_pred = (self._y_prob_test >= self.threshold).astype(int)
        cm = confusion_matrix(self._y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        labels = ["Benign", "Ransomware"]
        ax.set(xticks=[0,1], yticks=[0,1],
               xticklabels=labels, yticklabels=labels,
               xlabel="Predicted", ylabel="Actual",
               title=f"CNN Layer 2  (threshold={self.threshold:.4f})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        color="white" if cm[i,j]>cm.max()/2 else "black",
                        fontsize=14, fontweight="bold")
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
