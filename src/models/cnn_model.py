# src/models/cnn_model.py
# =============================================================================
# Layer 2 — RansomwareCNN
# 1D-CNN behavioral ransomware detector trained on sliding-window telemetry.
# =============================================================================

import os
import json
import math
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve, roc_curve,
)

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
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.random_state    = random_state

        self.model        = None
        self.scaler       = None
        self.threshold    = 0.5
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

        # Threshold tuning on validation set
        self.threshold = self._tune_threshold(X_va, y_va)

        # Test evaluation
        y_prob = self.model.predict(X_te, verbose=0).squeeze()
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
            "confusion_matrix": cm.tolist(),
            "y_test":           y_te,
            "test_proba":       y_prob,
            "history":          self._history,
        }

    # ── Threshold tuning ──────────────────────────────────────────────────────
    def _tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Precision-recall curve sweep targeting recall >= recall_target.

        Why PR-curve instead of Youden-J:
            Youden-J (TPR-FPR) maximises balanced accuracy across all thresholds
            and landed at 0.836 — far too high, giving only 61% test recall.
            The issue: Youden-J does not know we care more about missing
            ransomware than flagging benign. The PR-curve sweep directly finds
            the lowest threshold that meets the recall target.

        Logic:
            1. Sweep PR curve. Find all thresholds where recall >= recall_target.
            2. Among those, pick the one with the highest precision (best F1).
            3. If no threshold meets recall_target, pick threshold at recall_target*0.9.
            4. Apply a configurable floor only to prevent unsafe near-zero thresholds.
        """
        y_prob = self.model.predict(X_val, verbose=0).squeeze()

        precs, recs, thrs = precision_recall_curve(y_val, y_prob)
        # Note: precision_recall_curve returns arrays of len n+1; thrs has len n
        precs_t = precs[:-1]
        recs_t  = recs[:-1]

        # Find candidates meeting recall target
        candidates = [
            (float(t), float(p), float(r),
             2 * float(p) * float(r) / (float(p) + float(r) + 1e-9))
            for t, p, r in zip(thrs, precs_t, recs_t)
            if r >= self.recall_target
        ]

        if candidates:
            # Among candidates, pick highest F1 (best precision-recall balance)
            best = max(candidates, key=lambda x: x[3])
            chosen = max(float(best[0]), self.threshold_floor)
            print(f"  [Threshold] PR-curve: {chosen:.4f}  "
                  f"recall={best[2]:.3f}  precision={best[1]:.3f}  F1={best[3]:.3f}")
            return chosen

        # Fallback: recall target unachievable — use 90% of target
        relaxed = self.recall_target * 0.90
        candidates_relaxed = [
            (float(t), float(p), float(r),
             2 * float(p) * float(r) / (float(p) + float(r) + 1e-9))
            for t, p, r in zip(thrs, precs_t, recs_t)
            if r >= relaxed
        ]
        if candidates_relaxed:
            best = max(candidates_relaxed, key=lambda x: x[3])
            chosen = max(float(best[0]), self.threshold_floor)
            print(f"  [Threshold] Recall {self.recall_target:.0%} unachievable. "
                  f"Relaxed to {relaxed:.0%}: threshold={chosen:.4f}  "
                  f"recall={best[2]:.3f}  precision={best[1]:.3f}")
            return chosen

        # Last resort: max F1
        f1s = 2 * precs_t * recs_t / (precs_t + recs_t + 1e-9)
        chosen = max(float(thrs[np.argmax(f1s)]), self.threshold_floor)
        print(f"  [Threshold] Fallback max-F1: {chosen:.4f}")
        return chosen

    # ── Predict → SystemEvent ─────────────────────────────────────────────────
    def predict(self, window_array: np.ndarray,
                pid: int = 0, process_name: str = "unknown"):
        from src.engine.system_event import (
            SystemEvent, SEVERITY_HIGH, SEVERITY_NONE,
            ACTION_KILL_PROCESS, ACTION_SEND_ALERT,
            ACTION_LOG_EVENT, ACTION_NO_ACTION,
        )
        prob  = float(self.model.predict(
            window_array[np.newaxis, ...], verbose=0)[0, 0])
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
        )
        obj.threshold    = meta["threshold"]
        obj.feature_cols = meta.get("feature_cols")
        obj.model        = keras.models.load_model(
            os.path.join(model_dir, "model.keras"))
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
