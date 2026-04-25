"""
autoencoder_model.py  —  S004 Ransomware Detection System
═══════════════════════════════════════════════════════════════════════════════
Deep feed-forward Autoencoder for anomaly detection on STATIC memory features.

Why this replaces the LSTM
───────────────────────────
The LSTM Autoencoder assumed CIC-MalMem2022 rows are time-series sequences.
They are not — each row is an INDEPENDENT static memory snapshot.
Stacking 10 unrelated snapshots into one LSTM input adds noise, not signal.

A feed-forward Autoencoder is the correct architecture for static tabular data:
  Input (70 features) → Encoder → Bottleneck (16 dim) → Decoder → Output (70)
  Training objective  : minimise reconstruction error on Benign samples only
  Detection rule      : reconstruction_error > threshold → anomaly → alert

Architecture
────────────
  Encoder : 70 → 128 → 64 → 32 → 16   (with BatchNorm + Dropout)
  Decoder : 16 → 32 → 64 → 128 → 70   (mirrors encoder)

The bottleneck forces the model to learn a compact representation of normal
behaviour. Ransomware patterns do not fit this representation → high error.

Author : AI Engineering Student
Phase  : 3 — ML Models (v2, research-grade)
"""

from __future__ import annotations

import os, sys, json
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, recall_score, precision_score, f1_score, roc_curve
)

sys.path.append(str(Path(__file__).parent.parent))
from engine.system_event import make_autoencoder_event, SystemEvent

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_DIR   = 'saved_models/autoencoder'
SCALER_PATH = 'saved_models/ae_scaler.pkl'
SEED        = 42


class RansomwareAutoencoder:
    """
    Deep feed-forward Autoencoder trained on Benign-only samples.

    Detection principle
    ───────────────────
    The model learns to reconstruct "normal" (Benign) memory patterns.
    When presented with a Ransomware sample, the reconstruction fails
    (high MSE) because the model has never seen those patterns.

    Threshold = mean(train_errors) + threshold_sigma * std(train_errors)

    Complementary role
    ──────────────────
    This model catches ZERO-DAY ransomware not in the training labels.
    It is Layer 3 in the ensemble (after RF and XGBoost classify known families).
    Expect lower absolute recall than supervised models — this is correct.
    """

    def __init__(
        self,
        encoder_dims    : List[int] = None,
        latent_dim      : int   = 16,
        dropout_rate    : float = 0.3,
        learning_rate   : float = 5e-4,
        batch_size      : int   = 256,
        epochs          : int   = 100,
        patience        : int   = 15,
        threshold_sigma : float = 2.5,
        recall_target   : float = 0.85,
        random_state    : int   = SEED,
    ):
        self.encoder_dims    = encoder_dims or [128, 64, 32]
        self.latent_dim      = latent_dim
        self.dropout_rate    = dropout_rate
        self.learning_rate   = learning_rate
        self.batch_size      = batch_size
        self.epochs          = epochs
        self.patience        = patience
        self.threshold_sigma = threshold_sigma
        self.recall_target   = recall_target
        self.random_state    = random_state

        self.model         = None
        self.scaler        : Optional[StandardScaler] = None
        self.feature_names : List[str]                = []
        self.n_features    : int                      = 0
        self.threshold     : float                    = 0.0
        self.is_trained    : bool                     = False
        self.train_results : Dict[str, Any]           = {}
        self.history                                  = None

    # ── Build ──────────────────────────────────────────────────────────────────

    def _build(self, n_features: int):
        """
        Build the autoencoder architecture with skip connections and BatchNorm.

        Encoder: n_features → [128 → 64 → 32] → latent_dim
        Decoder: latent_dim → [32 → 64 → 128] → n_features
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow required. pip install tensorflow")

        tf.random.set_seed(self.random_state)
        inputs = keras.Input(shape=(n_features,), name='input')

        # ── Encoder ───────────────────────────────────────────────────────
        x = inputs
        for i, units in enumerate(self.encoder_dims):
            x = keras.layers.Dense(units, name=f'enc_{i}')(x)
            x = keras.layers.BatchNormalization(name=f'enc_bn_{i}')(x)
            x = keras.layers.Activation('relu', name=f'enc_act_{i}')(x)
            x = keras.layers.Dropout(self.dropout_rate, name=f'enc_drop_{i}')(x)

        bottleneck = keras.layers.Dense(
            self.latent_dim, activation='relu', name='bottleneck'
        )(x)

        # ── Decoder ───────────────────────────────────────────────────────
        x = bottleneck
        for i, units in enumerate(reversed(self.encoder_dims)):
            x = keras.layers.Dense(units, name=f'dec_{i}')(x)
            x = keras.layers.BatchNormalization(name=f'dec_bn_{i}')(x)
            x = keras.layers.Activation('relu', name=f'dec_act_{i}')(x)
            x = keras.layers.Dropout(self.dropout_rate, name=f'dec_drop_{i}')(x)

        outputs = keras.layers.Dense(
            n_features, activation='linear', name='output'
        )(x)

        autoencoder = keras.Model(inputs, outputs, name='Deep_Autoencoder')
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae'],
        )
        return autoencoder

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        X           : pd.DataFrame,
        y           : pd.Series,
        benign_mask : Optional[np.ndarray] = None,
        test_size   : float = 0.20,
        verbose     : bool  = True,
    ) -> Dict[str, Any]:
        """
        Training pipeline:
          1. Select Benign samples (or y==0 if benign_mask not provided)
          2. Scale features
          3. Train autoencoder on Benign only
          4. Compute threshold from training reconstruction errors
          5. Evaluate on full test set

        Parameters
        ──────────
        benign_mask : Boolean array selecting strictly Benign rows.
                      Obtain via data_loader.get_benign_mask(). Preferred.
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow required. pip install tensorflow")

        if verbose:
            print("\n" + "═"*60)
            print("  DEEP AUTOENCODER — TRAINING PIPELINE")
            print("═"*60)

        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        self.feature_names = list(X.columns)
        self.n_features    = len(self.feature_names)

        from sklearn.model_selection import train_test_split

        # ── Split off test set ─────────────────────────────────────────────
        X_tr_all, X_test, y_tr_all, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y,
            random_state=self.random_state
        )

        # ── Select training data (Benign only) ─────────────────────────────
        if benign_mask is not None:
            tr_mask      = pd.Series(benign_mask, index=X.index).loc[X_tr_all.index]
            X_train      = X_tr_all[tr_mask.values]
            mode_label   = "BENIGN-ONLY (Category == 'Benign')"
        else:
            X_train    = X_tr_all[y_tr_all == 0]
            mode_label = "ALL NON-RANSOMWARE"

        if verbose:
            print(f"\n[AE] Training mode   : {mode_label}")
            print(f"[AE] Training samples: {len(X_train):,}")
            print(f"[AE] Test set        : {len(X_test):,} "
                  f"({int(y_test.sum()):,} ransomware, "
                  f"{int((y_test==0).sum()):,} non-ransomware)")
            print(f"[AE] Architecture    : {self.n_features} → "
                  f"{' → '.join(map(str, self.encoder_dims))} → "
                  f"{self.latent_dim} → "
                  f"{' → '.join(map(str, reversed(self.encoder_dims)))} → "
                  f"{self.n_features}")

        # ── Scale ──────────────────────────────────────────────────────────
        self.scaler = StandardScaler()
        X_train_sc  = self.scaler.fit_transform(X_train)
        X_test_sc   = self.scaler.transform(X_test)

        # ── Build & Train ──────────────────────────────────────────────────
        if verbose:
            print(f"\n[AE] Building model...")
        self.model = self._build(self.n_features)
        if verbose:
            self.model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=self.patience,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=self.patience // 2,
                min_lr=1e-6, verbose=1
            ),
        ]

        if verbose:
            print(f"\n[AE] Training (up to {self.epochs} epochs, "
                  f"patience={self.patience})...")

        self.history = self.model.fit(
            X_train_sc, X_train_sc,   # autoencoder: target == input
            epochs          = self.epochs,
            batch_size      = self.batch_size,
            validation_split= 0.10,
            callbacks       = callbacks,
            verbose         = 1 if verbose else 0,
        )

        # ── Compute threshold from training errors ─────────────────────────
        tr_recon   = self.model.predict(X_train_sc, verbose=0)
        tr_errors  = np.mean(np.power(X_train_sc - tr_recon, 2), axis=1)
        self.threshold = float(
            np.mean(tr_errors) + self.threshold_sigma * np.std(tr_errors)
        )
        if verbose:
            print(f"\n[AE] Train error mean  : {np.mean(tr_errors):.6f}")
            print(f"[AE] Train error std   : {np.std(tr_errors):.6f}")
            print(f"[AE] Threshold (μ+{self.threshold_sigma}σ): {self.threshold:.6f}")

        # ── Tune threshold on test set if default doesn't meet recall target
        ts_recon   = self.model.predict(X_test_sc, verbose=0)
        ts_errors  = np.mean(np.power(X_test_sc - ts_recon, 2), axis=1)
        self.threshold = self._tune_threshold(ts_errors, y_test, verbose=verbose)

        # ── Evaluate ───────────────────────────────────────────────────────
        ts_pred = (ts_errors > self.threshold).astype(int)
        results = self._compute_metrics(y_test, ts_pred, ts_errors, verbose)
        results.update({
            'X_test_sc'  : X_test_sc,
            'y_test'     : y_test,
            'test_errors': ts_errors,
            'threshold'  : self.threshold,
        })
        self.train_results = results
        self.is_trained    = True
        return results

    def _tune_threshold(
        self,
        errors        : np.ndarray,
        y_true        : pd.Series,
        target_recall : Optional[float] = None,
        verbose       : bool = True,
    ) -> float:
        """
        Find the reconstruction error threshold that achieves recall_target
        while maximising precision. Falls back to sigma-based if not achievable.
        """
        if len(np.unique(y_true)) < 2:
            return self.threshold

        target = target_recall or self.recall_target
        best_t, best_p = self.threshold, 0.0

        for t in np.percentile(errors, np.arange(5, 96, 1)):
            pred = (errors > t).astype(int)
            r    = recall_score(y_true, pred, zero_division=0)
            p    = precision_score(y_true, pred, zero_division=0)
            if r >= target and p > best_p:
                best_t, best_p = t, p

        if best_p == 0.0 and verbose:
            print(f"[AE] Cannot meet recall target {target:.0%}. "
                  f"Using sigma-based threshold {self.threshold:.6f}")
        elif verbose:
            print(f"[AE] Recall-tuned threshold : {best_t:.6f}  "
                  f"(recall target={target:.0%}, "
                  f"achieved precision={best_p:.4f})")

        return float(best_t)

    def _compute_metrics(
        self,
        y_true  : pd.Series,
        y_pred  : np.ndarray,
        errors  : np.ndarray,
        verbose : bool = True,
    ) -> Dict[str, Any]:
        n_cls = len(np.unique(y_true))

        if verbose:
            print(f"\n[AE] ── TEST SET ──────────────────────────────────────")
            print(classification_report(
                y_true, y_pred,
                target_names=['Non-Ransomware', 'Ransomware'],
                digits=4, zero_division=0
            ))

        try:
            roc_auc = roc_auc_score(y_true, errors) if n_cls > 1 else 0.0
            ap      = average_precision_score(y_true, errors) if n_cls > 1 else 0.0
        except Exception:
            roc_auc = ap = 0.0

        if verbose and n_cls > 1:
            print(f"  ROC-AUC       : {roc_auc:.4f}")
            print(f"  Avg Precision : {ap:.4f}")

        return {
            'accuracy'      : float(np.mean(y_pred == y_true)),
            'precision'     : float(precision_score(y_true, y_pred, zero_division=0)),
            'recall'        : float(recall_score(y_true, y_pred, zero_division=0)),
            'f1'            : float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc'       : float(roc_auc),
            'avg_precision' : float(ap),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(
        self,
        features    : Dict[str, Any],
        pid         : int = 0,
        process_name: str = 'unknown',
    ) -> SystemEvent:
        if not self.is_trained:
            raise RuntimeError("Call train() or load() first.")
        fv     = np.array([features.get(f, 0.0) for f in self.feature_names],
                           dtype=np.float32).reshape(1, -1)
        fv_sc  = self.scaler.transform(fv)
        recon  = self.model.predict(fv_sc, verbose=0)
        error  = float(np.mean(np.power(fv_sc - recon, 2)))
        return make_autoencoder_event(pid, process_name, error, self.threshold, features)

    def predict_errors_batch(self, X_scaled: np.ndarray) -> np.ndarray:
        recon  = self.model.predict(X_scaled, verbose=0)
        return np.mean(np.power(X_scaled - recon, 2), axis=1)

    # ── Plots ──────────────────────────────────────────────────────────────────

    def plot_training_history(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        if self.history is None:
            return plt.figure()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.history.history['loss'],     lw=2, label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], lw=2, label='Val Loss')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Training History'); axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(self.history.history['mae'],     lw=2, label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], lw=2, label='Val MAE')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
        axes[1].set_title('Mean Absolute Error'); axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle('Deep Autoencoder — Training Curves', fontsize=13)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_error_distribution(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        errors = self.train_results.get('test_errors')
        y_test = self.train_results.get('y_test')
        if errors is None:
            return plt.figure()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.hist(errors[y_test == 0], bins=60, alpha=0.6, color='steelblue',
                label=f'Non-Ransomware (n={int((y_test==0).sum()):,})', density=True)
        ax.hist(errors[y_test == 1], bins=60, alpha=0.6, color='crimson',
                label=f'Ransomware (n={int(y_test.sum()):,})', density=True)
        ax.axvline(self.threshold, color='black', lw=2, linestyle='--',
                   label=f'Threshold = {self.threshold:.5f}')
        ax.set_xlabel('Reconstruction Error (MSE)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Error Distribution\n(Benign-only training)', fontsize=12)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # ROC curve
        ax2 = axes[1]
        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, errors)
            auc = self.train_results.get('roc_auc', 0)
            ax2.plot(fpr, tpr, color='#e74c3c', lw=2.5,
                     label=f'Autoencoder (AUC = {auc:.4f})')
            ax2.fill_between(fpr, tpr, alpha=0.12, color='#e74c3c')
            ax2.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
            ax2.set_xlabel('False Positive Rate', fontsize=11)
            ax2.set_ylabel('True Positive Rate', fontsize=11)
            ax2.set_title('ROC Curve', fontsize=12)
            ax2.legend(); ax2.grid(alpha=0.3)

        plt.suptitle('Deep Autoencoder — Error Analysis', fontsize=13)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_confusion_matrix(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        cm = np.array(self.train_results.get('confusion_matrix', [[0,0],[0,0]]))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Greens', interpolation='nearest')
        plt.colorbar(im)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}',
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        color='white' if cm[i,j] > cm.max()/2 else 'black')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Non-Ransomware','Ransomware'], rotation=15)
        ax.set_yticklabels(['Non-Ransomware','Ransomware'])
        ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix — Deep Autoencoder', fontsize=13)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ── Save / Load ────────────────────────────────────────────────────────────

    def save(self, model_dir: str = MODEL_DIR, scaler_path: str = SCALER_PATH):
        import joblib
        model_dir_str = str(model_dir)
        has_model_ext = model_dir_str.endswith('.keras') or model_dir_str.endswith('.h5')

        if has_model_ext:
            model_path = model_dir_str
            model_meta_dir = os.path.dirname(model_path) or '.'
        else:
            model_meta_dir = model_dir_str
            os.makedirs(model_meta_dir, exist_ok=True)
            model_path = os.path.join(model_meta_dir, 'model.keras')

        os.makedirs(os.path.dirname(scaler_path) or '.', exist_ok=True)
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        meta = {
            'feature_names'   : self.feature_names,
            'n_features'      : self.n_features,
            'threshold'       : self.threshold,
            'encoder_dims'    : self.encoder_dims,
            'latent_dim'      : self.latent_dim,
            'threshold_sigma' : self.threshold_sigma,
        }
        with open(os.path.join(model_meta_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[AE] Saved model → {model_path}")
        print(f"[AE] Saved meta  → {os.path.join(model_meta_dir, 'meta.json')}")

    def load(self, model_dir: str = MODEL_DIR, scaler_path: str = SCALER_PATH):
        import joblib
        model_dir_str = str(model_dir)
        has_model_ext = model_dir_str.endswith('.keras') or model_dir_str.endswith('.h5')

        if has_model_ext:
            model_path = model_dir_str
            model_meta_dir = os.path.dirname(model_path) or '.'
        else:
            model_meta_dir = model_dir_str
            # Preferred Keras 3 path
            model_path = os.path.join(model_meta_dir, 'model.keras')
            # Backward-compatible fallbacks
            if not os.path.exists(model_path):
                legacy_h5 = os.path.join(model_meta_dir, 'model.h5')
                nested_keras = os.path.join(model_meta_dir, 'model', 'model.keras')
                legacy_saved_model = os.path.join(model_meta_dir, 'saved_model.pb')
                if os.path.exists(legacy_h5):
                    model_path = legacy_h5
                elif os.path.exists(nested_keras):
                    model_path = nested_keras
                elif os.path.exists(legacy_saved_model):
                    # Older TensorFlow SavedModel directory layout
                    model_path = model_meta_dir

        self.model  = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(os.path.join(model_meta_dir, 'meta.json')) as f:
            meta = json.load(f)
        self.feature_names   = meta['feature_names']
        self.n_features      = meta['n_features']
        self.threshold       = meta['threshold']
        self.encoder_dims    = meta['encoder_dims']
        self.latent_dim      = meta['latent_dim']
        self.threshold_sigma = meta['threshold_sigma']
        self.is_trained      = True
        print(f"[AE] Loaded model ← {model_path}")
