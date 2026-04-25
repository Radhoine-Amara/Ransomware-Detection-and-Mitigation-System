"""
lstm_model.py
───────────────────────────────────────────────────────────────────────────────
LSTM Autoencoder for sequence-based ransomware detection.

Dataset : CIC-MalMem2022
Files   : data/datasets/Output1.csv
          data/datasets/output2.csv
          data/datasets/output3.csv

Trained ONLY on non-ransomware sequences.
Detects ransomware by measuring how poorly it reconstructs the input sequence.
High reconstruction error → behaviour is anomalous → ransomware suspected.

Author  : AI Engineering Student
Project : S004 — Ransomware Detection and Mitigation System
"""

from __future__ import annotations

import os, sys, json
import numpy  as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from typing  import Optional, List, Dict, Any, Tuple
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    classification_report, roc_auc_score,
    confusion_matrix, recall_score,
    precision_score, f1_score
)

sys.path.append(str(Path(__file__).parent))
from data_loader import load_cicmalmem, CICMALMEM_FILES

sys.path.append(str(Path(__file__).parent.parent))
from engine.system_event import make_lstm_event, SystemEvent

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[LSTM] TensorFlow not installed. Run: pip install tensorflow")

MODEL_SAVE_DIR   = "saved_models/lstm_autoencoder"
SCALER_SAVE_PATH = "saved_models/lstm_scaler.pkl"
RANDOM_STATE     = 42


class RansomwareLSTM:
    """
    LSTM Autoencoder for temporal sequence anomaly detection.

    While RF and IF look at a single snapshot of features,
    LSTM looks at a SEQUENCE of consecutive snapshots over time,
    capturing the characteristic progression pattern of ransomware.
    """

    def __init__(
        self,
        sequence_length: int   = 10,
        lstm_units     : int   = 64,
        latent_dim     : int   = 16,
        dropout_rate   : float = 0.2,
        learning_rate  : float = 0.001,
        epochs         : int   = 50,
        batch_size     : int   = 64,
    ):
        self.sequence_length = sequence_length
        self.lstm_units      = lstm_units
        self.latent_dim      = latent_dim
        self.dropout_rate    = dropout_rate
        self.learning_rate   = learning_rate
        self.epochs          = epochs
        self.batch_size      = batch_size

        self.model         = None
        self.scaler        : Optional[StandardScaler] = None
        self.feature_names : List[str]                = []
        self.threshold     : float                    = 0.0
        self.n_features    : int                      = 0
        self.is_trained    : bool                     = False
        self.train_results : Dict[str, Any]           = {}
        self.history                                  = None

    # ── Data loading ───────────────────────────────────────────────────────
    def load_data(
        self,
        file_paths: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the 3 CIC-MalMem2022 files.
        Returns (X, y) where y=1 means Ransomware.
        """
        X, y = load_cicmalmem(file_paths=file_paths, verbose=True)
        self.feature_names = list(X.columns)
        self.n_features    = len(self.feature_names)
        return X, y

    # ── Model architecture ─────────────────────────────────────────────────
    def _build_model(self, n_features: int):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow required. pip install tensorflow")
        inp = keras.Input(shape=(self.sequence_length, n_features))

        # Encoder
        x = keras.layers.LSTM(
            self.lstm_units, return_sequences=True,
            name='enc_lstm1')(inp)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        x = keras.layers.LSTM(
            self.latent_dim, return_sequences=False,
            name='enc_lstm2')(x)

        # Decoder
        x = keras.layers.RepeatVector(self.sequence_length)(x)
        x = keras.layers.LSTM(
            self.latent_dim, return_sequences=True,
            name='dec_lstm1')(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        x = keras.layers.LSTM(
            self.lstm_units, return_sequences=True,
            name='dec_lstm2')(x)
        out = keras.layers.TimeDistributed(
            keras.layers.Dense(n_features), name='output')(x)

        model = keras.Model(inp, out, name='LSTM_Autoencoder')
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss='mse'
        )
        return model

    # ── Sequence creation ──────────────────────────────────────────────────
    def _make_sequences(self, X: np.ndarray, step: int = 1) -> np.ndarray:
        """
        Convert 2D array (n_samples, n_features) into 3D sequences
        (n_sequences, sequence_length, n_features) using a sliding window.
        """
        seqs = []
        for i in range(0, len(X) - self.sequence_length + 1, step):
            seqs.append(X[i : i + self.sequence_length])
        return np.array(seqs)

    # ── Training ───────────────────────────────────────────────────────────
    def train(
        self,
        X        : pd.DataFrame,
        y        : pd.Series,
        test_size: float = 0.20,
    ) -> Dict[str, Any]:
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow required. pip install tensorflow")

        print("\n" + "="*60)
        print(" LSTM AUTOENCODER — TRAINING PIPELINE")
        print("="*60)

        self.feature_names = list(X.columns)
        self.n_features    = len(self.feature_names)

        from sklearn.model_selection import train_test_split
        X_tr_all, X_test, y_tr_all, y_test = train_test_split(
            X, y, test_size=test_size,
            stratify=y, random_state=RANDOM_STATE
        )

        # Train on NON-RANSOMWARE only
        X_train = X_tr_all[y_tr_all == 0]
        print(f"\n[LSTM] Training sequences from {len(X_train)} "
              f"non-ransomware samples")
        print(f"[LSTM] Test set: {len(X_test)} samples "
              f"({int(y_test.sum())} ransomware, "
              f"{int((y_test==0).sum())} non-ransomware)")

        # Scale
        self.scaler = StandardScaler()
        Xtr_sc = self.scaler.fit_transform(X_train)
        Xts_sc = self.scaler.transform(X_test)

        # Build sequences
        train_seqs = self._make_sequences(Xtr_sc, step=1)
        test_seqs  = self._make_sequences(Xts_sc, step=self.sequence_length)

        # Label for each test sequence = label of the LAST row in the window
        y_test_seq = np.array([
            int(y_test.iloc[min(
                i * self.sequence_length + self.sequence_length - 1,
                len(y_test) - 1
            )])
            for i in range(len(test_seqs))
        ])

        print(f"\n[LSTM] Train sequences shape: {train_seqs.shape}")
        print(f"[LSTM] Test  sequences shape: {test_seqs.shape}")

        # Build and train
        self.model = self._build_model(self.n_features)
        self.model.summary()

        cbs = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10,
                restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=5, verbose=1),
        ]

        print(f"\n[LSTM] Training (up to {self.epochs} epochs)...")
        self.history = self.model.fit(
            train_seqs, train_seqs,   # autoencoder: target = input
            epochs           = self.epochs,
            batch_size       = self.batch_size,
            validation_split = 0.1,
            callbacks        = cbs,
            verbose          = 1,
        )

        # Compute threshold from training errors
        tr_rec   = self.model.predict(train_seqs, verbose=0)
        tr_err   = np.mean(np.power(train_seqs - tr_rec, 2), axis=(1,2))
        self.threshold = float(np.mean(tr_err) + 2 * np.std(tr_err))
        print(f"\n[LSTM] Train error  mean  : {np.mean(tr_err):.6f}")
        print(f"[LSTM] Train error  std   : {np.std(tr_err):.6f}")
        print(f"[LSTM] Threshold (μ+2σ)  : {self.threshold:.6f}")

        # Evaluate
        ts_rec = self.model.predict(test_seqs, verbose=0)
        ts_err = np.mean(np.power(test_seqs - ts_rec, 2), axis=(1,2))
        ts_pred= (ts_err > self.threshold).astype(int)

        n  = min(len(ts_pred), len(y_test_seq))
        ts_pred    = ts_pred[:n]
        y_test_seq = y_test_seq[:n]
        ts_err_ev  = ts_err[:n]

        print("\n[LSTM] ── TEST SET ──────────────────────────────────────")
        print(classification_report(
            y_test_seq, ts_pred,
            target_names=['Non-Ransomware','Ransomware'],
            digits=4))
        try:
            auc = roc_auc_score(y_test_seq, ts_err_ev) \
                  if len(np.unique(y_test_seq)) > 1 else 0.0
        except Exception:
            auc = 0.0
        print(f"  ROC-AUC: {auc:.4f}")
        if len(np.unique(y_test_seq)) < 2:
            print("  [LSTM] Only one class in test set — ROC-AUC set to 0.0")

        res = {
            'accuracy' : float(np.mean(ts_pred == y_test_seq)),
            'precision': float(precision_score(y_test_seq, ts_pred, zero_division=0)),
            'recall'   : float(recall_score(y_test_seq, ts_pred, zero_division=0)),
            'f1'       : float(f1_score(y_test_seq, ts_pred, zero_division=0)),
            'roc_auc'  : float(auc),
            'confusion_matrix': confusion_matrix(y_test_seq, ts_pred).tolist(),
            'test_errors'     : ts_err_ev,
            'y_test_seq'      : y_test_seq,
            'threshold'       : self.threshold,
        }
        self.train_results = res
        self.is_trained    = True
        return res

    # ── Prediction ─────────────────────────────────────────────────────────
    def predict_sequence(
        self,
        feature_sequence: List[Dict[str, Any]],
        pid             : int = 0,
        process_name    : str = "unknown",
    ) -> SystemEvent:
        """
        Predict from a sequence of T feature snapshots.
        feature_sequence must have exactly sequence_length elements.
        """
        if not self.is_trained:
            raise RuntimeError("Call train() or load() first.")
        if len(feature_sequence) != self.sequence_length:
            raise ValueError(
                f"Need {self.sequence_length} steps, "
                f"got {len(feature_sequence)}")

        arr = np.array(
            [[s.get(f, 0.0) for f in self.feature_names]
             for s in feature_sequence],
            dtype=np.float32
        )
        sc  = self.scaler.transform(arr)
        inp = sc.reshape(1, self.sequence_length, self.n_features)
        rec = self.model.predict(inp, verbose=0)
        err = float(np.mean(np.power(inp - rec, 2)))

        return make_lstm_event(
            pid                  = pid,
            process_name         = process_name,
            reconstruction_error = err,
            threshold            = self.threshold,
            features             = feature_sequence[-1],
        )

    # ── Plots ──────────────────────────────────────────────────────────────
    def plot_training_history(self, save_path=None):
        if self.history is None:
            raise RuntimeError("Run train() first.")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history.history['loss'],     label='Train Loss', lw=2)
        ax.plot(self.history.history['val_loss'], label='Val Loss',   lw=2)
        ax.axhline(self.threshold, color='red', linestyle='--',
                   label=f'Threshold={self.threshold:.6f}')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.set_title('LSTM — Training History (CIC-MalMem2022)', fontsize=13)
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_error_distribution(self, save_path=None):
        errs   = self.train_results['test_errors']
        y_test = self.train_results['y_test_seq']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(errs[y_test==0], bins=50, alpha=0.6, color='steelblue',
                label=f'Non-Ransomware (n={int((y_test==0).sum())})',
                density=True)
        ax.hist(errs[y_test==1], bins=50, alpha=0.6, color='crimson',
                label=f'Ransomware (n={int(y_test.sum())})',
                density=True)
        ax.axvline(self.threshold, color='black', linestyle='--', lw=2,
                   label=f'Threshold={self.threshold:.6f}')
        ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('LSTM — Reconstruction Error Distribution\n'
                     'Ransomware should show higher error', fontsize=13)
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_confusion_matrix(self, save_path=None):
        cm = np.array(self.train_results['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Greens')
        plt.colorbar(im)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        color='white' if cm[i,j] > cm.max()/2 else 'black',
                        fontsize=16, fontweight='bold')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Non-Ransomware','Ransomware'], rotation=15)
        ax.set_yticklabels(['Non-Ransomware','Ransomware'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual',    fontsize=12)
        ax.set_title('Confusion Matrix — LSTM Autoencoder', fontsize=13)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ── Save / Load ────────────────────────────────────────────────────────
    def save(self, model_dir=MODEL_SAVE_DIR, scaler_path=SCALER_SAVE_PATH):
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

        self.model.save(model_path)

        scaler_dir = os.path.dirname(str(scaler_path))
        if scaler_dir:
            os.makedirs(scaler_dir, exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        meta = {
            'sequence_length': self.sequence_length,
            'n_features'     : self.n_features,
            'feature_names'  : self.feature_names,
            'threshold'      : self.threshold,
        }
        with open(os.path.join(model_meta_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        print(f"[LSTM] Saved model → {model_path}")
        print(f"[LSTM] Saved meta  → {os.path.join(model_meta_dir, 'meta.json')}")

    def load(self, model_dir=MODEL_SAVE_DIR, scaler_path=SCALER_SAVE_PATH):
        import joblib
        model_dir_str = str(model_dir)
        has_model_ext = model_dir_str.endswith('.keras') or model_dir_str.endswith('.h5')

        if has_model_ext:
            model_path = model_dir_str
            model_meta_dir = os.path.dirname(model_path) or '.'
        else:
            model_meta_dir = model_dir_str
            model_path = os.path.join(model_meta_dir, 'model.keras')

        self.model  = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(os.path.join(model_meta_dir, 'meta.json')) as f:
            meta = json.load(f)
        self.sequence_length = meta['sequence_length']
        self.n_features      = meta['n_features']
        self.feature_names   = meta['feature_names']
        self.threshold       = meta['threshold']
        self.is_trained      = True
        print(f"[LSTM] Loaded model ← {model_path}")
