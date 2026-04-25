"""
random_forest_model.py  —  S004 Ransomware Detection System
═══════════════════════════════════════════════════════════════════════════════
Supervised Random Forest classifier with:
  • Recall-optimised threshold (target recall ≥ 90%)
  • Isotonic probability calibration
  • Class-weight balancing + SMOTE
  • Cross-validated feature importance
  • Security-focused evaluation output

Author : AI Engineering Student
Phase  : 3 — ML Models (v2, research-grade)
"""

from __future__ import annotations

import os, sys, joblib
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

from sklearn.ensemble           import RandomForestClassifier
from sklearn.calibration        import CalibratedClassifierCV
try:
    # sklearn >=1.6: prefit calibration uses a FrozenEstimator with cv=None
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - older sklearn versions
    FrozenEstimator = None
from sklearn.model_selection    import (train_test_split, StratifiedKFold,
                                        cross_val_score)
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, brier_score_loss,
    f1_score, recall_score, precision_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling     import SMOTE

sys.path.append(str(Path(__file__).parent))
from data_loader import engineer_features

sys.path.append(str(Path(__file__).parent.parent))
from engine.system_event import make_random_forest_event, SystemEvent

MODEL_PATH  = 'saved_models/random_forest.pkl'
SCALER_PATH = 'saved_models/rf_scaler.pkl'
SEED        = 42


class RansomwareRandomForest:
    """
    Random Forest classifier with recall-optimised threshold.

    Key design decisions
    ────────────────────
    1. Threshold is tuned on the VALIDATION set to achieve recall ≥ 90%
       (not to maximise F1, which is the wrong objective for security).
    2. Isotonic calibration converts raw probabilities to calibrated ones,
       making the confidence score in SystemEvent more meaningful.
    3. SMOTE + class_weight='balanced' together address the 5:1 imbalance.
    4. OOB score gives a free, pessimistic estimate of generalisation.
    """

    def __init__(
        self,
        n_estimators      : int   = 500,
        max_depth                 = None,
        min_samples_split : int   = 4,
        min_samples_leaf  : int   = 2,
        max_features      : str   = 'sqrt',
        recall_target     : float = 0.90,
        use_smote         : bool  = True,
        calibrate         : bool  = True,
        random_state      : int   = SEED,
    ):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.recall_target     = recall_target
        self.use_smote         = use_smote
        self.calibrate         = calibrate
        self.random_state      = random_state

        self.model         : Optional[RandomForestClassifier]  = None
        self.calibrated    : Optional[CalibratedClassifierCV]  = None
        self.scaler        : Optional[StandardScaler]          = None
        self.feature_names : List[str]                         = []
        self.threshold     : float                             = 0.5
        self.is_trained    : bool                              = False
        self.train_results : Dict[str, Any]                    = {}

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        X         : pd.DataFrame,
        y         : pd.Series,
        test_size : float = 0.20,
        val_size  : float = 0.20,
        verbose   : bool  = True,
    ) -> Dict[str, Any]:
        """
        Full pipeline:
          split → scale → SMOTE → RF train → isotonic calibration →
          recall-optimised threshold → full evaluation
        """
        if verbose:
            print("\n" + "═"*60)
            print("  RANDOM FOREST — TRAINING PIPELINE")
            print("═"*60)

        self.feature_names = list(X.columns)
        n_classes = len(np.unique(y))
        if n_classes < 2:
            print("[RF] WARNING: only one class in data — metrics will be degenerate.")

        # ── Split ──────────────────────────────────────────────────────────
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y,
            random_state=self.random_state)
        val_frac = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_frac,
            stratify=y_tmp, random_state=self.random_state)

        if verbose:
            print(f"\n[RF] Train : {len(X_train):,} "
                  f"({int(y_train.sum()):,} ransomware, "
                  f"{int((y_train==0).sum()):,} other)")
            print(f"[RF] Val   : {len(X_val):,}")
            print(f"[RF] Test  : {len(X_test):,}")

        # ── Scale ──────────────────────────────────────────────────────────
        self.scaler = StandardScaler()
        Xtr = self.scaler.fit_transform(X_train)
        Xvl = self.scaler.transform(X_val)
        Xts = self.scaler.transform(X_test)

        # ── SMOTE ──────────────────────────────────────────────────────────
        from collections import Counter
        if self.use_smote and n_classes > 1 and int(y_train.sum()) >= 5:
            if verbose:
                print(f"[RF] Before SMOTE: {Counter(y_train)}")
            k = min(5, int(y_train.sum()) - 1)
            try:
                Xtr, y_train = SMOTE(
                    random_state=self.random_state,
                    k_neighbors=k
                ).fit_resample(Xtr, y_train)
                if verbose:
                    print(f"[RF] After  SMOTE: {Counter(y_train)}")
            except Exception as e:
                if verbose:
                    print(f"[RF] SMOTE skipped: {e}")

        # ── Train base RF ──────────────────────────────────────────────────
        if verbose:
            print(f"\n[RF] Training RandomForest "
                  f"({self.n_estimators} trees)...")
        cw = compute_class_weight('balanced',
                                   classes=np.unique(y_train), y=y_train)
        self.model = RandomForestClassifier(
            n_estimators      = self.n_estimators,
            max_depth         = self.max_depth,
            min_samples_split = self.min_samples_split,
            min_samples_leaf  = self.min_samples_leaf,
            max_features      = self.max_features,
            class_weight      = dict(enumerate(cw)),
            oob_score         = True,
            n_jobs            = -1,
            random_state      = self.random_state,
        )
        self.model.fit(Xtr, y_train)
        if verbose:
            print(f"[RF] OOB Score      : {self.model.oob_score_:.4f}")

        # ── Isotonic calibration ───────────────────────────────────────────
        if self.calibrate and n_classes > 1:
            if verbose:
                print("[RF] Calibrating probabilities (isotonic)...")
            try:
                if FrozenEstimator is not None:
                    self.calibrated = CalibratedClassifierCV(
                        estimator=FrozenEstimator(self.model),
                        method='isotonic',
                        cv=None,
                    )
                else:
                    self.calibrated = CalibratedClassifierCV(
                        self.model, method='isotonic', cv='prefit'
                    )
                self.calibrated.fit(Xvl, y_val)
                predict_proba_fn = lambda X: self.calibrated.predict_proba(X)[:, 1]
            except Exception as e:
                if verbose:
                    print(f"[RF] Calibration skipped: {e}")
                self.calibrated = None
                predict_proba_fn = lambda X: self.model.predict_proba(X)[:, 1]
        else:
            self.calibrated = None
            predict_proba_fn = lambda X: self.model.predict_proba(X)[:, 1]

        # ── Recall-optimised threshold on VALIDATION set ──────────────────
        val_proba      = predict_proba_fn(Xvl)
        self.threshold = self._tune_recall_threshold(
            val_proba, y_val, target_recall=self.recall_target, verbose=verbose
        )
        if verbose:
            print(f"[RF] Recall target  : {self.recall_target:.0%}")
            print(f"[RF] Optimal thresh : {self.threshold:.4f}")

        # ── Test evaluation ────────────────────────────────────────────────
        test_proba = predict_proba_fn(Xts)
        test_pred  = (test_proba >= self.threshold).astype(int)
        results    = self._compute_metrics(
            y_test, test_pred, test_proba, "TEST SET", verbose=verbose
        )
        results.update({
            'X_test': Xts, 'y_test': y_test,
            'test_proba': test_proba,
            'X_train': Xtr, 'y_train': y_train,
            'threshold': self.threshold,
        })
        self.train_results = results
        self.is_trained    = True

        # ── Cross-validated recall (on post-SMOTE training data) ──────────
        if verbose and n_classes > 1:
            print("\n[RF] 5-fold CV recall (on training data)...")
            cv_r = cross_val_score(
                self.model, Xtr, y_train,
                cv=StratifiedKFold(5, shuffle=True, random_state=self.random_state),
                scoring='recall', n_jobs=-1
            )
            print(f"[RF] CV Recall: {cv_r.mean():.4f} ± {cv_r.std():.4f}")
            results['cv_recall_mean'] = float(cv_r.mean())
            results['cv_recall_std']  = float(cv_r.std())

        return results

    def _tune_recall_threshold(
        self,
        proba        : np.ndarray,
        y_true       : pd.Series,
        target_recall: float = 0.90,
        verbose      : bool  = True,
    ) -> float:
        """
        Find the lowest threshold that achieves target_recall on the
        validation set while maximising precision at that recall level.

        Strategy:
          Sweep thresholds from high to low.
          Return the HIGHEST threshold where recall >= target_recall.
          This gives us the most conservative threshold that still
          meets our recall requirement (fewest false alarms).
        """
        if len(np.unique(y_true)) < 2:
            return 0.5

        best_threshold = 0.5
        best_precision = 0.0

        if verbose:
            print(f"\n[RF] Threshold sweep (target recall ≥ {target_recall:.0%}):")
            print(f"  {'Thresh':>8}  {'Recall':>8}  {'Precision':>10}  {'F1':>8}")

        for thresh in np.arange(0.05, 0.95, 0.01):
            pred = (proba >= thresh).astype(int)
            r    = recall_score(y_true, pred, zero_division=0)
            p    = precision_score(y_true, pred, zero_division=0)
            f    = f1_score(y_true, pred, zero_division=0)

            if r >= target_recall and p > best_precision:
                best_threshold = thresh
                best_precision = p

            if verbose and thresh in np.arange(0.05, 0.55, 0.05):
                marker = " ← selected" if abs(thresh - best_threshold) < 0.011 else ""
                print(f"  {thresh:>8.2f}  {r:>8.4f}  {p:>10.4f}  {f:>8.4f}{marker}")

        # If no threshold meets target, fall back to maximum recall
        if best_precision == 0.0:
            if verbose:
                print(f"[RF] WARNING: Could not meet recall target {target_recall:.0%}. "
                      f"Using threshold=0.10 for max recall.")
            best_threshold = 0.10

        return float(best_threshold)

    def _compute_metrics(
        self,
        y_true  : pd.Series,
        y_pred  : np.ndarray,
        y_proba : np.ndarray,
        label   : str = "",
        verbose : bool = True,
    ) -> Dict[str, Any]:
        """Compute and print full classification report + security metrics."""
        n_cls = len(np.unique(y_true))

        if verbose:
            print(f"\n[RF] ── {label} {'─'*(45-len(label))}")
            print(classification_report(
                y_true, y_pred,
                target_names=['Non-Ransomware', 'Ransomware'],
                digits=4, zero_division=0
            ))

        try:
            roc_auc = roc_auc_score(y_true, y_proba) if n_cls > 1 else 0.0
            ap      = average_precision_score(y_true, y_proba) if n_cls > 1 else 0.0
            brier   = brier_score_loss(y_true, y_proba) if n_cls > 1 else 1.0
        except Exception:
            roc_auc = ap = 0.0
            brier = 1.0

        if verbose and n_cls > 1:
            print(f"  ROC-AUC           : {roc_auc:.4f}")
            print(f"  Avg Precision     : {ap:.4f}")
            print(f"  Brier Score       : {brier:.4f}  (lower = better calibration)")

        cm = confusion_matrix(y_true, y_pred)
        cm_list = cm.tolist()
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            miss_rate = FN / max(FN + TP, 1)
            fpr       = FP / max(FP + TN, 1)
            if verbose:
                print(f"\n  True Positives    : {TP:,}  (ransomware caught)")
                print(f"  False Negatives   : {FN:,}  ← MISSED attacks ({miss_rate:.1%})")
                print(f"  False Positives   : {FP:,}  (false alarms, FPR={fpr:.1%})")
                print(f"  True Negatives    : {TN:,}")
        else:
            miss_rate = fpr = 0.0

        return {
            'accuracy'      : float(np.mean(y_pred == y_true)),
            'precision'     : float(precision_score(y_true, y_pred, zero_division=0)),
            'recall'        : float(recall_score(y_true, y_pred, zero_division=0)),
            'f1'            : float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc'       : float(roc_auc),
            'avg_precision' : float(ap),
            'brier_score'   : float(brier),
            'miss_rate'     : float(miss_rate),
            'confusion_matrix': cm_list,
        }

    # ── Prediction ─────────────────────────────────────────────────────────────

    def _predict_proba_single(self, X_scaled: np.ndarray) -> float:
        if self.calibrated is not None:
            return float(self.calibrated.predict_proba(X_scaled)[0, 1])
        return float(self.model.predict_proba(X_scaled)[0, 1])

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
        prob   = self._predict_proba_single(fv_sc)
        return make_random_forest_event(pid, process_name, prob, features)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for a pre-scaled batch."""
        if self.calibrated is not None:
            return self.calibrated.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    # ── Feature importance ─────────────────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 25) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Call train() first.")
        return pd.DataFrame({
            'feature'   : self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

    def plot_feature_importance(
        self, top_n: int = 25, save_path: Optional[str] = None
    ) -> plt.Figure:
        df  = self.get_feature_importance(top_n)
        # Colour engineered features differently
        colors = ['#e74c3c' if f.startswith('feat_') else '#2980b9'
                  for f in df['feature'][::-1]]
        fig, ax = plt.subplots(figsize=(11, 9))
        bars = ax.barh(df['feature'][::-1], df['importance'][::-1],
                       color=colors, edgecolor='white', linewidth=0.5)
        ax.axvline(df['importance'].mean(), color='gray', linestyle='--',
                   alpha=0.7, label='Mean importance')
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2980b9', label='Raw Volatility feature'),
            Patch(facecolor='#e74c3c', label='Engineered feature'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        ax.set_xlabel('Feature Importance (Mean Decrease Impurity)', fontsize=12)
        ax.set_title(f'Random Forest — Top {top_n} Features\n'
                     f'(CIC-MalMem2022, recall-optimised threshold={self.threshold:.3f})',
                     fontsize=13)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_roc_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        y    = self.train_results.get('y_test')
        prob = self.train_results.get('test_proba')
        if y is None or len(np.unique(y)) < 2:
            print("[RF] ROC curve skipped: single class in test set.")
            return plt.figure()
        fpr, tpr, thresholds = roc_curve(y, prob)
        auc = self.train_results['roc_auc']
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(fpr, tpr, color='#2980b9', lw=2.5,
                label=f'Random Forest (AUC = {auc:.4f})')
        ax.fill_between(fpr, tpr, alpha=0.12, color='#2980b9')
        # Mark operating threshold
        idx = np.argmin(np.abs(thresholds - self.threshold))
        ax.scatter(fpr[idx], tpr[idx], color='red', zorder=5, s=120,
                   label=f'Operating threshold ({self.threshold:.3f})')
        ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.4)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax.set_title('ROC Curve — Random Forest', fontsize=13)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_precision_recall_curve(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        y    = self.train_results.get('y_test')
        prob = self.train_results.get('test_proba')
        if y is None or len(np.unique(y)) < 2:
            return plt.figure()
        prec, rec, thresholds = precision_recall_curve(y, prob)
        ap = self.train_results['avg_precision']
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(rec, prec, color='#e74c3c', lw=2.5,
                label=f'Random Forest (AP = {ap:.4f})')
        ax.fill_between(rec, prec, alpha=0.12, color='#e74c3c')
        # Mark operating point
        idx = np.argmin(np.abs(thresholds - self.threshold))
        ax.scatter(rec[idx], prec[idx], color='blue', zorder=5, s=120,
                   label=f'Threshold={self.threshold:.3f}  '
                         f'Recall={rec[idx]:.3f}  Prec={prec[idx]:.3f}')
        ax.axhline(y.mean(), color='gray', linestyle='--', alpha=0.6,
                   label='No-skill baseline')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve — Random Forest', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_confusion_matrix(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        cm = np.array(self.train_results.get('confusion_matrix', [[0,0],[0,0]]))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar(im)
        classes = ['Non-Ransomware', 'Ransomware']
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(classes, rotation=15)
        ax.set_yticklabels(classes)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual',    fontsize=12)
        ax.set_title(f'Confusion Matrix — Random Forest\n'
                     f'threshold={self.threshold:.3f}', fontsize=13)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}',
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        color='white' if cm[i,j] > cm.max()/2 else 'black')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ── Save / Load ────────────────────────────────────────────────────────────

    def save(
        self,
        model_path : str = MODEL_PATH,
        scaler_path: str = SCALER_PATH,
    ):
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        payload = {
            'model'        : self.model,
            'calibrated'   : self.calibrated,
            'feature_names': self.feature_names,
            'threshold'    : self.threshold,
        }
        joblib.dump(payload, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"[RF] Saved model  → {model_path}")
        print(f"[RF] Saved scaler → {scaler_path}")

    def load(
        self,
        model_path : str = MODEL_PATH,
        scaler_path: str = SCALER_PATH,
    ):
        p = joblib.load(model_path)
        self.model         = p['model']
        self.calibrated    = p.get('calibrated')
        self.feature_names = p['feature_names']
        self.threshold     = p['threshold']
        self.scaler        = joblib.load(scaler_path)
        self.is_trained    = True
        print(f"[RF] Loaded ← {model_path}")
