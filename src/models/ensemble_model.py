"""
ensemble_model.py  —  S004 Ransomware Detection System
═══════════════════════════════════════════════════════════════════════════════
XGBoost and LightGBM classifiers + a weighted soft-voting ensemble.

Why add XGBoost / LightGBM
────────────────────────────
Random Forest is the standard baseline. XGBoost and LightGBM are gradient
boosting methods that consistently outperform RF on tabular data:
  • XGBoost: regularised gradient boosting, handles imbalance via scale_pos_weight
  • LightGBM: leaf-wise growth, faster than XGBoost, often higher recall
  • Ensemble: weighted soft voting of RF + XGB + LGB gives the best of all three

Research result expectation (CIC-MalMem2022):
  RF alone    : recall ≈ 90%, ROC-AUC ≈ 96%
  XGB alone   : recall ≈ 92%, ROC-AUC ≈ 97%
  LGB alone   : recall ≈ 91%, ROC-AUC ≈ 97%
  Ensemble    : recall ≈ 93–95%, ROC-AUC ≈ 98%

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

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from sklearn.calibration     import CalibratedClassifierCV
try:
    # sklearn >=1.6: prefit calibration uses FrozenEstimator + cv=None
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - older sklearn versions
    FrozenEstimator = None
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve,
    f1_score, recall_score, precision_score, brier_score_loss
)
from imblearn.over_sampling import SMOTE

sys.path.append(str(Path(__file__).parent.parent))
from engine.system_event import make_xgboost_event, make_lightgbm_event, make_ensemble_event, SystemEvent

SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost classifier
# ══════════════════════════════════════════════════════════════════════════════

class RansomwareXGBoost:
    """XGBoost classifier with recall-optimised threshold and SMOTE."""

    def __init__(
        self,
        n_estimators    : int   = 500,
        max_depth       : int   = 6,
        learning_rate   : float = 0.05,
        subsample       : float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int   = 3,
        gamma           : float = 0.1,
        reg_alpha       : float = 0.1,
        reg_lambda      : float = 1.0,
        eval_metric     : str   = 'aucpr',
        use_label_encoder: bool = False,
        recall_target   : float = 0.90,
        use_smote       : bool  = True,
        calibrate       : bool  = True,
        random_state    : int   = SEED,
        n_jobs          : int   = -1,
    ):
        try:
            import xgboost as xgb
            self._xgb = xgb
        except ImportError:
            raise ImportError("pip install xgboost")

        self.params = dict(
            n_estimators    = n_estimators,
            max_depth       = max_depth,
            learning_rate   = learning_rate,
            subsample       = subsample,
            colsample_bytree= colsample_bytree,
            min_child_weight= min_child_weight,
            gamma           = gamma,
            reg_alpha       = reg_alpha,
            reg_lambda      = reg_lambda,
            eval_metric     = eval_metric,
            random_state    = random_state,
            n_jobs          = n_jobs,
            use_label_encoder=use_label_encoder,
        )
        self.recall_target  = recall_target
        self.use_smote      = use_smote
        self.calibrate_flag = calibrate
        self.random_state   = random_state

        self.model         = None
        self.calibrated    = None
        self.scaler        : Optional[StandardScaler] = None
        self.feature_names : List[str]                = []
        self.threshold     : float                    = 0.5
        self.is_trained    : bool                     = False
        self.train_results : Dict[str, Any]           = {}

    def train(
        self,
        X         : pd.DataFrame,
        y         : pd.Series,
        test_size : float = 0.20,
        val_size  : float = 0.20,
        verbose   : bool  = True,
    ) -> Dict[str, Any]:
        if verbose:
            print("\n" + "═"*60)
            print("  XGBOOST — TRAINING PIPELINE")
            print("═"*60)

        self.feature_names = list(X.columns)
        from collections import Counter

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_size/(1-test_size),
            stratify=y_tmp, random_state=self.random_state)

        if verbose:
            print(f"[XGB] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

        self.scaler = StandardScaler()
        Xtr = self.scaler.fit_transform(X_train)
        Xvl = self.scaler.transform(X_val)
        Xts = self.scaler.transform(X_test)

        # Scale pos weight = ratio of negative to positive
        n_neg  = int((y_train == 0).sum())
        n_pos  = int((y_train == 1).sum())
        spw    = n_neg / max(n_pos, 1)

        if self.use_smote and n_pos >= 5:
            if verbose:
                print(f"[XGB] Before SMOTE: {Counter(y_train)}")
            k = min(5, n_pos - 1)
            try:
                Xtr, y_train = SMOTE(random_state=self.random_state, k_neighbors=k).fit_resample(Xtr, y_train)
                if verbose: print(f"[XGB] After SMOTE : {Counter(y_train)}")
            except Exception as e:
                if verbose: print(f"[XGB] SMOTE skipped: {e}")

        if verbose:
            print(f"[XGB] Training (scale_pos_weight={spw:.2f})...")

        self.model = self._xgb.XGBClassifier(
            **self.params,
            scale_pos_weight=spw,
        )
        try:
            # Older XGBoost sklearn API
            self.model.fit(
                Xtr, y_train,
                eval_set=[(Xvl, y_val)],
                early_stopping_rounds=40,
                verbose=False,
            )
        except TypeError:
            try:
                # Newer XGBoost sklearn API uses callback objects
                self.model.fit(
                    Xtr, y_train,
                    eval_set=[(Xvl, y_val)],
                    callbacks=[self._xgb.callback.EarlyStopping(rounds=40)],
                    verbose=False,
                )
            except Exception:
                # Final fallback: train without early stopping
                self.model.fit(Xtr, y_train)
        if verbose:
            best_it = getattr(self.model, 'best_iteration', None)
            if best_it is not None:
                print(f"[XGB] Best iteration: {best_it}")

        if self.calibrate_flag and len(np.unique(y_val)) > 1:
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
                predict_fn = lambda X: self.calibrated.predict_proba(X)[:, 1]
            except Exception as e:
                if verbose:
                    print(f"[XGB] Calibration skipped: {e}")
                self.calibrated = None
                predict_fn = lambda X: self.model.predict_proba(X)[:, 1]
        else:
            self.calibrated = None
            predict_fn = lambda X: self.model.predict_proba(X)[:, 1]

        val_proba      = predict_fn(Xvl)
        self.threshold = self._tune_recall(val_proba, y_val, verbose)
        test_proba     = predict_fn(Xts)
        test_pred      = (test_proba >= self.threshold).astype(int)
        results        = self._metrics(y_test, test_pred, test_proba, "TEST SET", verbose)
        results.update({'X_test': Xts, 'y_test': y_test, 'test_proba': test_proba})
        self.train_results = results
        self.is_trained    = True
        return results

    def _tune_recall(self, proba, y_true, verbose=True):
        if len(np.unique(y_true)) < 2: return 0.5
        best_t, best_p = 0.5, 0.0
        for t in np.arange(0.05, 0.90, 0.01):
            pred = (proba >= t).astype(int)
            r = recall_score(y_true, pred, zero_division=0)
            p = precision_score(y_true, pred, zero_division=0)
            if r >= self.recall_target and p > best_p:
                best_t, best_p = t, p
        if best_p == 0: best_t = 0.10
        if verbose: print(f"[XGB] Recall-optimised threshold: {best_t:.4f}")
        return float(best_t)

    def _metrics(self, y_true, y_pred, y_proba, label="", verbose=True):
        n_cls = len(np.unique(y_true))
        if verbose:
            print(f"\n[XGB] ── {label} ──────────────────────────────────")
            print(classification_report(y_true, y_pred,
                  target_names=['Non-Ransomware','Ransomware'],
                  digits=4, zero_division=0))
        try:
            auc = roc_auc_score(y_true, y_proba) if n_cls > 1 else 0.0
            ap  = average_precision_score(y_true, y_proba) if n_cls > 1 else 0.0
        except Exception: auc = ap = 0.0
        if verbose and n_cls > 1:
            print(f"  ROC-AUC: {auc:.4f} | Avg Precision: {ap:.4f}")
        return {
            'accuracy': float(np.mean(y_pred == y_true)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(auc), 'avg_precision': float(ap),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }

    def predict(self, features: Dict[str, Any], pid: int = 0,
                process_name: str = 'unknown') -> SystemEvent:
        if not self.is_trained: raise RuntimeError("Call train() first.")
        fv = np.array([features.get(f, 0.0) for f in self.feature_names],
                       dtype=np.float32).reshape(1, -1)
        fv_sc = self.scaler.transform(fv)
        if self.calibrated:
            prob = float(self.calibrated.predict_proba(fv_sc)[0, 1])
        else:
            prob = float(self.model.predict_proba(fv_sc)[0, 1])
        return make_xgboost_event(pid, process_name, prob, features)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        if self.calibrated: return self.calibrated.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, top_n: int = 25) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return pd.DataFrame({'feature': self.feature_names, 'importance': imp})\
               .sort_values('importance', ascending=False).head(top_n)

    def plot_feature_importance(self, top_n=25, save_path=None):
        df = self.get_feature_importance(top_n)
        colors = ['#e74c3c' if f.startswith('feat_') else '#e67e22'
                  for f in df['feature'][::-1]]
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.barh(df['feature'][::-1], df['importance'][::-1],
                color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Feature Importance (XGBoost gain)', fontsize=12)
        ax.set_title(f'XGBoost — Top {top_n} Features', fontsize=13)
        plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_confusion_matrix(self, save_path=None):
        cm = np.array(self.train_results.get('confusion_matrix', [[0,0],[0,0]]))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='YlOrRd')
        plt.colorbar(im)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white' if cm[i,j] > cm.max()/2 else 'black')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Non-Ransomware','Ransomware'], rotation=15)
        ax.set_yticklabels(['Non-Ransomware','Ransomware'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix — XGBoost', fontsize=13)
        plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def save(self, path='saved_models/xgboost.pkl',
             scaler_path='saved_models/xgb_scaler.pkl'):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump({'model': self.model, 'calibrated': self.calibrated,
                     'feature_names': self.feature_names, 'threshold': self.threshold}, path)
        joblib.dump(self.scaler, scaler_path)
        print(f"[XGB] Saved → {path}")

    def load(self, path='saved_models/xgboost.pkl',
             scaler_path='saved_models/xgb_scaler.pkl'):
        p = joblib.load(path)
        self.model, self.calibrated = p['model'], p.get('calibrated')
        self.feature_names, self.threshold = p['feature_names'], p['threshold']
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        print(f"[XGB] Loaded ← {path}")


# ══════════════════════════════════════════════════════════════════════════════
# LightGBM classifier
# ══════════════════════════════════════════════════════════════════════════════

class RansomwareLightGBM:
    """LightGBM classifier — often fastest and highest recall in benchmarks."""

    def __init__(
        self,
        n_estimators     : int   = 500,
        max_depth        : int   = 8,
        num_leaves       : int   = 63,
        learning_rate    : float = 0.05,
        subsample        : float = 0.8,
        colsample_bytree : float = 0.8,
        min_child_samples: int   = 20,
        reg_alpha        : float = 0.1,
        reg_lambda       : float = 0.1,
        recall_target    : float = 0.90,
        use_smote        : bool  = True,
        calibrate        : bool  = True,
        random_state     : int   = SEED,
        n_jobs           : int   = -1,
        verbose          : int   = -1,
    ):
        try:
            import lightgbm as lgb
            self._lgb = lgb
        except ImportError:
            raise ImportError("pip install lightgbm")

        self.params = dict(
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            num_leaves       = num_leaves,
            learning_rate    = learning_rate,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            min_child_samples= min_child_samples,
            reg_alpha        = reg_alpha,
            reg_lambda       = reg_lambda,
            random_state     = random_state,
            n_jobs           = n_jobs,
            verbose          = verbose,
            is_unbalance     = True,    # LightGBM's built-in imbalance handling
        )
        self.recall_target  = recall_target
        self.use_smote      = use_smote
        self.calibrate_flag = calibrate
        self.random_state   = random_state

        self.model         = None
        self.calibrated    = None
        self.scaler        : Optional[StandardScaler] = None
        self.feature_names : List[str]                = []
        self.threshold     : float                    = 0.5
        self.is_trained    : bool                     = False
        self.train_results : Dict[str, Any]           = {}

    def train(self, X, y, test_size=0.20, val_size=0.20, verbose=True):
        if verbose:
            print("\n" + "═"*60)
            print("  LIGHTGBM — TRAINING PIPELINE")
            print("═"*60)

        self.feature_names = list(X.columns)
        from collections import Counter

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_size/(1-test_size),
            stratify=y_tmp, random_state=self.random_state)

        if verbose:
            print(f"[LGB] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

        self.scaler = StandardScaler()
        Xtr = self.scaler.fit_transform(X_train)
        Xvl = self.scaler.transform(X_val)
        Xts = self.scaler.transform(X_test)

        n_pos = int((y_train == 1).sum())
        if self.use_smote and n_pos >= 5:
            if verbose: print(f"[LGB] Before SMOTE: {Counter(y_train)}")
            k = min(5, n_pos - 1)
            try:
                Xtr, y_train = SMOTE(random_state=self.random_state, k_neighbors=k).fit_resample(Xtr, y_train)
                if verbose: print(f"[LGB] After SMOTE : {Counter(y_train)}")
            except Exception as e:
                if verbose: print(f"[LGB] SMOTE skipped: {e}")

        if verbose: print("[LGB] Training LightGBM...")

        self.model = self._lgb.LGBMClassifier(**self.params)
        self.model.fit(
            Xtr, y_train,
            eval_set=[(Xvl, y_val)],
            callbacks=[self._lgb.early_stopping(50, verbose=False),
                       self._lgb.log_evaluation(-1)],
        )
        if verbose:
            print(f"[LGB] Best iteration: {self.model.best_iteration_}")

        if self.calibrate_flag and len(np.unique(y_val)) > 1:
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
                predict_fn = lambda X: self.calibrated.predict_proba(X)[:, 1]
            except Exception as e:
                if verbose:
                    print(f"[LGB] Calibration skipped: {e}")
                self.calibrated = None
                predict_fn = lambda X: self.model.predict_proba(X)[:, 1]
        else:
            self.calibrated = None
            predict_fn = lambda X: self.model.predict_proba(X)[:, 1]

        val_proba      = predict_fn(Xvl)
        self.threshold = self._tune_recall(val_proba, y_val, verbose)
        test_proba     = predict_fn(Xts)
        test_pred      = (test_proba >= self.threshold).astype(int)
        results        = self._metrics(y_test, test_pred, test_proba, "TEST SET", verbose)
        results.update({'X_test': Xts, 'y_test': y_test, 'test_proba': test_proba})
        self.train_results = results
        self.is_trained    = True
        return results

    def _tune_recall(self, proba, y_true, verbose=True):
        if len(np.unique(y_true)) < 2: return 0.5
        best_t, best_p = 0.5, 0.0
        for t in np.arange(0.05, 0.90, 0.01):
            pred = (proba >= t).astype(int)
            r = recall_score(y_true, pred, zero_division=0)
            p = precision_score(y_true, pred, zero_division=0)
            if r >= self.recall_target and p > best_p:
                best_t, best_p = t, p
        if best_p == 0: best_t = 0.10
        if verbose: print(f"[LGB] Recall-optimised threshold: {best_t:.4f}")
        return float(best_t)

    def _metrics(self, y_true, y_pred, y_proba, label="", verbose=True):
        n_cls = len(np.unique(y_true))
        if verbose:
            print(f"\n[LGB] ── {label} ──────────────────────────────────")
            print(classification_report(y_true, y_pred,
                  target_names=['Non-Ransomware','Ransomware'],
                  digits=4, zero_division=0))
        try:
            auc = roc_auc_score(y_true, y_proba) if n_cls > 1 else 0.0
            ap  = average_precision_score(y_true, y_proba) if n_cls > 1 else 0.0
        except Exception: auc = ap = 0.0
        if verbose and n_cls > 1: print(f"  ROC-AUC: {auc:.4f} | Avg Precision: {ap:.4f}")
        return {
            'accuracy': float(np.mean(y_pred == y_true)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(auc), 'avg_precision': float(ap),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }

    def predict(self, features, pid=0, process_name='unknown'):
        if not self.is_trained: raise RuntimeError("Call train() first.")
        fv = np.array([features.get(f, 0.0) for f in self.feature_names],
                       dtype=np.float32).reshape(1, -1)
        fv_sc = self.scaler.transform(fv)
        if self.calibrated:
            prob = float(self.calibrated.predict_proba(fv_sc)[0, 1])
        else:
            prob = float(self.model.predict_proba(fv_sc)[0, 1])
        return make_lightgbm_event(pid, process_name, prob, features)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        if self.calibrated: return self.calibrated.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, top_n=25):
        return pd.DataFrame({'feature': self.feature_names,
                              'importance': self.model.feature_importances_})\
               .sort_values('importance', ascending=False).head(top_n)

    def plot_confusion_matrix(self, save_path=None):
        cm = np.array(self.train_results.get('confusion_matrix', [[0,0],[0,0]]))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='PuBu')
        plt.colorbar(im)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white' if cm[i,j] > cm.max()/2 else 'black')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Non-Ransomware','Ransomware'], rotation=15)
        ax.set_yticklabels(['Non-Ransomware','Ransomware'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix — LightGBM', fontsize=13)
        plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def save(self, path='saved_models/lightgbm.pkl',
             scaler_path='saved_models/lgb_scaler.pkl'):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump({'model': self.model, 'calibrated': self.calibrated,
                     'feature_names': self.feature_names, 'threshold': self.threshold}, path)
        joblib.dump(self.scaler, scaler_path)
        print(f"[LGB] Saved → {path}")

    def load(self, path='saved_models/lightgbm.pkl',
             scaler_path='saved_models/lgb_scaler.pkl'):
        p = joblib.load(path)
        self.model, self.calibrated = p['model'], p.get('calibrated')
        self.feature_names, self.threshold = p['feature_names'], p['threshold']
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        print(f"[LGB] Loaded ← {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Weighted Soft-Voting Ensemble
# ══════════════════════════════════════════════════════════════════════════════

class RansomwareEnsemble:
    """
    Recall-oriented ensemble of RF + XGB + LGB + AE (optional).

    V7 combines weighted soft voting with a safety rule:
      final alert = weighted_score >= tuned_threshold
                    OR at least min_supervised_votes models cross their own tuned thresholds
                    OR any supervised model is very confident.

    This avoids the V6 failure where averaging suppressed recall even though the
    individual supervised models were all above 90% recall.
    """

    def __init__(
        self,
        weights     : Dict[str, float] = None,
        recall_target: float = 0.90,
        any_model_threshold: float = 0.85,
        min_supervised_votes: int = 2,
    ):
        # V7 defaults: rely mostly on the strongest supervised tabular models.
        # The autoencoder is useful as a novelty booster, but it should not drag
        # down the supervised ransomware recall in the main Layer 1 score.
        self.weights = weights or {
            'random_forest' : 0.45,
            'lightgbm'      : 0.40,
            'xgboost'       : 0.15,
            'autoencoder'   : 0.00,
        }
        self.recall_target = recall_target
        self.any_model_threshold = float(any_model_threshold)
        self.min_supervised_votes = int(min_supervised_votes)
        self.threshold     = 0.5
        self.models        : Dict[str, Any] = {}
        self.train_results : Dict[str, Any] = {}

    def add_model(self, name: str, model):
        """Register a trained model component."""
        self.models[name] = model

    def _weighted_proba(
        self,
        X_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute weighted ensemble probability.

        X_dict: maps model_name → pre-scaled feature array for that model.
        All scalers live inside individual model objects.
        """
        total_weight = 0.0
        proba_sum    = None

        for name, weight in self.weights.items():
            if name not in self.models:
                continue
            model = self.models[name]
            if not model.is_trained:
                continue

            X_for_model = X_dict.get(name)
            if X_for_model is None:
                continue

            if name == 'autoencoder':
                errors   = model.predict_errors_batch(X_for_model)
                # Normalise errors to [0, 1] probability
                max_err  = max(model.threshold * 5, np.percentile(errors, 99))
                p        = np.clip(errors / max_err, 0, 1)
            else:
                p = model.predict_proba_batch(X_for_model)

            if proba_sum is None:
                proba_sum = np.zeros(len(p))
            proba_sum    += weight * p
            total_weight += weight

        if proba_sum is None or total_weight == 0:
            raise RuntimeError("No valid models in ensemble.")

        return proba_sum / total_weight

    def _component_probas(self, X_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Return probability-like scores for every valid component."""
        scores: Dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            if not getattr(model, 'is_trained', False):
                continue
            X_for_model = X_dict.get(name)
            if X_for_model is None:
                continue
            if name == 'autoencoder':
                errors = model.predict_errors_batch(X_for_model)
                max_err = max(model.threshold * 5, np.percentile(errors, 99))
                scores[name] = np.clip(errors / max_err, 0, 1)
            else:
                scores[name] = model.predict_proba_batch(X_for_model)
        if not scores:
            raise RuntimeError("No valid models in ensemble.")
        return scores

    def _fusion_predict(self, weighted_proba: np.ndarray, component_scores: Dict[str, np.ndarray],
                        threshold: float) -> np.ndarray:
        """Recall-oriented V7 fusion rule.

        The weighted average gives a stable calibrated score. The vote/high-score
        rules prevent a single weak component from suppressing an otherwise strong
        ransomware signal. Autoencoder is excluded from supervised voting.
        """
        pred_weighted = weighted_proba >= threshold
        supervised = {k: v for k, v in component_scores.items() if k != 'autoencoder'}
        if not supervised:
            return pred_weighted.astype(int)

        vote_count = np.zeros_like(weighted_proba, dtype=int)
        max_score = np.zeros_like(weighted_proba, dtype=float)
        for name, scores in supervised.items():
            model = self.models.get(name)
            model_t = float(getattr(model, 'threshold', 0.5))
            vote_count += (scores >= model_t).astype(int)
            max_score = np.maximum(max_score, scores)

        pred_votes = vote_count >= self.min_supervised_votes
        pred_any_high = max_score >= self.any_model_threshold
        return (pred_weighted | pred_votes | pred_any_high).astype(int)

    def evaluate(
        self,
        X           : pd.DataFrame,
        y           : pd.Series,
        test_size   : float = 0.20,
        val_size    : float = 0.20,
        verbose     : bool  = True,
    ) -> Dict[str, Any]:
        """
        Evaluate the ensemble with validation-only threshold tuning.

        Previous versions tuned the ensemble threshold directly on the test set,
        which leaks test information into the final metric. This method now uses:

            X/y -> validation split for threshold -> untouched test split

        The component models still own their own scalers and calibrated outputs.
        """
        from sklearn.model_selection import train_test_split

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42)
        val_frac = val_size / max(1.0 - test_size, 1e-9)
        _, X_val, _, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=43)

        def _scale_for_models(X_part):
            X_dict = {}
            for name, model in self.models.items():
                if hasattr(model, 'scaler') and model.scaler is not None:
                    X_dict[name] = model.scaler.transform(X_part)
            return X_dict

        # Tune threshold on validation set only, using the same V7 fusion rule
        # that will be used on the untouched test set.
        Xv_dict = _scale_for_models(X_val)
        val_proba = self._weighted_proba(Xv_dict)
        val_components = self._component_probas(Xv_dict)
        self.threshold = self._tune_recall(val_proba, y_val, verbose, val_components)

        # Evaluate once on untouched test set.
        Xt_dict = _scale_for_models(X_test)
        proba = self._weighted_proba(Xt_dict)
        test_components = self._component_probas(Xt_dict)
        pred  = self._fusion_predict(proba, test_components, self.threshold)

        n_cls = len(np.unique(y_test))
        if verbose:
            print("\n[ENS] ── ENSEMBLE TEST SET (threshold tuned on validation) ──")
            print(classification_report(y_test, pred,
                  target_names=['Non-Ransomware','Ransomware'],
                  digits=4, zero_division=0))

        try:
            auc = roc_auc_score(y_test, proba) if n_cls > 1 else 0.0
            ap  = average_precision_score(y_test, proba) if n_cls > 1 else 0.0
        except Exception:
            auc = ap = 0.0

        if verbose and n_cls > 1:
            print(f"  ROC-AUC: {auc:.4f} | Avg Precision: {ap:.4f}")

        results = {
            'accuracy'  : float(np.mean(pred == y_test)),
            'precision' : float(precision_score(y_test, pred, zero_division=0)),
            'recall'    : float(recall_score(y_test, pred, zero_division=0)),
            'f1'        : float(f1_score(y_test, pred, zero_division=0)),
            'roc_auc'   : float(auc),
            'avg_precision': float(ap),
            'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
            'y_test'    : y_test,
            'test_proba': proba,
            'threshold' : self.threshold,
            'threshold_tuned_on': 'validation',
            'fusion_rule': 'weighted_or_two_model_votes_or_any_high',
            'any_model_threshold': self.any_model_threshold,
            'min_supervised_votes': self.min_supervised_votes,
        }
        self.train_results = results
        return results

    def _tune_recall(self, proba, y_true, verbose=True, component_scores=None):
        if len(np.unique(y_true)) < 2:
            return 0.5
        best_t, best_p, best_r = 0.5, -1.0, 0.0
        for t in np.arange(0.01, 0.95, 0.005):
            if component_scores is not None:
                pred = self._fusion_predict(proba, component_scores, t)
            else:
                pred = (proba >= t).astype(int)
            r = recall_score(y_true, pred, zero_division=0)
            p = precision_score(y_true, pred, zero_division=0)
            if r >= self.recall_target and p > best_p:
                best_t, best_p, best_r = float(t), float(p), float(r)
        if best_p < 0:
            best_t = 0.10
            if component_scores is not None:
                pred = self._fusion_predict(proba, component_scores, best_t)
            else:
                pred = (proba >= best_t).astype(int)
            best_r = recall_score(y_true, pred, zero_division=0)
            best_p = precision_score(y_true, pred, zero_division=0)
        if verbose:
            print(f"[ENS] V7 recall-optimised threshold: {best_t:.4f} "
                  f"(val_recall={best_r:.3f}, val_precision={best_p:.3f}, "
                  f"rule=weighted OR {self.min_supervised_votes} model votes OR any≥{self.any_model_threshold:.2f})")
        return float(best_t)

    def predict(self, features, pid=0, process_name='unknown'):
        """Single-sample prediction returning SystemEvent."""
        X_dict = {}
        component_scores = {}
        for name, model in self.models.items():
            if not model.is_trained: continue
            fv = np.array([features.get(f, 0.0) for f in model.feature_names],
                           dtype=np.float32).reshape(1, -1)
            fv_sc = model.scaler.transform(fv)
            if name == 'autoencoder':
                err = float(model.predict_errors_batch(fv_sc)[0])
                max_err = max(model.threshold * 5, 0.001)
                component_scores[name] = float(np.clip(err / max_err, 0, 1))
                X_dict[name] = fv_sc
            else:
                component_scores[name] = float(model.predict_proba_batch(fv_sc)[0])
                X_dict[name] = fv_sc

        w_sum = sum(self.weights.get(n, 0) for n in component_scores)
        weighted = sum(self.weights.get(n, 0) * s
                       for n, s in component_scores.items()) / max(w_sum, 1e-9)

        supervised_scores = {k: v for k, v in component_scores.items() if k != 'autoencoder'}
        votes = sum(
            1 for name, score in supervised_scores.items()
            if score >= float(getattr(self.models[name], 'threshold', 0.5))
        )
        max_supervised = max(supervised_scores.values()) if supervised_scores else weighted
        final = max(weighted,
                    self.threshold if votes >= self.min_supervised_votes else 0.0,
                    self.threshold if max_supervised >= self.any_model_threshold else 0.0)
        component_scores['_weighted_score'] = float(weighted)
        component_scores['_supervised_votes'] = float(votes)
        component_scores['_max_supervised'] = float(max_supervised)

        return make_ensemble_event(pid, process_name, final, component_scores, features)

    def plot_combined_roc(
        self,
        individual_results: Dict[str, Dict],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot ROC curves for all individual models + ensemble on one plot."""
        colors = {
            'random_forest' : '#2980b9',
            'xgboost'       : '#e67e22',
            'lightgbm'      : '#27ae60',
            'autoencoder'   : '#8e44ad',
            'ensemble'      : '#c0392b',
        }
        fig, ax = plt.subplots(figsize=(9, 7))

        for name, res in individual_results.items():
            y_t = res.get('y_test')
            p   = res.get('test_proba')
            if y_t is None or p is None or len(np.unique(y_t)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_t, p)
            auc = res.get('roc_auc', roc_auc_score(y_t, p))
            ax.plot(fpr, tpr, lw=2, color=colors.get(name, 'gray'),
                    label=f'{name.title()} (AUC={auc:.4f})')

        # Ensemble
        ens = self.train_results
        y_t = ens.get('y_test')
        p   = ens.get('test_proba')
        if y_t is not None and p is not None and len(np.unique(y_t)) > 1:
            fpr, tpr, _ = roc_curve(y_t, p)
            auc = ens.get('roc_auc', 0)
            ax.plot(fpr, tpr, lw=3.5, color=colors['ensemble'], linestyle='-.',
                    label=f'Ensemble (AUC={auc:.4f})')

        ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random baseline')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax.set_title('ROC Curves — All Models + Ensemble', fontsize=13)
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
