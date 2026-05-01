"""
hyperparameter_tuning.py  —  S004 Ransomware Detection System
═══════════════════════════════════════════════════════════════════════════════
Optuna-based hyperparameter optimisation for all supervised models.

Optimisation objective: RECALL on a stratified validation fold.
(Never F1 or accuracy — recall is the security-critical metric.)

Usage:
    python hyperparameter_tuning.py --model rf  --trials 50
    python hyperparameter_tuning.py --model xgb --trials 100
    python hyperparameter_tuning.py --model lgb --trials 100

Author : AI Engineering Student
Phase  : 3 — ML Models (v2, research-grade)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

from typing import Dict, Any, Optional
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import recall_score
from imblearn.over_sampling  import SMOTE

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config      import CICMALMEM_PATHS, RANDOM_SEED, REPORTS_DIR
from models.data_loader import load_cicmalmem

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[HPO] Optuna not installed. Run: pip install optuna")


# ══════════════════════════════════════════════════════════════════════════════
# Objective functions
# ══════════════════════════════════════════════════════════════════════════════

def rf_objective(trial, X_scaled: np.ndarray, y: np.ndarray) -> float:
    """Random Forest objective for Optuna — maximise recall."""
    from sklearn.ensemble import RandomForestClassifier

    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 800),
        'max_depth'        : trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 6),
        'max_features'     : trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
        'class_weight'     : 'balanced',
        'n_jobs'           : -1,
        'random_state'     : RANDOM_SEED,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(
        RandomForestClassifier(**params),
        X_scaled, y,
        cv=cv, scoring='recall', n_jobs=-1
    )
    return float(scores.mean())


def xgb_objective(trial, X_scaled: np.ndarray, y: np.ndarray) -> float:
    """XGBoost objective — maximise recall."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("pip install xgboost")

    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    spw   = n_neg / max(n_pos, 1)

    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 800),
        'max_depth'        : trial.suggest_int('max_depth', 3, 10),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
        'gamma'            : trial.suggest_float('gamma', 0, 1.0),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'scale_pos_weight' : spw,
        'eval_metric'      : 'aucpr',
        'use_label_encoder': False,
        'n_jobs'           : -1,
        'random_state'     : RANDOM_SEED,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(
        XGBClassifier(**params),
        X_scaled, y,
        cv=cv, scoring='recall', n_jobs=-1
    )
    return float(scores.mean())


def lgb_objective(trial, X_scaled: np.ndarray, y: np.ndarray) -> float:
    """LightGBM objective — maximise recall."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("pip install lightgbm")

    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 800),
        'max_depth'        : trial.suggest_int('max_depth', 3, 12),
        'num_leaves'       : trial.suggest_int('num_leaves', 15, 127),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'is_unbalance'     : True,
        'random_state'     : RANDOM_SEED,
        'n_jobs'           : -1,
        'verbose'          : -1,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(
        LGBMClassifier(**params),
        X_scaled, y,
        cv=cv, scoring='recall', n_jobs=-1
    )
    return float(scores.mean())


# ══════════════════════════════════════════════════════════════════════════════
# Main tuning runner
# ══════════════════════════════════════════════════════════════════════════════

def tune(
    model_name  : str,
    n_trials    : int = 50,
    file_paths  : Optional[list] = None,
    save_dir    : str = None,
    verbose     : bool = True,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter search for a given model.

    Parameters
    ──────────
    model_name : 'rf', 'xgb', or 'lgb'
    n_trials   : Number of Optuna trials
    file_paths : CIC-MalMem2022 CSV paths (defaults to config)
    save_dir   : Directory to save best params JSON

    Returns dict with best_params and best_recall.
    """
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("pip install optuna")

    if save_dir is None:
        save_dir = REPORTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  HYPERPARAMETER OPTIMISATION — {model_name.upper()}")
        print(f"  Trials: {n_trials}  |  Objective: Recall (5-fold CV)")
        print(f"{'═'*60}")

    # ── Load data ──────────────────────────────────────────────────────────
    X, y = load_cicmalmem(file_paths=file_paths, verbose=False)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_arr    = y.values

    # Apply SMOTE before CV to balance training folds
    if y_arr.sum() >= 5:
        k = min(5, int(y_arr.sum()) - 1)
        try:
            X_scaled, y_arr = SMOTE(
                random_state=RANDOM_SEED, k_neighbors=k
            ).fit_resample(X_scaled, y_arr)
            if verbose:
                from collections import Counter
                print(f"[HPO] After SMOTE: {Counter(y_arr)}")
        except Exception as e:
            if verbose:
                print(f"[HPO] SMOTE skipped: {e}")

    # ── Objective dispatch ─────────────────────────────────────────────────
    objectives = {
        'rf' : rf_objective,
        'xgb': xgb_objective,
        'lgb': lgb_objective,
    }
    if model_name not in objectives:
        raise ValueError(f"model_name must be one of {list(objectives.keys())}")

    objective_fn = lambda trial: objectives[model_name](trial, X_scaled, y_arr)

    # ── Run study ──────────────────────────────────────────────────────────
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    if verbose:
        print(f"[HPO] Running {n_trials} trials...")

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        show_progress_bar=verbose,
    )

    best_params = study.best_params
    best_recall = study.best_value

    if verbose:
        print(f"\n[HPO] Best Recall (5-fold CV): {best_recall:.4f}")
        print(f"[HPO] Best params:")
        for k, v in best_params.items():
            print(f"  {k:<25}: {v}")

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        'model'       : model_name,
        'n_trials'    : n_trials,
        'best_recall' : best_recall,
        'best_params' : best_params,
        'all_trials'  : [
            {'number': t.number, 'value': t.value, 'params': t.params}
            for t in study.trials if t.value is not None
        ],
    }

    save_path = os.path.join(save_dir, f'best_params_{model_name}.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\n[HPO] Results saved → {save_path}")

    return results


def load_best_params(model_name: str, save_dir: str = None) -> Dict[str, Any]:
    """Load previously saved best params."""
    if save_dir is None:
        save_dir = REPORTS_DIR
    path = os.path.join(save_dir, f'best_params_{model_name}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No tuning results found at {path}. "
            f"Run: python hyperparameter_tuning.py --model {model_name}"
        )
    with open(path) as f:
        data = json.load(f)
    return data['best_params']


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='S004 Hyperparameter Optimisation (Optuna)'
    )
    parser.add_argument('--model',  type=str, required=True,
                        choices=['rf', 'xgb', 'lgb'],
                        help='Model to tune')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results JSON')
    args = parser.parse_args()

    tune(
        model_name = args.model,
        n_trials   = args.trials,
        save_dir   = args.save_dir,
    )
