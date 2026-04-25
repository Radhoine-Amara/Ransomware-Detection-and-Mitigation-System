"""
train.py  —  S004 Ransomware Detection System
═══════════════════════════════════════════════════════════════════════════════
Main training orchestrator. Trains all models in sequence, runs evaluation,
saves checkpoints, and writes a summary report.

Usage:
    # Train all models with default config
    python train.py

    # Train specific model only
    python train.py --model rf
    python train.py --model xgb
    python train.py --model lgb
    python train.py --model autoencoder
    python train.py --model isolation_forest
    python train.py --model ensemble   # trains all then combines

    # Use best hyperparameters from tuning
    python train.py --use_tuned_params

    # Skip models whose saved artifacts already exist
    python train.py --skip_existing

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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config import (
    CICMALMEM_PATHS, RANDOM_SEED, MODELS_DIR, REPORTS_DIR, PLOTS_DIR,
    RF_CONFIG, XGB_CONFIG, LGB_CONFIG, IF_CONFIG, AE_CONFIG, ENSEMBLE_WEIGHTS,
    TEST_SIZE, VAL_SIZE,
)
from utils  import set_seed, get_logger, Timer, check_class_balance, format_metrics
from models.data_loader      import load_cicmalmem, get_benign_mask, get_category_series
from models.random_forest_model   import RansomwareRandomForest
from models.isolation_forest_model import RansomwareIsolationForest
from models.autoencoder_model     import RansomwareAutoencoder
from models.ensemble_model        import RansomwareXGBoost, RansomwareLightGBM, RansomwareEnsemble
from evaluate import run_full_evaluation, build_comparison_table, security_metrics

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

LOG_FILE = os.path.join(REPORTS_DIR, 'train.log')
logger   = get_logger('train', log_file=LOG_FILE)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data(verbose: bool = True):
    """
    Load CIC-MalMem2022, engineer features, return X, y, benign_mask, categories.
    """
    logger.info("Loading CIC-MalMem2022...")
    with Timer("Data loading"):
        X, y = load_cicmalmem(
            file_paths  = CICMALMEM_PATHS,
            verbose     = verbose,
            add_features= True,   # adds 15 engineered features → 70 total
        )
        benign_mask = get_benign_mask(file_paths=CICMALMEM_PATHS)
        categories  = get_category_series(file_paths=CICMALMEM_PATHS)

    check_class_balance(y, "CIC-MalMem2022 (after engineering)")

    # Raw features only (for ablation study)
    raw_cols = [c for c in X.columns if not c.startswith('feat_')]
    X_raw    = X[raw_cols].copy()

    logger.info(f"Dataset ready: {X.shape[0]:,} samples, {X.shape[1]} features")
    return X, y, X_raw, benign_mask, categories


# ══════════════════════════════════════════════════════════════════════════════
# Individual model trainers
# ══════════════════════════════════════════════════════════════════════════════

def train_random_forest(
    X, y,
    config        = None,
    use_tuned     = False,
    skip_existing = False,
) -> RansomwareRandomForest:
    model_path  = os.path.join(MODELS_DIR, 'random_forest.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'rf_scaler.pkl')

    if skip_existing and os.path.exists(model_path):
        logger.info("[RF] Skipping — saved model found. Loading...")
        rf = RansomwareRandomForest()
        rf.load(model_path, scaler_path)
        return rf

    cfg = config or RF_CONFIG.copy()

    if use_tuned:
        try:
            from hyperparameter_tuning import load_best_params
            best = load_best_params('rf')
            cfg.update(best)
            logger.info(f"[RF] Using tuned params: {best}")
        except FileNotFoundError:
            logger.warning("[RF] No tuned params found — using defaults.")

    logger.info("[RF] Starting training...")
    rf = RansomwareRandomForest(**cfg)
    rf.feature_names = list(X.columns)

    with Timer("RF training"):
        results = rf.train(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE)

    logger.info(f"[RF] {format_metrics(results)}")

    rf.save(model_path, scaler_path)
    rf.plot_feature_importance(
        top_n=25, save_path=os.path.join(PLOTS_DIR, 'rf_feature_importance.png'))
    rf.plot_roc_curve(save_path=os.path.join(PLOTS_DIR, 'rf_roc_curve.png'))
    rf.plot_confusion_matrix(save_path=os.path.join(PLOTS_DIR, 'rf_confusion_matrix.png'))
    rf.plot_precision_recall_curve(save_path=os.path.join(PLOTS_DIR, 'rf_pr_curve.png'))

    return rf


def train_xgboost(
    X, y,
    config        = None,
    use_tuned     = False,
    skip_existing = False,
) -> RansomwareXGBoost:
    model_path  = os.path.join(MODELS_DIR, 'xgboost.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'xgb_scaler.pkl')

    if skip_existing and os.path.exists(model_path):
        logger.info("[XGB] Skipping — saved model found. Loading...")
        xgb = RansomwareXGBoost()
        xgb.load(model_path, scaler_path)
        return xgb

    cfg = config or XGB_CONFIG.copy()
    if use_tuned:
        try:
            from hyperparameter_tuning import load_best_params
            best = load_best_params('xgb')
            cfg.update(best)
            logger.info(f"[XGB] Using tuned params: {best}")
        except FileNotFoundError:
            logger.warning("[XGB] No tuned params found — using defaults.")

    logger.info("[XGB] Starting training...")
    xgb = RansomwareXGBoost(**cfg)
    xgb.feature_names = list(X.columns)

    with Timer("XGB training"):
        results = xgb.train(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE)

    logger.info(f"[XGB] {format_metrics(results)}")
    xgb.save(model_path, scaler_path)
    xgb.plot_feature_importance(save_path=os.path.join(PLOTS_DIR, 'xgb_feature_importance.png'))
    xgb.plot_confusion_matrix(save_path=os.path.join(PLOTS_DIR, 'xgb_confusion_matrix.png'))
    return xgb


def train_lightgbm(
    X, y,
    config        = None,
    use_tuned     = False,
    skip_existing = False,
) -> RansomwareLightGBM:
    model_path  = os.path.join(MODELS_DIR, 'lightgbm.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'lgb_scaler.pkl')

    if skip_existing and os.path.exists(model_path):
        logger.info("[LGB] Skipping — saved model found. Loading...")
        lgb = RansomwareLightGBM()
        lgb.load(model_path, scaler_path)
        return lgb

    cfg = config or LGB_CONFIG.copy()
    if use_tuned:
        try:
            from hyperparameter_tuning import load_best_params
            best = load_best_params('lgb')
            cfg.update(best)
            logger.info(f"[LGB] Using tuned params: {best}")
        except FileNotFoundError:
            logger.warning("[LGB] No tuned params found — using defaults.")

    logger.info("[LGB] Starting training...")
    lgb = RansomwareLightGBM(**cfg)
    lgb.feature_names = list(X.columns)

    with Timer("LGB training"):
        results = lgb.train(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE)

    logger.info(f"[LGB] {format_metrics(results)}")
    lgb.save(model_path, scaler_path)
    lgb.plot_confusion_matrix(save_path=os.path.join(PLOTS_DIR, 'lgb_confusion_matrix.png'))
    return lgb


def train_isolation_forest(
    X, y, benign_mask,
    config        = None,
    skip_existing = False,
) -> RansomwareIsolationForest:
    model_path  = os.path.join(MODELS_DIR, 'isolation_forest.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'if_scaler.pkl')

    if skip_existing and os.path.exists(model_path):
        logger.info("[IF] Skipping — saved model found. Loading...")
        iso = RansomwareIsolationForest()
        iso.load(model_path, scaler_path)
        return iso

    cfg = config or IF_CONFIG.copy()
    logger.info("[IF] Starting training (Benign-only)...")
    iso = RansomwareIsolationForest(**cfg)
    iso.feature_names = list(X.columns)

    with Timer("IF training"):
        results = iso.train(X, y, benign_mask=benign_mask, test_size=TEST_SIZE)

    logger.info(f"[IF] {format_metrics(results)}")
    iso.save(model_path, scaler_path)
    iso.plot_score_distribution(save_path=os.path.join(PLOTS_DIR, 'if_score_distribution.png'))
    iso.plot_confusion_matrix(save_path=os.path.join(PLOTS_DIR, 'if_confusion_matrix.png'))
    return iso


def train_autoencoder(
    X, y, benign_mask,
    config        = None,
    skip_existing = False,
) -> RansomwareAutoencoder:
    model_dir   = os.path.join(MODELS_DIR, 'autoencoder')
    scaler_path = os.path.join(MODELS_DIR, 'ae_scaler.pkl')

    if skip_existing and os.path.exists(os.path.join(model_dir, 'meta.json')):
        logger.info("[AE] Skipping — saved model found. Loading...")
        ae = RansomwareAutoencoder()
        ae.load(model_dir, scaler_path)
        return ae

    cfg = config or AE_CONFIG.copy()
    logger.info("[AE] Starting training (Benign-only, deep autoencoder)...")

    try:
        ae = RansomwareAutoencoder(**cfg)
        ae.feature_names = list(X.columns)
        ae.n_features    = len(X.columns)

        with Timer("AE training"):
            results = ae.train(X, y, benign_mask=benign_mask, test_size=TEST_SIZE)

        logger.info(f"[AE] {format_metrics(results)}")
        ae.save(model_dir, scaler_path)
        ae.plot_training_history(save_path=os.path.join(PLOTS_DIR, 'ae_training_history.png'))
        ae.plot_error_distribution(save_path=os.path.join(PLOTS_DIR, 'ae_error_distribution.png'))
        ae.plot_confusion_matrix(save_path=os.path.join(PLOTS_DIR, 'ae_confusion_matrix.png'))
        return ae

    except RuntimeError as e:
        logger.warning(f"[AE] Skipped: {e}")
        return None


def train_ensemble(
    X, y,
    trained_models: dict,
) -> RansomwareEnsemble:
    logger.info("[ENS] Building ensemble from trained components...")

    ens = RansomwareEnsemble(
        weights      = ENSEMBLE_WEIGHTS,
        recall_target= 0.90,
    )
    for name, model in trained_models.items():
        if model is not None:
            ens.add_model(name, model)

    with Timer("Ensemble evaluation"):
        results = ens.evaluate(X, y, test_size=TEST_SIZE)

    logger.info(f"[ENS] {format_metrics(results)}")
    return ens


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    set_seed(RANDOM_SEED)
    logger.info("=" * 60)
    logger.info("  S004 RANSOMWARE DETECTION — TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    X, y, X_raw, benign_mask, categories = load_data(verbose=True)

    train_all = (args.model == 'all')
    trained   = {}

    # ── Random Forest ──────────────────────────────────────────────────────
    if train_all or args.model == 'rf':
        rf = train_random_forest(
            X, y,
            use_tuned     = args.use_tuned_params,
            skip_existing = args.skip_existing,
        )
        trained['random_forest'] = rf

    # ── XGBoost ───────────────────────────────────────────────────────────
    if train_all or args.model == 'xgb':
        try:
            xgb = train_xgboost(
                X, y,
                use_tuned     = args.use_tuned_params,
                skip_existing = args.skip_existing,
            )
            trained['xgboost'] = xgb
        except ImportError:
            logger.warning("[XGB] xgboost not installed. pip install xgboost")

    # ── LightGBM ──────────────────────────────────────────────────────────
    if train_all or args.model == 'lgb':
        try:
            lgb = train_lightgbm(
                X, y,
                use_tuned     = args.use_tuned_params,
                skip_existing = args.skip_existing,
            )
            trained['lightgbm'] = lgb
        except ImportError:
            logger.warning("[LGB] lightgbm not installed. pip install lightgbm")

    # ── Isolation Forest ──────────────────────────────────────────────────
    if train_all or args.model == 'isolation_forest':
        iso = train_isolation_forest(
            X, y, benign_mask,
            skip_existing = args.skip_existing,
        )
        trained['isolation_forest'] = iso

    # ── Deep Autoencoder ──────────────────────────────────────────────────
    if train_all or args.model == 'autoencoder':
        ae = train_autoencoder(
            X, y, benign_mask,
            skip_existing = args.skip_existing,
        )
        if ae is not None:
            trained['autoencoder'] = ae

    # ── Ensemble ──────────────────────────────────────────────────────────
    if train_all or args.model == 'ensemble':
        if len(trained) >= 2:
            ens = train_ensemble(X, y, trained)
            trained['ensemble'] = ens
        else:
            logger.warning("[ENS] Need ≥2 trained models for ensemble.")

    # ── Full evaluation ────────────────────────────────────────────────────
    if trained:
        logger.info("\n" + "=" * 60)
        logger.info("  RUNNING FULL EVALUATION")
        logger.info("=" * 60)

        # Separate display name mapping
        display_names = {
            'random_forest'   : 'Random Forest',
            'xgboost'         : 'XGBoost',
            'lightgbm'        : 'LightGBM',
            'isolation_forest': 'Isolation Forest',
            'autoencoder'     : 'Autoencoder',
            'ensemble'        : 'Ensemble',
        }
        display_trained = {display_names.get(k, k): v for k, v in trained.items()}

        run_full_evaluation(
            trained_models = display_trained,
            X              = X,
            y              = y,
            X_raw          = X_raw,
            save_dir       = PLOTS_DIR,
        )

    logger.info("\n✓ Training complete. All artifacts saved to:")
    logger.info(f"  Models  → {MODELS_DIR}")
    logger.info(f"  Plots   → {PLOTS_DIR}")
    logger.info(f"  Reports → {REPORTS_DIR}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S004 Training Orchestrator')
    parser.add_argument(
        '--model', type=str, default='all',
        choices=['all', 'rf', 'xgb', 'lgb',
                 'isolation_forest', 'autoencoder', 'ensemble'],
        help='Which model to train (default: all)'
    )
    parser.add_argument(
        '--use_tuned_params', action='store_true',
        help='Load best params from hyperparameter_tuning.py output'
    )
    parser.add_argument(
        '--skip_existing', action='store_true',
        help='Skip training if saved model already exists'
    )
    args = parser.parse_args()
    main(args)
