"""
config.py  —  S004 Ransomware Detection System
═══════════════════════════════════════════════════════════════════════════════
Central configuration for all model hyperparameters, paths, and evaluation
settings. Edit this file to change any setting — all scripts import from here.

Author : AI Engineering Student
Phase  : 3 — ML Models (v2, research-grade)
"""

from __future__ import annotations
import os

# ── Project root ───────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..'))

# ── Dataset paths ──────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'datasets')
REPORTS_DIR  = os.path.join(PROJECT_ROOT, 'reports')
PLOTS_DIR    = os.path.join(PROJECT_ROOT, 'reports', 'plots')
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'saved_models')

# CIC-MalMem2022 — add more paths here when you download additional files
CICMALMEM_PATHS = [
    os.path.join(DATA_DIR, 'MalMem2022.csv'),        # main merged file
    os.path.join(DATA_DIR, 'Output1.csv'),
    os.path.join(DATA_DIR, 'output2.csv'),
    os.path.join(DATA_DIR, 'output3.csv'),
    # Add Ransomware-only CSVs here when downloaded:
    # os.path.join(DATA_DIR, 'Ransomware_extra.csv'),
]

UNSW_TRAIN_PATH = os.path.join(
    DATA_DIR, 'UNSW-NB15', 'Training and Testing Sets',
    'UNSW_NB15_training-set.csv'
)
UNSW_TEST_PATH = os.path.join(
    DATA_DIR, 'UNSW-NB15', 'Training and Testing Sets',
    'UNSW_NB15_testing-set.csv'
)

# ── Global reproducibility ─────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Train / Val / Test split ratios ───────────────────────────────────────────
TEST_SIZE = 0.20   # 20% held-out test set (never touched during training)
VAL_SIZE  = 0.20   # 20% validation for threshold tuning + calibration

# ── Feature engineering ────────────────────────────────────────────────────────
ADD_ENGINEERED_FEATURES = True   # add 15 ratio/interaction features

# ── Class imbalance ────────────────────────────────────────────────────────────
ACTUAL_RANSOMWARE_RATIO = 0.165   # 16.5% of MalMem2022 is Ransomware
USE_SMOTE               = True

# ══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Random Forest ─────────────────────────────────────────────────────────────
RF_CONFIG = {
    'n_estimators'      : 500,
    'max_depth'         : None,
    'min_samples_split' : 4,
    'min_samples_leaf'  : 2,
    'max_features'      : 'sqrt',
    'recall_target'     : 0.90,   # threshold tuned to achieve this recall
    'use_smote'         : True,
    'calibrate'         : True,   # isotonic probability calibration
    'random_state'      : RANDOM_SEED,
}

# ── XGBoost ───────────────────────────────────────────────────────────────────
XGB_CONFIG = {
    'n_estimators'      : 500,
    'max_depth'         : 6,
    'learning_rate'     : 0.05,
    'subsample'         : 0.8,
    'colsample_bytree'  : 0.8,
    'min_child_weight'  : 3,
    'gamma'             : 0.1,
    'reg_alpha'         : 0.1,
    'reg_lambda'        : 1.0,
    'eval_metric'       : 'aucpr',   # area under PR curve — better for imbalance
    'use_label_encoder' : False,
    'recall_target'     : 0.90,
    'random_state'      : RANDOM_SEED,
    'n_jobs'            : -1,
}

# ── LightGBM ──────────────────────────────────────────────────────────────────
LGB_CONFIG = {
    'n_estimators'     : 500,
    'max_depth'        : 8,
    'num_leaves'       : 63,
    'learning_rate'    : 0.05,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
    'min_child_samples': 20,
    'reg_alpha'        : 0.1,
    'reg_lambda'       : 0.1,
    'recall_target'    : 0.90,
    'random_state'     : RANDOM_SEED,
    'n_jobs'           : -1,
    'verbose'          : -1,
}

# ── Isolation Forest ──────────────────────────────────────────────────────────
IF_CONFIG = {
    'n_estimators'  : 300,
    'contamination' : ACTUAL_RANSOMWARE_RATIO,
    'max_samples'   : 'auto',
    'recall_target' : 0.80,   # softer target — IF is for zero-days
    'random_state'  : RANDOM_SEED,
}

# ── Deep Autoencoder ──────────────────────────────────────────────────────────
AE_CONFIG = {
    'encoder_dims'    : [128, 64, 32],   # encoder layer sizes
    'latent_dim'      : 16,              # bottleneck dimension
    'dropout_rate'    : 0.3,
    'learning_rate'   : 5e-4,
    'batch_size'      : 256,
    'epochs'          : 100,
    'patience'        : 15,              # EarlyStopping patience
    'threshold_sigma' : 2.5,            # threshold = mean + sigma*std of train errors
    'recall_target'   : 0.85,
}

# ── Ensemble weights (for soft voting) ────────────────────────────────────────
ENSEMBLE_WEIGHTS = {
    'random_forest'    : 0.35,
    'xgboost'          : 0.35,
    'lightgbm'         : 0.20,
    'autoencoder'      : 0.10,
}

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# Minimum recall threshold below which a model is flagged as insufficient
MIN_ACCEPTABLE_RECALL = 0.88

# Cross-validation settings
CV_FOLDS   = 5
CV_SCORING = 'recall'   # primary CV metric

# ── SystemEvent thresholds ────────────────────────────────────────────────────
SEVERITY_THRESHOLDS = {
    'CRITICAL' : 0.90,   # confidence >= 90% → immediate process kill
    'HIGH'     : 0.75,   # confidence >= 75% → suspend + alert
    'MEDIUM'   : 0.50,   # confidence >= 50% → alert + monitor
}
