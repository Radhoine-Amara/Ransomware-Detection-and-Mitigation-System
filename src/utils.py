"""
utils.py  —  S004 Ransomware Detection System
═══════════════════════════════════════════════════════════════════════════════
Shared utility functions used across training, evaluation and the notebook.

Author : AI Engineering Student
Phase  : 3 — ML Models (v2, research-grade)
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility across numpy, random, and TF.
    Call this at the very start of train.py and the notebook.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ── Logging ────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a logger that writes to stdout and optionally to a file.

    Usage:
        logger = get_logger('train', log_file='logs/train.log')
        logger.info("Training started")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        '%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Timer context manager ──────────────────────────────────────────────────────

class Timer:
    """
    Simple context manager for timing code blocks.

    Usage:
        with Timer("RF training") as t:
            model.train(X, y)
        print(f"Training took {t.elapsed:.1f}s")
    """
    def __init__(self, label: str = "", verbose: bool = True):
        self.label   = label
        self.verbose = verbose
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.verbose and self.label:
            mins = int(self.elapsed // 60)
            secs = self.elapsed % 60
            if mins > 0:
                print(f"[Timer] {self.label}: {mins}m {secs:.1f}s")
            else:
                print(f"[Timer] {self.label}: {secs:.2f}s")


# ── Data utilities ─────────────────────────────────────────────────────────────

def check_class_balance(y: pd.Series, name: str = "dataset") -> Dict[str, Any]:
    """
    Print and return class balance statistics.

    Returns dict with: n_total, n_pos, n_neg, ratio, pct_positive
    """
    n_total = len(y)
    n_pos   = int(y.sum())
    n_neg   = n_total - n_pos
    ratio   = n_neg / max(n_pos, 1)
    pct_pos = n_pos / max(n_total, 1) * 100

    print(f"\n[Balance] {name}")
    print(f"  Total     : {n_total:,}")
    print(f"  Ransomware: {n_pos:,}  ({pct_pos:.1f}%)")
    print(f"  Other     : {n_neg:,}  ({100-pct_pos:.1f}%)")
    print(f"  Ratio     : {ratio:.1f}:1  ", end="")
    if ratio > 10:
        print("⚠ Severe imbalance — SMOTE strongly recommended")
    elif ratio > 3:
        print("Moderate imbalance — SMOTE recommended")
    else:
        print("Mild imbalance — class_weight='balanced' sufficient")

    return dict(n_total=n_total, n_pos=n_pos, n_neg=n_neg,
                ratio=ratio, pct_positive=pct_pos)


def check_missing(X: pd.DataFrame, threshold_pct: float = 50.0) -> List[str]:
    """
    Report missing values. Returns list of columns exceeding threshold_pct missing.
    Those columns should be dropped.
    """
    missing = X.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("[Missing] No missing values.")
        return []

    print(f"\n[Missing] {len(missing)} columns have missing values:")
    to_drop = []
    for col, n in missing.items():
        pct = n / len(X) * 100
        flag = " ← DROP" if pct > threshold_pct else ""
        print(f"  {col:<45}: {n:,} ({pct:.1f}%){flag}")
        if pct > threshold_pct:
            to_drop.append(col)
    return to_drop


def remove_low_variance(
    X       : pd.DataFrame,
    threshold: float = 0.001,
    verbose  : bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with near-zero variance. Returns (X_filtered, dropped_cols).
    """
    from sklearn.feature_selection import VarianceThreshold
    sel    = VarianceThreshold(threshold=threshold)
    X_arr  = sel.fit_transform(X)
    keep   = X.columns[sel.get_support()].tolist()
    dropped= [c for c in X.columns if c not in keep]

    if verbose:
        print(f"[VarFilter] Removed {len(dropped)} low-variance features "
              f"({len(keep)} remaining)")
        if dropped:
            print(f"  Dropped: {dropped[:10]}{'...' if len(dropped)>10 else ''}")

    return pd.DataFrame(X_arr, columns=keep, index=X.index), dropped


# ── Plot utilities ─────────────────────────────────────────────────────────────

def plot_confusion_matrix_pretty(
    cm         : np.ndarray,
    title      : str,
    cmap       : str = 'Blues',
    save_path  : Optional[str] = None,
) -> plt.Figure:
    """Reusable pretty confusion matrix plotter."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im)
    classes = ['Non-Ransomware', 'Ransomware']
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(classes, rotation=15)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual',    fontsize=12)
    ax.set_title(title, fontsize=13)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_feature_distributions(
    X      : pd.DataFrame,
    y      : pd.Series,
    top_n  : int = 6,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-class feature distributions for the top discriminative features."""
    diff = abs(X[y==1].mean() - X[y==0].mean()).sort_values(ascending=False)
    top  = diff.head(top_n).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    n_ransomware = int(y.sum())
    n_other      = len(y) - n_ransomware

    for i, feat in enumerate(top):
        ax = axes[i]
        ax.hist(X.loc[y==0, feat], bins=50, alpha=0.6, color='steelblue',
                label=f'Non-Ransomware (n={n_other:,})', density=True)
        if n_ransomware > 0:
            ax.hist(X.loc[y==1, feat], bins=50, alpha=0.6, color='crimson',
                    label=f'Ransomware (n={n_ransomware:,})', density=True)
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylabel('Density')
        ax.grid(alpha=0.3)

    plt.suptitle('Top Discriminative Features: Non-Ransomware vs Ransomware',
                 fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def save_figure(fig: plt.Figure, path: str, close: bool = True) -> str:
    """Save figure and optionally close it. Returns the path."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    if close:
        plt.close(fig)
    return path


# ── JSON / checkpoint utilities ────────────────────────────────────────────────

def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Save any JSON-serialisable state dict to disk."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a checkpoint saved by save_checkpoint."""
    with open(path) as f:
        return json.load(f)


def format_metrics(results: Dict[str, float]) -> str:
    """Return a compact one-line string of the key metrics."""
    return (
        f"Recall={results.get('recall', 0):.4f}  "
        f"Precision={results.get('precision', 0):.4f}  "
        f"F1={results.get('f1', 0):.4f}  "
        f"ROC-AUC={results.get('roc_auc', 0):.4f}  "
        f"MissRate={results.get('miss_rate', 1):.4f}"
    )
