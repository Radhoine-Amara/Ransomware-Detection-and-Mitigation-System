"""
evaluate.py  —  S004 Ransomware Detection System
═══════════════════════════════════════════════════════════════════════════════
Security-focused evaluation pipeline.

Produces:
  • Per-model metrics table (recall-first ordering)
  • Ablation study (raw features vs engineered features)
  • Threshold sensitivity analysis
  • Combined ROC + PR curve plots
  • Latex-ready results table for report/paper
  • JSON results file

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
import matplotlib.gridspec as gridspec

from typing import Dict, Any, List, Optional
from datetime import datetime

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    f1_score, recall_score, precision_score,
    confusion_matrix, brier_score_loss
)

# ── Project path ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config import PLOTS_DIR, REPORTS_DIR, MIN_ACCEPTABLE_RECALL

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Core evaluation functions
# ══════════════════════════════════════════════════════════════════════════════

def security_metrics(
    y_true  : np.ndarray,
    y_pred  : np.ndarray,
    y_proba : np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Compute the full set of security-relevant metrics for one model.

    Returns a dictionary with:
      recall, precision, f1, roc_auc, avg_precision,
      miss_rate (1-recall), fpr, brier_score, threshold_at_90_recall
    """
    n_cls = len(np.unique(y_true))

    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_proba)) if n_cls > 1 else 0.0
        ap      = float(average_precision_score(y_true, y_proba)) if n_cls > 1 else 0.0
        brier   = float(brier_score_loss(y_true, y_proba)) if n_cls > 1 else 1.0
    except Exception:
        roc_auc = ap = 0.0
        brier = 1.0

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        miss_rate = FN / max(FN + TP, 1)
        fpr_val   = FP / max(FP + TN, 1)
    else:
        miss_rate = fpr_val = 0.0

    # Find threshold that achieves exactly 90% recall
    t90 = _find_threshold_at_recall(y_true, y_proba, target_recall=0.90)

    return {
        'model'          : model_name,
        'recall'         : recall,
        'precision'      : precision,
        'f1'             : f1,
        'roc_auc'        : roc_auc,
        'avg_precision'  : ap,
        'brier_score'    : brier,
        'miss_rate'      : miss_rate,
        'fpr'            : fpr_val,
        'threshold_90r'  : t90,
        'meets_target'   : recall >= MIN_ACCEPTABLE_RECALL,
    }


def _find_threshold_at_recall(
    y_true        : np.ndarray,
    y_proba       : np.ndarray,
    target_recall : float = 0.90,
) -> float:
    """Return the highest threshold that achieves >= target_recall."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_t = 0.10
    best_p = 0.0
    for t in np.arange(0.05, 0.95, 0.005):
        pred = (y_proba >= t).astype(int)
        r    = recall_score(y_true, pred, zero_division=0)
        p    = precision_score(y_true, pred, zero_division=0)
        if r >= target_recall and p > best_p:
            best_t, best_p = t, p
    return float(best_t)


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(
    results_list: List[Dict[str, float]],
    save_path   : Optional[str] = None,
    print_table : bool = True,
) -> pd.DataFrame:
    """
    Build and print a formatted model comparison table sorted by recall.

    Parameters
    ──────────
    results_list : List of dicts from security_metrics()
    save_path    : If given, saves CSV to this path
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values('recall', ascending=False).reset_index(drop=True)

    display_cols = ['model', 'recall', 'precision', 'f1',
                    'roc_auc', 'avg_precision', 'miss_rate', 'meets_target']
    display_df   = df[display_cols].copy()

    # Format floats
    for col in ['recall','precision','f1','roc_auc','avg_precision','miss_rate']:
        display_df[col] = display_df[col].map('{:.4f}'.format)

    if print_table:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        print("\n" + "═"*100)
        print("  MODEL COMPARISON TABLE  (sorted by Recall — most important metric in security)")
        print("═"*100)
        print(display_df.to_string(index=False))
        print("═"*100)
        for _, row in df.iterrows():
            status = "✓ MEETS TARGET" if row['meets_target'] else "✗ BELOW TARGET"
            print(f"  {row['model']:<22}: Recall={float(row['recall']):.4f} {status}")
        print()

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[Eval] Table saved → {save_path}")

    return df


def build_latex_table(
    results_list: List[Dict[str, float]],
    save_path   : Optional[str] = None,
) -> str:
    """Generate a LaTeX table for the project report."""
    df = pd.DataFrame(results_list).sort_values('recall', ascending=False)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ransomware Detection Model Comparison (CIC-MalMem2022)}",
        r"\label{tab:model_comparison}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Model & Recall & Precision & F1 & ROC-AUC & Miss Rate \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        name  = str(row['model']).replace('_', ' ').title()
        r_val = float(row['recall'])
        bold  = r'\textbf{' if r_val >= MIN_ACCEPTABLE_RECALL else ''
        end_b = r'}' if bold else ''
        lines.append(
            f"{bold}{name}{end_b} & "
            f"{r_val:.4f} & "
            f"{float(row['precision']):.4f} & "
            f"{float(row['f1']):.4f} & "
            f"{float(row['roc_auc']):.4f} & "
            f"{float(row['miss_rate']):.4f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"[Eval] LaTeX table saved → {save_path}")
    return latex


# ══════════════════════════════════════════════════════════════════════════════
# Threshold sensitivity analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_threshold_sensitivity(
    models_probas : Dict[str, tuple],  # name → (y_true, y_proba)
    save_path     : Optional[str] = None,
) -> plt.Figure:
    """
    Plot Recall and Precision vs. threshold for multiple models.

    models_probas: dict mapping model_name → (y_true_array, y_proba_array)
    """
    colors = ['#2980b9', '#e67e22', '#27ae60', '#8e44ad', '#c0392b']
    thresholds = np.arange(0.05, 0.95, 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for idx, (name, (y_true, y_proba)) in enumerate(models_probas.items()):
        if len(np.unique(y_true)) < 2:
            continue
        color    = colors[idx % len(colors)]
        recalls  = []
        precs    = []
        for t in thresholds:
            pred = (y_proba >= t).astype(int)
            recalls.append(recall_score(y_true, pred, zero_division=0))
            precs.append(precision_score(y_true, pred, zero_division=0))

        axes[0].plot(thresholds, recalls,  lw=2, color=color, label=name)
        axes[1].plot(thresholds, precs,    lw=2, color=color, label=name)

    for ax, metric in zip(axes, ['Recall (TPR)', 'Precision']):
        ax.axhline(0.90, color='red', linestyle='--', alpha=0.5, label='90% target')
        ax.set_xlabel('Decision Threshold', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} vs Threshold', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Threshold Sensitivity Analysis\n'
                 'Recall must stay ≥ 90% — find the threshold that achieves this '
                 'with maximum precision', fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Eval] Threshold plot saved → {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ROC + PR combined plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_combined_curves(
    models_probas : Dict[str, tuple],  # name → (y_true, y_proba)
    operating_pts : Dict[str, float] = None,  # name → threshold
    save_path     : Optional[str] = None,
) -> plt.Figure:
    """
    Combined ROC and Precision-Recall curves for all models.
    Marks the operating threshold as a dot on each curve.
    """
    colors = {
        'Random Forest'    : '#2980b9',
        'XGBoost'          : '#e67e22',
        'LightGBM'         : '#27ae60',
        'Autoencoder'      : '#8e44ad',
        'Isolation Forest' : '#95a5a6',
        'Ensemble'         : '#c0392b',
    }
    default_color = '#7f8c8d'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for name, (y_true, y_proba) in models_probas.items():
        if len(np.unique(y_true)) < 2:
            continue
        color = colors.get(name, default_color)
        lw    = 3.0 if name == 'Ensemble' else 2.0
        ls    = '-.' if name == 'Ensemble' else '-'

        # ROC
        fpr, tpr, thr_roc = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax1.plot(fpr, tpr, lw=lw, color=color, linestyle=ls,
                 label=f'{name} (AUC={auc:.4f})')
        ax1.fill_between(fpr, tpr, alpha=0.05, color=color)

        # Mark operating point
        if operating_pts and name in operating_pts:
            t  = operating_pts[name]
            # Find closest threshold in roc thresholds
            idx = np.argmin(np.abs(thr_roc - t))
            ax1.scatter(fpr[idx], tpr[idx], color=color, s=80, zorder=5)

        # PR
        prec, rec, thr_pr = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax2.plot(rec, prec, lw=lw, color=color, linestyle=ls,
                 label=f'{name} (AP={ap:.4f})')

    # Baselines
    ax1.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random baseline')
    ax2.axhline(y_true.mean() if len(models_probas) else 0.5,
                color='gray', linestyle='--', alpha=0.5, label='No-skill baseline')

    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax1.set_title('ROC Curves — All Models', fontsize=13)
    ax1.legend(fontsize=9, loc='lower right'); ax1.grid(alpha=0.3)

    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves — All Models\n'
                  '(more informative than ROC for imbalanced data)', fontsize=13)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.suptitle('S004 — Ransomware Detection: Model Evaluation',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Eval] Combined curves saved → {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Ablation study
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(
    X_raw  : pd.DataFrame,   # only 55 raw Volatility features
    X_eng  : pd.DataFrame,   # 55 raw + 15 engineered = 70 features
    y      : pd.Series,
    save_path: Optional[str] = None,
    verbose  : bool = True,
) -> pd.DataFrame:
    """
    Ablation study: compare Random Forest performance with vs without
    the 15 engineered features.

    Returns a DataFrame with metric comparison.
    """
    from sklearn.ensemble        import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler

    if verbose:
        print("\n" + "═"*60)
        print("  ABLATION STUDY: Raw Features vs Engineered Features")
        print("═"*60)

    results = []

    for label, X in [("Raw (55 features)", X_raw), ("Engineered (70 features)", X_eng)]:
        Xtr, Xts, ytr, yts = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42)

        sc   = StandardScaler()
        Xtr  = sc.fit_transform(Xtr)
        Xts  = sc.transform(Xts)

        # Simple RF with same hyperparams for fair comparison
        clf = RandomForestClassifier(
            n_estimators=300, class_weight='balanced',
            random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)

        proba = clf.predict_proba(Xts)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        row = security_metrics(yts.values, pred, proba, label)
        results.append(row)

        if verbose:
            print(f"\n  {label}")
            print(f"    Recall    : {row['recall']:.4f}")
            print(f"    Precision : {row['precision']:.4f}")
            print(f"    F1        : {row['f1']:.4f}")
            print(f"    ROC-AUC   : {row['roc_auc']:.4f}")

    df = pd.DataFrame(results)[['model','recall','precision','f1','roc_auc','miss_rate']]

    if verbose:
        print(f"\n  Delta (Engineered - Raw):")
        for col in ['recall','precision','f1','roc_auc']:
            delta = float(df.iloc[1][col]) - float(df.iloc[0][col])
            arrow = "▲" if delta > 0 else "▼"
            print(f"    {col:<12}: {arrow} {abs(delta):.4f}")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n[Ablation] Results saved → {save_path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Summary JSON export
# ══════════════════════════════════════════════════════════════════════════════

def save_results_json(
    results_list: List[Dict[str, float]],
    save_path   : str,
    extra_meta  : Dict[str, Any] = None,
):
    """Save all evaluation results as a JSON file for reproducibility."""
    payload = {
        'timestamp'  : datetime.now().isoformat(),
        'models'     : results_list,
        'meta'       : extra_meta or {},
    }
    with open(save_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[Eval] Results JSON saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    trained_models : Dict[str, Any],      # name → trained model object
    X              : pd.DataFrame,
    y              : pd.Series,
    X_raw          : pd.DataFrame = None,  # for ablation (55 raw features only)
    save_dir       : str = None,
) -> Dict[str, Any]:
    """
    Run the complete evaluation pipeline.

    Parameters
    ──────────
    trained_models : Dict mapping display name → trained model object.
                     Each model must have .train_results with 'y_test' and 'test_proba'.
    X, y           : Full dataset for ablation study.
    X_raw          : 55-feature dataset for ablation (no engineering).
    save_dir       : Directory for all output plots and CSVs.

    Returns full results dict.
    """
    if save_dir is None:
        save_dir = PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    all_metrics   = []
    models_probas = {}
    operating_pts = {}

    for name, model in trained_models.items():
        res = model.train_results
        if not res:
            print(f"[Eval] Skipping {name} — no train_results (not trained?)")
            continue

        y_test  = np.array(res['y_test'])
        y_proba = np.array(res.get('test_proba', res.get('test_errors', [])))

        if len(y_proba) == 0:
            continue

        if hasattr(model, 'threshold'):
            t = model.threshold
        else:
            t = _find_threshold_at_recall(y_test, y_proba)

        y_pred = (y_proba >= t).astype(int)
        m      = security_metrics(y_test, y_pred, y_proba, name)

        all_metrics.append(m)
        models_probas[name] = (y_test, y_proba)
        operating_pts[name] = t

    # ── Comparison table ───────────────────────────────────────────────────
    comp_df = build_comparison_table(
        all_metrics,
        save_path=os.path.join(save_dir, '..', 'model_comparison.csv')
    )

    # ── LaTeX table ────────────────────────────────────────────────────────
    build_latex_table(
        all_metrics,
        save_path=os.path.join(save_dir, '..', 'model_comparison.tex')
    )

    # ── Combined ROC + PR curves ───────────────────────────────────────────
    plot_combined_curves(
        models_probas,
        operating_pts=operating_pts,
        save_path=os.path.join(save_dir, 'combined_roc_pr.png')
    )

    # ── Threshold sensitivity ──────────────────────────────────────────────
    supervised = {k: v for k, v in models_probas.items()
                  if k not in ('Isolation Forest', 'Autoencoder')}
    if supervised:
        plot_threshold_sensitivity(
            supervised,
            save_path=os.path.join(save_dir, 'threshold_sensitivity.png')
        )

    # ── Ablation study ─────────────────────────────────────────────────────
    if X_raw is not None and y is not None:
        run_ablation(
            X_raw, X, y,
            save_path=os.path.join(save_dir, '..', 'ablation_results.csv')
        )

    # ── JSON export ────────────────────────────────────────────────────────
    save_results_json(
        all_metrics,
        save_path=os.path.join(save_dir, '..', 'evaluation_results.json'),
        extra_meta={'dataset': 'CIC-MalMem2022', 'n_features': X.shape[1] if X is not None else 'N/A'}
    )

    return {'metrics': all_metrics, 'comparison': comp_df}
