# Improvements Applied to the Hybrid Ransomware EDR Source Code

This patched source folder implements the code-level improvements discussed for the hybrid Static + Dynamic + Orchestrator ransomware EDR system.

## Files changed

### `src/models/cnn_preprocessor.py`
- Preserves existing per-tick CSV labels instead of forcing every row in ransomware folders to label `1`.
- Uses folder-based labels only as a fallback when the CSV has no `label` column or invalid labels.
- Adds label-audit output showing benign/ransomware tick distribution and mixed/all-positive sessions.
- Adds cache metadata/versioning so older polluted `.npz` caches are rejected and reprocessed.
- Reduces positive class-weight cap from `8.0` to `5.0` to reduce over-prediction of ransomware.

### `src/models/cnn_model.py`
- Adds PR-AUC metric (`tf.keras.metrics.AUC(curve="PR", name="pr_auc")`).
- Uses `val_pr_auc` for EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint instead of `val_recall`.
- Adds `pr_auc`, `fpr`, `specificity`, and `balanced_accuracy` to returned training metrics.
- Changes standalone CNN `predict()` recommendations so Layer 2 alone does not recommend `kill_process`; mitigation is left to the orchestrator.
- Updates training-history plotting to show PR-AUC when available.

### `src/engine/edr_orchestrator.py`
- Replaced binary orchestration with a risk-state design: `SAFE`, `WATCH`, `SUSPICIOUS`, `HIGH_RISK`, `CRITICAL`.
- Uses Layer 1 as a forensic prior instead of ignoring medium scores.
- Uses the trained Layer 1 threshold if available, but reserves hard blocking for critical Layer 1 probabilities.
- Adds Layer 1 suspicion decay over time.
- Adds adaptive CNN thresholds based on effective Layer 1 risk.
- Adds non-overlapping strike counting to avoid treating overlapping CNN windows as independent proof.
- Adds an encryption-evidence gate based on interpretable telemetry metrics: write/read volume, write-read ratio, CPU/write coupling, write intensity, CPU activity, and open file pressure.
- Separates soft mitigation from hard mitigation: Layer 2 alerts usually suspend first; hard kill requires stronger confidence/context.
- Handles Layer 1 preprocessing/schema errors as `WATCH` instead of treating the process as clean.

### `src/models/ensemble_model.py`
- Fixes ensemble threshold leakage by tuning the ensemble threshold on a validation split, then evaluating once on an untouched test split.

## Important remaining steps

These code changes do not retrain your models. You still need to:

1. Regenerate the dynamic telemetry cache with `force_reprocess=True`.
2. Retrain the CNN so it learns the corrected labels.
3. Re-run Notepad, behavioral ransomware, and memory ransomware tests.
4. Build hard-benign negative-control sessions, such as 7-Zip compression, backup, file copy, antivirus scan, software installer, compiler build, browser download, and Office autosave.
5. Tune the evidence thresholds using validation data, not the final test set.

## Suggested retraining order

1. Reprocess CNN data.
2. Retrain CNN.
3. Save the new CNN model and scaler.
4. Reload the improved orchestrator.
5. Run process-level tests.
6. Measure window-level and process-level FPR.


### `src/explainability/edr_explain.py`
- Adds CNN occlusion explanation for Layer 2 windows.
- Adds optional TreeSHAP explanation for Layer 1, with fallback to feature importances when SHAP is unavailable.

### `src/evaluation/counterfactual_tests.py`
- Adds counterfactual testing helpers that suppress or inject encryption-like telemetry features and compare CNN probability changes.

## Notebook update

The main research notebook was also updated:

- `notebooks/03_Research_Grade_Pipeline.ipynb`

Its Layer 2 / Hybrid EDR section now matches the improved source code and includes corrected CNN preprocessing, PR-AUC-based CNN training, v5 orchestrator tests, occlusion explainability, and counterfactual validation.
