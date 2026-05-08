# Notebook Update Notes

Updated notebook:

- `notebooks/03_Research_Grade_Pipeline.ipynb`

## What changed

The previous Section 17 was replaced with an improved Layer 2 + Hybrid EDR section that matches the patched source code.

Main notebook updates:

1. Added an integration note at the top of the notebook explaining that old outputs are stale until the CNN cache is regenerated and the CNN is retrained.
2. Updated Layer 2 configuration to force dynamic cache reprocessing after the label-handling fix.
3. Updated CNN preprocessing to use the improved `cnn_preprocessor.py`, which preserves per-tick labels and prints label-audit output.
4. Updated CNN training to match `cnn_model.py`, including `val_pr_auc` monitoring and added metrics such as PR-AUC, FPR, specificity, and balanced accuracy.
5. Updated the orchestrator section to use `EDROrchestrator` v5 with:
   - risk states,
   - Layer 1 as a forensic prior,
   - Layer 1 score decay,
   - adaptive CNN thresholds,
   - non-overlapping strike counting,
   - encryption-evidence gating,
   - soft vs hard response separation.
6. Replaced the old Notepad / behavioral / memory tests with safer regression tests:
   - benign Notepad-like process should not be hard-blocked,
   - dynamic encryption behavior should trigger alert/mitigation only after evidence and persistence,
   - memory-forensics test recomputes engineered features from raw memory feature overrides.
7. Added explainability cells:
   - CNN occlusion explanation,
   - Layer 1 SHAP/feature-importance explanation.
8. Added counterfactual validation cells for suppressing or injecting encryption-like features.

## Important next step

Run the updated notebook from the beginning, or at least rerun Section 17 after all Layer 1 models are trained in memory.

The first improved dynamic run should use:

```python
FORCE_REPROCESS_DYNAMIC = True
```

After a clean cache is created, this can be changed to `False` for faster reruns.
