# V8.3 Research-Hardening Fixes Applied

This patch contains only changed/new files. Apply it over the latest V8.2 project.

## Source changes

### `src/models/autoencoder_model.py`
- Moved Autoencoder reconstruction-error normalisation into the model class.
- Added `test_proba`, `ae_normalized_threshold`, `ae_norm_max_error`, and normalised AE metrics inside `train()`.
- Added `predict_proba_batch()` for normalised AE anomaly probabilities.
- Saved/loaded AE normalisation metadata with the model.
- Prevents the previous notebook-order bug where AE recall could become zero if the notebook patch cell was skipped.

## Notebook changes

### `notebooks/03_Research_Grade_Pipeline.ipynb`
- Renumbered sections sequentially.
- Fixed the hard-benign `action` vs `actions` counting bug.
- Replaced hardcoded cross-family RF threshold `0.16` with `rf_tuned.threshold` fallback.
- Marked older registration/comparison tables as legacy/deprecated.
- Ensured XGBoost tuned v2 is included in final/canonical summaries.
- Marked the old Section 11 custom layered detector as deprecated and redirects users to the real `EDROrchestrator`.
- Added reproducibility/environment reporting.
- Added model artifact contract checks.
- Added Layer 1 calibration reliability curves.
- Added canonical summary tables for:
  - Layer 1 supervised models
  - Layer 1 ensemble modes
  - Layer 1 anomaly support models
  - Layer 2 CNN metrics
- Added process-level EDR outcome matrix and decision trace table.
- Added CNN threshold sensitivity analysis.
- Added hard-benign reason analysis.
- Fixed occlusion probability mismatch by using one selected window consistently.
- Added optional SHAP summary/beeswarm plot for Layer 1.
- Added safe demo policy, protected-process policy, deployment policy table, known limitations, and final regression checklist.
- Added explicit hard-benign assertions: no hard block and preferably no suspension.
- Strengthened counterfactual interpretation guidance.

## New terminal demo script

### `demo_realtime_edr.py`
- Safe terminal-based EDR demo runner.
- Dry-run by default: prints `WOULD SUSPEND` / `WOULD KILL` without dangerous actions.
- Supports:
  - built-in Notepad-like benign scenario
  - built-in ransomware-like encryption scenario
  - prepared memory-critical scenario
  - CSV replay mode for hard-benign files
  - basic live process telemetry mode using `psutil`
- Loads saved CNN model and scaler when available.
- Loads RF Layer 1 model if local RF artifacts exist; otherwise uses a safe WATCH-level demo static prior.
- Has protected-process safety logic.

## Suggested commands

```bash
# Notebook validation
jupyter notebook notebooks/03_Research_Grade_Pipeline.ipynb

# Terminal demos
python demo_realtime_edr.py --scenario notepad
python demo_realtime_edr.py --scenario ransomware
python demo_realtime_edr.py --mode replay --csv data/behavioral_raw/data/benign_021.csv
python demo_realtime_edr.py --mode live --name notepad.exe --duration 30
```

## Recommended notebook settings

```python
FORCE_REPROCESS_DYNAMIC = True
CNN_CALIBRATION_METHOD = "platt"
EDR_POLICY_MODE = "balanced"
```
