# V8 Fixes Applied — Balanced Mode

This version applies the next round of fixes after the executed V7 notebook.

## Goals

V8 is a balanced-mode release. It keeps the V7 process-level behavior that worked:

- Notepad-like benign process should not be blocked.
- Dynamic ransomware behavior should trigger at least a soft block.
- Layer 2 must not hard-kill when Layer 1 is SAFE or WATCH.
- Layer 1 V7 recall-oriented ensemble remains the authoritative ensemble path.

V8 adds improvements for cleaner reporting, thresholding, calibration, explainability, and hard-benign evaluation readiness.

## Source-code changes

### `src/models/cnn_preprocessor.py`

- Cache version increased to `5` so older V7 caches are automatically rejected.
- Label policy string updated to V8.
- Optional benign folders are now discovered automatically:
  - `data/behavioral_raw/hard_benign/`
  - `data/behavioral_raw/benign_hard/`
- These folders are treated as benign. Add future hard-benign workloads here.

### `src/models/cnn_model.py`

- Added optional probability calibration:
  - `calibration_method='platt'`
  - also supports `isotonic` or `none`.
- Added validation-derived threshold modes:
  - `sensitive`: recall-first operating point.
  - `balanced`: default EDR operating point.
  - `strict`: lower-FPR operating point for SAFE/WATCH contexts.
- Added FPR-constrained threshold selection using validation results.
- Added `predict_proba_windows()` and `predict_proba_raw_window()` so orchestrator inference can use the same calibration as training/evaluation.
- Save/load now preserves threshold modes and calibrator.

### `src/engine/edr_orchestrator.py`

- Updated to Hybrid EDR Orchestrator v8.
- Added `policy_mode='balanced'`.
- Added threshold-mode selection based on Layer 1 risk state:
  - SAFE/WATCH use stricter CNN threshold mode.
  - SUSPICIOUS uses balanced mode.
  - HIGH_RISK/CRITICAL use sensitive mode.
- Added calibrated CNN inference through `predict_proba_raw_window()` when available.
- Added risk timeline logging:
  - CNN probability over time.
  - adaptive threshold over time.
  - evidence hits.
  - risk accumulator score.
  - evidence streak.
  - counted strikes.
- Dynamic response events include `policy_mode`, `threshold_mode`, and a `risk_timeline_tail` for notebook/debug reporting.
- Hard kill remains protected: Layer 2 cannot hard-kill unless Layer 1 is at least SUSPICIOUS and CNN confidence reaches the kill threshold.

### `src/explainability/edr_explain.py`

- Occlusion helper now defaults to selecting less-saturated windows (`high=0.95`) for clearer explanations.

### `src/evaluation/counterfactual_tests.py`

- Improved counterfactual injection to create a more realistic encryption burst rather than uniformly multiplying all ticks.
- Stronger read/write/CPU/open-file floors are used.
- Derived features are still recomputed after mutation.
- Counterfactuals remain experimental until validated on hard-benign controls.

## Notebook changes

Updated:

- `notebooks/03_Research_Grade_Pipeline.ipynb`

Main notebook updates:

- Renamed section to V8.
- Added V8 configuration:
  - calibration method
  - threshold modes
  - strict/balanced FPR targets
  - hard-benign folder support
  - EDR policy mode
- Removed/renamed stale ensemble comparisons so they do not conflict with the final V8 summary.
- Added threshold-mode reporting after CNN training.
- Added process-level regression summary:
  - Notepad no alert.
  - Behavioral ransomware soft-blocks.
  - No hard kill while Layer 1 is SAFE/WATCH.
  - Synthetic memory test should be at least WATCH.
- Added risk timeline display around the behavioral ransomware detection step.
- Improved occlusion window selection to avoid saturated examples when possible.
- Counterfactual section is kept visible but explicitly marked experimental.
- Final notes explain where to place future hard-benign CSVs.

## How to run

Run the notebook from the beginning.

For the first V8 run, keep:

```python
FORCE_REPROCESS_DYNAMIC = True
CNN_RELABEL_POLICY = 'auto_evidence'
CNN_CALIBRATION_METHOD = 'platt'
EDR_POLICY_MODE = 'balanced'
```

After the cache is rebuilt successfully, later runs can set:

```python
FORCE_REPROCESS_DYNAMIC = False
```

## Expected behavior

- Test 1 / Notepad: no alert, no hard action.
- Test 2 / dynamic ransomware: alert with `suspend_process`, `send_alert`, and `log_event`.
- Test 3 / synthetic memory anomaly: WATCH state or higher, but no forced hard block.
- Layer 1 ensemble: final V8 summary is the authoritative ensemble result.
- CNN: may still have window-level false positives, but hard mitigation remains guarded by the orchestrator.

## Note on missing RF files

This package does not include `random_forest.pkl` or `random_forest_tuned.pkl` unless they were present in the input project directory. The code is compatible with them if you add them locally under `saved_models/`.
