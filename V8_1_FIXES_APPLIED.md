# V8.1 Hard-Benign + Reporting Fixes

This patch contains only the files changed after V8.

## Changed source files

- `src/models/cnn_preprocessor.py`
  - Detects `benign_021.csv` to `benign_031.csv` as hard-benign even if they remain under `data/behavioral_raw/data/`.
  - Also supports `data/behavioral_raw/hard_benign/` and `data/behavioral_raw/benign_hard/`.
  - Keeps hard-benign labels as `0` while preserving `sample_type`, `source_file`, and window metadata.
  - Bumps dynamic cache version so the dataset rebuilds.

- `src/models/cnn_model.py`
  - Makes `sensitive`, `balanced`, and `strict` threshold modes genuinely distinct.
  - Adds minimum threshold floors for each mode.
  - Adds `probability_diagnostics()` for calibrated benign/ransomware score distributions.

- `src/explainability/edr_explain.py`
  - Fixes occlusion to use calibrated CNN predictions.
  - Prevents derived-feature occlusion from being silently undone by recomputation.

- `src/evaluation/counterfactual_tests.py`
  - Uses calibrated CNN predictions.
  - Adds saturation-aware interpretation fields.

## Changed notebook

- `notebooks/03_Research_Grade_Pipeline.ipynb`
  - Updated to V8.1 configuration.
  - Adds hard-benign process-level evaluation.
  - Adds hard-benign window-level subgroup metrics when available.
  - Adds calibrated score diagnostics.
  - Adds real ransomware memory-row sanity test.
  - Fixes occlusion probability inconsistency.
  - Cleans counterfactual interpretation.

## Run settings

Run the notebook from the beginning with:

```python
FORCE_REPROCESS_DYNAMIC = True
CNN_CALIBRATION_METHOD = "platt"
EDR_POLICY_MODE = "balanced"
```

Keep your hard-benign CSVs at:

```text
data/behavioral_raw/data/benign_021.csv
...
data/behavioral_raw/data/benign_031.csv
```

or move them to:

```text
data/behavioral_raw/hard_benign/
data/behavioral_raw/benign_hard/
```
