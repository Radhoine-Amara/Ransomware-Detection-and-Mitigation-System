# V7 Fixes Applied

V7 is a targeted correction after the V6 notebook run. V6 made the system safer and fixed the Notepad false block, but it became too conservative and missed the behavioral ransomware regression test. V7 keeps the Notepad protection while restoring ransomware sensitivity.

## 1. Layer 2 label/window policy

Changed in `src/models/cnn_preprocessor.py`:

- Cache version increased to `4` so old V6 dynamic caches are rejected.
- Default window label fraction changed from `0.50` to `0.25`.
  - With `window_size=8`, a window is positive if at least 2 ticks are active-encryption ticks.
  - This prevents attack labels from becoming too sparse.
- Added asymmetric temporal label dilation:
  - `dilate_before_ticks = 1`
  - `dilate_after_ticks = 2`
- Kept active-encryption relabeling for all-positive ransomware CSVs.
- Class-weight cap raised from `5.0` to `8.0` because V7 labels are intentionally stricter and positives are sparse.
- Cache metadata now stores `majority_pos_fraction` and `relabel_policy`.

## 2. Layer 2 orchestrator risk accumulator

Changed in `src/engine/edr_orchestrator.py`:

- Orchestrator version updated to V7.
- `min_cnn_threshold` default reduced from `0.65` to `0.20`, so it no longer overrides the trained CNN threshold too aggressively.
- Added continuous Layer 2 risk accumulation:

```text
risk_t = risk_decay * risk_{t-1} + (1 - risk_decay) * current_signal
```

- Added evidence persistence tracking.
- Added fallback soft-block rule:

```text
persistent strong encryption evidence -> SOFT_BLOCK
```

This fixes the V6 issue where Test 2 had `max_L2 = 100%` and `evidence = True`, but no alert because only `1/3` non-overlapping strikes were counted.

## 3. Safer mitigation policy preserved

V7 still prevents CNN-only hard kills:

```text
Layer 1 SAFE/WATCH + Layer 2 strong evidence -> SOFT_BLOCK
Layer 1 SUSPICIOUS/HIGH_RISK + Layer 2 strong evidence -> HARD_BLOCK allowed
Layer 1 CRITICAL -> HARD_BLOCK allowed
```

Notepad remains protected because high CNN probability without encryption evidence does not accumulate risk and does not trigger mitigation.

## 4. Layer 1 ensemble recall-oriented fusion

Changed in `src/models/ensemble_model.py`:

- Default ensemble weights changed to:

```text
RF       = 0.45
LightGBM = 0.40
XGBoost  = 0.15
AE       = 0.00
```

- Added V7 fusion rule:

```text
alert if weighted_score >= tuned_threshold
OR at least 2 supervised models cross their own tuned thresholds
OR any supervised model is very confident (>= 0.85)
```

- Threshold is tuned on validation data using the same fusion rule used during test evaluation.
- This addresses the issue where all individual models had >90% recall but the average ensemble dropped to ~77–79% recall.

## 5. Notebook updated

Changed in `notebooks/03_Research_Grade_Pipeline.ipynb`:

- Updated Section 17 to V7 configuration.
- Set:

```python
FORCE_REPROCESS_DYNAMIC = True
CNN_MAJORITY_FRACTION = 0.25
CNN_RELABEL_POLICY = "auto_evidence"
```

- Added temporal label dilation settings.
- Added V7 orchestrator risk-accumulator parameters.
- Updated ensemble cells to use recall-oriented V7 fusion.
- Replaced stale/hard-coded final summary table with a table generated from actual result variables.
- Cleared notebook outputs because old V6 outputs are stale.

## What to rerun

Run the notebook from the beginning. Keep this on the first V7 run:

```python
FORCE_REPROCESS_DYNAMIC = True
```

After the first successful V7 run, you can set it to `False` to reuse the V7 cache.

## Expected behavior

- Test 1 Notepad: no hard block, should remain safe.
- Test 2 behavioral ransomware: should trigger at least SOFT_BLOCK through risk accumulator / evidence fallback.
- Test 3 synthetic memory: should remain WATCH or higher, but not forced into hard detection because the synthetic vector may not match real MalMem ransomware distribution.
- Layer 1 ensemble recall: should improve toward the individual model recalls and no longer rely on hard-coded summary values.
