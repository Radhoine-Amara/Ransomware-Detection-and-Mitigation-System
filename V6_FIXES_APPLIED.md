# V6 Fixes Applied After Notebook Review

This package applies the fixes recommended after reviewing the executed notebook results.

## 1. Dynamic CNN label fix: active-encryption relabeling

File: `src/models/cnn_preprocessor.py`

The previous fix preserved CSV labels, but the notebook showed that ransomware sessions were still all-positive. V6 adds `relabel_policy='auto_evidence'`.

Behavior:

- If a ransomware CSV has mixed per-tick labels, the labels are preserved.
- If a ransomware CSV is all-positive, labels are rebuilt from active encryption evidence.
- Benign CSVs remain label 0.

This helps the CNN learn active encryption behavior instead of folder/session identity.

New parameters:

```python
preprocess_dynamic_telemetry(
    ...,
    relabel_policy="auto_evidence",
    label_evidence_thresholds={...},
    force_reprocess=True,
)
```

The CNN cache version was increased to `3`, so old caches should be rejected.

## 2. Safer Layer 2 mitigation policy

File: `src/engine/edr_orchestrator.py`

Layer 2 can no longer hard-kill while Layer 1 is `SAFE` or `WATCH`. Strong Layer 2 evidence in those states produces a soft block first:

- suspend process
- preserve snapshot/log evidence
- send alert

Hard kill now requires:

- dynamic CNN evidence and persistence,
- Layer 1 at least `SUSPICIOUS`,
- CNN confidence above `kill_threshold`.

Layer 1 state thresholds were also adjusted:

```text
SAFE:       p1 < 0.10
WATCH:      0.10 <= p1 < 0.35
SUSPICIOUS: 0.35 <= p1 < 0.75
HIGH_RISK:  0.75 <= p1 < 0.90
CRITICAL:   p1 >= 0.90
```

This means the previous synthetic memory score around 0.145 becomes `WATCH`, not `SAFE`.

## 3. Counterfactual tests corrected

File: `src/evaluation/counterfactual_tests.py`

The earlier counterfactual helper mutated raw and derived features inconsistently. V6 recomputes derived features after every mutation:

- `write_read_ratio`
- `cpu_x_write`
- `io_write_intensity`

The injection test also uses absolute floors because multiplying a quiet benign I/O window by 10 can still leave it near zero.

## 4. Occlusion explainability improved

File: `src/explainability/edr_explain.py`

V6 adds coherent occlusion:

- if `io_write_bytes_delta` is occluded, derived features are recomputed;
- default baseline is now `scaler_center`, not the window median;
- saturated windows are explicitly discouraged for final report examples.

## 5. Notebook updated

File: `notebooks/03_Research_Grade_Pipeline.ipynb`

The notebook now uses:

- `CNN_RELABEL_POLICY = 'auto_evidence'`
- active-encryption label thresholds
- v6 Layer 1 thresholds
- safer orchestrator policy
- coherent occlusion
- corrected counterfactual tests

Notebook outputs were cleared because previous outputs are stale. Rerun the notebook from the beginning.

## Required next run

Use:

```python
FORCE_REPROCESS_DYNAMIC = True
CNN_RELABEL_POLICY = 'auto_evidence'
```

Then retrain the CNN and rerun the three tests.

## Expected changes after rerun

The next run should show:

- ransomware session labels are no longer 100% positive;
- CNN benign baseline probabilities should decrease;
- Notepad should remain unblocked;
- strong ransomware behavior should produce at least soft block, and hard block only when Layer 1 is sufficiently suspicious;
- counterfactual suppression/injection directions should become more meaningful.
