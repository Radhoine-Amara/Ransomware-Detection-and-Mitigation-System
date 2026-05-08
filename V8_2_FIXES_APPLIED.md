# V8.2 fixes applied

Changed files only patch.

## Main fixes

1. Added hard-benign protection in `src/engine/edr_orchestrator.py`.
   - SAFE/WATCH + generic high-I/O now becomes alert-only first.
   - SAFE/WATCH soft-block now requires stricter ransomware-specific evidence.
   - Layer 2 still cannot hard-kill unless Layer 1 is SUSPICIOUS/HIGH_RISK/CRITICAL.

2. Added state-aware soft-block thresholds.
   - SAFE/WATCH: stricter `watch_risk_soft_threshold`, longer persistence.
   - SUSPICIOUS/HIGH_RISK: existing faster response remains.

3. Updated the notebook configuration and summaries.
   - Hard-benign summary now separates alert-only, soft-block, and hard-block.
   - Hard-block count no longer treats snapshot/evidence preservation as a kill.
   - Regression checks include preferred no-suspension behavior for hard benign.

4. Added a Streamlit demo UI: `demo_edr_ui.py`.
   - Scenarios: Notepad benign, hard benign CSV, ransomware simulation, static high-risk memory.
   - Displays Layer 1 state, Layer 2/risk timeline, evidence, and final actions.

## Run instructions

Copy these changed files into your project, then run:

```bash
streamlit run demo_edr_ui.py
```

For notebook validation, rerun `notebooks/03_Research_Grade_Pipeline.ipynb` from the beginning with dynamic reprocessing enabled.
