# V8.4 Fixes Applied

Changed files:

- `src/engine/edr_orchestrator.py`
- `notebooks/03_Research_Grade_Pipeline.ipynb`

## Fixes

1. Implemented a ransomware-specific scoring gate instead of an all-or-nothing gate.
   - Uses multiple signals: write volume, read volume, ratio band, CPU/write coupling, write intensity, CPU activity, and open-file pressure.
   - Requires a minimum score plus sustained evidence and risk.

2. Adjusted SAFE/WATCH policy:
   - Generic high-I/O remains `ALERT_ONLY`.
   - Sustained ransomware-specific evidence becomes `SOFT_BLOCK`.
   - Hard kill remains blocked unless Layer 1 is at least `SUSPICIOUS`.

3. Preserved hard-benign safety:
   - Hard benign should remain `ALERT_ONLY` / no suspension / no kill.

4. Fixed process metric extraction:
   - Always returns both `action` and `actions`.
   - Carries ransomware-specific evidence diagnostics into summary tables.

5. Disabled legacy comparison tables:
   - Removed/marked old registration tables so the notebook has one canonical comparison path.
   - Canonical table includes `xgb_tuned_v2`.

6. Added a final consolidated system-results section:
   - Layer 1 models.
   - Layer 1 ensemble modes.
   - Layer 1 anomaly support.
   - Layer 2 CNN metrics.
   - Process-level EDR results.
   - Hard-benign summary.
   - Regression checklist.

7. Updated limitations to honestly frame the system as a research prototype.

## Expected V8.4 regression outcomes

- Notepad -> SAFE / log only.
- Hard benign -> ALERT_ONLY / no suspend / no kill.
- Dynamic ransomware -> SOFT_BLOCK / suspend + alert + log.
- Real ransomware memory row -> CRITICAL / hard response allowed.
