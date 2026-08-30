# Hybrid Ransomware Detection and Mitigation EDR

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A two-layer **Endpoint Detection and Response (EDR)** research prototype that detects ransomware-like activity by fusing **static memory forensics** with **dynamic behavioral telemetry**, then arbitrates the result through a risk-aware orchestrator before taking any mitigation action.

Built for the Computer and Network Security module.

> Ransomware usually follows the pattern **read files → encrypt content → write encrypted output**. This project detects that pattern while trying hard not to false-positive on benign but I/O-heavy workloads (backups, compression, file copies).

---

## Table of Contents

- [Architecture](#architecture)
- [Layer 1 — Static Memory Forensics](#layer-1--static-memory-forensics)
- [Layer 2 — Dynamic Behavioral Telemetry](#layer-2--dynamic-behavioral-telemetry)
- [Hybrid Orchestrator](#hybrid-orchestrator)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Installation](#installation)
- [Usage](#usage)
- [Demo Safety Notes](#demo-safety-notes)
- [Suggested Demo Flow](#suggested-demo-flow)
- [Results](#results)
- [Limitations](#limitations)
- [Ethical and Safety Statement](#ethical-and-safety-statement)

---

## Architecture

```text
                    +----------------------+
                    |  Monitored Process   |
                    +----------+-----------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
+-----------------------------+   +-----------------------------+
| Layer 1: Static Memory      |   | Layer 2: Dynamic Behavior   |
| Forensics                   |   | CNN Telemetry Monitoring    |
|                             |   |                             |
| RF / XGBoost / LightGBM     |   | CPU, I/O, memory, files     |
| Autoencoder anomaly support |   | 1D-CNN over time windows    |
+--------------+--------------+   +--------------+--------------+
               |                                 |
               +----------------+----------------+
                                v
                  +-----------------------------+
                  | Hybrid EDR Orchestrator     |
                  | risk state + evidence gate  |
                  | risk accumulator + policy   |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  | Mitigation Engine           |
                  | log / alert / suspend / kill|
                  +-----------------------------+
```

The system is not a single CNN classifier — it's a multi-layer pipeline that combines memory evidence, behavioral evidence, and an explicit decision policy before acting.

---

## Layer 1 — Static Memory Forensics

**Question answered:** *Does the memory state look malicious?*

Uses 57 memory-forensics features modeled on Volatility plugin output (see [FEATURE_INVENTORY.txt](FEATURE_INVENTORY.txt) for the full list):

| Feature group | Meaning |
|---|---|
| `malfind.*` | Code injection indicators |
| `psxview.*` | Hidden process anomalies |
| `ldrmodules.*` | Suspicious module/DLL inconsistencies |
| `handles.*` | Abnormal file, registry, mutex, event, or thread handles |
| `svcscan.*` | Service and driver manipulation |
| `pslist.*` | Process count, threads, handlers, and hierarchy |

**Models:**
- Supervised ensemble — Random Forest, XGBoost, LightGBM (**V8 fusion**), the main memory classifier
- Anomaly support — Autoencoder, Isolation Forest (novelty signal, does not replace the supervised classifier)

**Risk states produced:** `SAFE` → `WATCH` → `SUSPICIOUS` → `HIGH_RISK` → `CRITICAL`

This gives the orchestrator memory-forensic context — e.g. stronger mitigation is only allowed once Layer 1 reaches `SUSPICIOUS` or above.

---

## Layer 2 — Dynamic Behavioral Telemetry

**Question answered:** *Is the process currently behaving like ransomware?*

A 1D-CNN monitors 10 features over sliding time windows (window size **8 ticks**):

`cpu_percent`, `memory_rss_mb`, `memory_vms_mb`, `io_read_bytes_delta`, `io_write_bytes_delta`, `net_bytes_sent_delta`, `num_open_files`, `write_read_ratio`, `cpu_x_write`, `io_write_intensity`

The last three are derived features engineered to capture encryption-like behavior (read files + high CPU + heavy writes + many open files):

```text
write_read_ratio   = io_write_bytes_delta / (io_read_bytes_delta + 1)
cpu_x_write        = cpu_percent * io_write_bytes_delta
io_write_intensity = io_write_bytes_delta / (memory_rss_mb * 1024 + 1)
```

The CNN is treated as a **behavioral suspicion signal**, never used alone for hard blocking.

---

## Hybrid Orchestrator

Combines Layer 1 probability + risk state, Layer 2 CNN probability, encryption evidence, a risk accumulator, an evidence streak, and a ransomware-specific score — separating **generic high I/O** (backup tools, compression, file copies) from **ransomware-specific sustained encryption behavior**.

### Decision policy

| Scenario | Decision |
|---|---|
| Normal benign process | `SAFE` → `log_event` |
| Hard benign high-I/O workload | `ALERT_ONLY`, no suspend/kill |
| Dynamic ransomware-like behavior | `SOFT_BLOCK` → suspend + alert + log |
| Critical ransomware memory row | `CRITICAL`, hard response allowed |

A hard response is only permitted when Layer 1 memory evidence is `SUSPICIOUS`, `HIGH_RISK`, or `CRITICAL` — Layer 2 alone can never trigger a kill.

---

## Project Structure

```text
.
├── src/
│   ├── engine/            # EDROrchestrator, system event model
│   ├── collector/         # Live process telemetry monitor
│   ├── models/            # RF, XGBoost, LightGBM, CNN, Autoencoder, LSTM, ensemble
│   ├── evaluation/        # Counterfactual / robustness tests
│   ├── explainability/    # EDR decision explanations
│   ├── mitigation/        # Response actions (log/alert/suspend/kill)
│   ├── rule_engine.py, evaluate.py, train.py, utils.py, config.py
│
├── notebooks/
│   ├── 03_Research_Grade_Pipeline.ipynb   # Full technical pipeline
│   └── 04_Demo_Presentation.ipynb         # Shorter walkthrough for video
│
├── saved_models/          # Trained model weights, scalers, calibrators (Git LFS)
├── data/                  # Datasets + behavioral telemetry sessions (Git LFS)
├── reports/               # Evaluation results, plots, tables
│
├── demo_realtime_edr.py   # Terminal real-time EDR demo
├── demo_edr_ui.py         # Streamlit demo UI (`streamlit run demo_edr_ui.py`)
├── prepare_sequences.py   # Builds CNN training sequences from raw telemetry
├── requirements.txt
└── FEATURE_INVENTORY.txt  # Full 57-feature Layer 1 reference
```

> **Note:** `saved_models/random_forest.pkl` and `random_forest_tuned.pkl` are intentionally excluded (see [saved_models/README_MISSING_RANDOM_FOREST.md](saved_models/README_MISSING_RANDOM_FOREST.md)). Use the included XGBoost/LightGBM model or retrain RF locally if you need it.

Large binary artifacts (`*.csv`, `*.npz`, `*.pkl` under `data/` and `saved_models/`) are tracked with **Git LFS** — run `git lfs install` before cloning if you don't already have it set up.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| [`03_Research_Grade_Pipeline.ipynb`](notebooks/03_Research_Grade_Pipeline.ipynb) | Full pipeline: Layer 1 training/evaluation (RF/XGBoost/LightGBM/Autoencoder/Isolation Forest), leave-one-family-out validation, CNN preprocessing/training/calibration, orchestrator tests, hard-benign evaluation, final results, limitations |
| [`04_Demo_Presentation.ipynb`](notebooks/04_Demo_Presentation.ipynb) | Condensed walkthrough used for the video presentation |

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

If TensorFlow fails to install, pick a `tensorflow`/`tensorflow-cpu` version compatible with your Python version.

---

## Usage

### Run the full research notebook

```bash
jupyter notebook notebooks/03_Research_Grade_Pipeline.ipynb
```

Recommended config for a first full run:

```python
FORCE_REPROCESS_DYNAMIC = True
CNN_CALIBRATION_METHOD = "platt"
EDR_POLICY_MODE = "balanced"
```

Once the dynamic dataset is cached, subsequent runs can set `FORCE_REPROCESS_DYNAMIC = False`.

### Run the real-time terminal demo

```bash
# Live benign monitoring (Notepad)
python demo_realtime_edr.py --mode live --name notepad.exe --duration 30
# expect: SAFE / log_event, no alert, no block

# Hard benign high-I/O workload
python demo_realtime_edr.py --scenario hard_benign_dummy --duration 35
# expect: ALERT_ONLY or SAFE, no suspend, no kill

# Simulated ransomware behavior
python demo_realtime_edr.py --scenario ransomware_dummy --duration 35
# expect: SOFT_BLOCK -> suspend_process + send_alert + log_event

# Same, but allow mitigation against the controlled dummy process
python demo_realtime_edr.py --scenario ransomware_dummy --allow-demo-mitigation

# Critical memory scenario
python demo_realtime_edr.py --scenario memory_critical
# expect: Layer 1 CRITICAL, hard response allowed
```

### Run the Streamlit demo UI

```bash
streamlit run demo_edr_ui.py
```

---

## Demo Safety Notes

**Do not run real ransomware.** The demo scripts only use safe dummy workloads and dummy files, run mitigation in dry-run mode by default, and only allow controlled mitigation when explicitly enabled with `--allow-demo-mitigation`.

Recommended demo environment: an isolated VM, no real personal files, no shared folders, and a VM snapshot taken before testing.

The demo never targets protected system processes, e.g. `System`, `csrss.exe`, `wininit.exe`, `winlogon.exe`, `services.exe`, `lsass.exe`, `svchost.exe`, `explorer.exe` (Windows) or `init`, `systemd`, `sshd`, `bash`, `gnome-shell`, `Xorg` (Linux).

---

## Suggested Demo Flow

1. Introduce the project
2. Explain Layer 1 static memory forensics
3. Explain Layer 2 dynamic telemetry
4. Explain the orchestrator
5. Show final results in the presentation notebook
6. Live terminal demo: Notepad → hard benign dummy → ransomware dummy → memory-critical case
7. Discuss limitations
8. Conclude

The notebook proves the models statistically; the terminal script shows the system behaving live.

---

## Results

| Layer | Highlight |
|---|---|
| Layer 1 | V8 ensemble recall above 90% |
| Layer 2 | CNN window-level FPR improved vs. earlier versions; never used alone for hard blocking |
| Orchestrator | Evidence gate + risk accumulator separate generic high I/O from ransomware-specific behavior; prevents hard kills while Layer 1 is only `SAFE`/`WATCH` |

Expected end-to-end behavior:

| Input | Outcome |
|---|---|
| Notepad | No alert, no block |
| Hard benign high-I/O workload | Alert only, no suspend, no kill |
| Ransomware dummy | Soft block |
| Real ransomware memory row | Critical Layer 1 result |

Full evaluation artifacts (confusion matrices, ROC/PR curves, ablations, cross-family results, reproducibility manifests) are in [`reports/`](reports/).

---

## Limitations

- Not a full commercial EDR
- Layer 1 uses prepared memory features rather than continuous live Volatility acquisition
- CNN window-level false positives still occur; it is a supporting signal, not a standalone kill model
- More hard-benign workloads and ransomware families would improve evaluation coverage
- Explainability methods (occlusion, counterfactual testing) are still experimental

---

## Ethical and Safety Statement

This project does not require running real ransomware. All ransomware-like behavior in the demos is simulated safely with dummy files — the goal is to demonstrate detection and mitigation logic, not to execute malicious software.

---

## Conclusion

Layer 1 provides memory-forensics risk, Layer 2 provides behavioral evidence, and the orchestrator decides the safest response — the system never blindly trusts a single model. This makes it a suitable research prototype for ransomware detection and mitigation.

---

## License

Released under the [MIT License](LICENSE).
