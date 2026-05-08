# Hybrid Ransomware Detection and Mitigation EDR

## Project Overview

This project is a **Hybrid Endpoint Detection and Response (EDR) research prototype** for ransomware detection and mitigation. It was developed for the Computer and Network Security module.

The main goal is to detect ransomware-like activity by combining:

1. **Layer 1: Static Memory Forensics**
2. **Layer 2: Dynamic Behavioral Telemetry**
3. **A Hybrid Orchestrator**
4. **A Mitigation and Demo Layer**

The system is not only a CNN classifier. It is a multi-layer EDR-style pipeline that combines memory evidence, behavioral evidence, and a decision policy.

Ransomware usually follows a pattern like:

```text
Read files -> Encrypt content -> Write encrypted output
```

This project tries to detect that pattern safely while avoiding false positives on benign programs such as Notepad, file copying, compression, and backup-like workloads.

---

## Final Architecture

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

---

## Layer 1 — Static Memory Forensics

Layer 1 analyzes static memory-forensics features. It answers:

```text
Does the memory state look malicious?
```

It uses memory features similar to Volatility plugin outputs, including:

```text
malfind.*
psxview.*
ldrmodules.*
handles.*
pslist.*
svcscan.*
callbacks.*
```

Important examples:

| Feature Group | Meaning |
|---|---|
| `malfind.*` | Code injection indicators |
| `psxview.*` | Hidden process anomalies |
| `ldrmodules.*` | Suspicious module/DLL inconsistencies |
| `handles.*` | Abnormal file, registry, mutex, event, or thread handles |
| `svcscan.*` | Service and driver manipulation |
| `pslist.*` | Process count, threads, handlers, and process hierarchy |

### Layer 1 Models

Layer 1 uses supervised machine learning models:

```text
Random Forest
XGBoost
LightGBM
V8 ensemble fusion
```

It also includes anomaly detection models:

```text
Autoencoder
Isolation Forest
```

The supervised RF/XGBoost/LightGBM ensemble is the main ransomware memory classifier.  
The Autoencoder is used as a novelty/anomaly support signal. It does not replace the supervised classifier.

### Layer 1 Risk States

Layer 1 produces a memory risk state:

```text
SAFE
WATCH
SUSPICIOUS
HIGH_RISK
CRITICAL
```

Layer 1 is important because it gives the orchestrator memory-forensic context. For example, if Layer 1 is `SUSPICIOUS`, `HIGH_RISK`, or `CRITICAL`, the orchestrator can allow stronger mitigation.

---

## Layer 2 — Dynamic Behavioral Telemetry

Layer 2 monitors process behavior over time. It answers:

```text
Is the process currently behaving like ransomware?
```

The CNN uses 10 dynamic features:

```text
cpu_percent
memory_rss_mb
memory_vms_mb
io_read_bytes_delta
io_write_bytes_delta
net_bytes_sent_delta
num_open_files
write_read_ratio
cpu_x_write
io_write_intensity
```

### Derived Features

The derived features are used to represent encryption-like behavior.

```text
write_read_ratio = io_write_bytes_delta / (io_read_bytes_delta + 1)
cpu_x_write = cpu_percent * io_write_bytes_delta
io_write_intensity = io_write_bytes_delta / (memory_rss_mb * 1024 + 1)
```

These features help detect the ransomware loop:

```text
read files + high CPU + heavy writes + many open files
```

Layer 2 uses a CNN over time windows:

```text
window size = 8 ticks
features    = 10
```

The CNN is not used alone for hard blocking. It is treated as a behavioral suspicion signal.

---

## Hybrid Orchestrator

The orchestrator combines:

```text
Layer 1 probability
Layer 1 risk state
Layer 2 CNN probability
encryption evidence
risk accumulator
evidence streak
ransomware-specific score
```

The orchestrator separates:

```text
generic high I/O
```

from:

```text
ransomware-specific sustained encryption behavior
```

This is important because benign programs like backup tools, compression tools, and file copy operations can also generate high I/O.

### Final Decision Policy

| Scenario | Expected Decision |
|---|---|
| Normal benign process | `SAFE` / `log_event` |
| Hard benign high-I/O workload | `ALERT_ONLY` / no suspend / no kill |
| Dynamic ransomware-like behavior | `SOFT_BLOCK` / suspend + alert + log |
| Critical ransomware memory row | `CRITICAL` / hard response allowed |

A hard response is only allowed when Layer 1 memory evidence is strong enough, such as `SUSPICIOUS`, `HIGH_RISK`, or `CRITICAL`.

---

## Project Structure

Recommended final project structure:

```text
project_root/
├── src/
│   ├── engine/
│   │   └── edr_orchestrator.py
│   ├── models/
│   │   ├── cnn_model.py
│   │   ├── cnn_preprocessor.py
│   │   ├── random_forest_model.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── ensemble_model.py
│   │   └── autoencoder_model.py
│   ├── evaluation/
│   └── explainability/
│
├── notebooks/
│   ├── 03_Research_Grade_Pipeline.ipynb
│   └── 04_Demo_Presentation.ipynb
│
├── saved_models/
│   ├── cnn_model.keras
│   ├── cnn_scaler.pkl
│   ├── cnn_metadata.json
│   ├── random_forest.pkl
│   ├── random_forest_tuned.pkl
│   ├── xgboost.pkl
│   ├── xgboost_tuned.pkl
│   ├── lightgbm.pkl
│   ├── lightgbm_tuned.pkl
│   └── scaler / metadata files
│
├── data/
│   └── behavioral_raw/
│
├── demo_realtime_edr.py
├── requirements.txt
└── README.md
```

---

## Notebooks

### Full Research Notebook

```text
notebooks/03_Research_Grade_Pipeline.ipynb
```

This is the main technical notebook. It contains:

- Layer 1 model training and evaluation
- Random Forest, XGBoost, LightGBM results
- Autoencoder and Isolation Forest evaluation
- Leave-one-family-out validation
- CNN preprocessing and training
- CNN calibration and threshold analysis
- EDR orchestrator tests
- hard benign evaluation
- final consolidated results
- limitations and future work

### Presentation Notebook

```text
notebooks/04_Demo_Presentation.ipynb
```

This is a shorter notebook for the video presentation. It is used to explain the project clearly without showing the full long training pipeline.

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If TensorFlow causes an installation issue, install a TensorFlow version compatible with your Python version.

---

## Running the Full Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/03_Research_Grade_Pipeline.ipynb
```

Recommended configuration:

```python
FORCE_REPROCESS_DYNAMIC = True
CNN_CALIBRATION_METHOD = "platt"
EDR_POLICY_MODE = "balanced"
```

Run the notebook from the beginning.

After the first full run, if the dynamic dataset did not change, you can set:

```python
FORCE_REPROCESS_DYNAMIC = False
```

---

## Running the Presentation Notebook

Open:

```text
notebooks/04_Demo_Presentation.ipynb
```

Use this notebook for the video explanation.

---

## Running the Real-Time Demo Script

The demo script is:

```text
demo_realtime_edr.py
```

Place it in the project root folder.

### 1. Live Notepad Monitoring

On Windows:

```bash
python demo_realtime_edr.py --mode live --name notepad.exe --duration 30
```

Expected result:

```text
SAFE
log_event
no alert
no block
```

### 2. Hard Benign Dummy Workload

```bash
python demo_realtime_edr.py --scenario hard_benign_dummy --duration 35
```

Expected result:

```text
ALERT_ONLY or SAFE
no suspend
no kill
```

### 3. Ransomware Dummy Simulation

```bash
python demo_realtime_edr.py --scenario ransomware_dummy --duration 35
```

Expected result:

```text
SOFT_BLOCK
suspend_process + send_alert + log_event
```

### 4. Ransomware Dummy With Controlled Mitigation

```bash
python demo_realtime_edr.py --scenario ransomware_dummy --allow-demo-mitigation
```

This should only allow mitigation on the controlled dummy process started by the script.

### 5. Memory Critical Scenario

```bash
python demo_realtime_edr.py --scenario memory_critical
```

Expected result:

```text
Layer 1 CRITICAL
hard response allowed
```

---

## Demo Safety Notes

Do **not** run real ransomware.

The demo uses:

```text
safe dummy workloads
dummy files
dry-run mitigation by default
controlled mitigation only when explicitly allowed
```

Recommended demo environment:

```text
isolated virtual machine
no real personal files
no shared folders or read-only shared folder
VM snapshot before testing
dummy files only
```

The demo script should not target protected system processes such as:

```text
Windows:
System
csrss.exe
wininit.exe
winlogon.exe
services.exe
lsass.exe
svchost.exe
explorer.exe

Linux:
init
systemd
sshd
bash
gnome-shell
Xorg
```

---

## Suggested Video Demo Flow

Use this order:

```text
1. Introduce the project
2. Explain Layer 1 static memory forensics
3. Explain Layer 2 dynamic telemetry
4. Explain the orchestrator
5. Show final results in the presentation notebook
6. Run terminal demo:
   - Notepad
   - Hard benign dummy
   - Ransomware dummy
   - Memory critical case
7. Explain limitations
8. Conclude
```

The notebook proves the models statistically.  
The terminal script shows how the system behaves live.

---

## Important Results to Present

Final behavior to highlight:

```text
Notepad:
  no alert, no block

Hard benign:
  alert only, no suspend, no kill

Ransomware dummy:
  soft block

Real ransomware memory row:
  critical Layer 1 result
```

Layer 1:

```text
V8 ensemble recall above 90%
```

Layer 2:

```text
CNN window-level FPR improved compared to earlier versions
CNN is not used alone for hard blocking
```

Orchestrator:

```text
uses evidence gate
uses risk accumulator
separates generic high I/O from ransomware-specific behavior
prevents hard kill when Layer 1 is only SAFE or WATCH
```

---

## Main Improvements Made

Important improvements during development:

```text
1. Fixed CNN label handling.
2. Added active-encryption relabeling.
3. Added temporal label dilation.
4. Reduced CNN false positives.
5. Added CNN calibration.
6. Added CNN threshold modes.
7. Added hard benign evaluation.
8. Added Layer 1 V8 ensemble fusion.
9. Fixed ensemble recall drop.
10. Fixed action/actions counting bug.
11. Added hard benign ALERT_ONLY policy.
12. Added ransomware-specific scoring gate.
13. Added risk accumulator.
14. Added soft block vs hard block separation.
15. Added final consolidated summary.
16. Added reproducibility and artifact checks.
17. Added terminal real-time demo script.
```

---

## Limitations

This is a research prototype.

Current limitations:

```text
1. It is not a full commercial EDR.
2. Layer 1 live memory acquisition is represented using prepared memory features.
3. Full live Volatility extraction is heavier and not implemented as a continuous agent.
4. CNN window-level false positives still exist.
5. The CNN is used as a behavioral signal, not as a standalone kill model.
6. More hard benign workloads should be tested.
7. More ransomware families would improve evaluation.
8. Explainability methods such as occlusion and counterfactual testing are still experimental.
```

---

## Ethical and Safety Statement

This project does not require running real ransomware.

All ransomware-like behavior in the demo is simulated safely using dummy files.

The goal is to demonstrate detection and mitigation logic, not to execute malicious software.

---

## Final Conclusion

This project demonstrates a hybrid ransomware EDR prototype.

The system combines:

```text
Layer 1 static memory forensics
Layer 2 dynamic behavioral telemetry
risk-aware orchestrator
safe mitigation policy
```

The most important contribution is that the system does not blindly trust one model.

Instead:

```text
Layer 1 provides memory-forensics risk.
Layer 2 provides behavior evidence.
The orchestrator decides the safest response.
```

Final expected behavior:

```text
Normal benign process -> log only
Hard benign high-I/O workload -> alert only
Ransomware-like behavior -> soft block
Critical memory ransomware -> hard response allowed
```

This makes the project suitable as an advanced research prototype for ransomware detection and mitigation.
