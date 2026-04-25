# Behavioral Data Collection Guide
## For LSTM Temporal Sequence Model — S004 Ransomware Detection System

---

## Why You Need This

The CIC-MalMem2022 dataset contains **static memory snapshots** — each row is an
independent measurement with no temporal relationship to adjacent rows. This is
why the LSTM failed: stacking unrelated rows into sequences adds noise, not signal.

A proper LSTM needs **behavioral time-series data**: measurements of the SAME
process at SEQUENTIAL time steps (e.g., every 5 seconds). This guide shows you
exactly how to collect it.

---

## Part 1 — VM Setup for Safe Data Collection

### 1.1 VM Configuration (make it look like a real machine)

```
VirtualBox / VMware settings:
├── OS:           Windows 10 Enterprise (64-bit)
├── RAM:          8 GB
├── CPU:          4 cores
├── Disk:         80 GB (dynamically allocated)
├── Network:      Host-Only (NO internet — ransomware cannot call home)
├── Resolution:   1920 × 1080
└── Snapshot:     Take CLEAN_BASELINE before ANY malware execution
```

**Inside the VM — make it look real:**
```batch
:: Rename computer to something realistic
Rename-Computer -NewName "DESKTOP-K7F2B9" -Force

:: Create realistic user files
mkdir C:\Users\Ahmed\Documents\Work
mkdir C:\Users\Ahmed\Documents\University
mkdir C:\Users\Ahmed\Pictures

:: Copy some real-looking files (PDFs, DOCX) into these folders
:: This gives ransomware something to encrypt — without it some samples exit early
```

### 1.2 Install Monitoring Tools in VM

```batch
:: Install Python 3.11 in the VM (for the collector script)
:: Download from python.org, add to PATH

:: Then install:
pip install psutil watchdog pandas pywin32
```

### 1.3 Snapshot Policy

```
BEFORE every test run:
    Machine → Take Snapshot → Name: "CLEAN_BASELINE"

AFTER every test run:
    Machine → Restore Snapshot → "CLEAN_BASELINE"

NEVER run two malware samples without restoring snapshot between them.
Cross-contamination corrupts your labels.
```

---

## Part 2 — Data Collection Script

Save this as `collector.py` inside the Windows VM.

```python
"""
collector.py — Behavioral Time-Series Collector for S004 LSTM Training
Run this INSIDE the Windows 10 VM before executing malware samples.
It collects per-process behavioral features every INTERVAL seconds.

Usage:
    python collector.py --label 0 --duration 120 --output benign_session_001.csv
    python collector.py --label 1 --duration 120 --output ransomware_wannacry_001.csv
"""

import psutil
import time
import csv
import os
import argparse
import json
from datetime import datetime
from collections import defaultdict

# Feature snapshot interval in seconds
INTERVAL   = 5      # collect features every 5 seconds
MIN_CPU    = 0.0    # minimum CPU% to include a process (set > 0 to skip idle)

FEATURES = [
    'timestamp', 'session_id', 'step_number', 'pid', 'process_name',
    'cpu_percent', 'memory_rss_mb', 'memory_vms_mb',
    'num_threads', 'num_handles', 'num_fds',
    'io_read_bytes_delta', 'io_write_bytes_delta',
    'io_read_count_delta', 'io_write_count_delta',
    'net_bytes_sent_delta', 'net_bytes_recv_delta',
    'num_connections', 'num_open_files',
    'cpu_user_time_delta', 'cpu_system_time_delta',
    'label',   # 0 = benign, 1 = ransomware
]


def get_net_io():
    net = psutil.net_io_counters()
    return net.bytes_sent, net.bytes_recv


def collect_snapshot(session_id, step, label, prev_io, prev_net):
    """Collect one snapshot of all running processes."""
    rows = []
    now  = time.time()
    curr_net_sent, curr_net_recv = get_net_io()

    net_sent_delta = curr_net_sent - prev_net[0]
    net_recv_delta = curr_net_recv - prev_net[1]

    for proc in psutil.process_iter(attrs=[
        'pid', 'name', 'cpu_percent', 'memory_info',
        'num_threads', 'io_counters', 'connections',
        'open_files', 'cpu_times',
    ]):
        try:
            info = proc.info
            pid  = info['pid']
            name = info['name'] or 'unknown'

            mem  = info['memory_info']
            rss  = mem.rss / (1024 * 1024) if mem else 0
            vms  = mem.vms / (1024 * 1024) if mem else 0

            io   = info.get('io_counters')
            prev = prev_io.get(pid, (0, 0, 0, 0))
            if io:
                read_b_d  = max(0, io.read_bytes  - prev[0])
                write_b_d = max(0, io.write_bytes - prev[1])
                read_c_d  = max(0, io.read_count  - prev[2])
                write_c_d = max(0, io.write_count - prev[3])
                prev_io[pid] = (io.read_bytes, io.write_bytes,
                                io.read_count, io.write_count)
            else:
                read_b_d = write_b_d = read_c_d = write_c_d = 0

            cpu_t   = info.get('cpu_times')
            cpu_td  = cpu_t.user   - prev_io.get(f'{pid}_cpu_user', 0) if cpu_t else 0
            cpu_sd  = cpu_t.system - prev_io.get(f'{pid}_cpu_sys',  0) if cpu_t else 0
            if cpu_t:
                prev_io[f'{pid}_cpu_user'] = cpu_t.user
                prev_io[f'{pid}_cpu_sys']  = cpu_t.system

            n_conn  = len(info.get('connections') or [])
            n_files = len(info.get('open_files')  or [])

            row = {
                'timestamp'          : now,
                'session_id'         : session_id,
                'step_number'        : step,
                'pid'                : pid,
                'process_name'       : name,
                'cpu_percent'        : info.get('cpu_percent', 0) or 0,
                'memory_rss_mb'      : rss,
                'memory_vms_mb'      : vms,
                'num_threads'        : info.get('num_threads', 0) or 0,
                'num_handles'        : getattr(proc, 'num_handles', lambda: 0)(),
                'num_fds'            : 0,  # Windows: use num_handles instead
                'io_read_bytes_delta': read_b_d,
                'io_write_bytes_delta': write_b_d,
                'io_read_count_delta': read_c_d,
                'io_write_count_delta': write_c_d,
                'net_bytes_sent_delta': net_sent_delta,
                'net_bytes_recv_delta': net_recv_delta,
                'num_connections'    : n_conn,
                'num_open_files'     : n_files,
                'cpu_user_time_delta': max(0, cpu_td),
                'cpu_system_time_delta': max(0, cpu_sd),
                'label'              : label,
            }
            rows.append(row)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return rows, (curr_net_sent, curr_net_recv)


def main():
    parser = argparse.ArgumentParser(description='S004 Behavioral Data Collector')
    parser.add_argument('--label',    type=int, required=True,
                        help='0=benign, 1=ransomware')
    parser.add_argument('--duration', type=int, default=120,
                        help='Collection duration in seconds')
    parser.add_argument('--interval', type=float, default=INTERVAL,
                        help='Snapshot interval in seconds')
    parser.add_argument('--output',   type=str, required=True,
                        help='Output CSV file path')
    args = parser.parse_args()

    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    steps      = int(args.duration / args.interval)
    prev_io    = {}
    prev_net   = get_net_io()

    print(f"[Collector] Session  : {session_id}")
    print(f"[Collector] Label    : {'RANSOMWARE' if args.label else 'BENIGN'}")
    print(f"[Collector] Duration : {args.duration}s  ({steps} steps)")
    print(f"[Collector] Interval : {args.interval}s")
    print(f"[Collector] Output   : {args.output}")
    print(f"[Collector] Starting in 3 seconds...")
    time.sleep(3)

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FEATURES)
        writer.writeheader()

        for step in range(steps):
            t_start = time.time()
            rows, prev_net = collect_snapshot(
                session_id, step, args.label, prev_io, prev_net)
            writer.writerows(rows)
            f.flush()

            elapsed = time.time() - t_start
            sleep   = max(0, args.interval - elapsed)
            print(f"  Step {step+1:3d}/{steps} — "
                  f"{len(rows)} processes captured in {elapsed:.2f}s", end='\r')
            time.sleep(sleep)

    print(f"\n[Collector] Done. {steps} steps saved → {args.output}")


if __name__ == '__main__':
    main()
```

---

## Part 3 — Collection Protocol

### 3.1 Collecting Benign Samples

```batch
:: Session 1: Normal office work simulation
:: Start collector FIRST, then simulate activity
python collector.py --label 0 --duration 300 --output benign_office_001.csv

:: While collector runs, simulate normal activity:
:: - Open and type in Notepad
:: - Browse files in Explorer
:: - Open/close documents
:: - Let Chrome run in background
```

```batch
:: Session 2: File compression (similar to some ransomware behavior — important negative!)
python collector.py --label 0 --duration 180 --output benign_compression_001.csv
:: While running: use 7-Zip to compress a large folder
```

```batch
:: Session 3: Antivirus scan (also looks like ransomware — important negative!)
python collector.py --label 0 --duration 300 --output benign_avscan_001.csv
:: While running: run Windows Defender full scan
```

### 3.2 Collecting Ransomware Samples

**CRITICAL: Always restore snapshot before each sample.**

```batch
:: Step 1: Restore CLEAN_BASELINE snapshot

:: Step 2: Start collector BEFORE detonating malware
python collector.py --label 1 --duration 180 --output ransomware_wannacry_001.csv

:: Step 3: In a SEPARATE terminal, detonate the malware:
:: (place malware sample in VM, then execute)
:: WannaCry, Ryuk, LockBit samples from MalwareBazaar

:: Step 4: After collection, immediately restore snapshot again
```

### 3.3 Recommended Collection Targets

| Category | Sessions | Duration | Notes |
|----------|----------|----------|-------|
| Normal office work | 20 | 5 min each | Diverse activities |
| File compression   | 10 | 3 min each | 7-Zip, WinRAR |
| AV scan            | 5  | 5 min each | Windows Defender |
| Cloud sync         | 5  | 3 min each | Simulate OneDrive |
| WannaCry           | 15 | 3 min each | Most documented |
| Ryuk               | 15 | 3 min each | Enterprise target |
| LockBit            | 15 | 3 min each | Speed champion |
| Cerber             | 10 | 3 min each | Older family |

**Target: 50+ benign sessions, 50+ ransomware sessions across ≥3 families**

---

## Part 4 — Data Schema for LSTM Input

Each collected CSV has this per-row schema. One row = one process at one timestamp.

```
timestamp              : Unix timestamp
session_id             : Unique ID for this collection run
step_number            : Which 5-second step (0, 1, 2, ...)
pid                    : Process ID
process_name           : Process name (explorer.exe, malware.exe, etc.)
cpu_percent            : CPU usage % at this step
memory_rss_mb          : Physical memory usage (MB)
memory_vms_mb          : Virtual memory usage (MB)
num_threads            : Thread count
num_handles            : Handle count (Windows)
io_read_bytes_delta    : Bytes read since last step (DELTA, not cumulative)
io_write_bytes_delta   : Bytes written since last step
io_read_count_delta    : Read operations since last step
io_write_count_delta   : Write operations since last step
net_bytes_sent_delta   : Network bytes sent since last step
net_bytes_recv_delta   : Network bytes received since last step
num_connections        : Active network connections
num_open_files         : Open file handles
cpu_user_time_delta    : User-mode CPU time since last step
cpu_system_time_delta  : System-mode CPU time since last step
label                  : 0 = benign, 1 = ransomware
```

---

## Part 5 — Converting to LSTM Sequences

Once you have raw CSVs, use this script to convert to LSTM-ready sequences.

```python
"""
prepare_sequences.py — Convert behavioral CSVs to LSTM sequences

Usage:
    python prepare_sequences.py \
        --input_dir data/behavioral_raw/ \
        --output    data/behavioral_sequences.npz \
        --seq_len   10 \
        --pid_focus suspicious  # 'all' or 'suspicious'
"""

import os, glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLS = [
    'cpu_percent', 'memory_rss_mb', 'memory_vms_mb',
    'num_threads', 'num_handles',
    'io_read_bytes_delta', 'io_write_bytes_delta',
    'io_read_count_delta', 'io_write_count_delta',
    'net_bytes_sent_delta', 'net_bytes_recv_delta',
    'num_connections', 'num_open_files',
    'cpu_user_time_delta', 'cpu_system_time_delta',
]

def load_all_sessions(input_dir: str) -> pd.DataFrame:
    """Load and concatenate all CSV files from input_dir."""
    csvs  = glob.glob(os.path.join(input_dir, '*.csv'))
    dfs   = []
    for path in csvs:
        df = pd.read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_sequences(
    df       : pd.DataFrame,
    seq_len  : int = 10,
    step     : int = 1,
) -> tuple:
    """
    Build (X_sequences, y_sequences) from a behavioral DataFrame.

    Strategy: for each unique (session_id, pid) pair, extract sliding
    windows of seq_len consecutive steps. Label the sequence by the
    MAJORITY label across the window.

    Returns:
        X : shape (n_sequences, seq_len, n_features)
        y : shape (n_sequences,)
    """
    sequences = []
    labels    = []

    for (session, pid), group in df.groupby(['session_id', 'pid']):
        group = group.sort_values('step_number').reset_index(drop=True)
        feat  = group[FEATURE_COLS].fillna(0).values
        labs  = group['label'].values

        if len(feat) < seq_len:
            continue  # too short for a sequence

        for i in range(0, len(feat) - seq_len + 1, step):
            window = feat[i : i + seq_len]
            label  = int(np.round(np.mean(labs[i : i + seq_len])))
            sequences.append(window)
            labels.append(label)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels,    dtype=np.int32)
    return X, y


def main(input_dir, output_path, seq_len=10):
    print(f"Loading behavioral sessions from: {input_dir}")
    df = load_all_sessions(input_dir)
    print(f"Total rows: {len(df):,}")
    print(f"Sessions  : {df['session_id'].nunique()}")
    print(f"Label dist: {dict(df['label'].value_counts())}")

    # Normalize features
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS].fillna(0))

    # Build sequences
    print(f"\nBuilding sequences (seq_len={seq_len})...")
    X, y = build_sequences(df, seq_len=seq_len)

    print(f"Sequences shape: {X.shape}")
    print(f"Label dist     : {dict(zip(*np.unique(y, return_counts=True)))}")

    # Save
    np.savez_compressed(output_path, X=X, y=y)
    joblib.dump(scaler, output_path.replace('.npz', '_scaler.pkl'))
    print(f"\nSaved → {output_path}")
    print(f"Saved → {output_path.replace('.npz', '_scaler.pkl')}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output',    required=True)
    parser.add_argument('--seq_len',   type=int, default=10)
    args = parser.parse_args()
    main(args.input_dir, args.output, args.seq_len)
```

---

## Part 6 — Training the LSTM on Real Sequences

Once you have `behavioral_sequences.npz`, replace the static-data LSTM call in
the notebook with this:

```python
# Load behavioral sequences
data = np.load('data/behavioral_sequences.npz')
X_seq = data['X']   # shape: (n_sequences, 10, 15)
y_seq = data['y']   # shape: (n_sequences,)

# Use the existing RansomwareLSTM class from lstm_model.py
# But call it differently — pass pre-built sequences directly

from src.models.lstm_model import RansomwareLSTM

lstm = RansomwareLSTM(
    sequence_length = 10,
    lstm_units      = 64,
    latent_dim      = 16,
    epochs          = 100,
)

# The model's train() method still works — it builds sequences internally
# But if you want to use pre-built sequences directly:
lstm.feature_names = [
    'cpu_percent', 'memory_rss_mb', 'memory_vms_mb',
    'num_threads', 'num_handles',
    'io_read_bytes_delta', 'io_write_bytes_delta',
    'io_read_count_delta', 'io_write_count_delta',
    'net_bytes_sent_delta', 'net_bytes_recv_delta',
    'num_connections', 'num_open_files',
    'cpu_user_time_delta', 'cpu_system_time_delta',
]
lstm.n_features = len(lstm.feature_names)

# Now pass the pre-built dataframe version to lstm.train()
# or directly train the Keras model with X_seq / y_seq
```

---

## Part 7 — Minimum Dataset Size for LSTM

| Condition | Minimum |
|-----------|---------|
| Sessions per class | 30+ |
| Steps per session | 20+ |
| Ransomware families | 3+ |
| Total sequences after windowing | 5,000+ |
| Recommended for research quality | 20,000+ sequences |

With 50 sessions × 36 steps × sliding window of 10 with step 1:
50 × (36 - 10 + 1) sequences = 50 × 27 = **1,350 sequences per class minimum**.
Aim for 10× this: 100 sessions per class.

---

## Summary Checklist

```
□  Windows 10 VM configured (8GB RAM, 4 cores, Host-Only network)
□  CLEAN_BASELINE snapshot taken
□  collector.py installed and tested in VM
□  20+ benign sessions collected (office, compression, AV scan)
□  15+ ransomware sessions collected per family (WannaCry, Ryuk, LockBit)
□  prepare_sequences.py run on all CSVs → behavioral_sequences.npz
□  LSTM retrained on real sequences (not static CIC-MalMem2022 rows)
□  LSTM now achieves ROC-AUC >> 0.65 (was 0.50 on static data)
```
