#!/usr/bin/env python3
"""
Safe terminal demo runner for the Hybrid Ransomware EDR project.

This script is intended for your video/demo. It shows real-time style EDR logs
without requiring the full research notebook.

Recommended demo commands
-------------------------
# 1) Live benign process, after opening Notepad manually:
python demo_realtime_edr.py --mode live --name notepad.exe --duration 30

# Linux example:
python demo_realtime_edr.py --mode live --name gedit --duration 30

# 2) Generated hard-benign workload on dummy files:
python demo_realtime_edr.py --scenario hard_benign_dummy --duration 35

# 3) Safe ransomware-behavior dummy on generated demo files:
python demo_realtime_edr.py --scenario ransomware_dummy --duration 35

# 4) Prepared Layer-1 memory-forensics scenario:
python demo_realtime_edr.py --scenario memory_critical

Safety
------
By default this script runs in DRY-RUN mode. It prints actions such as
"WOULD SUSPEND" but does not suspend/kill processes.

If --allow-demo-mitigation is used, real mitigation is allowed ONLY for the
controlled dummy ransomware child process started by this script. The script
will refuse to mitigate protected OS processes.

Do not run real ransomware. The ransomware_dummy scenario only works inside a
controlled demo_workspace/ folder and writes harmless transformed copies.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
import subprocess
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

# Project imports. These should exist when the file is placed in the project root.
try:
    from src.models.cnn_model import RansomwareCNN
    from src.models.data_loader import RAW_FEATURE_COLUMNS, engineer_features
    from src.engine.edr_orchestrator import EDROrchestrator, RiskState
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import project modules. Place this script in the project root "
        "beside the src/ folder and run it from there.\n"
        f"Import error: {exc}"
    )

try:
    from src.models.random_forest_model import RansomwareRandomForest
except Exception:  # pragma: no cover
    RansomwareRandomForest = None

CNN_FEATURE_COLS = [
    "cpu_percent", "memory_rss_mb", "memory_vms_mb",
    "io_read_bytes_delta", "io_write_bytes_delta",
    "net_bytes_sent_delta", "num_open_files",
    "write_read_ratio", "cpu_x_write", "io_write_intensity",
]

PROTECTED_WINDOWS = {
    "system", "registry", "smss.exe", "csrss.exe", "wininit.exe", "winlogon.exe",
    "services.exe", "lsass.exe", "svchost.exe", "explorer.exe", "dwm.exe",
    "taskhostw.exe", "conhost.exe", "fontdrvhost.exe", "sihost.exe",
}
PROTECTED_LINUX = {
    "init", "systemd", "sshd", "bash", "zsh", "sh", "sudo", "gnome-shell",
    "xorg", "wayland", "dbus-daemon", "networkmanager", "containerd", "dockerd",
}


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


class ConstantStaticModel:
    """Fallback Layer-1 prior when RF artifacts are unavailable.

    This fallback exists only to keep the terminal demo runnable. It does not
    replace the real Layer-1 model used in the notebook/report.
    """

    def __init__(self, feature_names: List[str], probability: float = 0.12):
        self.feature_names = list(feature_names)
        self.threshold = 0.10
        self.probability = float(probability)
        self.scaler = None

    def predict(self, features: Dict[str, float], pid: int = 0, process_name: str = "unknown"):
        from src.engine.system_event import SystemEvent, SEVERITY_NONE, ACTION_LOG_EVENT
        return SystemEvent(
            alert=False,
            severity=SEVERITY_NONE,
            model_source="Demo_StaticPrior",
            confidence=self.probability,
            pid=pid,
            process_name=process_name,
            recommended_actions=[ACTION_LOG_EVENT],
            features={"demo_static_prior": self.probability},
            description="Demo static prior used because the Layer-1 RF artifact was not loaded.",
        )


def add_derived(tick: Dict[str, float]) -> Dict[str, float]:
    tick = dict(tick)
    read_b = float(tick.get("io_read_bytes_delta", 0.0))
    write_b = float(tick.get("io_write_bytes_delta", 0.0))
    cpu = float(tick.get("cpu_percent", 0.0))
    mem = max(float(tick.get("memory_rss_mb", 1.0)), 1.0)
    tick["write_read_ratio"] = write_b / (read_b + 1.0)
    tick["cpu_x_write"] = cpu * write_b
    tick["io_write_intensity"] = write_b / (mem * 1024.0 + 1.0)
    return tick


def load_cnn() -> RansomwareCNN:
    model_dir = PROJECT_ROOT / "saved_models" / "cnn"
    scaler_path = PROJECT_ROOT / "saved_models" / "cnn_scaler.pkl"
    if not model_dir.exists():
        raise FileNotFoundError(f"CNN model directory not found: {model_dir}")
    cnn = RansomwareCNN.load(str(model_dir), scaler_path=str(scaler_path))
    if getattr(cnn, "feature_cols", None) is None:
        cnn.feature_cols = CNN_FEATURE_COLS
    return cnn


def _build_static_dataframe(kind: str = "median") -> Tuple[List[str], np.ndarray, Dict[str, float]]:
    data_path = PROJECT_ROOT / "data" / "datasets" / "MalMem2022.csv"
    if not data_path.exists():
        names = [f"static_feature_{i}" for i in range(69)]
        arr = np.zeros(len(names), dtype=np.float32)
        return names, arr, {n: 0.0 for n in names}

    df = pd.read_csv(data_path)
    raw_cols = [c for c in RAW_FEATURE_COLUMNS if c in df.columns]
    raw_df = df[raw_cols].copy()

    if kind == "ransomware" and "Category" in df.columns:
        mask = df["Category"].astype(str).str.contains("ransomware", case=False, na=False)
        raw_row = raw_df[mask].iloc[[0]].copy() if mask.any() else raw_df.median(numeric_only=True).to_frame().T
    elif kind == "high":
        raw_row = raw_df.quantile(0.75, numeric_only=True).to_frame().T
    else:
        raw_row = raw_df.median(numeric_only=True).to_frame().T

    engineered = engineer_features(raw_row)
    drop_cols = [c for c in ["Class", "Category", "Filename", "Label"] if c in engineered.columns]
    if drop_cols:
        engineered = engineered.drop(columns=drop_cols)
    engineered = engineered.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    feature_names = list(engineered.columns)
    row_dict = {c: float(engineered.iloc[0][c]) for c in feature_names}
    return feature_names, engineered.iloc[0].astype(float).values.astype(np.float32), row_dict


def load_static_model_and_vector(kind: str = "median"):
    feature_names, static_vec, static_dict = _build_static_dataframe(kind)
    candidates = [
        (PROJECT_ROOT / "saved_models" / "random_forest_tuned.pkl", PROJECT_ROOT / "saved_models" / "rf_tuned_scaler.pkl"),
        (PROJECT_ROOT / "saved_models" / "random_forest.pkl", PROJECT_ROOT / "saved_models" / "rf_scaler.pkl"),
    ]
    static_model = None
    if RansomwareRandomForest is not None:
        for model_path, scaler_path in candidates:
            if model_path.exists() and scaler_path.exists():
                try:
                    rf = RansomwareRandomForest()
                    rf.load(str(model_path), str(scaler_path))
                    static_model = rf
                    log(f"Loaded Layer 1 model: {model_path.name}")
                    break
                except Exception as e:
                    log(f"Could not load {model_path.name}: {e}")

    if static_model is None:
        log("Layer 1 RF artifact not loaded; using WATCH-level demo static prior.")
        static_model = ConstantStaticModel(feature_names=feature_names, probability=0.12)

    # Align vector to the model's exact expected feature names when available.
    model_names = getattr(static_model, "feature_names", None)
    if model_names:
        aligned = np.array([static_dict.get(name, 0.0) for name in model_names], dtype=np.float32)
        static_vec = aligned
        log(f"Layer 1 static vector aligned to {len(model_names)} features.")
    else:
        log(f"Layer 1 static vector uses {len(static_vec)} features.")

    return static_model, static_vec


def build_orchestrator(static_kind: str = "median") -> Tuple[EDROrchestrator, np.ndarray]:
    cnn = load_cnn()
    static_model, static_vec = load_static_model_and_vector(static_kind)
    orchestrator = EDROrchestrator(
        static_model=static_model,
        cnn_model=cnn,
        policy_mode="balanced",
        require_encryption_evidence=True,
        history_max_steps=30,
        verbose=False,
    )
    return orchestrator, static_vec


def scenario_notepad(n: int = 40, seed: int = 1) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rows.append(add_derived({
            "cpu_percent": float(rng.uniform(0.5, 5.0)),
            "memory_rss_mb": float(rng.uniform(40, 90)),
            "memory_vms_mb": float(rng.uniform(90, 180)),
            "io_read_bytes_delta": float(rng.uniform(0, 4_000)),
            "io_write_bytes_delta": float(rng.uniform(0, 6_000)),
            "net_bytes_sent_delta": float(rng.uniform(0, 500)),
            "num_open_files": float(rng.integers(0, 4)),
        }))
    return rows


def csv_stream(path: Path) -> List[Dict[str, float]]:
    df = pd.read_csv(path)
    if "step_number" in df.columns:
        df = df.sort_values("step_number")
    rows: List[Dict[str, float]] = []
    base_cols = [c for c in CNN_FEATURE_COLS if c not in {"write_read_ratio", "cpu_x_write", "io_write_intensity"}]
    for _, row in df.iterrows():
        tick = {c: float(row[c]) if c in df.columns and pd.notna(row[c]) else 0.0 for c in base_cols}
        rows.append(add_derived(tick))
    return rows


def create_demo_files(root: Path, n_files: int = 80, size_bytes: int = 16_384) -> Path:
    input_dir = root / "input_files"
    output_dir = root / "output_files"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    for i in range(n_files):
        p = input_dir / f"document_{i:03d}.txt"
        if not p.exists() or p.stat().st_size < size_bytes:
            data = rng.integers(32, 126, size=size_bytes, dtype=np.uint8).tobytes()
            p.write_bytes(data)
    return root


def write_hard_benign_worker(workspace: Path, duration: int) -> Path:
    """Create a benign high-I/O worker.

    This simulates backup/copy/compression-like behavior. It reads files and
    writes benign copies/bundles inside the demo workspace only. It does not
    encrypt, rename original files, persist, or touch user folders.
    """
    worker = workspace / "hard_benign_worker.py"
    code = f"""
import time
from pathlib import Path

root = Path({str(workspace)!r})
input_dir = root / "input_files"
output_dir = root / "hard_benign_output"
output_dir.mkdir(exist_ok=True)
end = time.time() + {int(duration)}
files = list(input_dir.glob("*.txt"))
round_id = 0
while time.time() < end:
    round_id += 1
    bundle = output_dir / f"backup_bundle_{{round_id:04d}}.bin"
    try:
        with bundle.open("wb") as out:
            for p in files:
                data = p.read_bytes()
                # benign copy/backup behavior: write data as-is with separator
                out.write(data)
                out.write(b"\n---FILE_SEPARATOR---\n")
                time.sleep(0.01)
    except Exception:
        pass
    time.sleep(0.15)
"""
    worker.write_text(code, encoding="utf-8")
    return worker


def launch_dummy_hard_benign(duration: int = 30) -> subprocess.Popen:
    workspace = create_demo_files(PROJECT_ROOT / "demo_workspace_hard_benign", n_files=60, size_bytes=8192)
    worker = write_hard_benign_worker(workspace, duration)
    log(f"Prepared hard benign demo workspace: {workspace}")
    proc = subprocess.Popen([sys.executable, str(worker)], cwd=str(PROJECT_ROOT))
    log(f"Started controlled hard benign process PID={proc.pid}")
    return proc


def write_dummy_ransomware_worker(workspace: Path, duration: int) -> Path:
    worker = workspace / "ransomware_dummy_worker.py"
    code = f'''
import time
from pathlib import Path

root = Path({str(workspace)!r})
input_dir = root / "input_files"
output_dir = root / "output_files"
output_dir.mkdir(exist_ok=True)
end = time.time() + {int(duration)}
key = 0x5A
round_id = 0
files = list(input_dir.glob("*.txt"))
while time.time() < end:
    round_id += 1
    for p in files:
        try:
            data = p.read_bytes()
            # harmless CPU/file transformation: writes transformed copy only
            out = bytes((b ^ key) for b in data)
            (output_dir / (p.stem + f".{{round_id}}.simenc")).write_bytes(out)
        except Exception:
            pass
        # brief pause keeps the demo readable and avoids overloading the VM
        time.sleep(0.005)
'''
    worker.write_text(code, encoding="utf-8")
    return worker


def launch_dummy_ransomware(duration: int = 30) -> subprocess.Popen:
    workspace = create_demo_files(PROJECT_ROOT / "demo_workspace")
    worker = write_dummy_ransomware_worker(workspace, duration)
    log(f"Prepared safe dummy ransomware workspace: {workspace}")
    proc = subprocess.Popen([sys.executable, str(worker)], cwd=str(PROJECT_ROOT))
    log(f"Started controlled dummy ransomware process PID={proc.pid}")
    return proc


def find_process_by_name(name: str):
    if psutil is None:
        raise RuntimeError("psutil is not installed. Install it with: pip install psutil")
    target = name.lower()
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if (p.info.get("name") or "").lower() == target:
                return p
        except Exception:
            continue
    return None


def process_tick(proc, interval: float) -> Optional[Dict[str, float]]:
    if psutil is None:
        raise RuntimeError("psutil is required for live mode")
    try:
        # Store previous IO counters on the Process object wrapper.
        prev_io = getattr(proc, "_demo_prev_io", None)
        cpu = proc.cpu_percent(interval=interval)
        mem = proc.memory_info()
        rss = mem.rss / (1024 * 1024)
        vms = mem.vms / (1024 * 1024)
        try:
            io = proc.io_counters()
            rd = max(0, io.read_bytes - (prev_io.read_bytes if prev_io else io.read_bytes))
            wr = max(0, io.write_bytes - (prev_io.write_bytes if prev_io else io.write_bytes))
            proc._demo_prev_io = io
        except Exception:
            rd = wr = 0
        try:
            open_files = len(proc.open_files())
        except Exception:
            open_files = 0
        return add_derived({
            "cpu_percent": float(cpu),
            "memory_rss_mb": float(rss),
            "memory_vms_mb": float(vms),
            "io_read_bytes_delta": float(rd),
            "io_write_bytes_delta": float(wr),
            "net_bytes_sent_delta": 0.0,
            "num_open_files": float(open_files),
        })
    except psutil.NoSuchProcess:
        return None
    except Exception as exc:
        log(f"Could not sample process: {exc}")
        return None


def is_protected_process(name: str) -> bool:
    n = (name or "").lower()
    return n in PROTECTED_WINDOWS or n in PROTECTED_LINUX


def apply_or_print_actions(event, allow_demo_mitigation: bool, controlled_pid: Optional[int], target_pid: Optional[int], target_name: str):
    actions = list(getattr(event, "recommended_actions", []) or [])
    if not actions:
        log("Action: none")
        return

    for action in actions:
        if action in {"log_event", "monitor_only", "send_alert"}:
            log(f"Action: {action}")
            continue

        if not allow_demo_mitigation:
            log(f"DRY-RUN: WOULD {action} (mitigation disabled)")
            continue

        if target_pid is None or target_pid != controlled_pid:
            log(f"REFUSED {action}: only controlled dummy PID can be mitigated in demo mode")
            continue

        if is_protected_process(target_name):
            log(f"REFUSED {action}: protected process name {target_name}")
            continue

        if psutil is None:
            log(f"DRY-RUN: psutil unavailable for {action}")
            continue

        try:
            p = psutil.Process(target_pid)
            if action == "suspend_process":
                p.suspend()
                log(f"MITIGATION APPLIED: suspended controlled PID={target_pid}")
            elif action == "kill_process":
                p.terminate()
                log(f"MITIGATION APPLIED: terminated controlled PID={target_pid}")
            else:
                log(f"DRY-RUN: WOULD {action} (not implemented by demo script)")
        except Exception as exc:
            log(f"Could not apply {action}: {exc}")


def summarize_event(event) -> str:
    feats = getattr(event, "features", {}) or {}
    return (
        f"L1={feats.get('risk_state','?')} "
        f"L1p={float(feats.get('l1_probability_effective', feats.get('l1_probability', 0.0))):.3f} "
        f"L2max={float(feats.get('max_l2_probability', feats.get('l2_probability', 0.0))):.3f} "
        f"risk={float(feats.get('max_l2_risk_score', feats.get('l2_risk_score', 0.0))):.2f} "
        f"streak={int(feats.get('max_evidence_streak', feats.get('evidence_streak', 0)) or 0)} "
        f"specific={int(feats.get('max_specific_score', feats.get('specific_score', 0)) or 0)} "
        f"source={getattr(event, 'model_source', '?')} alert={getattr(event, 'alert', False)}"
    )


def evaluate_prefix(orchestrator: EDROrchestrator, static_vec: np.ndarray, prefix: List[Dict[str, float]], pid: int, name: str):
    return orchestrator.evaluate(
        static_vec,
        prefix,
        pid=pid,
        process_name=name,
        feature_cols=CNN_FEATURE_COLS,
    )


def run_replay(orchestrator: EDROrchestrator, static_vec: np.ndarray, stream: List[Dict[str, float]],
               process_name: str, pid: int, delay: float, allow_demo_mitigation: bool = False,
               controlled_pid: Optional[int] = None):
    log(f"Starting replay for {process_name} PID={pid} ticks={len(stream)}")
    prefix: List[Dict[str, float]] = []
    event = None
    for i, tick in enumerate(stream, start=1):
        prefix.append(tick)
        if i < orchestrator.cnn_model.window_size:
            log(f"tick={i:03d} warmup cpu={tick['cpu_percent']:.1f}% read={tick['io_read_bytes_delta']:.0f} write={tick['io_write_bytes_delta']:.0f}")
            time.sleep(delay)
            continue
        event = evaluate_prefix(orchestrator, static_vec, prefix, pid, process_name)
        log(f"tick={i:03d} {summarize_event(event)}")
        if event.alert:
            break
        time.sleep(delay)

    if event is None:
        event = evaluate_prefix(orchestrator, static_vec, prefix or stream, pid, process_name)

    print_final_event(event, allow_demo_mitigation, controlled_pid, pid, process_name)
    return event


def run_live_process(orchestrator: EDROrchestrator, static_vec: np.ndarray, proc, duration: int, interval: float,
                     allow_demo_mitigation: bool, controlled_pid: Optional[int] = None):
    name = proc.name()
    pid = proc.pid
    log(f"Starting LIVE monitor for {name} PID={pid} duration={duration}s")
    prefix: List[Dict[str, float]] = []
    event = None
    start = time.time()

    while time.time() - start < duration:
        tick = process_tick(proc, interval)
        if tick is None:
            log("Process ended or could not be sampled.")
            break
        prefix.append(tick)
        if len(prefix) < orchestrator.cnn_model.window_size:
            log(f"tick={len(prefix):03d} warmup cpu={tick['cpu_percent']:.1f}% read={tick['io_read_bytes_delta']:.0f} write={tick['io_write_bytes_delta']:.0f}")
            continue
        event = evaluate_prefix(orchestrator, static_vec, prefix, pid, name)
        log(f"tick={len(prefix):03d} {summarize_event(event)}")
        if event.alert:
            break

    if event is None:
        event = evaluate_prefix(orchestrator, static_vec, prefix, pid, name)
    print_final_event(event, allow_demo_mitigation, controlled_pid, pid, name)
    return event


def print_final_event(event, allow_demo_mitigation: bool, controlled_pid: Optional[int], pid: Optional[int], name: str):
    log("=" * 78)
    log(f"FINAL DECISION: alert={event.alert} severity={event.severity} source={event.model_source}")
    log(f"confidence={event.confidence} actions={event.recommended_actions}")
    log(f"description={event.description}")
    apply_or_print_actions(event, allow_demo_mitigation, controlled_pid, pid, name)
    log("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe real-time terminal demo for the Hybrid Ransomware EDR")
    parser.add_argument("--mode", choices=["scenario", "live"], default="scenario")
    parser.add_argument("--scenario", choices=["notepad", "hard_benign_dummy", "ransomware_dummy", "memory_critical"], default="notepad")
    parser.add_argument("--name", type=str, help="Process name for live mode, e.g. notepad.exe")
    parser.add_argument("--duration", type=int, default=35, help="Duration for live/dummy scenario")
    parser.add_argument("--interval", type=float, default=1.0, help="Live sampling interval in seconds")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay for generated scenario ticks")
    parser.add_argument("--allow-demo-mitigation", action="store_true", help="Allow mitigation only for controlled dummy child process")
    args = parser.parse_args()

    log("Hybrid Ransomware EDR terminal demo")
    log("Default mode is safe DRY-RUN. No real malware is used.")
    log("Layer 2 runs live/generated telemetry; Layer 1 uses saved RF or a demo prior.")

    static_kind = "ransomware" if args.scenario == "memory_critical" else "median"
    orchestrator, static_vec = build_orchestrator(static_kind=static_kind)

    if args.mode == "scenario":
        if args.scenario == "notepad":
            stream = scenario_notepad()
            run_replay(orchestrator, static_vec, stream, "notepad_demo.exe", 4242, args.delay)
        elif args.scenario == "memory_critical":
            # Quiet behavior + ransomware memory row: demonstrates Layer 1.
            stream = scenario_notepad(n=12)
            run_replay(orchestrator, static_vec, stream, "memory_ransomware_row", 4243, args.delay)
        elif args.scenario == "hard_benign_dummy":
            if psutil is None:
                raise SystemExit("psutil is required for hard_benign_dummy scenario. pip install psutil")
            child = launch_dummy_hard_benign(duration=args.duration)
            try:
                proc = psutil.Process(child.pid)
                run_live_process(
                    orchestrator, static_vec, proc,
                    duration=args.duration,
                    interval=args.interval,
                    allow_demo_mitigation=False,
                    controlled_pid=None,
                )
            finally:
                try:
                    if child.poll() is None:
                        child.terminate()
                except Exception:
                    pass
        elif args.scenario == "ransomware_dummy":
            if psutil is None:
                raise SystemExit("psutil is required for ransomware_dummy scenario. pip install psutil")
            child = launch_dummy_ransomware(duration=args.duration)
            try:
                proc = psutil.Process(child.pid)
                run_live_process(
                    orchestrator, static_vec, proc,
                    duration=args.duration,
                    interval=args.interval,
                    allow_demo_mitigation=args.allow_demo_mitigation,
                    controlled_pid=child.pid,
                )
            finally:
                # If not mitigated, stop the child at the end of the demo.
                try:
                    if child.poll() is None:
                        child.terminate()
                except Exception:
                    pass

    elif args.mode == "live":
        if not args.name:
            raise SystemExit("--name is required for live mode")
        if psutil is None:
            raise SystemExit("psutil is required for live mode. pip install psutil")
        proc = find_process_by_name(args.name)
        if proc is None:
            raise SystemExit(f"Process not found: {args.name}")
        # Live mode never applies real mitigation to arbitrary processes.
        run_live_process(orchestrator, static_vec, proc, args.duration, args.interval, allow_demo_mitigation=False)


if __name__ == "__main__":
    main()
