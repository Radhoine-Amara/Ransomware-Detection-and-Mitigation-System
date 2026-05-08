#!/usr/bin/env python3
"""
Terminal demo runner for the Hybrid Ransomware EDR project (V8.3).

Purpose
-------
This script is designed for a safe class/video demo. It prints real-time style
EDR logs in the terminal without requiring a full UI. By default it runs in
DRY-RUN mode: it prints what the EDR would do, but it does not suspend/kill
processes. Real mitigation is only allowed for explicitly controlled demo PIDs.

Examples
--------
python demo_realtime_edr.py --scenario notepad
python demo_realtime_edr.py --scenario ransomware
python demo_realtime_edr.py --scenario memory_critical
python demo_realtime_edr.py --mode replay --csv data/behavioral_raw/data/benign_021.csv
python demo_realtime_edr.py --mode live --name notepad.exe --duration 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
import signal
from pathlib import Path
from typing import Dict, List, Optional

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
except Exception:
    psutil = None

from src.models.cnn_model import RansomwareCNN
from src.models.data_loader import RAW_FEATURE_COLUMNS, engineer_features
from src.engine.edr_orchestrator import EDROrchestrator

try:
    from src.models.random_forest_model import RansomwareRandomForest
except Exception:
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
}
PROTECTED_LINUX = {"init", "systemd", "sshd", "bash", "zsh", "gnome-shell", "xorg", "wayland"}


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


class ConstantStaticModel:
    """Fallback Layer 1 prior when saved RF artifacts are unavailable.

    This keeps the terminal demo runnable. It does NOT replace Layer 1 in the
    notebook/report; it simply allows Layer 2/orchestrator behavior to be shown
    in a safe terminal demonstration.
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
            description="Demo static prior used because RF artifact was not loaded.",
        )


def load_cnn() -> RansomwareCNN:
    model_dir = PROJECT_ROOT / "saved_models" / "cnn"
    scaler_path = PROJECT_ROOT / "saved_models" / "cnn_scaler.pkl"
    if not model_dir.exists():
        raise FileNotFoundError(f"CNN model directory not found: {model_dir}")
    cnn = RansomwareCNN.load(str(model_dir), scaler_path=str(scaler_path))
    if cnn.feature_cols is None:
        cnn.feature_cols = CNN_FEATURE_COLS
    return cnn


def load_static_model(feature_names: List[str]):
    # Prefer tuned RF if the user added it locally; otherwise fallback.
    candidates = [
        (PROJECT_ROOT / "saved_models" / "random_forest_tuned.pkl", PROJECT_ROOT / "saved_models" / "rf_tuned_scaler.pkl"),
        (PROJECT_ROOT / "saved_models" / "random_forest.pkl", PROJECT_ROOT / "saved_models" / "rf_scaler.pkl"),
    ]
    if RansomwareRandomForest is not None:
        for model_path, scaler_path in candidates:
            if model_path.exists() and scaler_path.exists():
                try:
                    rf = RansomwareRandomForest()
                    rf.load(str(model_path), str(scaler_path))
                    log(f"Loaded Layer 1 static model: {model_path.name}")
                    return rf
                except Exception as e:
                    log(f"Could not load {model_path.name}: {e}")
    log("Layer 1 RF artifact not loaded; using WATCH-level demo static prior.")
    return ConstantStaticModel(feature_names=feature_names, probability=0.12)


def load_static_schema_and_vector(kind: str = "median") -> tuple[List[str], np.ndarray]:
    data_path = PROJECT_ROOT / "data" / "datasets" / "MalMem2022.csv"
    if not data_path.exists():
        # Fallback schema with 69 zeros; only works with ConstantStaticModel.
        names = [f"static_feature_{i}" for i in range(69)]
        return names, np.zeros(len(names), dtype=np.float32)

    df = pd.read_csv(data_path)
    raw_cols = [c for c in RAW_FEATURE_COLUMNS if c in df.columns]
    raw_df = df[raw_cols].copy()
    if kind == "ransomware" and "Category" in df.columns:
        mask = df["Category"].astype(str).str.contains("ransomware", case=False, na=False)
        if mask.any():
            raw_row = raw_df[mask].iloc[[0]].copy()
        else:
            raw_row = raw_df.median(numeric_only=True).to_frame().T
    elif kind == "high":
        raw_row = raw_df.quantile(0.75, numeric_only=True).to_frame().T
    else:
        raw_row = raw_df.median(numeric_only=True).to_frame().T

    engineered = engineer_features(raw_row)
    # Remove known non-feature metadata if present.
    drop_cols = [c for c in ["Class", "Category", "Filename"] if c in engineered.columns]
    if drop_cols:
        engineered = engineered.drop(columns=drop_cols)
    feature_names = list(engineered.columns)
    return feature_names, engineered.iloc[0].astype(float).values.astype(np.float32)


def add_derived(tick: Dict[str, float]) -> Dict[str, float]:
    tick = dict(tick)
    read_b = float(tick.get("io_read_bytes_delta", 0.0))
    write_b = float(tick.get("io_write_bytes_delta", 0.0))
    cpu = float(tick.get("cpu_percent", 0.0))
    mem = float(tick.get("memory_rss_mb", 1.0))
    tick["write_read_ratio"] = write_b / (read_b + 1.0)
    tick["cpu_x_write"] = cpu * write_b
    tick["io_write_intensity"] = write_b / (mem * 1024.0 + 1.0)
    return tick


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


def scenario_ransomware(n: int = 45, encrypt_start: int = 8, seed: int = 7) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        if i < encrypt_start:
            rows.append(add_derived({
                "cpu_percent": float(rng.uniform(2, 10)),
                "memory_rss_mb": float(rng.uniform(60, 110)),
                "memory_vms_mb": float(rng.uniform(120, 220)),
                "io_read_bytes_delta": float(rng.uniform(0, 20_000)),
                "io_write_bytes_delta": float(rng.uniform(0, 25_000)),
                "net_bytes_sent_delta": float(rng.uniform(0, 3_000)),
                "num_open_files": float(rng.integers(1, 8)),
            }))
        else:
            read = float(rng.uniform(1_000_000, 5_000_000))
            write = float(rng.uniform(2_000_000, 9_000_000))
            rows.append(add_derived({
                "cpu_percent": float(rng.uniform(70, 98)),
                "memory_rss_mb": float(rng.uniform(70, 130)),
                "memory_vms_mb": float(rng.uniform(160, 280)),
                "io_read_bytes_delta": read,
                "io_write_bytes_delta": write,
                "net_bytes_sent_delta": float(rng.uniform(0, 20_000)),
                "num_open_files": float(rng.integers(80, 240)),
            }))
    return rows


def csv_stream(path: Path) -> List[Dict[str, float]]:
    df = pd.read_csv(path)
    if "step_number" in df.columns:
        df = df.sort_values("step_number")
    rows = []
    for _, r in df.iterrows():
        tick = {c: float(r[c]) if c in df.columns and pd.notna(r[c]) else 0.0 for c in CNN_FEATURE_COLS if c not in {"write_read_ratio", "cpu_x_write", "io_write_intensity"}}
        rows.append(add_derived(tick))
    return rows


def find_process(name: str):
    if psutil is None:
        raise RuntimeError("psutil is not installed. Use --mode replay or --scenario instead.")
    name = name.lower()
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if (p.info.get("name") or "").lower() == name:
                return p
        except Exception:
            continue
    return None


def live_stream(process_name: str, duration: int, interval: float) -> List[Dict[str, float]]:
    if psutil is None:
        raise RuntimeError("psutil required for live mode. pip install psutil")
    proc = find_process(process_name)
    if proc is None:
        raise RuntimeError(f"Process not found: {process_name}")
    log(f"Monitoring live process {proc.name()} PID={proc.pid} for {duration}s")
    rows = []
    try:
        prev_io = proc.io_counters() if hasattr(proc, "io_counters") else None
    except Exception:
        prev_io = None
    start = time.time()
    while time.time() - start < duration:
        try:
            cpu = proc.cpu_percent(interval=interval)
            mem = proc.memory_info()
            rss = mem.rss / (1024 * 1024)
            vms = mem.vms / (1024 * 1024)
            try:
                io = proc.io_counters()
                rd = max(0, io.read_bytes - (prev_io.read_bytes if prev_io else io.read_bytes))
                wr = max(0, io.write_bytes - (prev_io.write_bytes if prev_io else io.write_bytes))
                prev_io = io
            except Exception:
                rd = wr = 0
            try:
                open_files = len(proc.open_files())
            except Exception:
                open_files = 0
            tick = add_derived({
                "cpu_percent": float(cpu),
                "memory_rss_mb": float(rss),
                "memory_vms_mb": float(vms),
                "io_read_bytes_delta": float(rd),
                "io_write_bytes_delta": float(wr),
                "net_bytes_sent_delta": 0.0,
                "num_open_files": float(open_files),
            })
            rows.append(tick)
            log(f"LIVE {proc.name()} PID={proc.pid} cpu={cpu:.1f}% read={rd} write={wr} open_files={open_files}")
        except psutil.NoSuchProcess:
            log("Process ended.")
            break
    return rows


def is_protected_process(name: str) -> bool:
    n = (name or "").lower()
    return n in PROTECTED_WINDOWS or n in PROTECTED_LINUX


def apply_or_print_actions(event, allow_demo_mitigation: bool = False, target_pid: Optional[int] = None, target_name: str = ""):
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
        if action not in {"suspend_process", "kill_process"}:
            log(f"DRY-RUN: WOULD {action} (not implemented in demo script)")
            continue
        if target_pid is None or is_protected_process(target_name):
            log(f"REFUSED {action}: protected or unknown process ({target_name}, pid={target_pid})")
            continue
        try:
            p = psutil.Process(target_pid) if psutil else None
            if p is None:
                log(f"DRY-RUN: psutil unavailable for {action}")
            elif action == "suspend_process":
                p.suspend()
                log(f"MITIGATION APPLIED: suspended PID={target_pid}")
            elif action == "kill_process":
                p.terminate()
                log(f"MITIGATION APPLIED: terminated PID={target_pid}")
        except Exception as e:
            log(f"Could not apply {action}: {e}")


def run_stream(orchestrator: EDROrchestrator, static_vec: np.ndarray, stream: List[Dict[str, float]],
               process_name: str, pid: int = 4242, delay: float = 0.05, allow_demo_mitigation: bool = False):
    log(f"Starting EDR replay for {process_name} PID={pid} ticks={len(stream)}")
    event = None
    prefix = []
    for i, tick in enumerate(stream, start=1):
        prefix.append(tick)
        if i < orchestrator.cnn_model.window_size:
            log(f"tick={i:03d} warmup cpu={tick['cpu_percent']:.1f} write={tick['io_write_bytes_delta']:.0f}")
            time.sleep(delay)
            continue
        event = orchestrator.evaluate(static_vec, prefix, pid=pid, process_name=process_name, feature_cols=CNN_FEATURE_COLS)
        feats = getattr(event, "features", {}) or {}
        log(
            f"tick={i:03d} L1={feats.get('risk_state','?')} "
            f"L2max={feats.get('max_l2_probability', feats.get('l2_probability', 0)):.3f} "
            f"risk={feats.get('max_l2_risk_score', feats.get('l2_risk_score', 0)):.2f} "
            f"evidence_streak={feats.get('max_evidence_streak', feats.get('evidence_streak', 0))} "
            f"decision={event.model_source} alert={event.alert}"
        )
        if event.alert:
            break
        time.sleep(delay)
    if event is None:
        event = orchestrator.evaluate(static_vec, stream, pid=pid, process_name=process_name, feature_cols=CNN_FEATURE_COLS)
    log("=" * 72)
    log(f"FINAL DECISION: alert={event.alert} severity={event.severity} source={event.model_source}")
    log(f"confidence={event.confidence} actions={event.recommended_actions}")
    log(f"description={event.description}")
    apply_or_print_actions(event, allow_demo_mitigation=allow_demo_mitigation, target_pid=pid, target_name=process_name)
    return event


def build_orchestrator(static_kind: str = "median"):
    feature_names, static_vec = load_static_schema_and_vector(static_kind)
    cnn = load_cnn()
    static_model = load_static_model(feature_names)
    orchestrator = EDROrchestrator(
        static_model=static_model,
        cnn_model=cnn,
        policy_mode="balanced",
        require_encryption_evidence=True,
        history_max_steps=30,
        verbose=False,
    )
    # If the loaded RF has a different feature schema, rebuild the static vector to match where possible.
    if getattr(static_model, "feature_names", None) and len(static_model.feature_names) != len(static_vec):
        log("Static model schema differs from demo vector; using zero-filled schema-aligned vector.")
        static_vec = np.zeros(len(static_model.feature_names), dtype=np.float32)
    return orchestrator, static_vec


def main():
    ap = argparse.ArgumentParser(description="Safe terminal demo for Hybrid Ransomware EDR")
    ap.add_argument("--mode", choices=["scenario", "replay", "live"], default="scenario")
    ap.add_argument("--scenario", choices=["notepad", "ransomware", "memory_critical"], default="notepad")
    ap.add_argument("--csv", type=str, help="CSV path for replay mode")
    ap.add_argument("--name", type=str, help="Process name for live mode, e.g. notepad.exe")
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--delay", type=float, default=0.05, help="Replay delay between ticks")
    ap.add_argument("--allow-demo-mitigation", action="store_true", help="Allow suspend/terminate only for explicit demo targets")
    args = ap.parse_args()

    log("Hybrid Ransomware EDR terminal demo — DRY-RUN by default")
    log("Layer 2 runs as live/replayed telemetry; Layer 1 uses loaded RF or demo static prior.")

    static_kind = "ransomware" if args.scenario == "memory_critical" else "median"
    orchestrator, static_vec = build_orchestrator(static_kind=static_kind)

    if args.mode == "scenario":
        if args.scenario == "notepad":
            stream = scenario_notepad()
            process_name = "notepad_demo.exe"
        elif args.scenario == "ransomware":
            stream = scenario_ransomware()
            process_name = "ransomware_sim.py"
        else:
            stream = scenario_notepad(n=12)
            process_name = "memory_ransomware_row"
        run_stream(orchestrator, static_vec, stream, process_name=process_name, pid=4242, delay=args.delay, allow_demo_mitigation=args.allow_demo_mitigation)

    elif args.mode == "replay":
        if not args.csv:
            raise SystemExit("--csv is required for replay mode")
        path = Path(args.csv)
        stream = csv_stream(path)
        run_stream(orchestrator, static_vec, stream, process_name=path.name, pid=5001, delay=args.delay, allow_demo_mitigation=args.allow_demo_mitigation)

    elif args.mode == "live":
        if not args.name:
            raise SystemExit("--name is required for live mode")
        stream = live_stream(args.name, args.duration, args.interval)
        run_stream(orchestrator, static_vec, stream, process_name=args.name, pid=6001, delay=0.0, allow_demo_mitigation=False)


if __name__ == "__main__":
    main()
