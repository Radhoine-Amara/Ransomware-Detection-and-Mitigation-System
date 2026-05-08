#!/usr/bin/env python3
# Live-only terminal demo runner for the Hybrid Ransomware EDR project.
# This version is for the video demo and does NOT replay training/evaluation CSVs.
# It uses live monitoring or controlled child processes generated at runtime.

from __future__ import annotations

import argparse
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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

try:
    from src.models.cnn_model import RansomwareCNN
    from src.engine.edr_orchestrator import EDROrchestrator
except Exception as exc:
    raise SystemExit(
        "Could not import project modules. Place this script in the project root "
        "beside the src/ folder and run it from there.\n"
        f"Import error: {exc}"
    )

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
    # Demo Layer-1 prior for live terminal mode.
    # The full notebook demonstrates Layer 1 using memory feature rows.
    # The terminal demo focuses on live telemetry and keeps memory state at WATCH.
    def __init__(self, probability: float = 0.12):
        self.feature_names = [f"demo_static_feature_{i}" for i in range(69)]
        self.threshold = 0.10
        self.probability = float(probability)
        self.scaler = None

    def predict(self, features, pid: int = 0, process_name: str = "unknown"):
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
            description="Live demo static prior: memory forensics is shown in the notebook; live mode focuses on telemetry.",
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


def build_orchestrator() -> Tuple[EDROrchestrator, np.ndarray]:
    cnn = load_cnn()
    static_model = ConstantStaticModel(probability=0.12)
    static_vec = np.zeros(len(static_model.feature_names), dtype=np.float32)
    orchestrator = EDROrchestrator(
        static_model=static_model,
        cnn_model=cnn,
        policy_mode="balanced",
        require_encryption_evidence=True,
        history_max_steps=30,
        verbose=False,
    )
    return orchestrator, static_vec


def create_demo_files(root: Path, n_files: int = 120, size_bytes: int = 24_576) -> Path:
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


def write_ransomware_dummy_worker(workspace: Path, duration: int) -> Path:
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
            # Harmless CPU-heavy transformation; writes transformed copies only.
            out = bytes((b ^ key) for b in data)
            (output_dir / (p.stem + f".{{round_id}}.simenc")).write_bytes(out)
        except Exception:
            pass
        time.sleep(0.003)
'''
    worker.write_text(code, encoding="utf-8")
    return worker


def write_hard_benign_worker(workspace: Path, duration: int) -> Path:
    worker = workspace / "hard_benign_worker.py"
    code = f'''
import time
import shutil
from pathlib import Path
root = Path({str(workspace)!r})
input_dir = root / "input_files"
output_dir = root / "output_files"
output_dir.mkdir(exist_ok=True)
end = time.time() + {int(duration)}
round_id = 0
files = list(input_dir.glob("*.txt"))
# Benign workload: sequential backup/copy behavior, no encryption pattern.
while time.time() < end:
    round_id += 1
    for p in files:
        try:
            dst = output_dir / f"backup_{{round_id}}_{{p.name}}"
            shutil.copyfile(p, dst)
        except Exception:
            pass
        time.sleep(0.012)
'''
    worker.write_text(code, encoding="utf-8")
    return worker


def launch_worker(kind: str, duration: int) -> subprocess.Popen:
    root = PROJECT_ROOT / "demo_workspace" / kind
    workspace = create_demo_files(root)
    if kind == "ransomware_dummy":
        worker = write_ransomware_dummy_worker(workspace, duration)
    elif kind == "hard_benign_dummy":
        worker = write_hard_benign_worker(workspace, duration)
    else:
        raise ValueError(f"Unknown worker kind: {kind}")
    log(f"Prepared generated demo workspace: {workspace}")
    proc = subprocess.Popen([sys.executable, str(worker)], cwd=str(PROJECT_ROOT))
    log(f"Started controlled {kind} process PID={proc.pid}")
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


def print_final_event(event, allow_demo_mitigation: bool, controlled_pid: Optional[int], pid: Optional[int], name: str):
    log("=" * 78)
    log(f"FINAL DECISION: alert={event.alert} severity={event.severity} source={event.model_source}")
    log(f"confidence={event.confidence} actions={event.recommended_actions}")
    log(f"description={event.description}")
    apply_or_print_actions(event, allow_demo_mitigation, controlled_pid, pid, name)
    log("=" * 78)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-only terminal demo for the Hybrid Ransomware EDR")
    parser.add_argument("--mode", choices=["scenario", "live"], default="scenario")
    parser.add_argument("--scenario", choices=["hard_benign_dummy", "ransomware_dummy"], default="ransomware_dummy")
    parser.add_argument("--name", type=str, help="Process name for live mode, e.g. notepad.exe")
    parser.add_argument("--duration", type=int, default=35, help="Duration for live/dummy scenario")
    parser.add_argument("--interval", type=float, default=1.0, help="Live sampling interval in seconds")
    parser.add_argument("--allow-demo-mitigation", action="store_true", help="Allow mitigation only for controlled dummy child process")
    args = parser.parse_args()

    if psutil is None:
        raise SystemExit("psutil is required for the live demo. Install it with: pip install psutil")

    log("Hybrid Ransomware EDR LIVE terminal demo")
    log("No training/evaluation CSV replay is used in this script.")
    log("Default mode is safe DRY-RUN. No real ransomware is used.")
    log("Layer 2 is sampled live; Layer 1 is represented by a safe WATCH-level demo prior.")

    orchestrator, static_vec = build_orchestrator()

    if args.mode == "live":
        if not args.name:
            raise SystemExit("--name is required for live mode")
        proc = find_process_by_name(args.name)
        if proc is None:
            raise SystemExit(f"Process not found: {args.name}")
        run_live_process(orchestrator, static_vec, proc, args.duration, args.interval, allow_demo_mitigation=False)
        return

    child = launch_worker(args.scenario, args.duration)
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
        try:
            if child.poll() is None:
                child.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
