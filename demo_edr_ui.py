"""
Streamlit demo UI for the Hybrid Ransomware EDR project.

Run:
    streamlit run demo_edr_ui.py

This UI is designed for presentation/demo purposes. It loads the saved CNN model
and scaler when available, uses a lightweight static-memory adapter for demo
scenarios, and replays safe telemetry streams through the real EDROrchestrator.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models.cnn_model import RansomwareCNN
from src.models.cnn_preprocessor import CNN_FEATURE_COLS_DERIVED, is_hard_benign_path
from src.engine.edr_orchestrator import EDROrchestrator

CNN_FEATURE_COLS = CNN_FEATURE_COLS_DERIVED
MODELS_DIR = PROJECT_ROOT / "saved_models"
BEHAVIORAL_RAW_DIR = PROJECT_ROOT / "data" / "behavioral_raw"


class DemoStaticModel:
    """Small adapter used for demo replay when full RF objects are not loaded.

    The orchestrator only needs a Layer 1 probability and feature_names length.
    For real training/evaluation, use the notebook with RF/XGB/LGBM models.
    """
    def __init__(self, p: float = 0.11, n_features: int = 69):
        self.p = float(p)
        self.threshold = 0.15
        self.feature_names = [f"f{i}" for i in range(n_features)]

    def predict(self, feature_dict, pid=0, process_name="unknown"):
        from src.engine.system_event import SystemEvent
        return SystemEvent(alert=self.p >= self.threshold, severity="NONE", model_source="DemoStatic", confidence=self.p, pid=pid, process_name=process_name, recommended_actions=["log_event"], features={"demo_static_probability": self.p})


def static_vector(n=69):
    return np.zeros(n, dtype=np.float32)


def benign_notepad_stream(n=30, seed=7):
    rng = np.random.default_rng(seed)
    stream = []
    for i in range(n):
        stream.append({
            "cpu_percent": float(rng.uniform(0.5, 4.0)),
            "memory_rss_mb": float(rng.uniform(35, 75)),
            "memory_vms_mb": float(rng.uniform(80, 180)),
            "io_read_bytes_delta": float(rng.integers(0, 4096)),
            "io_write_bytes_delta": float(rng.integers(0, 8192)),
            "net_bytes_sent_delta": 0.0,
            "num_open_files": float(rng.integers(1, 4)),
        })
    return stream


def ransomware_behavior_stream(n=36, encrypt_start=12, seed=42):
    rng = np.random.default_rng(seed)
    stream = []
    for i in range(n):
        if i < encrypt_start:
            tick = {
                "cpu_percent": float(rng.uniform(1, 8)),
                "memory_rss_mb": float(rng.uniform(40, 90)),
                "memory_vms_mb": float(rng.uniform(100, 220)),
                "io_read_bytes_delta": float(rng.integers(0, 8192)),
                "io_write_bytes_delta": float(rng.integers(0, 8192)),
                "net_bytes_sent_delta": 0.0,
                "num_open_files": float(rng.integers(2, 8)),
            }
        else:
            read_b = float(rng.integers(800_000, 2_500_000))
            write_b = float(rng.integers(1_500_000, 4_500_000))
            cpu = float(rng.uniform(45, 95))
            tick = {
                "cpu_percent": cpu,
                "memory_rss_mb": float(rng.uniform(30, 80)),
                "memory_vms_mb": float(rng.uniform(100, 240)),
                "io_read_bytes_delta": read_b,
                "io_write_bytes_delta": write_b,
                "net_bytes_sent_delta": float(rng.integers(0, 4096)),
                "num_open_files": float(rng.integers(80, 220)),
            }
        stream.append(tick)
    return stream


def load_csv_stream(path: Path):
    df = pd.read_csv(path)
    if "step_number" in df.columns:
        df = df.sort_values("step_number")
    stream = []
    for _, row in df.iterrows():
        tick = {}
        for col in CNN_FEATURE_COLS:
            if col in row:
                tick[col] = float(row[col])
            else:
                tick[col] = 0.0
        stream.append(tick)
    return stream


@st.cache_resource
def load_cnn():
    model_dir = MODELS_DIR / "cnn"
    scaler_path = MODELS_DIR / "cnn_scaler.pkl"
    if not (model_dir / "model.keras").exists():
        raise FileNotFoundError(f"Missing CNN model: {model_dir / 'model.keras'}")
    return RansomwareCNN.load(str(model_dir), scaler_path=str(scaler_path))


def build_orchestrator(cnn, static_p):
    return EDROrchestrator(
        static_model=DemoStaticModel(p=static_p),
        cnn_model=cnn,
        l1_threshold=None,
        n_consecutive=3,
        suspend_threshold=0.85,
        kill_threshold=0.97,
        min_cnn_threshold=0.20,
        require_encryption_evidence=True,
        non_overlapping_strikes=True,
        evidence_thresholds={
            "write_min_bytes": 256 * 1024,
            "read_min_bytes": 64 * 1024,
            "ratio_min": 0.50,
            "cpu_write_min": 5_000_000.0,
            "write_intensity_min": 512.0,
            "cpu_min": 15.0,
            "open_files_min": 5.0,
            "min_hits": 3,
            "specific_min_hits": 6,
            "specific_write_min_bytes": 1 * 1024 * 1024,
            "specific_read_min_bytes": 256 * 1024,
            "specific_ratio_min": 1.0,
            "specific_ratio_max": 25.0,
            "specific_cpu_write_min": 50_000_000.0,
            "specific_write_intensity_min": 1024.0,
            "specific_cpu_min": 20.0,
            "specific_open_files_min": 50.0,
        },
        risk_decay=0.85,
        risk_soft_threshold=0.50,
        evidence_persistence_windows=5,
        strong_evidence_hits=5,
        watch_risk_soft_threshold=0.70,
        watch_evidence_persistence_windows=8,
        watch_strong_evidence_hits=6,
        alert_only_risk_threshold=0.50,
        alert_only_persistence_windows=5,
        policy_mode="balanced",
        history_max_steps=50,
        verbose=False,
    )


st.set_page_config(page_title="Hybrid Ransomware EDR Demo", layout="wide")
st.title("Hybrid Ransomware EDR Demo")
st.caption("Static memory prior + dynamic CNN telemetry + evidence-aware orchestrator")

with st.sidebar:
    st.header("Scenario")
    scenario = st.selectbox("Choose demo", ["Notepad benign", "Hard benign CSV", "Ransomware simulation", "Static high-risk memory"])
    static_p = st.slider("Layer 1 static memory probability", 0.0, 1.0, 0.11, 0.01)
    if scenario == "Static high-risk memory":
        static_p = st.slider("Layer 1 static memory probability (high-risk demo)", 0.0, 1.0, 0.95, 0.01)
    hard_files = []
    data_dir = BEHAVIORAL_RAW_DIR / "data"
    if data_dir.exists():
        hard_files = [p for p in sorted(data_dir.glob("*.csv")) if is_hard_benign_path(str(p))]
    hb_choice = None
    if scenario == "Hard benign CSV":
        if hard_files:
            hb_choice = st.selectbox("Hard benign file", hard_files, format_func=lambda p: p.name)
        else:
            st.warning("No hard benign files found at data/behavioral_raw/data/benign_021.csv..031.csv")
    run = st.button("Run scenario", type="primary")

try:
    cnn = load_cnn()
    st.success("Loaded saved CNN model and scaler.")
except Exception as e:
    st.error(f"Could not load saved CNN: {e}")
    st.stop()

if run:
    orch = build_orchestrator(cnn, static_p)
    if scenario == "Notepad benign":
        stream = benign_notepad_stream()
        process_name = "notepad.exe"
    elif scenario == "Ransomware simulation":
        stream = ransomware_behavior_stream()
        process_name = "safe_ransomware_simulator.py"
    elif scenario == "Hard benign CSV" and hb_choice is not None:
        stream = load_csv_stream(hb_choice)
        process_name = hb_choice.name
    else:
        stream = benign_notepad_stream()
        process_name = "static_memory_demo.exe"

    event = orch.evaluate(static_vector(), stream, pid=4321, process_name=process_name, feature_cols=CNN_FEATURE_COLS)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alert", str(event.alert))
    c2.metric("Severity", event.severity)
    c3.metric("Confidence", f"{event.confidence:.3f}")
    c4.metric("Source", event.model_source)

    st.subheader("Recommended actions")
    st.write(event.recommended_actions)
    st.subheader("Decision explanation")
    st.write(event.description)

    features = event.features or {}
    st.subheader("Risk details")
    st.json({k: v for k, v in features.items() if k != "risk_timeline_tail"})

    timeline = features.get("risk_timeline_tail")
    if timeline:
        df = pd.DataFrame(timeline)
        st.subheader("Timeline")
        st.dataframe(df)
        if {"step", "l2_probability", "risk_score"}.issubset(df.columns):
            st.line_chart(df.set_index("step")[["l2_probability", "risk_score"]])
else:
    st.info("Select a scenario and click Run scenario.")
