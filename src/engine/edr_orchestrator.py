# src/engine/edr_orchestrator.py
# =============================================================================
# Hybrid EDR Orchestrator v8.4
# Static memory prior + dynamic CNN + evidence-aware state machine
# =============================================================================

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.engine.system_event import (
    SystemEvent,
    SEVERITY_NONE, SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL,
    ACTION_NO_ACTION, ACTION_LOG_EVENT, ACTION_SEND_ALERT,
    ACTION_KILL_PROCESS, ACTION_BLOCK_NETWORK, ACTION_SNAPSHOT_DIRECTORY,
    ACTION_MONITOR_ONLY, ACTION_SUSPEND_PROCESS,
)

# Backwards-compatible action names expected by earlier mitigation code.
ACTION_QUARANTINE = ACTION_SNAPSHOT_DIRECTORY
ACTION_NETWORK_BLOCK = ACTION_BLOCK_NETWORK
ACTION_SUSPEND_PROC = ACTION_SUSPEND_PROCESS


class RiskState:
    SAFE = "SAFE"
    WATCH = "WATCH"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"


class EDROrchestrator:
    """
    Hybrid EDR Orchestrator v8.4.

    Design goals
    ------------
    1. Layer 1 is a *forensic prior*, not a binary silo.
       Medium memory suspicion lowers the amount of Layer 2 evidence required.

    2. Layer 2 CNN output is a *behavioral suspicion score*, not an immediate
       kill authority. Hard mitigation requires persistence and interpretable
       encryption-like telemetry evidence.

    3. The strike rule can be non-overlapping, avoiding the false impression
       that windows (1-8), (2-9), and (3-10) are independent confirmations.

    4. Layer 1 risk can decay over time so old memory evidence does not remain
       permanently decisive.
    """

    DEFAULT_EVIDENCE_THRESHOLDS = {
        # These are deliberately conservative defaults and should be tuned on
        # your hard-benign negative controls and ransomware validation sessions.
        "write_min_bytes": 256 * 1024,      # average write bytes in a window
        "read_min_bytes": 64 * 1024,        # average read bytes in a window
        "ratio_min": 0.50,                 # max write/read ratio
        "cpu_write_min": 5_000_000.0,       # max cpu_percent * write_bytes
        "write_intensity_min": 512.0,       # max write / normalized memory
        "cpu_min": 15.0,                   # average CPU percent
        "open_files_min": 5.0,             # max open files
        "min_hits": 3,                     # number of conditions required

        # V8.2: Generic high-I/O is not enough to suspend a SAFE/WATCH process.
        # These stricter criteria are used to decide whether dynamic evidence is
        # ransomware-specific enough for SOFT_BLOCK when Layer 1 is only weak.
        "specific_min_hits": 6,
        "specific_write_min_bytes": 1 * 1024 * 1024,
        "specific_read_min_bytes": 256 * 1024,
        "specific_ratio_min": 1.0,
        "specific_ratio_max": 25.0,
        "specific_cpu_write_min": 50_000_000.0,
        "specific_write_intensity_min": 1024.0,
        "specific_cpu_min": 20.0,
        "specific_open_files_min": 50.0,

        # V8.4: scoring gate for weak-memory contexts. The gate is not
        # all-or-nothing: ransomware-specific behavior is identified by a
        # minimum number of concurrent signals plus sustained evidence.
        # This allows behavioral ransomware to SOFT_BLOCK while hard-benign
        # high-I/O workloads remain ALERT_ONLY.
        "specific_score_min": 5,
        "specific_streak_min": 5,
        "specific_risk_min": 0.50,
    }

    def __init__(
        self,
        static_model,
        cnn_model,
        l1_threshold: Optional[float] = None,
        n_consecutive: int = 3,
        suspend_threshold: float = 0.85,
        kill_threshold: float = 0.97,
        l1_watch_threshold: float = 0.10,
        l1_suspicious_threshold: float = 0.35,
        l1_high_threshold: float = 0.75,
        l1_critical_threshold: float = 0.90,
        min_cnn_threshold: float = 0.20,
        adaptive_alpha: float = 0.35,
        require_encryption_evidence: bool = True,
        non_overlapping_strikes: bool = True,
        l1_decay_tau_seconds: float = 60.0,
        l1_decay_floor: float = 0.10,
        evidence_thresholds: Optional[Dict[str, float]] = None,
        risk_decay: float = 0.85,
        risk_soft_threshold: float = 0.50,
        evidence_persistence_windows: int = 5,
        strong_evidence_hits: int = 5,
        # V8.2: stronger guardrails for SAFE/WATCH contexts.
        # Generic high-I/O becomes alert-only first. Suspension requires
        # ransomware-specific evidence.
        watch_risk_soft_threshold: float = 0.70,
        watch_evidence_persistence_windows: int = 8,
        watch_strong_evidence_hits: int = 6,
        alert_only_risk_threshold: float = 0.50,
        alert_only_persistence_windows: int = 5,
        policy_mode: str = "balanced",
        history_max_steps: int = 20,
        verbose: bool = False,
    ):
        self.static_model = static_model
        self.cnn_model = cnn_model

        # Use the trained Layer 1 threshold if available; otherwise fallback.
        self.l1_threshold = float(
            l1_threshold if l1_threshold is not None
            else getattr(static_model, "threshold", 0.50)
        )

        self.n_consecutive = int(n_consecutive)
        self.suspend_threshold = float(suspend_threshold)
        self.kill_threshold = float(kill_threshold)

        self.l1_watch_threshold = float(l1_watch_threshold)
        self.l1_suspicious_threshold = float(l1_suspicious_threshold)
        self.l1_high_threshold = float(l1_high_threshold)
        self.l1_critical_threshold = float(l1_critical_threshold)

        self.min_cnn_threshold = float(min_cnn_threshold)
        self.adaptive_alpha = float(adaptive_alpha)
        self.require_encryption_evidence = bool(require_encryption_evidence)
        self.non_overlapping_strikes = bool(non_overlapping_strikes)
        self.l1_decay_tau_seconds = float(l1_decay_tau_seconds)
        self.l1_decay_floor = float(l1_decay_floor)
        self.risk_decay = float(risk_decay)
        self.risk_soft_threshold = float(risk_soft_threshold)
        self.evidence_persistence_windows = int(evidence_persistence_windows)
        self.strong_evidence_hits = int(strong_evidence_hits)
        self.watch_risk_soft_threshold = float(watch_risk_soft_threshold)
        self.watch_evidence_persistence_windows = int(watch_evidence_persistence_windows)
        self.watch_strong_evidence_hits = int(watch_strong_evidence_hits)
        self.alert_only_risk_threshold = float(alert_only_risk_threshold)
        self.alert_only_persistence_windows = int(alert_only_persistence_windows)
        self.policy_mode = str(policy_mode or "balanced")
        self.history_max_steps = int(history_max_steps)
        self.verbose = verbose

        self.evidence_thresholds = dict(self.DEFAULT_EVIDENCE_THRESHOLDS)
        if evidence_thresholds:
            self.evidence_thresholds.update(evidence_thresholds)

        if getattr(cnn_model, "scaler", None) is None:
            raise ValueError(
                "cnn_model.scaler is None. Attach after training:\n"
                "  cnn.scaler = cnn_data['scaler']"
            )
        if getattr(cnn_model, "feature_cols", None) is None:
            raise ValueError(
                "cnn_model.feature_cols is None. Attach after training:\n"
                "  cnn.feature_cols = cnn_data['feature_cols']"
            )

    # ---------------------------------------------------------------------
    # Layer 1 helpers
    # ---------------------------------------------------------------------
    def _feature_dict_from_array(self, X: np.ndarray) -> Optional[Dict[str, float]]:
        names = getattr(self.static_model, "feature_names", None)
        if not names or len(names) != X.shape[1]:
            return None
        return {name: float(val) for name, val in zip(names, X.flatten())}

    def _l1_prob(self, static_features: np.ndarray, pid: int, process_name: str) -> float:
        """
        Get Layer 1 ransomware probability while avoiding raw/scaled mismatch.

        Preferred path: model.predict(feature_dict), because the Layer 1 wrapper
        owns feature order, scaler, and calibration. Fallbacks are kept only for
        compatibility with bare sklearn pipelines.
        """
        X = np.array(static_features, dtype=np.float32).reshape(1, -1)

        # Preferred wrapper path: validates feature order and applies scaler.
        feat_dict = self._feature_dict_from_array(X)
        if feat_dict is not None and hasattr(self.static_model, "predict"):
            try:
                event = self.static_model.predict(
                    feat_dict,
                    pid=pid or 0,
                    process_name=process_name or "unknown",
                )
                return float(getattr(event, "confidence", 0.0))
            except Exception as e:
                if self.verbose:
                    print(f"  [L1 Static] wrapper predict failed: {e}")

        # Wrapper batch path: predict_proba_batch expects pre-scaled input for
        # this project, so apply static_model.scaler if it exists.
        if hasattr(self.static_model, "predict_proba_batch"):
            X_in = X
            scaler = getattr(self.static_model, "scaler", None)
            if scaler is not None:
                try:
                    X_in = scaler.transform(X)
                except Exception as e:
                    raise ValueError(
                        "Layer 1 scaler failed. Do not treat this process as benign; "
                        "check the static feature schema/order."
                    ) from e
            proba = self.static_model.predict_proba_batch(X_in)
            return float(np.asarray(proba).reshape(-1)[0])

        # Bare sklearn Pipeline or model. Use only if exposed directly.
        if hasattr(self.static_model, "predict_proba"):
            proba = self.static_model.predict_proba(X)
            proba = np.asarray(proba)
            if proba.ndim == 2:
                return float(proba[0, 1])
            return float(proba.reshape(-1)[0])

        if hasattr(self.static_model, "model") and hasattr(self.static_model.model, "predict_proba"):
            proba = self.static_model.model.predict_proba(X)
            return float(proba[0, 1])

        raise AttributeError(
            "static_model must expose predict(feature_dict), predict_proba_batch(), "
            "or predict_proba()."
        )

    def _decay_l1_prob(self, p1: float, elapsed_seconds: float) -> float:
        """
        Decay suspicious Layer 1 evidence toward a low WATCH floor.

        We do not make a clean score become suspicious over time; only excess
        memory risk above l1_decay_floor fades. This is safer for an EDR policy
        than decaying all probabilities toward 0.5.
        """
        if p1 <= self.l1_decay_floor or self.l1_decay_tau_seconds <= 0:
            return float(p1)
        decay = math.exp(-max(elapsed_seconds, 0.0) / self.l1_decay_tau_seconds)
        return float(self.l1_decay_floor + (p1 - self.l1_decay_floor) * decay)

    def _risk_state(self, p1: float) -> str:
        if p1 >= self.l1_critical_threshold:
            return RiskState.CRITICAL
        if p1 >= self.l1_high_threshold:
            return RiskState.HIGH_RISK
        if p1 >= self.l1_suspicious_threshold:
            return RiskState.SUSPICIOUS
        if p1 >= self.l1_watch_threshold:
            return RiskState.WATCH
        return RiskState.SAFE

    def _cnn_threshold_mode_for_state(self, risk_state: str) -> str:
        """Choose a CNN operating point based on Layer 1 state.

        V8 formalises threshold modes:
          SAFE/WATCH      -> strict or balanced threshold, evidence required
          SUSPICIOUS      -> balanced threshold
          HIGH/CRITICAL   -> sensitive threshold
        The policy remains balanced at the process level because hard kill is
        still blocked unless Layer 1 is at least SUSPICIOUS.
        """
        if risk_state in (RiskState.HIGH_RISK, RiskState.CRITICAL):
            return "sensitive"
        if risk_state == RiskState.SUSPICIOUS:
            return "balanced"
        # SAFE/WATCH: reduce window-level FPs. Evidence fallback can still soft-block.
        return "strict" if self.policy_mode in ("balanced", "safe") else "balanced"

    def _adaptive_cnn_threshold(self, p1_effective: float, risk_state: str = None) -> tuple[float, str]:
        modes = getattr(self.cnn_model, "threshold_modes", {}) or {}
        mode = self._cnn_threshold_mode_for_state(risk_state or self._risk_state(p1_effective))
        base = float(modes.get(mode, {}).get("threshold", getattr(self.cnn_model, "threshold", 0.5)))
        # Layer 1 suspicion lowers Layer 2 threshold, but never below min.
        risk_excess = max(0.0, p1_effective - self.l1_suspicious_threshold)
        return float(max(self.min_cnn_threshold, base - self.adaptive_alpha * risk_excess)), mode

    def _required_strikes(self, risk_state: str) -> int:
        if risk_state == RiskState.CRITICAL:
            return 1
        if risk_state == RiskState.HIGH_RISK:
            return max(1, self.n_consecutive - 2)
        if risk_state == RiskState.SUSPICIOUS:
            return max(2, self.n_consecutive - 1)
        return self.n_consecutive

    def _risk_accumulator_signal(self, l2_prob: float, evidence_present: bool,
                                 evidence: Dict[str, float]) -> float:
        """Continuous Layer 2 risk signal for bursty CNN outputs.

        V6 used strict non-overlapping strikes. That protected Notepad but missed
        attacks when the CNN was bursty. V7 keeps the strike rule but also
        accumulates risk from high CNN scores plus interpretable encryption
        evidence. No evidence -> no risk accumulation.
        """
        if not evidence_present:
            return 0.0
        hits = float(evidence.get("evidence_hits", 0.0))
        evidence_strength = min(1.0, hits / 7.0)
        return float(max(l2_prob, evidence_strength))

    def _is_weak_memory_state(self, risk_state: str) -> bool:
        """True when Layer 1 is not strong enough to justify process suspension
        from generic high-I/O behavior alone."""
        return risk_state in (RiskState.SAFE, RiskState.WATCH)

    def _ransomware_specific_evidence(self, evidence: Dict[str, float]) -> bool:
        """Score-based ransomware-specific evidence gate for SAFE/WATCH.

        V8.2 used an all-conditions gate. That protected hard-benign workloads,
        but it also overcorrected: ransomware simulation became ALERT_ONLY. V8.4
        uses a score instead of an all-or-nothing rule. A process must show a
        sustained combination of ransomware-like signals before a SAFE/WATCH
        process is suspended.

        Key idea:
          generic high I/O       -> ALERT_ONLY
          sustained encryption   -> SOFT_BLOCK

        The function writes diagnostic fields back into ``evidence`` so the
        notebook can explain why a process was or was not soft-blocked.
        """
        t = self.evidence_thresholds
        ratio = float(evidence.get("max_write_read_ratio", 0.0))

        signals = {
            "specific_write_volume": evidence.get("avg_write_bytes", 0.0) >= float(t.get("specific_write_min_bytes", 1 * 1024 * 1024)),
            "specific_read_volume": evidence.get("avg_read_bytes", 0.0) >= float(t.get("specific_read_min_bytes", 256 * 1024)),
            "specific_ratio_band": ratio >= float(t.get("specific_ratio_min", 1.0)) and ratio <= float(t.get("specific_ratio_max", 25.0)),
            "specific_cpu_write_coupling": evidence.get("max_cpu_x_write", 0.0) >= float(t.get("specific_cpu_write_min", 50_000_000.0)),
            "specific_write_intensity": evidence.get("max_io_write_intensity", 0.0) >= float(t.get("specific_write_intensity_min", 1024.0)),
            "specific_cpu_activity": evidence.get("avg_cpu_percent", 0.0) >= float(t.get("specific_cpu_min", 20.0)),
            "specific_open_file_pressure": evidence.get("max_open_files", 0.0) >= float(t.get("specific_open_files_min", 50.0)),
        }
        score = int(sum(bool(v) for v in signals.values()))
        evidence.update({k: float(v) for k, v in signals.items()})
        evidence["ransomware_specific_score"] = float(score)

        # Core ransomware behavior: heavy writes + CPU/write coupling + either
        # many open files or abnormal write intensity. This avoids suspending a
        # plain high-throughput file copy or backup that lacks encryption-like
        # coupling.
        core_present = bool(
            signals["specific_write_volume"]
            and signals["specific_cpu_write_coupling"]
            and (signals["specific_open_file_pressure"] or signals["specific_write_intensity"])
        )
        sustained = bool(
            evidence.get("evidence_streak", 0.0) >= float(t.get("specific_streak_min", 5))
            and evidence.get("l2_risk_score", 0.0) >= float(t.get("specific_risk_min", 0.50))
        )
        passed = bool(
            score >= int(t.get("specific_score_min", 5))
            and core_present
            and sustained
        )
        evidence["ransomware_specific_core_present"] = float(core_present)
        evidence["ransomware_specific_sustained"] = float(sustained)
        evidence["ransomware_specific_passed"] = float(passed)
        return passed

    def _state_soft_policy(self, risk_state: str) -> Tuple[float, int, int]:
        """Return state-aware risk/streak/hit requirements for SOFT_BLOCK."""
        if self._is_weak_memory_state(risk_state):
            return (self.watch_risk_soft_threshold,
                    self.watch_evidence_persistence_windows,
                    self.watch_strong_evidence_hits)
        return (self.risk_soft_threshold,
                self.evidence_persistence_windows,
                self.strong_evidence_hits)

    # ---------------------------------------------------------------------
    # Layer 2 helpers
    # ---------------------------------------------------------------------
    def _ensure_tick_features(self, tick: dict, feat_cols: list) -> dict:
        """Compute derived features if missing, matching cnn_preprocessor formulas."""
        tick = dict(tick)
        if "write_read_ratio" in feat_cols and "write_read_ratio" not in tick:
            read_b = float(tick.get("io_read_bytes_delta", 0.0))
            write_b = float(tick.get("io_write_bytes_delta", 0.0))
            tick["write_read_ratio"] = write_b / (read_b + 1.0)

        if "cpu_x_write" in feat_cols and "cpu_x_write" not in tick:
            cpu_pct = float(tick.get("cpu_percent", 0.0))
            write_b = float(tick.get("io_write_bytes_delta", 0.0))
            tick["cpu_x_write"] = cpu_pct * write_b

        if "io_write_intensity" in feat_cols and "io_write_intensity" not in tick:
            write_b = float(tick.get("io_write_bytes_delta", 0.0))
            mem_rss = float(tick.get("memory_rss_mb", 0.0))
            tick["io_write_intensity"] = write_b / (mem_rss * 1024.0 + 1.0)

        return tick

    def _encryption_evidence(self, window_raw: np.ndarray, feat_cols: List[str]) -> Tuple[bool, Dict[str, float]]:
        """
        Interpretable evidence gate for active encryption-like behavior.
        Returns (is_present, metrics). The gate is intentionally separate from
        the CNN so hard blocking is based on behavior + model confidence.
        """
        idx = {c: i for i, c in enumerate(feat_cols)}

        def col(name: str, default: float = 0.0) -> np.ndarray:
            if name not in idx:
                return np.full((window_raw.shape[0],), default, dtype=np.float32)
            return window_raw[:, idx[name]].astype(np.float32)

        read_b = col("io_read_bytes_delta")
        write_b = col("io_write_bytes_delta")
        ratio = col("write_read_ratio")
        cpu_x_write = col("cpu_x_write")
        intensity = col("io_write_intensity")
        cpu = col("cpu_percent")
        open_files = col("num_open_files")

        metrics = {
            "avg_write_bytes": float(np.mean(write_b)),
            "avg_read_bytes": float(np.mean(read_b)),
            "max_write_read_ratio": float(np.max(ratio)),
            "max_cpu_x_write": float(np.max(cpu_x_write)),
            "max_io_write_intensity": float(np.max(intensity)),
            "avg_cpu_percent": float(np.mean(cpu)),
            "max_open_files": float(np.max(open_files)),
        }
        t = self.evidence_thresholds
        checks = {
            "write_volume": metrics["avg_write_bytes"] >= t["write_min_bytes"],
            "read_volume": metrics["avg_read_bytes"] >= t["read_min_bytes"],
            "read_write_ratio": metrics["max_write_read_ratio"] >= t["ratio_min"],
            "cpu_write_coupling": metrics["max_cpu_x_write"] >= t["cpu_write_min"],
            "write_intensity": metrics["max_io_write_intensity"] >= t["write_intensity_min"],
            "cpu_activity": metrics["avg_cpu_percent"] >= t["cpu_min"],
            "open_file_pressure": metrics["max_open_files"] >= t["open_files_min"],
        }
        hits = int(sum(checks.values()))
        metrics.update({f"evidence_{k}": float(v) for k, v in checks.items()})
        metrics["evidence_hits"] = float(hits)
        return hits >= int(t.get("min_hits", 3)), metrics

    def _predict_l2(self, window_raw: np.ndarray, scaler) -> float:
        if hasattr(self.cnn_model, "predict_proba_raw_window"):
            return float(self.cnn_model.predict_proba_raw_window(window_raw, scaler=scaler))
        window_scaled = scaler.transform(window_raw)
        return float(self.cnn_model.model.predict(window_scaled[np.newaxis, ...], verbose=0)[0, 0])

    # ---------------------------------------------------------------------
    # Response builders
    # ---------------------------------------------------------------------
    def _static_response(self, p1: float, pid: int, process_name: str, state: str,
                         elapsed_ms: float) -> SystemEvent:
        if state == RiskState.CRITICAL:
            return SystemEvent(
                alert=True,
                severity=SEVERITY_CRITICAL,
                model_source="EDR_Layer1_Static_CRITICAL",
                confidence=round(p1, 4),
                pid=pid,
                process_name=process_name,
                recommended_actions=[ACTION_KILL_PROCESS, ACTION_QUARANTINE,
                                     ACTION_SEND_ALERT, ACTION_NETWORK_BLOCK],
                features={"l1_probability": p1, "risk_state": state},
                raw_prediction=1,
                description=(
                    f"HARD BLOCK BY LAYER 1 — static memory probability "
                    f"{p1:.1%} ≥ critical threshold {self.l1_critical_threshold:.0%}. "
                    f"Latency={elapsed_ms:.1f}ms."
                ),
            )

        # High Layer 1 memory risk is a soft block, not automatic kill.
        return SystemEvent(
            alert=True,
            severity=SEVERITY_HIGH,
            model_source="EDR_Layer1_Static_HIGH_RISK",
            confidence=round(p1, 4),
            pid=pid,
            process_name=process_name,
            recommended_actions=[ACTION_SUSPEND_PROC, ACTION_SEND_ALERT, ACTION_LOG_EVENT],
            features={"l1_probability": p1, "risk_state": state},
            raw_prediction=1,
            description=(
                f"SOFT BLOCK BY LAYER 1 — static memory probability {p1:.1%} "
                f"indicates high forensic risk, but is below the critical hard-block "
                f"threshold {self.l1_critical_threshold:.0%}. Process should be "
                f"suspended and monitored with Layer 2 confirmation."
            ),
        )

    def _alert_only_response(
        self,
        confidence: float,
        step: int,
        pid: int,
        process_name: str,
        p1_effective: float,
        risk_state: str,
        evidence: Dict[str, float],
        theta: float,
        trigger_reason: str = "generic_high_io_alert_only",
    ) -> SystemEvent:
        """Non-disruptive response for generic high-I/O evidence in SAFE/WATCH.

        This is the V8.2 hard-benign protection layer. It keeps analyst visibility
        without suspending backup/compression/copy-like workloads unless the
        evidence becomes ransomware-specific or Layer 1 is stronger.
        """
        return SystemEvent(
            alert=True,
            severity=SEVERITY_MEDIUM,
            model_source="EDR_Layer2_ALERT_ONLY_GENERIC_IO",
            confidence=round(confidence, 4),
            pid=pid,
            process_name=process_name,
            recommended_actions=[ACTION_SEND_ALERT, ACTION_LOG_EVENT],
            features={"l1_effective_probability": p1_effective,
                      "risk_state": risk_state, "cnn_threshold": theta,
                      "l2_probability": confidence,
                      "trigger_reason": trigger_reason,
                      "policy_mode": self.policy_mode,
                      "ransomware_specific_evidence": 0.0,
                      **evidence},
            raw_prediction=1,
            description=(
                f"ALERT ONLY at step {step} — generic high-I/O/encryption-like "
                f"evidence observed, but Layer 1 is {risk_state} and the evidence "
                f"did not pass the stricter ransomware-specific gate. "
                f"No suspension or kill is recommended. L1_effective={p1_effective:.1%}; "
                f"risk={evidence.get('l2_risk_score', 0):.2f}; "
                f"evidence_hits={int(evidence.get('evidence_hits', 0))}."
            ),
        )

    def _dynamic_response(
        self,
        confidence: float,
        step: int,
        pid: int,
        process_name: str,
        p1_effective: float,
        risk_state: str,
        evidence: Dict[str, float],
        strikes: int,
        required_strikes: int,
        theta: float,
        trigger_reason: str = "strike_rule",
    ) -> SystemEvent:
        evidence_note = (
            f"evidence_hits={int(evidence.get('evidence_hits', 0))}; "
            f"avg_write={evidence.get('avg_write_bytes', 0):.0f}; "
            f"avg_read={evidence.get('avg_read_bytes', 0):.0f}; "
            f"max_ratio={evidence.get('max_write_read_ratio', 0):.2f}; "
            f"max_cpu_x_write={evidence.get('max_cpu_x_write', 0):.0f}"
        )

        # Hard block is intentionally NOT allowed while Layer 1 is SAFE/WATCH.
        # Strong Layer 2 behavior with clean/weak memory evidence should first
        # become a soft block (suspend + preserve evidence), because Layer 2 is
        # the more false-positive-prone signal. Hard kill requires both:
        #   1) confirmed dynamic encryption evidence and persistence, and
        #   2) at least SUSPICIOUS Layer 1 memory prior, and
        #   3) CNN confidence above the kill threshold.
        hard_allowed = (
            risk_state in (RiskState.SUSPICIOUS, RiskState.HIGH_RISK, RiskState.CRITICAL)
            and confidence >= self.kill_threshold
        )

        if hard_allowed:
            return SystemEvent(
                alert=True,
                severity=SEVERITY_CRITICAL,
                model_source="EDR_Layer2_CNN_HARD_EVIDENCE",
                confidence=round(confidence, 4),
                pid=pid,
                process_name=process_name,
                recommended_actions=[ACTION_KILL_PROCESS, ACTION_QUARANTINE,
                                     ACTION_SEND_ALERT, ACTION_NETWORK_BLOCK],
                features={"l1_effective_probability": p1_effective,
                          "risk_state": risk_state, "cnn_threshold": theta,
                          "l2_probability": confidence,
                          "trigger_reason": trigger_reason,
                          "policy_mode": self.policy_mode,
                          **evidence},
                raw_prediction=1,
                description=(
                    f"HARD BLOCK at step {step} — CNN={confidence:.1%} ≥ "
                    f"kill_threshold={self.kill_threshold:.0%}; "
                    f"strikes={strikes}/{required_strikes}; trigger={trigger_reason}; "
                    f"L1_effective={p1_effective:.1%} ({risk_state}); "
                    f"{evidence_note}."
                ),
            )

        return SystemEvent(
            alert=True,
            severity=SEVERITY_HIGH,
            model_source="EDR_Layer2_CNN_SOFT_EVIDENCE",
            confidence=round(confidence, 4),
            pid=pid,
            process_name=process_name,
            recommended_actions=[ACTION_SUSPEND_PROC, ACTION_SEND_ALERT, ACTION_LOG_EVENT],
            features={"l1_effective_probability": p1_effective,
                      "risk_state": risk_state, "cnn_threshold": theta,
                      "l2_probability": confidence,
                      "trigger_reason": trigger_reason,
                      "policy_mode": self.policy_mode,
                      **evidence},
            raw_prediction=1,
            description=(
                f"SOFT BLOCK at step {step} — CNN={confidence:.1%} ≥ adaptive "
                f"threshold={theta:.1%}; strikes={strikes}/{required_strikes}; trigger={trigger_reason}; "
                f"L1_effective={p1_effective:.1%} ({risk_state}); "
                f"{evidence_note}. Hard kill withheld because Layer 1 is not "
                f"sufficiently suspicious or CNN confidence did not reach "
                f"the hard threshold {self.kill_threshold:.0%}."
            ),
        )

    # ---------------------------------------------------------------------
    # Main evaluation
    # ---------------------------------------------------------------------
    def evaluate(
        self,
        static_features: np.ndarray,
        telemetry_stream: list,
        window_size: int = None,
        pid: int = None,
        process_name: str = "unknown",
        feature_cols: list = None,
    ) -> SystemEvent:
        """Evaluate one process through static memory + dynamic telemetry."""
        t0 = time.time()
        ws = window_size or self.cnn_model.window_size
        feat_cols = feature_cols or self.cnn_model.feature_cols
        scaler = self.cnn_model.scaler

        # Layer 1: static memory prior.
        try:
            l1_prob_initial = self._l1_prob(static_features, pid, process_name)
            l1_error = None
        except Exception as e:
            # Schema/scaler failures should never be treated as clean.
            l1_prob_initial = self.l1_watch_threshold
            l1_error = str(e)

        risk_state_initial = self._risk_state(l1_prob_initial)

        if self.verbose:
            print(f"  [L1 Static] P={l1_prob_initial:.4f} "
                  f"state={risk_state_initial} l1_threshold={self.l1_threshold:.4f}")
            if l1_error:
                print(f"  [L1 Static] preprocessing/schema error: {l1_error}")

        # Immediate static actions only at high/critical forensic risk.
        elapsed_ms = (time.time() - t0) * 1000
        if risk_state_initial == RiskState.CRITICAL:
            return self._static_response(l1_prob_initial, pid, process_name,
                                         risk_state_initial, elapsed_ms)
        if risk_state_initial == RiskState.HIGH_RISK:
            return self._static_response(l1_prob_initial, pid, process_name,
                                         risk_state_initial, elapsed_ms)

        tick_buf: List[np.ndarray] = []
        l2_scores: List[float] = []
        counted_strikes = 0
        max_counted_strikes = 0
        last_counted_step = -10**9
        risk_score = 0.0
        max_risk_score = 0.0
        evidence_streak = 0
        max_evidence_streak = 0
        last_evidence: Dict[str, float] = {}
        history_records: List[Dict[str, float]] = []
        last_theta = float(getattr(self.cnn_model, "threshold", 0.5))
        last_threshold_mode = "balanced"

        for step_idx, tick in enumerate(telemetry_stream):
            tick = self._ensure_tick_features(tick, feat_cols)
            try:
                raw_vec = np.array([float(tick[col]) for col in feat_cols], dtype=np.float32)
            except KeyError as e:
                missing = [c for c in feat_cols if c not in tick]
                raise ValueError(
                    f"Telemetry tick missing columns: {missing or [str(e)]}. "
                    f"Expected: {feat_cols}"
                ) from e

            tick_buf.append(raw_vec)
            if len(tick_buf) < ws:
                continue

            elapsed = time.time() - t0
            l1_effective = self._decay_l1_prob(l1_prob_initial, elapsed)
            risk_state = self._risk_state(l1_effective)
            theta, threshold_mode = self._adaptive_cnn_threshold(l1_effective, risk_state)
            required = self._required_strikes(risk_state)
            last_theta = theta
            last_threshold_mode = threshold_mode

            window_raw = np.array(tick_buf[-ws:], dtype=np.float32)
            l2_prob = self._predict_l2(window_raw, scaler)
            l2_scores.append(l2_prob)

            evidence_present, evidence = self._encryption_evidence(window_raw, feat_cols)

            signal = self._risk_accumulator_signal(l2_prob, evidence_present, evidence)
            risk_score = self.risk_decay * risk_score + (1.0 - self.risk_decay) * signal
            max_risk_score = max(max_risk_score, risk_score)

            if evidence_present:
                evidence_streak += 1
            else:
                evidence_streak = 0
            max_evidence_streak = max(max_evidence_streak, evidence_streak)

            evidence["l2_risk_score"] = float(risk_score)
            evidence["evidence_streak"] = float(evidence_streak)
            ransomware_specific = self._ransomware_specific_evidence(evidence)
            evidence["ransomware_specific_evidence"] = float(ransomware_specific)
            state_risk_soft, state_persistence, state_strong_hits = self._state_soft_policy(risk_state)
            evidence["risk_soft_threshold"] = float(state_risk_soft)
            evidence["state_evidence_persistence_windows"] = float(state_persistence)
            evidence["state_strong_evidence_hits"] = float(state_strong_hits)
            evidence["threshold_mode"] = threshold_mode
            history_records.append({
                "step": float(step_idx + 1),
                "l2_probability": float(l2_prob),
                "cnn_threshold": float(theta),
                "threshold_mode": threshold_mode,
                "evidence_present": float(bool(evidence_present)),
                "evidence_hits": float(evidence.get("evidence_hits", 0.0)),
                "ransomware_specific_evidence": float(ransomware_specific),
                "risk_score": float(risk_score),
                "evidence_streak": float(evidence_streak),
                "counted_strikes": float(counted_strikes),
            })
            if len(history_records) > self.history_max_steps:
                history_records = history_records[-self.history_max_steps:]
            last_evidence = evidence

            if self.verbose:
                print(f"  [L2 CNN] step={step_idx+1:3d} P={l2_prob:.4f} "
                      f"thr={theta:.4f} state={risk_state} "
                      f"evidence={evidence_present} strikes={counted_strikes}/{required} "
                      f"risk={risk_score:.3f} ev_streak={evidence_streak}")

            # V7 fallback: strong/persistent encryption evidence should trigger
            # SOFT_BLOCK even when CNN scores are bursty and do not satisfy the
            # non-overlapping strike rule. This does not hard-kill while Layer 1
            # is SAFE/WATCH; _dynamic_response enforces that policy.
            state_risk_soft, state_persistence, state_strong_hits = self._state_soft_policy(risk_state)
            weak_memory = self._is_weak_memory_state(risk_state)
            persistent_evidence = (
                evidence_present
                and evidence_streak >= state_persistence
                and risk_score >= state_risk_soft
            )
            strong_persistent_evidence = (
                evidence_present
                and evidence.get("evidence_hits", 0.0) >= state_strong_hits
                and evidence_streak >= max(3, state_persistence - 2)
            )

            # V8.2 hard-benign protection:
            # In SAFE/WATCH, generic high-I/O evidence becomes ALERT_ONLY first.
            # SOFT_BLOCK requires ransomware-specific evidence.
            generic_alert_only = (
                weak_memory
                and evidence_present
                and not ransomware_specific
                and evidence_streak >= self.alert_only_persistence_windows
                and risk_score >= self.alert_only_risk_threshold
            )
            if generic_alert_only:
                return self._alert_only_response(
                    confidence=max(l2_prob, risk_score),
                    step=step_idx + 1,
                    pid=pid,
                    process_name=process_name,
                    p1_effective=l1_effective,
                    risk_state=risk_state,
                    evidence={**evidence, "risk_timeline_tail": history_records[-self.history_max_steps:]},
                    theta=theta,
                    trigger_reason="generic_high_io_alert_only",
                )

            if persistent_evidence or strong_persistent_evidence:
                if weak_memory and not ransomware_specific:
                    # Keep monitoring until generic high-I/O is either resolved or
                    # becomes ransomware-specific enough for suspension.
                    pass
                else:
                    return self._dynamic_response(
                        confidence=max(l2_prob, risk_score),
                        step=step_idx + 1,
                        pid=pid,
                        process_name=process_name,
                        p1_effective=l1_effective,
                        risk_state=risk_state,
                        evidence={**evidence, "risk_timeline_tail": history_records[-self.history_max_steps:]},
                        strikes=max(counted_strikes, 1),
                        required_strikes=required,
                        theta=theta,
                        trigger_reason="risk_accumulator_evidence_fallback",
                    )

            strike_evidence_ok = (evidence_present or not self.require_encryption_evidence)
            if weak_memory and self.require_encryption_evidence:
                strike_evidence_ok = strike_evidence_ok and ransomware_specific

            if l2_prob >= theta and strike_evidence_ok:
                # Avoid treating overlapping windows as independent evidence.
                enough_gap = (step_idx + 1 - last_counted_step) >= (ws if self.non_overlapping_strikes else 1)
                if enough_gap:
                    counted_strikes += 1
                    max_counted_strikes = max(max_counted_strikes, counted_strikes)
                    last_counted_step = step_idx + 1
                elif self.verbose:
                    print("  [L2 CNN] high window ignored for strike count "
                          "because it overlaps previous counted strike")

                if counted_strikes >= required:
                    return self._dynamic_response(
                        confidence=l2_prob,
                        step=step_idx + 1,
                        pid=pid,
                        process_name=process_name,
                        p1_effective=l1_effective,
                        risk_state=risk_state,
                        evidence={**evidence, "risk_timeline_tail": history_records[-self.history_max_steps:]},
                        strikes=counted_strikes,
                        required_strikes=required,
                        theta=theta,
                        trigger_reason="non_overlapping_strike_rule",
                    )
            else:
                # Decay strike memory instead of hard reset; a single quiet tick
                # should not erase all context, but repeated clean windows will.
                if counted_strikes > 0:
                    counted_strikes = max(0, counted_strikes - 1)

        # No mitigation. Return SAFE/WATCH/SUSPICIOUS diagnostic event.
        max_l2 = max(l2_scores) if l2_scores else 0.0
        final_l1 = self._decay_l1_prob(l1_prob_initial, time.time() - t0)
        final_state = self._risk_state(final_l1)
        elapsed_ms = (time.time() - t0) * 1000

        if l1_error:
            return SystemEvent(
                alert=True,
                severity=SEVERITY_LOW,
                model_source="EDR_WATCH_L1_PREPROCESSING_ERROR",
                confidence=round(max(max_l2, final_l1), 4),
                pid=pid,
                process_name=process_name,
                recommended_actions=[ACTION_LOG_EVENT, ACTION_MONITOR_ONLY],
                features={"l1_error": l1_error, "l1_probability": final_l1,
                          "max_l2_probability": max_l2,
                          "max_l2_risk_score": max_risk_score,
                          "max_evidence_streak": max_evidence_streak,
                          "last_threshold_mode": last_threshold_mode,
                          "risk_timeline_tail": history_records[-self.history_max_steps:],
                          **last_evidence},
                description=(
                    f"WATCH — Layer 1 preprocessing/schema error occurred, so the "
                    f"process was not marked clean. No dynamic mitigation triggered. "
                    f"max_L2={max_l2:.1%}, strikes={max_counted_strikes}/{self.n_consecutive}."
                ),
            )

        alert = final_state in (RiskState.SUSPICIOUS,)
        severity = SEVERITY_MEDIUM if alert else SEVERITY_NONE
        actions = [ACTION_SEND_ALERT, ACTION_LOG_EVENT] if alert else [ACTION_LOG_EVENT]
        return SystemEvent(
            alert=alert,
            severity=severity,
            model_source="EDR_NO_MITIGATION" if alert else "EDR_SAFE",
            confidence=round(max(max_l2, final_l1), 4),
            pid=pid,
            process_name=process_name,
            recommended_actions=actions,
            features={"l1_initial_probability": l1_prob_initial,
                      "l1_effective_probability": final_l1,
                      "risk_state": final_state,
                      "max_l2_probability": max_l2,
                      "last_cnn_threshold": last_theta,
                      "max_counted_strikes": max_counted_strikes,
                      "max_l2_risk_score": max_risk_score,
                      "max_evidence_streak": max_evidence_streak,
                      "last_threshold_mode": last_threshold_mode,
                      "risk_timeline_tail": history_records[-self.history_max_steps:],
                      **last_evidence},
            description=(
                f"No mitigation over {len(telemetry_stream)} steps. "
                f"L1_initial={l1_prob_initial:.1%}, L1_effective={final_l1:.1%} "
                f"({final_state}), max_L2={max_l2:.1%}, adaptive_thr={last_theta:.1%}, "
                f"max_counted_strikes={max_counted_strikes}/{self.n_consecutive}, "
                f"max_l2_risk={max_risk_score:.2f}, max_evidence_streak={max_evidence_streak}. "
                f"Latency={elapsed_ms:.1f}ms."
            ),
        )

    # ---------------------------------------------------------------------
    # Batch replay
    # ---------------------------------------------------------------------
    def batch_replay(
        self,
        telemetry_df,
        static_feature_fn,
        window_size: int = None,
        sort_col: str = "step_number",
        label_col: str = "label",
        feature_cols: list = None,
        verbose: bool = True,
    ) -> list:
        """Replay orchestrator over every (session_id, pid) in a DataFrame."""
        feat_cols = feature_cols or self.cnn_model.feature_cols
        ws = window_size or self.cnn_model.window_size
        results = []

        for (session_id, pid), grp in telemetry_df.groupby(["session_id", "pid"]):
            grp = grp.sort_values(sort_col)
            true_label = int(grp[label_col].max()) if label_col in grp.columns else -1
            proc_name = (str(grp["process_name"].iloc[0])
                         if "process_name" in grp.columns else "unknown")

            try:
                static_feats = static_feature_fn(grp)
            except Exception:
                n_static = len(getattr(self.static_model, "feature_names", [])) or 55
                static_feats = np.zeros(n_static, dtype=np.float32)

            # Send raw/base columns when possible so the orchestrator can compute
            # derived features consistently. If caller only has feat_cols, use them.
            available = [c for c in feat_cols if c in grp.columns]
            stream = grp[available].to_dict(orient="records")
            event = self.evaluate(
                static_features=static_feats,
                telemetry_stream=stream,
                window_size=ws,
                pid=int(pid),
                process_name=proc_name,
                feature_cols=feat_cols,
            )

            record = {
                "session_id": session_id,
                "pid": int(pid),
                "process_name": proc_name,
                "true_label": true_label,
                "alert": event.alert,
                "severity": event.severity,
                "model_source": event.model_source,
                "confidence": event.confidence,
                "n_steps": len(grp),
                "description": event.description,
            }
            record.update({f"feature_{k}": v for k, v in event.features.items()
                           if isinstance(v, (int, float, str))})

            if verbose:
                ok = "?" if true_label < 0 else ("✓" if int(event.alert) == true_label else "✗")
                det = event.model_source if event.alert else "SAFE"
                print(f"  {ok} session={session_id} pid={pid:6d} "
                      f"true={true_label} → {det} conf={event.confidence:.3f}")

            results.append(record)

        return results
