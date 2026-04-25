"""
system_event.py  —  S004 Ransomware Detection System
═══════════════════════════════════════════════════════════════════════════════
Standard output format produced by every detection component.

Every model (RF, XGBoost, LightGBM, Isolation Forest, Autoencoder,
Rule Engine) translates its raw prediction into a SystemEvent.
The Mitigation System reads ONLY SystemEvents — it never imports any model.

Author : AI Engineering Student
Phase  : 3 — ML Models (v2, research-grade)
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# ── Severity constants ─────────────────────────────────────────────────────────
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_HIGH     = "HIGH"
SEVERITY_MEDIUM   = "MEDIUM"
SEVERITY_LOW      = "LOW"
SEVERITY_NONE     = "NONE"

# ── Action constants ───────────────────────────────────────────────────────────
ACTION_KILL_PROCESS        = "kill_process"
ACTION_SUSPEND_PROCESS     = "suspend_process"
ACTION_SEND_ALERT          = "send_alert"
ACTION_LOG_EVENT           = "log_event"
ACTION_BLOCK_NETWORK       = "block_network"
ACTION_SNAPSHOT_DIRECTORY  = "snapshot_directory"
ACTION_MONITOR_ONLY        = "monitor_only"
ACTION_NO_ACTION           = "no_action"


@dataclass
class SystemEvent:
    """
    Standardised event produced by every detection component.

    Fields
    ──────
    event_id            : UUID uniquely identifying this event
    timestamp           : Unix timestamp
    pid                 : Suspicious process ID
    process_name        : Suspicious process name
    alert               : True = suspicious activity detected
    severity            : CRITICAL / HIGH / MEDIUM / LOW / NONE
    model_source        : Which component fired (RuleEngine, RandomForest, etc.)
    confidence          : Float 0.0–1.0 = probability of ransomware
    triggered_rule      : Rule ID if from RuleEngine, else None
    features            : Feature snapshot that caused the alert
    recommended_actions : List of action strings for Mitigation System
    raw_prediction      : Raw binary label (1=ransomware, 0=benign)
    anomaly_score       : Raw score for IF / Autoencoder (None for others)
    description         : Human-readable explanation
    """

    event_id           : str           = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp          : float         = field(default_factory=time.time)
    pid                : int           = 0
    process_name       : str           = "unknown"

    alert              : bool          = False
    severity           : str           = SEVERITY_NONE
    model_source       : str           = "unknown"
    confidence         : float         = 0.0
    triggered_rule     : Optional[str] = None

    features           : Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str]      = field(default_factory=list)

    raw_prediction     : int            = 0
    anomaly_score      : Optional[float]= None
    description        : str            = ""

    # ── Helpers ───────────────────────────────────────────────────────────
    def is_critical(self)     -> bool: return self.severity == SEVERITY_CRITICAL
    def is_high_or_above(self)-> bool: return self.severity in (SEVERITY_CRITICAL, SEVERITY_HIGH)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id"            : self.event_id,
            "timestamp"           : self.timestamp,
            "pid"                 : self.pid,
            "process_name"        : self.process_name,
            "alert"               : self.alert,
            "severity"            : self.severity,
            "model_source"        : self.model_source,
            "confidence"          : round(self.confidence, 4),
            "triggered_rule"      : self.triggered_rule,
            "features"            : self.features,
            "recommended_actions" : self.recommended_actions,
            "raw_prediction"      : self.raw_prediction,
            "anomaly_score"       : self.anomaly_score,
            "description"         : self.description,
        }

    def __repr__(self) -> str:
        s = "🔴 ALERT" if self.alert else "🟢 NORMAL"
        return (f"SystemEvent({s} | {self.severity} | src={self.model_source} "
                f"| pid={self.pid} | conf={self.confidence:.2%})")


# ══════════════════════════════════════════════════════════════════════════════
# Factory functions — one per model source
# ══════════════════════════════════════════════════════════════════════════════

def _severity_from_confidence(conf: float) -> str:
    if conf >= 0.90: return SEVERITY_CRITICAL
    if conf >= 0.75: return SEVERITY_HIGH
    if conf >= 0.50: return SEVERITY_MEDIUM
    return SEVERITY_NONE

def _actions_from_severity(sev: str) -> List[str]:
    if sev == SEVERITY_CRITICAL:
        return [ACTION_KILL_PROCESS,    ACTION_SEND_ALERT, ACTION_LOG_EVENT]
    if sev == SEVERITY_HIGH:
        return [ACTION_SUSPEND_PROCESS, ACTION_SEND_ALERT, ACTION_LOG_EVENT]
    if sev == SEVERITY_MEDIUM:
        return [ACTION_SEND_ALERT,      ACTION_LOG_EVENT]
    return [ACTION_NO_ACTION]


def make_rule_engine_event(
    pid              : int,
    process_name     : str,
    rule_id          : str,
    severity         : str,
    response_actions : List[str],
    features         : Dict[str, Any],
    description      : str,
) -> SystemEvent:
    return SystemEvent(
        pid=pid, process_name=process_name,
        alert=True, severity=severity,
        model_source="RuleEngine",
        confidence=1.0 if severity == SEVERITY_CRITICAL else 0.95,
        triggered_rule=rule_id,
        features=features,
        recommended_actions=response_actions,
        raw_prediction=1,
        anomaly_score=None,
        description=description,
    )


def make_random_forest_event(
    pid             : int,
    process_name    : str,
    ransomware_proba: float,
    features        : Dict[str, Any],
) -> SystemEvent:
    alert = ransomware_proba >= 0.5
    sev   = _severity_from_confidence(ransomware_proba) if alert else SEVERITY_NONE
    return SystemEvent(
        pid=pid, process_name=process_name,
        alert=alert, severity=sev,
        model_source="RandomForest",
        confidence=ransomware_proba,
        features=features,
        recommended_actions=_actions_from_severity(sev),
        raw_prediction=int(alert),
        anomaly_score=None,
        description=f"Random Forest: {ransomware_proba:.1%} ransomware probability.",
    )


def make_xgboost_event(
    pid             : int,
    process_name    : str,
    ransomware_proba: float,
    features        : Dict[str, Any],
) -> SystemEvent:
    alert = ransomware_proba >= 0.5
    sev   = _severity_from_confidence(ransomware_proba) if alert else SEVERITY_NONE
    return SystemEvent(
        pid=pid, process_name=process_name,
        alert=alert, severity=sev,
        model_source="XGBoost",
        confidence=ransomware_proba,
        features=features,
        recommended_actions=_actions_from_severity(sev),
        raw_prediction=int(alert),
        anomaly_score=None,
        description=f"XGBoost: {ransomware_proba:.1%} ransomware probability.",
    )


def make_lightgbm_event(
    pid             : int,
    process_name    : str,
    ransomware_proba: float,
    features        : Dict[str, Any],
) -> SystemEvent:
    alert = ransomware_proba >= 0.5
    sev   = _severity_from_confidence(ransomware_proba) if alert else SEVERITY_NONE
    return SystemEvent(
        pid=pid, process_name=process_name,
        alert=alert, severity=sev,
        model_source="LightGBM",
        confidence=ransomware_proba,
        features=features,
        recommended_actions=_actions_from_severity(sev),
        raw_prediction=int(alert),
        anomaly_score=None,
        description=f"LightGBM: {ransomware_proba:.1%} ransomware probability.",
    )


def make_isolation_forest_event(
    pid          : int,
    process_name : str,
    anomaly_score: float,
    features     : Dict[str, Any],
) -> SystemEvent:
    # score_samples: more negative = more anomalous
    confidence = min(1.0, max(0.0, -anomaly_score))
    is_anomaly = anomaly_score < -0.3
    if anomaly_score < -0.5: sev = SEVERITY_HIGH
    elif is_anomaly         : sev = SEVERITY_MEDIUM
    else                    : sev = SEVERITY_NONE
    return SystemEvent(
        pid=pid, process_name=process_name,
        alert=is_anomaly, severity=sev,
        model_source="IsolationForest",
        confidence=confidence,
        features=features,
        recommended_actions=_actions_from_severity(sev),
        raw_prediction=int(is_anomaly),
        anomaly_score=anomaly_score,
        description=f"Isolation Forest anomaly score: {anomaly_score:.4f}.",
    )


def make_autoencoder_event(
    pid                  : int,
    process_name         : str,
    reconstruction_error : float,
    threshold            : float,
    features             : Dict[str, Any],
) -> SystemEvent:
    confidence = min(1.0, reconstruction_error / max(threshold, 1e-9))
    is_anomaly = reconstruction_error > threshold
    if reconstruction_error > threshold * 2.5: sev = SEVERITY_HIGH
    elif is_anomaly                           : sev = SEVERITY_MEDIUM
    else                                      : sev = SEVERITY_NONE
    return SystemEvent(
        pid=pid, process_name=process_name,
        alert=is_anomaly, severity=sev,
        model_source="Autoencoder",
        confidence=confidence,
        features=features,
        recommended_actions=_actions_from_severity(sev),
        raw_prediction=int(is_anomaly),
        anomaly_score=reconstruction_error,
        description=f"Autoencoder reconstruction error: {reconstruction_error:.6f} "
                    f"(threshold: {threshold:.6f}).",
    )


def make_ensemble_event(
    pid         : int,
    process_name: str,
    final_proba : float,
    component_scores: Dict[str, float],
    features    : Dict[str, Any],
) -> SystemEvent:
    alert = final_proba >= 0.5
    sev   = _severity_from_confidence(final_proba) if alert else SEVERITY_NONE
    desc  = (f"Ensemble probability: {final_proba:.1%}. "
             f"Components: " +
             ", ".join(f"{k}={v:.3f}" for k, v in component_scores.items()))
    return SystemEvent(
        pid=pid, process_name=process_name,
        alert=alert, severity=sev,
        model_source="Ensemble",
        confidence=final_proba,
        features=features,
        recommended_actions=_actions_from_severity(sev),
        raw_prediction=int(alert),
        anomaly_score=None,
        description=desc,
    )
