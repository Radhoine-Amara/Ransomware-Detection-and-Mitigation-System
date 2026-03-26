"""
Feature extraction pipeline.

Transforms raw CSV logs (system metrics + file-system events) produced by
``src/collector/monitor.py`` into a labelled feature DataFrame ready for
model training.
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System-metrics features
# ---------------------------------------------------------------------------

def load_system_metrics(path: str) -> pd.DataFrame:
    """Load raw system-metrics CSV and parse the timestamp column."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def extract_system_features(df: pd.DataFrame, window: str = "10s") -> pd.DataFrame:
    """Compute rolling-window statistics over system metrics.

    Args:
        df: Raw system-metrics DataFrame (output of :func:`load_system_metrics`).
        window: Rolling window size as a pandas offset string (e.g. ``"10s"``).

    Returns:
        DataFrame with one row per original row and added ``*_mean`` /
        ``*_std`` columns for each numeric metric.
    """
    df = df.sort_values("timestamp").set_index("timestamp")
    numeric_cols = df.select_dtypes("number").columns.tolist()
    rolled = df[numeric_cols].rolling(window)
    means = rolled.mean().rename(columns=lambda c: f"{c}_mean")
    stds = rolled.std().rename(columns=lambda c: f"{c}_std")
    features = pd.concat([df[numeric_cols], means, stds], axis=1).reset_index()
    return features


# ---------------------------------------------------------------------------
# File-event features
# ---------------------------------------------------------------------------

def load_file_events(path: str) -> pd.DataFrame:
    """Load raw file-system-event CSV and parse the timestamp column."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def extract_file_features(df: pd.DataFrame, window: str = "10s") -> pd.DataFrame:
    """Aggregate file-system events into rolling-window counts.

    Counts how many ``created``, ``modified``, ``deleted``, and ``moved``
    events occurred in the rolling window.  High rename / delete rates are
    a common indicator of ransomware activity.

    Args:
        df: Raw file-event DataFrame (output of :func:`load_file_events`).
        window: Rolling window size as a pandas offset string.

    Returns:
        DataFrame indexed by timestamp with event-count columns.
    """
    df = df.sort_values("timestamp").set_index("timestamp")
    event_dummies = pd.get_dummies(df["event_type"])
    for col in ["created", "modified", "deleted", "moved"]:
        if col not in event_dummies.columns:
            event_dummies[col] = 0
    rolled = event_dummies.rolling(window).sum()
    rolled.columns = [f"file_{c}_count" for c in rolled.columns]
    return rolled.reset_index()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def build_feature_dataframe(
    system_metrics_path: str,
    file_events_path: str,
    output_path: str,
    window: str = "10s",
) -> pd.DataFrame:
    """Merge system and file features into a single DataFrame and save it.

    Args:
        system_metrics_path: Path to raw system-metrics CSV.
        file_events_path: Path to raw file-events CSV.
        output_path: Where to write the processed feature CSV.
        window: Rolling window size.

    Returns:
        Merged feature DataFrame.
    """
    sys_feat = extract_system_features(load_system_metrics(system_metrics_path), window)
    file_feat = extract_file_features(load_file_events(file_events_path), window)

    # Merge on nearest timestamp (tolerance = window)
    merged = pd.merge_asof(
        sys_feat.sort_values("timestamp"),
        file_feat.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info("Saved feature DataFrame (%d rows) to %s", len(merged), output_path)
    return merged
