"""
Machine-learning model training and evaluation for ransomware detection.

Supports:
- RandomForestClassifier (default)
- Training / evaluation with standard sklearn metrics
- Saving and loading trained models with joblib
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str, label_column: str = "label") -> tuple[pd.DataFrame, pd.Series]:
    """Load a processed feature CSV and split into *X* and *y*.

    Args:
        path: Path to the feature CSV produced by the feature-extraction pipeline.
        label_column: Name of the binary label column (1 = ransomware, 0 = benign).

    Returns:
        Tuple ``(X, y)`` where *X* is a DataFrame of feature columns and *y*
        is the label Series.
    """
    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {path}")
    y = df[label_column]
    X = df.drop(columns=[label_column])
    # Drop non-numeric / timestamp columns
    X = X.select_dtypes(include=[np.number])
    return X, y


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> tuple:
    """Train a RandomForestClassifier and return the fitted model and metrics.

    Args:
        X: Feature matrix.
        y: Binary label vector.
        test_size: Fraction of data reserved for the test split.
        random_state: RNG seed for reproducibility.
        n_estimators: Number of trees in the random forest.

    Returns:
        Tuple ``(model, scaler, report_dict)`` where *report_dict* contains
        the classification report as a dict.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))
    logger.info("Confusion matrix:\n%s", cm)

    return model, scaler, report


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(model, scaler, model_dir: str):
    """Persist the trained model and scaler to *model_dir*.

    Args:
        model: Fitted sklearn estimator.
        scaler: Fitted StandardScaler.
        model_dir: Directory where ``model.joblib`` and ``scaler.joblib``
            will be written.
    """
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    logger.info("Saved model and scaler to %s", model_dir)


def load_model(model_dir: str):
    """Load a previously saved model and scaler from *model_dir*.

    Returns:
        Tuple ``(model, scaler)``.
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    return model, scaler


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict(model, scaler, X: pd.DataFrame) -> np.ndarray:
    """Run inference on a feature DataFrame.

    Args:
        model: Fitted sklearn estimator.
        scaler: Fitted StandardScaler.
        X: Feature matrix (must have the same columns used during training).

    Returns:
        Array of binary predictions (1 = ransomware, 0 = benign).
    """
    X_scaled = scaler.transform(X.select_dtypes(include=[np.number]))
    return model.predict(X_scaled)
