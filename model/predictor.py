from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "fingu​ard_model.pkl"
THRESHOLD_PATH = BASE_DIR / "finguard_threshold.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# LOAD THRESHOLD
# ============================================================

if THRESHOLD_PATH.exists():
    threshold = joblib.load(THRESHOLD_PATH)

    # Handle numpy/scalar/list-like saved thresholds safely
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = float(threshold[0])
else:
    threshold = 0.30


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _prepare_transaction(transaction):
    """
    Prepare a transaction DataFrame so that its columns match
    the features used by the trained model.
    """

    if not isinstance(transaction, pd.DataFrame):
        transaction = pd.DataFrame(transaction)

    transaction = transaction.copy()

    # Remove target if accidentally supplied
    if "Class" in transaction.columns:
        transaction = transaction.drop(columns=["Class"])

    # Get feature names from the trained model
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)

        missing_features = [
            feature
            for feature in expected_features
            if feature not in transaction.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing model features: "
                + ", ".join(missing_features)
            )

        # Correct column order
        transaction = transaction[expected_features]

    return transaction


def _get_probability(transaction):
    """
    Return fraud probability for a transaction.
    """

    prepared = _prepare_transaction(transaction)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Loaded model does not support predict_proba()."
        )

    probability = float(
        model.predict_proba(prepared)[0][1]
    )

    # Safety bounds
    probability = max(0.0, min(1.0, probability))

    return probability


def _get_risk_level(probability):
    """
    Convert fraud probability into a human-readable risk level.
    """

    if probability >= threshold:
        return "HIGH RISK"

    elif probability >= 0.10:
        return "MEDIUM RISK"

    else:
        return "LOW RISK"


def _get_decision(probability):
    """
    Convert probability into an operational decision.
    """

    if probability >= threshold:
        return "BLOCK / INVESTIGATE"

    elif probability >= 0.10:
        return "MANUAL REVIEW"

    else:
        return "APPROVE"


# ============================================================
# PREDICT TRANSACTION
# ============================================================

def predict_transaction(transaction):
    """
    Predict fraud probability, risk level and recommended
    operational decision for one transaction.

    Parameters
    ----------
    transaction : pandas.DataFrame
        One transaction containing the model features.

    Returns
    -------
    dict
        fraud_probability
        fraud_percentage
        risk_level
        decision
        threshold
    """

    probability = _get_probability(transaction)

    risk_level = _get_risk_level(probability)

    decision = _get_decision(probability)

    return {
        "fraud_probability": probability,
        "fraud_percentage": probability * 100,
        "risk_level": risk_level,
        "decision": decision,
        "threshold": threshold,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(transaction=None):
    """
    Return global Random Forest feature importance.

    The transaction argument is accepted for compatibility
    with the Streamlit application.
    """

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "The loaded model does not provide feature_importances_."
        )

    importances = list(model.feature_importances_)

    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        if transaction is None:
            raise ValueError(
                "Feature names are unavailable. "
                "Provide a transaction DataFrame."
            )

        if not isinstance(transaction, pd.DataFrame):
            transaction = pd.DataFrame(transaction)

        transaction = transaction.copy()

        if "Class" in transaction.columns:
            transaction = transaction.drop(columns=["Class"])

        features = list(transaction.columns)

    if len(importances) != len(features):
        raise ValueError(
            "Number of model features does not match "
            "number of feature importance values."
        )

    importance_df = pd.DataFrame(
        {
            "feature": features,
            "importance": importances,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance_df


# ============================================================
# BATCH FRAUD DETECTION
# ============================================================

def
