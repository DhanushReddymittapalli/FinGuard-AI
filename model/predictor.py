from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "finguard_model.pkl"
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
    threshold = float(
        joblib.load(THRESHOLD_PATH)
    )
else:
    threshold = 0.30


# ============================================================
# PREDICT TRANSACTION
# ============================================================

def predict_transaction(transaction):
    """
    Predict fraud probability for one transaction.

    Parameters
    ----------
    transaction : pandas.DataFrame
        One transaction containing the model features.

    Returns
    -------
    dict
        Fraud probability, risk level and threshold.
    """

    if not isinstance(transaction, pd.DataFrame):
        transaction = pd.DataFrame(transaction)

    transaction = transaction.copy()

    # Remove target column if accidentally supplied
    if "Class" in transaction.columns:
        transaction = transaction.drop(
            columns=["Class"]
        )

    # Make sure the model receives the same
    # features and order used during training.
    if hasattr(model, "feature_names_in_"):

        expected_features = list(
            model.feature_names_in_
        )

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

        transaction = transaction[
            expected_features
        ]

    probability = float(
        model.predict_proba(transaction)[0][1]
    )

    if probability >= threshold:
        risk_level = "HIGH RISK"
    elif probability >= 0.10:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "LOW RISK"

    return {
        "fraud_probability": probability,
        "risk_level": risk_level,
        "threshold": threshold
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(transaction):
    """
    Return feature importance for the transaction.

    Uses Random Forest feature importance.
    """

    if not isinstance(transaction, pd.DataFrame):
        transaction = pd.DataFrame(transaction)

    transaction = transaction.copy()

    if "Class" in transaction.columns:
        transaction = transaction.drop(
            columns=["Class"]
        )

    # Align features with the trained model
    if hasattr(model, "feature_names_in_"):

        expected_features = list(
            model.feature_names_in_
        )

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

        features = expected_features

    else:
        features = list(
            transaction.columns
        )

    # Random Forest feature importance
    if hasattr(model, "feature_importances_"):

        importances = list(
            model.feature_importances_
        )

        if len(importances) != len(features):
            raise ValueError(
                "Number of model features does not "
                "match feature importance values."
            )

        importance_df = pd.DataFrame(
            {
                "feature": features,
                "importance": importances
            }
        )

        # Sort from most important to least important
        importance_df = importance_df.sort_values(
            by="importance",
            ascending=False
        ).reset_index(drop=True)

        return importance_df

    raise ValueError(
        "The loaded model does not provide "
        "feature_importances_."
    )
