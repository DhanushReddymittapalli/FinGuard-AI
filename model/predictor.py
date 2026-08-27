from pathlib import Path
import joblib
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "fingu​​ard_model.pkl"
THRESHOLD_PATH = BASE_DIR / "fingu​​ard_threshold.pkl"


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
    threshold = float(joblib.load(THRESHOLD_PATH))
else:
    # Default threshold
    threshold = 0.30


# ============================================================
# PREDICT TRANSACTION
# ============================================================

def predict_transaction(transaction):
    """
    Predict fraud probability, risk level and recommended
    decision for one transaction.

    Parameters
    ----------
    transaction : pandas.DataFrame
        One transaction containing the same features used
        during model training.

    Returns
    -------
    dict
        fraud_probability
        risk_level
        threshold
        decision
    """

    # --------------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------------

    if not isinstance(transaction, pd.DataFrame):
        transaction = pd.DataFrame(transaction)

    transaction = transaction.copy()

    # --------------------------------------------------------
    # Remove target column if supplied
    # --------------------------------------------------------

    if "Class" in transaction.columns:
        transaction = transaction.drop(columns=["Class"])

    # --------------------------------------------------------
    # Get expected model features
    # --------------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        expected_features = list(model.feature_names_in_)

        # Find missing features
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

        # Keep exactly the features used during training
        transaction = transaction[expected_features]

    else:
        # Fallback for models without feature_names_in_
        expected_features = list(transaction.columns)

    # --------------------------------------------------------
    # Predict fraud probability
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(transaction)[0][1]
    )

    # Keep probability between 0 and 1
    probability = max(0.0, min(1.0, probability))

    # --------------------------------------------------------
    # Determine risk level
    # --------------------------------------------------------

    if probability >= threshold:
        risk_level = "HIGH RISK"

    elif probability >= 0.10:
        risk_level = "MEDIUM RISK"

    else:
        risk_level = "LOW RISK"

    # --------------------------------------------------------
    # Recommended decision
    # --------------------------------------------------------

    if risk_level == "HIGH RISK":
        decision = "BLOCK TRANSACTION"

    elif risk_level == "MEDIUM RISK":
        decision = "REVIEW TRANSACTION"

    else:
        decision = "APPROVE TRANSACTION"

    # --------------------------------------------------------
    # Return complete prediction
    # --------------------------------------------------------

    return {
        "fraud_probability": probability,
        "risk_level": risk_level,
        "threshold": threshold,
        "decision": decision
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(transaction):
    """
    Return Random Forest feature importance.

    Parameters
    ----------
    transaction : pandas.DataFrame
        Transaction data used for prediction.

    Returns
    -------
    pandas.DataFrame
        Feature names and importance values.
    """

    if not isinstance(transaction, pd.DataFrame):
        transaction = pd.DataFrame(transaction)

    transaction = transaction.copy()

    # --------------------------------------------------------
    # Remove target column
    # --------------------------------------------------------

    if "Class" in transaction.columns:
        transaction = transaction.drop(columns=["Class"])

    # --------------------------------------------------------
    # Align features with trained model
    # --------------------------------------------------------

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

        features = expected_features

    else:
        features = list(transaction.columns)

    # --------------------------------------------------------
    # Get feature importance
    # --------------------------------------------------------

    if hasattr(model, "feature_importances_"):

        importances = list(model.feature_importances_)

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

        # Sort most important to least important
        importance_df = (
            importance_df
            .sort_values(
                by="importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        return importance_df

    # --------------------------------------------------------
    # Model does not support feature importance
    # --------------------------------------------------------

    raise ValueError(
        "The loaded model does not provide "
        "feature_importances_."
    )
