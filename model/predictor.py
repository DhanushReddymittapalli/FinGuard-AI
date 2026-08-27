from pathlib import Path
import joblib
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

# predictor.py is inside /model
# Model files are in the repository root
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
    try:
        threshold = float(joblib.load(THRESHOLD_PATH))
    except Exception:
        threshold = 0.30
else:
    threshold = 0.30


# ============================================================
# GET EXPECTED FEATURES
# ============================================================

def get_expected_features():
    """
    Return the feature names expected by the trained model.
    """

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # Fallback for models saved without feature names
    return None


# ============================================================
# PREPARE TRANSACTION
# ============================================================

def prepare_transaction(transaction):
    """
    Prepare a transaction DataFrame so that it matches
    the features used during model training.
    """

    if not isinstance(transaction, pd.DataFrame):
        transaction = pd.DataFrame(transaction)

    transaction = transaction.copy()

    # Remove target column if supplied
    if "Class" in transaction.columns:
        transaction = transaction.drop(columns=["Class"])

    expected_features = get_expected_features()

    if expected_features is not None:

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

        # Keep exactly the training feature order
        transaction = transaction[expected_features]

    return transaction


# ============================================================
# PREDICT TRANSACTION
# ============================================================

def predict_transaction(transaction):
    """
    Predict fraud probability and risk level.

    Returns:
        fraud_probability
        risk_level
        threshold
        decision
    """

    transaction = prepare_transaction(transaction)

    # Get fraud probability
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(transaction)

        # For binary classification, column 1 is normally
        # the positive/fraud class.
        if probabilities.shape[1] >= 2:
            probability = float(probabilities[0][1])
        else:
            probability = float(probabilities[0][0])

    else:

        prediction = int(model.predict(transaction)[0])

        probability = 1.0 if prediction == 1 else 0.0

    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    if probability >= threshold:

        risk_level = "HIGH RISK"
        decision = "BLOCK / MANUAL REVIEW"

    elif probability >= 0.10:

        risk_level = "MEDIUM RISK"
        decision = "REVIEW TRANSACTION"

    else:

        risk_level = "LOW RISK"
        decision = "APPROVE TRANSACTION"

    return {
        "fraud_probability": probability,
        "risk_level": risk_level,
        "threshold": threshold,
        "decision": decision
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(transaction=None):
    """
    Return Random Forest feature importance.

    The transaction parameter is accepted for compatibility
    with the Streamlit application.
    """

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "The loaded model does not provide feature_importances_."
        )

    expected_features = get_expected_features()

    if expected_features is None:

        if transaction is not None:

            if not isinstance(transaction, pd.DataFrame):
                transaction = pd.DataFrame(transaction)

            features = list(transaction.columns)

        else:

            features = [
                f"Feature {i + 1}"
                for i in range(len(model.feature_importances_))
            ]

    else:

        features = expected_features

    importances = list(model.feature_importances_)

    if len(features) != len(importances):
        raise ValueError(
            "Number of model features does not match "
            "number of feature importance values."
