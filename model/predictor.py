from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

# predictor.py is inside: model/
# Model files are in the repository root.
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

threshold = 0.30

if THRESHOLD_PATH.exists():
    try:
        threshold = float(joblib.load(THRESHOLD_PATH))
    except Exception:
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

    return None


# ============================================================
# PREPARE TRANSACTION
# ============================================================

def prepare_transaction(transaction):
    """
    Prepare a transaction DataFrame for prediction.

    Removes the target column if present and keeps the
    same feature order used during model training.
    """

    if not isinstance(transaction, pd.DataFrame):
        transaction = pd.DataFrame(transaction)

    transaction = transaction.copy()

    # Remove target column if supplied
    if "Class" in transaction.columns:
        transaction = transaction.drop(columns=["Class"])

    expected_features = get_expected_features()

    # If the model contains feature names, use them.
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
    Predict fraud probability, risk level and decision.

    Returns:
        fraud_probability
        risk_level
        threshold
        decision
    """

    transaction = prepare_transaction(transaction)

    # --------------------------------------------------------
    # FRAUD PROBABILITY
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(transaction)

        # Binary classification
        if probabilities.shape[1] >= 2:

            # Normally class 1 = fraud.
            classes = list(getattr(model, "classes_", [0, 1]))

            if 1 in classes:
                fraud_index = classes.index(1)
            else:
                # Fall back to the second probability
                fraud_index = 1

            probability = float(
                probabilities[0][fraud_index]
            )

        else:
            probability = float(
                probabilities[0][0]
            )

    else:

        prediction = int(
            model.predict(transaction)[0]
        )

        probability = 1.0 if prediction == 1 else 0.0


    # --------------------------------------------------------
    # RISK CLASSIFICATION
    # --------------------------------------------------------

    if probability >= threshold:

        risk_level = "HIGH RISK"
        decision = "BLOCK / MANUAL REVIEW"

    elif probability >= 0.10:

        risk_level = "MEDIUM RISK"
        decision = "REVIEW TRANSACTION"

    else:

        risk_level = "LOW RISK"
        decision = "APPROVE TRANSACTION"


    # --------------------------------------------------------
    # ALWAYS RETURN ALL REQUIRED VALUES
    # --------------------------------------------------------

    return {
        "fraud_probability": probability,
        "risk_level": risk_level,
        "threshold": threshold,
        "decision": decision,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(transaction=None):
    """
    Return Random Forest feature importance.

    Returns a DataFrame containing:
        Feature
        Importance
    """

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "The loaded model does not provide "
            "feature_importances_."
        )

    importances = list(model.feature_importances_)

    expected_features = get_expected_features()

    # --------------------------------------------------------
    # USE MODEL FEATURE NAMES WHEN AVAILABLE
    # --------------------------------------------------------

    if expected_features is not None:

        features = expected_features

    # --------------------------------------------------------
    # OTHERWISE USE TRANSACTION COLUMN NAMES
    # --------------------------------------------------------

    elif transaction is not None:

        if not isinstance(transaction, pd.DataFrame):
            transaction = pd.DataFrame(transaction)

        transaction = transaction.copy()

        if "Class" in transaction.columns:
            transaction = transaction.drop(columns=["Class"])

        features = list(transaction.columns)

    # --------------------------------------------------------
    # LAST FALLBACK
    # --------------------------------------------------------

    else:

        features = [
            f"Feature {i + 1}"
            for i in range(len(importances))
        ]


    # --------------------------------------------------------
    # GUARANTEE SAME LENGTH
    # --------------------------------------------------------

    if len(features) != len(importances):

        features = [
            f"Feature {i + 1}"
            for i in range(len(importances))
        ]


    # --------------------------------------------------------
    # CREATE RESULT
    # --------------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "Feature
