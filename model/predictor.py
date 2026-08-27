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
# LOAD DECISION THRESHOLD
# ============================================================

if THRESHOLD_PATH.exists():
    threshold = float(joblib.load(THRESHOLD_PATH))
else:
    threshold = 0.30


# ============================================================
# MODEL FEATURES
# ============================================================

if hasattr(model, "feature_names_in_"):
    MODEL_FEATURES = list(model.feature_names_in_)
else:
    MODEL_FEATURES = None


# ============================================================
# PREPARE TRANSACTION
# ============================================================

def _prepare_transaction(transaction):
    """
    Prepare one transaction so that it matches
    the features used while training the model.
    """

    if isinstance(transaction, pd.Series):
        transaction = transaction.to_frame().T

    elif not isinstance(transaction, pd.DataFrame):
        transaction = pd.DataFrame(transaction)

    transaction = transaction.copy()

    # Remove target column if supplied
    if "Class" in transaction.columns:
        transaction = transaction.drop(columns=["Class"])

    # Make sure all values are numeric
    for column in transaction.columns:
        transaction[column] = pd.to_numeric(
            transaction[column],
            errors="coerce"
        )

    # Replace invalid values
    transaction = transaction.fillna(0)

    # Match training feature order
    if MODEL_FEATURES is not None:

        missing_features = [
            feature
            for feature in MODEL_FEATURES
            if feature not in transaction.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing model features: "
                + ", ".join(missing_features)
            )

        transaction = transaction[MODEL_FEATURES]

    return transaction


# ============================================================
# PREDICT TRANSACTION
# ============================================================

def predict_transaction(transaction):
    """
    Predict fraud probability for one transaction.

    Returns:
        dict containing:
        - fraud_probability
        - risk_level
        - threshold
        - decision
    """

    transaction = _prepare_transaction(transaction)

    probability = float(
        model.predict_proba(transaction)[0][1]
    )

    if probability >= threshold:
        risk_level = "HIGH RISK"
        decision = "BLOCK / REVIEW"

    elif probability >= 0.10:
        risk_level = "MEDIUM RISK"
        decision = "REVIEW"

    else:
        risk_level = "LOW RISK"
        decision = "ALLOW"

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
    Return global Random Forest feature importance.

    This shows which features are generally important
    to the trained model.
    """

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "The loaded model does not provide "
            "feature_importances_."
        )

    importances = list(model.feature_importances_)

    if MODEL_FEATURES is not None:
        features = MODEL_FEATURES
    elif transaction is not None:

        if isinstance(transaction, pd.Series):
            features = list(transaction.index)
        else:
            features = list(transaction.columns)

        if "Class" in features:
            features.remove("Class")

    else:
        features = [
            f"Feature_{i}"
            for i in range(len(importances))
        ]

    if len(features) != len(importances):
        raise ValueError(
            "Number of features does not match "
            "number of importance values."
        )

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": importances
    })

    return (
        importance_df
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def get_shap_explanation(transaction, top_n=10):
    """
    Generate a transaction-level SHAP explanation.

    Returns a DataFrame containing:
        feature
        value
        shap_value
        absolute_impact
        impact
    """

    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP is not installed. Add 'shap' "
            "to requirements.txt."
        )

    transaction = _prepare_transaction(transaction)

    # TreeExplainer works well with Random Forest
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(transaction)

    # Handle different SHAP versions / output formats
    if isinstance(shap_values, list):

        if len(shap_values) > 1:
            values = shap_values[1][0]
        else:
            values = shap_values[0][0]

    else:

        values = shap_values

        # Newer SHAP versions can return:
        # (samples, features, classes)
        if hasattr(values, "ndim"):

            if values.ndim == 3:
                values = values[0, :, 1]

            elif values.ndim == 2:
                values = values[0]

    values = list(values)

    feature_names = list(transaction.columns)
    feature_values = transaction.iloc[0].tolist()

    explanation = pd.DataFrame({
        "feature": feature_names,
        "value": feature_values,
        "shap_value": values
    })

    explanation["absolute_impact"] = (
        explanation["shap_value"].abs()
    )

    explanation["impact"] = explanation["shap_value"].apply(
        lambda x: (
            "Increases fraud risk"
            if x > 0
            else "Decreases fraud risk"
        )
    )

    explanation = (
        explanation
        .sort_values(
            by="absolute_impact",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return explanation


# ============================================================
# COMPLETE TRANSACTION ANALYSIS
# ============================================================

def analyze_transaction(transaction, top_n=10):
    """
    Run prediction + SHAP explanation together.
    """

    prediction = predict_transaction(transaction)

    try:
        explanation = get_shap_explanation(
            transaction,
            top_n=top_n
        )
    except Exception:
        explanation = get_feature_importance(
            transaction
        ).head(top_n)

        explanation["value"] = 0
        explanation["shap_value"] = 0
        explanation["absolute_impact"] = (
            explanation["importance"]
        )
        explanation["impact"] = "Global model importance"

    return {
        "prediction": prediction,
        "explanation": explanation
    }
