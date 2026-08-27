from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "finguard_model.pkl"
THRESHOLD_PATH = BASE_DIR / "finguard_threshold.pkl"

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)


def predict_transaction(transaction):
    """
    Predict fraud probability and risk level for a transaction.
    """

    probability = float(model.predict_proba(transaction)[0][1])

    if probability >= threshold:
        risk = "HIGH RISK"
    elif probability >= 0.10:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    return {
        "fraud_probability": probability,
        "risk_level": risk
    }


def get_feature_importance(transaction):
    """
    Return the most important model features for fraud detection.
    """

    import pandas as pd

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    importance = pd.DataFrame({
        "feature": transaction.columns,
        "importance": model.feature_importances_
    })

    return importance.sort_values(
        "importance",
        ascending=False
    )
