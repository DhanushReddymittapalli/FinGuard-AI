from pathlib import Path
import joblib
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "finguard_model.pkl"
THRESHOLD_PATH = BASE_DIR / "finguard_threshold.pkl"

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

# SHAP explainer for Random Forest
explainer = shap.TreeExplainer(model)


def predict_transaction(transaction):
    """
    Predict fraud probability and risk level.
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
    Explain why this specific transaction received
    its fraud prediction using SHAP.
    """

    shap_values = explainer.shap_values(transaction)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

        if hasattr(values, "ndim") and values.ndim == 2:
            values = values[:, 1]

    importance = pd.DataFrame({
        "feature": transaction.columns,
        "SHAP Impact": values,
        "absolute_impact": abs(values)
    })

    importance = importance.sort_values(
        "absolute_impact",
        ascending=False
    )

    return importance
