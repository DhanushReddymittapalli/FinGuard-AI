from pathlib import Path
import joblib
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "finguard_model.pkl"
THRESHOLD_PATH = BASE_DIR / "finguard_threshold.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# LOAD THRESHOLD
# ============================================================

threshold = float(joblib.load(THRESHOLD_PATH))


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)


# ============================================================
# PREDICT TRANSACTION
# ============================================================

def predict_transaction(transaction):

    probability = float(
        model.predict_proba(transaction)[0][1]
    )

    if probability >= threshold:
        risk = "HIGH RISK"
    elif probability >= 0.10:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    return {
        "fraud_probability": probability,
        "risk_level": risk,
        "threshold": threshold
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(transaction):

    try:

        shap_values = explainer.shap_values(transaction)

        if isinstance(shap_values, list):

            if len(shap_values) > 1:
                values = shap_values[1][0]
            else:
                values = shap_values[0][0]

        else:

            values = shap_values

            if hasattr(values, "ndim"):

                if values.ndim == 3:
                    values = values[0, :, 1]

                elif values.ndim == 2:
                    values = values[0]

        values = list(values)

        importance = pd.DataFrame(
            {
                "feature": transaction.columns,
                "SHAP Impact": values,
                "absolute_impact": [
                    abs(value) for value in values
                ]
            }
        )

        importance = importance.sort_values(
            "absolute_impact",
            ascending=False
        )

        return importance.reset_index(drop=True)

    except Exception:

        # Random Forest fallback
        if hasattr(model, "feature_importances_"):

            importance = pd.DataFrame(
                {
                    "feature": transaction.columns,
                    "SHAP Impact": model.feature_importances_,
                    "absolute_impact": model.feature_importances_
                }
            )

            importance = importance.sort_values(
                "absolute_impact",
                ascending=False
            )

            return importance.reset_index(drop=True)

        return pd.DataFrame(
            columns=[
                "feature",
                "SHAP Impact",
                "absolute_impact"
            ]
        )
        def predict_transaction_batch(transactions):
    """
    Predict fraud probability for multiple transactions.
    """

    probabilities = model.predict_proba(transactions)[:, 1]

    return probabilities.tolist()
