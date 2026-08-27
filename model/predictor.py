from pathlib import Path
import joblib
import pandas as pd
import shap


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "fingu​​ard_model.pkl"
THRESHOLD_PATH = BASE_DIR / "fingu​​ard_threshold.pkl"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
threshold = float(joblib.load(THRESHOLD_PATH))


# --------------------------------------------------
# SHAP EXPLAINER
# --------------------------------------------------

explainer = shap.TreeExplainer(model)


# --------------------------------------------------
# PREDICT TRANSACTION
# --------------------------------------------------

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


# --------------------------------------------------
# FEATURE IMPORTANCE / EXPLANATION
# --------------------------------------------------

def get_feature_importance(transaction):

    try:

        shap_values = explainer.shap_values(transaction)

        # Handle different SHAP output formats
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

        values = pd.Series(
            values,
            index=transaction.columns
        )

        importance = pd.DataFrame({
            "feature": values.index,
            "SHAP Impact": values.values,
            "absolute_impact": values.abs().values
        })

        importance = importance.sort_values(
            "absolute_impact",
            ascending=False
        )

        return importance.reset_index(drop=True)

    except Exception:

        # Fallback to Random Forest feature importance
        if hasattr(model, "feature_importances_"):

            importance = pd.DataFrame({
                "feature": transaction.columns,
                "SHAP Impact": model.feature_importances_,
                "absolute_impact": model.feature_importances_
            })

            return importance.sort_values(
                "absolute_impact",
                ascending=False
            ).reset_index(drop=True)

        return pd.DataFrame(
            columns=[
                "feature",
                "SHAP Impact",
                "absolute_impact"
            ]
        )
