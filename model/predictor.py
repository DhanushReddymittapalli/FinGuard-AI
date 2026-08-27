from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "finguard_model.pkl"
THRESHOLD_PATH = BASE_DIR / "finguard_threshold.pkl"


# Load model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# Load threshold
threshold = 0.30

if THRESHOLD_PATH.exists():
    try:
        threshold = float(joblib.load(THRESHOLD_PATH))
    except Exception:
        threshold = 0.30


def get_expected_features():
    """Return features expected by the trained model."""

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return None


def prepare_transaction(transaction):
    """Prepare transaction for model prediction."""

    if not isinstance(transaction, pd.DataFrame):
        transaction = pd.DataFrame(transaction)

    transaction = transaction.copy()

    if "Class" in transaction.columns:
        transaction = transaction.drop(columns=["Class"])

    expected = get_expected_features()

    if expected is not None:
        missing = [
            col for col in expected
            if col not in transaction.columns
        ]

        if missing:
            raise ValueError(
                "Missing model features: "
                + ", ".join(missing)
            )

        transaction = transaction[expected]

    return transaction


def predict_transaction(transaction):
    """Predict fraud probability and risk."""

    transaction = prepare_transaction(transaction)

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(transaction)

        classes = list(
            getattr(model, "classes_", [0, 1])
        )

        if 1 in classes:
            fraud_index = classes.index(1)
        else:
            fraud_index = 1

        probability = float(
            probabilities[0][fraud_index]
        )

    else:

        prediction = int(
            model.predict(transaction)[0]
        )

        probability = 1.0 if prediction == 1 else 0.0


    # Risk classification

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
        "decision": decision,
    }


def get_feature_importance(transaction=None):
    """Return model feature importance."""

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "This model does not provide feature_importances_."
        )

    importances = list(
        model.feature_importances_
    )

    expected = get_expected_features()

    if expected is not None:

        features = expected

    elif transaction is not None:

        if not isinstance(transaction, pd.DataFrame):
            transaction = pd.DataFrame(transaction)

        if "Class" in transaction.columns:
            transaction = transaction.drop(columns=["Class"])

        features = list(transaction.columns)

    else:

        features = [
            f"Feature {i + 1}"
            for i in range(len(importances))
        ]

    if len(features) != len(importances):

        features = [
            f"Feature {i + 1}"
            for i in range(len(importances))
        ]

    result = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    })

    return result.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)
