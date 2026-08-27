import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from model.predictor import (
    predict_transaction,
    get_feature_importance,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "finguard_model.pkl"
THRESHOLD_PATH = BASE_DIR / "finguard_threshold.pkl"
DEMO_DATA_PATH = BASE_DIR / "demo_transactions_small.csv"
FULL_DATA_PATH = BASE_DIR / "transactions.csv"


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ FinGuard AI")

st.subheader(
    "AI-Powered Credit Card Fraud Detection"
)

st.write(
    "An AI-powered machine-learning system for detecting "
    "and analyzing suspicious credit-card transactions."
)


# ============================================================
# LOAD DEMO DATASET
# ============================================================

@st.cache_data
def load_demo_dataset():
    if not DEMO_DATA_PATH.exists():
        raise FileNotFoundError(
            "demo_transactions_small.csv was not found "
            "in the repository root."
        )

    return pd.read_csv(DEMO_DATA_PATH)


try:
    df = load_demo_dataset()

except Exception as e:
    st.error(f"Unable to load demo dataset: {e}")
    st.stop()


# ============================================================
# FRAUD MONITORING DASHBOARD
# ============================================================

st.header("📊 Fraud Monitoring Dashboard")

total_transactions = len(df)

if "Class" in df.columns:
    fraud_cases = int(df["Class"].sum())
else:
    fraud_cases = 0

legitimate_cases = total_transactions - fraud_cases

if total_transactions > 0:
    fraud_rate = (
        fraud_cases / total_transactions
    ) * 100
else:
    fraud_rate = 0.0


# ============================================================
# DASHBOARD METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Transactions",
        f"{total_transactions:,}",
    )

with col2:
    st.metric(
        "Fraud Cases",
        f"{fraud_cases:,}",
    )

with col3:
    st.metric(
        "Legitimate",
        f"{legitimate_cases:,}",
    )

with col4:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%",
    )


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

st.header("📈 Dataset Distribution")

if "Class" in df.columns:

    chart_df = pd.DataFrame(
        {
            "Type": [
                "Fraud",
                "Legitimate",
            ],
            "Count": [
                fraud_cases,
                legitimate_cases,
            ],
        }
    )

    st.bar_chart(
        chart_df.set_index("Type")
    )

else:

    st.info(
        "Class column is not available in the demo dataset."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("⚙️ Model Performance")

evaluation_results = None

try:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "finguard_model.pkl was not found."
        )

    evaluation_model = joblib.load(MODEL_PATH)

    # --------------------------------------------------------
    # Select evaluation dataset
    # --------------------------------------------------------

    if FULL_DATA_PATH.exists():

        evaluation_df = pd.read_csv(
            FULL_DATA_PATH
        )

    else:

        evaluation_df = df.copy()

    if "Class" not in evaluation_df.columns:

        st.warning(
            "The evaluation dataset does not contain "
            "a Class column. Model metrics cannot be calculated."
        )

    else:

        y_true = (
            evaluation_df["Class"]
            .astype(int)
        )

        # ----------------------------------------------------
        # Determine model features
        # ----------------------------------------------------

        if hasattr(
            evaluation_model,
            "feature_names_in_",
        ):

            model_features = list(
                evaluation_model.feature_names_in_
            )

        else:

            model_features = [
                column
                for column in evaluation_df.columns
                if column != "Class"
            ]

        # ----------------------------------------------------
        # Check missing features
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in model_features
            if feature not in evaluation_df.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing model features: "
                + ", ".join(missing_features)
            )

        # ----------------------------------------------------
        # Prepare evaluation data
        # ----------------------------------------------------

        X_eval = evaluation_df[
            model_features
        ]

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = evaluation_model.predict(
            X_eval
        )

        # ----------------------------------------------------
        # Fraud probabilities
        # ----------------------------------------------------

        if hasattr(
            evaluation_model,
            "predict_proba",
        ):

            probabilities = (
                evaluation_model.predict_proba(
                    X_eval
                )
            )

            if hasattr(
                evaluation_model,
                "classes_",
            ):

                classes = list(
                    evaluation_model.classes_
                )

                if 1 in classes:

                    fraud_index = classes.index(1)

                    y_probability = (
                        probabilities[:, fraud_index]
                    )

                else:

                    y_probability = (
                        probabilities[:, -1]
                    )

            else:

                y_probability = (
                    probabilities[:, -1]
                )

        else:

            y_probability = (
                y_pred.astype(float)
            )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        try:

            roc_auc = roc_auc_score(
                y_true,
                y_probability,
            )

        except Exception:

            roc_auc = 0.0

        try:

            pr_auc = average_precision_score(
                y_true,
                y_probability,
            )

        except Exception:

            pr_auc = 0.0

        evaluation_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "y_true": y_true,
            "y_pred": y_pred,
        }

        # ----------------------------------------------------
        # Display metrics
        # ----------------------------------------------------

        st.subheader("📊 Classification Metrics")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "Accuracy",
                f"{accuracy:.3f}",
            )

        with m2:
            st.metric(
                "Precision",
                f"{precision:.3f}",
            )

        with m3:
            st.metric(
                "Recall",
                f"{recall:.3f}",
            )

        m4, m5, m6 = st.columns(3)

        with m4:
            st.metric(
                "F1 Score",
                f"{f1:.3f}",
            )

        with m5:
            st.metric(
                "ROC-AUC",
                f"{roc_auc:.3f}",
            )

        with m6:
            st.metric(
                "PR-AUC",
                f"{pr_auc:.3f}",
            )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        st.subheader("🔲 Confusion Matrix")

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual Legitimate",
                "Actual
