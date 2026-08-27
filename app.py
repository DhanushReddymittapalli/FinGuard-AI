import streamlit as st
import pandas as pd
from pathlib import Path

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
DEMO_DATA_PATH = BASE_DIR / "demo_transactions_small.csv"


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
# HEADER
# ============================================================

st.title("🛡️ FinGuard AI")

st.subheader(
    "AI-Powered Credit Card Fraud Detection"
)

st.write(
    "FinGuard AI uses machine learning to detect "
    "suspicious credit-card transactions and classify "
    "their risk level."
)


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


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

with col2:
    st.metric(
        "Fraud Cases",
        f"{fraud_cases:,}"
    )

with col3:
    st.metric(
        "Legitimate",
        f"{legitimate_cases:,}"
    )

with col4:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

st.header("📈 Dataset Distribution")

if "Class" in df.columns:

    distribution_df = pd.DataFrame(
        {
            "Type": [
                "Fraud",
                "Legitimate"
            ],
            "Transactions": [
                fraud_cases,
                legitimate_cases
            ],
        }
    )

    st.bar_chart(
        distribution_df.set_index("Type")
    )

else:

    st.info(
        "Class column is not available in the dataset."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("⚙️ Model Performance")

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:
    st.metric(
        "Model",
        "Random Forest"
    )

with model_col2:
    st.metric(
        "ROC-AUC",
        "0.953"
    )

with model_col3:
    st.metric(
        "PR-AUC",
        "0.854"
    )

st.caption(
    "Decision threshold: 0.30"
)


# ============================================================
# ADVANCED BATCH FRAUD DETECTION
# ============================================================

st.header("📁 Advanced Batch Fraud Detection")

st.write(
    "Upload a CSV containing transactions. "
    "FinGuard AI will analyze every transaction and "
    "generate fraud probability, risk level, and "
    "recommended decision."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        batch_df = pd.read_csv(
            uploaded_file
        )

        st.success(
            f"Loaded {len(batch_df):,} transactions."
        )

        st.subheader("📋 Uploaded Data")

        st.dataframe(
            batch_df.head(10),
            use_container_width=True
        )

        if st.button(
            "🚨 Analyze All Transactions",
            type="primary",
            key="batch_button"
        ):

            predictions = []

            progress_bar = st.progress(0)

            status_text = st.empty()

            total_rows = len(batch_df)


            for index in range(total_rows):

                row = batch_df.iloc[index]

                transaction = (
                    row.to_frame().T
                )

                try:

                    prediction = predict_transaction(
                        transaction
                    )

                    probability = float(
                        prediction["fraud_probability"]
                    )

                    risk = prediction[
                        "risk_level"
                    ]

                    decision = prediction[
                        "decision"
                    ]

                    error_message = ""

                except Exception as e:

                    probability = 0.0

                    risk = "ERROR"

                    decision = "UNABLE TO ANALYZE"

                    error_message = str(e)


                predictions.append(
                    {
                        "Fraud Probability": probability,
                        "Risk Level": risk,
                        "Decision": decision,
                        "Analysis Error": error_message,
                    }
                )


                progress_bar.progress(
                    (index + 1) / total_rows
                )

                status_text.write(
                    f"Analyzing transaction "
                    f"{index + 1:,} of "
                    f"{total_rows:,}"
                )


            progress_bar.empty()
            status_text.empty()


            # ------------------------------------------------
            # CREATE RESULT DATAFRAME
            # ------------------------------------------------

            prediction_df = pd.DataFrame(
                predictions
            )

            analyzed_df = batch_df.copy()

            analyzed_df[
                "Fraud Probability"
            ] = prediction_df[
                "Fraud Probability"
            ]

            analyzed_df[
                "Risk Level"
            ] = prediction_df[
                "Risk Level"
            ]

            analyzed_df[
                "Decision"
            ] = prediction_df[
                "Decision"
            ]

            analyzed_df[
                "Analysis Error"
            ] = prediction_df[
                "Analysis Error"
            ]


            st.success(
                "✅ Batch analysis completed."
            )


            # ------------------------------------------------
            # BATCH SUMMARY
            # ------------------------------------------------

            st.subheader(
                "📊 Batch Analysis Summary"
            )

            analyzed_count = len(
                analyzed_df
            )

            high_risk_count = int(
                (
                    analyzed_df["Risk Level"]
                    == "HIGH RISK"
                ).sum()
            )

            medium_risk_count = int(
                (
                    analyzed_df["Risk Level"]
                    == "MEDIUM RISK"
                ).sum()
            )

            low_risk_count = int(
                (
                    analyzed_df["Risk Level"]
                    == "LOW RISK"
                ).sum()
            )

            error_count = int(
                (
                   
