import streamlit as st
import pandas as pd
from pathlib import Path

from model.predictor import predict_transaction, get_feature_importance


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide",
)

# ============================================================
# HEADER
# ============================================================

st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "FinGuard AI uses machine learning to detect suspicious "
    "credit-card transactions and classify their risk level."
)

# ============================================================
# LOAD DEMO DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEMO_FILE = BASE_DIR / "demo_transactions_small.csv"


@st.cache_data
def load_demo_data():
    return pd.read_csv(DEMO_FILE)


try:
    df = load_demo_data()
except Exception as error:
    st.error(
        f"Unable to load demo_transactions_small.csv: {error}"
    )
    st.stop()

# ============================================================
# DASHBOARD
# ============================================================

st.header("📊 Fraud Monitoring Dashboard")

total_transactions = len(df)

if "Class" in df.columns:
    fraud_cases = int(
        pd.to_numeric(
            df["Class"],
            errors="coerce"
        ).fillna(0).sum()
    )
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

    chart_df = pd.DataFrame(
        {
            "Type": [
                "Fraud",
                "Legitimate"
            ],
            "Count": [
                fraud_cases,
                legitimate_cases
            ],
        }
    )

    st.bar_chart(
        chart_df.set_index("Type")
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
    "Upload a CSV containing transactions. FinGuard AI will "
    "analyze each transaction and generate fraud probability, "
    "risk level, and recommended decision."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
)

if uploaded_file is not None:

    try:

        batch_df = pd.read_csv(
            uploaded_file
        )

        st.success(
            f"Loaded {len(batch_df):,} transactions."
        )

        st.subheader(
            "📋 Uploaded Data"
        )

        st.dataframe(
            batch_df.head(10),
            use_container_width=True
        )

        if st.button(
            "🚨 Analyze All Transactions",
            type="primary",
            key="batch_analysis_button",
        ):

            if len(batch_df) == 0:

                st.warning(
                    "The uploaded CSV contains no transactions."
                )

            else:

                predictions = []

                progress_bar = st.progress(
                    0.0
                )

                status_text = st.empty()

                total_rows = len(batch_df)

                for index in range(total_rows):

                    row = batch_df.iloc[index]

                    transaction = row.to_frame().T

                    try:

                        result = predict_transaction(
                            transaction
                        )

                        probability = float(
                            result["fraud_probability"]
                        )

                        risk_level = str(
                            result["risk_level"]
                        )

                        decision = str(
                            result["decision"]
                        )

                        error_message = ""

                    except Exception as error:

                        probability = 0.0

                        risk_level = "ERROR"

                        decision = "UNABLE TO ANALYZE"

                        error_message = str(
                            error
                        )

                    predictions.append(
                        {
                            "Fraud Probability": probability,
                            "Risk Level": risk_level,
                            "Decision": decision,
                            "Analysis Error": error_message,
                        }
                    )

                    progress_bar.progress(
                        (index + 1) / total_rows
                    )

                    status_text.write(
                        f"Analyzing transaction "
                        f"{index + 1} of {total_rows}..."
                    )

                progress_bar.empty()

                status_text.empty()

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

                # ====================================================
                # BATCH SUMMARY
                # ====================================================

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
                        analyzed_df["Risk Level"]
                        == "ERROR"
                    ).sum()
                )

                if analyzed_count > 0:

                    high_risk_rate = (
                        high_risk_count
                        / analyzed_count
                    ) * 100

                else:

                    high_risk_rate = 0.0

                summary_col1, summary_col2, summary_col3, summary_col4 = (
                    st.columns(4)
                )

                with summary_col1:

                    st.metric(
                        "Analyzed",
                        f"{analyzed_count:,}"
                    )

                with summary_col2:

                    st.metric(
                        "High Risk",
                        f"{high_risk_count:,}"
                    )

                with summary_col3:

                    st.metric(
                        "Medium Risk",
                        f"{medium_risk_count:,}"
                    )

                with summary_col4:

                    st.metric(
                        "High-Risk Rate",
                        f"{high_risk_rate:.2f}%"
                    )

                if error_count > 0:

                    st.warning(
                        f"⚠️ {error_count:,} transaction(s) "
                        "could not be analyzed."
                    )

                # ====================================================
                # RISK DISTRIBUTION
                # ====================================================

                st.subheader(
                    "📊 Risk Distribution"
                )

                risk_df = pd.DataFrame(
                    {
                        "Risk": [
                            "HIGH RISK",
                            "MEDIUM RISK",
                            "LOW RISK",
                        ],
                        "Transactions": [
                            high_risk_count,
                            medium_risk_count,
                            low_risk_count,
                        ],
                    }
                )

                st.bar_chart(
                    risk_df.set_index("Risk")
                )

                # ====================================================
                # MOST SUSPICIOUS TRANSACTIONS
                # ====================================================

                st.subheader(
                    "🚨 Most Suspicious Transactions"
                )

                suspicious_df = (
                    analyzed_df
                    .sort_values(
                        by="Fraud Probability",
                        ascending=False,
                    )
                    .head(10)
                )

                st.dataframe(
                    suspicious_df,
                    use_container_width=True
                )

                # ====================================================
                # DOWNLOAD REPORT
                # ====================================================

                st.subheader(
                    "📥 Download Analysis Report"
                )

                download_data = (
                    analyzed_df
                    .to_csv(
                        index=False
                    )
                    .encode("utf-8")
                )

                st.download_button(
                    label="📥 Download Fraud Analysis CSV",
                    data=download_data,
                    file_name="finguard_batch_analysis.csv",
                    mime="text/csv",
                    key="download_report_button",
                )

    except Exception as error:

        st.error(
            f"❌ Could not process the uploaded CSV: {error}"
        )

# ============================================================
# SINGLE TRANSACTION ANALYSIS
# ============================================================

st.header(
    "🔎 Analyze Real Transaction"
)

st.write(
    "Select a transaction row from the demo dataset and run "
    "the fraud detection model."
)

if len(df) > 0:

    selected_index = st.number_input(
        "Transaction Row",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
        key="transaction_row",
    )

    selected_row = df.iloc[
        int(selected_index)
    ]

    st.dataframe(
        selected_row.to_frame().T,
        use_container_width=True
    )

    if st.button(
        "🚨 Analyze Transaction",
        type="primary",
        key="single_transaction_button",
    ):

        try:

            transaction = (
                selected_row
                .to_frame()
                .T
            )

            prediction = predict_transaction(
                transaction
            )

            probability = float(
                prediction["fraud_probability"]
            )

            risk_level = str(
                prediction["risk_level"]
            )

            decision = str(
                prediction["decision"]
            )

            threshold = float(
                prediction["threshold"]
            )

            # ====================================================
            # PREDICTION RESULT
            # ====================================================

            st.subheader(
                "🎯 Prediction Result"
            )

            result_col1, result_col2, result_col3 = (
                st.columns(3)
            )

            with result_col1:

                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%"
                )

            with result_col2:

                st.metric(
                    "Risk Level",
                    risk_level
                )

            with result_col3:

                st.metric(
                    "Threshold",
                    f"{threshold:.2f}"
                )

            # ====================================================
            # RISK MESSAGE
            # ====================================================

            if risk_level == "HIGH RISK":

                st.error(
                    "🚨 HIGH RISK — Transaction should be "
                    "blocked or manually reviewed."
                )

            elif risk_level == "MEDIUM RISK":

                st.warning(
                    "⚠️ MEDIUM RISK — Transaction requires "
                    "additional review."
                )

            else:

                st.success(
                    "✅ LOW RISK — Transaction appears "
                    "legitimate."
                )

            # ====================================================
            # DECISION
            # ====================================================

            st.info(
                f"Recommended Decision: **{decision}**"
            )

            # ====================================================
            # ACTUAL DATASET LABEL
            # ====================================================

            if "Class" in selected_row.index:

                try:

                    actual
