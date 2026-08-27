import streamlit as st
import pandas as pd

from model.predictor import predict_transaction, get_feature_importance


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide",
)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
@st.cache_data
def load_demo_dataset():
    return pd.read_csv("demo_transactions_small.csv")


def run_prediction(row):
    """Run the existing FinGuard model on one transaction row."""
    transaction = row.to_frame().T
    return predict_transaction(transaction)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "FinGuard AI uses machine learning to detect suspicious "
    "credit-card transactions and classify their risk level."
)


# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
try:
    df = load_demo_dataset()

except FileNotFoundError:
    st.error(
        "demo_transactions_small.csv was not found. "
        "Make sure it is in the repository root."
    )
    st.stop()

except Exception as exc:
    st.error(f"Could not load the demo dataset: {exc}")
    st.stop()


# ------------------------------------------------------------
# FRAUD MONITORING DASHBOARD
# ------------------------------------------------------------
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


m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

with m2:
    st.metric(
        "Fraud Cases",
        f"{fraud_cases:,}"
    )

with m3:
    st.metric(
        "Legitimate",
        f"{legitimate_cases:,}"
    )

with m4:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )


# ------------------------------------------------------------
# DATASET DISTRIBUTION
# ------------------------------------------------------------
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
        "Class column is not available in the demo dataset."
    )


# ------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------
st.header("⚙️ Model Performance")

p1, p2, p3 = st.columns(3)

with p1:
    st.metric(
        "Model",
        "Random Forest"
    )

with p2:
    st.metric(
        "ROC-AUC",
        "0.953"
    )

with p3:
    st.metric(
        "PR-AUC",
        "0.854"
    )

st.caption(
    "Decision threshold: 0.30"
)


# ------------------------------------------------------------
# ADVANCED BATCH FRAUD DETECTION
# ------------------------------------------------------------
st.header("📁 Advanced Batch Fraud Detection")

st.write(
    "Upload a CSV containing transactions. FinGuard AI will "
    "analyze each transaction and generate fraud probability, "
    "risk level, and recommended decision."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        batch_df = pd.read_csv(uploaded_file)

        st.success(
            f"Loaded {len(batch_df):,} transactions."
        )

        st.subheader("📋 Uploaded Data")

        st.dataframe(
            batch_df.head(10),
            use_container_width=True
        )

        if len(batch_df) == 0:

            st.warning(
                "The uploaded CSV is empty."
            )

        else:

            if st.button(
                "🚨 Analyze All Transactions",
                type="primary",
                key="batch_button"
            ):

                predictions = []
                total_rows = len(batch_df)

                progress = st.progress(0)
                status = st.empty()

                for index, row in batch_df.iterrows():

                    try:

                        prediction = run_prediction(row)

                        probability = safe_float(
                            prediction.get(
                                "fraud_probability",
                                0.0
                            )
                        )

                        risk = str(
                            prediction.get(
                                "risk_level",
                                "UNKNOWN"
                            )
                        )

                        decision = str(
                            prediction.get(
                                "decision",
                                "UNKNOWN"
                            )
                        )

                        error_message = ""

                    except Exception as exc:

                        probability = 0.0
                        risk = "ERROR"
                        decision = "UNABLE TO ANALYZE"
                        error_message = str(exc)

                    predictions.append(
                        {
                            "Fraud Probability": probability,
                            "Risk Level": risk,
                            "Decision": decision,
                            "Analysis Error": error_message,
                        }
                    )

                    progress.progress(
                        (index + 1) / total_rows
                    )

                    status.write(
                        f"Analyzing transaction "
                        f"{index + 1:,} of "
                        f"{total_rows:,}..."
                    )

                progress.empty()
                status.empty()

                prediction_df = pd.DataFrame(
                    predictions
                )

                analyzed_df = batch_df.copy()

                analyzed_df[
                    "Fraud Probability"
                ] = prediction_df[
                    "Fraud Probability"
                ].values

                analyzed_df[
                    "Risk Level"
                ] = prediction_df[
                    "Risk Level"
                ].values

                analyzed_df[
                    "Decision"
                ] = prediction_df[
                    "Decision"
                ].values

                analyzed_df[
                    "Analysis Error"
                ] = prediction_df[
                    "Analysis Error"
                ].values

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
                        analyzed_df["Risk Level"]
                        == "ERROR"
                    ).sum()
                )

                if analyzed_count > 0:

                    detected_rate = (
                        high_risk_count
                        / analyzed_count
                    ) * 100

                else:

                    detected_rate = 0.0


                s1, s2, s3, s4 = st.columns(4)

                with s1:
                    st.metric(
                        "Analyzed",
                        f"{analyzed_count:,}"
                    )

                with s2:
                    st.metric(
                        "High Risk",
                        f"{high_risk_count:,}"
                    )

                with s3:
                    st.metric(
                        "Medium Risk",
                        f"{medium_risk_count:,}"
                    )

                with s4:
                    st.metric(
                        "High-Risk Rate",
                        f"{detected_rate:.2f}%"
                    )


                if error_count > 0:

                    st.warning(
                        f"⚠️ {error_count:,} "
                        "transaction(s) could not "
                        "be analyzed."
                    )


                # ------------------------------------------------
                # RISK DISTRIBUTION
                # ------------------------------------------------
                st.subheader(
                    "📌 Risk Distribution"
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


                # ------------------------------------------------
                # MOST SUSPICIOUS TRANSACTIONS
                # ------------------------------------------------
                st.subheader(
                    "🚨 Most Suspicious Transactions"
                )

                suspicious_df = (
                    analyzed_df
                    .sort_values(
                        by="Fraud Probability",
                        ascending=False
                    )
                    .head(10)
                )

                display_suspicious = (
                    suspicious_df.copy()
                )

                display_suspicious[
                    "Fraud Probability"
                ] = (
                    display_suspicious[
                        "Fraud Probability"
                    ] * 100
                ).round(2)

                st.dataframe(
                    display_suspicious,
                    use_container_width=True
                )


                # ------------------------------------------------
                # DOWNLOAD REPORT
                # ------------------------------------------------
                st.subheader(
                    "📥 Download Analysis Report"
                )

                download_data = (
                    analyzed_df
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(
                    label="📄 Download Fraud Analysis CSV",
                    data=download_data,
                    file_name=(
                        "finguard_batch_analysis.csv"
                    ),
                    mime="text/csv",
                    key="download_report"
                )


    except Exception as exc:

        st.error(
            f"❌ Batch processing failed: {exc}"
        )


# ------------------------------------------------------------
# SINGLE TRANSACTION ANALYSIS
# ------------------------------------------------------------
st.header("🔎 Analyze Real Transaction")

st.write(
    "Select a transaction from the demo dataset "
    "and run the fraud detection model."
)


if len(df) > 0:

    selected_index = st.number_input(
        "Transaction Row",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
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
        key="single_button"
    ):

        try:

            prediction = run_prediction(
                selected_row
            )

            probability = safe_float(
                prediction.get(
                    "fraud_probability",
                    0.0
                )
            )

            risk_level = str(
                prediction.get(
                    "risk_level",
                    "UNKNOWN"
                )
            )

            decision = str(
                prediction.get(
                    "decision",
                    "UNKNOWN"
                )
            )

            threshold = safe_float(
                prediction.get(
                    "threshold",
                    0.30
                ),
                0.30
            )


            # ------------------------------------------------
            # PREDICTION RESULT
            # ------------------------------------------------
            st.subheader(
                "🎯 Prediction Result"
            )

            r1, r2, r3 = st.columns(3)

            with r1:

                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%"
                )

            with r2:

                st.metric(
                    "Risk Level",
                    risk_level
                )

            with r3:

                st.metric(
                    "Threshold",
                    f"{threshold:.2f}"
                )


            # ------------------------------------------------
            # RISK MESSAGE
            # ------------------------------------------------
            if risk_level == "HIGH RISK":

                st.error(
                    "🚨 HIGH RISK — Transaction should "
                    "be blocked or manually reviewed."
                )

            elif risk_level == "MEDIUM RISK":

                st.warning(
                    "⚠️ MEDIUM RISK — Transaction "
                    "requires review."
                )

            else:

                st.success(
                    "✅ LOW RISK — Transaction "
                    "appears legitimate."
                )


            # ------------------------------------------------
            # DECISION
            # ------------------------------------------------
            st.info(
                f"Recommended Decision: **{decision}**"
            )


            # ------------------------------------------------
            # ACTUAL DATASET LABEL
            # ------------------------------------------------
            if "Class" in selected_row.index:

                actual_value = safe_float(
                    selected_row["Class"]
                )

                if actual_value == 1:

                    actual_label = (
                        "🚨 Fraudulent Transaction"
                    )

                else:

                    actual_label = (
                        "✅ Legitimate Transaction"
                    )

                st.info(
                    f"Dataset Label: **{actual_label}**"
                )


            # ------------------------------------------------
            # FEATURE IMPORTANCE
            # ------------------------------------------------
            st.subheader(
                "🔍 AI Explanation"
            )

            try:

                importance_df = (
                    get_feature_importance(
                        selected_row.to_frame().T
                    )
                )

                if isinstance(
                    importance_df,
                    pd.DataFrame
                ):

                    st.dataframe(
                        importance_df,
                        use_container_width=True
                    )

                else:

                   
