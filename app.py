import streamlit as st
import pandas as pd

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
# HEADER
# ============================================================

st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "AI-powered machine-learning system for detecting "
    "and analyzing suspicious credit-card transactions."
)


# ============================================================
# LOAD DEMO DATASET
# ============================================================

try:
    df = pd.read_csv("demo_transactions_small.csv")
except Exception as e:
    st.error(f"Could not load demo dataset: {e}")
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
    fraud_rate = (fraud_cases / total_transactions) * 100
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
# BATCH FRAUD DETECTION
# ============================================================

st.header("📁 Advanced Batch Fraud Detection")

st.write(
    "Upload a CSV containing transactions. "
    "FinGuard AI will analyze each transaction "
    "and assign a fraud probability, risk level, "
    "and recommended decision."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
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

                try:

                    transaction = row.to_frame().T

                    prediction = predict_transaction(
                        transaction
                    )

                    probability = float(
                        prediction["fraud_probability"]
                    )

                    risk = prediction["risk_level"]

                    decision = prediction["decision"]

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
                        "Error": error_message,
                    }
                )

                progress_bar.progress(
                    (index + 1) / total_rows
                )

                status_text.write(
                    f"Analyzing transaction "
                    f"{index + 1:,} / {total_rows:,}"
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
                "Error"
            ]

            st.success(
                "✅ Batch analysis completed."
            )

            # ==================================================
            # BATCH SUMMARY
            # ==================================================

            st.subheader(
                "📊 Batch Analysis Summary"
            )

            analyzed_count = len(analyzed_df)

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
                    * 100
                )
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
                    f"⚠️ {error_count:,} transaction(s) "
                    "could not be analyzed."
                )

            # ==================================================
            # RISK DISTRIBUTION
            # ==================================================

            st.subheader(
                "📈 Risk Distribution"
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

            # ==================================================
            # MOST SUSPICIOUS TRANSACTIONS
            # ==================================================

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

            st.dataframe(
                suspicious_df,
                use_container_width=True
            )

            # ==================================================
            # DOWNLOAD REPORT
            # ==================================================

            st.subheader(
                "📥 Download Analysis Report"
            )

            download_data = analyzed_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Fraud Analysis CSV",
                data=download_data,
                file_name="finguard_batch_analysis.csv",
                mime="text/csv",
                key="download_report",
            )

    except Exception as e:

        st.error(
            f"❌ Batch processing failed: {e}"
        )


# ============================================================
# SINGLE TRANSACTION ANALYSIS
# ============================================================

st.header("🔎 Analyze Real Transaction")

st.write(
    "Select a transaction row from the demo dataset "
    "and run the fraud detection model."
)


if len(df) > 0:

    selected_index = st.number_input(
        "Transaction Row",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )

    selected_row = df.iloc[
        int(selected_index)
    ]

    st.dataframe(
        selected_row.to_frame().T,
        use_container_width=True,
    )

    if st.button(
        "🚨 Analyze Transaction",
        type="primary",
        key="single_button",
    ):

        try:

            transaction = selected_row.to_frame().T

            prediction = predict_transaction(
                transaction
            )

            probability = float(
                prediction["fraud_probability"]
            )

            risk_level = prediction[
                "risk_level"
            ]

            decision = prediction[
                "decision"
            ]

            threshold = float(
                prediction["threshold"]
            )

            # ================================================
            # PREDICTION RESULT
            # ================================================

            st.subheader(
                "🎯 Prediction Result"
            )

            p1, p2, p3 = st.columns(3)

            with p1:
                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%"
                )

            with p2:
                st.metric(
                    "Risk Level",
                    risk_level
                )

            with p3:
                st.metric(
                    "Threshold",
                    f"{threshold:.2f}"
                )

            # ================================================
            # RISK MESSAGE
            # ================================================

            if risk_level == "HIGH RISK":

                st.error(
                    "🚨 HIGH RISK — "
                    "Transaction should be blocked "
                    "or manually reviewed."
                )

            elif risk_level == "MEDIUM RISK":

                st.warning(
                    "⚠️ MEDIUM RISK — "
                    "Transaction requires review."
                )

            else:

                st.success(
                    "✅ LOW RISK — "
                    "Transaction appears legitimate."
                )

            st.info(
                f"Recommended Decision: **{decision}**"
            )

            # ================================================
            # ACTUAL DATASET LABEL
            # ================================================

            if "Class" in selected_row.index:

                actual_label = selected_row[
                    "Class"
                ]

                if actual_label == 1:

                    st.error(
                        "Dataset Label: 🚨 Fraud"
                    )

                else:

                    st.info(
                        "Dataset Label: ✅ Legitimate"
                    )

            # ================================================
            # AI EXPLANATION
            # ================================================

            st.subheader(
                "🔍 AI Explanation"
            )

            try:

                importance_df = (
                    get_feature_importance(
                        transaction
                    )
                )

                st.dataframe(
                    importance_df.head(10),
                    use_container_width=True,
                )

                st.caption(
                    "The current explanation uses "
                    "Random Forest feature importance."
                )

            except Exception as e:

                st.warning(
                    f"AI explanation unavailable: {e}"
                )

        except Exception as e:

            st.error(
                f"❌ Transaction analysis failed: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FinGuard AI | Random Forest Fraud Detection | "
    "Credit Card Fraud Detection Dataset"
)
