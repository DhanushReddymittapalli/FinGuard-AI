import streamlit as st
import pandas as pd

from model.predictor import (
    predict_transaction,
    get_feature_importance,
)


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

st.caption(
    "AI-powered machine-learning system for detecting "
    "and analyzing suspicious credit-card transactions."
)


# ============================================================
# LOAD DATASET
# ============================================================

try:
    df = pd.read_csv("demo_transactions_small.csv")
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ============================================================
# DATASET DASHBOARD
# ============================================================

st.header("📊 Fraud Monitoring Dashboard")

total_transactions = len(df)

if "Class" in df.columns:
    fraud_transactions = int(df["Class"].sum())
else:
    fraud_transactions = 0

legitimate_transactions = (
    total_transactions - fraud_transactions
)

fraud_rate = (
    fraud_transactions / total_transactions * 100
    if total_transactions > 0
    else 0
)


c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

with c2:
    st.metric(
        "Fraud Cases",
        f"{fraud_transactions:,}"
    )

with c3:
    st.metric(
        "Legitimate",
        f"{legitimate_transactions:,}"
    )

with c4:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

st.header("📈 Dataset Distribution")

if "Class" in df.columns:

    distribution = pd.DataFrame({
        "Type": ["Fraud", "Legitimate"],
        "Count": [
            fraud_transactions,
            legitimate_transactions
        ]
    })

    st.bar_chart(
        distribution.set_index("Type")
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("⚙️ Model Performance")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Model", "Random Forest")

with m2:
    st.metric("ROC-AUC", "0.953")

with m3:
    st.metric("PR-AUC", "0.854")

st.caption("Decision threshold: 0.30")


# ============================================================
# ADVANCED BATCH FRAUD DETECTION
# ============================================================

st.header("📁 Advanced Batch Fraud Detection")

st.write(
    "Upload a CSV file containing transactions. "
    "FinGuard AI will analyze every transaction "
    "and assign a fraud probability and risk level."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    key="batch_upload"
)


if uploaded_file is not None:

    try:

        batch_df = pd.read_csv(uploaded_file)

        st.success(
            f"Successfully loaded {len(batch_df):,} transactions."
        )

        st.subheader("📋 Uploaded Dataset")

        st.dataframe(
            batch_df.head(10),
            use_container_width=True
        )

        if st.button(
            "🚨 Analyze All Transactions",
            type="primary"
        ):

            progress = st.progress(0)
            status = st.empty()

            results = []

            total_rows = len(batch_df)

            for i, (_, row) in enumerate(batch_df.iterrows()):

                try:

                    prediction = predict_transaction(
                        row.to_frame().T
                    )

                    probability = prediction[
                        "fraud_probability"
                    ]

                    risk = prediction[
                        "risk_level"
                    ]

                    decision = prediction[
                        "decision"
                    ]

                except Exception as e:

                    probability = 0.0
                    risk = "ERROR"
                    decision = str(e)

                results.append({
                    "Fraud Probability": probability,
                    "Risk Level": risk,
                    "Decision": decision
                })

                progress.progress(
                    (i + 1) / total_rows
                    if total_rows > 0
                    else 1.0
                )

                status.text(
                    f"Analyzing transaction "
                    f"{i + 1:,} of {total_rows:,}"
                )

            result_df = pd.DataFrame(results)

            analyzed_df = batch_df.copy()

            analyzed_df["Fraud Probability"] = (
                result_df["Fraud Probability"]
            )

            analyzed_df["Risk Level"] = (
                result_df["Risk Level"]
            )

            analyzed_df["Decision"] = (
                result_df["Decision"]
            )

            status.empty()
            progress.empty()

            # ==================================================
            # BATCH SUMMARY
            # ==================================================

            st.success(
                "✅ Batch analysis completed successfully."
            )

            st.subheader("📊 Batch Analysis Summary")

            analyzed_count = len(analyzed_df)

            high_risk = int(
                (analyzed_df["Risk Level"] == "HIGH RISK").sum()
            )

            medium_risk = int(
                (analyzed_df["Risk Level"] == "MEDIUM RISK").sum()
            )

            low_risk = int(
                (analyzed_df["Risk Level"] == "LOW RISK").sum()
            )

            error_count = int(
                (analyzed_df["Risk Level"] == "ERROR").sum()
            )

            detected_fraud_rate = (
                high_risk / analyzed_count * 100
                if analyzed_count > 0
                else 0
            )

            b1, b2, b3, b4 = st.columns(4)

            with b1:
                st.metric(
                    "Analyzed",
                    f"{analyzed_count:,}"
                )

            with b2:
                st.metric(
                    "High Risk",
                    f"{high_risk:,}"
                )

            with b3:
                st.metric(
                    "Medium Risk",
                    f"{medium_risk:,}"
                )

            with b4:
                st.metric(
                    "Detected Rate",
                    f"{detected_fraud_rate:.2f}%"
                )

            if error_count > 0:
                st.warning(
                    f"{error_count} transaction(s) "
                    "could not be analyzed."
                )

            # ==================================================
            # RISK DISTRIBUTION
            # ==================================================

            st.subheader("📈 Risk Distribution")

            risk_distribution = pd.DataFrame({
                "Risk Level": [
                    "HIGH RISK",
                    "MEDIUM RISK",
                    "LOW RISK"
                ],
                "Transactions": [
                    high_risk,
                    medium_risk,
                    low_risk
                ]
            })

            st.bar_chart(
                risk_distribution.set_index(
                    "Risk Level"
                )
            )

            # ==================================================
            # MOST SUSPICIOUS TRANSACTIONS
            # ==================================================

            st.subheader("🚨 Most Suspicious Transactions")

            suspicious = (
                analyzed_df
                .sort_values(
                    "Fraud Probability",
                    ascending=False
                )
                .head(10)
            )

            display_columns = [
                column
                for column in [
                    "Fraud Probability",
                    "Risk Level",
                    "Decision"
                ]
                if column in suspicious.columns
            ]

            st.dataframe(
                suspicious[
                    display_columns
                    + [
                        column
                        for column in suspicious.columns
                        if column not in display_columns
                    ]
                ],
                use_container_width=True
            )

            # ==================================================
            # DOWNLOAD REPORT
            # ==================================================

            st.subheader("📥 Download Fraud Analysis Report")

            csv_data = analyzed_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Analyzed CSV",
                data=csv_data,
                file_name="finguard_batch_analysis.csv",
                mime="text/csv",
            )


# ============================================================
# SINGLE TRANSACTION ANALYSIS
# ============================================================

st.header("🔎 Analyze Real Transaction")

st.write(
    "Select a transaction from the demo dataset "
    "and analyze it using the trained Random Forest model."
)


if len(df) > 0:

    row_number = st.number_input(
        "Transaction Row",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    selected_row = df.iloc[
        int(row_number)
    ]

    st.dataframe(
        selected_row.to_frame().T,
        use_container_width=True
    )

    if st.button(
        "🚨 Analyze Transaction",
        type="primary",
        key="single_analysis"
    ):

        try:

            prediction = predict_transaction(
                selected_row.to_frame().T
            )

            probability = prediction[
                "fraud_probability"
            ]

            risk_level = prediction[
                "risk_level"
            ]

            decision = prediction[
                "decision"
            ]

            threshold = prediction[
                "threshold"
            ]

            # ================================================
            # PREDICTION RESULT
            # ================================================

            st.subheader("🎯 Prediction Result")

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
            # DECISION MESSAGE
            # ================================================

            if risk_level == "HIGH RISK":

                st.error(
                    "🚨 HIGH RISK — Transaction should be blocked "
                    "or manually reviewed."
                )

            elif risk_level == "MEDIUM RISK":

                st.warning(
                    "⚠️ MEDIUM RISK — Transaction requires review."
                )

            else:

                st.success(
                    "✅ LOW RISK — Transaction appears legitimate."
                )

            # ================================================
            # DECISION
            # ================================================

            st.info(
                f"Recommended Decision: **{decision}**"
            )

            # ================================================
            # DATASET LABEL
            # ================================================

            if "Class" in selected_row.index:

                actual_label = selected_row["Class"]

                if actual_label == 1:
                    st.error(
                        "Dataset Label: 🚨 Fraud"
                    )
                else:
                    st.info(
                        "Dataset Label: ✅ Legitimate"
                    )

            # ================================================
            # FEATURE IMPORTANCE
            # ================================================

            st.subheader("🔍 AI Explanation")

            try:

                importance_df = get_feature_importance(
                    selected_row.to_frame().T
                )

                st.dataframe(
                    importance_df.head(10),
                    use_container_width=True
                )

                st.caption(
                    "Feature importance shows which features "
                    "are most influential to the trained model."
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
