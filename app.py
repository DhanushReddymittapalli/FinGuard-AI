import streamlit as st
import pandas as pd
from pathlib import Path

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
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEMO_DATA = BASE_DIR / "demo_transactions_small.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    if not DEMO_DATA.exists():
        raise FileNotFoundError(
            "demo_transactions_small.csv was not found."
        )

    return pd.read_csv(DEMO_DATA)


try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
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
# DASHBOARD
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

    distribution = pd.DataFrame(
        {
            "Type": [
                "Fraud",
                "Legitimate"
            ],
            "Transactions": [
                fraud_cases,
                legitimate_cases
            ]
        }
    )

    st.bar_chart(
        distribution.set_index("Type")
    )

else:

    st.info(
        "The dataset does not contain a Class column."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("⚙️ Model Performance")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Model",
        "Random Forest"
    )

with m2:
    st.metric(
        "ROC-AUC",
        "0.953"
    )

with m3:
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
    "FinGuard AI will analyze every transaction "
    "and generate fraud probability, risk level, "
    "and recommended decision."
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

            results = []

            progress = st.progress(0)

            status = st.empty()

            total_rows = len(batch_df)

            for i in range(total_rows):

                row = batch_df.iloc[i]

                transaction = (
                    row.to_frame().T
                )

                try:

                    result = predict_transaction(
                        transaction
                    )

                    probability = float(
                        result["fraud_probability"]
                    )

                    risk = result["risk_level"]

                    decision = result["decision"]

                    error = ""

                except Exception as e:

                    probability = 0.0

                    risk = "ERROR"

                    decision = (
                        "UNABLE TO ANALYZE"
                    )

                    error = str(e)

                results.append(
                    {
                        "Fraud Probability": probability,
                        "Risk Level": risk,
                        "Decision": decision,
                        "Analysis Error": error
                    }
                )

                progress.progress(
                    (i + 1) / total_rows
                )

                status.write(
                    f"Analyzing transaction "
                    f"{i + 1:,} of {total_rows:,}"
                )

            progress.empty()
            status.empty()

            result_df = pd.DataFrame(
                results
            )

            analyzed_df = batch_df.copy()

            analyzed_df[
                "Fraud Probability"
            ] = result_df[
                "Fraud Probability"
            ]

            analyzed_df[
                "Risk Level"
            ] = result_df[
                "Risk Level"
            ]

            analyzed_df[
                "Decision"
            ] = result_df[
                "Decision"
            ]

            analyzed_df[
                "Analysis Error"
            ] = result_df[
                "Analysis Error"
            ]

            st.success(
                "✅ Batch analysis completed."
            )

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.subheader(
                "📊 Batch Analysis Summary"
            )

            analyzed_count = len(
                analyzed_df
            )

            high_risk = int(
                (
                    analyzed_df["Risk Level"]
                    == "HIGH RISK"
                ).sum()
            )

            medium_risk = int(
                (
                    analyzed_df["Risk Level"]
                    == "MEDIUM RISK"
                ).sum()
            )

            low_risk = int(
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
                    high_risk
                    / analyzed_count
                ) * 100

            else:

                high_risk_rate = 0.0


            s1, s2, s3, s4 = st.columns(4)

            with s1:
                st.metric(
                    "Analyzed",
                    f"{analyzed_count:,}"
                )

            with s2:
                st.metric(
                    "High Risk",
                    f"{high_risk:,}"
                )

            with s3:
                st.metric(
                    "Medium Risk",
                    f"{medium_risk:,}"
                )

            with s4:
                st.metric(
                    "High-Risk Rate",
                    f"{high_risk_rate:.2f}%"
                )


            # ------------------------------------------------
            # RISK DISTRIBUTION
            # ------------------------------------------------

            st.subheader(
                "📊 Risk Distribution"
            )

            risk_df = pd.DataFrame(
                {
                    "Risk": [
                        "HIGH RISK",
                        "MEDIUM RISK",
                        "LOW RISK"
                    ],
                    "Transactions": [
                        high_risk,
                        medium_risk,
                        low_risk
                    ]
                }
            )

            st.bar_chart(
                risk_df.set_index("Risk")
            )


            # ------------------------------------------------
            # MOST SUSPICIOUS
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

            st.dataframe(
                suspicious_df,
                use_container_width=True
            )


            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            st.subheader(
                "📥 Download Analysis Report"
            )

            csv_data = (
                analyzed_df
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                label="📥 Download Fraud Analysis CSV",
                data=csv_data,
                file_name="finguard_batch_analysis.csv",
                mime="text/csv",
                key="download_report"
            )

            if error_count > 0:

                st.warning(
                    f"⚠️ {error_count:,} transaction(s) "
                    "could not be analyzed."
                )

    except Exception as e:

        st.error(
            f"Batch processing failed: {e}"
        )


# ============================================================
# SINGLE TRANSACTION
# ============================================================

st.header("🔍 Analyze Real Transaction")

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

            risk_level = prediction[
                "risk_level"
            ]

            decision = prediction[
                "decision"
            ]

            threshold = float(
                prediction["threshold"]
            )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

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


            # ------------------------------------------------
            # RISK MESSAGE
            # ------------------------------------------------

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


            # ------------------------------------------------
            # DECISION
            # ------------------------------------------------

            st.info(
                f"Recommended Decision: **{decision}**"
            )


            # ------------------------------------------------
            # ACTUAL LABEL
            # ------------------------------------------------

            if "Class" in selected_row.index:

                actual_class = int(
                    selected_row["Class"]
                )

                if actual_class == 1:

                    st.warning(
                        "📌 Dataset Label: "
                        "**Fraudulent Transaction**"
                    )

                else:

                    st.success(
                        "📌 Dataset Label: "
                        "**Legitimate Transaction**"
                    )


            # ------------------------------------------------
            # AI EXPLANATION
            # ------------------------------------------------

            st.subheader(
                "🔎 AI Explanation"
            )

            try:

                explanation = (
                    get_feature_importance(
                        transaction
                    )
                )

                if isinstance(
                    explanation,
                    pd.DataFrame
                ):

                    st.dataframe(
                        explanation,
                        use_container_width=True
                    )

                elif isinstance(
                    explanation,
                    dict
                ):

                    explanation_df = pd.DataFrame(
                        {
                            "Feature": list(
                                explanation.keys()
                            ),
                            "Importance": list(
                                explanation.values()
                            )
                        }
                    )

                    explanation_df = (
                        explanation_df
                        .sort_values(
                            by="Importance",
                            ascending=False
                        )
                    )

                    st.dataframe(
                        explanation_df,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "Feature importance "
                        "is not available."
                    )

                st.caption(
                    "Explanation is based on "
                    "Random Forest feature importance."
                )

            except Exception as e:

                st.warning(
                    f"Explanation unavailable: {e}"
                )


        except Exception as e:

            st.error(
                f"Transaction analysis failed: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FinGuard AI | Random Forest Fraud Detection | "
    "Credit Card Fraud Detection Dataset"
)
