import streamlit as st
import pandas as pd

from model.predictor import predict_transaction, get_feature_importance


st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_demo_dataset():
    return pd.read_csv("demo_transactions_small.csv")


def make_transaction(row):
    return row.to_frame().T


def get_probability(prediction):
    try:
        return float(prediction.get("fraud_probability", 0.0))
    except (TypeError, ValueError):
        return 0.0


def get_threshold(prediction):
    try:
        return float(prediction.get("threshold", 0.30))
    except (TypeError, ValueError):
        return 0.30


st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")
st.write(
    "FinGuard AI uses machine learning to detect suspicious "
    "credit-card transactions and classify their risk level."
)


# =========================
# LOAD DEMO DATASET
# =========================
try:
    df = load_demo_dataset()
except FileNotFoundError:
    st.error(
        "demo_transactions_small.csv was not found. "
        "Place it in the repository root."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the demo dataset: {exc}")
    st.stop()


# =========================
# DASHBOARD
# =========================
st.header("📊 Fraud Monitoring Dashboard")

total_transactions = len(df)

if "Class" in df.columns:
    class_values = pd.to_numeric(df["Class"], errors="coerce").fillna(0)
    fraud_cases = int((class_values == 1).sum())
else:
    fraud_cases = 0

legitimate_cases = total_transactions - fraud_cases

if total_transactions:
    fraud_rate = fraud_cases / total_transactions * 100
else:
    fraud_rate = 0.0

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Transactions", f"{total_transactions:,}")

with c2:
    st.metric("Fraud Cases", f"{fraud_cases:,}")

with c3:
    st.metric("Legitimate", f"{legitimate_cases:,}")

with c4:
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")


# =========================
# DATASET DISTRIBUTION
# =========================
st.header("📈 Dataset Distribution")

if "Class" in df.columns:
    distribution = pd.DataFrame(
        {
            "Type": ["Fraud", "Legitimate"],
            "Transactions": [fraud_cases, legitimate_cases],
        }
    )
    st.bar_chart(distribution.set_index("Type"))
else:
    st.info("Class column is not available in the dataset.")


# =========================
# MODEL PERFORMANCE
# =========================
st.header("⚙️ Model Performance")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Model", "Random Forest")

with m2:
    st.metric("ROC-AUC", "0.953")

with m3:
    st.metric("PR-AUC", "0.854")

st.caption("Decision threshold: 0.30")


# =========================
# BATCH ANALYSIS
# =========================
st.header("📁 Advanced Batch Fraud Detection")

st.write(
    "Upload a CSV containing transactions. FinGuard AI will "
    "analyze each transaction and assign fraud probability, "
    "risk level, and recommended decision."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        if batch_df.empty:
            st.warning("The uploaded CSV is empty.")
        else:
            st.success(f"Loaded {len(batch_df):,} transactions.")

            st.subheader("📋 Uploaded Data")
            st.dataframe(
                batch_df.head(10),
                use_container_width=True,
            )

            analyze_batch = st.button(
                "🚨 Analyze All Transactions",
                type="primary",
                key="batch_button",
            )

            if analyze_batch:
                results = []
                total_rows = len(batch_df)

                progress = st.progress(0)
                status = st.empty()

                for position, (_, row) in enumerate(batch_df.iterrows(), start=1):
                    try:
                        prediction = predict_transaction(make_transaction(row))

                        probability = get_probability(prediction)
                        risk = str(prediction.get("risk_level", "UNKNOWN"))
                        decision = str(prediction.get("decision", "UNKNOWN"))
                        error_message = ""

                    except Exception as exc:
                        probability = 0.0
                        risk = "ERROR"
                        decision = "UNABLE TO ANALYZE"
                        error_message = str(exc)

                    results.append(
                        {
                            "Fraud Probability": probability,
                            "Risk Level": risk,
                            "Decision": decision,
                            "Analysis Error": error_message,
                        }
                    )

                    progress.progress(position / total_rows)
                    status.write(
                        f"Analyzing transaction {position:,} of {total_rows:,}..."
                    )

                progress.empty()
                status.empty()

                result_df = pd.DataFrame(results)
                analyzed_df = batch_df.copy()

                for column in result_df.columns:
                    analyzed_df[column] = result_df[column].values

                st.success("✅ Batch analysis completed.")

                # Batch summary
                st.subheader("📊 Batch Analysis Summary")

                analyzed_count = len(analyzed_df)
                high_risk_count = int(
                    (analyzed_df["Risk Level"] == "HIGH RISK").sum()
                )
                medium_risk_count = int(
                    (analyzed_df["Risk Level"] == "MEDIUM RISK").sum()
                )
                low_risk_count = int(
                    (analyzed_df["Risk Level"] == "LOW RISK").sum()
                )
                error_count = int(
                    (analyzed_df["Risk Level"] == "ERROR").sum()
                )

                high_risk_rate = (
                    high_risk_count / analyzed_count * 100
                    if analyzed_count
                    else 0.0
                )

                s1, s2, s3, s4 = st.columns(4)

                with s1:
                    st.metric("Analyzed", f"{analyzed_count:,}")

                with s2:
                    st.metric("High Risk", f"{high_risk_count:,}")

                with s3:
                    st.metric("Medium Risk", f"{medium_risk_count:,}")

                with s4:
                    st.metric("High-Risk Rate", f"{high_risk_rate:.2f}%")

                if error_count:
                    st.warning(
                        f"⚠️ {error_count:,} transaction(s) could not be analyzed."
                    )

                # Risk distribution
                st.subheader("📌 Risk Distribution")

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

                st.bar_chart(risk_df.set_index("Risk"))

                # Most suspicious
                st.subheader("🚨 Most Suspicious Transactions")

                suspicious_df = (
                    analyzed_df
                    .sort_values(
                        "Fraud Probability",
                        ascending=False,
                    )
                    .head(10)
                    .copy()
                )

                display_df = suspicious_df.copy()
                display_df["Fraud Probability"] = (
                    display_df["Fraud Probability"] * 100
                ).round(2)

                st.dataframe(
                    display_df,
                    use_container_width=True,
                )

                # Download
                st.subheader("📥 Download Analysis Report")

                csv_data = analyzed_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "📄 Download Fraud Analysis CSV",
                    data=csv_data,
                    file_name="finguard_batch_analysis.csv",
                    mime="text/csv",
                    key="download_report",
                )

    except Exception as exc:
        st.error(f"❌ Batch processing failed: {exc}")


# =========================
# SINGLE TRANSACTION
# =========================
st.header("🔎 Analyze Real Transaction")

st.write(
    "Select a transaction from the demo dataset and run the fraud detection model."
)

if not df.empty:
    selected_index = st.number_input(
        "Transaction Row",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )

    selected_row = df.iloc[int(selected_index)]

    st.dataframe(
        selected_row.to_frame().T,
        use_container_width=True,
    )

    analyze_single = st.button(
        "🚨 Analyze Transaction",
        type="primary",
        key="single_button",
    )

    if analyze_single:
        try:
            transaction = make_transaction(selected_row)
            prediction = predict_transaction(transaction)

            probability = get_probability(prediction)
            risk_level = str(prediction.get("risk_level", "UNKNOWN"))
            decision = str(prediction.get("decision", "UNKNOWN"))
            threshold = get_threshold(prediction)

            st.subheader("🎯 Prediction Result")

            p1, p2, p3 = st.columns(3)

            with p1:
                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%",
                )

            with p2:
                st.metric("Risk Level", risk_level)

            with p3:
                st.metric("Threshold", f"{threshold:.2f}")

            if risk_level == "HIGH RISK":
                st.error(
                    "🚨 HIGH RISK — Transaction should be blocked or manually reviewed."
                )
            elif risk_level == "MEDIUM RISK":
                st.warning(
                    "⚠️ MEDIUM RISK — Transaction requires review."
                )
            elif risk_level == "LOW RISK":
                st.success(
                    "✅ LOW RISK — Transaction appears legitimate."
                )
            else:
                st.info("Risk level returned by the model: " + risk_level)

            st.info(f"Recommended Decision: **{decision}**")

            if "Class" in selected_row.index:
                try:
                    actual = int(float(selected_row["Class"]))
                    if actual == 1:
                        label = "🚨 Fraudulent Transaction"
                    else:
                        label = "✅ Legitimate Transaction"
                    st.info(f"Dataset Label: **{label}**")
                except (TypeError, ValueError):
                    pass

            st.subheader("🔍 AI Explanation")

            try:
                importance = get_feature_importance(transaction)

                if isinstance(importance, pd.DataFrame):
                    st.dataframe(
                        importance,
                        use_container_width=True,
                    )
                else:
                    st.write(importance)

                st.caption(
                    "Explanation is based on Random Forest feature importance."
                )

            except Exception as exc:
                st.warning(
                    f"Feature importance is unavailable: {exc}"
                )

        except Exception as exc:
            st.error(
                f"❌ Transaction analysis failed: {exc}"
            )
else:
    st.warning("The demo dataset contains no transactions.")


# =========================
# FOOTER
# =========================
st.divider()

st.caption(
    "FinGuard AI | Random Forest Fraud Detection | "
    "Credit Card Fraud Detection Dataset"
)
