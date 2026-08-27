import streamlit as st
import pandas as pd

from model.predictor import predict_transaction, get_feature_importance


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("demo_transactions_small.csv")


try:
    df = load_data()
except Exception as e:
    st.error("❌ Could not load demo_transactions_small.csv")
    st.error(str(e))
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "An AI-powered machine-learning system for detecting "
    "and analyzing suspicious credit-card transactions."
)


# ============================================================
# FRAUD MONITORING DASHBOARD
# ============================================================

st.header("📊 Fraud Monitoring Dashboard")

total_transactions = len(df)

if "Class" in df.columns:
    fraud_transactions = int(df["Class"].sum())
else:
    fraud_transactions = 0

legitimate_transactions = total_transactions - fraud_transactions

if total_transactions > 0:
    fraud_rate = (fraud_transactions / total_transactions) * 100
else:
    fraud_rate = 0.0


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

distribution = pd.DataFrame(
    {
        "Transaction Type": ["Fraud", "Legitimate"],
        "Count": [
            fraud_transactions,
            legitimate_transactions
        ]
    }
)

st.bar_chart(
    distribution.set_index("Transaction Type")
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

st.caption("Decision threshold: 0.30")


# ============================================================
# BATCH FRAUD DETECTION
# ============================================================

st.header("📁 Batch Fraud Detection")

st.write(
    "Upload a CSV file containing transactions and "
    "FinGuard AI will analyze them automatically."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        batch_df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Transactions")

        st.dataframe(
            batch_df.head(10),
            use_container_width=True
        )

        if st.button(
            "🔍 Analyze Uploaded Transactions",
            use_container_width=True
        ):

            with st.spinner("Analyzing transactions..."):

                probabilities = []

                for index in range(len(batch_df)):

                    transaction = batch_df.iloc[[index]].copy()

                    if "Class" in transaction.columns:
                        transaction = transaction.drop(
                            columns=["Class"]
                        )

                    try:
                        prediction = predict_transaction(
                            transaction
                        )

                        probability = float(
                            prediction["fraud_probability"]
                        )

                    except Exception:
                        probability = 0.0

                    probabilities.append(probability)

            results = batch_df.copy()

            results["Fraud Probability"] = probabilities

            results["Risk Level"] = results[
                "Fraud Probability"
            ].apply(
                lambda x:
                    "🔴 HIGH RISK"
                    if x >= 0.30
                    else (
                        "🟠 MEDIUM RISK"
                        if x >= 0.10
                        else "🟢 LOW RISK"
                    )
            )

            results["Recommended Action"] = results[
                "Fraud Probability"
            ].apply(
                lambda x:
                    "🚫 BLOCK"
                    if x >= 0.30
                    else (
                        "🔎 MANUAL REVIEW"
                        if x >= 0.10
                        else "✅ APPROVE"
                    )
            )

            st.success(
                f"✅ Analyzed {len(results):,} transactions."
            )

            st.dataframe(
                results,
                use_container_width=True
            )

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_data,
                file_name="finguard_fraud_results.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as e:
        st.error("❌ Batch analysis failed.")
        st.exception(e)


# ============================================================
# ANALYZE REAL TRANSACTION
# ============================================================

st.header("🔎 Analyze Real Transaction")

st.write("Select a transaction row")

if len(df) > 0:

    selected = st.number_input(
        "Transaction Row",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    row = df.iloc[[int(selected)]].copy()

    st.dataframe(
        row,
        use_container_width=True
    )

    if st.button(
        "🚨 Analyze Transaction",
        use_container_width=True
    ):

        try:

            # Keep original label for display only
            actual_label = None

            if "Class" in row.columns:
                actual_label = int(
                    row["Class"].iloc[0]
                )

                transaction = row.drop(
                    columns=["Class"]
                ).copy()

            else:
                transaction = row.copy()

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            result = predict_transaction(
                transaction
            )

            probability = float(
                result["fraud_probability"]
            )

            threshold = float(
                result.get("threshold", 0.30)
            )

            risk_level = result.get(
                "risk_level",
                "LOW RISK"
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.subheader("🎯 Prediction Result")

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

            if probability >= threshold:

                st.error(
                    "🚨 HIGH RISK — Potential Fraud"
                )

            elif probability >= 0.10:

                st.warning(
                    "⚠️ MEDIUM RISK — Manual Review Recommended"
                )

            else:

                st.success(
                    "✅ LOW RISK — Transaction Appears Legitimate"
                )


            # ------------------------------------------------
            # ACTUAL LABEL
            # ------------------------------------------------

            if actual_label is not None:

                if actual_label == 1:
                    st.info(
                        "Dataset label: 🚨 Fraud"
                    )
                else:
                    st.info(
                        "Dataset label: ✅ Legitimate"
                    )


            # ------------------------------------------------
            # FEATURE IMPORTANCE
            # ------------------------------------------------

            st.subheader("🔍 AI Explanation")

            try:

                importance = get_feature_importance(
                    transaction
                )

                if isinstance(importance, pd.DataFrame):

                    st.dataframe(
                        importance,
                        use_container_width=True
                    )

                    if (
                        "feature" in importance
