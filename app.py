import streamlit as st
import pandas as pd
from model import predictor


st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("demo_transactions_small.csv")


# ==============================
# HEADER
# ==============================

st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "AI-powered machine-learning system for detecting "
    "and analyzing suspicious credit-card transactions."
)


# ==============================
# DASHBOARD
# ==============================

st.write("### 📊 Fraud Monitoring Dashboard")

total_transactions = len(df)

if "Class" in df.columns:
    fraud_transactions = int(df["Class"].sum())
else:
    fraud_transactions = 0

legitimate_transactions = total_transactions - fraud_transactions

if total_transactions > 0:
    fraud_rate = (
        fraud_transactions / total_transactions
    ) * 100
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


# ==============================
# DATASET DISTRIBUTION
# ==============================

st.write("### 📈 Dataset Distribution")

chart_data = pd.DataFrame(
    {
        "Transaction Type": [
            "Fraud",
            "Legitimate"
        ],
        "Count": [
            fraud_transactions,
            legitimate_transactions
        ]
    }
)

st.bar_chart(
    chart_data.set_index("Transaction Type")
)


# ==============================
# MODEL PERFORMANCE
# ==============================

st.write("### ⚙️ Model Performance")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Model", "Random Forest")

with m2:
    st.metric("ROC-AUC", "0.953")

with m3:
    st.metric("PR-AUC", "0.854")

with m4:
    st.metric("Threshold", "0.30")


# ==============================
# BATCH FRAUD DETECTION
# ==============================

st.write("### 📁 Batch Fraud Detection")

st.write(
    "Upload a CSV file containing transactions "
    "and FinGuard AI will analyze them automatically."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    batch_df = pd.read_csv(uploaded_file)

    st.write("### 📋 Uploaded Transactions")

    st.dataframe(
        batch_df.head(10),
        use_container_width=True
    )

    prediction_data = batch_df.copy()

    if "Class" in prediction_data.columns:
        prediction_data = prediction_data.drop(
            columns=["Class"]
        )

    if st.button(
        "🚨 Analyze Uploaded Transactions",
        use_container_width=True
    ):

        try:

            probabilities = (
                predictor.predict_transaction_batch(
                    prediction_data
                )
            )

            results = batch_df.copy()

            results["Fraud Probability"] = probabilities

            results["Risk Level"] = results[
                "Fraud Probability"
            ].apply(
                lambda x:
                    "HIGH RISK 🚨"
                    if x >= 0.30
                    else (
                        "MEDIUM RISK ⚠️"
                        if x >= 0.10
                        else "LOW RISK ✅"
                    )
            )

            results["Recommended Action"] = results[
                "Fraud Probability"
            ].apply(
                lambda x:
                    "BLOCK"
                    if x >= 0.30
                    else (
                        "MANUAL REVIEW"
                        if x >= 0.10
                        else "APPROVE"
                    )
            )

            st.success(
                f"✅ Analyzed {len(results):,} transactions."
            )

            st.dataframe(
                results,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "Batch prediction failed."
            )

            st.exception(e)


# ==============================
# SINGLE TRANSACTION
# ==============================

st.write("### 🔍 Analyze Real Transaction")

row = st.number_input(
    "Select Transaction Row",
    min_value=0,
    max_value=max(len(df) - 1, 0),
    value=0,
    step=1
)

selected = df.iloc[[int(row)]].copy()

if "Class" in selected.columns:

    actual_label = int(
        selected["Class"].iloc[0]
    )

    transaction = selected.drop(
        columns=["Class"]
    )

else:

    actual_label = None
    transaction = selected.copy()


st.dataframe(
    transaction,
    use_container_width=True
)


# ==============================
# ANALYZE
# ==============================

if st.button(
    "🚨 Analyze Transaction",
    use_container_width=True
):

    try:

        result = predictor.predict_transaction(
            transaction
        )

        probability = float(
            result["fraud_probability"]
        )

        risk = result["risk_level"]

        threshold = float(
            result.get(
                "threshold",
                0.30
            )
        )


        # ==============================
        # FRAUD ANALYSIS
        # ==============================

        st.write("### 📊 Fraud Analysis")

        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

        with r2:
            st.metric(
                "Risk Level",
                risk
            )


        # ==============================
        # RISK EXPLANATION
        # ==============================

        if risk.startswith("HIGH"):

            st.error(
                "🚨 HIGH FRAUD RISK\n\n"
                "The model estimates a high probability "
                "of fraudulent activity."
            )

        elif risk.startswith("MEDIUM"):

            st.warning(
                "⚠️ MEDIUM FRAUD RISK\n\n"
                "Additional verification is recommended."
            )

        else:

            st.success(
                "✅ LOW FRAUD RISK\n\n"
                "The model estimates a low probability "
                "of fraud."
            )


        # ==============================
        # RECOMMENDED ACTION
        # ==============================

        st.write("### 🛡️ Recommended Action")

        if probability >= threshold:

            st.error(
                "🚫 BLOCK TRANSACTION\n\n"
                "High fraud risk detected. "
                "The transaction should be investigated."
            )

        elif probability >= 0.10:

            st.warning(
                "🔎 MANUAL REVIEW REQUIRED\n\n"
                "Suspicious behavior detected. "
                "Additional verification is recommended."
            )

        else:

            st.success(
                "✅ APPROVE TRANSACTION\n\n"
                "Low fraud risk detected. "
                "The transaction can proceed."
            )


        # ==============================
        # RISK SCORE
        # ==============================

        st.write("### 🎯 Fraud Risk Score")

        risk_score = round(
            probability * 100
        )

        st.progress(
            min(max(risk_score, 0), 100)
        )

        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )


        # ==============================
        # WHY FLAGGED
        # ==============================

        st.write(
            "### 🔎 Why was this transaction flagged?"
        )

        try:

            importance = (
                predictor.get_feature_importance(
                    transaction
                )
            )

            if (
                importance is not None
                and not importance.empty
            ):

                st.dataframe(
                    importance.head(5),
                    use_container_width=True
                )

            else:

                st.info(
                    "Feature importance is not available."
                )

        except Exception:

            st.info(
                "Feature importance could not be calculated."
            )


        # ==============================
        # ACTUAL LABEL
        # ==============================

        if actual_label is not None:

            st.write(
                "### 🎯 Actual Dataset Label"
            )

            if actual_label == 1:

                st.error("🚨 FRAUD")

            else:

                st.success("✅ LEGITIMATE")


            # ==============================
            # VERIFICATION
            # ==============================

            st.write(
                "### 🧠 Model Verification"
            )

            predicted_fraud = (
                probability >= threshold
            )

            if predicted_fraud == bool(actual_label):

                st.success(
                    "✅ Model prediction matches "
                    "the actual dataset label."
                )

            else:

                st.warning(
                    "⚠️ Model prediction differs from "
                    "the actual dataset label."
                )

    except Exception as e:

        st.error(
            "❌ Transaction analysis failed."
        )

        st.exception(e)


# ==============================
# FOOTER
# ==============================

st.write("---")

st.caption(
    "FinGuard AI | Random Forest Fraud Detection | "
    "ULB Credit Card Fraud Detection Dataset"
)
