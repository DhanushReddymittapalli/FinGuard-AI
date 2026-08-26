import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# Load model
model = joblib.load("finguard_model.pkl")

# Load optimized threshold
threshold = joblib.load("finguard_threshold.pkl")

# Load demo data
df = pd.read_csv("demo_transactions_small.csv")

# -------------------------------
# HEADER
# -------------------------------

st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "Machine-learning system for detecting suspicious "
    "credit-card transactions."
)

# -------------------------------
# DASHBOARD
# -------------------------------

st.write("### 📊 Fraud Monitoring Dashboard")

total_transactions = len(df)
fraud_transactions = int(df["Class"].sum())
legitimate_transactions = total_transactions - fraud_transactions
fraud_rate = (fraud_transactions / total_transactions) * 100

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Transactions", f"{total_transactions:,}")

with c2:
    st.metric("Fraud Cases", f"{fraud_transactions:,}")

with c3:
    st.metric("Legitimate", f"{legitimate_transactions:,}")

with c4:
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

# -------------------------------
# RISK DISTRIBUTION
# -------------------------------

st.write("### 📈 Dataset Distribution")

chart_data = pd.DataFrame({
    "Transaction Type": [
        "Legitimate",
        "Fraud"
    ],
    "Count": [
        legitimate_transactions,
        fraud_transactions
    ]
})

st.bar_chart(
    chart_data.set_index("Transaction Type")
)

# -------------------------------
# MODEL PERFORMANCE
# -------------------------------

st.write("### 🤖 Model Performance")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Model", "Random Forest")

with m2:
    st.metric("ROC-AUC", "0.953")

with m3:
    st.metric("PR-AUC", "0.854")

with m4:
    st.metric("Threshold", "0.30")

# -------------------------------
# TRANSACTION ANALYSIS
# -------------------------------

st.write("### 🔍 Analyze Real Transaction")

row = st.number_input(
    "Select Transaction Row",
    min_value=0,
    max_value=len(df) - 1,
    value=4,
    step=1
)

selected = df.iloc[[int(row)]].copy()

actual_label = int(selected["Class"].iloc[0])

transaction = selected.drop(
    columns=["Class"]
)

st.dataframe(
    transaction,
    use_container_width=True
)

if st.button(
    "🚨 Analyze Transaction",
    use_container_width=True
):

    probability = model.predict_proba(
        transaction
    )[0][1]

    if probability >= threshold:
        risk = "HIGH RISK 🚨"

    elif probability >= 0.10:
        risk = "MEDIUM RISK ⚠️"

    else:
        risk = "LOW RISK ✅"

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

    if probability >= threshold:

        st.error(
            "🚨 HIGH FRAUD RISK\n\n"
            "The model estimates a high probability "
            "of fraudulent activity."
        )

    elif probability >= 0.10:

        st.warning(
            "⚠️ MEDIUM FRAUD RISK\n\n"
            "Additional review is recommended."
        )

    else:

        st.success(
            "✅ LOW FRAUD RISK\n\n"
            "The model estimates a low probability of fraud."
        )

    st.write("### 🎯 Actual Dataset Label")

    if actual_label == 1:

        st.error("🚨 FRAUD")

    else:

        st.success("✅ LEGITIMATE")

# -------------------------------
# FOOTER
# -------------------------------

st.write("---")

st.caption(
    "FinGuard AI | Random Forest Fraud Detection | "
    "ULB Credit Card Fraud Detection Dataset"
)
