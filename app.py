import streamlit as st
import pandas as pd

from model.predictor import predict_transaction, get_feature_importance


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("demo_transactions_small.csv")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "Machine-learning system for detecting suspicious "
    "credit-card transactions."
)


# --------------------------------------------------
# FRAUD MONITORING DASHBOARD
# --------------------------------------------------

st.write("### 📊 Fraud Monitoring Dashboard")

total_transactions = len(df)

fraud_transactions = int(df["Class"].sum())

legitimate_transactions = (
    total_transactions - fraud_transactions
)

fraud_rate = (
    fraud_transactions / total_transactions
) * 100


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


# --------------------------------------------------
# DATASET DISTRIBUTION
# --------------------------------------------------

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


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.write("### ⚙️ Model Performance")

m1, m2, m3, m4 = st.columns(4)

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

with m4:
    st.metric(
        "Threshold",
        "0.30"
    )


# --------------------------------------------------
# TRANSACTION ANALYSIS
# --------------------------------------------------

st.write("### 🔍 Analyze Real Transaction")

row = st.number_input(
    "Select Transaction Row",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)


# Get selected transaction

selected = df.iloc[[int(row)]].copy()

actual_label = int(
    selected["Class"].iloc[0]
)

transaction = selected.drop(
    columns=["Class"]
)


# Show transaction

st.dataframe(
    transaction,
    use_container_width=True
)


# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button(
    "🚨 Analyze Transaction",
    use_container_width=True
):

    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------

    result = predict_transaction(transaction)

    probability = result["fraud_probability"]

    risk = result["risk_level"]


    # --------------------------------------------------
    # FRAUD ANALYSIS
    # --------------------------------------------------

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


    # --------------------------------------------------
    # WHY WAS IT FLAGGED?
    # --------------------------------------------------

    st.write(
        "### 🔎 Why was this transaction flagged?"
    )

    importance = get_feature_importance(
        transaction
    )

    if not importance.empty:

        st.dataframe(
            importance.head(5),
            use_container_width=True
        )

    else:

        st.info(
            "Feature importance information is "
            "not available for this model."
        )


    # --------------------------------------------------
    # RISK EXPLANATION
    # --------------------------------------------------

    if risk.startswith("HIGH"):

        st.error(
            "🚨 HIGH FRAUD RISK\n\n"
            "The model estimates a high probability "
            "of fraudulent activity."
        )

    elif risk.startswith("MEDIUM"):

        st.warning(
            "⚠️ MEDIUM FRAUD RISK\n\n"
            "Additional review is recommended."
        )

    else:

        st.success(
            "✅ LOW FRAUD RISK\n\n"
            "The model estimates a low probability "
            "of fraud."
        )


    # --------------------------------------------------
    # RECOMMENDED ACTION
    # --------------------------------------------------

    st.write("### 🛡️ Recommended Action")


    if risk.startswith("HIGH"):

        st.error(
            "🚫 BLOCK TRANSACTION\n\n"
            "High fraud risk detected. "
            "Block the transaction and investigate "
            "immediately."
        )

    elif risk.startswith("MEDIUM"):

        st.warning(
            "🔎 MANUAL REVIEW REQUIRED\n\n"
            "Suspicious activity detected. "
            "Additional verification is recommended "
            "before approval."
        )

    else:

        st.success(
            "✅ APPROVE TRANSACTION\n\n"
            "Low fraud risk detected. "
            "The transaction can proceed."
        )


    # --------------------------------------------------
    # RISK SCORE
    # --------------------------------------------------

    st.write("### 🎯 Fraud Risk Score")

    risk_score = round(
        probability * 100
    )

    st.progress(
        min(risk_score, 100)
    )

    st.metric(
        "Risk Score",
        f"{risk_score}/100"
    )


    if risk_score >= 30:

        st.write(
            f"🔴 **Elevated risk:** "
            f"{risk_score}/100"
        )

    else:

        st.write(
            f"🟢 **Low risk:** "
            f"{risk_score}/100"
        )


    # --------------------------------------------------
    # ACTUAL DATASET
