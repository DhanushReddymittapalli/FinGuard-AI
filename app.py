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

# Load demo transactions
demo_df = pd.read_csv("demo_transactions.csv")

st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "FinGuard AI uses a trained Random Forest machine-learning "
    "model to estimate the probability that a transaction is fraudulent."
)

st.info(
    "Model trained on the ULB Credit Card Fraud Detection benchmark dataset."
)

# --------------------------------------------------
# TRANSACTION SELECTION
# --------------------------------------------------

st.write("### 🧪 Test a Real Dataset Transaction")

demo_position = st.number_input(
    "Select Transaction Row",
    min_value=0,
    max_value=len(demo_df) - 1,
    value=0,
    step=1
)

# Get transaction by POSITION, not original index
selected_transaction = demo_df.iloc[[int(demo_position)]].copy()

# Actual label
actual_label = int(selected_transaction["Class"].iloc[0])

# Remove target column before prediction
transaction = selected_transaction.drop(
    columns=["Class"]
)

st.write("### 📋 Selected Transaction")

st.dataframe(
    transaction,
    use_container_width=True
)

# --------------------------------------------------
# ANALYZE
# --------------------------------------------------

if st.button(
    "🚨 Analyze Transaction",
    use_container_width=True
):

    probability = model.predict_proba(
        transaction
    )[0][1]

    # Risk classification
    if probability >= threshold:

        risk_level = "HIGH RISK 🚨"

    elif probability >= 0.10:

        risk_level = "MEDIUM RISK ⚠️"

    else:

        risk_level = "LOW RISK ✅"

    # Results
    st.write("### 📊 Fraud Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

    with col2:

        st.metric(
            "Risk Level",
            risk_level
        )

    # Risk message
    if probability >= threshold:

        st.error(
            "🚨 HIGH FRAUD RISK\n\n"
            "The model estimates a high probability "
            "of fraudulent activity."
        )

    elif probability >= 0.10:

        st.warning(
            "⚠️ MEDIUM FRAUD RISK\n\n"
            "The transaction should receive additional review."
        )

    else:

        st.success(
            "✅ LOW FRAUD RISK\n\n"
            "The model estimates a low probability of fraud."
        )

    # Actual dataset result
    st.write("### 🎯 Actual Dataset Label")

    if actual_label == 1:

        st.error(
            "🚨 FRAUD — Actual dataset label is FRAUD."
        )

    else:

        st.success(
            "✅ LEGITIMATE — Actual dataset label is LEGITIMATE."
        )

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.write("---")

st.write("### 🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Model",
        "Random Forest"
    )

with col2:

    st.metric(
        "Decision Threshold",
        f"{threshold:.2f}"
    )

with col3:

    st.metric(
        "Features",
        "30"
    )

st.caption(
    "FinGuard AI | Machine Learning Fraud Detection System"
)
