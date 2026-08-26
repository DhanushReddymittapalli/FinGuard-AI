import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# Load model and threshold
model = joblib.load("finguard_model.pkl")
threshold = joblib.load("finguard_threshold.pkl")

# Load demo transactions
demo_df = pd.read_csv("demo_transactions.csv")

st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "FinGuard AI uses a trained Random Forest model "
    "to estimate the probability that a transaction is fraudulent."
)

st.info(
    "Model trained on the ULB Credit Card Fraud Detection benchmark dataset."
)

# --------------------------------------------------
# DEMO TRANSACTION
# --------------------------------------------------

st.write("### 🧪 Test a Real Dataset Transaction")

demo_index = st.number_input(
    "Select Demo Transaction",
    min_value=0,
    max_value=len(demo_df) - 1,
    value=0,
    step=1
)

demo_transaction = demo_df.drop(
    columns=["Class"]
).iloc[[demo_index]]

st.write("Selected transaction:")
st.dataframe(
    demo_transaction,
    use_container_width=True
)

# --------------------------------------------------
# ANALYZE TRANSACTION
# --------------------------------------------------

if st.button(
    "🚨 Analyze Transaction",
    use_container_width=True
):

    transaction = demo_transaction.copy()

    probability = model.predict_proba(
        transaction
    )[0][1]

    if probability >= threshold:
        risk_level = "HIGH RISK 🚨"
    elif probability >= 0.10:
        risk_level = "MEDIUM RISK ⚠️"
    else:
        risk_level = "LOW RISK ✅"

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

    if probability >= threshold:
        st.error(
            "🚨 HIGH FRAUD RISK\n\n"
            "This transaction has a high estimated "
            "probability of fraudulent activity."
        )

    elif probability >= 0.10:
        st.warning(
            "⚠️ MEDIUM FRAUD RISK\n\n"
            "This transaction requires additional review."
        )

    else:
        st.success(
            "✅ LOW FRAUD RISK\n\n"
            "This transaction has a low estimated probability of fraud."
        )

    # Actual class, only for demonstration
    if "Class" in demo_df.columns:

        actual_class = demo_df.iloc[demo_index]["Class"]

        if actual_class == 1:
            st.write("### 🎯 Actual Dataset Label")
            st.error("Actual label: FRAUD")

        else:
            st.write("### 🎯 Actual Dataset Label")
            st.success("Actual label: LEGITIMATE")

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
