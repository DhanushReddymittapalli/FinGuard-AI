import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load("finguard_model.pkl")
threshold = joblib.load("finguard_threshold.pkl")
demo_df = pd.read_csv("demo_transactions.csv")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "FinGuard AI uses a trained Random Forest machine-learning model "
    "to estimate the probability that a credit-card transaction is fraudulent."
)

st.info(
    "Model trained on the ULB Credit Card Fraud Detection benchmark dataset."
)

# --------------------------------------------------
# DATASET FEATURES
# --------------------------------------------------

feature_names = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]

# --------------------------------------------------
# TRANSACTION INPUT
# --------------------------------------------------

st.write("### 🔍 Transaction Analysis")

values = {}

col1, col2 = st.columns(2)

with col1:

    values["Time"] = st.number_input(
        "Transaction Time",
        value=0.0
    )

    values["V1"] = st.number_input("V1", value=0.0)
    values["V2"] = st.number_input("V2", value=0.0)
    values["V3"] = st.number_input("V3", value=0.0)
    values["V4"] = st.number_input("V4", value=0.0)
    values["V5"] = st.number_input("V5", value=0.0)
    values["V6"] = st.number_input("V6", value=0.0)
    values["V7"] = st.number_input("V7", value=0.0)
    values["V8"] = st.number_input("V8", value=0.0)
    values["V9"] = st.number_input("V9", value=0.0)
    values["V10"] = st.number_input("V10", value=0.0)
    values["V11"] = st.number_input("V11", value=0.0)
    values["V12"] = st.number_input("V12", value=0.0)
    values["V13"] = st.number_input("V13", value=0.0)
    values["V14"] = st.number_input("V14", value=0.0)

with col2:

    values["V15"] = st.number_input("V15", value=0.0)
    values["V16"] = st.number_input("V16", value=0.0)
    values["V17"] = st.number_input("V17", value=0.0)
    values["V18"] = st.number_input("V18", value=0.0)
    values["V19"] = st.number_input("V19", value=0.0)
    values["V20"] = st.number_input("V20", value=0.0)
    values["V21"] = st.number_input("V21", value=0.0)
    values["V22"] = st.number_input("V22", value=0.0)
    values["V23"] = st.number_input("V23", value=0.0)
    values["V24"] = st.number_input("V24", value=0.0)
    values["V25"] = st.number_input("V25", value=0.0)
    values["V26"] = st.number_input("V26", value=0.0)
    values["V27"] = st.number_input("V27", value=0.0)
    values["V28"] = st.number_input("V28", value=0.0)

    values["Amount"] = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )

# --------------------------------------------------
# FRAUD ANALYSIS
# --------------------------------------------------

if st.button(
    "🚨 Analyze Transaction",
    use_container_width=True
):

    transaction = pd.DataFrame(
        [values],
        columns=feature_names
    )

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

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.write("### 📊 Fraud Analysis")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

    with result_col2:

        st.metric(
            "Risk Level",
            risk_level
        )

    # --------------------------------------------------
    # RISK MESSAGE
    # --------------------------------------------------

    if probability >= threshold:

        st.error(
            "🚨 HIGH FRAUD RISK\n\n"
            "This transaction has a high estimated probability "
            "of fraudulent activity. Additional verification "
            "is recommended."
        )

    elif probability >= 0.10:

        st.warning(
            "⚠️ MEDIUM FRAUD RISK\n\n"
            "This transaction shows unusual characteristics. "
            "Additional review is recommended."
        )

    else:

        st.success(
            "✅ LOW FRAUD RISK\n\n"
            "The transaction has a low estimated probability "
            "of fraud."
        )

    # --------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------

    st.write("### 🤖 Model Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.metric(
            "Model",
            "Random Forest"
        )

    with info_col2:

        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )

    with info_col3:

        st.metric(
            "Features",
            "30"
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.write("---")

st.caption(
    "FinGuard AI | Machine Learning Fraud Detection System"
)
