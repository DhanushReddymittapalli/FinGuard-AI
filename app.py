import streamlit as st
import pandas as pd

from model.predictor import predict_transaction, get_feature_importance


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# LOAD DEMO DATASET
# ============================================================

@st.cache_data
def load_demo_dataset():
    try:
        return pd.read_csv("demo_transactions_small.csv")

    except FileNotFoundError:
        st.error(
            "demo_transactions_small.csv was not found "
            "in the repository root."
        )
        st.stop()

    except Exception as exc:
        st.error(f"Could not load the demo dataset: {exc}")
        st.stop()


df = load_demo_dataset()


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ FinGuard AI")

st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "FinGuard AI uses machine learning to detect suspicious "
    "credit-card transactions and classify their risk level."
)


# ============================================================
# FRAUD MONITORING DASHBOARD
# ============================================================

st.header("📊 Fraud Monitoring Dashboard")

total_transactions = len(df)

if "Class" in df.columns:
    fraud_cases = int(
        pd.to_numeric(
            df["Class"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )
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

    chart_df = pd.DataFrame(
        {
            "Type": [
                "Fraud",
                "Legitimate"
            ],
            "Count": [
                fraud_cases,
                legitimate_cases
            ],
        }
    )

    st.bar_chart(
        chart_df.set_index("Type")
    )

else:
    st.info(
        if "Class" in df.columns:
