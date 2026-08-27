import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ FinGuard AI")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.success("✅ FinGuard AI is running successfully!")

df = pd.read_csv("demo_transactions_small.csv")

st.write("### 📊 Dataset")
st.metric("Total Transactions", len(df))

if "Class" in df.columns:
    fraud = int(df["Class"].sum())
    legitimate = len(df) - fraud

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Transactions", len(df))

    with c2:
        st.metric("Fraud", fraud)

    with c3:
        st.metric("Legitimate", legitimate)

    st.dataframe(df.head(10), use_container_width=True)
    
