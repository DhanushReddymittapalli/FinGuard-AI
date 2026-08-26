import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="💰"
)

st.title("💰 FinGuard AI")
st.subheader("AI-Based Personal Finance & Fraud Risk Analyzer")

st.write(
    "Analyze financial transactions and identify potential fraud risk "
    "using machine learning."
)

# Load transaction dataset
df = pd.read_csv("transactions.csv")

st.write("### 📊 Transaction Data")
st.dataframe(df, use_container_width=True)

# Check required columns
required_columns = ["amount", "frequency", "risk"]

if all(column in df.columns for column in required_columns):

    # Machine learning data
    X = df[["amount", "frequency"]]
    y = df["risk"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Model evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    st.metric(
    "Model Accuracy",
        f"{accuracy * 100:.1f}%"
    )
    cm = confusion_matrix(y_test, predictions)

st.write("### 🔍 Confusion Matrix")
st.dataframe(cm)

     User prediction
st.write("### 🔍 Check a Transaction")
    

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=500.0,
        step=50.0
    )

    frequency = st.number_input(
        "Transaction Frequency",
        min_value=1,
        value=2,
        step=1
    )

    if st.button("🔎 Analyze Transaction"):

        user_data = pd.DataFrame({
            "amount": [amount],
            "frequency": [frequency]
        })

        result = model.predict(user_data)[0]

        if result == 1:
            st.error("⚠️ High Fraud Risk Detected")
        else:
            st.success("✅ Transaction Appears Safe")

else:
    st.error(
        "Dataset must contain these columns: "
        "amount, frequency, risk"
    )
st.write("### 📊 Fraud Risk Distribution")

risk_counts = df["risk"].value_counts().rename(
    index={0: "Safe", 1: "Fraud Risk"}
)

st.bar_chart(risk_counts)
