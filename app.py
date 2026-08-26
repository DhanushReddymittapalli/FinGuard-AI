import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Sample transaction data
data = {
    "amount": [120, 450, 80, 1500, 200, 75, 2200, 300, 90, 1800,
               110, 500, 2500, 60, 700, 130, 1900, 350, 100, 2100],
    "frequency": [2, 3, 1, 8, 2, 1, 9, 3, 1, 7,
                  2, 4, 10, 1, 5, 2, 8, 3, 1, 9],
    "risk": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
             0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

# Machine learning model
X = df[["amount", "frequency"]]
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Model accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.metric("Model Accuracy", f"{accuracy * 100:.1f}%")

st.write("### 📊 Transaction Data")
st.dataframe(df)

# User prediction
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

if st.button("Analyze Transaction"):
    result = model.predict([[amount, frequency]])[0]

    if result == 1:
        st.error("⚠️ High Fraud Risk Detected")
        st.warning("Review this transaction carefully.")
    else:
        st.success("✅ Low Fraud Risk")
        st.info("Transaction appears normal.")

st.write("### 💡 Financial Insight")

average_amount = df["amount"].mean()

st.info(
    f"The average transaction amount in the dataset is "
    f"₹{average_amount:.0f}."
)

st.write("### 🚀 Future Improvements")
st.write(
    "- Add real transaction datasets\n"
    "- Add spending category analysis\n"
    "- Add monthly expense forecasting\n"
    "- Add interactive charts\n"
    "- Improve fraud detection accuracy"
)
