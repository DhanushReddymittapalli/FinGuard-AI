# 💰 FinGuard AI
## 🚀 Live Demo

[Open FinGuard AI](https://uwg.streamlit.app/)

## 🤖 AI-Based Personal Finance & Fraud Risk Analyzer

FinGuard AI is a machine learning application that analyzes financial transactions and identifies potential fraud risk.

The project uses transaction amount and transaction frequency to classify transactions as **Safe** or **High Fraud Risk**.

## 🚀 Features

- 💳 Transaction analysis
- 🤖 Machine learning fraud detection
- 📊 Transaction data visualization
- 🎯 Fraud risk prediction
- 📈 Model accuracy evaluation
- 🌐 Interactive Streamlit web application

## 🛠️ Technologies

- Python
- Pandas
- Scikit-learn
- Random Forest
- Streamlit

## 🧠 Machine Learning Model

The project uses a **Random Forest Classifier** to learn patterns from transaction data and predict potential fraud risk.

### Input Features

- Transaction Amount
- Transaction Frequency

### Output

- ✅ Safe Transaction
- ⚠️ High Fraud Risk

## 📊 Model Evaluation

The model is evaluated using **accuracy score** on the test dataset.

## 📁 Project Structure

```text
FinGuard-AI/
├── app.py
├── transactions.csv
├── requirements.txt
└── README.md
pip install -r requirements.txt
streamlit run app.py
## 📸 Application Screenshot

![FinGuard AI Application](Screenshot_20260826_143423.jpg)
