# 🛡️ FinGuard AI

## AI-Powered Credit Card Fraud Detection System

FinGuard AI is a machine-learning based fraud detection system designed to identify suspicious credit-card transactions and estimate their probability of fraudulent activity.

The system uses a Random Forest classifier trained on the ULB Credit Card Fraud Detection benchmark dataset and provides an interactive Streamlit dashboard for transaction analysis.

---

## 🚀 Features

- 🤖 Random Forest fraud detection model
- 📊 Fraud probability scoring
- 🚨 Low / Medium / High risk classification
- 📈 ROC-AUC and PR-AUC evaluation
- 🎯 Optimized fraud decision threshold
- 📋 Confusion matrix and classification metrics
- 🔍 Real transaction analysis
- 📊 Interactive monitoring dashboard
- 🌐 Streamlit deployment
- 💾 Saved machine-learning model

---

## 📂 Dataset

The model was trained using the ULB Credit Card Fraud Detection benchmark dataset.

The original dataset contains:

- **284,807 transactions**
- **492 fraudulent transactions**
- **284,315 legitimate transactions**
- **30 predictive features**
- Severe class imbalance

The dataset contains anonymized features (`V1`–`V28`) produced through PCA, along with transaction time and transaction amount.

---

## 🧠 Machine Learning

### Model

**Random Forest Classifier**

The model uses:

- `n_estimators = 100`
- `class_weight = "balanced"`
- `random_state = 42`

Class weighting was used to address the severe imbalance between legitimate and fraudulent transactions.

---

## 📊 Model Evaluation

The dataset was split using a stratified train/test split.

### Results

| Metric | Score |
|---|---:|
| ROC-AUC | **0.953** |
| PR-AUC | **0.854** |
| Fraud Precision | **~92%** |
| Fraud Recall | **~85%** |
| F1 Score | **~88%** |

The fraud decision threshold was optimized using the Precision-Recall curve.

### Optimized Threshold

**0.30**

This threshold provides a better balance between detecting fraudulent transactions and limiting false alarms.

---

## 🔍 Fraud Risk Scoring

FinGuard AI converts the model's fraud probability into risk levels:

| Probability | Risk |
|---|---|
| < 10% | 🟢 Low Risk |
| 10% – 30% | 🟡 Medium Risk |
| ≥ 30% | 🔴 High Risk |

---

## 📊 Dashboard

The deployed application provides:

- Total transactions
- Fraud cases
- Legitimate transactions
- Fraud rate
- Dataset distribution
- Model performance
- Transaction-level fraud analysis
- Fraud probability
- Risk classification

---

## 🏗️ Project Structure

```text
FinGuard-AI/
│
├── app.py
├── finguard_model.pkl
├── finguard_threshold.pkl
├── demo_transactions_small.csv
├── requirements.txt
└── README.md
