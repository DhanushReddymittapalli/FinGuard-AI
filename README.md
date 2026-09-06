# 🛡️ FinGuard AI

## AI-Powered Credit Card Fraud Detection & Explainable AI

FinGuard AI is an end-to-end machine learning project for detecting suspicious credit-card transactions and estimating their probability of fraudulent activity.

The project combines a production-style fraud detection application with a research component focused on model comparison, classification thresholds, error analysis, and Explainable AI (XAI) using SHAP.

---

## 🚀 Project Highlights

- 🤖 Machine learning-based fraud detection
- 🌲 Random Forest fraud detection application
- 🚀 XGBoost research experiments
- 📊 Fraud probability scoring
- 🚨 Low / Medium / High risk classification
- 📈 ROC-AUC and PR-AUC evaluation
- 🎯 Classification threshold analysis
- 🔍 False-positive and false-negative analysis
- 🧠 SHAP-based Explainable AI
- 📊 Global feature importance
- 🔎 Individual transaction explanations
- 🌐 Interactive Streamlit dashboard
- 💾 Saved machine-learning model

---

# 💳 FinGuard AI Application

The main FinGuard AI application uses a Random Forest classifier to identify potentially fraudulent transactions.

The Streamlit dashboard provides an interactive interface for analyzing transaction data and understanding fraud risk.

### Application Features

- Total transaction monitoring
- Fraud and legitimate transaction statistics
- Fraud probability prediction
- Risk-level classification
- Transaction-level analysis
- Model performance metrics
- Interactive visualizations

---

## 📂 Dataset

The project uses the **ULB Credit Card Fraud Detection** benchmark dataset.

The original dataset contains:

- **284,807 transactions**
- **492 fraudulent transactions**
- **284,315 legitimate transactions**
- **30 predictive features**
- Severe class imbalance

The predictive variables include anonymized PCA-transformed features (`V1`–`V28`), transaction time, and transaction amount.

Because the dataset is highly imbalanced, accuracy alone is not an appropriate evaluation metric. Precision, recall, F1-score, ROC-AUC, and especially PR-AUC provide more meaningful evaluation.

---

# 🧠 Machine Learning Application

## Random Forest

The deployed FinGuard AI application uses a Random Forest classifier.

Configuration:

- `n_estimators = 100`
- `class_weight = "balanced"`
- `random_state = 42`

Class weighting helps address the severe imbalance between legitimate and fraudulent transactions.

---

## 📊 Application Model Results

| Metric | Score |
|---|---:|
| ROC-AUC | **0.953** |
| PR-AUC | **0.854** |
| Precision | **~92%** |
| Recall | **~85%** |
| F1-score | **~88%** |

The fraud decision threshold was optimized using the Precision-Recall curve.

### Optimized Threshold

**0.30**

This threshold was selected to provide a practical balance between detecting fraudulent transactions and reducing false alarms.

---

## 🚨 Fraud Risk Scoring

FinGuard AI converts predicted fraud probability into three risk levels:

| Fraud Probability | Risk Level |
|---|---|
| < 10% | 🟢 Low Risk |
| 10% – 30% | 🟡 Medium Risk |
| ≥ 30% | 🔴 High Risk |

---

# 🔬 FinGuard AI Research

The repository also contains a dedicated research component:

**Explainable Machine Learning for Credit Card Fraud Detection**

The research investigates:

- Machine learning model comparison
- Classification threshold effects
- Precision-recall trade-offs
- Prediction error analysis
- SHAP-based feature importance
- Individual transaction explanations

---

## 🧪 Research Models

Three machine learning approaches were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

The research evaluates the models using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score

---

## 🚀 XGBoost Research Results

The XGBoost research model achieved:

| Metric | Score |
|---|---:|
| ROC-AUC | **0.9816** |
| PR-AUC | **0.8565** |
| Precision | **0.6614** |
| Recall | **0.8571** |
| F1-score | **0.7467** |

These results demonstrate strong discrimination between fraudulent and legitimate transactions while maintaining high fraud-detection recall.

---

# 🎯 Threshold Analysis

The research evaluates multiple classification thresholds to understand how changing the decision boundary affects:

- Precision
- Recall
- F1-score
- False positives
- False negatives

This analysis demonstrates that fraud detection is not simply a binary prediction problem; the decision threshold can be adjusted depending on the desired balance between fraud detection and false alarms.

---

# 🔍 Error Analysis

Prediction errors were analyzed using:

- True Positives
- True Negatives
- False Positives
- False Negatives

This provides insight into where the model succeeds and where incorrect fraud or legitimate classifications occur.

---

# 🧠 Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) was applied to the XGBoost model to investigate why the model makes particular fraud predictions.

SHAP analysis was performed on:

- **1,000 test transactions**
- **30 features**

The most influential features in the analyzed sample were:

1. **V14**
2. **V4**
3. **V10**
4. **V12**
5. **V11**

SHAP provides both global feature importance and individual transaction-level explanations.

---

# 📊 Research Outputs

The research outputs are available in:

`FinGuard-AI-Research/results/`

They include:

- `FinGuard_model_comparison_results.csv`
- `FinGuard_model_comparison.png`
- `FinGuard_threshold_analysis.csv`
- `FinGuard_threshold_tradeoff.png`
- `FinGuard_error_analysis.csv`
- `FinGuard_shap_feature_importance.csv`
- `FinGuard_shap_feature_importance.png`
- `FinGuard_individual_explanation.png`

---

# 📓 Research Notebook

The complete research workflow is available in:

`FinGuard_XAI_Research.ipynb`

The notebook includes:

- Data preparation
- Train/test splitting
- Model training
- Model comparison
- Evaluation metrics
- Threshold analysis
- Error analysis
- SHAP explainability
- Visualization and result generation

---

# 🏗️ Project Structure

```text
FinGuard-AI/
│
├── app.py
├── finguard_model.pkl
├── finguard_threshold.pkl
├── demo_transactions_small.csv
├── requirements.txt
├── transactions.csv
├── README.md
│
└── FinGuard-AI-Research/
    ├── FinGuard_XAI_Research.ipynb
    ├── notebooks/
    │   └── .gitkeep
    ├── results/
    │   ├── FinGuard_model_comparison_results.csv
    │   ├── FinGuard_model_comparison.png
    │   ├── FinGuard_threshold_analysis.csv
    │   ├── FinGuard_threshold_tradeoff.png
    │   ├── FinGuard_error_analysis.csv
    │   ├── FinGuard_shap_feature_importance.csv
    │   ├── FinGuard_shap_feature_importance.png
    │   └── FinGuard_individual_explanation.png
    └── README.md
