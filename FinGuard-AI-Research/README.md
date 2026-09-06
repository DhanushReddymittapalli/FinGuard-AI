# FinGuard AI Research

## Explainable Machine Learning for Credit Card Fraud Detection

This research project extends FinGuard AI by investigating machine learning models, classification thresholds, error patterns, and Explainable AI (XAI) techniques for credit card fraud detection.

The goal is to evaluate different machine learning approaches and improve the interpretability of fraud detection predictions using SHAP (SHapley Additive exPlanations).

---

## Research Objective

The main objective of this research is to:

- Compare multiple machine learning models for credit card fraud detection.
- Evaluate model performance using appropriate classification metrics.
- Study the effect of classification thresholds on fraud detection.
- Analyze false positives and false negatives.
- Identify important transaction features using SHAP.
- Improve the interpretability of machine learning-based fraud detection.

---

## Research Questions

1. Which machine learning model provides the best fraud detection performance?
2. How does the classification threshold affect precision, recall, and F1-score?
3. Which transaction features contribute most to fraud predictions?
4. Can Explainable AI make fraud detection decisions easier to interpret?

---

## Dataset

The experiments use the **Credit Card Fraud Detection dataset**.

- Total transactions: **284,807**
- Fraudulent transactions: **492**
- Legitimate transactions: **284,315**
- Predictive features: **30**

The dataset contains severe class imbalance. Therefore, accuracy alone is not an appropriate evaluation measure. PR-AUC, precision, recall, and F1-score are important metrics for this research.

---

## Research Methodology

### 1. Data Preparation

The dataset was divided into training and testing sets using a **stratified 80/20 split**.

Stratification was used to preserve the proportion of fraudulent and legitimate transactions in both sets.

### 2. Machine Learning Models

Three machine learning models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

### 3. Evaluation Metrics

The models were evaluated using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score

These metrics provide a more meaningful evaluation for highly imbalanced fraud detection data.

---

## XGBoost Results

The XGBoost model achieved the following results:

| Metric | Score |
|---|---:|
| ROC-AUC | **0.9816** |
| PR-AUC | **0.8565** |
| Precision | **0.6614** |
| Recall | **0.8571** |
| F1-score | **0.7467** |

The results demonstrate that XGBoost provides strong discrimination between fraudulent and legitimate transactions while maintaining
