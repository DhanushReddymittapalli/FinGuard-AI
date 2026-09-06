# FinGuard AI Research

## Explainable Machine Learning for Credit Card Fraud Detection

This research project extends FinGuard AI by investigating machine learning models, decision thresholds, error patterns, and Explainable AI (XAI) techniques for credit card fraud detection.

## Research Objective

The objective is to compare different machine learning approaches and improve the interpretability of fraud detection predictions using SHAP (SHapley Additive exPlanations).

## Research Questions

1. Which machine learning model provides the best fraud detection performance?
2. How does the classification threshold affect precision and recall?
3. Which transaction features contribute most to fraud predictions?
4. Can Explainable AI make fraud detection decisions easier to interpret?

## Dataset

The experiments use the Credit Card Fraud Detection dataset.

- Total transactions: 284,807
- Fraudulent transactions: 492
- Legitimate transactions: 284,315
- Features: 30

The dataset contains severe class imbalance, making PR-AUC, precision, recall, and F1-score important evaluation metrics.

## Research Methodology

### Data Preparation

The dataset was divided into training and testing sets using a stratified 80/20 split.

### Machine Learning Models

Three models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

### Evaluation Metrics

The models were evaluated using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score

### Threshold Analysis

Different classification thresholds were investigated to study the trade-off between detecting fraudulent transactions and reducing false positives.

### Error Analysis

False positives and false negatives were analyzed to understand model limitations and identify areas for improvement.

### Explainable AI

SHAP was applied to the XGBoost model.

SHAP analysis was performed on 1,000 test transactions across 30 features.

The most influential features identified were:

1. V14
2. V4
3. V10
4. V12
5. V11

## XGBoost Results

The XGBoost experiment achieved:

- ROC-AUC: **0.9816**
- PR-AUC: **0.8565**
- Precision: **0.6614**
- Recall: **0.8571**
- F1-score: **0.7467**

## Research Outputs

The `results/` directory contains:

- Model comparison results
- Model comparison visualization
- Threshold analysis
- Threshold trade-off visualization
- Error analysis
- SHAP feature importance results
- SHAP feature importance visualization
- Individual SHAP explanation

## Project Structure

```text
FinGuard-AI-Research/
├── data/
├── notebooks/
├── models/
├── results/
└── README.md
