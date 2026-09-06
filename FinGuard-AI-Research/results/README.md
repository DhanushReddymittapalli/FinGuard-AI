# FinGuard AI Research

## Research Objective

This research extends FinGuard AI with model comparison, threshold analysis, error analysis, and Explainable AI to improve fraud detection performance and interpretability.

## Model Performance

The XGBoost model achieved:

- ROC-AUC: 0.9816
- PR-AUC: 0.8565
- Precision: 0.6614
- Recall: 0.8571
- F1-score: 0.7467

## Explainable AI Analysis

SHAP (SHapley Additive exPlanations) was used to interpret the XGBoost fraud detection model.

SHAP analysis was performed on 1,000 test transactions across 30 features.

The most influential features were:

1. V14
2. V4
3. V10
4. V12
5. V11
## Research Methodology

### Dataset

The Credit Card Fraud Detection dataset was used for the experiments. The dataset contains 284,807 transactions with 492 fraudulent transactions and 284,315 legitimate transactions.

### Data Preparation

The data was divided into training and testing sets using a stratified 80/20 split to preserve the fraud-to-legitimate transaction ratio.

### Models

Three machine learning approaches were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The models were evaluated using ROC-AUC, PR-AUC, Precision, Recall, and F1-score.

### Explainable AI

SHAP was applied to the XGBoost model to understand the contribution of individual features to fraud predictions. A sample of 1,000 test transactions and 30 features was analyzed.

### Evaluation

Additional threshold analysis and error analysis were performed to study the trade-off between detecting fraudulent transactions and reducing false positives.
## Research Contribution

The project combines machine learning, model evaluation, threshold analysis, error analysis, and Explainable AI to develop a more interpretable fraud detection framework.
