# FinGuard AI Research

## Explainable Machine Learning for Credit Card Fraud Detection

This research project extends FinGuard AI by investigating machine learning models, classification thresholds, error patterns, and Explainable AI (XAI) techniques for credit card fraud detection.

The goal is to evaluate different machine learning approaches and improve the interpretability of fraud detection predictions using SHAP (SHapley Additive exPlanations).

---

## Research Objective

The main objectives of this research are to:

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

The results demonstrate strong discrimination between fraudulent and legitimate transactions while maintaining high fraud detection recall.

---

## Threshold Analysis

Different classification thresholds were evaluated to investigate the trade-off between precision and recall.

Changing the classification threshold affects the number of transactions predicted as fraudulent.

This analysis helps study how a fraud detection system can balance:

- Detecting fraudulent transactions.
- Reducing false positives.
- Improving precision.
- Maintaining acceptable recall.

The complete threshold analysis and visualization are available in the `results/` directory.

---

## Error Analysis

Error analysis was performed to study:

- True positives
- True negatives
- False positives
- False negatives

This analysis helps identify model limitations and provides insight into incorrect fraud and legitimate transaction predictions.

---

## Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) was applied to the XGBoost model to improve prediction interpretability.

SHAP analysis was performed on:

- **1,000 test transactions**
- **30 features**

The most influential features based on mean absolute SHAP value were:

1. **V14**
2. **V4**
3. **V10**
4. **V12**
5. **V11**

SHAP provides both global feature importance and individual transaction-level explanations.

---

## Key Research Findings

- XGBoost achieved a **ROC-AUC of 0.9816**.
- Threshold selection significantly affects the precision-recall trade-off.
- SHAP identified **V14, V4, V10, V12, and V11** as the most influential features in the analyzed sample.
- Error analysis provides insight into false-positive and false-negative predictions.
- SHAP explanations provide additional interpretability for individual predictions.
- Combining machine learning with Explainable AI can make fraud detection systems easier to analyze and understand.

---

## Research Outputs

The `results/` directory contains the following research outputs:

- `FinGuard_model_comparison_results.csv` — model performance results
- `FinGuard_model_comparison.png` — model comparison visualization
- `FinGuard_threshold_analysis.csv` — threshold performance analysis
- `FinGuard_threshold_tradeoff.png` — threshold trade-off visualization
- `FinGuard_error_analysis.csv` — prediction error analysis
- `FinGuard_shap_feature_importance.csv` — SHAP feature importance values
- `FinGuard_shap_feature_importance.png` — SHAP feature importance visualization
- `FinGuard_individual_explanation.png` — individual transaction SHAP explanation

---

## Research Notebook

The complete experimental workflow is available in:

**`FinGuard_XAI_Research.ipynb`**

The notebook contains the data preparation, model training, evaluation, threshold analysis, error analysis, and SHAP explainability experiments.

---

## Project Structure

```text
FinGuard-AI-Research/
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
