# Credit Scoring Model (CodeAlpha Machine Learning Internship - Task 1)

## Overview
This repository contains the implementation of a **Credit Scoring Model** developed during the Machine Learning Internship at **CodeAlpha**. The primary goal is to assess an individual's creditworthiness and predict the likelihood of loan default based on historical financial and demographic data.

---

## Dataset
- **Dataset:** German Credit Dataset (1,000 instances, 20 features)
- **Target Variable:** `credit_risk` (Good Credit Risk vs. Bad Credit Risk)
- **Features Included:**
  - Account status
  - Duration in months
  - Credit history & past payment behavior
  - Purpose of credit
  - Credit amount
  - Savings account balance
  - Employment duration & installment rate
  - Personal status, housing, age, and existing credits

---

## Workflow & Implementation

1. **Data Preprocessing & Exploration:**
   - Renamed raw abbreviated columns to clear descriptive identifiers.
   - Checked for missing values and confirmed data integrity.
   - Performed train-test split (80% training, 20% testing) with stratification to handle class distribution.
   - Applied `StandardScaler` to normalize feature ranges for distance/gradient-based algorithms.

2. **Model Training:**
   - **Logistic Regression:** Serves as an interpretable baseline classification model.
   - **Random Forest Classifier:** An ensemble tree-based algorithm to capture non-linear relationships.

3. **Evaluation Metrics:**
   - **Precision, Recall, and F1-Score:** To measure trade-offs across risk classes.
   - **ROC-AUC Score & ROC Curve:** Plotted to benchmark the discriminative power between both classifiers.
   - **Confusion Matrix:** Evaluated false positives and false negatives.

4. **Inference Pipeline:**
   - Implemented a prediction function (`predict_credit_risk`) capable of taking arbitrary new applicant records and outputting risk predictions alongside model confidence.

---

## Tech Stack & Libraries
- **Language:** Python
- **Environment:** Google Colab / Jupyter Notebook
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## How to Run
1. Open the `.ipynb` notebook directly in Google Colab or locally in Jupyter Notebook:
   ```bash
   jupyter notebook task1_credit_scoring_model.ipynb
