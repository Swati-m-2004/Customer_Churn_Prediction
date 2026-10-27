# Customer Churn Prediction & Retention Analysis

**Internship Day 36 — Project Day 1 of 4**

## Overview
This project predicts whether a subscription customer is likely to leave (churn) and explores factors associated with churn. It compares Logistic Regression, Decision Tree, and Random Forest; evaluates ROC-AUC and class-specific metrics; and exports feature importance plus a retention-insights checklist.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn. Jupyter Notebook is optional; Streamlit may be added later.

## Dataset
Default: IBM Telco Customer Churn CSV:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

If unavailable, download a compatible IBM Telco Customer Churn CSV and set `DATA_PATH` in the Python script. The CSV should contain `Churn` values `Yes` and `No`. Common fields include `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, and `InternetService`.

## Install and run
```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
python customer_churn_prediction.py
```
The default dataset requires internet access. For a local file, set `DATA_PATH = "your_file.csv"`.

## Workflow
1. Inspect data types, missing values, duplicates, and dataset shape.
2. Clean churn labels and encode Yes as 1, No as 0.
3. Convert `TotalCharges` to numeric where available.
4. Exclude identifiers and target fields from predictors.
5. Impute missing values, scale numeric features, and one-hot encode categorical fields in a pipeline.
6. Explore churn distribution, tenure, monthly charges, contract, and internet service.
7. Train three classifiers with a stratified train/test split.
8. Compare accuracy, churn precision, recall, F1-score, and ROC-AUC.
9. Plot confusion matrices and ROC curves.
10. Export Random Forest feature importance.
11. Use `predict_churn()` for a customer record.

## Outputs
The `customer_churn_outputs/` folder contains EDA charts, confusion matrices, ROC comparison, `model_comparison.csv`, `cleaned_churn_dataset.csv`, `feature_importance.csv`, and `retention_recommendations_template.txt`.

## Evaluation notes
Churn data can be imbalanced, so accuracy alone can be misleading. Review churn recall, precision, F1-score, confusion matrices, and ROC-AUC. The sample prediction uses a 0.50 threshold; real deployments should select a threshold based on business costs and validation.

## Retention ideas
Use the actual model and EDA findings to investigate proactive support, clearer contract/renewal information, relevant offers, and outreach before renewal. These are hypotheses to test, not guaranteed outcomes. Avoid unfair targeting; model scores should not be the sole basis for customer decisions.

## Interview questions
1. What is customer churn?
2. Why is churn prediction a classification problem?
3. Why can accuracy be misleading on imbalanced data?
4. What is ROC-AUC?
5. What is the difference between precision and recall?
6. Why use one-hot encoding?
7. What does feature importance mean?
8. How can churn prediction be used responsibly?

**Author:** Swati Muttannavar — AI & ML Track
