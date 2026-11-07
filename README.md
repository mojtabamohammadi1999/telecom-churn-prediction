# Telecom Customer Churn Prediction

Predicting whether a telecom customer will leave, using account and service data.
The main challenge is the **imbalanced target** (~73% stayed, ~27% churned), so I compare
several ways of handling it and judge them with recall and F1 instead of accuracy alone.

## Dataset
- Telco Customer Churn (IBM sample data, via Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- ~7,000 customers, 20 features (demographics, services, contract, billing)
- Download the CSV, rename it to `Telecom_customer_churn.csv` and put it next to the notebook.

## What's inside
1. Data cleaning (fixing `TotalCharges`, removing 11 blank rows)
2. EDA (churn vs. tenure and monthly charges)
3. Preprocessing (binary encoding, one-hot encoding, MinMax scaling)
4. Train/test split **before** any scaling or resampling to avoid data leakage
5. Logistic Regression with five approaches to class imbalance

## Results (held-out test set, real class distribution)

| Method | Accuracy | Recall (churn) | F1 (churn) |
|---|---|---|---|
| No weights | 0.803 | 0.54 | 0.59 |
| Class weight [1, 1.5] | 0.793 | 0.68 | 0.64 |
| Undersampling | 0.742 | 0.78 | 0.62 |
| Oversampling | 0.740 | 0.79 | 0.62 |
| SMOTE | 0.761 | 0.73 | 0.62 |

**Takeaways**
- Class weighting gave the best F1 for the churn class with the simplest approach.
- Resampling increased recall but lowered precision (~0.51).
- The scaler and all resampling are fitted on the training set only.

## How to run
```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn jupyter
jupyter notebook churn_modeling.ipynb
```

## Next steps
Tree-based models (Random Forest, XGBoost) and decision-threshold tuning.