# E-Commerce_Fraud_Detection
# Fraud Detection using Machine Learning

## 📌 Project Overview

This project focuses on detecting fraudulent transactions using a
dataset of **1,472,952 rows**.\
Multiple machine learning models were applied, and their performance was
evaluated using accuracy, classification reports, ROC-AUC, and confusion
matrices.\
Additionally, exploratory data analysis (EDA) was performed to
understand fraud patterns across various features.

------------------------------------------------------------------------

## 🧾 Dataset

Key features used for analysis and modeling include: - **IsFraudulent**
(Target variable: 0 = Not Fraud, 1 = Fraud) - **TransactionAmount** -
**PaymentMethod** (bank transfer, debit card, PayPal, credit card) -
**ProductCategory** (electronics, clothing, health & beauty, etc.) -
**TransactionHour** - **CustomerAge**

------------------------------------------------------------------------

## 📊 Exploratory Data Analysis (EDA) Insights

### Fraud vs Non-Fraud Transactions

-   Fraudulent transactions are **rare compared to non-fraudulent
    ones**, showing a class imbalance problem.

### Transaction Amount by Fraud Status

-   Fraudulent transactions generally involve **higher transaction
    amounts** compared to non-fraudulent ones.

### Fraud by Payment Method

-   Fraud occurs across all payment methods, with **similar distribution
    patterns**.

### Fraud by Product Category

-   All product categories show fraud cases, but **fraud proportion
    remains consistently low**.

### Fraudulent Transactions by Hour of Day

-   Fraud is **most common during late-night to early-morning hours
    (0--5 AM)**.

### Fraud by Customer Age Distribution

-   Fraud is most common among **customers aged 25--45**, following the
    same trend as non-fraudulent transactions.

------------------------------------------------------------------------

## 🤖 Machine Learning Models Used

The following models were trained and compared: 1. **Logistic
Regression** 2. **Decision Tree** 3. **Random Forest** 4. **XGBoost** 5.
**CatBoost** 6. **Neural Network (MLP)** 7. **HistGradient Boosting
Classifier** 8. **Support Vector Machine (SVM)** (optional, slower on
large datasets)

------------------------------------------------------------------------

## 📈 Evaluation Metrics

Each model was evaluated using: - **Accuracy** - **Classification Report
(Precision, Recall, F1-score)** - **ROC-AUC Score** - **Confusion Matrix
(Visualized)**

Example evaluation function:

``` python
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    print(f"\n {model_name} Results")
    print("-"*40)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    if y_proba is not None:
        print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
    print("="*40)
```

------------------------------------------------------------------------

## 📊 Model Comparison

### Accuracy Comparison Visualization

Bar chart was used to compare accuracy across models, annotated with
exact values.

### Confusion Matrix Visualization

Heatmaps were plotted for each model to show **True Positive, True
Negative, False Positive, and False Negative counts**.

------------------------------------------------------------------------

## ⚡ Performance Notes

-   Random Forest and XGBoost were **slow on the large dataset**,
    requiring parameter tuning (e.g., `n_estimators`, `max_depth`) or
    sampling for faster execution.\
-   Neural Network (MLP) was optimized by reducing hidden layers and
    iterations for faster training.

------------------------------------------------------------------------

## ✅ Conclusion

-   Fraud detection is challenging due to **severe class imbalance**.\
-   Fraudulent transactions often involve **higher amounts and occur
    late at night**.\
-   Ensemble methods like **XGBoost, Random Forest, and CatBoost**
    provided better performance than simple models.\
-   Further improvements can be made using **SMOTE, anomaly detection,
    or deep learning techniques**.

------------------------------------------------------------------------

## 📌 Next Steps

-   Handle **class imbalance** with oversampling/undersampling.\
-   Deploy best-performing model using **Flask API + React Dashboard**.\
-   Explore **real-time fraud detection pipeline**.
