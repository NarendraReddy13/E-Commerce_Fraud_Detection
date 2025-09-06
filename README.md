# Fraud Detection in E-Commerce Transactions 🛒🔍

## 📌 Project Overview

This project focuses on detecting **fraudulent e-commerce transactions** using **machine learning and deep learning models**.
The dataset contains over **1.47 million transactions**, labeled as **fraudulent (1)** or **non-fraudulent (0)**.

By analyzing **transaction patterns, payment methods, product categories, customer behavior, and time-of-day effects**, this project builds models that can **accurately classify frauds in highly imbalanced data**.

---

## 📊 Dataset Details

* **Rows:** 1,472,952
* **Columns:**

  * TransactionID, CustomerID, TransactionAmount, TransactionDate
  * PaymentMethod, ProductCategory, Quantity
  * CustomerAge, CustomerLocation, DeviceUsed, IPAddress
  * ShippingAddress, BillingAddress
  * **IsFraudulent (Target: 0 or 1)**
  * AccountAgeDays, TransactionHour

---

## 🔍 Key Insights from Data

* **Fraud Imbalance:** Only a small portion of transactions are fraudulent (\~5%).
* **Transaction Amounts:** Fraudulent transactions often involve **higher amounts** with many outliers.
* **Time of Day:** Most frauds happen during **late-night/early-morning hours (0–5 AM)**.
* **Customer Age:** Fraud distribution peaks among **25–45 year-old customers**, similar to normal transactions.
* **Payment Methods:** Fraud is present across **bank transfer, debit card, PayPal, and credit card**, with no single method dominating.
* **Product Categories:** Fraud is spread across **electronics, clothing, health & beauty, home & garden, toys & games**.

---

## 🤖 Machine Learning Models Applied

We experimented with **8 models** for binary classification (fraud vs. non-fraud):

1. **Logistic Regression** – baseline linear classifier
2. **Decision Tree** – interpretable tree-based model
3. **Random Forest** – ensemble of decision trees
4. **XGBoost** – gradient boosting with high accuracy
5. **CatBoost** – boosting optimized for categorical features
6. **MLP (Neural Network)** – multi-layer perceptron
7. **HistGradientBoosting** – fast gradient boosting method
8. **Support Vector Classifier (SVC)** – margin-based classifier

---

## 📈 Model Evaluation Metrics

For each model, we evaluate using:

* ✅ **Accuracy**
* ✅ **Precision, Recall, F1-score**
* ✅ **ROC-AUC**
* ✅ **Confusion Matrix**

Example evaluation function:

```python
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    
    print(f"\n🔹 {model_name} Results")
    print("-"*40)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    if y_proba is not None:
        print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
    print("="*40)
```

---

## 📊 Visualizations

### Fraud vs. Non-Fraud Distribution

Shows the **class imbalance** in the dataset.
*(Non-fraudulent transactions \~1.4M vs. fraudulent \~70K)*

### Transaction Amount by Fraud Status

Fraudulent transactions tend to have **larger values with extreme outliers**.

### Fraud by Payment Method

Fraud exists across **all payment methods**, but always fewer than non-frauds.

### Fraud by Product Category

Fraud is spread across all categories with no dominant category.

### Fraud by Hour of Day

Fraud is **more common at night (0–5 AM)** compared to daytime.

### Fraud by Customer Age

Fraud distribution peaks among **25–45 year-olds**, matching normal customer age distribution.

---

## 📌 How to Run the Project

1. Clone the repo:

```bash
git clone https://github.com/NarendraReddy13/E-Commerce_Fraud_Detection.git
cd E-Commerce_Fraud_Detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Jupyter Notebook:

```bash
jupyter notebook
```

4. Train models and view results.

---

## 🚀 Future Improvements

* Apply **SMOTE / class-weighting** to handle imbalance
* Use **AutoML (PyCaret, H2O)** for better model selection
* Deploy the best model as a **Flask / FastAPI API**
* Integrate with a **real-time fraud monitoring dashboard**

---
