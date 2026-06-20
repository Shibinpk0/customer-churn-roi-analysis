# 📞 Customer Churn Prediction & ROI Optimizer
## 🎯 Overview
Telecom companies often spend millions on retention campaigns by offering discounts to customers who may never leave. This project uses Machine Learning, Explainable AI, and ROI analysis to help businesses identify high-risk customers, optimize retention spending, and maximize profitability.

The system not only predicts customer churn but also explains the reasons behind each prediction and evaluates the financial impact of retention strategies.

---

## ✨ Key Features

### 🔍 Churn Prediction

* Predicts customers likely to churn using advanced machine learning models.
* Handles imbalanced datasets effectively.
* Achieved **84.16% AUC-ROC** using XGBoost.

### 📊 Explainable AI (SHAP)

* Provides transparent predictions through SHAP explanations.
* Visualizes feature contributions for each customer.
* Helps business teams understand the factors driving churn.

### 💰 ROI Optimization Engine

* Calculates retention campaign costs and expected savings.
* Identifies the most profitable customer segments to target.
* Enables data-driven decision-making for retention strategies.

### 🎚️ Dynamic Risk Threshold Analysis

* Interactive slider to adjust churn-risk thresholds.
* Instantly updates:

  * Customers targeted
  * Campaign costs
  * Expected revenue retained
  * Net ROI

---

## 📸 Dashboard Demonstrations

### 1️⃣ Individual Customer Risk & SHAP Explanation

<img width="1894" height="901" alt="model_benchmarking" src="https://github.com/user-attachments/assets/08ba766f-da74-45b5-8046-7895febb0a96" />


The SHAP Waterfall Plot explains exactly why a customer is predicted to churn.

**Example:**

* Month-to-Month Contract → Increases churn risk
* Short Tenure → Increases churn risk
* High Monthly Charges → Increases churn risk

This eliminates the "black-box" problem and improves stakeholder trust.

---

### 2️⃣ Dynamic Cost-Benefit Threshold Analysis

![ROI Threshold Analysis](<img width="1909" height="722" alt="roi_threshold_demo" src="https://github.com/user-attachments/assets/7790a65d-9205-40b3-adc7-bb53791c555c" />
)

Business users can adjust the churn-risk threshold and observe the impact on:

* Targeted Customers
* Retention Cost
* Revenue Saved
* Net ROI

This helps determine the most profitable retention strategy.

---

### 3️⃣ Model Performance Comparison

![Model Comparison](<img width="1894" height="901" alt="model_benchmarking" src="https://github.com/user-attachments/assets/66fb0d82-6387-496c-bc84-39fc09cb26d9" />
)

| Model               | AUC-ROC    |
| ------------------- | ---------- |
| Logistic Regression | 82.31%     |
| Random Forest       | 83.47%     |
| LightGBM            | 83.92%     |
| **XGBoost**         | **84.16%** |

**Selected Model:** XGBoost

Chosen for its superior performance on imbalanced customer churn data.

---

## 🛠️ Tech Stack

### Data Engineering

* SQLite
* SQL Queries
* Pandas

### Machine Learning

* XGBoost
* Scikit-Learn
* SMOTE / Class Imbalance Handling

### Explainable AI

* SHAP
* TreeExplainer
* Waterfall Plots

### Deployment & Automation

* Streamlit
* GitHub Actions
* Streamlit Cloud

---

## 📈 Business Impact

This project enables organizations to:

✅ Reduce unnecessary retention spending

✅ Improve customer retention strategies

✅ Understand churn drivers

✅ Maximize campaign ROI

✅ Build trust through explainable predictions

---

## 🚀 Installation

```bash
# Clone repository
git clone <repository-url>

# Move into project folder
cd customer-churn-roi-optimizer

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

---

## 🔗 Live Demo

**Interactive Dashboard:**

[[The LINK]](https://customer-churn-roi-analysis-5eypgnz4mknfzwttaqolzw.streamlit.app)

---

## 👨‍💻 Author

**Shibin**

Final Year Student | Data Science & Machine Learning Enthusiast


For internships and recruiters, this version looks significantly more professional and can help showcase your ML, business analytics, explainable AI, and deployment skills in one project.
