📞 Customer Churn Prediction & ROI Optimizer
🎯 Business Problem
Telecom companies waste millions giving retention discounts to customers who were never going to leave. This project builds an ML pipeline to identify the exact customers likely to churn, calculates the financial ROI of targeting them, and explains why they are leaving using Explainable AI (SHAP).

📸 Dashboard Demonstrations
1. Individual Customer Risk & SHAP Explanation
(Drag and drop your shap_waterfall_demo.png here using the GitHub UI)

Instead of a black-box prediction, the model uses a SHAP Waterfall plot to show the business team exactly why a user is churning (e.g., Month-to-month contract adds +0.57 to churn risk).

2. Dynamic Cost-Benefit Threshold
(Drag and drop your roi_threshold_demo.png here)

A dynamic slider allowing business stakeholders to adjust the risk threshold. Shows in real-time how targeting fewer people lowers campaign costs but might reduce Net ROI.

3. Model Benchmarking
<img width="1894" height="902" alt="shap_waterfall_demo" src="https://github.com/user-attachments/assets/096be406-a762-45cc-a155-db5df2aac9b7" />


XGBoost was selected over Logistic Regression, Random Forest, and LightGBM due to its superior 84.16% AUC-ROC on imbalanced data.

🛠️ Tech Stack
Data Pipeline: SQLite, SQL Queries, Pandas
Machine Learning: XGBoost, Scikit-Learn, Class Imbalance Handling
Explainability: SHAP (TreeExplainer, Waterfall Plots)
Deployment: Streamlit Cloud, GitHub Actions

🚀 How to Run
Clone the repo: git clone <your-link>
Install dependencies: pip install -r requirements.txt
Run app: streamlit run app.py

🔗 Live Application
Click here to view the live interactive dashboard
