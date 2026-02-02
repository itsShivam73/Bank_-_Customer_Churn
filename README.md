✅ Bank Customer Churn Prediction using Machine Learning
📌 Project Overview
Customer churn is one of the biggest challenges faced by banks today. Losing customers directly impacts revenue and business growth. This project focuses on predicting whether a customer will churn (leave the bank) based on their demographic and account-related information.
The goal is to build a machine learning model that can identify at-risk customers early, allowing the bank to take proactive retention actions.

🎯 Problem Statement
The objective of this project is:
To predict customer churn in a banking system
To compare multiple machine learning algorithms
To improve churn detection performance using threshold tuning
To evaluate models using precision, recall, F1-score, confusion matrix, and precision-recall curve

📂 Dataset Information
The dataset contains customer-related features such as:
Credit score
Geography
Gender
Age
Tenure
Balance
Number of products
Estimated salary
Churn label (Target variable)

Target variable:
0 → Customer stays
1 → Customer churns

⚙️ Machine Learning Models Used
Three different algorithms were trained and evaluated:
Model	Type
Logistic Regression	Baseline Linear Model
Random Forest	Bagging Ensemble Model
XGBoost Classifier	Boosting Ensemble Model

This combination provides a strong comparison across different ML families.

🧪 Model Evaluation Metrics
Since churn prediction is an imbalanced classification problem, accuracy alone is not sufficient. The following metrics were used:
Precision
Recall
F1-score
Confusion Matrix
Precision–Recall Curve
Threshold tuning for business optimization

📌 Confusion Matrix Comparison
Confusion matrices were plotted for all three models:
Logistic Regression
Random Forest
XGBoost
This helped analyze:
True churn predictions
Missed churn customers (False Negatives)
Wrong churn alerts (False Positives)

🔥 Threshold Tuning (Business Optimization)
In churn prediction, the cost of missing a churn customer is higher than falsely flagging a loyal customer.
Instead of using the default threshold (0.5), threshold tuning was applied:
y_prob = best_model_xgb.predict_proba(x_test_scaled)[:,1]
y_pred = (y_prob > 0.35).astype(int)

✅ Result Improvement
Threshold tuning significantly improved churn detection:
Recall improved from 66% → 80%
Missed churn customers reduced from 129 → 78
True churn detections increased from 264 → 315

​
This makes the model more useful for real banking retention strategies.
📉 Precision–Recall Curve Analysis
A Precision–Recall Curve was plotted to evaluate performance under class imbalance.
This curve helped select the optimal threshold based on recall–precision tradeoff.

📊 Key Insights
Logistic Regression captured more churn customers but produced many false churn alerts.
Random Forest and XGBoost provided better balance.
XGBoost + Threshold tuning achieved the best churn recall performance.
Precision–Recall analysis is more reliable than accuracy in churn prediction problems.

🚀 Project Workflow
Data Cleaning and Preprocessing
Feature Scaling and Encoding
Model Training (LR, RF, XGB)
Hyperparameter Tuning
Confusion Matrix Visualization
Precision–Recall Curve Analysis
Threshold Optimization for Churn Recall


📌 Conclusion
This project demonstrates an end-to-end machine learning pipeline for churn prediction with real-world business optimization. By applying threshold tuning and precision–recall analysis, the model achieves improved churn detection, making it highly valuable for banking applications.

👤 Author
Shivam Pandey
Data Science Student | Machine Learning Enthusiast

Suggestions and improvements are always welcome!
