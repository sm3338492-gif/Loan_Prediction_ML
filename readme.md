# Loan Approval Prediction using Machine Learning

## 📌 Overview
This project is a simple Machine Learning model that predicts whether a loan should be approved or not based on user financial data.

## 🎯 Objective
To build a classification model that helps in decision-making for loan approval using basic ML techniques.

## 🧠 Model Used
- Logistic Regression

## 📊 Features Used
- Annual Salary
- Bank Balance
- Employment Status

## ⚙️ Techniques Applied
- Data Preprocessing
- Feature Engineering
- Feature Scaling (StandardScaler)
- Train-Test Split
- Model Evaluation

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

## 🔍 Output
- **1 → Loan Approved**
- **0 → Loan Not Approved**

## 🚀 How It Works
1. User inputs salary, bank balance, and employment status
2. Data is scaled using StandardScaler
3. Model predicts probability of approval
4. Final decision is made based on threshold

## 💡 Key Learnings
- Importance of data preprocessing and scaling
- Handling imbalanced datasets
- Understanding evaluation metrics
- Building a complete ML pipeline

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn
python loan_prediction.py
