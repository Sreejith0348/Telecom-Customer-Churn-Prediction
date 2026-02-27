# Telecom Customer Churn Prediction using Machine Learning

This project builds and compares multiple supervised machine learning models to predict customer churn. The objective is to identify customers at risk of leaving, enabling proactive retention strategies.

# Problem Statement

Customer churn significantly impacts business revenue.
This project aims to:
Predict whether a customer will churn
Compare multiple ML algorithms
Optimize models using hyperparameter tuning
Evaluate performance using proper classification metrics

# Models Implemented

Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Naive Bayes
LightGBM
CatBoost

# Techniques Used

Data Cleaning & Preprocessing
Feature Encoding
Train-Test Split
SMOTE (Handling Class Imbalance)
Hyperparameter Tuning (RandomizedSearchCV)
ROC-AUC Evaluation
Confusion Matrix Analysis
Feature Importance Analysis

# Project Structure
Telecom-customer-churn/
│
├── data/          # Raw & processed datasets
├── notebooks/     # EDA_and_preprocessing , modeling_building
├── models/           # Saved models
├── outputs/       # figures, reports
├── requirements.txt
└── README.md


# Evaluation Metrics

Accuracy
Precision
Recall
F1-Score
ROC-AUC Score


# Setup
1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Run notebooks step by step

