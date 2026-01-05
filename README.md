# Customer Churn Prediction using Amazon SageMaker (XGBoost)

This project implements an **end-to-end customer churn prediction system** using **Amazon SageMaker** and **XGBoost (Script Mode)**.  
The goal is to predict whether a customer will churn based on behavioral, transactional, and demographic features.

---

## 🚀 Project Overview

- **Problem**: Binary classification – predict customer churn (Yes / No)
- **Dataset**: E-commerce customer data (Kaggle)
- **Model**: XGBoost (binary logistic)
- **Platform**: Amazon SageMaker
- **Deployment**: Real-time inference endpoint
- **Status**: Fully working, end-to-end pipeline


---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **XGBoost**
- **Amazon SageMaker**
- **Amazon S3**
- **Boto3**

---

## 📊 Dataset Details

Key features include:
- Tenure
- Preferred login device
- Payment mode
- Order history
- Satisfaction score
- Complaint history
- Cashback amount

---

## ⚙️ Implementation Steps

1. Load and clean dataset
2. Handle missing values
3. Reorder target column for XGBoost compatibility
4. Upload processed data to S3
5. Train XGBoost model using SageMaker Script Mode
6. Deploy real-time endpoint
7. Perform inference on sample data
8. Delete endpoint to avoid billing

---

## ✅ Results

- Successfully trained a churn prediction model
- Deployed a live inference endpoint
- Model achieved strong predictive performance (validated during training)
- Fully reproducible and cloud-scalable solution

