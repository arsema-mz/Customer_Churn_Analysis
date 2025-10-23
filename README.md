# 🧠 Customer Churn Analysis and Prediction

This project aims to predict customer churn using machine learning techniques. The dataset includes customer demographics, service usage patterns, and account information. By identifying customers at risk of leaving, businesses can take preventive actions to improve retention.

## 🎯 Objectives

* Perform exploratory data analysis (EDA) to uncover churn patterns.
* Engineer relevant features (demographics, usage behavior, billing, etc.).
* Train and evaluate machine learning models for churn prediction.

## 🧩 Tasks Overview
| Task | Description |
|------|--------------|
| **1. Data Preprocessing** | Cleaned missing values, encoded categorical features, and scaled numerical columns. |
| **2. Exploratory Data Analysis (EDA)** | Visualized churn distribution, correlation heatmaps, and service usage patterns. |
| **3. Feature Selection** | Selected top 10 predictive features based on correlation and importance ranking. |
| **4. Model Selection** | Compared Logistic Regression, Random Forest, and XGBoost models. |
| **5. Model Training** | Trained final model on selected features and saved as `final_gradient_boosting_model.pkl`. |
| **6. Model Evaluation** | Evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC. |

## 🧮 Model Performance
| Metric | Score |
|--------|--------|
| Accuracy | **0.795** |
| Precision | **0.645** |
| Recall | **0.499** |
| F1-Score | **0.562** |
| ROC-AUC | **0.840** |

## 🧰 Tools and Libraries
- Python 3.x  
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  
- joblib  

## 🚀 How to Run
1. Clone this repository  
   ```bash
   git clone https://github.com/arsema-mz/customer-churn-analysis.git
   cd customer-churn-prediction

