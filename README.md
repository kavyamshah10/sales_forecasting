# 📊 Sales Forecasting System

## 🔥 Project Overview
This project is an end-to-end **Sales Forecasting System** built using Machine Learning and Streamlit.

It performs:
- Model training and evaluation using historical data
- Comparison of multiple ML models
- Forecasting future sales (2013) using time-series techniques
- Interactive visualization through a Streamlit app

---

## 🚀 Features

### 📈 1. Model Evaluation (2012)
- Automatic comparison of **Actual vs Predicted Sales**
- Displays performance metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score
- Weekly aggregated sales visualization

---

### 🔮 2. Sales Forecasting (2013)
- Predicts future sales using **recursive forecasting**
- User can:
  - Select Store
  - Select number of weeks
- Outputs:
  - Forecast graph
  - Prediction table
  - Downloadable CSV file

---

## 🧠 Machine Learning Approach

### 🔍 Models Compared
The following models were trained and evaluated:

- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  

---

### 📊 Evaluation Metrics (Model Selection)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

The best model was selected based on lowest prediction error.

---

### ✅ Final Model Selected
**XGBoost Regressor** was chosen as the final model due to its superior performance compared to other models.

---

### 📈 Metrics in App
In addition to MAE and RMSE, the app also displays:

- R² Score → Indicates how well the model explains variance in data

---

## 🔧 Feature Engineering

- Time-based features:
  - Year, Month, Week, Day, Day_of_week

- Lag features:
  - Lag_1 (previous week sales)
  - Lag_7 (previous 7 weeks sales)

- Rolling statistics:
  - Rolling_mean_7 (7-week average)

---

## 🔁 Forecasting Technique

The project uses **Recursive Forecasting**:

1. Predict next week sales  
2. Add prediction to dataset  
3. Use updated data for next prediction  
4. Repeat for multiple future weeks  

---

APP:[]