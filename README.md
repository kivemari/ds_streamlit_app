# ds_streamlit_app

# 🧠 Stroke Risk Prediction App

This project is a web-based application that predicts the risk of stroke using patient health data. It uses an ensemble machine learning model built with Logistic Regression, Random Forest, and XGBoost.

## 🚀 Demo

Try it live on [Streamlit Cloud]()

## 🔍 Features

- Predict stroke risk based on medical features
- Uses SMOTE to handle class imbalance
- Ensemble learning with tuned models
- Built with `scikit-learn`, `XGBoost`, `Streamlit`, and `imblearn`

## 📁 Project Structure

stroke-predictor-app/
├── main.py # Streamlit web app
├── model/
│ └── model.joblib # Trained ensemble model
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── .gitignore
