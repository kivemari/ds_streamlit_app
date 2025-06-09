import streamlit as st
import pandas as pd
import joblib
import numpy as np
from preprocessing import add_features


# Load model
model = joblib.load("model/model.joblib")

# Title
st.title("Stroke Risk Prediction App")

# Form inputs
st.header("Enter Patient Details")
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 0, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Submit button
if st.button("Predict"):
    # Prepare input as a DataFrame
    input_df = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status]
    })

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Display results
    st.subheader("Prediction Result")
    st.write("💡 Stroke Risk:", "Yes" if prediction == 1 else "No")
    st.write(f"📊 Probability of Stroke: {probability:.2f}")

