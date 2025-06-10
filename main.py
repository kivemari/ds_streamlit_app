import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from preprocessing import add_features

model = joblib.load("model/model.joblib")

st.title("Stroke Risk Prediction App")

st.header("Enter Patient Details")
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 18, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level mg/dL", min_value=50.0, max_value=300.0, value=100.0)
st.markdown(
    "🔗 **Need to convert glucose values from mmol/L to mg/dL?** "
    "[Use this online converter](https://www.diabetes.co.uk/blood-sugar-converter.html)"
)
bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=22.0)
st.markdown(
    "🔗 **Prefer to calculate it yourself?** "
    "[Use the NIH BMI Calculator](https://www.nhlbi.nih.gov/calculate-your-bmi)"
)
smoking_status = st.selectbox("Smoking Status", ["never_smoked", "formerly_smoked", "smokes", "unknown"])

if st.button("Predict"):
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

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write("💡 Stroke Risk:", "Yes" if prediction == 1 else "No")
    st.write(f"📊 Probability of Stroke: {probability:.2f}")
