import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('hypertension_model0.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Hypertension Prediction App")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 120, 30)
diabetes = st.selectbox("Diabetes", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking = st.selectbox("Smoking History", ["never", "former", "current"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=500, value=100)

# Convert categorical inputs
gender = 1 if gender == "Male" else 0
smoking_map = {"never": 0, "former": 1, "current": 2}
smoking = smoking_map[smoking]

# Predict
input_data = np.array([[gender, age, diabetes, heart_disease, smoking, bmi, hba1c, glucose]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

st.subheader("Prediction Result")
if prediction == 1:
    st.error("You may have Hypertension.")
else:
    st.success("You are likely healthy.")
