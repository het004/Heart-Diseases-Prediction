import streamlit as st
import pandas as pd
import numpy as np
import os
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.utils import load_object

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Title and description
st.title("Heart Disease Prediction App")
st.write("Enter the patient details below to predict the likelihood of heart disease. This tool uses a machine learning model to assess risk based on numerical and categorical health metrics.")

# Note about categorical values
st.info("Please select valid categories as per the training data to avoid errors.")

# Create a form for user input
with st.form("heart_disease_form"):
    st.header("Patient Information")

    # Organize inputs in two columns for numerical features
    st.subheader("Numerical Features")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (years)", min_value=18, max_value=100, value=50, step=1)
        blood_pressure = st.slider("Blood Pressure (mmHg)", min_value=80.0, max_value=200.0, value=120.0, step=0.5)
        cholesterol_level = st.slider("Cholesterol Level (mg/dL)", min_value=100.0, max_value=400.0, value=200.0, step=0.5)
        bmi = st.slider("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        sleep_hours = st.slider("Sleep Hours per Night", min_value=2.0, max_value=12.0, value=7.0, step=0.5)

    with col2:
        triglyceride_level = st.slider("Triglyceride Level (mg/dL)", min_value=50.0, max_value=500.0, value=150.0, step=0.5)
        fasting_blood_sugar = st.slider("Fasting Blood Sugar (mg/dL)", min_value=50.0, max_value=200.0, value=90.0, step=0.5)
        crp_level = st.slider("CRP Level (mg/L)", min_value=0.0, max_value=15.0, value=2.0, step=0.1)
        homocysteine_level = st.slider("Homocysteine Level (µmol/L)", min_value=5.0, max_value=30.0, value=10.0, step=0.1)

    # Categorical inputs in two columns
    st.subheader("Categorical Features")
    col3, col4 = st.columns(2)

    with col3:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        exercise_habits = st.selectbox("Exercise Habits", options=["Low", "Medium", "High"])
        smoking = st.selectbox("Smoking", options=["Yes", "No"])
        family_heart_disease = st.selectbox("Family Heart Disease", options=["Yes", "No"])
        diabetes = st.selectbox("Diabetes", options=["Yes", "No"])
        high_blood_pressure = st.selectbox("High Blood Pressure", options=["Yes", "No"])

    with col4:
        low_hdl_cholesterol = st.selectbox("Low HDL Cholesterol", options=["Yes", "No"])
        high_ldl_cholesterol = st.selectbox("High LDL Cholesterol", options=["Yes", "No"])
        alcohol_consumption = st.selectbox("Alcohol Consumption", options=["Low", "Medium", "High"])
        stress_level = st.selectbox("Stress Level", options=["Low", "Medium", "High"])
        sugar_consumption = st.selectbox("Sugar Consumption", options=["Low", "Medium", "High"])

    # Submit button
    submitted = st.form_submit_button("Predict Heart Disease Status")

# Process prediction when form is submitted
if submitted:
    try:
        # Create CustomData instance
        data = CustomData(
            age=age,
            blood_pressure=blood_pressure,
            cholesterol_level=cholesterol_level,
            bmi=bmi,
            sleep_hours=sleep_hours,
            triglyceride_level=triglyceride_level,
            fasting_blood_sugar=fasting_blood_sugar,
            crp_level=crp_level,
            homocysteine_level=homocysteine_level,
            gender=gender,
            exercise_habits=exercise_habits,
            smoking=smoking,
            family_heart_disease=family_heart_disease,
            diabetes=diabetes,
            high_blood_pressure=high_blood_pressure,
            low_hdl_cholesterol=low_hdl_cholesterol,
            high_ldl_cholesterol=high_ldl_cholesterol,
            alcohol_consumption=alcohol_consumption,
            stress_level=stress_level,
            sugar_consumption=sugar_consumption
        )

        # Convert to DataFrame
        data_df = data.get_data_as_data_frame()

        # Initialize prediction pipeline
        pipeline = PredictPipeline()

        # Make prediction
        prediction = pipeline.predict(data_df)

        # Get prediction probability
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Model or preprocessor file not found in artifacts folder.")
        model = load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)
        data_scaled = preprocessor.transform(data_df)
        prob = model.predict_proba(data_scaled)[0]
        prob_yes = prob[1] * 100  # Probability of "Yes" (heart disease)

        # Display result
        st.success(f"Prediction: **{prediction[0]}**")
        st.write(f"Probability of Heart Disease: **{prob_yes:.1f}%**")
        if prediction[0] == "Yes":
            st.warning("The model predicts a risk of heart disease. Please consult a healthcare professional for further evaluation.")
        else:
            st.info("The model predicts no risk of heart disease. Continue maintaining a healthy lifestyle.")

    except ValueError as ve:
        st.error(f"Input Error: {str(ve)}")
        st.warning("Ensure all categorical inputs match the training data categories and numerical inputs are within valid ranges.")
    except FileNotFoundError as fnfe:
        st.error(f"File Error: {str(fnfe)}")
        st.warning("Please ensure the model and preprocessor files are present in the artifacts folder. Run the training pipeline to generate them.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.warning("This could be due to a model issue or data mismatch. Check the logs and ensure the training pipeline has been run successfully.")

# Add footer
st.markdown("---")
st.markdown("**Heart Disease Prediction Model** | Built with Streamlit | © 2025")