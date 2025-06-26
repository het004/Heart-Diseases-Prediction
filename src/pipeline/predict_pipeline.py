import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            target_encoder_path = os.path.join("artifacts", "target_encoder.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            target_encoder = load_object(file_path=target_encoder_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            # Decode predictions to original labels (e.g., "Yes"/"No")
            decoded_preds = target_encoder.inverse_transform(preds.astype(int))
            return decoded_preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 age: int,
                 blood_pressure: float,
                 cholesterol_level: float,
                 bmi: float,
                 sleep_hours: float,
                 triglyceride_level: float,
                 fasting_blood_sugar: float,
                 crp_level: float,
                 homocysteine_level: float,
                 gender: str,
                 exercise_habits: str,
                 smoking: str,
                 family_heart_disease: str,
                 diabetes: str,
                 high_blood_pressure: str,
                 low_hdl_cholesterol: str,
                 high_ldl_cholesterol: str,
                 alcohol_consumption: str,
                 stress_level: str,
                 sugar_consumption: str):

        self.age = age
        self.blood_pressure = blood_pressure
        self.cholesterol_level = cholesterol_level
        self.bmi = bmi
        self.sleep_hours = sleep_hours
        self.triglyceride_level = triglyceride_level
        self.fasting_blood_sugar = fasting_blood_sugar
        self.crp_level = crp_level
        self.homocysteine_level = homocysteine_level
        self.gender = gender
        self.exercise_habits = exercise_habits
        self.smoking = smoking
        self.family_heart_disease = family_heart_disease
        self.diabetes = diabetes
        self.high_blood_pressure = high_blood_pressure
        self.low_hdl_cholesterol = low_hdl_cholesterol
        self.high_ldl_cholesterol = high_ldl_cholesterol
        self.alcohol_consumption = alcohol_consumption
        self.stress_level = stress_level
        self.sugar_consumption = sugar_consumption

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.age],
                "Blood Pressure": [self.blood_pressure],
                "Cholesterol Level": [self.cholesterol_level],
                "BMI": [self.bmi],
                "Sleep Hours": [self.sleep_hours],
                "Triglyceride Level": [self.triglyceride_level],
                "Fasting Blood Sugar": [self.fasting_blood_sugar],
                "CRP Level": [self.crp_level],
                "Homocysteine Level": [self.homocysteine_level],
                "Gender": [self.gender],
                "Exercise Habits": [self.exercise_habits],
                "Smoking": [self.smoking],
                "Family Heart Disease": [self.family_heart_disease],
                "Diabetes": [self.diabetes],
                "High Blood Pressure": [self.high_blood_pressure],
                "Low HDL Cholesterol": [self.low_hdl_cholesterol],
                "High LDL Cholesterol": [self.high_ldl_cholesterol],
                "Alcohol Consumption": [self.alcohol_consumption],
                "Stress Level": [self.stress_level],
                "Sugar Consumption": [self.sugar_consumption]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)