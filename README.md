## 🫀 Heart Disease Prediction
A comprehensive machine learning project that predicts the likelihood of heart disease based on patient health metrics. This project features an end-to-end machine learning pipeline with a user-friendly Streamlit web application for real-time predictions.
## 🌟 Project Overview
This project uses advanced machine learning algorithms to predict heart disease risk based on 20 different health parameters including demographic information, clinical measurements, lifestyle factors, and biomarkers. The system is designed to assist healthcare professionals and individuals in early heart disease risk assessment.
## 🎯 Key Features

Multiple ML Models: Implements 7 different machine learning algorithms with hyperparameter tuning
Interactive Web App: Streamlit-based user interface for easy prediction
End-to-End Pipeline: Complete MLOps pipeline from data ingestion to model deployment
Comprehensive EDA: Detailed exploratory data analysis with visualizations
Real-time Predictions: Instant heart disease risk assessment with probability scores
Error Handling: Robust exception handling and logging system
Model Persistence: Automated model saving and loading capabilities

## 🏗️ Project Architecture
##📊 Dataset Information
The dataset contains 21 features for heart disease prediction:
Numerical Features (9):

Age: Patient age in years
Blood Pressure: Systolic blood pressure (mmHg)
Cholesterol Level: Total cholesterol (mg/dL)
BMI: Body Mass Index
Sleep Hours: Hours of sleep per night
Triglyceride Level: Triglycerides (mg/dL)
Fasting Blood Sugar: Fasting glucose (mg/dL)
CRP Level: C-Reactive Protein (mg/L)
Homocysteine Level: Homocysteine (µmol/L)

Categorical Features (11):

Gender: Male/Female
Exercise Habits: Low/Medium/High
Smoking: Yes/No
Family Heart Disease: Yes/No
Diabetes: Yes/No
High Blood Pressure: Yes/No
Low HDL Cholesterol: Yes/No
High LDL Cholesterol: Yes/No
Alcohol Consumption: Low/Medium/High
Stress Level: Low/Medium/High
Sugar Consumption: Low/Medium/High

Target Variable:

Heart Disease Status: Yes/No (binary classification)

## 🤖 Machine Learning Models
The project implements and compares 7 different algorithms:

Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree Classifier
Random Forest Classifier
XGBoost Classifier
CatBoost Classifier
AdaBoost Classifier

Each model undergoes hyperparameter tuning using GridSearchCV, and the best-performing model is automatically selected and saved.
## 🛠️ Installation & Setup
Prerequisites

Python 3.8 or higher
pip package manager

Installation Steps

Clone the repository

git clone https://github.com/het004/Heart-Diseases-Prediction.git
cd Heart-Diseases-Prediction


Create virtual environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Install the package

pip install -e .

## 🚀 Usage
Running the Streamlit Web Application
streamlit run app.py

The application will be available at http://localhost:8501
Using the Web Interface

Open the Streamlit app in your browser
Fill in patient information:
Adjust numerical sliders for health metrics
Select appropriate categorical values


Click "Predict Heart Disease Status"
View results:
Prediction: Yes/No
Probability percentage
Risk assessment message



Training the Model
To retrain the model with new data:
python src/components/Data_ingestion.py

This will:

Load and split the dataset
Perform data preprocessing
Train multiple models
Select the best performer
Save model artifacts

Making Predictions Programmatically
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Create patient data
patient_data = CustomData(
    age=45,
    blood_pressure=140.0,
    cholesterol_level=220.0,
    bmi=28.5,
    sleep_hours=6.5,
    triglyceride_level=180.0,
    fasting_blood_sugar=110.0,
    crp_level=3.2,
    homocysteine_level=12.0,
    gender="Male",
    exercise_habits="Low",
    smoking="Yes",
    family_heart_disease="Yes",
    diabetes="No",
    high_blood_pressure="Yes",
    low_hdl_cholesterol="Yes",
    high_ldl_cholesterol="Yes",
    alcohol_consumption="Medium",
    stress_level="High",
    sugar_consumption="High"
)

# Make prediction
pipeline = PredictPipeline()
prediction = pipeline.predict(patient_data.get_data_as_data_frame())
print(f"Heart Disease Prediction: {prediction[0]}")

## 📈 Model Performance
The project automatically evaluates all models using:

Accuracy Score
Precision Score
Recall Score
F1 Score

The best model is selected based on accuracy, with a minimum threshold of 60% required for deployment.
## 🔍 Exploratory Data Analysis
Comprehensive EDA is available in the notebooks/EDA Heart Disease.ipynb notebook, including:

Data distribution analysis
Correlation matrices
Feature importance analysis
Visualization of relationships between variables
Statistical summaries

## 📁 File Descriptions
Core Files

app.py: Main Streamlit application with interactive UI
requirements.txt: All project dependencies
setup.py: Package configuration and metadata

Source Code (src/)

exception.py: Custom exception handling for better error management
logger.py: Logging configuration for tracking application behavior
utils.py: Utility functions for model evaluation and object persistence

Components (src/components/)

Data_ingestion.py: Handles data loading and train/test splitting
Data_transformation.py: Feature engineering and preprocessing pipeline
Model_Trainer.py: Model training, evaluation, and selection logic

Pipelines (src/pipeline/)

predict_pipeline.py: Prediction pipeline with CustomData class
train_pipeline.py: Training pipeline orchestration

## 🎨 Web Interface Features
The Streamlit application provides:

Intuitive Input Forms: Organized sliders and dropdowns for easy data entry
Real-time Validation: Input validation and error handling
Interactive Results: Clear prediction display with probability scores
Risk Assessment: Contextual health recommendations
Responsive Design: Works on desktop and mobile devices

## 🔧 Technical Stack

Frontend: Streamlit
Backend: Python
ML Libraries: scikit-learn, XGBoost, CatBoost
Data Processing: pandas, numpy
Visualization: matplotlib, seaborn, plotly
Model Persistence: pickle, dill
Web Framework: HTML/CSS (Bootstrap)

## 📊 Data Pipeline

Data Ingestion: Load dataset and split into train/test sets
Data Transformation:
Handle missing values
Encode categorical variables
Scale numerical features
Feature engineering


Model Training:
Train multiple algorithms
Hyperparameter tuning
Model evaluation
Best model selection


Model Deployment: Save artifacts and deploy via Streamlit

## 🚨 Error Handling
The project includes comprehensive error handling:

Custom Exception Classes: Detailed error messages with file and line information
Input Validation: Ensures valid data types and ranges
Model Loading Checks: Verifies model artifacts exist before prediction
Graceful Degradation: User-friendly error messages in the web interface

## 📝 Logging
Comprehensive logging system tracks:

Data ingestion progress
Model training metrics
Prediction requests
Error occurrences
Performance statistics

## 🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
👨‍💻 Author
Het Shah

Email: shahheta1973@gmail.com
GitHub: @het004

## 🙏 Acknowledgments

Dataset contributors and medical research community
Open source machine learning libraries
Streamlit for the excellent web framework
Healthcare professionals for domain expertise

## 📞 Support
If you encounter any issues or have questions:
Contact the author via email

## ⚠️ Medical Disclaimer
This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.
