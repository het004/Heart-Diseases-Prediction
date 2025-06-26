import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Define the configuration for data transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    target_encoder_file_path = os.path.join('artifacts', 'target_encoder.pkl')

class DataTransformation:
    '''
    This class is responsible for data transformation.
    '''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function creates the preprocessing object for data transformation.
        '''
        try:
            # Define numerical and categorical columns
            numerical_columns = [
                'Age', 'Blood Pressure', 'Cholesterol Level', 
                'BMI', 'Sleep Hours', 'Triglyceride Level', 
                'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level'
            ]
            categorical_columns = [
                'Gender', 'Exercise Habits', 'Smoking', 
                'Family Heart Disease', 'Diabetes', 
                'High Blood Pressure', 'Low HDL Cholesterol', 
                'High LDL Cholesterol', 'Alcohol Consumption', 
                'Stress Level', 'Sugar Consumption'
            ]

            # Pipeline for numerical columns: impute missing values with median, then scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical columns: impute missing values with mode, one-hot encode, then scale
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns scaling completed")
            logging.info("Categorical columns encoding and scaling completed")

            # Combine the numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function applies the data transformation to the training and test datasets.
        
        Parameters:
        - train_path (str): Path to the training dataset CSV file.
        - test_path (str): Path to the test dataset CSV file.
        
        Returns:
        - tuple: (train_arr, test_arr, preprocessor_obj_file_path, target_encoder_file_path)
        '''
        try:
            # Read the training and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")

            # Get the preprocessing object for features
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target column and feature columns
            target_column_name = "Heart Disease Status"
            numerical_columns = [
                'Age', 'Blood Pressure', 'Cholesterol Level', 
                'BMI', 'Sleep Hours', 'Triglyceride Level', 
                'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level'
            ]
            categorical_columns = [
                'Gender', 'Exercise Habits', 'Smoking', 
                'Family Heart Disease', 'Diabetes', 
                'High Blood Pressure', 'Low HDL Cholesterol', 
                'High LDL Cholesterol', 'Alcohol Consumption', 
                'Stress Level', 'Sugar Consumption'
            ]

            # Separate features and target for training and test sets
            input_train_feature = train_df.drop(columns=[target_column_name], axis=1)
            target_train_feature = train_df[target_column_name]

            input_test_feature = test_df.drop(columns=[target_column_name], axis=1)
            target_test_feature = test_df[target_column_name]

            # Apply preprocessing to training and test features
            logging.info("Applying preprocessing object on training and testing datasets")
            input_train_feature_arr = preprocessing_obj.fit_transform(input_train_feature)
            input_test_feature_arr = preprocessing_obj.transform(input_test_feature)

            # Encode the target variable
            logging.info("Encoding target variable")
            target_encoder = LabelEncoder()
            target_train_feature_encoded = target_encoder.fit_transform(target_train_feature)
            target_test_feature_encoded = target_encoder.transform(target_test_feature)

            # Combine the transformed features with the encoded target variable
            train_arr = np.c_[
                input_train_feature_arr, np.array(target_train_feature_encoded)
            ]
            test_arr = np.c_[
                input_test_feature_arr, np.array(target_test_feature_encoded)
            ]

            # Save the preprocessing object
            logging.info("Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Save the target encoder
            logging.info("Saved target encoder")
            save_object(
                file_path=self.data_transformation_config.target_encoder_file_path,
                obj=target_encoder
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.target_encoder_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)