import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        model_report = {}
        fitted_models = {}
        for model_name, model in models.items():
            logging.info(f"Training {model_name}")
            # Get hyperparameters
            model_params = param.get(model_name, {})
            # Perform GridSearchCV
            gs = GridSearchCV(
                model,
                model_params,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                error_score=0
            )
            # Fit model
            gs.fit(X_train, y_train)
            # Store best model
            fitted_models[model_name] = gs.best_estimator_
            # Predict on test set
            y_pred = fitted_models[model_name].predict(X_test)
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            model_report[model_name] = accuracy
            # Log metrics
            precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            logging.info(
                f"{model_name} - Accuracy: {accuracy:.4f}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            )
        return model_report, fitted_models
    except Exception as e:
        raise CustomException(e)

def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e)
