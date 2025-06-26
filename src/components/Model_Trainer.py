import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path=None):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "KNeighbors-Neighbors Classifier": KNeighborsClassifier(),
                "DecisionTree": DecisionTreeClassifier(random_state=42),
                "RandomForest": RandomForestClassifier(random_state=42),
                #"SupportVector Machine": SVC(probability=True, random_state=42),
                "XGBoost": XGBClassifier(use_label_encoder=False,eval_metric='logloss', random_state=42),
                "CatBoostClassifier": CatBoostClassifier(verbose=False, random_state=42),
                "AdaBoostClassifier": AdaBoostClassifier(random_state=42)
            }

            params = {
                "Logistic Regression": {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
                },
                "KNeighborsClassifier": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
                },
                "DecisionTree": {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30]
                },
                "RandomForest": {
                'n_estimators': [8, 16, 32, 64, 128],
                'max_depth': [None, 10, 20]
                },
                "XGBoost": {
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200]
                },
                "CatBoostClassifier": {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [50, 100]
                },
                "AdaBoostClassifier": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1]
    }
}


            # Evaluate models and get fitted models
            model_report, fitted_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Get best model score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = fitted_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model: {best_model_name} with accuracy: {best_model_score}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            # Predict and compute accuracy
            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            logging.info(f"Test set accuracy: {accuracy}")

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)