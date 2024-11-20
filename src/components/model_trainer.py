from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models, tune_model_hyperparameters

import os
import sys
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','best_model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            # Models that require feature scaling
            models_requiring_scaling = [
                ('Logistic Regression', LogisticRegression(max_iter=1000)),
                ('XGB Classifier (scaled)', XGBClassifier())
            ]
            # Models that do not require feature scaling
            models_not_requiring_scaling = [
                ('Random Forest Classifier', RandomForestClassifier(n_estimators=100)),
                ('Decision Tree Classifier', DecisionTreeClassifier()),
                ('XGB Classifier (no scaling)', XGBClassifier()),
                ('LGB Classifier', lgb.LGBMClassifier())
            ]
            
            logging.info("Scaling the features for models that require it")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Evaluate models requiring scaling
            logging.info("Evaluating models requiring scaling")
            results_requiring_scaling = evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test, models_requiring_scaling)

            # Evaluate models not requiring scaling
            logging.info("Evaluating models not requiring scaling")
            results_no_scaling = evaluate_models(X_train, y_train, X_test, y_test, models_not_requiring_scaling)

            # Combine results
            all_results = results_requiring_scaling + results_no_scaling

            # Filter models with accuracy >= 86%
            top_models = [result for result in all_results if result['accuracy'] >= 0.86]
            logging.info(f"Top-performing models (>= 86% accuracy): {top_models}")
            
            # Hyperparameters for models
            params = {
                "Logistic Regression": {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
                "XGB Classifier (scaled)": {"learning_rate": [0.01, 0.1, 0.2], "n_estimators": [50, 100, 200]},
                "Random Forest Classifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
                "Decision Tree Classifier": {"criterion": ["gini", "entropy"], "max_depth": [None, 10, 20]},
                "XGB Classifier (no scaling)": {"learning_rate": [0.01, 0.1, 0.2], "n_estimators": [50, 100, 200]},
                "LGB Classifier": {"num_leaves": [31, 50, 100], "learning_rate": [0.01, 0.1, 0.2], "n_estimators": [50, 100, 200]},
            }
            # Extract model names for easier checking
            models_requiring_scaling_names = [model[0] for model in models_requiring_scaling]
            models_not_requiring_scaling_names = [model[0] for model in models_not_requiring_scaling]
            
            # Variable to store details of the best model after tuning
            best_model_details = {
                "model_name": None,
                "best_params": None,
                "model_object": None,
                "accuracy": 0
            }
            
            # Tune top models only
            for top_model in top_models:
                model_name = top_model['model']
                logging.info(f"Tuning hyperparameters for: {model_name}")
                
                # Check if the model requires scaling
                if model_name in models_requiring_scaling_names:
                    logging.info(f"Using scaled data for tuning {model_name}")
                    X_train_used, y_train_used, X_test_used = X_train_scaled, y_train, X_test_scaled
                elif model_name in models_not_requiring_scaling_names:
                    logging.info(f"Using non-scaled data for tuning {model_name}")
                    X_train_used, y_train_used, X_test_used = X_train, y_train, X_test
                else:
                    logging.warning(f"Unknown model scaling requirement for: {model_name}. Skipping tuning.")
                    continue  # Skip tuning if the model's scaling requirement is unknown

                # Tune hyperparameters if grid is defined
                if model_name in params:
                    tuned_model = tune_model_hyperparameters(top_model['model_object'], params[model_name], X_train_used, y_train_used)
                    y_test_pred = tuned_model.predict(X_test_used)
                    accuracy = accuracy_score(y_test, y_test_pred)

                    logging.info(f"Tuned model: {model_name}, Accuracy: {accuracy}, Params: {tuned_model.get_params()}")

                    # Update best model details if current model has higher accuracy
                    if accuracy > best_model_details["accuracy"]:
                        best_model_details.update({
                            "model_name": model_name,
                            "best_params": tuned_model.get_params(),
                            "model_object": tuned_model,
                            "accuracy": accuracy
                        })
                else:
                    logging.warning(f"No hyperparameter grid defined for: {model_name}")
        
            # Save the best model
            logging.info(f"Saving the best model {best_model_details}")
            save_object(self.model_trainer_config.trained_model_file_path, best_model_details)      
            
            return best_model_details
            
        except Exception as e:
            raise CustomException(e,sys)
        