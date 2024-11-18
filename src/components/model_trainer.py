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
from utils import save_object, evaluate_models

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
            
            logging.info("Scaling the features if required")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            models_requiring_scaling = [
                ('Logistic Regression', LogisticRegression(max_iter=1000)),
                ('XGB Classifier', XGBClassifier())
            ]

            models_not_requiring_scaling = [
                ('Random Forest Classifier', RandomForestClassifier(n_estimators=100)),
                ('Decision Tree Classifier', DecisionTreeClassifier()),
                ('XGB Classifier', XGBClassifier()),
                ('LGB Classifier', lgb.LGBMClassifier())
            ]
            
            # Evaluate models requiring scaling
            logging.info("Evaluating models requiring scaling")
            results_requiring_scaling = evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test, models_requiring_scaling)
            
            # Evaluate models without scaling
            logging.info("Evaluating models not requiring scaling")
            results_no_scaling = evaluate_models(X_train, y_train, X_test, y_test, models_not_requiring_scaling)

            # Combine results from both lists
            results = results_requiring_scaling + results_no_scaling

            # Create a DataFrame from results
            results_df = pd.DataFrame(results)
            logging.info(f"Model evaluation results: \n{results_df}")
            
            best_model_name = results_df.loc[results_df['Model Accuracy'].idxmax(), 'Model Name']
            best_model_accuracy = results_df['Model Accuracy'].max()

            logging.info(f"Best model: {best_model_name} with accuracy: {best_model_accuracy}")

            ''' 
            # Dictionary to map model names to their corresponding classes
            model_dict = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'XGB Classifier': XGBClassifier(),
                'Random Forest Classifier': RandomForestClassifier(),
                'Decision Tree Classifier': DecisionTreeClassifier(),
                'LGB Classifier': lgb.LGBMClassifier()
            }

            # Get the best model from the dictionary
            best_model = model_dict.get(best_model_name)

            # Save the best model
            if best_model:
                # Train the best model before saving, as untrained models won’t have the necessary attributes
                logging.info(f"Training the best model before saving: {best_model_name}")
                # Train the best model with the appropriate data
                if best_model_name in ['Logistic Regression', 'XGB Classifier']:  # models requiring scaling
                    best_model.fit(X_train_scaled, y_train)
                else:  # models not requiring scaling
                    best_model.fit(X_train, y_train)
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path, 
                    obj=best_model
                )
            else:
                logging.error(f"Model {best_model_name} not found in the dictionary")
            '''
            save_object(
                    file_path=self.model_trainer_config.trained_model_file_path, 
                    obj=best_model_name
                )            
            return best_model_accuracy
            
        except Exception as e:
            raise CustomException(e,sys)
        