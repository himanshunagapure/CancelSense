import dill 
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException

"""# Dynamically add 'src' to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)"""

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.

    Args:
        file_path (str): Path to save the object.
        obj: Python object to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise Exception(f"Error saving object to {file_path}: {e}")
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the given list of models and return their accuracies on the test data.

    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        X_test (DataFrame): Test features
        y_test (Series): Test target
        models (list of tuples): List of model names and objects

    Returns:
        list of dict: List of results containing model name, object, and accuracy
    """
    try:
        results = []
        for name, model in models:
            logging.info(f"Training and evaluating model: {name}")
            model.fit(X_train, y_train)  # Train the model
            y_pred = model.predict(X_test)  # Predict on test data
            accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
            logging.info(f"Model: {name}, Accuracy: {accuracy}")
            
            # Store model results
            results.append({
                "model": name,
                "model_object": model,
                "accuracy": accuracy
            })

        # Sort results by accuracy in descending order
        results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
        # Log the sorted results as a DataFrame for readability
        logging.info(f"Model evaluation results:\n{pd.DataFrame(results)}")
        return results

    except Exception as e:
        raise CustomException(e, sys)

def tune_model_hyperparameters(model, param_grid, X_train, y_train):
    """
    Tune hyperparameters of a given model using GridSearchCV.

    Args:
        model (object): The model to tune
        param_grid (dict): Hyperparameter grid for the model
        X_train (DataFrame): Training features
        y_train (Series): Training target

    Returns:
        object: The best model after hyperparameter tuning
    """
    logging.info(f"Starting hyperparameter tuning for: {model.__class__.__name__}")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)
    grid_search.fit(X_train, y_train)  # Fit the grid search

    # Log the best parameters and score
    logging.info(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    logging.info(f"Best score for {model.__class__.__name__}: {grid_search.best_score_}")

    return grid_search.best_estimator_

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except ModuleNotFoundError as e:
        logging.error(f"Module not found: {e}. Check if all dependencies are installed and modules are available.")
        raise CustomException(f"Error loading object. Module not found: {e}", sys)
    except Exception as e:
        raise CustomException(e, sys)
        