import dill 
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from exception import CustomException

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
    try:
        results = []
        for name, model in models:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            results.append({
                'Model Name': name,
                'Model Accuracy': accuracy
            })

        return results
    except Exception as e:
        raise CustomException(e, sys)
