import dill 
import os
import sys

import numpy as np
import pandas as pd

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
