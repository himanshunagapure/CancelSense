import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging
from utils import save_object

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    param_grid: dict = None
    max_iter: int = 1000  # Default max iterations for LogisticRegression
  
class DataTransformation:
    def __init__(self):
        # Initialize feature selection configuration with default parameter grid
        self.feature_selection_config = DataTransformationConfig(
            param_grid={'C': [1000, 200, 100, 50, 10, 1, 0.1, 0.01, 0.001, 0.005, 0.05]}
        )
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, scaler_path: str):
        """
        Perform feature selection using an embedded Lasso-based method.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            scaler_path (str): Path to save the fitted scaler object.

        Returns:
            pd.DataFrame: DataFrame containing the selected features.
            list: List of selected feature names.
        """
        try:
            logging.info("Starting feature selection process.")

            # Scale the feature set
            logging.info("Scaling features using StandardScaler.")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Save the scaler object
            save_object(scaler_path, scaler)
            logging.info(f"Scaler saved at: {scaler_path}")

            # Initialize Logistic Regression with L1 penalty
            logistic = LogisticRegression(
                penalty='l1', solver='liblinear', max_iter=self.feature_selection_config.max_iter
            )

            # Grid search to find the best regularization parameter
            logging.info("Performing GridSearchCV to find the best regularization parameter.")
            grid_search = GridSearchCV(
                logistic,
                self.feature_selection_config.param_grid,
                cv=5,
                scoring='accuracy'
            )
            grid_search.fit(X_scaled, y)
            best_C = grid_search.best_params_['C']
            logging.info(f"Best regularization parameter (C): {best_C}")

            # Feature selection based on the best regularization parameter
            logging.info("Selecting features based on Logistic Regression model with best C.")
            selected_model = SelectFromModel(
                LogisticRegression(C=best_C, penalty='l1', solver='liblinear')
            )
            selected_model.fit(X_scaled, y)

            # Extracting selected features
            selected_features = X.columns[selected_model.get_support()]
            logging.info(f"Selected features: {list(selected_features)}")

            return X[selected_features], list(selected_features)

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path: str, test_path: str, target_column: str):
        """
        Initiates feature selection on training and testing datasets.

        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.
            target_column (str): Target column name.

        Returns:
            tuple: Transformed training array, testing array, path to the saved scaler object.
        """
        try:
            logging.info("Reading train and test data.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Splitting data into features and target.")
            if target_column not in train_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in train_df.")
            X_train = train_df.drop(columns=[target_column])
            #X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Performing feature selection on training data.")
            X_train_selected, selected_features = self.select_features(
                X_train, y_train, self.feature_selection_config.preprocessor_obj_file_path
            )

            logging.info("Applying selected features to testing data.")
            X_test_selected = X_test[selected_features]

            logging.info("Converting selected data to arrays.")
            train_arr = pd.concat([X_train_selected, y_train.reset_index(drop=True)], axis=1).values
            test_arr = pd.concat([X_test_selected, y_test.reset_index(drop=True)], axis=1).values

            logging.info("Data Transformation completed successfully.")

            return (
                train_arr,
                test_arr,
                self.feature_selection_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

