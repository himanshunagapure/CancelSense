import os
import sys

from src.exception import CustomException
from src.logger import logging
#from src.utils import save_object
from utils import save_object


from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    param_grid: dict = None
    max_iter: int = 1000  # Default max iterations for LogisticRegression.
    
class DataTransformation:
    def __init__(self):
        # Initialize feature selection configuration with default parameter grid
        self.feature_selection_config = DataTransformationConfig(
            param_grid={'C': [1000, 200, 100, 50, 10, 1, 0.1, 0.01, 0.001, 0.005, 0.05]}
        )
        self.numerical_columns = []
        self.categorical_columns = []
        
    def get_data_transformer_object(self):
        """
        Creates a preprocessor object for transforming incoming data.
                
        Returns:
            ColumnTransformer: Preprocessor object for data transformation.
        """
        try:
            # Define columns
            numerical_columns = [ 'lead_time', 'is_repeated_guest', 
                                 'previous_cancellations', 'previous_bookings_not_canceled', 
                                 'booking_changes', 'days_in_waiting_list', 'adr', 
                                 'required_car_parking_spaces', 'total_of_special_requests', 
                                 'total_guests', 'total_stay_length', 'is_family', 'is_deposit_given', 
                                 'is_room_upgraded'] 
            categorical_columns = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 
                                 'customer_type']
            
            # Custom transformer to preserve column names after log transformation
            class NamedFunctionTransformer(FunctionTransformer):
                def get_feature_names_out(self, input_features=None):
                    return input_features

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
                ("log_transform", NamedFunctionTransformer(np.log1p, validate=True)),  # Log transformation
                ("scaler", StandardScaler())  # Scaling
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with 0
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ],
                verbose_feature_names_out=False  # This will simplify feature names
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    # Assuming preprocessing_obj is your ColumnTransformer
    def get_feature_names(self, preprocessor):
        """
        Extract feature names after transformation by a ColumnTransformer.

        Args:
            preprocessor (ColumnTransformer): The fitted ColumnTransformer.
            numerical_columns (list): List of numerical column names.
            categorical_columns (list): List of categorical column names.

        Returns:
            list: List of feature names after transformation.
        """
        try:
            # Get feature names for numerical columns (unchanged after transformation)
            numerical_features = self.numerical_columns

            # Get feature names for categorical columns (after one-hot encoding)
            categorical_transformer = preprocessor.named_transformers_['cat_pipeline']
            encoder = categorical_transformer.named_steps['one_hot_encoder']
            categorical_features = encoder.get_feature_names_out(self.categorical_columns)

            # Combine all feature names
            feature_names = list(numerical_features) + list(categorical_features)
            return feature_names
        
        except Exception as e:
            raise CustomException(e, sys)

        
    def select_features(self, X: pd.DataFrame, y: pd.Series):
        """
        Perform feature selection using an embedded Lasso-based method.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.

        Returns:
            pd.DataFrame: DataFrame containing the selected features.
            list: List of selected feature names.
        """
        try:
            logging.info("Starting feature selection process.")

            # Initialize Logistic Regression with L1 penalty
            logistic = LogisticRegression(
                penalty='l1', solver='liblinear', 
                max_iter=self.feature_selection_config.max_iter
            )

            # Grid search to find the best regularization parameter
            logging.info("Performing GridSearchCV to find the best regularization parameter.")
            grid_search = GridSearchCV(
                logistic,
                self.feature_selection_config.param_grid,
                cv=5,
                scoring='accuracy'
            )
            #grid_search.fit(X_scaled, y)
            grid_search.fit(X, y)

            best_C = grid_search.best_params_['C']
            logging.info(f"Best regularization parameter (C): {best_C}")

            # Feature selection based on the best regularization parameter
            logging.info("Selecting features based on Logistic Regression model with best C.")
            selected_model = SelectFromModel(
                LogisticRegression(C=best_C, penalty='l1', solver='liblinear')
            )
            selected_model.fit(X, y)

            # Extracting selected features
            selected_features = X.columns[selected_model.get_support()].tolist()
            logging.info(f"Selected features: {selected_features}")

            return X[selected_features], selected_features

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Initiates feature selection on training and testing datasets.

        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.

        Returns:
            tuple: Transformed training array, testing array, path to the saved scaler object.
        """
        try:
            logging.info("Reading train and test data.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            target_column = 'is_canceled'
            
            logging.info("Splitting data into features and target.")
            if target_column not in train_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in train_df.")
            
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            logging.info("Initializing numerical and categorical columns.")
            # Dynamically determine numerical and categorical columns
            self.numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Clip numerical columns to handle invalid values
            X_train[self.numerical_columns] = X_train[self.numerical_columns].clip(lower=0)
            X_test[self.numerical_columns] = X_test[self.numerical_columns].clip(lower=0)

            logging.info("Creating and fitting preprocessor.")
            preprocessing_obj = self.get_data_transformer_object()
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)
            
            # Get feature names
            feature_names = preprocessing_obj.get_feature_names_out()
            
            # Create DataFrames with proper column names
            X_train_preprocessed = pd.DataFrame(X_train_transformed, columns=feature_names)
            X_test_preprocessed = pd.DataFrame(X_test_transformed, columns=feature_names)
            
            logging.info("Performing feature selection on training data.")
            X_train_selected, selected_features = self.select_features(X_train_preprocessed, y_train)
            X_test_selected = X_test_preprocessed[selected_features]

            logging.info("Converting selected data to arrays.")
            train_arr = pd.concat([X_train_selected, y_train.reset_index(drop=True)], axis=1).values
            test_arr = pd.concat([X_test_selected, y_test.reset_index(drop=True)], axis=1).values

            logging.info("Saving preprocessor object.")
            print(self.feature_selection_config.preprocessor_obj_file_path)
            print(preprocessing_obj)
            save_object(
                self.feature_selection_config.preprocessor_obj_file_path,
                preprocessing_obj  
            )
            
            logging.info("Data Transformation completed successfully.")
            return (
                train_arr,
                test_arr,
                self.feature_selection_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

