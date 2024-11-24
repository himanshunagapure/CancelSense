import sys
import pandas as pd
import numpy as np
import os
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.utils import load_object
import dill 

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "best_model.pkl")
            print("Before Loading")
            model_details = load_object(file_path=model_path)
            logging.info(f"Loading Model: {model_details}")
            print("After Loading Model")
            
            # Load the saved preprocessor
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            logging.info(f"Trying to read preprocessor Model")
            #preprocessor = load_object(file_path=preprocessor_path)
            
            preprocessor = None 
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found ❌: {preprocessor_path}")
            logging.info(f"Found preprocessor path at {preprocessor_path}")
            
            with open(preprocessor_path, "rb") as file_obj:
                preprocessor = dill.load(file_obj)
            logging.info(f"Loaded preprocessor successfully")

            preprocessed_data = preprocessor.transform(features)
            print("After Preprocessing")
            
            pred = model_details['model_object'].predict(preprocessed_data)
            return pred
            
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                hotel: str, arrival_date_month: str, meal: str, 
                country: str, market_segment: str, 
                customer_type: str, lead_time: int, is_repeated_guest: str, 
                previous_cancellations: int, 
                previous_bookings_not_canceled: int, booking_changes: int, 
                days_in_waiting_list: int, adr: float, 
                required_car_parking_spaces: int, total_of_special_requests: int, 
                total_guests: int, total_stay_length: int, 
                is_family: str, is_deposit_given: str, is_room_upgraded: str):
        self.hotel = hotel
        self.arrival_date_month = arrival_date_month
        self.meal = meal
        self.country = country
        self.market_segment = market_segment
        self.customer_type = customer_type
        self.lead_time = lead_time
        self.is_repeated_guest = is_repeated_guest
        self.previous_cancellations = previous_cancellations
        self.previous_bookings_not_canceled = previous_bookings_not_canceled
        self.booking_changes = booking_changes
        self.days_in_waiting_list = days_in_waiting_list
        self.adr = adr
        self.required_car_parking_spaces = required_car_parking_spaces
        self.total_of_special_requests = total_of_special_requests
        self.total_guests = total_guests
        self.total_stay_length = total_stay_length
        self.is_family = is_family
        self.is_deposit_given = is_deposit_given
        self.is_room_upgraded = is_room_upgraded
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "hotel": [self.hotel],
                "arrival_date_month": [self.arrival_date_month],
                "meal": [self.meal],
                "country": [self.country],
                "market_segment": [self.market_segment],
                "customer_type": [self.customer_type],
                "lead_time": [self.lead_time],
                "is_repeated_guest": [self.is_repeated_guest],
                "previous_cancellations": [self.previous_cancellations],
                "previous_bookings_not_canceled": [self.previous_bookings_not_canceled],
                "booking_changes": [self.booking_changes],
                "days_in_waiting_list": [self.days_in_waiting_list],
                "adr": [self.adr],
                "required_car_parking_spaces": [self.required_car_parking_spaces],
                "total_of_special_requests": [self.total_of_special_requests],
                "total_guests": [self.total_guests],
                "total_stay_length": [self.total_stay_length],
                "is_family": [self.is_family],
                "is_deposit_given": [self.is_deposit_given],
                "is_room_upgraded": [self.is_room_upgraded]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)