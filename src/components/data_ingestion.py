'''
Purpose:
We read data from any source (like manual, mongodb, api etc.) 
and with that data we did train-test split
and we saved this raw file, train file, and test file inside artifact folder
'''

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from data_transformation import DataTransformation
from model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            #Collecting data
            #To make code platform-independent and avoid issues, we use Python's os.path to construct file paths dynamically.
            data_path = os.path.join('notebook','data','hotel_booking_cleaned1.csv')
            df = pd.read_csv(data_path)
            logging.info('Read the dataset as dataframe')

            #create artifact directory if not present
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            #Store the raw data at ra_data_path before train-test split    
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            #Imported Data already is converted to numerical values 
            # print(df.dtypes)

            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

       
#For testing 
if __name__=="__main__":
    obj=DataIngestion()
    train_data_p, test_data_p = obj.initiate_data_ingestion()
     
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_p,test_data_p)
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



 