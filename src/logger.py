import logging
import os
from datetime import datetime

#Formats the current date and time into a string (e.g., 11_16_2024_15_30_45) to create unique log file names.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#setting up log file path
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

#Final Log file path.Example: /current/working/directory/logs/11_16_2024_15_30_45.log 
LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

#Configuring the Logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,   
)

'''
For testing logger.py file
if __name__=="__main__":
    logging.info("Logging has started")
'''