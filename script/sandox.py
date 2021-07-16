import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_manipulation.check_dataset import check_landmarks_len
from data_manipulation import AnalyseData, format_data
from data_manipulation import DataFormator
import pandas as pd
from database_connector import  SFTPConnector
from dotenv import load_dotenv
load_dotenv("env_file/.env_path")

PATH_TO_TIME_ON_TASK_VIDEO = os.environ.get("PATH_TO_TIME_ON_TASK_VIDEO")
PATH_TO_TIME_ON_TASK_MERGE = os.environ.get("PATH_TO_TIME_ON_TASK_MERGE")
PATH_TO_TIME_ON_TASK_CROSS = os.environ.get("PATH_TO_TIME_ON_TASK_CROSS")

PATH_TO_DEBT_VIDEO = os.environ.get("PATH_TO_DEBT_VIDEO")
PATH_TO_DEBT_MERGE = os.environ.get("PATH_TO_DEBT_MERGE")
PATH_TO_DEBT_CROSS = os.environ.get("PATH_TO_DEBT_CROSS")

PATH_TO_LANDMARKS_DESFAM_F_5_MIN = os.environ.get("PATH_TO_LANDMARKS_DESFAM_F_5_MIN")
PATH_TO_LANDMARKS_DESFAM_F_FULL = os.environ.get("PATH_TO_LANDMARKS_DESFAM_F_FULL")

WINDOWS_SIZE = int(os.environ.get("WINDOWS_SIZE"))



format_data = DataFormator()

format_data.generate_cross_dataset_by_week(PATH_TO_DEBT_VIDEO)

del format_data