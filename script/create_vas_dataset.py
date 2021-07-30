import argparse, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_manipulation.check_dataset import check_landmarks_len
from data_manipulation import DataFormator
import pandas as pd

from dotenv import load_dotenv
load_dotenv("env_file/.env_path")

PATH_TO_TIME_ON_TASK_VIDEO_VAS = os.environ.get("PATH_TO_TIME_ON_TASK_VIDEO_VAS")
PATH_TO_TIME_ON_TASK_MERGE_VAS = os.environ.get("PATH_TO_TIME_ON_TASK_MERGE_VAS")
PATH_TO_TIME_ON_TASK_CROSS_VIDEO_VAS = os.environ.get("PATH_TO_TIME_ON_TASK_CROSS_VIDEO_VAS")
dataformator = DataFormator()

#dataformator.create_VAS_dataset()
#dataformator.create_dataset_from_measure_folder(PATH_TO_TIME_ON_TASK_VIDEO_VAS,[30],PATH_TO_TIME_ON_TASK_MERGE_VAS)
dataformator.generate_cross_dataset(PATH_TO_TIME_ON_TASK_VIDEO_VAS, PATH_TO_TIME_ON_TASK_CROSS_VIDEO_VAS)