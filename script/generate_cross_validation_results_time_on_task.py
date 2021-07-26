import os, sys
from dotenv import load_dotenv

load_dotenv("env_file/.env_path")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fatigue_model import CrossValidation

PATH_TO_TIME_ON_TASK_CROSS = os.environ.get("PATH_TO_TIME_ON_TASK_CROSS")
PATH_TO_RESULTS_CROSS_PREDICTIONS_TIME_ON_TASK = os.environ.get("PATH_TO_RESULTS_CROSS_PREDICTIONS_TIME_ON_TASK")
PATH_TO_TIME_ON_TASK_VIDEO = os.environ.get("PATH_TO_TIME_ON_TASK_VIDEO")


cross_validator = CrossValidation(PATH_TO_TIME_ON_TASK_CROSS, PATH_TO_TIME_ON_TASK_VIDEO, PATH_TO_RESULTS_CROSS_PREDICTIONS_TIME_ON_TASK)
cross_validator.train_cross_model()

del cross_validator