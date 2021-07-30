import os, sys
from dotenv import load_dotenv

load_dotenv("env_file/.env_path")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fatigue_model import CrossValidationMultiClass

PATH_TO_TIME_ON_TASK_CROSS_VIDEO_VAS = os.environ.get("PATH_TO_TIME_ON_TASK_CROSS_VIDEO_VAS")
PATH_TO_RESULTS_CROSS_PREDICTIONS_TIME_ON_TASK_VAS = os.environ.get("PATH_TO_RESULTS_CROSS_PREDICTIONS_TIME_ON_TASK_VAS")
PATH_TO_TIME_ON_TASK_VIDEO_VAS = os.environ.get("PATH_TO_TIME_ON_TASK_VIDEO_VAS")


cross_validator = CrossValidationMultiClass(PATH_TO_TIME_ON_TASK_CROSS_VIDEO_VAS, PATH_TO_TIME_ON_TASK_VIDEO_VAS, PATH_TO_RESULTS_CROSS_PREDICTIONS_TIME_ON_TASK_VAS)
cross_validator.train_cross_model()

del cross_validator