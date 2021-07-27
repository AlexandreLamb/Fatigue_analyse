import os, sys
from dotenv import load_dotenv

load_dotenv("env_file/.env_path")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fatigue_model import CrossValidation

PATH_TO_DEBT_CROSS = os.environ.get("PATH_TO_DEBT_CROSS")
PATH_TO_RESULTS_CROSS_PREDICTIONS_DEBT = os.environ.get("PATH_TO_RESULTS_CROSS_PREDICTIONS_DEBT")
PATH_TO_DEBT_VIDEO = os.environ.get("PATH_TO_DEBT_VIDEO")

PATH_TO_DEBT_WEEK_TRAIN = os.environ.get("PATH_TO_DEBT_WEEK_TRAIN")
PATH_TO_DEBT_WEEK_TEST = os.environ.get("PATH_TO_DEBT_WEEK_TEST")



cross_validator = CrossValidation(PATH_TO_DEBT_CROSS, PATH_TO_DEBT_VIDEO, PATH_TO_RESULTS_CROSS_PREDICTIONS_DEBT)
#cross_validator.train_cross_model()
cross_validator.train_cross_model_debt_by_week(PATH_TO_DEBT_WEEK_TRAIN, PATH_TO_DEBT_WEEK_TEST)

del cross_validator