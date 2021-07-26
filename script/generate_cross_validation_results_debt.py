import os, sys
from dotenv import load_dotenv

load_dotenv("env_file/.env_path")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fatigue_model import CrossValidation

PATH_TO_DEBT_CROSS = os.environ.get("PATH_TO_DEBT_CROSS")
PATH_TO_RESULTS_CROSS_PREDICTIONS_DEBT = os.environ.get("PATH_TO_RESULTS_CROSS_PREDICTIONS_DEBT")
PATH_TO_DEBT_VIDEO = os.environ.get("PATH_TO_TIME_ON_DEBT")

cross_validator = CrossValidation(PATH_TO_DEBT_CROSS, PATH_TO_DEBT_VIDEO, PATH_TO_RESULTS_CROSS_PREDICTIONS_DEBT)
cross_validator.train_cross_model()

del cross_validator