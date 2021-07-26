import os, sys
from dotenv import load_dotenv

load_dotenv("env_file/.env_path")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fatigue_model import CrossValidation

PATH_TO_DEBT_CROSS = os.environ.get("PATH_TO_DEBT_CROSS")
print(PATH_TO_DEBT_CROSS)

cross_validator = CrossValidation(PATH_TO_DEBT_CROSS)
cross_validator.train_cross_model()

del cross_validator