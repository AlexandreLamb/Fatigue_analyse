import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fatigue_model.hyper_parameter_tunning import ModelTunning
from dotenv import load_dotenv
load_dotenv("env_file/.env_path")
 
PATH_TO_TIME_ON_TASK_MERGE = os.environ.get("PATH_TO_TIME_ON_TASK_MERGE")
PATH_TO_DEBT_MERGE = os.environ.get("PATH_TO_DEBT_MERGE")

json_path = "fatigue_model/model_parameter/hparms_lstm.json"
path_to_merge_dataset_folder = PATH_TO_TIME_ON_TASK_MERGE
mt = ModelTunning(json_path, path_to_merge_dataset_folder, isTimeSeries = True, batch_size=32)
mt.initialize_model("LSTM")
mt.tune_model()
del mt
path_to_merge_dataset_folder = PATH_TO_DEBT_MERGE
mt = ModelTunning(json_path, path_to_merge_dataset_folder, isTimeSeries = True, batch_size=32)
mt.initialize_model("LSTM")
mt.tune_model()
del mt