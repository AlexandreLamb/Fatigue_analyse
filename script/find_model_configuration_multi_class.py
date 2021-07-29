import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fatigue_model.hyper_parameter_tunning import ModelTunningMultiClass
from dotenv import load_dotenv
load_dotenv("env_file/.env_path")
 
PATH_TO_TIME_ON_TASK_MERGE_VAS = os.environ.get("PATH_TO_TIME_ON_TASK_MERGE_VAS")
PATH_TO_DEBT_MERGE = os.environ.get("PATH_TO_DEBT_MERGE")

json_path = "fatigue_model/model_parameter/hparms_lstm_multi_class.json"
path_to_merge_dataset_folder = PATH_TO_TIME_ON_TASK_MERGE_VAS
mt = ModelTunningMultiClass(json_path, path_to_merge_dataset_folder, isTimeSeries = True, batch_size=32)
mt.initialize_model("LSTM")
mt.tune_model()
del mt
