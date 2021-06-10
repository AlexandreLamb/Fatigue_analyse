import os
from dotenv import load_dotenv
load_dotenv("env_file/.env")

REMOTE_DATA_URI = os.environ.get("REMOTE_DATA_URI")

dataset_path_uri = os.path.join(REMOTE_DATA_URI,"dataset")
landmarks_path_uri = os.path.join(REMOTE_DATA_URI,"landmarks","DESFAM_F")
landmarks_path_uri = os.path.join(REMOTE_DATA_URI,"results")

desfam_f_folder = ["full","5_min_cut"]

dataset_folder = ["temporal","non_temporal"]
temporal_folder = ["time_on_task", "debt"]
temporal_sub_commmun_folder = ["video", "merge","cross_dataset"]

results_folder = ["predictions","cross_predictions"]
results_sub_folder = ["time_on_task", "debt"]




os.makedirs()