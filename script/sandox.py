import argparse, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database_connector import SFTPConnector

import pandas as pd

from dotenv import load_dotenv
load_dotenv("env_file/.env_path")

PATH_TO_RESULTS_CROSS_PREDICTIONS_TIME_ON_TASK = os.environ.get("PATH_TO_RESULTS_CROSS_PREDICTIONS_TIME_ON_TASK")


sftp = SFTPConnector()
base_path = os.path.join(PATH_TO_RESULTS_CROSS_PREDICTIONS_TIME_ON_TASK,"20210729-160224")
res = sftp.list_dir_remote(base_path)
df_all = pd.DataFrame()
for dir_name in [name for name in res if name != "metrics_train_model.csv"]:
    df_all = df_all.append(sftp.read_remote_df(os.path.join(base_path, dir_name,"metrics.csv")))
print(df_all.sort_values("binary_accuracy"))
print(df_all[df_all["binary_accuracy"]>0.1]["binary_accuracy"].mean())

for dir_name in [name for name in res if name != "metrics_train_model.csv"]:
    df = sftp.read_remote_df(os.path.join(base_path, dir_name,"pred.csv"))
    #print(dir_name)    
    first_interval = df["target_real"].value_counts()[0]
    #print()
    #print(df[0:first_interval].describe())
    #print(df[0:first_interval]["target_round"].value_counts().index[0] == 0)
    #print(df[first_interval:]["target_round"].value_counts().index[0] == 1)
    #print(df[first_interval:].describe())
    #print()
    

del sftp