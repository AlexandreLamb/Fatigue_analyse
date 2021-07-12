import os, sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv("env_file/.env_path")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database_connector import SFTPConnector


def check_dataset_label_mean(path_base):
    sftp = SFTPConnector()        
    dir =  sftp.list_dir_remote(path_base)
    csv_to_read = [path_base+"/"+folder+"/"+folder+".csv" for folder in dir ]

    for csv in csv_to_read:
        df = pd.read_csv(csv)
        if not df.describe().loc["mean"].between(0.4,0.6).item() :
            #print("/".join(csv.split('/')[0:-1]))
            base_path = "/".join(csv.split('/')[0:-1])
            file_to_remove = sftp.list_dir_remote(base_path)
            for file in file_to_remove:
                sftp.remove_file_remote(os.path.join(base_path,file))
            sftp.remove_dir_remote(base_path)


check_dataset_label_mean("/home/fatigue_database/fatigue_database/dataset/temporal/DESFAM_F/time_on_task/video")
