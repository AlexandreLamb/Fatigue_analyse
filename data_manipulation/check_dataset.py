from genericpath import isdir
import os, sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv("env_file/.env_path")
PATH_TO_TIME_ON_TASK_VIDEO = os.environ.get("PATH_TO_TIME_ON_TASK_VIDEO")
PATH_TO_LANDMARKS_DESFAM_F_5_MIN = os.environ.get("PATH_TO_LANDMARKS_DESFAM_F_FULL")
PATH_TO_LANDMARKS_DESFAM_F = os.environ.get("PATH_TO_LANDMARKS_DESFAM_F")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database_connector import SFTPConnector

def check_landmarks_len(path_base):
    sftp = SFTPConnector()        
    dir =  sftp.list_dir_remote(path_base)
    video_infos_path = os.path.join(PATH_TO_LANDMARKS_DESFAM_F,"videos_infos.csv")
    video_infos = pd.DataFrame(sftp.read_remote_df(video_infos_path, index_col=1)["frame_count"])
    fps = 10
    video_length = 45*60
    threshold = (fps * video_length)*0.2
    total_frame = (fps * video_length) - threshold
    csv_to_read = [path_base+"/"+file for file in dir]
    for csv in csv_to_read:
        df = pd.read_csv(csv)
        video_name = "_".join(csv.split('/')[-1].split('_')[0:-2])
        video_infos.loc[video_name,"landmarks_length"] = len(df)
        video_infos.loc[video_name,"frame_ratio"] = len(df)/video_infos.loc[video_name]["frame_count"]
          
    return list(video_infos[video_infos["landmarks_length"] > total_frame].index)

def check_dataset_len(path_base):
    sftp = SFTPConnector()        
    dir =  sftp.list_dir_remote(path_base)
    csv_to_read = [path_base+"/"+folder+"/"+folder+".csv" for folder in dir ]
    for csv in csv_to_read:
        df = pd.read_csv(csv)
        print("----------------")
        print(csv.split('/')[-2])
        print(" length is " + str(len(df)))
        print("mean is "+  str(df.describe().loc["mean"]))
        print("----------------")


def check_dataset_label_mean(path_base):
    sftp = SFTPConnector()        
    dir =  sftp.list_dir_remote(path_base)
    csv_to_read = [path_base+"/"+folder+"/"+folder+".csv" for folder in dir ]

    for csv in csv_to_read:
        df = pd.read_csv(csv)
        if not df.describe().loc["mean"].between(0.4,0.6).item() :
            #print("/".join(csv.split('/')[0:-1]))
            print(csv)
            print(df.describe())
            """base_path = "/".join(csv.split('/')[0:-1])
            file_to_remove = sftp.list_dir_remote(base_path)
            for file in file_to_remove:
                sftp.remove_file_remote(os.path.join(base_path,file))
            sftp.remove_dir_remote(base_path)"""


check_landmarks_len(PATH_TO_LANDMARKS_DESFAM_F_5_MIN)
