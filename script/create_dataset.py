import argparse, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_manipulation import AnalyseData
from data_manipulation import DataFormator
import pandas as pd
from database_connector import  SFTPConnector
from dotenv import load_dotenv
load_dotenv("env_file/.env_path")

PATH_TO_TIME_ON_TASK_VIDEO = os.environ.get("PATH_TO_TIME_ON_TASK_VIDEO")
PATH_TO_TIME_ON_TASK_MERGE = os.environ.get("PATH_TO_TIME_ON_TASK_MERGE")
PATH_TO_TIME_ON_TASK_CROSS = os.environ.get("PATH_TO_TIME_ON_TASK_CROSS")

PATH_TO_DEBT_VIDEO = os.environ.get("PATH_TO_DEBT_VIDEO")
PATH_TO_DEBT_MERGE = os.environ.get("PATH_TO_DEBT_MERGE")
PATH_TO_DEBT_CROSS = os.environ.get("PATH_TO_DEBT_CROSS")

PATH_TO_LANDMARKS_DESFAM_F_5_MIN = os.environ.get("PATH_TO_LANDMARKS_DESFAM_F_5_MIN")
WINDOWS_SIZE = int(os.environ.get("WINDOWS_SIZE"))

def create_dataset(dataset_path, path_folder_to_save, path_folder_cross, dataset_type):
    sftp = SFTPConnector()
    csv_array_name  = sftp.list_dir_remote(PATH_TO_LANDMARKS_DESFAM_F_5_MIN)
    csv_array_path = [PATH_TO_LANDMARKS_DESFAM_F_5_MIN + "/" +  name for name in csv_array_name]
    dataformat = DataFormator()
    analyse_data = AnalyseData()
    measure_full = ["frame","ear","eyebrow_nose","eye_area","jaw_dropping","eyebrow_eye"]
    for index, csv_landmarks_path in enumerate(csv_array_path) :
        analyse_data.load_csv(csv_landmarks_path)
        video_name = csv_array_name[index].split("_mtcnn")[0]
        #TODO: make a function who take a json file of meaurec
        analyse_data.measure_ear()
        analyse_data.measure_eyebrow_nose()
        analyse_data.nose_wrinkles()
        analyse_data.jaw_dropping()
        analyse_data.measure_eye_area()
        if dataset_type == "time_on_task":
            df_measures = dataformat.make_label_df(num_min = 5, video_name = video_name, measures =measure_full , df_measure= analyse_data.df_measure, fps = 10)
        elif dataset_type == "debt" :
            df_measures = dataformat.generate_dataset_debt_sleep(video_name = video_name, measures =measure_full , df_measure= analyse_data.df_measure, fps = 10)        
        df_temporal, df_label = dataformat.make_df_temporal_label([WINDOWS_SIZE] , df_measures)
        df_tab = dataformat.make_df_feature(df_temporal, df_label, [WINDOWS_SIZE])
        df_merge = dataformat.concat_dataset(df_tab)
        for df_to_save in df_tab:
            dataformat.save_df(df_to_save, video_name, dataset_path, measure= df_to_save.columns[0])
        dataformat.save_df(df_merge, video_name, dataset_path)
    dataformat.create_dataset_from_measure_folder(dataset_path, [WINDOWS_SIZE], path_folder_to_save = path_folder_to_save)
    dataformat.generate_cross_dataset(dataset_path, path_folder_cross)
    del dataformat
    del analyse_data

#create_dataset(PATH_TO_TIME_ON_TASK_VIDEO, PATH_TO_TIME_ON_TASK_MERGE, PATH_TO_TIME_ON_TASK_CROSS, dataset_type="time_on_task")
#create_dataset(PATH_TO_DEBT_VIDEO, PATH_TO_DEBT_MERGE, PATH_TO_DEBT_CROSS, dataset_type="debt")


def generate_cross():
    """[summary]
    """
    dataformat = DataFormator()
    dataformat.generate_cross_dataset(PATH_TO_TIME_ON_TASK_VIDEO, PATH_TO_TIME_ON_TASK_CROSS)

generate_cross()
