import argparse, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_manipulation import AnalyseData
from data_manipulation import DataFormator
import json
import os
import pandas as pd
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database_connector import read_remote_df, save_remote_df, list_dir_remote
from dotenv import load_dotenv
load_dotenv("env_file/.env_path")

PATH_TO_DEBT_VIDEO = os.environ.get("PATH_TO_DEBT_VIDEO")
PATH_TO_DEBT_MERGE = os.environ.get("PATH_TO_DEBT_MERGE")

parser=argparse.ArgumentParser()


parser.add_argument('--path', help='path to the landamrks folder')
parser.add_argument('--windows', nargs= '+', type = int, help= 'windows size')
#parser.add_argument('--json_measure', help='path to measure json file')

args=parser.parse_args()

windows_size = args.windows
print(windows_size)
#json_measure = json.load(args.json_measure)

csv_array_name  = list_dir_remote(args.path)
print(csv_array_name)

csv_array_path = [args.path + "/" +  name for name in csv_array_name]

df_ear_all = pd.DataFrame()
measure_full = ["frame","ear","eyebrow_nose","eye_area","jaw_dropping","eyebrow_eye"]
for index, csv_landmarks_path in enumerate(csv_array_path) :
    video_name = csv_array_name[index].split("_mtcnn.")[0]

    analyse_data = AnalyseData(csv_landmarks_path)
    #TODO: make a function who take a json file of meaurec
    analyse_data.measure_ear()
    analyse_data.measure_eyebrow_nose()
    analyse_data.nose_wrinkles()
    analyse_data.jaw_dropping()
    analyse_data.measure_eye_area()
    df_measures = DataFormator.generate_dataset_debt_sleep(video_name = video_name, measures =measure_full , df_measure= analyse_data.df_measure, fps = 10)
    df_temporal, df_label = DataFormator.make_df_temporal_label(windows_size , df_measures)
    df_tab = DataFormator.make_df_feature(df_temporal, df_label, windows_size)
    df_merge = DataFormator.concat_dataset(df_tab)
    for df_to_save in df_tab:
        DataFormator.save_df(df_to_save, video_name, measure = df_to_save.columns[0], dataset_path = PATH_TO_DEBT_VIDEO)
    DataFormator.save_df(df_merge, video_name, dataset_path = PATH_TO_DEBT_VIDEO)
DataFormator.create_dataset_from_measure_folder(PATH_TO_DEBT_VIDEO, windows_size, path_folder_to_save = PATH_TO_DEBT_MERGE)
