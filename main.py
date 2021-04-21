import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_manipulation import AnalyseData
from data_manipulation import DataFormator

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

video_infos_path = "data/stage_data_out/videos_infos.csv"

path_folder = "data/stage_data_out/landmarks_csv/mtcnn"
csv_name_arr = os.listdir(path_folder)


df_ear_all = pd.DataFrame()
for csv_name in csv_name_arr :
    video_name = csv_name.split("/")[-1].split(".csv")[0].split("_mtcnn")[0]
    analyse_data = AnalyseData(path_folder+"/"+csv_name)
    analyse_data.measure_ear()
    analyse_data.measure_eyebrow_nose()
    analyse_data.nose_wrinkles()
    analyse_data.jaw_dropping()
    analyse_data.measure_eye_area()

    df_ear = DataFormator.make_label_df(num_min = 5, measures=["frame","ear", "ear_l", "ear_r","eyebrow_nose","eyebrow_eye", "jaw_dropping", "eye_area"], video_name = video_name, df_measure= analyse_data.df_measure)
    df_ear_all = df_ear_all.append(df_ear)
    #DataFormator.save_df(df_ear, video_name)
    """
    df_temporal, df_label = DataFormator.make_df_temporal_label([10],df_ear)  
    df_tab = DataFormator.make_df_feature(df_temporal, df_label, [300])
    """
DataFormator.save_df(df_ear_all,'dataset_ear', 1)