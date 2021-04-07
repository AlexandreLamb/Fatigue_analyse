from video_transforme import  VideoToLandmarks
from data_analyse import AnalyseData
from format_data import DataFormator

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

import os
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

video_infos_path = "data/stage_data_out/videos_infos.csv"

path_array =  [
"data/stage_data_out/DESFAM_Semaine 2-Vendredi_Go-NoGo_H69_hog.csv",
"data/stage_data_out/DESFAM_Semaine 2-Vendredi_Go-NoGo_H71_hog.csv",
"data/stage_data_out/DESFAM_Semaine 2-Vendredi_PVT_H63_hog.csv",
"data/stage_data_out/DESFAM_Semaine 2-Vendredi_PVT_H64_hog.csv",
"data/stage_data_out/DESFAM_Semaine 2-Vendredi_PVT_H68_hog.csv",
"data/stage_data_out/DESFAM_Semaine 2-Vendredi_PVT_H69_hog.csv",
"data/stage_data_out/DESFAM_Semaine 2-Vendredi_PVT_H70_hog.csv",
"data/stage_data_out/DESFAM_Semaine 2-Vendredi_PVT_H71_hog.csv",
"data/stage_data_out/DESFAM_Semaine-2-Vendredi_PVT_H66_hog.csv"]

path_array_test = [
    "data/stage_data_out/test_dataset.csv"
]

df_ear_all = pd.DataFrame()
for video_path in path_array :
    video_name = video_path.split("/")[-1].split(".csv")[0].split("_hog")[0]
    analyse_data = AnalyseData(video_path)
    analyse_data.measure_ear()

    df_ear = DataFormator.make_label_df(num_min = 5, video_name = video_name, df_measure= analyse_data.df_measure)
    df_ear_all = df_ear_all.append(df_ear)
    DataFormator.save_df(df_ear, video_name)
    """
    df_temporal, df_label = DataFormator.make_df_temporal_label([10],df_ear)  
    df_tab = DataFormator.make_df_feature(df_temporal, df_label, [300])
    """
DataFormator.save_df(df_ear_all,'Merge_Dataset')