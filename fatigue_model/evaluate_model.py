import tensorflow as tf 
import pandas as pd 
import io
import itertools
import numpy as np 
import json
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorboard.plugins.hparams import api as hp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os
from dotenv import load_dotenv
load_dotenv("env_file/.env_path")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database_connector import read_remote_df, save_remote_df, list_dir_remote
from data_processing import DataPreprocessing

PATH_TO_RESULTS_PREDICTIONS = os.environ.get("PATH_TO_RESULTS_PREDICTIONS")
PATH_TO_TIME_ON_TASK_VIDEO_TO_TEST = os.environ.get("PATH_TO_TIME_ON_TASK_VIDEO_TO_TEST") 

##TODOMODEL/load/save model tf
model_path = "/home/simeon/Desktop/Fatigue_analyse/tensorboard/model/ear_30_eye_area_30_jaw_dropping_30_eyebrow_eye_30"
model = tf.keras.models.load_model(model_path)

video_to_test = list_dir_remote(PATH_TO_TIME_ON_TASK_VIDEO_TO_TEST)
df_evaluate_metrics = pd.DataFrame(columns=["video_name","binary_accuracy", "binary_crossentropy", "mean_squared_error"]).set_index("video_name")
for video in video_to_test:
    preprocessing = DataPreprocessing(os.path.join(PATH_TO_TIME_ON_TASK_VIDEO_TO_TEST,video,"/"+video+".csv"), isTimeSeries = True, batch_size = 1, evaluate = True)
    preprocessing.dataset = preprocessing.dataset.batch(preprocessing.batch_size)

    _ ,binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(preprocessing.dataset)
    df_evaluate_metrics.loc[video] = [binary_accuracy, binary_crossentropy, mean_squared_error]
    
    predictions = model.predict(preprocessing.dataset)
    df_video = read_remote_df(os.path.join(PATH_TO_TIME_ON_TASK_VIDEO_TO_TEST,video,"/"+video+".csv"))
    measure_list = list(df_video)
    df = pd.DataFrame(np.squeeze(predictions), columns = [measure for measure in measure_list if measure != "target"])
    y_pred_list = []
    y_pred = []
    for idx in df.index:
        df.loc[idx, "pred_mean"] = df.loc[idx].mean() 
        df.loc[idx, "pred_max"] = df.loc[idx].max() 
    df.loc[lambda df: df["pred_mean"] < 0.5,"target_pred_mean"] = 0
    df.loc[lambda df: df["pred_mean"] >= 0.5,"target_pred_mean"] = 1
    df.loc[lambda df: df["pred_max"] < 0.5,"target_pred_max"] = 0
    df.loc[lambda df: df["pred_max"] >= 0.5,"target_pred_max"] = 1
    df["target_real"] = df_video["target"]
    path_to_csv = os.path.join(PATH_TO_RESULTS_PREDICTIONS, video, video+"_pred.csv")
    save_remote_df(path_to_csv, df, index=False)

save_remote_df(os.path.join(PATH_TO_RESULTS_PREDICTIONS, "metrics.csv"), df_evaluate_metrics)

