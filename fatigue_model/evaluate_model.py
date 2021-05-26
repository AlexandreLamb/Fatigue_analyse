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
from data_processing import DataPreprocessing
import matplotlib.pyplot as plt
import time
import os

model_path = "tensorboard/model/20210526-173148/model_lstm"
model = tf.keras.models.load_model(model_path)

video_to_test = os.listdir("data/stage_data_out/dataset_temporal/Irba_40_min")
df_evaluate_metrics = pd.DataFrame(columns=["video_name","binary_accuracy", "binary_crossentropy", "mean_squared_error"]).set_index("video_name")
for video in video_to_test:
    preprocessing = DataPreprocessing("data/stage_data_out/dataset_temporal/Irba_40_min/"+video+"/"+video+".csv", isTimeSeries = True, batch_size = 1, evaluate = True)
    preprocessing.dataset = preprocessing.dataset.batch(preprocessing.batch_size)

    _ ,binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(preprocessing.dataset)
    df_evaluate_metrics.loc[video] = [binary_accuracy, binary_crossentropy, mean_squared_error]
    
    predictions = model.predict(preprocessing.dataset)
    measure_list = list(pd.read_csv("data/stage_data_out/dataset_temporal/Irba_40_min/"+video+"/"+video+".csv"))
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
    path_folder_to_save = "data/stage_data_out/predictions/"+video
    path_to_csv = path_folder_to_save + "/"+video+"_pred.csv"
    if os.path.exists(path_folder_to_save) == False:
                os.makedirs(path_folder_to_save)
    df.to_csv(path_to_csv, index = False)

df_evaluate_metrics.to_csv("data/stage_data_out/predictions/metrics.csv")


