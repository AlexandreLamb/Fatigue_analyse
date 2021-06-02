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
import os, sys

from tensorflow.python.keras.activations import sigmoid 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_processing import DataPreprocessing
from logger import logging


MODEL = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64,input_shape=(5,30) , return_sequences=True),
    #tf.keras.layers.Dense(units=32, activation = "relu"),
    tf.keras.layers.LSTM(64,input_shape=(5,30) , return_sequences=True),
    #tf.keras.layers.Dense(units=64, activation = "relu"),
    tf.keras.layers.LSTM(64,input_shape=(5,30) , return_sequences=True),
    tf.keras.layers.Dense(units=1, activation = "sigmoid")
])


def train_evaluate_model(path_to_dataset, df_metrics_model_train):
    video_exclude = path_to_dataset.split("/")[-2].split("exclude_")[-1]
    dp = DataPreprocessing(path_to_dataset,batch_size= 32, isTimeSeries = True) 
    logging.info("path to dataset")
    logging.info(path_to_dataset)
    train = dp.train
    test = dp.test 
    val = dp.val
    
    MODEL.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(),
              metrics=["binary_accuracy","binary_crossentropy","mean_squared_error"])
    MODEL.summary()
    logging.info("start fit model")
    MODEL.fit(
        train, 
        validation_data= val,
        epochs=1000,
        shuffle=True,
        verbose =1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='auto')]) 
    path_to_model_to_save = "tensorboard/model/"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + "/model_lstm_exclude_"+video_exclude
   
    logging.info("SAVING... into " + path_to_model_to_save)
    MODEL.save(path_to_model_to_save)
    logging.info("SAVE !")

    _ ,binary_accuracy, binary_crossentropy, mean_squared_error = MODEL.evaluate(test)
    df_metrics_model_train.loc[video_exclude] = [binary_accuracy, binary_crossentropy, mean_squared_error]
    return path_to_model_to_save, video_exclude
    
    
    
def evaluate_model(model_path, video_exclude):
    MODEL = tf.keras.models.load_model(model_path)
    logging.info("model_path")
    logging.info(model_path)
    logging.info("video_exclude")
    logging.info(video_exclude)
    
    df_evaluate_metrics = pd.DataFrame(columns=["video_exclude","binary_accuracy", "binary_crossentropy", "mean_squared_error"]).set_index("video_exclude")
    
    preprocessing = DataPreprocessing("data/stage_data_out/dataset_temporal/Irba_40_min/"+video_exclude+"/"+video_exclude+".csv", isTimeSeries = True, batch_size = 1, evaluate = True)
    preprocessing.dataset = preprocessing.dataset.batch(preprocessing.batch_size)
    
    _ ,binary_accuracy, binary_crossentropy, mean_squared_error = MODEL.evaluate(preprocessing.dataset)
    df_evaluate_metrics.loc[video_exclude] = [binary_accuracy, binary_crossentropy, mean_squared_error]
    
    predictions = MODEL.predict(preprocessing.dataset)
    
    df_video = pd.read_csv("data/stage_data_out/dataset_temporal/Irba_40_min/"+video_exclude+"/"+video_exclude+".csv")
    measure_list = list(df_video)
    df = pd.DataFrame(np.squeeze(predictions), columns = [measure for measure in measure_list if measure != "target"])
    for idx in df.index:
        df.loc[idx, "pred_mean"] = df.loc[idx].mean() 
        df.loc[idx, "pred_max"] = df.loc[idx].max() 
    df.loc[lambda df: df["pred_mean"] < 0.5,"target_pred_mean"] = 0
    df.loc[lambda df: df["pred_mean"] >= 0.5,"target_pred_mean"] = 1
    df.loc[lambda df: df["pred_max"] < 0.5,"target_pred_max"] = 0
    df.loc[lambda df: df["pred_max"] >= 0.5,"target_pred_max"] = 1
    df["target_real"] = df_video["target"]
    path_folder_to_save = "data/stage_data_out/cross_predictions/"+video_exclude
    path_to_csv = path_folder_to_save + "/"+video_exclude+"_pred.csv"
    if os.path.exists(path_folder_to_save) == False:
                os.makedirs(path_folder_to_save)
    df.to_csv(path_to_csv, index = False)
    logging.info("SAVING...")
    df_evaluate_metrics.to_csv("data/stage_data_out/cross_predictions/"+video_exclude+"/metrics.csv")
    logging.info("SAVE !")



    
    
    
def train_cross_model():
    cross_dataset_path = "data/stage_data_out/dataset_temporal/cross-dataset/"
    folder_dataset = os.listdir(cross_dataset_path)
    path_dataset = [ cross_dataset_path + folder + "/dataset.csv" for folder in  folder_dataset ]
    df_metrics_model_train = pd.DataFrame(columns=["video_exclude","binary_accuracy","binary_crossentropy","mean_squared_error"]).set_index("video_exclude")
    for dataset in path_dataset:
        logging.info("start with  :" + dataset)
        path_to_model, video_to_exclude = train_evaluate_model(dataset, df_metrics_model_train)
        evaluate_model(path_to_model, video_to_exclude)
   
train_cross_model()