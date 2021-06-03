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
from itertools import combinations
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


def train_evaluate_model(path_to_dataset, df_metrics_model_train, df, date_id):
    video_exclude = path_to_dataset.split("/")[-2].split("exclude_")[-1]
    measure_combinaition = [measure for measure in list(df) if measure != "target"]
    dp = DataPreprocessing(path_to_dataset = None,batch_size= 32, isTimeSeries = True, df_dataset = df) 
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
    path_to_model_to_save = "tensorboard/model/"+date_id+ "/model_lstm_exclude_"+video_exclude+"/"+"_".join(measure_combinaition)
   
    logging.info("SAVING... into " + path_to_model_to_save)
    MODEL.save(path_to_model_to_save)
    logging.info("SAVE !")

    _ ,binary_accuracy, binary_crossentropy, mean_squared_error = MODEL.evaluate(test)
    df_metrics_model_train.loc[(video_exclude, "_".join(measure_combinaition))] = [binary_accuracy, binary_crossentropy, mean_squared_error]
    return path_to_model_to_save, video_exclude, df_metrics_model_train
    
    
    
def evaluate_model(model_path, video_exclude, cross_measures): 
    MODEL = tf.keras.models.load_model(model_path)
    logging.info("model_path")
    logging.info(model_path)
    logging.info("video_exclude")
    logging.info(video_exclude)
    
    df_evaluate_metrics = pd.DataFrame(columns=["video_exclude", "measure_combination", "binary_accuracy", "binary_crossentropy", "mean_squared_error"]).set_index(["video_exclude","measure_combination"])
    df = pd.read_csv("data/stage_data_out/dataset_temporal/Irba_40_min/"+video_exclude+"/"+video_exclude+".csv")
    sub_df = df[cross_measures + ["target"]]
    preprocessing = DataPreprocessing(path_to_dataset = None, isTimeSeries = True, batch_size = 1, evaluate = True, df_dataset=sub_df)
    preprocessing.dataset = preprocessing.dataset.batch(preprocessing.batch_size)
    
    _ ,binary_accuracy, binary_crossentropy, mean_squared_error = MODEL.evaluate(preprocessing.dataset)
    df_evaluate_metrics.loc[(video_exclude, cross_measures)] = [binary_accuracy, binary_crossentropy, mean_squared_error]
    
    predictions = MODEL.predict(preprocessing.dataset)
    
    measure_list = list(sub_df)
    df_pred = pd.DataFrame(np.squeeze(predictions), columns = [measure for measure in measure_list if measure != "target"])
    for idx in df_pred.index:
        df_pred.loc[idx, "pred_mean"] = df_pred.loc[idx].mean() 
        df_pred.loc[idx, "pred_max"] = df_pred.loc[idx].max() 
    df_pred.loc[lambda df_pred: df_pred["pred_mean"] < 0.5,"target_pred_mean"] = 0
    df_pred.loc[lambda df_pred: df_pred["pred_mean"] >= 0.5,"target_pred_mean"] = 1
    df_pred.loc[lambda df_pred: df_pred["pred_max"] < 0.5,"target_pred_max"] = 0
    df_pred.loc[lambda df_pred: df_pred["pred_max"] >= 0.5,"target_pred_max"] = 1
    df_pred["target_real"] = sub_df["target"]
    path_folder_to_save = "data/stage_data_out/cross_predictions/"+video_exclude+"/"+cross_measures
    path_to_csv = path_folder_to_save + "/pred.csv"
    if os.path.exists(path_folder_to_save) == False:
                os.makedirs(path_folder_to_save)
    df_pred.to_csv(path_to_csv, index = False)
    logging.info("SAVING...")
    df_evaluate_metrics.to_csv("data/stage_data_out/cross_predictions/" + video_exclude + "/" + cross_measures + "/metrics.csv")
    logging.info("SAVE !")
   
def train_cross_model():
    cross_combinations = lambda df : sum([list(map(list, combinations(df, i))) for i in range(len(df) + 1) if i != 0],[])
    cross_dataset_path = "data/stage_data_out/dataset_temporal/cross-dataset/"
    folder_dataset = os.listdir(cross_dataset_path)
    path_dataset = [ cross_dataset_path + folder + "/dataset.csv" for folder in  folder_dataset ]
    date_id = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    df_metrics_model_train = pd.DataFrame(columns=["video_exclude","measure_combination","binary_accuracy","binary_crossentropy","mean_squared_error"]).set_index(["video_exclude","measure_combination"])
    for dataset in path_dataset:
        df = pd.read_csv(dataset)
        cross_combinations_measure = cross_combinations([measure for measure in list(df) if measure != "target"])
        for measures in cross_combinations_measure:
            sub_df = df[measures + ["target"]]
            logging.info("start with  :" + dataset)
            logging.info("start with comniation of measure : " + measures)
            path_to_model, video_to_exclude, df_metrics_model_train = train_evaluate_model(dataset, df_metrics_model_train, sub_df, date_id)
            evaluate_model(path_to_model, video_to_exclude, measures)
    df_metrics_model_train.to_csv("data/stage_data_out/cross_predictions/metrics_train_model.csv")

train_cross_model()