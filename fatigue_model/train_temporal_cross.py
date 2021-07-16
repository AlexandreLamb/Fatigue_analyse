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
from fatigue_model.data_processing import DataPreprocessing
from dotenv import load_dotenv
load_dotenv("env_file/.env_path")

from database_connector import SFTPConnector
PATH_TO_RESULTS_CROSS_PREDICTIONS = os.environ.get("PATH_TO_RESULTS_CROSS_PREDICTIONS")
PATH_TO_TIME_ON_TASK_VIDEO = os.environ.get("PATH_TO_TIME_ON_TASK_VIDEO")
PATH_TO_TIME_ON_TASK_CROSS = os.environ.get("PATH_TO_TIME_ON_TASK_CROSS")

class CrossValidation:
    def __init__(self, cross_dataset_path):
        self.sftp = SFTPConnector()
        self.cross_dataset_path = cross_dataset_path



    def define_model(self, measure_len) :
        return tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64,input_shape=(measure_len,30) , return_sequences=True),
        #tf.keras.layers.Dense(units=32, activation = "relu"),
        tf.keras.layers.LSTM(32,input_shape=(measure_len,30) , return_sequences=True),
        #tf.keras.layers.Dense(units=64, activation = "relu"),
        tf.keras.layers.LSTM(256,input_shape=(measure_len,30) , return_sequences=True),
        tf.keras.layers.Dense(units=1, activation = "sigmoid")
        ])



    def train_evaluate_model(self, path_to_dataset, df_metrics_model_train, df, date_id):
        video_exclude = path_to_dataset.split("/")[-2].split("exclude_")[-1]
        measure_combinaition = [measure for measure in list(df) if measure != "target"]
        dp = DataPreprocessing(path_to_dataset = None,batch_size= 32, isTimeSeries = True, df_dataset = df) 
        train = dp.train
        test = dp.test 
        val = dp.val
        model = self.define_model(len(measure_combinaition))
        model.compile(optimizer='adam',
                loss=tf.losses.BinaryCrossentropy(),
                metrics=["binary_accuracy","binary_crossentropy","mean_squared_error"])
        model.summary()
        model.fit(
            train, 
            validation_data= val,
            epochs=1000,
            shuffle=True,
            verbose =1,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='auto')]) 
        path_to_model_to_save = "tensorboard/model/"+date_id+ "/model_lstm_exclude_"+video_exclude
    
        model.save(path_to_model_to_save)

        _ ,binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(test)
        
        df_metrics_model_train.loc[(video_exclude),["binary_accuracy", "binary_crossentropy", "mean_squared_error"]] = [binary_accuracy, binary_crossentropy, mean_squared_error]
        return path_to_model_to_save, video_exclude, df_metrics_model_train
        
        
        
    def evaluate_model(self, model_path, video_exclude): 
        model = tf.keras.models.load_model(model_path)
            
        df_evaluate_metrics = pd.DataFrame(columns=["video_exclude", "measure_combination", "binary_accuracy", "binary_crossentropy", "mean_squared_error"]).set_index(["video_exclude","measure_combination"])
        df = self.sftp.read_remote_df(os.path.join(PATH_TO_TIME_ON_TASK_VIDEO, video_exclude, video_exclude+".csv"))
        preprocessing = DataPreprocessing(path_to_dataset = None, isTimeSeries = True, batch_size = 1, evaluate = True, df_dataset=df)
        preprocessing.dataset = preprocessing.dataset.batch(preprocessing.batch_size)
        
        _ ,binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(preprocessing.dataset)
        df_evaluate_metrics.loc[(video_exclude), ["binary_accuracy", "binary_crossentropy", "mean_squared_error"]] = [binary_accuracy, binary_crossentropy, mean_squared_error]
        
        predictions = model.predict(preprocessing.dataset)
        
        measure_list = list(df)
        df_pred = pd.DataFrame(np.squeeze(predictions), columns = [measure for measure in measure_list if measure != "target"])
        for idx in df_pred.index:
            df_pred.loc[idx, "pred_mean"] = df_pred.loc[idx].mean() 
            df_pred.loc[idx, "pred_max"] = df_pred.loc[idx].max() 
        df_pred.loc[lambda df_pred: df_pred["pred_mean"] < 0.5,"target_pred_mean"] = 0
        df_pred.loc[lambda df_pred: df_pred["pred_mean"] >= 0.5,"target_pred_mean"] = 1
        df_pred.loc[lambda df_pred: df_pred["pred_max"] < 0.5,"target_pred_max"] = 0
        df_pred.loc[lambda df_pred: df_pred["pred_max"] >= 0.5,"target_pred_max"] = 1
        df_pred["target_real"] = df["target"]
        
        path_to_csv_pred =os.join(PATH_TO_RESULTS_CROSS_PREDICTIONS, video_exclude, "/pred.csv") 
        path_to_csv_metrics =os.join(PATH_TO_RESULTS_CROSS_PREDICTIONS, video_exclude, "/metrics.csv") 
        self.sftp.save_remote_df(path_to_csv_pred, df_pred, index = False)
        self.sftp.save_remote_df(path_to_csv_metrics, df_evaluate_metrics)

    def train_cross_model(self):
        folder_dataset = self.sftp.list_dir_remote(self.cross_dataset_path)
        path_dataset = [ os.path.join(self.cross_dataset_path, folder, "dataset.csv") for folder in  folder_dataset ]
        date_id = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        df_metrics_model_train = pd.DataFrame(columns=["video_exclude","binary_accuracy","binary_crossentropy","mean_squared_error"]).set_index(["video_exclude"])
        for dataset in path_dataset:
            df = self.sftp.read_remote_df(dataset)
            path_to_model, video_to_exclude, df_metrics_model_train = self.train_evaluate_model(dataset, df_metrics_model_train, df, date_id)
            self.evaluate_model(path_to_model, video_to_exclude)
        path_to_csv_metrics_model_train = os.path.join(PATH_TO_RESULTS_CROSS_PREDICTIONS,"metrics_train_model.csv")
        self.sftp.save_remote_df(path_to_csv_metrics_model_train, df_metrics_model_train)
        
    """   
    def train_cross_measure_model(self):
        cross_combinations = lambda df : sum([list(map(list, combinations(df, i))) for i in range(len(df) + 1) if i != 0],[])
        cross_dataset_path = PATH_TO_TIME_ON_TASK_CROSS
        folder_dataset = self.sftp.list_dir_remote(cross_dataset_path)
        path_dataset = [ cross_dataset_path + folder + "/dataset.csv" for folder in  folder_dataset ]
        date_id = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        df_metrics_model_train = pd.DataFrame(columns=["video_exclude","measure_combination","binary_accuracy","binary_crossentropy","mean_squared_error"]).set_index(["video_exclude","measure_combination"])
        for dataset in path_dataset:
            df = self.sftp.read_remote_df(dataset)
            cross_combinations_measure = cross_combinations([measure for measure in list(df) if measure != "target"])
            for measures in cross_combinations_measure:
                sub_df = df[measures + ["target"]]
                path_to_model, video_to_exclude, df_metrics_model_train = self.train_evaluate_model(dataset, df_metrics_model_train, sub_df, date_id)
                self.evaluate_model(path_to_model, video_to_exclude, measures)
        path_to_csv_metrics_model_train = os.path.join(PATH_TO_RESULTS_CROSS_PREDICTIONS,"metrics_train_model.csv")
        self.sftp.save_remote_df(path_to_csv_metrics_model_train, df_metrics_model_train)
    """