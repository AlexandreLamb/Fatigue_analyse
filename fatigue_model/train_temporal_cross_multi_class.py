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
import copy
from itertools import combinations
from tensorflow.python.keras.activations import sigmoid 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fatigue_model.data_processing import DataPreprocessing
from dotenv import load_dotenv
load_dotenv("env_file/.env_path")

from database_connector import SFTPConnector

class CrossValidationMultiClass:
    def __init__(self, cross_dataset_path, video_dataset_path, prediction_dataset_path):
        self.sftp = SFTPConnector()
        self.cross_dataset_path = cross_dataset_path
        self.video_dataset_path = video_dataset_path
        self.prediction_dataset_path = prediction_dataset_path
        self.date_id = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    def define_model(self, measure_len) :
        return tf.keras.models.Sequential([
         tf.keras.layers.LSTM(64, input_shape=(measure_len,30), return_sequences=True),
        #tf.keras.layers.Dense(units=32, activation = "relu"),
        tf.keras.layers.LSTM(64, return_sequences=True),
        #tf.keras.layers.Dense(units=64, activation = "relu"),
        tf.keras.layers.LSTM(64, return_sequences=True),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(units=4, activation = "sigmoid"),

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
                loss=tf.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.metrics.SparseCategoricalAccuracy(), tf.metrics.SparseCategoricalCrossentropy()])
        model.summary()
        model.fit(
            train, 
            epochs=1000,
            validation_data= val,
            shuffle=True,
            verbose =1,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='auto')]) 
        path_to_model_to_save = "tensorboard/model/"+date_id+ "/model_lstm_exclude_"+video_exclude
    
        model.save(path_to_model_to_save)

        _ , sparse_categorical_accuracy, sparse_categorical_crossentropy = model.evaluate(test)
        
        df_metrics_model_train.loc[(video_exclude),["sparse_categorical_accuracy", "sparse_categorical_crossentropy"]] = [sparse_categorical_accuracy, sparse_categorical_crossentropy]
        
        return path_to_model_to_save, video_exclude, df_metrics_model_train
        
        
        
    def test_model(self, model_path, video_exclude, df_evaluate_metrics): 
        model = tf.keras.models.load_model(model_path)
            
        df = self.sftp.read_remote_df(os.path.join(self.video_dataset_path, video_exclude, video_exclude+".csv"))
        df_copy = copy.copy(df)
        preprocessing = DataPreprocessing(path_to_dataset = None, isTimeSeries = True, batch_size = 1, evaluate = True, df_dataset=df)
        
        
        #shuffle_dataset =  copy.copy(preprocessing.dataset)
        #shuffle_dataset = shuffle_dataset.shuffle(buffer_size = preprocessing.batch_size)
        preprocessing.dataset = preprocessing.dataset.batch(preprocessing.batch_size)
        
        _ , sparse_categorical_accuracy, sparse_categorical_crossentropy = model.evaluate(preprocessing.dataset)
        df_evaluate_metrics.loc[(video_exclude), ["sparse_categorical_accuracy", "sparse_categorical_crossentropy"]] = [sparse_categorical_accuracy, sparse_categorical_crossentropy]
        predictions = model.predict(preprocessing.dataset)
        print(predictions)
        measure_list = list(df_copy)
        df_pred = pd.DataFrame(np.squeeze(predictions), columns = ["target_pred"])
        """ 
        df_pred.loc[lambda df_pred: df_pred["target_pred"] < 0.5,"target_round"] = 0
        df_pred.loc[lambda df_pred: df_pred["target_pred"] >= 0.5,"target_round"] = 1
        """
        df_pred["target_real"] = df_copy["target"]
        
        path_to_csv_pred =os.path.join(self.prediction_dataset_path, self.date_id, video_exclude, "pred.csv") 
        self.sftp.save_remote_df(path_to_csv_pred, df_pred, index = False)
        return df_evaluate_metrics

    def train_cross_model(self):
        folder_dataset = self.sftp.list_dir_remote(self.cross_dataset_path)
        path_dataset = [ os.path.join(self.cross_dataset_path, folder, "dataset.csv") for folder in  folder_dataset ]
        
        df_metrics_model_train = pd.DataFrame(columns=["sparse_categorical_accuracy", "sparse_categorical_crossentropy"]).set_index(["video_exclude"])
        df_evaluate_metrics = pd.DataFrame(columns=["sparse_categorical_accuracy", "sparse_categorical_crossentropy"]).set_index(["video_exclude"])
    
        for dataset in path_dataset:
            df = self.sftp.read_remote_df(dataset)
            path_to_model, video_to_exclude, df_metrics_model_train = self.train_evaluate_model(dataset, df_metrics_model_train, df, self.date_id)
            df_evaluate_metrics = self.test_model(path_to_model, video_to_exclude, df_evaluate_metrics)
            
        path_to_csv_metrics_model_train = os.path.join(self.prediction_dataset_path, self.date_id, "metrics_train_model.csv")
        path_to_csv_metrics_model_evaluate = os.path.join(self.prediction_dataset_path, self.date_id, "metrics_evaluate_model.csv")
        
        self.sftp.save_remote_df(path_to_csv_metrics_model_train, df_metrics_model_train)
        self.sftp.save_remote_df(path_to_csv_metrics_model_evaluate, df_evaluate_metrics)
        