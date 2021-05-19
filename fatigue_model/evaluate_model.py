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


model_path = "fatigue_model/model_save/20210518-171300/model_0"

model = tf.keras.models.load_model(model_path)
preprocessing = DataPreprocessing("data/stage_data_out/dataset_temporal/Irba_40_min/DESFAM-F_H92_VENDREDI/DESFAM-F_H92_VENDREDI.csv", isTimeSeries = True, batch_size = 1, evaluate = True)
preprocessing.dataset = preprocessing.dataset.batch(preprocessing.batch_size)
for feature_batch, label_batch in preprocessing.dataset.take(1):
    print('A shape of features:', tf.rank(feature_batch))
    print('A shape of targets:', tf.rank(label_batch.shape))
    print('A shape of features:', feature_batch.shape)
    print('A shape of targets:', label_batch.shape)
    print('A batch of features:', feature_batch.numpy())
    print('A batch of targets:', label_batch.numpy())

_ ,binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(preprocessing.dataset)

print("binary_accuracy on test : " + str(binary_accuracy) )
print("binary_crossentropy on test : " + str(binary_crossentropy) )
print("mean_squared_error on test : " + str(mean_squared_error) )


predictions = model.predict(preprocessing.dataset)

measure_list = list(pd.read_csv("data/stage_data_out/dataset_temporal/Irba_40_min/DESFAM-F_H92_VENDREDI/DESFAM-F_H92_VENDREDI.csv"))
df = pd.DataFrame(np.squeeze(predictions), columns = [measure for measure in measure_list if measure != "target"])
df.to_csv("pred.csv")
print(df)
y_pred_list = []
y_pred = []
for idx in df.index:
    df.loc[idx, "pred_mean"] = df.loc[idx].mean() 
    df.loc[idx, "pred_max"] = df.loc[idx].max() 

print(df)
df.loc[lambda df: df["pred_mean"] < 0.5,"target_pred_mean"] = 0
df.loc[lambda df: df["pred_mean"] >= 0.5,"target_pred_mean"] = 1
df.loc[lambda df: df["pred_max"] < 0.5,"target_pred_max"] = 0
df.loc[lambda df: df["pred_max"] >= 0.5,"target_pred_max"] = 1
print(df)
df.to_csv("data/stage_data_out/predictions/pred.csv")
