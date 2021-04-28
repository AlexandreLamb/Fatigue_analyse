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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_processing import DataPreprocessing

dp = DataPreprocessing(path_to_dataset = "/home/simeon/Desktop/Fatigue_analyse/data/stage_data_out/measure_dataset/DESFAM_Semaine 2-Vendredi_Go-NoGo_H63/DESFAM_Semaine 2-Vendredi_Go-NoGo_H63_90.csv",batch_size= 1, isTimeSeries = True) 
for feature_batch, label_batch in dp.train.take(1):
    print('A batch of features:', feature_batch.shape)
    print('A batch of targets:', label_batch.shape)

train = dp.train
test = dp.test 
val = dp.val

"""model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32,activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])"""
model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=["binary_accuracy","binary_crossentropy","mean_squared_error"])

model.fit(
        train, 
        validation_data= val,
        epochs=30,
        shuffle=True,
        verbose =1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')]) 

model.save("tensorboard/model/"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + "/model_ann_dense")
_ ,binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(test)

print("binary_accuracy on test : " + str(binary_accuracy) )
print("binary_crossentropy on test : " + str(binary_crossentropy) )
print("mean_squared_error on test : " + str(mean_squared_error) )

