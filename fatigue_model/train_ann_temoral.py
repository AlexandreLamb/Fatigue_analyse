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
dp = DataPreprocessing(path_to_dataset = "data/stage_data_out/dataset_temporal/Merge_Dataset/dataset_merge_30_17_21_26_05_2021.csv",batch_size= 32, isTimeSeries = True) 
for feature_batch, label_batch in dp.train.take(1):
    print('A rank of features:', tf.rank(feature_batch))
    print('A rank of targets:', tf.rank(label_batch.shape))
    print('A shape of features:', feature_batch.shape)
    print('A shape of targets:', label_batch.shape)
    print('A batch of features:', feature_batch.numpy())
    print('A batch of targets:', label_batch.numpy())

train = dp.train
test = dp.test 
val = dp.val

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64,input_shape=(5,30) , return_sequences=True),
    #tf.keras.layers.Dense(units=32, activation = "relu"),
    tf.keras.layers.LSTM(64,input_shape=(5,30) , return_sequences=True),
    #tf.keras.layers.Dense(units=64, activation = "relu"),
    tf.keras.layers.LSTM(64,input_shape=(5,30) , return_sequences=True),

    tf.keras.layers.Dense(units=1, activation = "sigmoid")
])

model.compile(optimizer='rmsprop',
              loss=tf.losses.BinaryCrossentropy(),
              metrics=["binary_accuracy","binary_crossentropy","mean_squared_error"])
#model.summary()
model.fit(
        train, 
        validation_data= val,
        epochs=1000,
        shuffle=True,
        verbose =1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')]) 

model.save("tensorboard/model/"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + "/model_lstm")
_ ,binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(test)

print("binary_accuracy on test : " + str(binary_accuracy) )
print("binary_crossentropy on test : " + str(binary_crossentropy) )
print("mean_squared_error on test : " + str(mean_squared_error) )

