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
from tensorflow.python.keras.backend import sparse_categorical_crossentropy
from tensorflow.python.keras.metrics import sparse_categorical_accuracy 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fatigue_model.data_processing import DataPreprocessing

def train_temporal():   
    df = pd.read_csv("/home/simeon/Documents/Fatigue_analyse/dataset_merge_30_2021_07_28_14_00.csv")
    dp = DataPreprocessing(path_to_dataset=None, batch_size= 32, isTimeSeries = True, df_dataset=df) 

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
        tf.keras.layers.LSTM(64, return_sequences=True),
        #tf.keras.layers.Dense(units=64, activation = "relu"),
        tf.keras.layers.LSTM(64, return_sequences=True),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(units=4, activation = "sigmoid"),
    ])
    model.summary()
    model.compile(  optimizer='adam',
                    loss=tf.losses.SparseCategoricalCrossentropy(),
                    metrics=[tf.metrics.SparseCategoricalAccuracy(), tf.metrics.SparseCategoricalCrossentropy()])
    history = model.fit(
            train, 
            validation_data= val,
            epochs=1000,
            shuffle=True,
            verbose =1,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')]) 

    #model.save("tensorboard/model/"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + "/model_lstm")
    _ , sparse_categorical_accuracy, sparse_categorical_crossentropy = model.evaluate(test)

    print("sparse_categorical_accuracy on test : " + str(sparse_categorical_accuracy) )
    print("sparse_categorical_crossentropy on test : " + str(sparse_categorical_crossentropy) )
    plt.plot(history.epoch, history.history["loss"], 'g', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

train_temporal()