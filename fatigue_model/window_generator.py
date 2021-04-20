import pandas as pd
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fatigue_model.data_processing import DataPreprocessing
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

def compile_and_fit(model, preprocessing, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.BinaryAccuracy()])
    history = model.fit(preprocessing.train, epochs=20, validation_data=preprocessing.val, callbacks=[early_stopping])
    return history


df = pd.read_csv("data/stage_data_out/dataset/DESFAM_Semaine 2-Vendredi_PVT_H64/ear_10.csv", index_col=0)

dp = DataPreprocessing(1,"data/stage_data_out/dataset/DESFAM_Semaine 2-Vendredi_PVT_H64/ear_10.csv", True)
for feature_batch, label_batch in dp.dataset.take(1):
    print('A batch of features:', feature_batch.numpy())
    print('A batch of targets:', label_batch.numpy() )

dense = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
])

history = compile_and_fit(dense, dp)
dense.save("fatigue_model/model_1")
val_performance ={}
performance = {}
val_performance['Dense'] = dense.evaluate(dp.val)
performance['Dense'] = dense.evaluate(dp.test, verbose=0)

print(val_performance)
print(performance)

y_true = np.concatenate([y for x, y in dp.dataset], axis=0)

predictions = dense.predict(dp.dataset)
print(predictions)
y_pred = []
for pred in predictions:
    if pred >=0.5:
        y_pred.append(1)
    else: y_pred.append(0)

conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)


"""
def parse_time_series(columns):
    array_serie=[]
    for serie in list(columns):
        parse_serie = serie.replace("[","").replace("]","").split(",")
        parse_serie = [ float(element_floated) for element_floated in parse_serie ]
        array_serie.append(parse_serie)
    return array_serie



time_series = parse_time_series(df["ear_10"])

time_label = [np.ones(1)*label for label in list(df["target"])]

print(len(time_series))

dataset = tf.keras.preprocessing.timeseries_dataset_from_array(time_series, time_label, sequence_length = 1, batch_size=3)

for feature_batch, label_batch in dataset.take(1):
    print('A batch of features:', feature_batch.numpy())
    print('A batch of targets:', label_batch.numpy() )

"""