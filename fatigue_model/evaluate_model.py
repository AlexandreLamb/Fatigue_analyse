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



model_path = "/home/simeon/Documents/model_1250"

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
y_pred_list = []
y_pred = []
print(np.squeeze(predictions))
for pred in np.squeeze(predictions):
    y_pred_list.append(pred.mean())
"""    if pred.mean() >=0.5:
        y_pred.append(1)
    else: y_pred.append(0)
y_true = target.tolist()

conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)
"""
df = pd.DataFrame(y_pred_list, columns=["pred"])
print(df)
df.loc[lambda df: df["pred"] < 0.5,"target_pred"] = 0
df.loc[lambda df: df["pred"] >= 0.5,"target_pred"] = 1
print(df)
df.to_csv("data/stage_data_out/predictions/pred.csv")
df_plot = pd.DataFrame(columns=["pred"])

for frame in df.index :
    df_plot.loc[frame] =  df[df.loc[ frame * (600):  (frame+1) * (600) ]]["target_pred"].mean()

df_plot["target_pred"].plot(title="target_pred" + " by min")
plt.show()
df.plot()
plt.show()
print(df)