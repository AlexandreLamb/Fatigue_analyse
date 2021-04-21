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



model_path = "tensorboard/model/20210421-111446/model_ann_dense"

model = tf.keras.models.load_model(model_path)

#df = pd.read_csv("data/stage_data_out/dataset/Merge_Dataset/Merge_Dataset.csv", index_col=0)
df = pd.read_csv("data/stage_data_out/dataset_ear/dataset_ear/dataset_ear_1.csv", index_col=0)

train, test = train_test_split(df, test_size = 0.3)

target = test.pop("Target")


_ ,binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(x =dict(test), y = target)

print("binary_accuracy on test : " + str(binary_accuracy) )
print("binary_crossentropy on test : " + str(binary_crossentropy) )
print("mean_squared_error on test : " + str(mean_squared_error) )


predictions = model.predict(dict(test))
y_pred = []
for pred in predictions:
    if pred >=0.5:
        y_pred.append(1)
    else: y_pred.append(0)
y_true = target.tolist()

conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)