import tensorflow as tf
import sklearn.metrics as skm
import pandas as pd
import numpy as np

df = pd.read_csv("data/stage_data_out/dataset/Merge_Dataset/Merge_Dataset.csv", index_col=0)
target = df.pop('Target')

model = tf.keras.models.load_model("tensorboard/model")

predictions = model.predict(dict(df))

y_pred = []
for pred in predictions:
    if pred >=0.5:
        y_pred.append(1)
    else: y_pred.append(0)
y_true = target.tolist()

conf_mat = skm.confusion_matrix(y_true, y_pred)
print(conf_mat)
