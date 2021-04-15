import tensorflow as tf
import pandas as pd
import numpy as np

df = pd.read_csv("data/stage_data_out/dataset/Merge_Dataset/Merge_Dataset.csv", index_col=0)
target = df.pop('Target')

model = tf.keras.models.load_model("tensorboard/model/20210414-110714/model_0")

predictions = model.predict(dict(df))

df_prediction = pd.DataFrame(predictions, index = df.index)
df_prediction.insert(1, "target", target)
df_prediction.to_csv("data/stage_data_out/dataset/prediciton.csv")

print((df_prediction))
print((df_prediction[df_prediction[0]>0.9]))
print("test")

"""
y_pred = []
for pred in predictions:
    if pred >=0.5:
        y_pred.append(1)
    else: y_pred.append(0)
y_true = target.tolist()

conf_mat = skm.confusion_matrix(y_true, y_pred)
print(conf_mat)
"""