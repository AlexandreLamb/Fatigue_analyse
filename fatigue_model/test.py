import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def parse_time_series( columns):
        array_serie=[]
        for serie in list(columns):
            parse_serie = serie.replace("[","").replace("]","").split(",")
            parse_serie = [ float(element_floated) for element_floated in parse_serie ]
            array_serie.append(parse_serie)
        return array_serie
"""    
df = pd.read_csv("data/stage_data_out/dataset/DESFAM_Semaine 2-Vendredi_PVT_H64/ear_10.csv", index_col=0)
target = df.pop('target')
time_series = parse_time_series(df["ear_10"])"""
"""time_label = [np.ones(1)*label for label in list(target)]
dataset = tf.keras.preprocessing.timeseries_dataset_from_array(time_series, time_label, sequence_length = 1, batch_size=1)
for feature_batch, label_batch in dataset.take(1):
    print('A batch of features:', feature_batch.numpy())
    print('A batch of targets:', label_batch.numpy() )
"""
model = tf.keras.models.load_model("fatigue_model/model_1")
model.summary()
predictions = model.predict([[[[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]]]])
"""
df_prediction = pd.DataFrame(predictions, index = df.index)
df_prediction.insert(1, "target", target)
df_prediction.to_csv("data/stage_data_out/dataset/prediciton.csv")
"""
print(predictions)
"""
y_pred = []
for pred in predictions:
    if pred >=0.5:
        y_pred.append(1)
    else: y_pred.append(0)
y_true = target.tolist()

conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)
"""