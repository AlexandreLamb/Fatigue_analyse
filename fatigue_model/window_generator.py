import pandas as pd
import numpy as np

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, df, label_columns=None):
        # Store the raw data.
        self.df = df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
"""
df = pd.read_csv("data/stage_data_out/dataset/DESFAM_Semaine 2-Vendredi_PVT_H64/ear_10.csv", index_col="frame")
w1 = WindowGenerator(30,1,1, df, ["Target"])
print(w1)
"""


import tensorflow as tf
import numpy as np
df = pd.read_csv("data/stage_data_out/dataset/DESFAM_Semaine 2-Vendredi_PVT_H64/ear_10.csv", index_col=0)


time_serie = np.fromstring(df["ear_10"][0], dtype=float)
time_label = np.ones(len(time_serie))
print(len(time_serie))

dataset = tf.keras.preprocessing.timeseries_dataset_from_array(time_serie, time_label, sequence_length = len(time_serie))

for feature_batch, label_batch in dataset.take(1):
    print('A batch of features:', feature_batch.numpy())
    print('A batch of targets:', label_batch.numpy() )

