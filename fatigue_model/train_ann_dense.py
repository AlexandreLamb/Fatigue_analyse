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

df = pd.read_csv("data/stage_data_out/dataset_ear/dataset_ear/dataset_ear_1.csv", index_col=0)
print(df.dtypes)
print(df.describe())
print(df.head(5))

target = df.pop('Target')
dataset = tf.data.Dataset.from_tensor_slices((dict(df), target.values))
print(dataset)

for feature_batch, label_batch in dataset.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ear:', feature_batch['ear'])
    print('A batch of targets:', label_batch )
    
dataset_size = dataset.reduce(0, lambda x, _: x + 1).numpy()
dataset = dataset.shuffle(buffer_size = dataset_size)

train_size = int(0.7*dataset_size)
val_size = int(0.15*dataset_size)
test_size = int(0.15*dataset_size)

train = dataset.take(train_size)
val = dataset.skip(train_size)
val = dataset.take(val_size)
test = dataset.skip(train_size + val_size)
test = dataset.take(test_size)

train_size = train.reduce(0, lambda x, _: x + 1).numpy()
val_size = val.reduce(0, lambda x, _: x + 1).numpy()
test_size = test.reduce(0, lambda x, _: x + 1).numpy()

print("Full dataset size:", dataset_size)
print("Train dataset size:", train_size)
print("Val dataset size:", val_size)
print("Test dataset size:", test_size)    

BATCH_SIZE = 256

train = train.shuffle(buffer_size = train_size)
train = train.batch(BATCH_SIZE)

val = val.shuffle(buffer_size = val_size)
val = val.batch(BATCH_SIZE)

test = test.batch(BATCH_SIZE)
example_batch = next(iter(train))[0]


def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


def make_numerical_feature_col(numerical_column, normalize = False):
    def get_normalization_layer(name, dataset):
        # Create a Normalization layer for our feature.
        normalizer = preprocessing.Normalization()
        # Prepare a Dataset that only yields our feature.
        feature_ds = dataset.map(lambda x, y: x[name])
        print(feature_ds)
        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)
        print(feature_ds)
        return normalizer
    
    for column_name in numerical_column:
        numeric_col = tf.keras.Input(shape=(1,), name=column_name)
        if normalize : 
            normalization_layer = get_normalization_layer(column_name, train)
            encoded_numeric_col = normalization_layer(numeric_col) 
        else : 
            encoded_numeric_col = feature_column.numeric_column(column_name)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)
    return all_inputs, encoded_features

all_inputs = []
encoded_features = []
numerical_features = ["ear","eyebrow_nose","eyebrow_eye", "jaw_dropping", "eye_area"]
all_inputs, encoded_features = make_numerical_feature_col(numerical_features, normalize = True)

all_features = []
all_features = tf.keras.layers.concatenate(encoded_features)

x = tf.keras.layers.BatchNormalization()(all_features)
x = tf.keras.layers.Dense(32,activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512,activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512,activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512,activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.BatchNormalization()(x)

output = tf.keras.layers.Dense(1, activation="sigmoid")(x)


model = tf.keras.Model(all_inputs,output)

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

print(test)
