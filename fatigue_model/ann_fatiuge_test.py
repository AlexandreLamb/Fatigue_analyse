#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
import datetime


# ## Load Data and Create Dataset

# In[4]:


df = pd.read_csv("data/stage_data_out/dataset/Merge_Dataset/Merge_Dataset.csv", index_col=0)
print(df.dtypes)
print(df.describe())
print(df.head(6))


# In[5]:


target = df.pop('Target')
dataset = tf.data.Dataset.from_tensor_slices((dict(df), target.values))
print(dataset)


# In[6]:


for feature_batch, label_batch in dataset.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ear:', feature_batch['ear'])
  print('A batch of targets:', label_batch )


# ## Splitting, Shuffing, Batching  data

# ### Splitting and Shuffling

# In[7]:


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


# ### Shuffling, Batching

# In[8]:


BATCH_SIZE = 32

train = train.shuffle(buffer_size = train_size)
train = train.batch(BATCH_SIZE)

val = val.shuffle(buffer_size = val_size)
val = val.batch(BATCH_SIZE)

test = test.batch(BATCH_SIZE)


# ## Feature Engineering

# In[9]:


example_batch = next(iter(train))[0]


# In[10]:


def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

# POnly if we have features with different scale
def normalize_numerical_features(df, features):
  def get_mean_std(x):
    return df[x].mean(), df[x].std()
  for column in features: 
    mean, std = get_mean_std(column)
    def z_score(col):
      return (col - mean)/std    
    def _numeric_column_normalized(column_name, normalizer_fn):
      return tf.feature_column.numeric_column(column_name, normalizer_fn=normalizer_fn)
    return _numeric_column_normalized(column,z_score)
  
def get_normalization_layer(name, dataset):
  # Create a Normalization layer for our feature.
  normalizer = preprocessing.Normalization()
  # Prepare a Dataset that only yields our feature.
  feature_ds = dataset.map(lambda x, y: x[name])
  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)
  return normalizer

def make_numerical_feature_col(numerical_column, normalize = False):
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


# In[11]:


all_inputs = []
encoded_features = []
numerical_features = ["ear","ear_l","ear_r"]
all_inputs, encoded_features = make_numerical_feature_col(numerical_features, normalize = True)


# In[12]:


all_features = []
all_features = tf.keras.layers.concatenate(encoded_features)


# ## Model

# ## Hyper Parameter tuning (Hparams & tensor board)

# ### Define log dir

# In[13]:


logdir = "tensorboard/logs/fit/tunning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"


# ### Define model parameter

# In[14]:


HP_NUM_UNITS_1 = hp.HParam('num_units_1', hp.Discrete([32,64]))
HP_NUM_UNITS_2 = hp.HParam('num_units_2', hp.Discrete([512]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.5, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu']))
HP_ACTIVATION_OUTPUT = hp.HParam('activation_output', hp.Discrete(['sigmoid']))


METRIC_BINARY_ACCURACY = "binary_accuracy"
METRIC_BINARY_CROSSENTROPY = "binary_crossentropy"
METRIC_MSE = "mean_squared_error"

NUMBER_OF_TARGET = 1
metrics = ["binary_accuracy","binary_crossentropy","mean_squared_error"]


# ### Initialize hyper parameter for the log

# In[15]:


with tf.summary.create_file_writer(logdir).as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS_1, HP_NUM_UNITS_2, HP_DROPOUT, HP_ACTIVATION, HP_ACTIVATION_OUTPUT, HP_OPTIMIZER],
    metrics=[ hp.Metric(METRIC_BINARY_ACCURACY, display_name='Binary Accuracy'),
              hp.Metric(METRIC_BINARY_CROSSENTROPY, display_name='Binary Cross Entropy'),
              hp.Metric(METRIC_MSE, display_name='MSE'),
    ],
  )


# ### Define the model

# In[20]:


def modeling(hparams):
    
    x = tf.keras.layers.BatchNormalization()(all_features)
    x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS_1],activation=hparams[HP_ACTIVATION])(x)
    x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS_2],activation=hparams[HP_ACTIVATION])(x)
    x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS_2],activation=hparams[HP_ACTIVATION])(x)
    x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)
    x = tf.keras.layers.BatchNormalization()(x)

    output = tf.keras.layers.Dense(NUMBER_OF_TARGET, activation=hparams[HP_ACTIVATION_OUTPUT])(x)
    model = tf.keras.Model(all_inputs,output)
    return model


# In[25]:


def train_test_model(hparams, session_num):
    model = modeling(hparams)
    model.summary()
    model.compile(
        optimizer = hparams[HP_OPTIMIZER],
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = ["binary_accuracy","binary_crossentropy","mean_squared_error"],
    )
    model.fit(
        train, 
        validation_data= val,
        epochs=30,
        shuffle=True,
        verbose =1,
        callbacks=[ 
            tf.keras.callbacks.TensorBoard(log_dir = logdir),  # log metrics
            hp.KerasCallback(logdir, hparams),  # log hparams
            tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10),
        ]
    ) 
    model.save("tensorboard/model/"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + "/model_" + str(session_num))
    _, binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(test)
    return binary_accuracy, binary_crossentropy, mean_squared_error


# ### Define a method to run the the training and testing model function and logs the paramete

# In[18]:


def run(run_dir, hparams, session_num):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        binary_accuracy, binary_crossentropy, mean_squared_error = train_test_model(hparams, session_num)
        tf.summary.scalar(METRIC_BINARY_ACCURACY, binary_accuracy, step=1)
        tf.summary.scalar(METRIC_BINARY_CROSSENTROPY, binary_crossentropy, step=1)
        tf.summary.scalar(METRIC_MSE, mean_squared_error, step=1)


# ### Tunning the model

# In[24]:


session_num = 0
import time

for num_units_1 in HP_NUM_UNITS_1.domain.values:
  for num_units_2 in HP_NUM_UNITS_2.domain.values:
      for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
          for activation in HP_ACTIVATION.domain.values:
            for activation_output in HP_ACTIVATION_OUTPUT.domain.values:
              hparams = {
                HP_NUM_UNITS_1: num_units_1,
                HP_NUM_UNITS_2: num_units_2,
                HP_DROPOUT : dropout_rate,
                HP_OPTIMIZER: optimizer,
                HP_ACTIVATION: activation,
                HP_ACTIVATION_OUTPUT: activation_output
              }
              print(hparams)
              run_name = "run-%d" % session_num
              print('--- Starting trial: %s' % run_name)
              print({h.name: hparams[h] for h in hparams})
              print({type(h) for h in hparams})
              print(hparams[HP_NUM_UNITS_1])
              print(type(hparams[HP_NUM_UNITS_1]))
              time.sleep(60)
              run(logdir + run_name, hparams, session_num)
              session_num += 1          


# In[1]:


get_ipython().run_line_magic('reload_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', "--logdir 'tensorboard/logs/fit' --port=8080")


# In[24]:


get_ipython().system('jupyter nbconvert --to script ann_fatiuge_test.ipynb')

