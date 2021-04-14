#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
import sklearn.metrics
import datetime


# ## Load Data and Create Dataset

# In[8]:


df = pd.read_csv("data/stage_data_out/dataset/Merge_Dataset/Merge_Dataset.csv", index_col=0)
print(df.dtypes)
print(df.describe())
print(df.head(5))


# In[9]:


target = df.pop('Target')
dataset = tf.data.Dataset.from_tensor_slices((dict(df), target.values))
print(dataset)


# In[10]:


for feature_batch, label_batch in dataset.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ear:', feature_batch['ear'])
  print('A batch of targets:', label_batch )


# ## Splitting, Shuffing, Batching  data

# ### Splitting and Shuffling

# In[11]:


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

# In[12]:


BATCH_SIZE = 32

train = train.shuffle(buffer_size = train_size)
train = train.batch(BATCH_SIZE)

val = val.shuffle(buffer_size = val_size)
val = val.batch(BATCH_SIZE)

test = test.batch(BATCH_SIZE)


# ## Feature Engineering

# In[13]:


example_batch = next(iter(train))[0]


# In[14]:


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


# In[15]:


all_inputs = []
encoded_features = []
numerical_features = ["ear","ear_l","ear_r"]
all_inputs, encoded_features = make_numerical_feature_col(numerical_features, normalize = True)


# In[16]:


all_features = []
all_features = tf.keras.layers.concatenate(encoded_features)


# ## Model

# ## Hyper Parameter tuning (Hparams & tensor board)

# ### Define log dir

# In[17]:


logdir = "tensorboard/logs/fit/tunning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"


# ### Define model parameter

# In[18]:


HP_NUM_UNITS_1 = hp.HParam('num_units_1', hp.Discrete([32]))
HP_NUM_UNITS_2 = hp.HParam('num_units_2', hp.Discrete([512]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.5,0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu']))
HP_ACTIVATION_OUTPUT = hp.HParam('activation_output', hp.Discrete(['sigmoid']))


METRIC_BINARY_ACCURACY = "binary_accuracy"
METRIC_BINARY_CROSSENTROPY = "binary_crossentropy"
METRIC_MSE = "mean_squared_error"

NUMBER_OF_TARGET = 1
metrics = ["binary_accuracy","binary_crossentropy","mean_squared_error"]


# ### Initialize hyper parameter for the log

# In[ ]:


with tf.summary.create_file_writer(logdir).as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS_1, HP_NUM_UNITS_2, HP_DROPOUT, HP_ACTIVATION, HP_ACTIVATION_OUTPUT, HP_OPTIMIZER],
    metrics=[ hp.Metric(METRIC_BINARY_ACCURACY, display_name='Binary Accuracy'),
              hp.Metric(METRIC_BINARY_CROSSENTROPY, display_name='Binary Cross Entropy'),
              hp.Metric(METRIC_MSE, display_name='MSE'),
    ],
  )


# ### Define the model

# In[ ]:


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


# In[ ]:


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    
    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)
    
    return image


# In[ ]:


def log_confusion_matrix(epoch, logs):
    
    # Use the model to predict the values from the test_images.
    test_pred_raw = model.predict(test_images)
    
    test_pred = np.argmax(test_pred_raw, axis=1)
    
    # Calculate the confusion matrix using sklearn.metrics
    cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
    
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)
    
    # Log the confusion matrix as an image summary.
    with tf.summary.create_file_writer(logdir + '/cm').as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


# In[ ]:


def modeling(hparams):
    
    x = tf.keras.layers.BatchNormalization()(all_features)
    x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS_1],activation=hparams[HP_ACTIVATION])(x)
    x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS_2],activation=hparams[HP_ACTIVATION])(x)
    x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)
    x = tf.keras.layers.BatchNormalization()(x)

    output = tf.keras.layers.Dense(NUMBER_OF_TARGET, activation=hparams[HP_ACTIVATION_OUTPUT])(x)
    model = tf.keras.Model(all_inputs,output)
    return model


# In[24]:


def train_test_model(hparams):
    model = modeling(hparams)
    model.summary()
    model.compile(
        optimizer = hparams[HP_OPTIMIZER],
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = ["binary_accuracy","binary_crossentropy","mean_squared_error"],
    )
    model.fit(
        train, 
        validation_data= val,
        epochs=10,
        shuffle=True,
        verbose =1,
        callbacks=[ 
            tf.keras.callbacks.TensorBoard(log_dir = logdir, histogram_freq = 1),  # log metrics
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix),
            hp.KerasCallback(logdir, hparams),  # log hparams
            tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=10),
        ]
    ) 
    _, binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(test)
    return binary_accuracy, binary_crossentropy, mean_squared_error


# ### Define a method to run the the training and testing model function and logs the paramete

# In[25]:


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        binary_accuracy, binary_crossentropy, mean_squared_error = train_test_model(hparams)
        tf.summary.scalar(METRIC_BINARY_ACCURACY, binary_accuracy, step=1)
        tf.summary.scalar(METRIC_BINARY_CROSSENTROPY, binary_crossentropy, step=1)
        tf.summary.scalar(METRIC_MSE, mean_squared_error, step=1)


# ### Tunning the model

# In[26]:


session_num = 0
 
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
              run_name = "run-%d" % session_num
              print('--- Starting trial: %s' % run_name)
              print({h.name: hparams[h] for h in hparams})
              run(logdir + run_name, hparams)
              session_num += 1          


# In[1]:


get_ipython().run_line_magic('reload_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', "--logdir 'tensorboard/logs/fit' --port=8080")


# In[24]:


get_ipython().system('jupyter nbconvert --to script ann_fatiuge_test.ipynb')

