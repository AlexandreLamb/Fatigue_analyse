
import tensorflow as tf 
import pandas as pd 
import numpy as np
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import time
class DataPreprocessing():
    def __init__(self, path_to_dataset, isTimeSeries=False, batch_size=32, evaluate = False):
        self.path_to_dataset = path_to_dataset
        self.dataset = None
        self.train = None
        self.test = None
        self.val = None
        
        self.all_features = []
        self.all_inputs = []
        self.encoded_features = []
        self.numerical_column = None
        
        self.batch_size = batch_size
        self.isTimeSeries = isTimeSeries
        self.evaluate = evaluate
        self.initialize()
    
    def make_numerical_feature_col(self, normalize = False):    
        def get_normalization_layer(name, dataset):
            # Create a Normalization layer for our feature.
            normalizer = preprocessing.Normalization()
            # Prepare a Dataset that only yields our feature.
            feature_ds = dataset.map(lambda x, y: x[name])
            # Learn the statistics of the data.
            normalizer.adapt(feature_ds)
            return normalizer
        
        for column_name in self.numerical_column:
            numeric_col = tf.keras.Input(shape=(1,), name=column_name)
            if normalize : 
                normalization_layer = get_normalization_layer(column_name, self.train)
                encoded_numeric_col = normalization_layer(numeric_col) 
            else : 
                encoded_numeric_col = feature_column.numeric_column(column_name)
            self.all_inputs.append(numeric_col)
            self.encoded_features.append(encoded_numeric_col)
        self.all_features = tf.keras.layers.concatenate(self.encoded_features)
    
    def make_train_val_test_dataset(self):     
        dataset_size = self.dataset.reduce(0, lambda x, _: x + 1).numpy()
        self.dataset = self.dataset.shuffle(buffer_size = dataset_size)

        train_size = int(0.7*dataset_size)
        val_size = int(0.15*dataset_size)
        test_size = int(0.15*dataset_size)

        self.train = self.dataset.take(train_size)
        self.val = self.dataset.skip(train_size)
        self.val = self.dataset.take(val_size)
        self.test = self.dataset.skip(train_size + val_size)
        self.test = self.dataset.take(test_size)

        train_size = self.train.reduce(0, lambda x, _: x + 1).numpy()
        val_size = self.val.reduce(0, lambda x, _: x + 1).numpy()
        test_size = self.test.reduce(0, lambda x, _: x + 1).numpy()

        print("Full dataset size:", dataset_size)
        print("Train dataset size:", train_size)
        print("Val dataset size:", val_size)
        print("Test dataset size:", test_size)

        #if self.isTimeSeries == False:
        self.train = self.train.batch(self.batch_size)
        self.val = self.val.batch(self.batch_size)
        self.test = self.test.batch(self.batch_size)

        self.train = self.train.shuffle(buffer_size = train_size)
        self.val = self.val.shuffle(buffer_size = val_size)
        
        for feature_batch, label_batch in self.train.take(1):
                print('A shape of features:', tf.rank(feature_batch))
                print('A shape of targets:', tf.rank(label_batch.shape))
                print('A shape of features:', feature_batch.shape)
                print('A shape of targets:', label_batch.shape)
                print('A batch of features:', feature_batch.numpy())
                print('A batch of targets:', label_batch.numpy())

        
    def load_dataset(self):                  
        #df = pd.read_csv(self.path_to_dataset, index_col=0)
        df = pd.read_csv(self.path_to_dataset)
        if self.isTimeSeries :          
            target = df.pop("target")
            time_label = [np.ones(1)*label for label in list(target)]
            time_series = np.array(self.parse_time_series(df), dtype=np.float32)
            #time_series = np.squeeze(time_series)
            #self.dataset = tf.keras.preprocessing.timeseries_dataset_from_array(time_series, time_label, sequence_length = 1, batch_size=self.batch_size)
            self.append_additional_features(time_series,[[1],[2]])
            
            self.dataset = tf.data.Dataset.from_tensor_slices((time_series, time_label))
            #self.dataset = self.dataset.map(lambda features, label: (tf.squeeze(features), label))
            
           
        else :
            target = df.pop('Target')
            self.numerical_column = list(df.columns)
            self.dataset = tf.data.Dataset.from_tensor_slices((dict(df), target.values))

    def initialize(self):
        self.load_dataset()
        if self.evaluate == False:
            self.make_train_val_test_dataset()
        else : 
            self.dataset.batch(self.batch_size)
        if self.isTimeSeries == False :
            self.make_numerical_feature_col(normalize=True)
            
    def noramlize_time_series(self, time_series): 
        df_to_normalize = pd.DataFrame(time_series)
        mean = df_to_normalize.mean(axis = 1)
        std = df_to_normalize.std(axis = 1)
        df_to_normalize = df_to_normalize.sub(mean, axis = 0) 
        df_to_normalize = df_to_normalize.div(std, axis = 0)
        return list(df_to_normalize.values)
    
    def parse_time_serie(self, columns):
        array_serie=[]
        for serie in list(columns):
            parse_serie = serie.replace("[","").replace("]","").split(",")
            parse_serie = [ float(element_floated) for element_floated in parse_serie ]
            array_serie.append(parse_serie)
        return array_serie
    
    def parse_time_series(self, df):
        array_series = []
        array_measure = []
        list_measures = [measure for measure in list(df) if measure != "target"]
        for series in df[list_measures].values:
            parse_series = [serie.replace("[","").replace("]","").split(",") for serie in series]
            for float_serie in parse_series:
                array_series.append([ float(element_floated) for element_floated in float_serie ])
            array_measure.append(array_series)
            array_series = []
        return array_measure
    
    def append_additional_features(self, time_series, features_addition):
        ##TODO : see how to fix input_dim of layer
        embedding_layer = tf.keras.layers.Embedding(len(time_series[0][0]), len(time_series[0][0]))
        features_addition_embeding = embedding_layer(tf.constant(features_addition))
        for index, serie in enumerate(time_series):
            serie =  np.append(serie,np.squeeze(features_addition_embeding.numpy()), axis=0)
        return time_series
           

"""
dp = DataPreprocessing("data/stage_data_out/dataset_ear/DESFAM-F_H99_VENDREDI/DESFAM-F_H99_VENDREDI.csv",batch_size=12 ,isTimeSeries = True)
for feature_batch, label_batch in dp.train.take(1):
    print('A batch of features:', feature_batch)
    print('A batch of targets:', label_batch)
"""