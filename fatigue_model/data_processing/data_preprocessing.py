
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow import constant

import time, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from logger import logging

class DataPreprocessing():
    def __init__(self, path_to_dataset, isTimeSeries=False, batch_size=32, evaluate = False):
        self.path_to_dataset = path_to_dataset
        self.dataset = None
        self.train = None
        self.test = None
        self.val = None
        self.dataset_size = None
        self.all_features = []
        self.all_inputs = []
        self.encoded_features = []
        self.numerical_column = None
        
        self.batch_size = batch_size
        self.isTimeSeries = isTimeSeries
        self.evaluate = evaluate
        t = time.process_time()
        self.initialize()
        elapsed_time = time.process_time() - t
        logging.info("time function : initialize : " + str(elapsed_time))

    
    def make_train_val_test_dataset(self):
        
        self.dataset = self.dataset.shuffle(buffer_size = self.dataset_size)

        train_size = int(0.7*self.dataset_size)
        val_size = int(0.15*self.dataset_size)
        test_size = int(0.15*self.dataset_size)

        self.train = self.dataset.take(train_size)
        self.val = self.dataset.skip(train_size)
        self.val = self.dataset.take(val_size)
        self.test = self.dataset.skip(train_size + val_size)
        self.test = self.dataset.take(test_size)

        self.train = self.train.batch(self.batch_size)
        self.val = self.val.batch(self.batch_size)
        self.test = self.test.batch(self.batch_size)

        self.train = self.train.shuffle(buffer_size = train_size)
        self.val = self.val.shuffle(buffer_size = val_size)

        
    def load_dataset(self):                  
        df = pd.read_csv(self.path_to_dataset)
        self.dataset_size = len(df)
        if self.isTimeSeries :          
            target = df.pop("target")
            time_label = [np.ones(1)*label for label in list(target)]
            t = time.process_time()
            time_series = np.array(self.parse_time_series_fast(df), dtype=np.float32)
            elapsed_time = time.process_time() - t
            logging.info("time to parse time serie : " +str(elapsed_time))
            #time_series = np.squeeze(time_series)
            #self.dataset = tf.keras.preprocessing.timeseries_dataset_from_array(time_series, time_label, sequence_length = 1, batch_size=self.batch_size)
            #self.append_additional_features(time_series,[[1],[2]])
            
            self.dataset = tf.data.Dataset.from_tensor_slices((time_series, time_label))
            #self.dataset = self.dataset.map(lambda features, label: (tf.squeeze(features), label))
            
           
        else :
            target = df.pop('Target')
            self.numerical_column = list(df.columns)
            self.dataset = tf.data.Dataset.from_tensor_slices((dict(df), target.values))

    def initialize(self):
        t = time.process_time()
        self.load_dataset()
        elapsed_time = time.process_time() - t
        logging.info("time to load_dataset function : " +str(elapsed_time))
        if self.evaluate == False:
            t = time.process_time()
            self.make_train_val_test_dataset()
            elapsed_time = time.process_time() - t
            logging.info("time to make_train_val_test_dataset function : " +str(elapsed_time))
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
        def replace_split(serie):
            return serie.replace("[","").replace("]","").split(",")
        array_series = []
        array_measure = []
        list_measures = [measure for measure in list(df) if measure != "target"]
        for series in df[list_measures].itertuples(index=False):
            parse_series = list(map(replace_split,series))
            for float_serie in parse_series:
                array_series.append([ float(element_floated) for element_floated in float_serie ])
            array_measure.append(array_series)
            array_series = []
        return array_measure
    
    def parse_time_series_fast(self, df):
        array_measure = []
        list_measures = [measure for measure in list(df) if measure != "target"]
        format = lambda x : x.replace("[","").replace("]","").split(",")
        to_float = lambda x : [float(el) for el in x]
        for series in df[list_measures][borne_inf:borne_sup].applymap(format).applymap(to_float).itertuples(index=False):
                array_measure.append([s for s in series])
        return array_measure
    
    def append_additional_features(self, time_series, features_addition):
        ##TODO : see how to fix input_dim of layer
        embedding_layer = Embedding(len(time_series[0][0]), len(time_series[0][0]))
        features_addition_embeding = embedding_layer(constant(features_addition))
        for index, serie in enumerate(time_series):
            serie =  np.append(serie,np.squeeze(features_addition_embeding.numpy()), axis=0)
        return time_series
    
dp = DataPreprocessing("data/stage_data_out/dataset_temporal/cross-dataset/exclude_DESFAM_F_H91_VENDREDI/dataset.csv",batch_size= 32, isTimeSeries = True) 
