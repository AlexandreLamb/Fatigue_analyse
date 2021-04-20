
import tensorflow as tf 
import pandas as pd 
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

class DataPreprocessing()
    def __init__(self, batch_size, path_to_dataset):
        self.all_inputs = []
        self.encoded_features = []
        self.batch_size = batch_size
        self.all_features = []
        self.path_to_dataset = path_to_dataset
        self.dataset = None
        self.numerical_column = None
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
        
        for column_name in numerical_column:
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

        train = train.shuffle(buffer_size = train_size)
        train = train.batch(self.batch_size)

        val = val.shuffle(buffer_size = val_size)
        val = val.batch(self.batch_size)

        test = test.batch(self.batch_size)
        
        def load_dataset(self):                  
            df = pd.read_csv(self.path_to_dataset, index_col=0)
            target = df.pop('Target')
            self.numerical_column = list(df.columns)
            self.dataset = tf.data.Dataset.from_tensor_slices((dict(df), target.values))

        def initialize(self):
            self.load_dataset()
            self.make_train_val_test_dataset()
            self.make_numerical_feature_col()