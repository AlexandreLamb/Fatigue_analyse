import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

class DenseAnn():
    @staticmethod
    def get_model(hparams ,number_of_target):
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(hparams["num_units_1"], return_sequences=True),
            tf.keras.layers.Dense(units=number_of_target)
        ])
        return model
        
      