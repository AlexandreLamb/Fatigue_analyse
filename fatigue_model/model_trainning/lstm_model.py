import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
class LSTMAnn():
    @staticmethod
    def get_model(inputs_features, all_inputs, hparams ,number_of_target):       
        model = tf.keras.models.Sequential([])
        for layer in np.arange(hparams["number_of_lstm_layer"])+1:
            model.add(tf.keras.layers.SimpleRNN(hparams["num_units_"+str(layer)], return_sequences=True))
            #model.add(tf.keras.layers.LSTM(hparams["num_units_"+str(layer)], return_sequences=True))
        model.add(tf.keras.layers.Dense(units=number_of_target, activation=hparams["activation_output"]))
        return model
        
       