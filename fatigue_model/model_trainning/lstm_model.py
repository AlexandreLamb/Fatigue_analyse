import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

class LSTMAnn():
    @staticmethod
    def get_model(inputs_features, all_inputs, hparams ,number_of_target):       
        for layer in hparams["number_of_lstm_layer"]:
            model.add(tf.keras.layers.LSTM(hparams["num_units_"+str(layer)], return_sequences=True))
        model.add(tf.keras.layers.Dense(units=number_of_target, activation_output=hparams["activation_output"]))
        return model
        
       