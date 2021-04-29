import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

class DenseAnn():
    @staticmethod
    def get_model(inputs_features, all_inputs, hparams ,number_of_target):
        x = tf.keras.layers.BatchNormalization()(inputs_features)
        x = tf.keras.layers.Dense(hparams["num_units_1"],activation=hparams["activation"])(x)
        x = tf.keras.layers.Dropout(hparams["dropout_1"])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(hparams["num_units_2"],activation=hparams["activation"])(x)
        x = tf.keras.layers.Dropout(hparams["dropout_2"])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(hparams["num_units_3"],activation=hparams["activation"])(x)
        x = tf.keras.layers.Dropout(hparams["dropout_3"])(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(number_of_target, activation=hparams["activation_output"])(x)
        model = tf.keras.Model(all_inputs,output)
        return model
        
      