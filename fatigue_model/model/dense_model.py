

class DenseAnn():
    @static
    def get_model(self):
        x = tf.keras.layers.BatchNormalization()(inputs_features)
        x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS_1],activation=hparams[HP_ACTIVATION])(x)
        x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS_2],activation=hparams[HP_ACTIVATION])(x)
        x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS_2],activation=hparams[HP_ACTIVATION])(x)
        x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(self.number_of_target, activation=hparams[HP_ACTIVATION_OUTPUT])(x)
        model = tf.keras.Model(inputs_features,output)
        return model
        
      