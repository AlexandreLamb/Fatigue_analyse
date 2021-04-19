

class DenseANn():
    def __init__(inputs_features, hparams):
        self.hparams = hparams
        self.inputs_features = inputs_features
    
        

    def get_model(self):
        x = tf.keras.layers.BatchNormalization()(self.inputs_features)
        x = tf.keras.layers.Dense(self.hparams[HP_NUM_UNITS_1],activation=self.hparams[HP_ACTIVATION])(x)
        x = tf.keras.layers.Dropout(self.hparams[HP_DROPOUT])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(self.hparams[HP_NUM_UNITS_2],activation=self.hparams[HP_ACTIVATION])(x)
        x = tf.keras.layers.Dropout(self.hparams[HP_DROPOUT])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(self.hparams[HP_NUM_UNITS_2],activation=self.hparams[HP_ACTIVATION])(x)
        x = tf.keras.layers.Dropout(self.hparams[HP_DROPOUT])(x)
        x = tf.keras.layers.BatchNormalization()(x)

        output = tf.keras.layers.Dense(self.number_of_target, activation=self.hparams[HP_ACTIVATION_OUTPUT])(x)
        model = tf.keras.Model(self.inputs_features,output)
        self.models.append(model)
        
        
    def tune_model(self):
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
                    run(logdir + run_name, hparams, session_num)
                    session_num += 1          
        
