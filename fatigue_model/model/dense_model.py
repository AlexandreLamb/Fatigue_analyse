

class DenseANn():
    def __init__(inputs_features, hparams):
        self.hparams = hparams
        self.inputs_features = inputs_features
    
        self.HP_NUM_UNITS_1 = hp.HParam('num_units_1', hp.Discrete([32]))
        self.HP_NUM_UNITS_2 = hp.HParam('num_units_2', hp.Discrete([512]))
        self.HP_NUM_UNITS_3 = hp.HParam('num_units_3', hp.Discrete([512]))
        self.HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.5, 0.5))
        self.HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
        self.HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu']))
        self.HP_ACTIVATION_OUTPUT = hp.HParam('activation_output', hp.Discrete(['sigmoid']))

        self.METRIC_BINARY_ACCURACY = "binary_accuracy"
        self.METRIC_BINARY_CROSSENTROPY = "binary_crossentropy"
        self.METRIC_MSE = "mean_squared_error"

        self.NUMBER_OF_TARGET = 1
        self.METRICS = ["binary_accuracy","binary_crossentropy","mean_squared_error"]

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
        
