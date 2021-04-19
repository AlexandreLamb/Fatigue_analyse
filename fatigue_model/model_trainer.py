


class ModelTunning():
    def __init__(self, json_path):      
        self.metrics= []
        self.save_model_on_training = True
        self.model = None
        self.inputs_features = DataPreprocessing.get_features()
        self.hparams_combined = Hparams("fatigue_model/model/hparms.json").hparams_combined
        
    def initialize_model(self, model_name):
        if model_name == "Dense" :
            self.model = DenseAnn()
            
    def creat_file_logger(self):
            
    with tf.summary.create_file_writer(self.logdir).as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS_1, HP_NUM_UNITS_2, HP_DROPOUT, HP_ACTIVATION, HP_ACTIVATION_OUTPUT, HP_OPTIMIZER],
        metrics=[ hp.Metric(METRIC_BINARY_ACCURACY, display_name='Binary Accuracy'),
                hp.Metric(METRIC_BINARY_CROSSENTROPY, display_name='Binary Cross Entropy'),
                hp.Metric(METRIC_MSE, display_name='MSE'),
        ],
    )
     
    def train_test_model(self, session_num):
        model = modeling(self.hparams)
        model.summary()
        model.compile(
            optimizer = self.hparams[HP_OPTIMIZER],
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = ["binary_accuracy","binary_crossentropy","mean_squared_error"],
        )
        model.fit(
            train, 
            validation_data= val,
            epochs=30,
            shuffle=True,
            verbose =1,
            callbacks=[ 
                tf.keras.callbacks.TensorBoard(log_dir = logdir),  # log metrics
                hp.KerasCallback(logdir, self.hparams),  # log self.hparams
                tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10),
            ]
        )
        if self.save_model_on_training : 
            model.save("tensorboard/model/"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + "/model_" + str(session_num))
        _, binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(test)
        return binary_accuracy, binary_crossentropy, mean_squared_error

    def run(run_dir, hparams, session_num):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            binary_accuracy, binary_crossentropy, mean_squared_error = train_test_model(hparams, session_num)
            tf.summary.scalar(METRIC_BINARY_ACCURACY, binary_accuracy, step=1)
            tf.summary.scalar(METRIC_BINARY_CROSSENTROPY, binary_crossentropy, step=1)
            tf.summary.scalar(METRIC_MSE, mean_squared_error, step=1)

    def tune_model(self):
        for hparams in self.hparams_combined:    
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run(logdir + run_name, hparams, session_num)
            session_num += 1          
                    
                    
                    
                    
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
                    
                    