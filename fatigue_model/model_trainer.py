


class Model():
    def __init__(self):      
        self.metrics= []
        self.epochs
        self.save_model_on_training
        self.number_of_target
        self.models = []
        self.inputs_features = 
    


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
                    
                    