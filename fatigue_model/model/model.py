


class Model():
    def __init__(self):
        
        self.metrics= []
        self.epochs
        self.save_model_on_training
        self.number_of_target
        
    def dense_model(self, hparams):
        
        x = tf.keras.layers.BatchNormalization()(all_features)
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
        model = tf.keras.Model(all_inputs,output)
        return model


    def train_test_model(self, hparams, session_num):
        model = modeling(hparams)
        model.summary()
        model.compile(
            optimizer = hparams[HP_OPTIMIZER],
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
                hp.KerasCallback(logdir, hparams),  # log hparams
                tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10),
            ]
        )
        if self.save_model_on_training : 
            model.save("tensorboard/model/"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + "/model_" + str(session_num))
        _, binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(test)
        return binary_accuracy, binary_crossentropy, mean_squared_error

