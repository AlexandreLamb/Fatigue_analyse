import datetime


class ModelTunning():
    def __init__(self, json_path):      
        self.metrics= []
        self.save_model_on_training = True
        self.model = None
        
        self.inputs_features = DataPreprocessing.get_features()
        self.val = DataPreprocessing.get_val()
        self.training = DataPreprocessing.get_training()
        self.test = DataPreprocessing.get_test()
        
        self.hptuner = Hparams("fatigue_model/model/hparms.json")
        self.hparams_combined = hptuner.hparams_combined
        self.hparams_discrete = hptuner.hparams.get("hp.Discrete")
        self.hpmetrics = hptuner.hpmetrics.get("metrics")
        
        self.logdir = "tensorboard/logs/fit/tunning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"

    def initialize_model(self, model_name):
        if model_name == "Dense" :
            self.model = DenseAnn().get_model()
            
    def creat_file_logger(self):
            
    with tf.summary.create_file_writer(self.logdir).as_default():
    hp.hparams_config(
        hparams=self.hparams,
        metrics=self.hpmetrics,
    )
     
    def train_test_model(self, hparams, session_num):
        model = modeling(self.hparams)
        model.summary()
        model.compile(
            optimizer = [hparam for hparam in hparams if  hparam.name == "optimizer"],
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = self.hpmetrics,
        )
        model.fit(
            self.train, 
            validation_data= self.val,
            epochs=self.epochs,
            shuffle=True,
            verbose =1,
            callbacks=[ 
                tf.keras.callbacks.TensorBoard(log_dir = self.logdir),  # log metrics
                hp.KerasCallback(self.logdir, self.hparams),  # log self.hparams
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
        session_num = 0
        for hparams in self.hparams_combined:    
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run(self.logdir + run_name, hparams, session_num)
            session_num += 1          
                    
        
                    