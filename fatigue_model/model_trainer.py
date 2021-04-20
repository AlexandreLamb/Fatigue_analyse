import datetime


class ModelTunning():
    def __init__(self, json_path):      
        self.metrics= []
        self.save_model_on_training = True
        self.model_generator = None
        
        self.preprocessing = DataPreprocessing(batch_size, path_to_dataset)
        self.all_features = DataPreprocessing.all_features
        self.all_inputs = DataPreprocessing.all_inputs
        self.val = DataPreprocessing.val
        self.training = DataPreprocessing.training
        self.test = DataPreprocessing.test
        
        self.hptuner = Hparams("fatigue_model/model/hparms.json")
        self.hparams_combined = hptuner.hparams_combined
        self.hparams_discrete = hptuner.hparams.get("hp.Discrete")
        self.hpmetrics = hptuner.hpmetrics.get("metrics")
        self.number_of_target = hptuner.hpmetrics.get("num_of_target")
        
        self.logdir = "tensorboard/logs/fit/tunning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"

    def initialize_model(self, model_name):
        if model_name == "Dense" :
            self.model_generator = DenseAnn()
            
    def create_file_logger(self):           
        with tf.summary.create_file_writer(self.logdir).as_default():
            hp.hparams_config(
                hparams=self.hparams,
                metrics=self.hpmetrics,
            )
     
    def train_test_model(self, hparams, session_num):
        model = self.model.getModel(self.all_features, self.hparams, self.number_of_target)
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

    def run(self, run_dir, hparams, session_num):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            binary_accuracy, binary_crossentropy, mean_squared_error = train_test_model(hparams, session_num)
            tf.summary.scalar("METRIC_BINARY_ACCURACY", binary_accuracy, step=1)
            tf.summary.scalar("METRIC_BINARY_CROSSENTROPY", binary_crossentropy, step=1)
            tf.summary.scalar("METRIC_MSE", mean_squared_error, step=1)

    def tune_model(self):
        self.create_file_logger()
        session_num = 0
        for hparams in self.hparams_combined:    
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run(self.logdir + run_name, hparams, session_num)
            session_num += 1          
                    
        
                    