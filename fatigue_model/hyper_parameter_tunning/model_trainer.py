import os
import sys
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import DenseAnn
from data_processing import DataPreprocessing
from hparams import Hparams
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

class ModelTunning():
    def __init__(self, json_path, path_to_dataset):      
        self.save_model_on_training = True
        self.model_generator = None
        
        self.preprocessing = DataPreprocessing(path_to_dataset)
        self.all_features = self.preprocessing.all_features
        self.all_inputs = self.preprocessing.all_inputs
        self.val = self.preprocessing.val
        self.train = self.preprocessing.train
        self.test = self.preprocessing.test
        
        self.hptuner = Hparams(json_path)
        self.hparams_combined = self.hptuner.hparams_combined
        self.hparams_discrete = self.hptuner.hparams.get("hp.Discrete")
        self.hparams_real_inerval = self.hptuner.hparams.get("hp.RealInterval")
        self.hpmetrics = self.hptuner.hpmetrics
        self.number_of_target = self.hptuner.other_params.get("num_of_target")
        self.epochs = self.hptuner.other_params.get("epochs")
        self.logdir = "tensorboard/logs/fit/tunning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"

    def initialize_model(self, model_name):
        if model_name == "Dense" :
            self.model_generator = DenseAnn()
            
    def create_file_logger(self):           
        with tf.summary.create_file_writer(self.logdir).as_default():
            hp.hparams_config(
                hparams=self.hparams_discrete + self.hparams_real_inerval,
                metrics=list(self.hpmetrics.values()),
            )
     
    def train_test_model(self, hparams, session_num):
        model = self.model_generator.get_model(self.all_features, self.all_inputs, hparams, self.number_of_target)
        model.summary()
        print(self.hpmetrics)
        model.compile(
            optimizer = hparams["optimizer"],
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = list(self.hpmetrics),
        )
        model.fit(
            self.train, 
            validation_data= self.val,
            epochs=self.epochs,
            shuffle=True,
            verbose =1,
            callbacks=[ 
                tf.keras.callbacks.TensorBoard(log_dir = self.logdir),  # log metrics
                hp.KerasCallback(self.logdir, hparams),  # log self.hparams
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
            binary_accuracy, binary_crossentropy, mean_squared_error = self.train_test_model(hparams, session_num)
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
            self.run(self.logdir + run_name, {h.name: hparams[h] for h in hparams}, session_num)
            session_num += 1          
                    
json_path = "fatigue_model/model/hparms.json"
dataset_path = "data/stage_data_out/dataset_ear/dataset_ear/dataset_ear_1.csv"
mt = ModelTunning(json_path, dataset_path)
mt.initialize_model("Dense")
mt.tune_model()
## TODO : make global variable across module for path