import os
import sys
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fatigue_model.model_parameter import DenseAnn, LSTMAnn
from fatigue_model.data_processing import DataPreprocessing
from fatigue_model.hyper_parameter_tunning import Hparams
from utils import get_last_date_item
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from database_connector import SFTPConnector

from dotenv import load_dotenv

load_dotenv("env_file/.env_path")

PATH_TO_DEBT_MERGE = os.environ.get("PATH_TO_DEBT_MERGE")
PATH_TO_TIME_ON_TASK_MERGE = os.environ.get("PATH_TO_TIME_ON_TASK_MERGE")
PATH_TO_TENSORBOARD = os.environ.get("PATH_TO_TENSORBOARD")
PATH_TO_MODELS = os.environ.get("PATH_TO_MODELS")
class ModelTunning():
    def __init__(self, json_path, path_to_merge_dataset_folder, isTimeSeries, batch_size =32,):      
        self.save_model_on_training = True
        self.model_generator = None
        
        self.preprocessing = DataPreprocessing(get_last_date_item(path_to_merge_dataset_folder), isTimeSeries = isTimeSeries, batch_size = batch_size)
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
        self. date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logdir = "tensorboard/logs/fit/tunning/" + self.date_id +"/"

        self.sftp = SFTPConnector()
    def initialize_model(self, model_name):
        if model_name == "Dense" :
            self.model_generator = DenseAnn()
        elif model_name == "LSTM" :
            self.model_generator = LSTMAnn()
            
    def create_file_logger(self):   
        with tf.summary.create_file_writer(self.logdir).as_default():
            hp.hparams_config(
                hparams=self.hparams_discrete + self.hparams_real_inerval,
                metrics=list(self.hpmetrics.values()),
            )
        
     
    def train_test_model(self, hparams, session_num):
        model = self.model_generator.get_model(self.all_features, self.all_inputs, hparams, self.number_of_target)
        print(list(self.hpmetrics))
        model.compile(
            optimizer = hparams["optimizer"],
            loss = tf.keras.losses.BinaryCrossentropy(),
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
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
            ]
        )
        model.summary()
        if self.save_model_on_training : 
            model_path = "fatigue_model/model_save/"+ self.date_id + "/model_" + str(session_num)
            model.save(model_path)
            #self.sftp.put_dir(PATH_TO_MODELS, model_path)         
            self.sftp.put_dir(os.path.join(PATH_TO_MODELS, self.date_id, "model_" + str(session_num)), model_path)         
            
        _, binary_accuracy, binary_crossentropy, mean_squared_error = model.evaluate(self.test)
        return binary_accuracy, binary_crossentropy, mean_squared_error

    def run(self, run_dir, hparams, session_num):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            binary_accuracy, binary_crossentropy, mean_squared_error = self.train_test_model(hparams, session_num)
            tf.summary.scalar("binary_accuracy", binary_accuracy, step=1)
            tf.summary.scalar("binary_crossentropy", binary_crossentropy, step=1)
            tf.summary.scalar("mean_squared_error", mean_squared_error, step=1)

    def tune_model(self):
        self.create_file_logger()
        session_num = 0
        for hparams in self.hparams_combined:    
            run_name = "run-%d" % session_num
            #hparams.update({"session_num" : session_num})
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            self.run(self.logdir + run_name, {h.name: hparams[h] for h in hparams}, session_num)
            session_num += 1  
        self.sftp.put_dir(os.path.join(PATH_TO_TENSORBOARD, self.date_id), self.logdir)         

                
def train():    
    json_path = "fatigue_model/model_parameter/hparms_lstm.json"

    path_to_merge_dataset_folder = PATH_TO_TIME_ON_TASK_MERGE

    mt = ModelTunning(json_path, path_to_merge_dataset_folder, isTimeSeries = True, batch_size=32)
    mt.initialize_model("LSTM")
    mt.tune_model()
    del mt

    path_to_merge_dataset_folder = PATH_TO_DEBT_MERGE

    mt = ModelTunning(json_path, path_to_merge_dataset_folder, isTimeSeries = True, batch_size=32)
    mt.initialize_model("LSTM")
    mt.tune_model()
    del mt


## TODO : make global variable across module for path