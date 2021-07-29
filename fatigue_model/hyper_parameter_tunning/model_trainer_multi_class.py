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
class ModelTunningMultiClass():
    """Class for tunning the configuration of a models with a json file
    """
    def __init__(self, json_path, path_to_merge_dataset_folder, isTimeSeries, batch_size =32,):
        """Init function who load the dataset and preprocess it, it also load and read the configuration to test from the json file.

        :param json_path: Path to the json descriptor configuration
        :type json_path: str
        :param path_to_merge_dataset_folder: Path to the folder who contain the merge dataset in the database 
        :type path_to_merge_dataset_folder: str
        :param isTimeSeries: An boolean for indicate if the preprocessing dataset has to be in time serie mode
        :type isTimeSeries: bool
        :param batch_size: The size of batch for the dataset, defaults to 32
        :type batch_size: int, optional
        """
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
        """Function who initialize the model, you can choice between LSTM or ANN

        :param model_name: Name of the model type to use "LSTM" or "Dense"
        :type model_name: str
        """
        if model_name == "Dense" :
            self.model_generator = DenseAnn()
        elif model_name == "LSTM" :
            self.model_generator = LSTMAnn()
            
    def create_file_logger(self):   
        """Function who create the file for log the metrics and current configuration test
        """
        with tf.summary.create_file_writer(self.logdir).as_default():
            hp.hparams_config(
                hparams=self.hparams_discrete + self.hparams_real_inerval,
                metrics=list(self.hpmetrics.values()),
            )
        
     
    def train_test_model(self, hparams, session_num):
        """Fucntion who run train session for a configuration of the model, save the model into the database folder 'models', and test the configuraiton on test dataset

        :param hparams: a set of configuration for the model (like number of units, number hideen layer, ...)
        :type hparams: dict
        :param session_num: The number of the current session
        :type session_num: int
        :return: This functions return the values of the metrics on the test dataset
        :rtype: multiple int
        """
        model = self.model_generator.get_model(self.all_features, self.all_inputs, hparams, self.number_of_target)
        print(list(self.hpmetrics))
        model.compile(
            optimizer = hparams["optimizer"],
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy() ],
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
            
        _, sparse_categorical_accuracy = model.evaluate(self.test)
        return sparse_categorical_accuracy

    def run(self, run_dir, hparams, session_num):
        """function who create the file that log the metrics return by the train_test_model function and run train_test_model

        :param run_dir: path to directory to log metrics
        :type run_dir: str
        :param hparams:  a set of configuration for the model (like number of units, number hideen layer, ...)
        :type hparams: dict
        :param session_num: The number of the current session
        :type session_num: int
        """
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            sparse_categorical_accuracy = self.train_test_model(hparams, session_num)
            tf.summary.scalar("sparse_categorical_accuracy", sparse_categorical_accuracy, step=1)

    def tune_model(self):
        """main function who loop over the different set of hparams for run the model tunning with run function
        """
        for feature_batch, label_batch in self.train.take(1):
            print('A rank of features:', tf.rank(feature_batch))
            print('A rank of targets:', tf.rank(label_batch.shape))
            print('A shape of features:', feature_batch.shape)
            print('A shape of targets:', label_batch.shape)
            print('A batch of features:', feature_batch.numpy())
            print('A batch of targets:', label_batch.numpy())
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

    


## TODO : make global variable across module for path