import json
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf 


def read_json(path):  
    with open(path) as f:
        data = json.load(f)
    return data

def json_to_hparams(hparams_json):
    hparams = {    "hp.Discrete" : [] ,
                    "hp.RealInterval" : [] ,
                }
    hpmetrics = {   "metrics" : [] ,
                    "num_of_target" : None
                }
    for hparam, value in hparams_json.get("hp.Discrete").items():
        hparams.get("hp.Discrete").append(hp.HParam(hparam, hp.Discrete(value)))
        
    for hparam, value in hparams_json.get("hp.RealInterval").items():
        hparams.get("hp.RealInterval").append(hp.HParam(hparam, hp.RealInterval(value[0],value[1])))
    
    for metrics in hparams_json.get("metrics"):
        hpmetrics.get("metrics").append(hp.Metric(metrics, display_name = metrics))
    hpmetrics.update({"num_of_target" : hparams_json.get("num_of_target")}) 
    
    return hparams

import itertools
import numpy as np

def tune_model(hparams_arr):
    len_of_loops = len(hparams_arr.get("hp.Discrete")) + len(hparams_arr.get("hp.RealInterval"))
    discrete_value = [val.domain.values for val in hparams_arr.get("hp.Discrete")]
    
    real_inteval =  [[val for val in (hparam_interval.domain.min_value, hparam_interval.domain.max_value)] for hparam_interval in hparams_arr.get("hp.RealInterval")]
    
    
    print(real_inteval)
    print(list(itertools.product(*discrete_value)))
    
    """
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
                                print(hparams)
    """
        
path = "fatigue_model/model/hparms.json"

hparams_json = read_json(path)
hparams_arr = json_to_hparams(hparams_json)
tune_model(hparams_arr)