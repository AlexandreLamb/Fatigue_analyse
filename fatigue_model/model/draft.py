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
    name_hparams = [hparam.name for hparam in hparams_arr.get("hp.Discrete")]  + [hparam.name for hparam in hparams_arr.get("hp.RealInterval")]
    
    hparam_combination = []
    hparams = []
    discrete_value = [val.domain.values for val in hparams_arr.get("hp.Discrete")]
    real_inteval =  [[float(val) for val in np.arange(hparam_interval.domain.min_value, hparam_interval.domain.max_value+0.1,0.1)] for hparam_interval in hparams_arr.get("hp.RealInterval")]
    
    [hparam_combination.append(discrete) for discrete in discrete_value]
    [hparam_combination.append(interval) for interval in real_inteval]
    
    for combination in list(itertools.product(*hparam_combination)):
        combination = list(combination)
        for index, name in enumerate(name_hparams):
            combination[index] = { name : combination[index]}
        hparams.append(combination)
        
    for hparam in hparams:
       print(hparam)
        
path = "fatigue_model/model/hparms.json"

hparams_json = read_json(path)
hparams_arr = json_to_hparams(hparams_json)
tune_model(hparams_arr)