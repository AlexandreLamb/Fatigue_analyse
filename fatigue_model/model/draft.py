import json
from tensorboard.plugins.hparams import api as hp

def read_json(path):  
    with open(path) as f:
        data = json.load(f)
    return data

def json_to_hparams(hparams_json):
    hparams_arr = { "hp.Discrete" : [] ,
                    "hp.RealInterval" : [] ,
                    "metrics" : [] ,
                    "num_of_target" : None
                }
    for hparam, value in hparams_json.get("hp.Discrete").items():
        hparams_arr.get("hp.Discrete").append(hp.HParam(hparam, hp.Discrete(value)))
        
    for hparam, value in hparams_json.get("hp.RealInterval").items():
        hparams_arr.get("hp.RealInterval").append(hp.HParam(hparam, hp.RealInterval(value[0],value[1])))
    
    for metrics in hparams_json.get("metrics"):
        print(metrics)
        hparams_arr.get("metrics").append(hp.Metrics(metrics, display_name = metrics))
    hparams_arr.     
        
path = "fatigue_model/model/hparms.json"

hparams_json = read_json(path)
json_to_hparams(hparams_json)