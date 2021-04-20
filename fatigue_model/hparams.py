import json
from tensorboard.plugins.hparams import api as hp
import numpy as np
import itertools

class Hparams():
    def __init__(self, json_path):
        self.json_path = json_path
        self.json_data = self.read_json()
        self.hparams = {    "hp.Discrete" : [] ,
                    "hp.RealInterval" : [] ,
                }
        self.hpmetrics = {   "metrics" : [] ,
                    "num_of_target" : None
                }
        self.hparams_combined = []
        self.json_to_hparams()
        self.create_hparam_combination()
        
    def read_json(self):  
        with open(self.json_path) as json_file:
            data = json.load(json_file)
        return data
    
    def json_to_hparams(self):
        
        for hparam, value in self.json_data.get("hp.Discrete").items():
            self.hparams.get("hp.Discrete").append(hp.HParam(hparam, hp.Discrete(value)))
            
        for hparam, value in self.json_data.get("hp.RealInterval").items():
            self.hparams.get("hp.RealInterval").append(hp.HParam(hparam, hp.RealInterval(value[0],value[1])))
        
        for metrics in self.json_data.get("metrics"):
            self.hpmetrics.get("metrics").append(hp.Metric(metrics, display_name = metrics))
        
        self.hpmetrics.update({"num_of_target" : self.json_data.get("num_of_target")}) 
    
        
    def create_hparam_combination(self):
        name_hparams = [hparam.name for hparam in self.hparams.get("hp.Discrete")]  + [hparam.name for hparam in self.hparams.get("hp.RealInterval")]
        
        hparam_combination = []
        
        discrete_value = [val.domain.values for val in self.hparams.get("hp.Discrete")]
        real_inteval =  [[float(val) for val in np.arange(hparam_interval.domain.min_value, hparam_interval.domain.max_value+0.1,0.1)] for hparam_interval in self.hparams.get("hp.RealInterval")]
        
        [hparam_combination.append(discrete) for discrete in discrete_value]
        [hparam_combination.append(interval) for interval in real_inteval]
        for combination in list(itertools.product(*hparam_combination)):
            combination = list(combination)
            hparam_obj = {}
            for index, hparam in enumerate(self.hparams.get("hp.Discrete") + self.hparams.get("hp.RealInterval")):
                hparam_obj.update({ hparam : combination[index]})
            self.hparams_combined.append(hparam_obj)
    
    def tune_model(self):
        for hparams in self.hparams_combined:    
            print(hparams)
            print({h.name: hparams[h] for h in hparams})
            time.sleep(10) 
        
hp = Hparams("fatigue_model/model/hparms.json")
#hp.tune_model()

hpmetrics = hp.hpmetrics.get("metrics")
number_of_tar = hp.hpmetrics.get("num_of_target")

print(number_of_tar)