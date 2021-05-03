import pandas as pd
def parse_time_series(df):
    array_series = []
    array_measure = []
    list_measures = [measure for measure in list(df) if measure != "target"]
    for series in df[list_measures].values:
        parse_series = [serie.replace("[","").replace("]","").split(",") for serie in series]
        for float_serie in parse_series:
            array_series.append([ float(element_floated) for element_floated in float_serie ])
        array_measure.append(array_series)
        array_series = []
    return array_measure
    
    
            
def noramlize_time_series(time_series): 
    df_to_normalize = pd.DataFrame(time_series)
    mean = df_to_normalize.mean(axis = 1)
    std = df_to_normalize.std(axis = 1)
    df_to_normalize = df_to_normalize.sub(mean, axis = 0) 
    df_to_normalize = df_to_normalize.div(std, axis = 0)
    return list(df_to_normalize.values)
        

df = pd.read_csv("data/stage_data_out/dataset_ear/DESFAM-F_H99_VENDREDI/DESFAM-F_H99_VENDREDI.csv", index_col=0)

import numpy as np
array_series = parse_time_series(df.loc[0:10])
df_to_normalize = pd.DataFrame(array_series)
print(np.mean(list(df_to_normalize[0])))
print(np.std(list(df_to_normalize[0])))

#print(list(df_to_normalize[0]) - np.ones(len(df_to_normalize[0]),) )

