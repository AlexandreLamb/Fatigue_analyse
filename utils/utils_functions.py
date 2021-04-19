import pandas as pd
from datetime import datetime
import numpy as np

def make_landmarks_header():
    csv_header = []
    for i in range(1,69):
        csv_header.append("landmarks_"+str(i)+"_x")
        csv_header.append("landmarks_"+str(i)+"_y")
    return csv_header

def parse_path_to_name(path):
    name_with_extensions = path.split("/")[-1]
    name = name_with_extensions.split(".")[0]
    return name

def paths_to_df(csv_array):
    df_array = []
    for path in csv_array:
        df_array.append(pd.read_csv(path).rename(columns={"Unnamed: 0" : "frame"}))
    return df_array

def generate_columns_name(windows):
    columns_name = []
    for i in range(1,windows+1):
        columns_name.append("t_"+str(i))
    return columns_name

def make_box_from_landmarks(row, threeshold_px = 20):

    left = int(row["landmarks_1_x"] - threeshold_px)
    top = int(np.mean([row["landmarks_20_y"] , row["landmarks_25_y"]]) - 2* threeshold_px)
    height = int( row["landmarks_9_y"] - top - threeshold_px)
    width = int(row["landmarks_17_x"] - left + threeshold_px)
    return {"top" : top, "left" : left, "height" : height, "width" : width} 


date_id = lambda : datetime.now().strftime("%H_%M_%S_%d_%m_%Y")

make_landmarks_pair  = lambda marks : list(zip(marks[::2],marks[1::2]))
