import pandas as pd
from datetime import datetime
import numpy as np
import re

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


def parse_video_name(video_name_list):
    #video_name_list = list(pd.read_csv("/home/simeon/Documents/Fatigue_analyse/data/stage_data_out/videos_infos.csv")["video_name"])
    subject_condition = list(pd.read_csv("data/stage_data_out/sujets_data_pvt_perf.csv", sep=";", index_col = [0,1]).index)
    jour_1_to_parse = ["LUNDI", "lundi"]
    jour_2_to_parse = ["VENDREDI", "vendredi"]

    string_to_remove = ["DESFAM", "DESFAM-F", "PVT", "P1", "P2", "DEBUT", "FIN", "retard","min","de" , "avant PVT", "avant", "PVT", "F", "Semaine","1", "08", "8"]
    subject_list = []
    for subject_to_parse in video_name_list:
        subject_clean = []
        subject_split = re.split("_|\s+",subject_to_parse)
        #print(subject_split)
        for sub in subject_split:
            #print("str split :" +str(subject_split))
            if sub not in string_to_remove :
                subject_clean.append(sub)
        condition =  [cond[1] for cond  in subject_condition if cond[0] == subject_clean[0]][0]
        if subject_clean[1] in jour_1_to_parse:
            subject_list.append(subject_clean[0] +"_"+condition +  "_T1")
        else :
            subject_list.append(subject_clean[0] +"_"+condition + "_T2")
    return subject_list

date_id = lambda : datetime.now().strftime("%H_%M_%S_%d_%m_%Y")

make_landmarks_pair  = lambda marks : list(zip(marks[::2],marks[1::2]))

#print(parse_video_name(["DESFAM-F_H92_LUNDI_P1"]))
