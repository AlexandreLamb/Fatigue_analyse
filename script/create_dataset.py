import argparse, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_manipulation import AnalyseData
from data_manipulation import DataFormator
import json
import os

parser=argparse.ArgumentParser()


parser.add_argument('--path', help='path to the landamrks folder')
#parser.add_argument('--json_measure', help='path to measure json file')

args=parser.parse_args()

#json_measure = json.load(args.json_measure)

csv_array_name  = os.listdir(args.path)

csv_array_path = [args.path + "/" +  name for name in csv_array_name]

df_ear_all = pd.DataFrame()

for index, csv_landmarks_path in enumerate(csv_array_path) :
    video_name = csv_array_name[index].split("_"+args.path.split("/")[-1]+".")[0]
    
    analyse_data = AnalyseData(csv_landmarks_path)
    #TODO: make a function who take a json file of meaurec
    analyse_data.measure_ear()

    df_ear = DataFormator.make_label_df(num_min = 5, video_name = video_name, df_measure= analyse_data.df_measure)
    df_ear_all = df_ear_all.append(df_ear)
    DataFormator.save_df(df_ear, video_name)

DataFormator.save_df(df_ear_all,'Merge_Dataset')
