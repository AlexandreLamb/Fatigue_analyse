import pandas as pd
import numpy as np
from utils import paths_to_df
class DataFormator:   
    def __init__(self):
        self.df_csv_files = {}
        self.df_formatted = None
        self.df_merge = None
        self.face_recognitions_type = ["cnn", "hog"]
        self.data_folder = "data/data_out/"
        
    def load_csv(self, video_name):
        for face_type in self.face_recognitions_type:
            self.df_csv_files.update({
                face_type : pd.read_csv(self.data_folder+video_name+"_"+face_type+".csv").rename(columns={ "Unnamed: 0" : "frame"})
            })
            
    def fusion_csv(self, mode):
        row_list = []
        if mode == "independent" :
            self.df_formatted = self.df_csv_files.get("hog")
            for key, value in self.df_csv_files.items():
                if key not in ["hog"]:
                    for frame in value["frame"] :
                        if self.df_formatted[self.df_formatted["frame"] == frame].empty:
                            self.df_formatted =  self.df_formatted.append(value[value["frame"] == frame])        
            self.df_formatted.sort_values(by="frame")
        if mode == "mean":
            self.df_formatted = self.df_csv_files.get("hog")
            self.df_formatted.set_index("frame")
            for key, value in self.df_csv_files.items():
                if key not in ["hog"]:
                    value.set_index("frame")
                    for frame in value["frame"] :
                        if self.df_formatted[self.df_formatted["frame"] == frame].empty:
                            self.df_formatted =  self.df_formatted.append(value[value["frame"] == frame])
                        else: 
                            print(self.df_formatted.loc[self.df_formatted["frame"] == frame, self.df_formatted.columns != "frame"] + value.loc[value["frame"] == frame, value.columns != "frame"]/2)
            self.df_formatted.sort_values(by="frame")
            
    def merge_csv(self, csv_array_path):
        count_divide_dict = {}
        df_array = paths_to_df(csv_array_path)
        self.df_merge = df_array[0].rename(columns={"Unnamed: 0" : "frame"})
        #self.df_merge.set_index("frame")
        for df in df_array[1:]:
            for frame in df["frame"]:
                if self.df_merge[self.df_merge["frame"] == frame].empty:
                    self.df_merge =  self.df_merge.append(df[df["frame"] == frame])
                else: 
                    #print(self.df_merge.loc[self.df_merge["frame"] == frame, self.df_merge.columns != "frame"] + df.loc[df["frame"] == frame, df.columns != "frame"])
                if count_divide_dict.get(frame) == None :
                    #count_divide_dict.update({frame, 1})
                    count_divide_dict[frame] = 1
                else :
                    count_divide_dict.update({frame, count_divide_dict.get(frame)+1})
                print(count_divide_dict)
                self.df_merge.sort_values(by="frame")
        for frame, divider in count_divide_dict.items():
            self.df_merge[self.df_merge["frame"] == frame] / divider
       
        
df = DataFormator()
csv_array = ["data/data_out/DESFAM_Semaine 2-Vendredi_Go-NoGo_H64_hog.csv", "data/data_out/DESFAM_Semaine 2-Vendredi_Go-NoGo_H69_hog.csv"]
df.merge_csv(csv_array)