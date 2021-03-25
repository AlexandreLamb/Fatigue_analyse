import pandas as pd
import numpy as np
from utils import paths_to_df, parse_path_to_name
from datetime import datetime
class DataFormator:   
    VIDEOS_INFOS_PATH = "data/stage_data_out/videos_infos.csv"

    def __init__(self):
        self.df_csv_files = {}
        self.df_formatted = None
        self.df_merge = None
        self.face_recognitions_type = ["cnn", "hog"]
        self.data_folder = "data/data_out/"
        #self.video_info_path("data/stage_data_out/videos_infos.csv")
        
    def load_csv_by_face_recognitions(self, video_name):
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
    
    def is_df_has_empty_frame(self, df, frame):
        index_array = df.index.values
        if frame in index_array: return False
        else: return True
    
    ## TODO: Make frame index of all the DF by default
    
    def merge_csv(self, csv_array_path):
        count_divide_dict = {}
        df_array = paths_to_df(csv_array_path)
        self.df_merge = df_array[0].set_index("frame")
        for df in df_array[1:]:
            df = df.set_index("frame")     
            for frame, landmarks in df.iterrows():
                if self.is_df_has_empty_frame(self.df_merge, frame):
                    if count_divide_dict.get(frame) == None :
                        count_divide_dict[frame] = 1
                    self.df_merge = self.df_merge.append(landmarks)
                else:   
                    if count_divide_dict.get(frame) == None :
                        count_divide_dict[frame] = 1
                    self.df_merge.loc[frame] =  self.df_merge.loc[frame] + landmarks              
                    count_divide_dict.update({frame : count_divide_dict.get(frame)+1})
        self.df_merge = self.df_merge.sort_index()
        for frame, divider in count_divide_dict.items():
            self.df_merge.loc[frame] = self.df_merge.loc[frame].divide(divider)
        date_id = datetime.now().strftime("%H_%M_%d_%m_%Y")
        csv_path = "data/data_out/df_merge_"+date_id+".csv"
        self.df_merge.to_csv(csv_path)
        df_videos_infos = pd.read_csv("data/data_out/videos_infos.csv")
        df_videos_merge_infos = pd.DataFrame(columns = ["video_name","fps"])
        df_videos_merge_infos = df_videos_merge_infos.append({
                                                                'video_name' : "df_merge_"+date_id,
                                                                'fps' :list(df_videos_infos[df_videos_infos["video_name"]==parse_path_to_name(csv_array_path[0])]["fps"])[0],
                                                                    },
                                                                    ignore_index=True)
        df_videos_merge_infos.to_csv("data/data_out/videos_infos.csv", mode="a", header=False)
        return csv_path.split(".")[0]
    @staticmethod
    def make_label_df(num_min, video_name, df_measure= pd.DataFrame(), path = None):
        df_video_infos = pd.read_csv(DataFormator.VIDEOS_INFOS_PATH)
        fps = list(df_video_infos[df_video_infos["video_name"] == video_name]["fps"])[0]
        num_sec = num_min*60
        num_frame_by_num_min = int(fps*num_sec)
        if path != None:
            df = pd.read_csv(path)
        if not df_measure.empty:
            df = df_measure
            df_label = df.append( pd.DataFrame(columns=['Target']))

            df_label.loc[lambda df_label: df_label["frame"] <= num_frame_by_num_min,"Target"] = 0

            df_label.loc[lambda df_label: df_label["frame"] > num_frame_by_num_min,"Target"] = 1
            
        return df_label


## TODO:  add video anme and stuff in csv video infos

## TODO: make mother class for herite some commun variable (csv_infos ect...)