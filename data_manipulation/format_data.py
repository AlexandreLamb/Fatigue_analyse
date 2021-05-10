import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import paths_to_df, parse_path_to_name, generate_columns_name
from datetime import datetime
import os
import json

class DataFormator:   
    VIDEOS_INFOS_PATH = "data/stage_data_out/videos_infos.csv"

    def __init__(self):
        self.df_csv_files = {}
        self.df_formatted = None
        self.df_merge = None
        self.face_recognitions_type = ["cnn", "hog"]
        self.data_folder = "data/stage_data_out/"

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
    def make_label_df(num_min, video_name, measures, df_measure= pd.DataFrame(), path = None):
        
        df_video_infos = pd.read_csv(DataFormator.VIDEOS_INFOS_PATH)
        fps = list(df_video_infos[df_video_infos["video_name"] == video_name]["fps"])[0]
        num_sec = num_min*60
        num_frame_by_num_min = int(fps*num_sec)
        if path != None:
            df = pd.read_csv(path)
        if not df_measure.empty:
            df = df_measure[measures]
            df_label = df.append( pd.DataFrame(columns=['Target']))

            df_label.loc[lambda df_label: df_label["frame"] <= num_frame_by_num_min,"Target"] = 0

            df_label.loc[lambda df_label: df_label["frame"] > num_frame_by_num_min,"Target"] = 1
        df_label = df_label.set_index("frame")
        columns_measures = [col for col in df_label.columns if col !=  "Target"]
        df_label[columns_measures] = (df_label[columns_measures]-df_label[columns_measures].min())/(df_label[columns_measures].max()-df_label[columns_measures].min())
        return df_label
    
    @staticmethod
    def make_df_temporal_label(windows_array, df_measure):
        measures_list = list(df_measure)
        df_measure.sort_index(inplace=True)
        #print(measures_list)
        measures_list.remove("Target")
        df_temporal, df_label = DataFormator.create_df_temporal_label(list(measures_list), windows_array)
        for window in windows_array:
            #print(df_measure.index.max())
            for index in df_measure.index:
                if index + window < df_measure.index.max():
                    for measure_name in measures_list:
                        #print("frame : " + str(index))
                        #print("measure : " + str(measure_name))
                        #print("window size: "  + str(window))
                        if index+window-1 in df_measure.index :
                            #print("windows frames : " + str(list(df_measure.loc[index : index+window-1].index)))
                            measure = df_measure.loc[index : index+window-1][measure_name]
                            if len(measure)==window:
                                df_temporal.loc[index,measure_name+"_"+str(window)]=list(measure)
                                label = 0 if df_measure.loc[index : index+window-1]["Target"].sum() == 0 else 1
                                df_label = df_label.append(pd.DataFrame([label], columns=[window]))            
                        #os.system('clear')
        return df_temporal, df_label
        
    @staticmethod
    def create_df_temporal_label(measure_name_array, windows_array):
        col = []
        for measure_name in measure_name_array:
            for windows in windows_array:
                col.append(measure_name+"_"+str(windows))
        df_temporal  = pd.DataFrame(columns=col, dtype='object')
        df_label = pd.DataFrame(columns=windows_array)
        return df_temporal, df_label  
    
    @staticmethod
    def make_df_feature(df_temporal, df_label, windows_array):
        df_tab=[]
        measures_list = list(df_temporal)
        #print(measures_list)
        """
        for window in windows_array:
            measures_list.remove("frame_"+str(window))
            measures_list.remove("Target_"+str(window))
        """
        #print(df_label)
        #print(measures_list)
        for measure in measures_list:
            df_features = pd.DataFrame(df_temporal[df_temporal[measure].notna()][measure])
            df_features = df_features.set_index(np.arange(len(df_features)))
            df_target = pd.DataFrame(df_label[df_label[int(measure.split("_")[-1])].notna()][int(measure.split("_")[-1])]).rename(columns = {int(measure.split("_")[-1]) : "target"})
            df_target = df_target.set_index(np.arange(len(df_target)))
            #print(df_features.join(df_target))
            df_tab.append(df_features.join(df_target))
        return df_tab

    @staticmethod
    def save_df(df, video_name, windows=""):
        dataset_path = "data/stage_data_out/dataset/Irba_40_min"
        if os.path.exists(os.path.join(dataset_path,video_name)) == False:
            os.mkdir(os.path.join(dataset_path,video_name))
        if windows == "":
            df.to_csv(os.path.join(dataset_path,video_name,video_name+".csv"))
        else : 
            df.to_csv(os.path.join(dataset_path,video_name,video_name+"_"+str(windows)+".csv"))

    
    
    @staticmethod
    def concat_dataset(dataset_array):
        #[df.pop("target") for index, df in enumerate(dataset_array) if index !=len(dataset_array)-1]
        for df in dataset_array:
            target  = df.pop("target")
        df_concat = pd.concat(dataset_array, axis =1)
        df_concat["target"] = target
        print(df_concat)
        return df_concat
    
    @staticmethod
    def convert_df_temporal_array_into_df(dataset_to_convert):
        df = pd.DataFrame(columns = generate_columns_name(10))
        for i, row in dataset_to_convert.iterrows():
            array = json.loads(row["ear_10"])
            df.loc[i] = array
        df["target"] = dataset_to_convert["target"]
        return df
    
    @staticmethod
    def create_dataset_from_measure_folder(path_to_measure_folder, windows):
        dir_measures = os.listdir(path_to_measure_folder)
        date_id = datetime.now().strftime("%H_%M_%d_%m_%Y")
        path_csv_arr = [path_to_measure_folder+"/"+ dir_name+"/"+dir_name+".csv" for dir_name in dir_measures]
        df_measures = pd.DataFrame()
        for path in path_csv_arr:
            print(pd.read_csv(path, index_col=0)["target"].sum())
            df_measures = df_measures.append(pd.read_csv(path, index_col=0), ignore_index=True)
        print(df_measures["target"].sum())
        df_measures.to_csv("data/stage_data_out/dataset/Merge_Dataset/dataset_merge_"+str(windows[0])+"_"+date_id+".csv")
        
        
## TODO:  add video anme and stuff in csv video infos

## TODO: make mother class for herite some commun variable (csv_infos ect...)

## TODO: fiw 'Unnamed: 0' columns  (coreseponding to frame) in df_temporal

