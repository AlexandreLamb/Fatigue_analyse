import pandas as pd
import numpy as np
import os, sys, timeit
from dotenv import load_dotenv

load_dotenv("env_file/.env_path")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import paths_to_df, parse_path_to_name
from datetime import datetime
from database_connector import SFTPConnector
import os
import re
import json
PATH_TO_DATASET_NON_TEMPORAL = os.environ.get("PATH_TO_DATASET_NON_TEMPORAL")
PATH_TO_DATASET_TEMPORAL = os.environ.get("PATH_TO_DATASET_TEMPORAL")
PATH_TO_LANDMARKS_DESFAM_F = os.environ.get("PATH_TO_LANDMARKS_DESFAM_F")
PATH_TO_IRBA_DATA_PVT = os.environ.get("PATH_TO_IRBA_DATA_PVT")
PATH_TO_IRBA_DATA_VAS = os.environ.get("PATH_TO_IRBA_DATA_VAS")
PATH_TO_TIME_ON_TASK_VIDEO = os.environ.get("PATH_TO_TIME_ON_TASK_VIDEO")
PATH_TO_TIME_ON_TASK_VIDEO_VAS = os.environ.get("PATH_TO_TIME_ON_TASK_VIDEO_VAS")

class DataFormator:  
    
    def __init__(self):
        self.sftp = SFTPConnector()
        #self.video_info_path("data/stage_data_out/videos_infos.csv")
    def make_label_df(self, num_min, video_name, measures, df_measure= pd.DataFrame(), path = None, fps =None): 
        df_video_infos = self.sftp.read_remote_df(os.path.join(PATH_TO_LANDMARKS_DESFAM_F, "videos_infos.csv"))
        if fps == None:
            fps = list(df_video_infos[df_video_infos["video_name"] == video_name]["fps"])[0]
        num_sec = num_min*60
        num_frame_by_num_min = int(fps*num_sec)
        frame_threeshold = 3 * 60 * fps
        
        #print("num frame by min : " +str(num_frame_by_num_min))
        if path != None:
            df = self.sftp.save_remote_df(path)
        if not df_measure.empty:
            
            df = df_measure[measures]
            df_label = df.append(pd.DataFrame(columns=['Target']))
            #print(df_label[df_label["frame"].between(0,num_frame_by_num_min*2)])
            df_label.loc[lambda df_label: df_label["frame"].between(frame_threeshold,num_frame_by_num_min + frame_threeshold),"Target"] = 0

            df_label.loc[lambda df_label: df_label["frame"].between(len(df_label) -num_frame_by_num_min -frame_threeshold, len(df_label) - frame_threeshold),"Target"] = 1
            df_label.dropna(inplace=True)

        df_label = df_label.set_index("frame")
        columns_measures = [col for col in df_label.columns if col !=  "Target"]
        df_label[columns_measures] = (df_label[columns_measures]-df_label[columns_measures].min())/(df_label[columns_measures].max()-df_label[columns_measures].min())
        print(df_label)       
        remote_path = os.path.join(PATH_TO_DATASET_NON_TEMPORAL,video_name,video_name+".csv")
        print("saving non temporal")
        self.sftp.save_remote_df(remote_path, df)
        
        return df_label
    ##TODO : find why there is 6000 frame isntead of 3000 frame

    def make_df_temporal_label(self, windows_array, df_measure):
        measures_list = list(df_measure)
        df_measure.sort_index(inplace=True)
        #print(measures_list)
        measures_list.remove("Target")
        df_temporal, df_label = self.create_df_temporal_label(list(measures_list), windows_array)
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
        
    def create_df_temporal_label(self, measure_name_array, windows_array):
        col = []
        for measure_name in measure_name_array:
            for windows in windows_array:
                col.append(measure_name+"_"+str(windows))
        df_temporal  = pd.DataFrame(columns=col, dtype='object')
        df_label = pd.DataFrame(columns=windows_array)
        return df_temporal, df_label  
    
    def make_df_feature(self, df_temporal, df_label, windows_array):
        df_tab=[]
        measures_list = list(df_temporal)
        #print(measures_list)
        """
        for window in windows_array:
            measures_list.remove("frame_"+str(window))
            measures_list.remove("Target_"+str(window))
        """
        for measure in measures_list:
            df_features = pd.DataFrame(df_temporal[df_temporal[measure].notna()][measure])
            df_features = df_features.set_index(np.arange(len(df_features)))
            #df_target = pd.DataFrame(df_label[df_label[int(measure.split("_")[-1])].notna()][int(measure.split("_")[-1])]).rename(columns = {int(measure.split("_")[-1]) : "target"})
            df_target = df_label.rename(columns ={ windows_array[0] : "target" })
            df_target = df_target.set_index(np.arange(len(df_target)))
            #print(df_features.join(df_target))
            df_tab.append(df_features.join(df_target))
        return df_tab

    def save_df(self, df, video_name, dataset_path, measure=""):
        print("saving " + video_name)
        if measure == "":
            self.sftp.save_remote_df(os.path.join(dataset_path,video_name,video_name+".csv"),df, index = False)
        else : 
            self.sftp.save_remote_df(os.path.join(dataset_path,video_name,video_name+"_"+str(measure)+".csv"), df, index =False)
             
    def concat_dataset(self, dataset_array):
        #[df.pop("target") for index, df in enumerate(dataset_array) if index !=len(dataset_array)-1]
        for df in dataset_array:
            target  = df.pop("target")
        df_concat = pd.concat(dataset_array, axis =1)
        df_concat["target"] = target
        #print(df_concat)
        return df_concat
    
    def create_dataset_from_measure_folder(self, path_to_measure_folder, windows, path_folder_to_save):
        ##TODODATABASE: list dir methode 
        dir_measures = self.sftp.list_dir_remote( path_to_measure_folder)
        date_id = datetime.now().strftime("%Y_%m_%d_%H_%M")
        path_csv_arr = [path_to_measure_folder+"/"+ dir_name+"/"+dir_name+".csv" for dir_name in dir_measures]
        print(path_csv_arr)
        df_measures = pd.DataFrame()
        for path in path_csv_arr:
            print(path)
            df_measures = df_measures.append(self.sftp.read_remote_df(path), ignore_index=False)
        if os.path.isdir("temp") == False :
                    os.makedirs("temp" )
        df_measures.to_csv("temp/temp.csv", index=False)
        path_folder_to_put = os.path.join(path_folder_to_save,"dataset_merge_"+str(windows[0])+"_"+date_id+".csv")
        self.sftp.put_file(path_folder_to_put, os.getcwd()+"/temp/temp.csv")
        if os.path.exists("temp/temp.csv"):
            os.remove("temp/temp.csv")
    
    def generate_cross_dataset_by_week(self, path_to_dataset):
        dir_measures = self.sftp.list_dir_remote(path_to_dataset)
        path_csv_arr = [path_to_dataset+"/"+ dir_name+"/"+dir_name+".csv" for dir_name in dir_measures]
        
        dataframe_lundi = pd.DataFrame()   
        for path in path_csv_arr:
            video_name = path.split("/")[-2]
            dataframe = self.sftp.read_remote_df(path)
            print(video_name + " read")
            if "LUNDI" in re.split("_| ", video_name) : 
                dataframe_lundi = dataframe_lundi.append(dataframe, ignore_index=False)
            else:
                ##TODO save dataframe and put it instead of thath :
                self.sftp.save_remote_df(os.path.join(PATH_TO_DATASET_TEMPORAL,"debt","cross_dataset_week","test",video_name,"dataset.csv"), dataframe, index=False)
        
        if os.path.isdir("temp") == False :
            os.makedirs("temp" )
        dataframe_lundi.to_csv("temp/temp.csv", index=False)
        self.sftp.put_file(os.path.join(PATH_TO_DATASET_TEMPORAL,"debt","cross_dataset_week","train","dataset.csv"), os.getcwd()+"/temp/temp.csv")
        if os.path.exists("temp/temp.csv"):
            os.remove("temp/temp.csv")
        
       
    def generate_cross_dataset(self, path_to_dataset, path_to_dataset_to_save):
        dir_measures = self.sftp.list_dir_remote(path_to_dataset)
        path_csv_arr = [path_to_dataset+"/"+ dir_name+"/"+dir_name+".csv" for dir_name in dir_measures]
        
        dataframe_dict = {}    
        for path in path_csv_arr:
            video_name = path.split("/")[-2]
            dataframe = self.sftp.read_remote_df(path)
            dataframe_dict.update({video_name : dataframe})
                 
        df_measures = pd.DataFrame()
        for video_exclude in dir_measures:            
            for video_name in [path.split(sep="/")[-2] for path in path_csv_arr if path != path_to_dataset +"/"+ video_exclude +"/"+ video_exclude + ".csv"]:
                df_measures = df_measures.append(dataframe_dict[video_name], ignore_index=False)
            path_folder_to_save = os.path.join(path_to_dataset_to_save,"exclude_"+video_exclude,"dataset.csv")
            print(path_folder_to_save)
            if os.path.isdir("temp") == False :
                    os.makedirs("temp" )
            df_measures.to_csv("temp/temp.csv", index=False)
            self.sftp.put_file(path_folder_to_save, os.getcwd()+"/temp/temp.csv")
            if os.path.exists("temp/temp.csv"):
                os.remove("temp/temp.csv")
            df_measures = pd.DataFrame()
            
    def generate_dataset_debt_sleep(self, video_name, measures, df_measure= pd.DataFrame(), path = None, fps =None): 
        df_pvt_total = self.sftp.read_remote_df(os.path.join(PATH_TO_IRBA_DATA_PVT,"sujets_data_pvt_perf.csv"), sep=";", index_col = [0,1])
        subject_conditions = dict(df_pvt_total.index)
        subject_to_label = video_name.split("_")[2]
        condition = subject_conditions[subject_to_label]
        if path != None:
            df = self.sftp.save_remote_df(path)
        if not df_measure.empty:
            df = df_measure[measures]
            df_label = df.append( pd.DataFrame(columns=['Target']))
            #print(df_label[df_label["frame"].between(0,num_frame_by_num_min*2)])
            if condition == "dette" : 
                df_label["Target"] = 1
            else :
                df_label["Target"] = 0
        df_label = df_label.set_index("frame")
        columns_measures = [col for col in df_label.columns if col !=  "Target"]
        df_label[columns_measures] = (df_label[columns_measures]-df_label[columns_measures].min())/(df_label[columns_measures].max()-df_label[columns_measures].min())
        return df_label
    
    def create_VAS_dataset(self):
                
        df = self.sftp.read_remote_df(PATH_TO_IRBA_DATA_VAS, usecols=["Subject", "Fatigue (Apres PVT)", "Fatigue (Avant PVT)", "Date"])

        df.loc[lambda df: df["Fatigue (Apres PVT)"].between(0, 25), "Fatigue (Apres PVT)"] = 0
        df.loc[lambda df: df["Fatigue (Apres PVT)"].between(25, 50), "Fatigue (Apres PVT)"] = 1
        df.loc[lambda df: df["Fatigue (Apres PVT)"].between(50, 75), "Fatigue (Apres PVT)"] = 2
        df.loc[lambda df: df["Fatigue (Apres PVT)"].between(75, 100), "Fatigue (Apres PVT)"] = 3

        df.loc[lambda df: df["Fatigue (Avant PVT)"].between(0, 25), "Fatigue (Avant PVT)"] = 0
        df.loc[lambda df: df["Fatigue (Avant PVT)"].between(25, 50), "Fatigue (Avant PVT)"] = 1
        df.loc[lambda df: df["Fatigue (Avant PVT)"].between(50, 75), "Fatigue (Avant PVT)"] = 2
        df.loc[lambda df: df["Fatigue (Avant PVT)"].between(75, 100), "Fatigue (Avant PVT)"] = 3
        
        df.loc[lambda df: df["Date"].apply(lambda el: pd.to_datetime(el,dayfirst=True).day_name()) == "Monday", "Day"] = "LUNDI"
        df.loc[lambda df: df["Date"].apply(lambda el: pd.to_datetime(el,dayfirst=True).day_name()) == "Friday", "Day"] = "VENDREDI"
        df.pop("Date")
        df["Subject_Day"] = df["Subject"] + "_" + df["Day"]
        df.set_index("Subject_Day", inplace=True)
        
        videos_names = self.sftp.list_dir_remote(PATH_TO_TIME_ON_TASK_VIDEO)
        subject_list = df
        for name in videos_names:
            name_sanitize = "_".join(re.split("_| ", name)[2:4])
            if name_sanitize in list(df.index):
                df_dataset = self.sftp.read_remote_df(os.path.join(PATH_TO_TIME_ON_TASK_VIDEO, name, name+".csv"))
                df_dataset.loc[lambda df: df["target"] == 0, "target"] = df.loc[name_sanitize, "Fatigue (Avant PVT)"] 
                df_dataset.loc[lambda df: df["target"] == 1, "target"] = df.loc[name_sanitize, "Fatigue (Apres PVT)"] 
                self.sftp.save_remote_df(os.path.join(PATH_TO_TIME_ON_TASK_VIDEO_VAS,name_sanitize,name_sanitize+".csv"),df_dataset, index=False )
            else:
                print("Not")
        
        
## TODO:  add video anme and stuff in csv video infos

## TODO: make mother class for herite some commun variable (csv_infos ect...)

## TODO: fiw 'Unnamed: 0' columns  (coreseponding to frame) in df_temporal
