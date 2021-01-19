import pandas as pd
import numpy as np

class DataFormator:   
    def __init__(self):
        self.df_csv_files = {}
        self.df_formatted = None
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
            print(self.df_formatted)
            for key, value in self.df_csv_files.items():
                if key not in ["hog"]:
                    value.set_index("frame")
                    for frame in value["frame"] :
                        if self.df_formatted[self.df_formatted["frame"] == frame].empty:
                            self.df_formatted =  self.df_formatted.append(value[value["frame"] == frame])
                        else: 
                            print(self.df_formatted.loc[self.df_formatted["frame"] == frame, self.df_formatted.columns != "frame"] + value.loc[value["frame"] == frame, value.columns != "frame"]/2)
            self.df_formatted.sort_values(by="frame")
            
            
df = DataFormator()
df.load_csv("DESFAM_Semaine 2-Vendredi_Go-NoGo_H69")
#df.fusion_csv(mode = "mean")

def measure_euclid_dist(landmarks_1, landmarks_2, df):
        x_1 = "landmarks_"+str(landmarks_1)+"_x"
        y_1 = "landmarks_"+str(landmarks_1)+"_y"
        x_2 = "landmarks_"+str(landmarks_2)+"_x"
        y_2 = "landmarks_"+str(landmarks_2)+"_y"
        a = df[[x_1,y_1]].rename(columns={x_1 : "x", y_1 :"y"})
        b = df[[x_2,y_2]].rename(columns={x_2 : "x", y_2 :"y"})
        return (a-b).apply(np.linalg.norm,axis=1)

print(measure_euclid_dist(38, 42, df.df_csv_files.get("cnn")))
print(measure_euclid_dist(38, 42, df.df_csv_files.get("hog")))