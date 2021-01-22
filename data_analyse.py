import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from utils_functions import parse_path_to_name

VIDEOS_INFOS_PATH = "data/data_out/videos_infos.csv"

class AnalyseData():
    def __init__(self, csv_path):
        self.df_landmarks = pd.read_csv(csv_path).rename(columns={"Unnamed: 0" : "frame"})
        self.df_measure = pd.DataFrame( self.df_landmarks["frame"])
        self.df_videos_infos = pd.read_csv(VIDEOS_INFOS_PATH)
        self.video_name = parse_path_to_name(csv_path)
        self.measures_computes = []

    def measure_euclid_dist(self, landmarks_1, landmarks_2):
        x_1 = "landmarks_"+str(landmarks_1)+"_x"
        y_1 = "landmarks_"+str(landmarks_1)+"_y"
        x_2 = "landmarks_"+str(landmarks_2)+"_x"
        y_2 = "landmarks_"+str(landmarks_2)+"_y"
        a = self.df_landmarks[[x_1,y_1]].rename(columns={x_1 : "x", y_1 :"y"})
        b = self.df_landmarks[[x_2,y_2]].rename(columns={x_2 : "x", y_2 :"y"})
        return (a-b).apply(np.linalg.norm,axis=1)

   #not finished
    def measure_yawning_frequency(self):
        self.df_measure["mouth"] = (self.measure_euclid_dist(62,68) + self.measure_euclid_dist(63,67) + self.measure_euclid_dist(64,68)) / (2*self.measure_euclid_dist(61,65))
        #find the value that indicates a yawning
        #if this value is reached, add +1 on the frequency count

    #function that displays the blinking
    def measure_blinking(self):
        #self.df_measure["eye_l"] = (self.measure_euclid_dist(38,42) + self.measure_euclid_dist(39,41)) / (2*self.measure_euclid_dist(37,40))
        #self.df_measure["eye_r"] = (self.measure_euclid_dist(44,48) + self.measure_euclid_dist(45,47)) / (2*self.measure_euclid_dist(43,46))
        #self.df_measure["eye"]   = (self.df_measure["eye_r"] +self.df_measure["eye_l"])/2
        self.df_measure["eye_l"] = (self.measure_euclid_dist(39,41))
        self.df_measure["eye_r"] = (self.measure_euclid_dist(45,47))
        self.df_measure["eye"]   = (self.df_measure["eye_r"] +self.df_measure["eye_l"])/2

    #function that computes the number of blinks
    def blinking_frequency(self, threshold):
        self.df_measure["eye_l"] = (self.measure_euclid_dist(39,41))
        self.df_measure["eye_r"] = (self.measure_euclid_dist(45,47))
        self.df_measure["eye"]   = (self.df_measure["eye_r"] +self.df_measure["eye_l"])/2
        #we select the values that are below 3.0
        self.df_measure["eyes_frame"] = self.df_measure[self.df_measure["eye"].between(2.0,3.0)]["frame"]
        #find the blinking frequency for each minute
        for i in range(0,int(self.df_measure["frame"].max()/threshold)):
            blinking_measures2 = self.df_measure[self.df_measure["eyes_frame"].between(i*threshold,threshold*(i+1))]["eye"]
            peaks= find_peaks(np.array(blinking_measures2), height=3.0)
            self.df_measure["blinking_frequence"] = len(peaks[0])
        print(self.df_measure["blinking_frequence"])

    def measure_ear(self): # calculate
        self.df_measure["ear_l"] = (self.measure_euclid_dist(38,42) + self.measure_euclid_dist(39,41)) / (2*self.measure_euclid_dist(37,40))
        self.df_measure["ear_r"] = (self.measure_euclid_dist(44,48) + self.measure_euclid_dist(45,47)) / (2*self.measure_euclid_dist(43,46))
        self.df_measure["ear"]   = (self.df_measure["ear_r"] +self.df_measure["ear_l"])/2
        self.df_measure["ear_std"] = self.df_measure["ear"].std()
        self.measures_computes.append({"measure" : "ear" , "axis_x" : "frame"})

    def measure_eyebrows_nose(self):
        self.df_measure["eyebrowns_nose_l"] = (self.measure_euclid_dist(20,32))
        self.df_measure["eyebrowns_nose_r"] = (self.measure_euclid_dist(25,36))
        self.df_measure["eyebrowns_nose"]   = (self.df_measure["eyebrowns_nose_r"] + self.df_measure["eyebrowns_nose_r"])/ 2
        #print(self.df_measure)

    def measure_eye_area(self):
        self.df_measure["eye_area_l"] = (self.measure_euclid_dist(37, 40) / 2) * ((self.measure_euclid_dist(38, 42) + self.measure_euclid_dist(39,41)) /2) * np.pi
        self.df_measure["eye_area_r"] = (self.measure_euclid_dist(43, 46) / 2) * ((self.measure_euclid_dist(44, 48) + self.measure_euclid_dist(45,47)) /2) * np.pi
        self.df_measure["eye_area"] = (self.df_measure["eye_area_l"] + self.df_measure["eye_area_r"])/2

    def measure_mean_eye_area(self, threshold, percent = False):
        self.measure_eye_area()
        self.df_measure["eye_area_theshold"] = pd.DataFrame(np.arange(self.df_measure["frame"].max()/threshold))
        max_eye_area = self.df_measure["eye_area"].max()
        eye_area_mean = []
        for i in range(0,len(np.arange(self.df_measure["frame"].max()/threshold))):
            if percent : eye_area_mean.append(self.df_measure[self.df_measure["frame"].between(i*threshold,threshold*(i+1))]["eye_area"].mean()*100/max_eye_area)
            else : eye_area_mean.append(self.df_measure[self.df_measure["frame"].between(i*threshold,threshold*(i+1))]["eye_area"].mean())
        self.df_measure["eye_area_mean_over_"+str(threshold)+"_frame"] = pd.DataFrame(eye_area_mean)
        self.df_measure["eye_area_mean_over_"+str(threshold)+"_frame_std"] = self.df_measure["eye_area_mean_over_"+str(threshold)+"_frame"].std()
        self.measures_computes.append({"measure" : "eye_area_mean_over_"+str(threshold)+"_frame" , "axis_x" : "eye_area_theshold"})

    def measure_mean_eye_area_curve(self, threshold, percent = False):
        self.measure_eye_area()
        self.df_measure["eye_area_theshold"] = pd.DataFrame(np.arange(self.df_measure["frame"].max()/threshold))
        max_eye_area = self.df_measure["eye_area"].max()
        eye_area_mean = []
        for i in range(0,len(np.arange(self.df_measure["frame"].max()/threshold))):
            if percent : eye_area_mean.append(self.df_measure[self.df_measure["frame"].between(i,threshold*(i+1))]["eye_area"].mean()*100/max_eye_area)
            else : eye_area_mean.append(self.df_measure[self.df_measure["frame"].between(i,threshold*(i+1))]["eye_area"].mean())
        self.df_measure["eye_area_mean_over_"+str(threshold)+"_frame"] = pd.DataFrame(eye_area_mean)

    def plot_measure(self, measure, axis_x = "frame"):
        discontinuities_frame  = self.find_discontinuities()
        video_fps = self.df_videos_infos[self.df_videos_infos["video_name"] == self.video_name]["fps"]
        #print(float(video_fps))
        if axis_x == "frame" :
            for index in discontinuities_frame:
                plt.plot(self.df_measure[self.df_measure[axis_x].between(index[0],index[1])][axis_x]/float(video_fps), self.df_measure[self.df_measure[axis_x].between(index[0],index[1])][measure])
            plt.xlabel("sec")
        else :
            plt.plot(self.df_measure[axis_x], self.df_measure[measure])
            plt.xlabel(axis_x)
        plt.ylabel(measure)
        plt.show()

    def plot_points_measure(self, measure):
        discontinuities_frame  = self.find_discontinuities()
        video_fps = self.df_videos_infos[self.df_videos_infos["video_name"] == self.video_name]["fps"]
        for index in discontinuities_frame:
            plt.scatter(self.df_measure[self.df_measure["frame"].between(index[0],index[1])]["frame"]/video_fps[0], self.df_measure[self.df_measure["frame"].between(index[0],index[1])][measure])
        plt.xlabel("sec")
        plt.ylabel(measure)
        plt.show()

    def find_discontinuities(self):
        cmp = 0
        discontinuities_frame = [0]
        for index, row in self.df_measure.iterrows():
            if(row['frame'] != cmp) :
                discontinuities_frame.append(cmp-1)
                discontinuities_frame.append(row['frame'])
                cmp = row['frame']
            cmp = cmp + 1
        discontinuities_frame.append( cmp-1)
        result = zip(discontinuities_frame[::2], discontinuities_frame[1::2])
        return list(result)
    
    ##TODO : Make DF for corespondance of measure and axis : ex : ear -> sec, mean_eaye_area -> threeshold  
    def plot_multiple_measures(self):
        number_subplot = len(self.measures_computes)
        fig, axes = plt.subplots(number_subplot) 
        video_fps = self.df_videos_infos[self.df_videos_infos["video_name"] == self.video_name]["fps"]
        
        for axe_number, measure in enumerate(self.measures_computes):
            
            if measure.get("measure")+"_std" in self.df_measure.columns:   
                axes[axe_number].errorbar(x = self.df_measure[measure.get("axis_x")]/float(video_fps), y = self.df_measure[measure.get("measure")], yerr = self.df_measure[measure.get("measure")], ecolor ='g'  )
            else : 
                axes[axe_number].plot(self.df_measure[measure.get("axis_x")]/float(video_fps), self.df_measure[measure.get("measure")])
            axes[axe_number].set(xlabel=measure.get("axis_x"), ylabel=measure.get("measure"))
        plt.show()
    
    def compute_std(self, measure_name):
        self.df_measure[measure_name+"_std"] = self.df_measure[measure_name].std()

ad = AnalyseData("data/data_out/DESFAM_Semaine 2-Vendredi_Go-NoGo_H69.csv")
threshold = int(ad.df_videos_infos[ad.df_videos_infos["video_name"] == ad.video_name]["fps"].item() * 30)

ad.measure_ear()
ad.measure_mean_eye_area(threshold)

ad.plot_multiple_measures()
