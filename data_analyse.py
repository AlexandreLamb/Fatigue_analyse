import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.signal import find_peaks
from itertools import groupby
from operator import itemgetter
from utils import parse_path_to_name
from matplotlib.figure import Figure

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

    def measure_yawning_frequency(self, threshold):
        self.df_measure["mouth"] = self.measure_euclid_dist(63,67)
        #find the value that indicates a yawning // we select the values that are above 38
        self.df_measure["mouth_frame"] = self.df_measure[self.df_measure["mouth"] > (38)]["frame"]
        #we decide the axis we want for the display
        self.df_measure["Time (in mins)"] = pd.DataFrame(np.arange((self.df_measure["frame"].max()/threshold)))
        #we select only the highest peak
        y_frequency_list = []
        #we want the yawning frequency for each minute (hence the threshold)
        for i in range(0,int(self.df_measure["frame"].max()/threshold)):
            yawning_measures = self.df_measure[self.df_measure["mouth_frame"].between(i*threshold,threshold*(i+1))]["mouth"]
            yawning_array = (np.array(yawning_measures))
            peaks= find_peaks(yawning_array, height = 30, distance = 5)
            peaks_values = peaks[0]
            peaks_cleaned = np.diff(peaks_values)
            peaks_highest = [peaks_values[0] for peaks_values in groupby(peaks_cleaned)]
            if (len(peaks_values) != 0):
                y_frequency_list.append(len(peaks_highest))
                print(y_frequency_list)
            else: 
                y_frequency_list.append(0)
        self.df_measure["yawning_frequency"] = pd.DataFrame(y_frequency_list)
        return self.df_measure["yawning_frequency"]
    
    #function that computes the number of blinks
    def blinking_frequency(self, threshold):
        self.df_measure["eye_l"] = self.measure_euclid_dist(39,41)
        self.df_measure["eye_r"] = self.measure_euclid_dist(45,47)
        self.df_measure["eye"]   = (self.df_measure["eye_r"] +self.df_measure["eye_l"])/2
        #we select the values that are below 8
        self.df_measure["eyes_frame"] = self.df_measure[self.df_measure["eye"].between(2.0,8.0)]["frame"]
        #print(self.df_measure["eyes_frame"])
        #we decide the axis we want for the display
        self.df_measure["Time (in mins)"] = pd.DataFrame(np.arange((self.df_measure["frame"].max()/threshold)))
        #find the blinking frequency for each minute
        b_frequency_list = []
        for i in range(0,int(self.df_measure["frame"].max()/threshold)):
            blinking_measures = self.df_measure[self.df_measure["eyes_frame"].between(i*threshold,threshold*(i+1))]["eye"]
            peaks= find_peaks(np.array(blinking_measures))
            b_frequency_list.append(len(peaks[0]))
            #print(b_frequency_list)
        self.df_measure["blinking_frequency"] = pd.DataFrame(b_frequency_list)
        return self.df_measure["blinking_frequency"]

    def measure_ear(self): # calculate
        self.df_measure["ear_l"] = (self.measure_euclid_dist(38,42) + self.measure_euclid_dist(39,41)) / (2*self.measure_euclid_dist(37,40))
        self.df_measure["ear_r"] = (self.measure_euclid_dist(44,48) + self.measure_euclid_dist(45,47)) / (2*self.measure_euclid_dist(43,46))
        self.df_measure["ear"]   = (self.df_measure["ear_r"] +self.df_measure["ear_l"])/2
        self.df_measure["ear_std"] = self.df_measure["ear"].std()
        self.measures_computes.append({"measure" : "ear" , "axis_x" : "frame"})
        return self.measures_computes

    def measure_perclos(self, threshold, percentage):
        #first, we get the distances for each eye and do the mean of it 
        self.df_measure["eye_l"] = self.measure_euclid_dist(39,41)
        self.df_measure["eye_r"] = self.measure_euclid_dist(45,47)
        self.df_measure["eye"]   = (self.df_measure["eye_r"] + self.df_measure["eye_l"])/2
        self.df_measure["eyes_frame"] = self.df_measure[self.df_measure["eye"].between(0,10.0)]["frame"]
        self.df_measure["Time (in mins)"] = pd.DataFrame(np.arange((self.df_measure["frame"].max()/threshold)))
        #we get the percentage for the PERCLOS measure (either 80 or 70)
        percentage1 = percentage 
        percentage2 = 100 - percentage
        perclos_list = []
        #find the the 4 timestamps needed for the perclos measure BY MINUTE
        for i in range(0,int(self.df_measure["frame"].max()/threshold)):
            #find the highest distance aka the largest pupil
            pupil_measures = self.df_measure[self.df_measure["eyes_frame"].between(i*threshold,threshold*(i+1))]
            pupil_measures = pupil_measures[["eye", "frame"]]
            pupil_measures_array = np.array(pupil_measures["eye"])
            highest_value = max(pupil_measures_array)
            borne1 = (percentage1 * highest_value)/100
            borne2 = (percentage2 * highest_value)/100
            t1 = pupil_measures[pupil_measures["eye"] < borne1]["frame"].min()
            t2 = pupil_measures[pupil_measures["eye"] < borne2]["frame"].min()
            pupil = pupil_measures[pupil_measures["frame"] > t1]
            t3 = pupil[pupil["eye"] > borne2]["frame"].min()
            t4 = pupil[pupil["eye"] > borne1]["frame"].min()
            #if the values doesn't go below one of the "bornes" the timestamp equals 0 to simplify the perclos calculus
            if(math.isnan(t1)): 
                t1 = 0
                perclos = (t3 - t2)/(t4 - t1)
            if(math.isnan(t2)): 
                t2 = 0
                perclos = (t3 - t2)/(t4 - t1)
            if(math.isnan(t3)): 
                t3 = 0
                perclos = (t3 - t2)/(t4 - t1)
            if(math.isnan(t4)): 
                t4 = 0
                perclos = 0
            perclos_list.append(perclos)
        self.df_measure["perclos_measure"] = pd.DataFrame(perclos_list)
        return self.df_measure["perclos_measure"]
            
    def measure_microsleep(self, microsleep):
        frames_per_second = (int(self.df_measure["frame"].max()))/60
        self.df_measure["eye_l"] = (self.measure_euclid_dist(39,41))
        self.df_measure["eye_r"] = (self.measure_euclid_dist(45,47))
        self.df_measure["eye"]   = (self.df_measure["eye_r"] +self.df_measure["eye_l"])/2
        self.df_measure["eyes_frame"] = self.df_measure[self.df_measure["eye"] <2.5]["frame"]
        microsleep_frequency = 0
        frame = self.df_measure["eyes_frame"].dropna()
        for k, g in groupby(enumerate(frame), lambda ix : ix[0] - ix[1]):
            frame_map = list(map(itemgetter(1), g))
            frame_len = len(frame_map)
            if frame_len > (microsleep * frames_per_second):
                    microsleep_frequency = microsleep_frequency + 1
        self.df_measure["microsleep_measure"] = microsleep_frequency 
  
    def measure_eyebrow_nose(self):
        self.df_measure["eyebrow_nose_l"] = (self.measure_euclid_dist(20,32))
        self.df_measure["eyebrow_nose_r"] = (self.measure_euclid_dist(25,36))
        self.df_measure["eyebrow_nose"]   = (self.df_measure["eyebrow_nose_r"] + self.df_measure["eyebrow_nose_r"])/ 2
        return self.df_measure["eyebrow_nose"]

    def nose_wrinkles(self):
        self.df_measure["eyebrow_eye_l"] = (self.measure_euclid_dist(22,40))
        self.df_measure["eyebrow_eye_r"] = (self.measure_euclid_dist(23,43))
        self.df_measure["eyebrow_eye"]   = (self.df_measure["eyebrow_eye_l"] + self.df_measure["eyebrow_eye_r"])/ 2
        return self.df_measure

    def eyes_angle(self):

        #right eye
        #angle 1
        #vector 1
        r1_v1_p1_x1 = "landmarks_"+str(45)+"_x"
        r1_v1_p1_y1 = "landmarks_"+str(45)+"_y"
        r1_v1_p2_x2 = "landmarks_"+str(43)+"_x"
        r1_v1_p2_y2 = "landmarks_"+str(43)+"_y"
        r1_v1_p1_x = (self.df_landmarks[[r1_v1_p1_x1,r1_v1_p1_y1]])["landmarks_45_x"]
        r1_v1_p1_y = (self.df_landmarks[[r1_v1_p1_x1,r1_v1_p1_y1]])["landmarks_45_y"]
        r1_v1_p2_x = (self.df_landmarks[[r1_v1_p2_x2,r1_v1_p2_y2]])["landmarks_43_x"]
        r1_v1_p2_y = (self.df_landmarks[[r1_v1_p2_x2,r1_v1_p2_y2]])["landmarks_43_y"]

        #vector 2
        r1_v2_p1_x1 = "landmarks_"+str(43)+"_x"
        r1_v2_p1_y1 = "landmarks_"+str(43)+"_y"
        r1_v2_p2_x2 = "landmarks_"+str(46)+"_x"
        r1_v2_p2_y2 = "landmarks_"+str(46)+"_y"
        r1_v2_p1_x = (self.df_landmarks[[r1_v2_p1_x1,r1_v2_p1_y1]])["landmarks_43_x"]
        r1_v2_p1_y = (self.df_landmarks[[r1_v2_p1_x1,r1_v2_p1_y1]])["landmarks_43_y"]
        r1_v2_p2_x = (self.df_landmarks[[r1_v2_p2_x2,r1_v2_p2_y2]])["landmarks_46_x"]
        r1_v2_p2_y = (self.df_landmarks[[r1_v2_p2_x2,r1_v2_p2_y2]])["landmarks_46_y"]

        # Get nicer vector form
        r1_vA = [(r1_v1_p1_x - r1_v1_p2_x), ( r1_v1_p1_y - r1_v1_p2_y)]
        r1_vB = [(r1_v2_p1_x - r1_v2_p2_x), ( r1_v2_p1_y - r1_v2_p2_y)]
        # Get dot prod
        r1_dot_prod = r1_vA[0]*r1_vB[0]+r1_vA[1]*r1_vB[1]
        # Get magnitudes
        r1_magA = (r1_vA[0]*r1_vA[0]+r1_vA[1]*r1_vA[1])**0.5
        r1_magB = (r1_vB[0]*r1_vB[0]+r1_vB[1]*r1_vB[1])**0.5
        # Get angle in radians and then convert to degrees
        r_angle1_array = r1_dot_prod/r1_magB/r1_magA
        r_angle1_list = []
        for i in range(0,len(r_angle1_array)):
            r_angle1_measures = r_angle1_array[i]
            r_angle1 = math.acos(r_angle1_measures)
            r_angle1_list.append(r_angle1)
        self.df_measure["right_angle1"] = r_angle1_list 
            
        #angle 2
        #vector 1
        r2_v1_p1_x1 = "landmarks_"+str(46)+"_x"
        r2_v1_p1_y1 = "landmarks_"+str(46)+"_y"
        r2_v1_p2_x2 = "landmarks_"+str(47)+"_x"
        r2_v1_p2_y2 = "landmarks_"+str(47)+"_y"
        r2_v1_p1_x = (self.df_landmarks[[r2_v1_p1_x1,r2_v1_p1_y1]])["landmarks_46_x"]
        r2_v1_p1_y = (self.df_landmarks[[r2_v1_p1_x1,r2_v1_p1_y1]])["landmarks_46_y"]
        r2_v1_p2_x = (self.df_landmarks[[r2_v1_p2_x2,r2_v1_p2_y2]])["landmarks_47_x"]
        r2_v1_p2_y = (self.df_landmarks[[r2_v1_p2_x2,r2_v1_p2_y2]])["landmarks_47_y"]
        
        #vector 2
        r2_v2_p1_x1 = "landmarks_"+str(47)+"_x"
        r2_v2_p1_y1 = "landmarks_"+str(47)+"_y"
        r2_v2_p2_x2 = "landmarks_"+str(43)+"_x"
        r2_v2_p2_y2 = "landmarks_"+str(43)+"_y"
        r2_v2_p1_x = (self.df_landmarks[[r2_v2_p1_x1,r2_v2_p1_y1]])["landmarks_47_x"]
        r2_v2_p1_y = (self.df_landmarks[[r2_v2_p1_x1,r2_v2_p1_y1]])["landmarks_47_y"]
        r2_v2_p2_x = (self.df_landmarks[[r2_v2_p2_x2,r2_v2_p2_y2]])["landmarks_43_x"]
        r2_v2_p2_y = (self.df_landmarks[[r2_v2_p2_x2,r2_v2_p2_y2]])["landmarks_43_y"]

        # Get nicer vector form
        r2_vA = [(r2_v1_p1_x - r2_v1_p2_x), ( r2_v1_p1_y - r2_v1_p2_y)]
        r2_vB = [(r2_v2_p1_x - r2_v2_p2_x), ( r2_v2_p1_y - r2_v2_p2_y)]
        # Get dot prod
        r2_dot_prod = r2_vA[0]*r2_vB[0]+r2_vA[1]*r2_vB[1]
        # Get magnitudes
        r2_magA = (r2_vA[0]*r2_vA[0]+r2_vA[1]*r2_vA[1])**0.5
        r2_magB = (r2_vB[0]*r2_vB[0]+r2_vB[1]*r2_vB[1])**0.5
        # Get angle in radians and then convert to degrees
        r_angle2_array = r2_dot_prod/r2_magB/r2_magA
        r_angle2_list = []
        for i in range(0,len(r_angle2_array)):
            r_angle2_measures = r_angle2_array[i]
            r_angle2 = math.acos(r_angle2_measures)
            r_angle2_list.append(r_angle2)
        self.df_measure["right_angle2"] = r_angle2_list 

        """""""""""""""""""""""""""""""""""
        """""""""""""""""""""""""""""""""""

        #left eye
        #angle 1
        #vector 1
        l1_v1_p1_x1 = "landmarks_"+str(38)+"_x"
        l1_v1_p1_y1 = "landmarks_"+str(38)+"_y"
        l1_v1_p2_x2 = "landmarks_"+str(40)+"_x"
        l1_v1_p2_y2 = "landmarks_"+str(40)+"_y"
        l1_v1_p1_x = (self.df_landmarks[[l1_v1_p1_x1,l1_v1_p1_y1]])["landmarks_38_x"]
        l1_v1_p1_y = (self.df_landmarks[[l1_v1_p1_x1,l1_v1_p1_y1]])["landmarks_38_y"]
        l1_v1_p2_x = (self.df_landmarks[[l1_v1_p2_x2,l1_v1_p2_y2]])["landmarks_40_x"]
        l1_v1_p2_y = (self.df_landmarks[[l1_v1_p2_x2,l1_v1_p2_y2]])["landmarks_40_y"]

        #vector 2
        l1_v2_p1_x1 = "landmarks_"+str(40)+"_x"
        l1_v2_p1_y1 = "landmarks_"+str(40)+"_y"
        l1_v2_p2_x2 = "landmarks_"+str(37)+"_x"
        l1_v2_p2_y2 = "landmarks_"+str(37)+"_y"
        l1_v2_p1_x = (self.df_landmarks[[l1_v2_p1_x1,l1_v2_p1_y1]])["landmarks_40_x"]
        l1_v2_p1_y = (self.df_landmarks[[l1_v2_p1_x1,l1_v2_p1_y1]])["landmarks_40_y"]
        l1_v2_p2_x = (self.df_landmarks[[l1_v2_p2_x2,l1_v2_p2_y2]])["landmarks_37_x"]
        l1_v2_p2_y = (self.df_landmarks[[l1_v2_p2_x2,l1_v2_p2_y2]])["landmarks_37_y"]

        # Get nicer vector form
        l1_vA = [(l1_v1_p1_x - l1_v1_p2_x), ( l1_v1_p1_y - l1_v1_p2_y)]
        l1_vB = [(l1_v2_p1_x - l1_v2_p2_x), ( l1_v2_p1_y - l1_v2_p2_y)]
        # Get dot prod
        l1_dot_prod = l1_vA[0]*l1_vB[0]+l1_vA[1]*l1_vB[1]
        # Get magnitudes
        l1_magA = (l1_vA[0]*l1_vA[0]+l1_vA[1]*l1_vA[1])**0.5
        l1_magB = (l1_vB[0]*l1_vB[0]+l1_vB[1]*l1_vB[1])**0.5
        # Get angle in radians and then convert to degrees
        l_angle1_array = l1_dot_prod/l1_magB/l1_magA
        l_angle1_list = []
        for i in range(0,len(l_angle1_array)):
            l_angle1_measures = l_angle1_array[i]
            l_angle1 = math.acos(l_angle1_measures)
            l_angle1_list.append(l_angle1)
        self.df_measure["left_angle1"] = l_angle1_list 

        #angle 2
        #vector 1
        l2_v1_p1_x1 = "landmarks_"+str(37)+"_x"
        l2_v1_p1_y1 = "landmarks_"+str(37)+"_y"
        l2_v1_p2_x2 = "landmarks_"+str(42)+"_x"
        l2_v1_p2_y2 = "landmarks_"+str(42)+"_y"
        l2_v1_p1_x = (self.df_landmarks[[l2_v1_p1_x1,l2_v1_p1_y1]])["landmarks_37_x"]
        l2_v1_p1_y = (self.df_landmarks[[l2_v1_p1_x1,l2_v1_p1_y1]])["landmarks_37_y"]
        l2_v1_p2_x = (self.df_landmarks[[l2_v1_p2_x2,l2_v1_p2_y2]])["landmarks_42_x"]
        l2_v1_p2_y = (self.df_landmarks[[l2_v1_p2_x2,l2_v1_p2_y2]])["landmarks_42_y"]
        
        #vector 2
        l2_v2_p1_x1 = "landmarks_"+str(42)+"_x"
        l2_v2_p1_y1 = "landmarks_"+str(42)+"_y"
        l2_v2_p2_x2= "landmarks_"+str(40)+"_x"
        l2_v2_p2_y2 = "landmarks_"+str(40)+"_y"
        l2_v2_p1_x = (self.df_landmarks[[l2_v2_p1_x1,l2_v2_p1_y1]])["landmarks_42_x"]
        l2_v2_p1_y = (self.df_landmarks[[l2_v2_p1_x1,l2_v2_p1_y1]])["landmarks_42_y"]
        l2_v2_p2_x = (self.df_landmarks[[l2_v2_p2_x2,l2_v2_p2_y2]])["landmarks_40_x"]
        l2_v2_p2_y = (self.df_landmarks[[l2_v2_p2_x2,l2_v2_p2_y2]])["landmarks_40_y"]

        # Get nicer vector form
        l2_vA = [(l2_v1_p1_x - l2_v1_p2_x), ( l2_v1_p1_y - l2_v1_p2_y)]
        l2_vB = [(l2_v2_p1_x - l2_v2_p2_x), ( l2_v2_p1_y - l2_v2_p2_y)]
        # Get dot prod
        l2_dot_prod = l2_vA[0]*l2_vB[0]+l2_vA[1]*l2_vB[1]
        # Get magnitudes
        l2_magA = (l2_vA[0]*l2_vA[0]+l2_vA[1]*l2_vA[1])**0.5
        l2_magB = (l2_vB[0]*l2_vB[0]+l2_vB[1]*l2_vB[1])**0.5
        # Get angle in radians and then convert to degrees
        l_angle2_array = l2_dot_prod/l2_magB/l2_magA
        l_angle2_list = []
        for i in range(0,len(l_angle2_array)):
            l_angle2_measures = l_angle2_array[i]
            l_angle2 = math.acos(l_angle2_measures)
            l_angle2_list.append(l_angle2)
        self.df_measure["left_angle2"] = l_angle2_list 
            
        return self.df_measure["left_angle1"], self.df_measure["left_angle2"], self.df_measure["right_angle1"], self.df_measure["right_angle2"]

    def jaw_dropping(self):
        self.df_measure["jaw_dropping"] = self.measure_euclid_dist(52,9)
        return self.df_measure["jaw_dropping"]
        
    def measure_eye_area(self):
        self.df_measure["eye_area_l"] = (self.measure_euclid_dist(37, 40) / 2) * ((self.measure_euclid_dist(38, 42) + self.measure_euclid_dist(39,41)) /2) * np.pi
        self.df_measure["eye_area_r"] = (self.measure_euclid_dist(43, 46) / 2) * ((self.measure_euclid_dist(44, 48) + self.measure_euclid_dist(45,47)) /2) * np.pi
        self.df_measure["eye_area"] = (self.df_measure["eye_area_l"] + self.df_measure["eye_area_r"])/2
        return self.df_measure["eye_area"]

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
        return self.measures_computes

    def measure_mean_eye_area_curve(self, threshold, percent = False):
        self.measure_eye_area()
        self.df_measure["eye_area_theshold"] = pd.DataFrame(np.arange(self.df_measure["frame"].max()/threshold))
        max_eye_area = self.df_measure["eye_area"].max()
        eye_area_mean = []
        for i in range(0,len(np.arange(self.df_measure["frame"].max()/threshold))):
            if percent : eye_area_mean.append(self.df_measure[self.df_measure["frame"].between(i,threshold*(i+1))]["eye_area"].mean()*100/max_eye_area)
            else : eye_area_mean.append(self.df_measure[self.df_measure["frame"].between(i,threshold*(i+1))]["eye_area"].mean())
        self.df_measure["eye_area_mean_over_"+str(threshold)+"_frame"] = pd.DataFrame(eye_area_mean)

    def plot_measure(self, measure, title, axis_x = "frame"):
        fig = Figure(figsize=(6,5))
        plt = fig.add_subplot(111)
        discontinuities_frame  = self.find_discontinuities()
        #c est ca qui merde
        video_fps = list(self.df_videos_infos[self.df_videos_infos["video_name"] == self.video_name]["fps"])
        print(video_fps)
        if axis_x == "frame" :
            for index in discontinuities_frame:
                plt.plot(self.df_measure[self.df_measure[axis_x].between(index[0],index[1])][axis_x]/video_fps[0], self.df_measure[self.df_measure[axis_x].between(index[0],index[1])][measure])
            plt.set_xlabel("Time (in sec)")
        else :
            plt.plot(self.df_measure[axis_x], self.df_measure[measure])
            plt.set_xlabel(axis_x, fontsize=10)
        plt.set_ylabel(measure,  fontsize=10)
        plt.set_title(title, fontsize=15)
        return fig
    
    def plot_multi_measure(self, measures, title, axis_x = "frame"):
        fig = Figure(figsize=(5,4))
        plt = fig.add_subplot(111)
        video_fps = list(self.df_videos_infos[self.df_videos_infos["video_name"] == self.video_name]["fps"])
        for measure in measures:
            if axis_x == "frame" :      
                plt.plot(self.df_measure[axis_x]/video_fps[0], self.df_measure[measure])
                plt.set_xlabel("Time (in sec)")
            else :
                plt.plot(self.df_measure[axis_x], self.df_measure[measure])
                plt.set_xlabel(axis_x, fontsize=10)
        plt.set_ylabel(measure, fontsize=10)
        plt.set_title(title, fontsize=15)
        return fig

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
    
    def compute_std(self, measure_name, file_path):
        self.df_measure[measure_name+"_std"] = self.df_measure[measure_name].std()
        return self.df_measure[measure_name+"_std"]
    
    def save_csv(self, measure, file_path, axis_x = "frame"):
        df_to_save = pd.DataFrame()
        video_fps = list(self.df_videos_infos[self.df_videos_infos["video_name"] == self.video_name]["fps"])
        if axis_x == "frame" :
                df_to_save["seconds"] = self.df_measure[axis_x]/video_fps[0]
                df_to_save["measure"] = self.df_measure[measure]
        else :
            df_to_save[axis_x] = self.df_measure[axis_x]
            df_to_save[measure] = self.df_measure[measure]
        print(df_to_save)
        df_to_save.to_csv(file_path)
            
             

      
# ad = AnalyseData("data/data_out/DESFAM_Semaine-2-Vendredi_PVT_H63_hog.csv")
# threshold = int(ad.df_videos_infos[ad.df_videos_infos["video_name"] == ad.video_name]["fps"].item() * 30)

# #EAR measure
# ad.measure_ear()
# ad.plot_measure("ear")

# #mean eye are measure
# ad.measure_mean_eye_area(30)
# ad.plot_measure("eye_area_mean_over_30_frame", "eye_area_theshold")

# #blinking measure
# ad.blinking_frequency(1500)
# ad.plot_measure("blinking_frequency", axis_x = "Time (in mins)")
# ad.plot_measure("eye")

# ad.measure_eyebrow_nose()
# ad.plot_measure("eyebrow_nose")

# ad.eyes_angle()
# ad.plot_multi_measure(["left_angle1","left_angle2"])
# ad.plot_multi_measure(["right_angle1","right_angle2"])

# ad.measure_perclos(1500, 80)
# ad.plot_measure("perclos_measure", axis_x = "Time (in mins)")

# ad.measure_microsleep(1)
# ad.plot_measure("microsleep_measure")

   