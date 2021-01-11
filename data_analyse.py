import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
landmarks_eyes_left = np.arange(36,42)
landmarks_eyes_rigth = np.arange(42,48)


def compute_ear():



df = pd.read_csv("data/landmarks.csv")

df["ear_l"] = (df["euclid_dist_38_42_l"]+df["euclid_dist_39_41_l"])/(2*df["euclid_dist_37_40_l"])
df["ear_r"] = (df["euclid_dist_44_48_r"]+df["euclid_dist_45_47_r"])/(2*df["euclid_dist_43_46_r"])
df["ear"] = (df["ear_l"]+df["ear_r"])/2

df[('ear')].plot()


plt.xlabel("frame")
plt.ylabel("eye aspect ratio")
plt.show()
"""

class AnalyseData():
    def __init__(self, csv_path):
        self.df_landmarks = pd.read_csv(csv_path).rename(columns={"Unnamed: 0" : "frame"})
        self.df_measure = pd.DataFrame()
        self.df_measure["frame"] = self.df_landmarks["frame"]

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

    #not finished
    def measure_blinking_frequency(self):
        self.df_measure["eye_l"] = (self.measure_euclid_dist(38,42) + self.measure_euclid_dist(39,41)) / (2*self.measure_euclid_dist(37,40))
        self.df_measure["eye_r"] = (self.measure_euclid_dist(44,48) + self.measure_euclid_dist(45,47)) / (2*self.measure_euclid_dist(43,46))
        self.df_measure["eye"]   = (self.df_measure["eye_r"] +self.df_measure["eye_l"])/2
        #when this measurement equals 0, we have a blink
        #if this happens, we add +1 on the blinking count


    def measure_ear(self): # calculate
        self.df_measure["ear_l"] = (self.measure_euclid_dist(38,42) + self.measure_euclid_dist(39,41)) / (2*self.measure_euclid_dist(37,40))
        self.df_measure["ear_r"] = (self.measure_euclid_dist(44,48) + self.measure_euclid_dist(45,47)) / (2*self.measure_euclid_dist(43,46))
        self.df_measure["ear"]   = (self.df_measure["ear_r"] +self.df_measure["ear_l"])/2

    def measure_eyebrows_nose(self):
        print(self.df_landmarks)
        self.df_measure["eyebrowns_nose_l"] = (self.measure_euclid_dist(20,32))
        self.df_measure["eyebrowns_nose_r"] = (self.measure_euclid_dist(25,36))
        self.df_measure["eyebrowns_nose"]   = (self.df_measure["eyebrowns_nose_r"] + self.df_measure["eyebrowns_nose_r"])/ 2
        print(self.df_measure)

    def plot_measure(self, measure):
        discontinuities_frame  = self.find_discontinuities()
        print(discontinuities_frame[0][1])
        for index in discontinuities_frame:
            plt.plot(self.df_measure[self.df_measure["frame"].between(index[0],index[1])]["frame"], self.df_measure[self.df_measure["frame"].between(index[0],index[1])][measure])
        plt.xlabel("frame")
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


ad = AnalyseData("data/data_out/IRBA_extrait_1.mp4landmarks.csv")
ad.find_discontinuities()
ad.measure_ear()
ad.measure_yawning_frequency()
ad.plot_measure("mouth")
