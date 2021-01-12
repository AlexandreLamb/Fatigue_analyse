from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import numpy as np
import pandas as pd
import os

SHAPE_PREDICTOR_PATH = "data/data_in/shape_predictor/shape_predictor_68_face_landmarks.dat"

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

class VideoToLandmarks:
    def __init__(self, path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        self.df_landmarks = pd.DataFrame(columns = make_landmarks_header())
        self.df_videos_infos = pd.DataFrame(columns = ["video_name","fps"])
        self.path = path
        self.videos = []
        self.video_infos_path = "data/data_out/videos_infos.csv"

    def load_data_video(self):
        print("loading at path : "  + self.path)
        if(os.path.isdir(self.path)):
            for video_name in os.listdir(self.path):
                print("loading video : " + video_name)
                cap = cv2.VideoCapture(os.path.join(self.path,video_name))
                video_name = parse_path_to_name(video_name)
                self.videos.append({
                    "video_name" : video_name,
                    "video" : cap
                })
                self.df_videos_infos = self.df_videos_infos.append({'video_name' : video_name, 'fps' : cap.get(cv2.CAP_PROP_FPS)}, ignore_index=True)
        else:
            print("loading video : " + self.path.split("/")[-1])
            cap = cv2.VideoCapture(os.path.join(self.path))
            video_name = parse_path_to_name(self.path)
            self.videos.append({
                "video_name" : video_name,
                "video" : cap
            })
            self.df_videos_infos = self.df_videos_infos.append({'video_name' : video_name, 'fps' : cap.get(cv2.CAP_PROP_FPS)}, ignore_index=True)
        if os.path.isfile(self.video_infos_path) :
            self.df_videos_infos.to_csv(self.video_infos_path, mode="a", header=False)
        else :
            self.df_videos_infos.to_csv(self.video_infos_path, mode="w")


    def place_landmarks(self, image, count):
        print("place landmarks on image " + str(count))
        image = imutils.resize(image, width=600)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # détecter les visages
        rects = self.detector(gray, 1)
        # Pour chaque visage détecté, recherchez le repère.
        for rect in rects:
            # déterminer les repères du visage for the face region, then
            # convertir le repère du visage (x, y) en un array NumPy
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            self.df_landmarks.loc[count]= shape.ravel()

    def transoform_videos_to_landmarks(self):
        for video in self.videos:
            print("Writing video : " + str(video.get("video_name")))
            csv_path_name = "data/data_out/"+video.get("video_name")+".csv"
            success, image = video.get("video").read()
            count = 0;
            while success:
                success, image = video.get("video").read()
                if success:
                    self.place_landmarks(image,count)
                    count += 1
            self.df_landmarks.to_csv(csv_path_name,header=True,mode="w")

    def load_and_transform(self):
        self.load_data_video()
        self.transoform_videos_to_landmarks()
