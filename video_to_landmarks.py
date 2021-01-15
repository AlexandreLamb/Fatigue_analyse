from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import numpy as np
import pandas as pd
import os
from log import logging
from face_recognitions import FaceRecognitionHOG, FaceRecognitionCNN
from utils import make_landmarks_header, parse_path_to_name

SHAPE_PREDICTOR_PATH ="data/data_in/models/shape_predictor_68_face_landmarks.dat"

class VideoToLandmarks:
    def __init__(self, path):
        self.df_landmarks = pd.DataFrame(columns = make_landmarks_header())
        self.df_videos_infos = pd.DataFrame(columns = ["video_name","fps"])
        self.path = path
        self.video_infos_path = "data/data_out/videos_infos.csv"
        self.videos = []
        self.face_recognitions = {
                                    "hog" : FaceRecognitionHOG(),
                                    "cnn" : FaceRecognitionCNN()
        }

    def check_if_video_already_exists(self, name):
        if os.path.exists(self.video_infos_path):
            video_infos = pd.read_csv(self.video_infos_path)
            if name in video_infos["video_name"]:
                return True
            else:
                return False
        else return False

    def load_data_video(self):
        logging.info("loading at path : "  + str(self.path))
        if(os.path.isdir(self.path)):
            for video_name in os.listdir(self.path):
                logging.info("loading video : " + video_name)
                cap = cv2.VideoCapture(os.path.join(self.path,video_name))
                video_name = parse_path_to_name(video_name)
                self.videos.append({
                    "video_name" : video_name,
                    "video" : cap
                })
                if self.check_if_video_already_exists(video_name):
                    self.df_videos_infos = self.df_videos_infos.append({'video_name' : video_name, 'fps' : cap.get(cv2.CAP_PROP_FPS)}, ignore_index=True)
        else:
            print("loading video : " + self.path.split("/")[-1])
            cap = cv2.VideoCapture(os.path.join(self.path))
            video_name = parse_path_to_name(self.path)
            self.videos.append({
                "video_name" : video_name,
                "video" : cap
            })
            if self.check_if_video_already_exists(video_name):
                self.df_videos_infos = self.df_videos_infos.append({'video_name' : video_name, 'fps' : cap.get(cv2.CAP_PROP_FPS)}, ignore_index=True)
        if os.path.isfile(self.video_infos_path) :
            self.df_videos_infos.to_csv(self.video_infos_path, mode="a", header=False)
        else :
            self.df_videos_infos.to_csv(self.video_infos_path, mode="w")

    def transoform_videos_to_landmarks(self, face_recognition_type):
        for video in self.videos:
            print("Writing video : " + str(video.get("video_name")))
            csv_path_name = "data/data_out/"+video.get("video_name")+".csv"
            success, image = video.get("video").read()
            count = 0;
            while success:
                success, img = video.get("video").read()
                if success:
                    marks = self.face_recognitions.get(face_recognition_type).place_landmarks(img, count)
                    print(marks.ravel())
                    if len(marks) > 0:
                        self.df_landmarks.loc[count] = marks
                    else:
                        logging.info("No face detect on image "+str(count))
                    count += 1
            self.df_landmarks.to_csv(csv_path_name,header=True,mode="w")

    def load_and_transform(self):
        self.load_data_video()
        self.transoform_videos_to_landmarks("hog")



vl = VideoToLandmarks("data/data_in/videos/DESFAM_Semaine 2-Vendredi_Go-NoGo_H69.mp4")
vl.load_and_transform()
