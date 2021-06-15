from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logger import logging
from video_transforme.face_recognitions import FaceRecognitionHOG, FaceRecognitionMtcnn
from utils import make_landmarks_header, parse_path_to_name
from dotenv import load_dotenv
load_dotenv("env_file/.env_path")
from database_connector import read_remote_df, save_remote_df, list_dir_remote
PATH_TO_LANDMARKS_DESFAM_F_5_MIN= os.environ.get("PATH_TO_LANDMARKS_DESFAM_F_5_MIN")
PATH_TO_LANDMARKS_DESFAM_F_FULL= os.environ.get("PATH_TO_LANDMARKS_DESFAM_F_FULL")
PATH_TO_LANDMARKS_DESFAM_F= os.environ.get("PATH_TO_LANDMARKS_DESFAM_F")

SHAPE_PREDICTOR_PATH ="data/data_in/models/shape_predictor_68_face_landmarks.dat"

class VideoToLandmarks:
    def __init__(self, path):
        self.df_landmarks = pd.DataFrame(columns = make_landmarks_header()).rename(index={0 : "frame"})
        self.df_videos_infos = pd.DataFrame(columns = ["video_name","fps","frame_count"])
        self.path = path
        self.video_infos_path = os.path.join(PATH_TO_LANDMARKS_DESFAM_F,"videos_infos.csv")
        self.videos = []
        self.face_recognitions = {
                                    "hog" : FaceRecognitionHOG(),
                                    "mtcnn" : FaceRecognitionMtcnn()
                                   # "cnn" : FaceRecognitionCNN()
        }

    def check_if_video_already_exists(self, name):
        if os.path.exists(self.video_infos_path):
            video_infos = read_remote_df(self.video_infos_path)
            if name in list(video_infos["video_name"]):
                return False
            else:
                return True
        else:
            return True
    ##TODO: save filepaht in video infos csv
    ##TODO: Blindage video infos a tester 
    def load_data_video(self):
        logging.info("loading at path : "  + str(self.path))
        if(os.path.isdir(self.path)):
            for video_name in list_dir_remote(self.path):
                logging.info("loading video : " + video_name)
                cap = cv2.VideoCapture(os.path.join(self.path,video_name))
                video_name = parse_path_to_name(video_name)
                self.videos.append({
                    "video_name" : video_name,
                    "video" : cap
                })
                if self.check_if_video_already_exists(video_name):
                    self.df_videos_infos = self.df_videos_infos.append({
                                                                        'video_name' : video_name,
                                                                        'fps' : cap.get(cv2.CAP_PROP_FPS),
                                                                        'frame_count' : cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                                                        },
                                                                        ignore_index=True)
        else:
            logging.info("loading video : " + self.path.split("/")[-1])
            cap = cv2.VideoCapture(os.path.join(self.path))
            video_name = parse_path_to_name(self.path)
            self.videos.append({
                "video_name" : video_name,
                "video" : cap
            })
            if self.check_if_video_already_exists(video_name):
                self.df_videos_infos = self.df_videos_infos.append({
                                                                    'video_name' : video_name,
                                                                    'fps' : cap.get(cv2.CAP_PROP_FPS),
                                                                    'frame_count' : cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                                                    },
                                                                    ignore_index=True)
        if os.path.isfile(self.video_infos_path) :
            save_remote_df(self.video_infos_path, self.df_videos_infos, mode="a", header=False)
        else :
            save_remote_df(self.video_infos_path, self.df_videos_infos, mode="w")
            
        self.df_videos_infos =read_remote_df(self.video_infos_path)
    def save_landmarks_pics(self, marks, img, face_recognition_type, coun, video_name):
        marks_pair = list(zip(marks[::2],marks[1::2]))
        for mark in marks_pair:
            cv2.circle(img, (mark[0], mark[1]), 2, (0,255,0), -1, cv2.LINE_AA)
        cv2.imwrite("data/stage_data_out/landmarks_pics/"++"_image_"+str(face_recognition_type)+"_"+str(count)+".jpg", img)

    def progression_of_place_landmarks(self, count, video_name, frame_total_1= -1, frame_total_2 = None):
        if frame_total_1 == -1:
            frame_count = self.df_videos_infos[self.df_videos_infos["video_name"] == video_name]["frame_count"]
            os.system("clear")
            print(str(count)+ " on " + str(list(frame_count)[0]) + " frame analyse")
        else : 
            frame_count_1 = int(frame_total_1)
            if frame_total_2 == None:
                frame_count_2 = -1
            else : 
                frame_count_2 = int(frame_total_2)             
            os.system("clear")
            if count <= frame_count_1 : 
                print(str(count)+ " on " + str(frame_count_1) + " frame analyse")
            if count >= frame_count_1 and frame_count_2 != -1 :
                print(str(count)+ " on " + str(frame_count_2) + " frame analyse")

    def transoform_videos_to_landmarks(self, face_recognition_type):
        for video in self.videos:
            video_name = video.get("video_name")
            video_fps = list(self.df_videos_infos[self.df_videos_infos["video_name"] == video_name]["fps"])[0]
            frame_count = int(list(self.df_videos_infos[self.df_videos_infos["video_name"] == video_name]["frame_count"])[0])

            logging.info("Writing video : " + str(video_name))
            csv_path_name = os.path.join(PATH_TO_LANDMARKS_DESFAM_F_FULL,video_name+"_"+str(face_recognition_type)+"_all.csv")
            success, image = video.get("video").read()
            count = 0
            self.df_landmarks = pd.DataFrame(columns = make_landmarks_header()).rename(index={0 : "frame"})
            if os.path.isfile(csv_path_name) : 
                logging.info(os.path.isfile(csv_path_name)) 
                logging.info(csv_path_name) 
                self.df_landmarks = read_remote_df(csv_path_name, index_col="frame")
                count = len(self.df_landmarks)
            while success:
                success, img = video.get("video").read()
                if (self.df_landmarks.index == count).any() == False :
                    if success:
                        marks = self.face_recognitions.get(face_recognition_type).place_landmarks(img, count)
                        if len(marks) > 0:             
                            self.df_landmarks.loc[count] = marks
                            save_remote_df(csv_path_name, self.df_landmarks, header=True, mode="w", index_label="frame")
                        else:
                            logging.info("No face detect on image "+str(count))
                        self.progression_of_place_landmarks(count, video_name)
                        count += 1
            save_remote_df(csv_path_name, self.df_landmarks, header=True, mode="w")
    def transoform_videos_with_sec_to_landmarks(self, face_recognition_type, sec):
        for video in self.videos:
            video_name = video.get("video_name")
            video_fps = list(self.df_videos_infos[self.df_videos_infos["video_name"] == video_name]["fps"])[0]
            frame_count = int(list(self.df_videos_infos[self.df_videos_infos["video_name"] == video_name]["frame_count"])[0])
            
            logging.info("Writing video : " + str(video_name))
            csv_path_name = os.path.join(PATH_TO_LANDMARKS_DESFAM_F_5_MIN,video_name+"_"+str(face_recognition_type)+"_"+str(sec)+".csv")
            success, image = video.get("video").read()
            count = 0
            self.df_landmarks = pd.DataFrame(columns = make_landmarks_header()).rename(index={0 : "frame"})
            if os.path.isfile(csv_path_name) : 
                logging.info(os.path.isfile(csv_path_name)) 
                logging.info(csv_path_name) 
                self.df_landmarks = read_remote_df(csv_path_name, index_col="frame")
                count = len(self.df_landmarks)
            while success:
                if count in range(0,int(sec*video_fps)+1) or count in range(frame_count-int(sec*video_fps), frame_count+1) :
                    success, img = video.get("video").read()
                    if (self.df_landmarks.index == count).any() == False :
                        if success:
                            marks = self.face_recognitions.get(face_recognition_type).place_landmarks(img, count)
                            if len(marks) > 0:         
                                self.df_landmarks.loc[count] = marks
                                save_remote_df(csv_path_name, self.df_landmarks, header=True, mode="w", index_label="frame")
                            else:
                                logging.info("No face detect on image "+str(count))
                    else : 
                        logging.info("There is already landmarks place on image "+str(count))
                self.progression_of_place_landmarks(count, video_name, sec*video_fps, frame_count)
                if count == frame_count:
                    success = False
                count += 1
                if count == int(sec*video_fps) : 
                    count = frame_count-int(sec*video_fps)
            save_remote_df(csv_path_name, self.df_landmarks, header=True, mode="w")

             
    def load_and_transform(self, detector):
        self.load_data_video()
        self.transoform_videos_to_landmarks(detector)
        
    def load_and_transform_with_sec(self, detector, minutes):
        self.load_data_video()
        self.transoform_videos_with_sec_to_landmarks(detector, minutes*60)

    def load_and_transform_mtcnn(self):
        cap = cv2.VideoCapture(self.path)
        success, img = cap.read()     
         
        mtcnn = FaceRecognitionMtcnn()
        count = 0
        while success:  
            marks = mtcnn.place_landmarks(img, count)
            marks_pair = list(zip(marks[::2],marks[1::2]))
            np.savetxt("data/stage_data_out/marks_pair",marks_pair)
            for mark in marks_pair:
                cv2.circle(img, (mark[0], mark[1]), 2, (0,255,0), -1, cv2.LINE_AA)
            cv2.imwrite("data/stage_data_out/landmarks_pics_mtcnn/image_"+str(count)+".jpg", img)

            success, img = cap.read()
            count = count + 1


