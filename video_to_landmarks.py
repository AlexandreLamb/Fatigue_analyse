from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from log import logging

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
        self.model_face_detector= "data/data_in/models/res10_300x300_ssd_iter_140000.caffemodel"
        self.config_file = "data/data_in/models/deploy.prototxt"
        self.model_landmarks = "data/data_in/models/pose_model"
        self.face_detector =  cv2.dnn.readNetFromCaffe(self.config_file, self.model_face_detector)
        self.landmark_model = keras.models.load_model(self.model_landmarks)

    def face_detection_dnn(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        res = self.face_detector.forward()
        faces = []
        for i in range(res.shape[2]):
            confidence = res[0, 0, i, 2]
            if confidence > 0.5:
                box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                faces.append([x, y, x1, y1])
        return faces

    def move_box(self, box, offset):
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    def get_square_box(self, box):
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:                   # Already a square.
            return box
        elif diff > 0:                  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:                           # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    def detect_landmarks_dnn(self, img, face):
        offset_y = int(abs((face[3] - face[1]) * 0.1))
        box_moved = self.move_box(face, [0, offset_y])
        facebox = self.get_square_box(box_moved)

        face_img = img[facebox[1]: facebox[3],
                         facebox[0]: facebox[2]]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # # Actual detection.
        predictions = self.landmark_model.signatures["predict"](tf.constant([face_img], dtype=tf.uint8))

        # Convert predictions to landmarks.
        marks = np.array(predictions['output']).flatten()[:136]
        marks = np.reshape(marks, (-1, 2))
        marks *= (facebox[2] - facebox[0])
        marks[:, 0] += facebox[0]
        marks[:, 1] += facebox[1]
        marks = marks.astype(np.uint)

        return marks

    def draw_marks(image, marks, color=(0, 255, 0)):
        for mark in marks:
            cv2.circle(np.array(image), (mark[0], mark[1]), 2, color, -1, cv2.LINE_AA)

    def place_landmarks_dnn(self, img, count) :
        print("place landmarks on image "+str(count))
        rects = self.face_detection_dnn(img)
        for rect in rects:
            marks = self.detect_landmarks_dnn(img,rect)
            self.df_landmarks.loc[count]= marks.ravel()

    def load_data_video(self):
        print("loading at path : "  + str(self.path))
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

    def place_landmarks_hog(self, image, count):
        print("place landmarks on image " + str(count))
        image = imutils.resize(image, width=600)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # détecter les visages
        rects = self.detector(gray, 1)
        #print(rects)
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
                success, img = video.get("video").read()
                if success:
                    self.place_landmarks_dnn(img, count)
                    count += 1
            self.df_landmarks.to_csv(csv_path_name,header=True,mode="w")

    def load_and_transform(self):
        logging.info("test")
        self.load_data_video()
        #self.transoform_videos_to_landmarks()


class FaceRecognitionHOG:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    def place_landmarks_hog(self, image, count):
        logging.info("place landmarks on image " + str(count))
        image = imutils.resize(image, width=600)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # détecter les visages
        rects = self.detector(gray, 1)
        #print(rects)
        # Pour chaque visage détecté, recherchez le repère.
        for rect in rects:
            # déterminer les repères du visage for the face region, then
            # convertir le repère du visage (x, y) en un array NumPy
            marks = self.predictor(gray, rect)
            marks = face_utils.shape_to_np(marks)
            return marks

class FaceRecognitionCNN:
    def __init__(self):
        self.model_face_detector= "data/data_in/models/res10_300x300_ssd_iter_140000.caffemodel"
        self.config_file = "data/data_in/models/deploy.prototxt"
        self.model_landmarks = "data/data_in/models/pose_model"
        self.face_detector =  cv2.dnn.readNetFromCaffe(self.config_file, self.model_face_detector)
        self.landmark_model = keras.models.load_model(self.model_landmarks)

    def face_detection_dnn(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        res = self.face_detector.forward()
        faces = []
        for i in range(res.shape[2]):
            confidence = res[0, 0, i, 2]
            if confidence > 0.5:
                box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                faces.append([x, y, x1, y1])
        return faces

    def move_box(self, box, offset):
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    def get_square_box(self, box):
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y
        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)
        if diff == 0:                   # Already a square.
            return box
        elif diff > 0:                  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:                           # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1
        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'
        return [left_x, top_y, right_x, bottom_y]

    def draw_marks(image, marks, color=(0, 255, 0)):
        for mark in marks:
            cv2.circle(np.array(image), (mark[0], mark[1]), 2, color, -1, cv2.LINE_AA)

    def detect_landmarks_dnn(self, img, face):
        offset_y = int(abs((face[3] - face[1]) * 0.1))
        box_moved = self.move_box(face, [0, offset_y])
        facebox = self.get_square_box(box_moved)
        face_img = img[facebox[1]: facebox[3], facebox[0]: facebox[2]]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # # Actual detection.
        predictions = self.landmark_model.signatures["predict"](tf.constant([face_img], dtype=tf.uint8))
        # Convert predictions to landmarks.
        marks = np.array(predictions['output']).flatten()[:136]
        marks = np.reshape(marks, (-1, 2))
        marks *= (facebox[2] - facebox[0])
        marks[:, 0] += facebox[0]
        marks[:, 1] += facebox[1]
        marks = marks.astype(np.uint)
        return marks

    def place_landmarks_dnn(self, img, count) :
        logging.info("place landmarks on image "+str(count))
        rects = self.face_detection_dnn(img)
        if len(rects) == 1 :
            marks = self.detect_landmarks_dnn(img,rects[0])
            return marks
        else:
            logging.info("Detect more than 1 face on img number " +str(count))
            marks = self.detect_landmarks_dnn(img,rects[0])
            return marks


vl = VideoToLandmarks("data/data_in/videos/DESFAM_Semaine 2-Vendredi_Go-NoGo_H69.mp4")
vl.load_and_transform()
