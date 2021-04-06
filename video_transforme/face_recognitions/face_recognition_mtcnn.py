from mtcnn import MTCNN
import cv2
import dlib
import pandas as pd
from imutils import face_utils

SHAPE_PREDICTOR_PATH ="data/data_in/models/shape_predictor_68_face_landmarks.dat"

class FaceRecognitionMtcnn:
    def __init__(self):
        self.detector = MTCNN()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    def place_landmarks(self, img, count):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_informations = self.detector.detect_faces(img_gray)
        box = face_informations[0].get("box")
        rectangle_box = dlib.rectangle(box[0]+box[2], box[1], box[0], box[1]+box[3])
        marks = self.predictor(img_gray, rectangle_box)
        marks = face_utils.shape_to_np(marks)
        return marks.ravel()
           