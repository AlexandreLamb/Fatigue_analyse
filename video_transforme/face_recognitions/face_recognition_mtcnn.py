from mtcnn import MTCNN
import cv2
import dlib
import pandas as pd
from imutils import face_utils
from . import FaceRecognitionHOG
import numpy as np

SHAPE_PREDICTOR_PATH ="data/data_in/models/shape_predictor_68_face_landmarks.dat"

class FaceRecognitionMtcnn:
    """Class who implement the algorithm of the face recognition with MTCNN classifier based on https://github.com/ipazc/mtcnn
    """
    def __init__(self):
        self.detector = MTCNN()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    def place_landmarks(self, img, count):
        """Main function for place the landmarks into an image and return the coodroniates (x,y), if  no face detected on image, use HOG to detect face and place landamrks with dlib predictor

        :param img: Image (numpy array) to place landamrks 
        :type img: Numpy Array (int)
        :param count: Index of wich frame is analysis
        :type count: int
        :return: Array of landamrks position (x,y)
        :rtype: Array (int)
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("data/data_temp/face_in_gray_before_porcessing"+str(count)+".jpg",img_gray)
        face_informations = self.detector.detect_faces(img_gray)
        if len(face_informations) == 1:
            box = face_informations[0].get("box")
            rectangle_box = dlib.rectangle(box[0]+box[2], box[1], box[0], box[1]+box[3])
            marks = self.predictor(img_gray, rectangle_box)
            marks = face_utils.shape_to_np(marks)
        elif len(face_informations) > 1:
            conf_max  = max([obj.get("confidence") for obj in face_informations])
            index_conf_max = [index for index , obj in enumerate(face_informations) if obj.get("confidence") == conf_max]
            box = face_informations[index_conf_max[0]].get("box")
            rectangle_box = dlib.rectangle(box[0]+box[2], box[1], box[0], box[1]+box[3])
            marks = self.predictor(img_gray, rectangle_box)
            marks = face_utils.shape_to_np(marks)
        else :
            hog_recogniton = FaceRecognitionHOG()
            marks = hog_recogniton.place_landmarks_from_mtcnn(img,count)
        return marks.ravel()
           