from mtcnn import MTCNN
import cv2
import dlib
import pandas as pd
from imutils import face_utils
from face_recognitions import FaceRecognitionHOG


SHAPE_PREDICTOR_PATH ="data/data_in/models/shape_predictor_68_face_landmarks.dat"

class FaceRecognitionMtcnn:
    def __init__(self):
        self.detector = MTCNN()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    def place_landmarks(self, img, count):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_informations = self.detector.detect_faces(img_gray)
        if len(face_informations) == 1:
            logging.info("Detect 1 face on img number " +str(count))
            box = face_informations[0].get("box")
            rectangle_box = dlib.rectangle(box[0]+box[2], box[1], box[0], box[1]+box[3])
            marks = self.predictor(img_gray, rectangle_box)
            marks = face_utils.shape_to_np(marks)
        elif len(face_informations) > 1:
            logging.info("Detect more than 1 face on img number " +str(count) + " get default first face detect")
        
        else :
            logging.info("No face on img number " +str(count) + " try with HOG classifier")
            hog_recogniton = FaceRecognitionHOG()
            marks = hog_recogniton.place_landmarks_from_mtcnn(img,count)
            if len(marks) == 0:
                logging.info("No face on img number " +str(count)+ " even with HOG")

        return marks.ravel()
           