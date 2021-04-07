import dlib
import cv2
from logger import logging
from imutils import face_utils
import imutils

SHAPE_PREDICTOR_PATH ="data/data_in/models/shape_predictor_68_face_landmarks.dat"

class FaceRecognitionHOG:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    def place_landmarks(self, image, count, img_size = None):
        logging.info("place landmarks on image " + str(count))
        if img_size != None:
            image = imutils.resize(image, width=img_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # détecter les visages
        rects = self.detector(gray, 1)
        #print(rects)
        # Pour chaque visage détecté, recherchez le repère.
        if len(rects) == 1 :
            marks = self.predictor(gray, rects[0])
            marks = face_utils.shape_to_np(marks)            
        elif len(rects) >= 1:
            logging.info("Detect more than 1 face on img number " +str(count) + " get default first face detect")
            marks = self.predictor(gray, rects[0])
            marks = face_utils.shape_to_np(marks)
        else :
            marks = np.array([])
        return marks.ravel()
        
    def place_landmarks_from_mtcnn(self, image, count, img_size = None):
        logging.info("place landmarks on image " + str(count))
        if img_size != None:
            image = imutils.resize(image, width=img_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # détecter les visages
        rects = self.detector(gray, 1)
        #print(rects)
        # Pour chaque visage détecté, recherchez le repère.
        if len(rects) == 1 :
            marks = self.predictor(gray, rects[0])
            marks = face_utils.shape_to_np(marks)            
        elif len(rects) >= 1:
            logging.info("Detect more than 1 face on img number " +str(count) + " get default first face detect")
            marks = self.predictor(gray, rects[0])
            marks = face_utils.shape_to_np(marks)
        else :
            marks = np.array([])
        return marks