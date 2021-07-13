import dlib
import cv2
from imutils import face_utils
import imutils
import numpy as np

SHAPE_PREDICTOR_PATH ="data/data_in/models/shape_predictor_68_face_landmarks.dat"

class FaceRecognitionHOG:
    """Class who implement the algorithm of the face recognition with Hog classifier based on dlib implementation and place landamrks with dlib predictor
    """
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        
    def place_landmarks(self, image, count, img_size = None):
        """Main function for place landmarks on a image and retrun the landamrks coordinates (x,y)

        :param image: Image (numpy array) to place landamrks 
        :type image: Numpy Array (int)
        :param count: Index of wich frame is analysis
        :type count: int
        :param img_size: If needed a size to resize the image, defaults to None
        :type img_size: int , optional
        :return: Array of landamrks position (x,y)
        :rtype: Array (int)
        """
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
            marks = self.predictor(gray, rects[0])
            marks = face_utils.shape_to_np(marks)
        else :
            marks = np.array([])
        return marks.ravel()
        
    def place_landmarks_from_mtcnn(self, image, count, img_size = None):
        """Function whose use when the toher face dector algorithme MTCNN can't find face into the image

        :param image: Image (numpy array) to place landamrks 
        :type image: Numpy Array (int)
        :param count: Index of wich frame is analysis
        :type count: int
        :param img_size: If needed a size to resize the image, defaults to None
        :type img_size: int , optional
        :return: Array of landamrks position (x,y)
        :rtype: Array (int)
        """
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
            marks = self.predictor(gray, rects[0])
            marks = face_utils.shape_to_np(marks)
        else :
            marks = np.array([])
        return marks