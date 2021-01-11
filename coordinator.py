import os
from video_to_landmarks import VideoToLandmarks



SHAPE_PREDICTOR_PATH = "data/data_in/shape_predictor/shape_predictor_68_face_landmarks.dat"


class Coordinatore() :
    def __init__(self, path_provide) :
        self.videoTrasnformer = VideoToLandmarks(path_provide)

    def create_csv_landmarks(self):
        self.videoTrasnformer.load_and_transform()
