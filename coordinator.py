import os
from video_to_frame import VideoUtils
from images_to_landmarks import ImagesToLandmarks



DATA_DIR_PATH = "data/data_in/videos"
SHAPE_PREDICTOR_PATH = "data/data_in/shape_predictor/shape_predictor_68_face_landmarks.dat"
videoUtils = VideoUtils()

videoUtils.load_data_video_from_dir(DATA_DIR_PATH)

class Coordinatore() :
    def __init__(self, path_provide) :
        self.path_provide = path_provide

    def create_csv_landmarks(self):
        videoUtils = VideoUtils()

        if(os.path.isfile(self.path_provide)):
            videoUtils.load_data_video_from_file(self.path_provide)
        else:
            videoUtils.load_data_video_from_dir(self.path_provide)
        file_path_tab = videoUtils.transoform_video_to_frames()
        imagesToLandmarks = ImagesToLandmarks(SHAPE_PREDICTOR_PATH,file_path_tab)
        imagesToLandmarks.place_landmarks()
