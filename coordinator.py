from video_to_frame import VideoUtils
from images_to_landmarks import ImagesToLandmarks

DATA_DIR_PATH = "data/data_in/videos"
videoUtils = VideoUtils()

videoUtils.load_data_video_from_dir(DATA_DIR_PATH)
