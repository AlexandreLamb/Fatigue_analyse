import cv2
import argparse
import os

class VideoUtils:
	def __init__(self):
		self.file_path = None
		self.dir_path = None
		self.data_dir = None
		self.videos = []

	def load_data_video_from_file(self, file_path):
		self.videos.append({
			"video_name" : file_path.split("/")[-1],
			"video" : cv2.VideoCapture(file_path)
		})

	def load_data_video_from_dir(self, dir_path):
		print(os.listdir(dir_path))
		for video_name in os.listdir(dir_path):
			print(os.path.join(dir_path,video_name))
			self.videos.append({
				"video_name" : video_name,
				"video" : cv2.VideoCapture(os.path.join(dir_path,video_name))
			})

	def transoform_video_to_frames(self,data_dir):
		for video in self.videos:
			print(video.get("video_name"))
"""

vidcap = cv2.VideoCapture(args["video"])

success,image = vidcap.read()


count = 0;
while success:
  success,image = vidcap.read()
  if success:
   	cv2.imwrite("data/images/frame%d.jpg" % count, image)
  if cv2.waitKey(10) == 27:
      break
  count += 1
"""
