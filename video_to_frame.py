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

	def transoform_video_to_frames(self):
		file_path_tab = []
		for video in self.videos:
			print("Writing video : " + str(video.get("video_name")))
			file_path_name = "data/data_in/images/"+video.get("video_name")
			file_path_tab.append(file_path_name)
			if(os.path.isdir(file_path_name) == False):
				success, image = video.get("video").read()
				count = 0;
				os.mkdir("data/data_in/images/"+video.get("video_name"))
				while success:
				  success, image = video.get("video").read()
				  if success:
				   	cv2.imwrite("data/data_in/images/"+video.get("video_name")+"/frame"+str(count)+".jpg", image)
				  if cv2.waitKey(10) == 27:
				      break
				  count += 1
		return file_path_tab
