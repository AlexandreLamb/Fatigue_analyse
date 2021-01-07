from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import numpy as np
import pandas as pd
import os

"""
INPUT : Image Folder
OUTPUT : CSV file in data Directory
CSV file :
	- name : landmarks.csv
	- description : one row per frame / one row containes coordinates (landmarks_n_x, landmarks_n_y) of the frame

"""
def make_landmarks_header():
	csv_header = []
	for i in range(1,69):
		csv_header.append("landmarks_"+str(i)+"_x")
		csv_header.append("landmarks_"+str(i)+"_y")
	return csv_header


class ImagesToLandmarks:
	def __init__(self, shape_predictor_path, images_dir_path):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(shape_predictor_path)
		self.df_landmarks = pd.DataFrame(columns = make_landmarks_header())
		self.images_dir_path = images_dir_path
		#self.number_images = len(os.listdir(images_dir_path))
		#self.file_name = images_dir_path.split("/")[-1]

	def place_landmarks(self):
		for file_dir in self.images_dir_path:
			number_images = len(os.listdir(file_dir))
			file_name = file_dir.split("/")[-1].split(".")[0]
			file_path_name = "data/data_in/images/"+file_name+"_landmarks.csv"
			if(os.path.isdir(file_path_name) == False):
				for index in range(0,number_images):
					image = cv2.imread(file_dir+"/frame"+str(index)+".jpg")
					image = imutils.resize(image, width=600)
					gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					# détecter les visages
					rects = self.detector(gray, 1)
					# Pour chaque visage détecté, recherchez le repère.
					for rect in rects:
						# déterminer les repères du visage for the face region, then
						# convertir le repère du visage (x, y) en un array NumPy
						shape = self.predictor(gray, rect)
						shape = face_utils.shape_to_np(shape)
						self.df_landmarks.loc[index]= shape.ravel()
						print("Process "+str(index)+" on "+str(number_images))
				self.df_landmarks.to_csv(file_path_name,header=True,mode="w")


"""
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=False,
				help="access path to the images folder to analyze")
args = vars(ap.parse_args())
# initialiser le détecteur de visage de dlib (basé sur HOG)
detector = dlib.get_frontal_face_detector()
# répertoire de modèles pré-formés
predictor = dlib.shape_predictor("data/shape_predictor/shape_predictor_68_face_landmarks.dat")


df_landmarks = pd.DataFrame(columns=make_landmarks_header())
number_images = len(os.listdir("data/images"))
for index in range(0,20):
	image = cv2.imread("data/images/frame"+str(index)+".jpg")
	image = imutils.resize(image, width=600)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# détecter les visages
	rects = detector(gray, 1)

	# Pour chaque visage détecté, recherchez le repère.
	for rect in rects:
		# déterminer les repères du visage for the face region, then
		# convertir le repère du visage (x, y) en un array NumPy
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		df_landmarks.loc[index]= shape.ravel()

		print("Process "+str(index)+" on "+str(number_images))
df_landmarks.to_csv("data/landmarks.csv",header=True,mode="w")
"""
