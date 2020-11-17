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


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=False,
				help="access path to the images folder to analyze")
args = vars(ap.parse_args())
# initialiser le détecteur de visage de dlib (basé sur HOG)
detector = dlib.get_frontal_face_detector()
# répertoire de modèles pré-formés
predictor = dlib.shape_predictor("data/shape_predictor/shape_predictor_68_face_landmarks.dat")


df_landmarks = pd.DataFrame(columns=make_landmarks_header())
number_images = len(os.listdir(args["folder"]))


for index in range(0,number_images):
	image = cv2.imread(str(args["folder"]) + "/frame"+str(index)+".jpg")
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
