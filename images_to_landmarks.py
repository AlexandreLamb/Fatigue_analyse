from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import numpy as np
import pandas as pd

landmarks_eyes_left = np.arange(36,42)
landmarks_eyes_rigth = np.arange(42,48)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
				help="chemin d'accès à l'image d'entrée")
args = vars(ap.parse_args())
# initialiser le détecteur de visage de dlib (basé sur HOG)
detector = dlib.get_frontal_face_detector()
# répertoire de modèles pré-formés
predictor = dlib.shape_predictor("data/shape_predictor/shape_predictor_68_face_landmarks.dat")

for index in range(0,5548):
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
		# convertir le rectangle de Dlib en un cadre de sélection de style OpenCV
		# dessiner le cadre de sélection
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		euclid_dist_38_42_l = np.linalg.norm(shape[landmarks_eyes_left[1]]-shape[landmarks_eyes_left[5]])
		euclid_dist_39_41_l = np.linalg.norm(shape[landmarks_eyes_left[2]]-shape[landmarks_eyes_left[4]])
		euclid_dist_37_40_l = np.linalg.norm(shape[landmarks_eyes_left[0]]-shape[landmarks_eyes_left[3]])

		euclid_dist_44_48_r = np.linalg.norm(shape[landmarks_eyes_rigth[1]]-shape[landmarks_eyes_rigth[5]])
		euclid_dist_45_47_r = np.linalg.norm(shape[landmarks_eyes_rigth[2]]-shape[landmarks_eyes_rigth[4]])
		euclid_dist_43_46_r = np.linalg.norm(shape[landmarks_eyes_rigth[0]]-shape[landmarks_eyes_rigth[3]])

		df = pd.DataFrame({"id" :["frame"+str(index)] ,
								"euclid_dist_38_42_l" : [euclid_dist_38_42_l],
								"euclid_dist_39_41_l" : [euclid_dist_39_41_l],
								"euclid_dist_37_40_l" : [euclid_dist_37_40_l] ,
								"mean_38_39_41_42_l" : [(euclid_dist_38_42_l+euclid_dist_39_41_l)/2],
								"euclid_dist_44_48_r" : [euclid_dist_44_48_r],
								"euclid_dist_45_47_r" : [euclid_dist_45_47_r],
								"euclid_dist_43_46_r" : [euclid_dist_43_46_r],
								"mean_44_45_47_48_r" : [(euclid_dist_44_48_r+euclid_dist_45_47_r)/2],
							})
		df.to_csv("data.csv",header=False,mode="a")
		print("Process "+str(index)+" on 500")
