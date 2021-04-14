import cv2
import dlib
import os
import pandas as pd
import numpy as np

make_landmarks_pair  = lambda marks : list(zip(marks[::2],marks[1::2]))

def make_boxfrom_landmarks(row, threeshold_px = 20):
    left = int(row["landmarks_1_x"] - threeshold_px)
    top = int(np.mean([row["landmarks_20_y"] , row["landmarks_25_y"]]) - 2* threeshold_px)
    height = int(top - row["landmarks_9_y"] - threeshold_px)
    width = int(row["landmarks_17_x"] - left + threeshold_px)
    return {"top" : top, "left" : left, "height" : height, "width" : width} 
def read_image(img, count) : 
    cv2.imshow("frame_" +str(count), img)
    key = cv2.waitKey(0)
    sucess = True
    print(key)
    if key == 27:
        cv2.destroyAllWindows()
    elif key == 13 or key == 10:
        print("ok")
        cv2.destroyAllWindows()
    elif key == 115:
        print("save")
        cv2.destroyAllWindows()
        sucess = False
    return sucess

def place_landmarks_on_img(img, row):
    threeshold_px = 20
    left = int(row["landmarks_1_x"] - threeshold_px)
    top = int(np.mean([row["landmarks_20_y"] , row["landmarks_25_y"]]) - 2* threeshold_px)
    height = int(top - row["landmarks_9_y"] - threeshold_px)
    width = int(row["landmarks_17_x"] - left + threeshold_px)
    
    left_down_pts = [row["landmarks_1_x"],row["landmarks_9_y"]]
    print((left, top))
    print((left+width,top-height))
    cv2.rectangle(img, (left, top),(left+width,top-height),(0,255,0), 2)
    return img

def manual_classify_image(path_to_landmarks_csv, path_to_video) : 
    df_landmarks = pd.read_csv(path_to_landmarks_csv, index_col="frame")       
    cap = cv2.VideoCapture(path_to_video)
    sucess, image = cap.read()
    img_index = 0 
    while sucess:
        if img_index in df_landmarks.index :
            image = place_landmarks_on_img(image, df_landmarks.loc[img_index])
            sucess = read_image(image, img_index)
        if sucess:
            sucess, image = cap.read()
        img_index += 1
        
#manual_classify_image("data/stage_data_out/DESFAM_Semaine 2-Vendredi_PVT_H64_hog.csv","data/data_in/videos/DESFAM_Semaine 2-Vendredi_PVT_H64.mp4" )
def place_landmarks_on_img( img, landmarks):
        print("start_place")
        marks_pair = make_landmarks_pair(landmarks)
        for mark in marks_pair:
            cv2.circle(img, (mark[0], mark[1]), 2, (0,255,0), -1, cv2.LINE_AA)
        print("finish place")
        return img
img = cv2.imread("dlib_trainer/dataset/afw/1634816_1.jpg")
landmarks = [402, 520,
403, 544,
406, 572,
410 ,600,
419, 628,
435, 652,
451, 671,
471, 685,
497, 689,
521, 688,
545, 673,
568, 654]
place_landmarks_on_img(img,landmarks)
cv2.imshow('frame',img)
cv2.waitKey(0)