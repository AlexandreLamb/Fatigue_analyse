import numpy as np
import cv2
import os
import random
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *


def video_cut_movie(path_to_video):    
    video_name = path_to_video.split("/")[-1].split(".")[0]
    subject_info = video_name.split("_")
    random_sequence_order = [chr(el) for el in random.sample(range(65,69),4)]
    df_sequence = pd.DataFrame(columns=["subject", "day", "0 min", "15 min", "30 min", "45 min"]).set_index(["subject", "day"])
    df_sequence.loc[(subject_info[-2], subject_info[-1]),["0 min", "15 min", "30 min", "45 min"]] = random_sequence_order
    video_infos = VideoFileClip(path_to_video)
    
    windows_to_cut_frame = [0, int(15*60) , 30*60, video_infos.duration-10]
    
    time_to_cut = 10
    for index, window in enumerate(windows_to_cut_frame):
        video = VideoFileClip(path_to_video)
        video = video.subclip(window, window + time_to_cut)
        create_folder_video(video_name)    
        #video.ipython_display()
        video.write_videofile("data/stage_data_out/image_for_irba/"+video_name+"/"+video_name+"_"+random_sequence_order[index]+".mp4")
        video.close()
        #ffmpeg_extract_subclip(path_to_video, window, window + time_to_cut , targetname="data/stage_data_out/image_for_irba/"+video_name+"_"+random_sequence_order[index]+".avi")
        print(video_name+"_"+random_sequence_order[index] + " is save")
    save_csv_sequences_order(df_sequence)
    
def video_cut(path_to_video):
    
    video_infos = cv2.VideoCapture(path_to_video)
    fps = video_infos.get(cv2.CAP_PROP_FPS)
    frame_count =video_infos.get(cv2.CAP_PROP_FRAME_COUNT)
    time_to_cut_sec = 23
    number_frame_to_cut = int(time_to_cut_sec  * fps)
    windows_to_cut_frame = [0, int(15*60*fps) , int(30*60*fps), int(frame_count - number_frame_to_cut) ]
    video_name = path_to_video.split("/")[-1].split(".")[0]
    subject_info = video_name.split("_")
    
    random_sequence_order = [chr(el) for el in random.sample(range(65,69),4)]
    df_sequence = pd.DataFrame(columns=["subject", "day", "0 min", "15 min", "30 min", "45 min"]).set_index(["subject", "day"])
    df_sequence.loc[(subject_info[-2], subject_info[-1]),["0 min", "15 min", "30 min", "45 min"]] = random_sequence_order

    for index, windows_to_cut in enumerate(windows_to_cut_frame):
        
        img_array = read_video_sequence (path_to_video, number_frame_to_cut, windows_to_cut) 
        if len(img_array) > 0 :
            create_folder_video(video_name)
            write_save_video(img_array, video_name, random_sequence_order[index])
            print(video_name+"_"+str(random_sequence_order[index]) + " is save")
            save_csv_sequences_order(df_sequence)
        else :
            print("error, img array Null : " + img_array)   
    
def read_video_sequence(path_to_video, number_frame_to_cut,windows_to_cut):
    cap = cv2.VideoCapture(path_to_video)
    img_array = []
    for frame in range(number_frame_to_cut+1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame+windows_to_cut)
            #cap.set(cv2.CAP_PROP_POS_FRAMES,(frame+windows_to_cut)/frame_count)
            _, image = cap.read()
            position = cap.get(cv2.CAP_PROP_POS_FRAMES)
            img_array.append(image)
    cap.release()
    return img_array

def write_save_video(img_array,video_name,random_sequence_order):
    height, width, layers = img_array[0].shape
    size = (width,height)
    out = cv2.VideoWriter("data/stage_data_out/image_for_irba/"+video_name+"/"+video_name+"_"+str(random_sequence_order)+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def create_folder_video(video_name):
    path_folder_to_save = "data/stage_data_out/image_for_irba/"+video_name
    if os.path.exists(path_folder_to_save) == False:
        os.makedirs(path_folder_to_save)
        
def save_csv_sequences_order(df_sequence):
    if os.path.isfile("data/stage_data_out/image_for_irba/sequence_order.csv") == False:
        df_sequence.to_csv("data/stage_data_out/image_for_irba/sequence_order.csv",mode="w", header=True ) 
    else : df_sequence.to_csv("data/stage_data_out/image_for_irba/sequence_order.csv",mode="a", header=False )

def get_file_path(subject_list):
    subject_list_lundi = [ "DESFAM_F_"+subject+"_LUNDI.avi" for subject in subject_list ]
    subject_list_vendredi = [ "DESFAM_F_"+subject+"_VENDREDI.avi" for subject in subject_list ]
    file_path = [PATH_TO_HDD_VIDEO_FOLDER_1+video_path for video_path in subject_list_vendredi + subject_list_lundi if os.path.isfile(PATH_TO_HDD_VIDEO_FOLDER_1+video_path) == True]
    file_path = file_path + [PATH_TO_HDD_VIDEO_FOLDER_2+video_path for video_path in subject_list_vendredi + subject_list_lundi if os.path.isfile(PATH_TO_HDD_VIDEO_FOLDER_2+video_path) == True]
    return file_path
   
def convert_csv_to_xlsx_save():
    df_sequence = pd.read_csv("data/stage_data_out/image_for_irba/sequence_order.csv", index_col=["subject","day"]) 
    df_sequence.sort_index().to_excel("data/stage_data_out/image_for_irba/sequence_order.xlsx")

 

PATH_TO_HDD_VIDEO_FOLDER_1 = "/mnt/feb02e35-bf58-4dba-aec4-589661cff1a5/data/OneDrive/IRBA/Video_IRBA_40_min/"
PATH_TO_HDD_VIDEO_FOLDER_2 = "/mnt/7b914d1c-f145-4023-9f2f-2cd288d7db76/DESFAM-F/"
subject_list = ["H90", "H91", "H95", "H98", "H103", "H105"]
file_path = get_file_path(subject_list)

for path in file_path:
    video_cut_movie(path)
convert_csv_to_xlsx_save()

##TODO: why is so long
