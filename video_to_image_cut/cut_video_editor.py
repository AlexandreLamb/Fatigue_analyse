import numpy as np
import cv2
import os
import random
import pandas as pd
    
def video_cut(path_to_video):
    
    video_infos = cv2.VideoCapture(path_to_video)
    fps = video_infos.get(cv2.CAP_PROP_FPS)
    frame_count =video_infos.get(cv2.CAP_PROP_FRAME_COUNT)
    time_to_cut_sec = 15
    offset_sec = 120
    offset_frame = int(offset_sec * fps)
    number_frame_to_cut = int(time_to_cut_sec  * fps)
    windows_to_cut_frame = [offset_frame, int(15*60*fps) , int(30*60*fps), int(frame_count - number_frame_to_cut - offset_frame) ]
    windows_to_cut_frame_shift = [offset_frame+number_frame_to_cut, int(15*60*fps)+number_frame_to_cut , int(30*60*fps)+number_frame_to_cut, int(frame_count - number_frame_to_cut - offset_frame) +number_frame_to_cut ]
    video_name = path_to_video.split("/")[-1].split(".")[0]
    subject_info = video_name.split("_")
    print("------ "+video_name+ " ------")
    print("video total frame : " +str(frame_count))
    print("video fps : " +str(fps))
    print("windows to cut : " + str(list(zip(windows_to_cut_frame, windows_to_cut_frame_shift))))
    print("---------------------------")

    random_sequence_order = [chr(el) for el in random.sample(range(65,69),4)]
    df_sequence = pd.DataFrame(columns=["subject", "day", "0 min", "15 min", "30 min", "45 min"]).set_index(["subject", "day"])
    df_sequence.loc[(subject_info[-2], subject_info[-1]),["0 min", "15 min", "30 min", "45 min"]] = random_sequence_order
    create_folder_video(video_name)
    save_csv_sequences_order(df_sequence)
    windows_to_cut = list(zip(windows_to_cut_frame,windows_to_cut_frame_shift))
    
    cap = cv2.VideoCapture(path_to_video)
    img_array = []
    success, image = cap.read()
    cmp_frame  = 0
    cmp_sequence = 0
    while(success) : 
            if [el for el in windows_to_cut if el[0]<cmp_frame<el[1] ] :
                img_array.append(image)
                print(cmp_frame)
            if len([el[1] for el in windows_to_cut if el[0]<=cmp_frame<=el[1]]) > 0:
                if cmp_frame == [el[1] for el in windows_to_cut if el[0]<=cmp_frame<=el[1]][0]-1 : 
                    height, width, layers = img_array[0].shape
                    size = (width,height)
                    out = cv2.VideoWriter("data/stage_data_out/image_for_irba/"+video_name+"/"+video_name+"_"+str(random_sequence_order[cmp_sequence])+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
                    os.utime("data/stage_data_out/image_for_irba/"+video_name+"/"+video_name+"_"+str(random_sequence_order[cmp_sequence])+".avi",(0,0))
                    for i in range(len(img_array)):
                        out.write(img_array[i])
                    print(video_name+"_"+str(random_sequence_order[cmp_sequence]) + " is save")
                    out.release()
                    cmp_sequence = cmp_sequence + 1
                    img_array = []
            success, image = cap.read()
            cmp_frame = cmp_frame + 1
    cap.release()
                  
 

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
    if os.path.exists("data/stage_data_out/image_for_irba/sequence_order.csv"):
        df_sequence.to_csv("data/stage_data_out/image_for_irba/sequence_order.csv", mode="a", header=False)
    else:
        df_sequence.to_csv("data/stage_data_out/image_for_irba/sequence_order.csv", mode="w", header=True)
         

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
subject_list = ["H90", "H91", "H95", "H98", "H103"]
file_path = get_file_path(subject_list)

for path in file_path:
    video_cut(path)
convert_csv_to_xlsx_save()
##TODO: why is so long
