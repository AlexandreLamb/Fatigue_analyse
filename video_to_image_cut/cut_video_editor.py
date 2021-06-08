import numpy as np
import cv2
import os
import random
import pandas as pd
from pandas.core.algorithms import mode

def video_cut(path_to_video):
    cap = cv2.VideoCapture(path_to_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count =cap.get(cv2.CAP_PROP_FRAME_COUNT)
    time_to_cut_sec = 10
    number_frame_to_cut = int(time_to_cut_sec  * fps)
    windows_to_cut_frame = [0, int(15*60*fps) , 30*60*fps, frame_count - number_frame_to_cut ]
    video_name = path_to_video.split("/")[-1].split(".")[0]
    subject_info = video_name.split("_")
    random_sequence_order = random.sample(range(0,4),4)
    df_sequence = pd.DataFrame(columns=["subject", "day", "0 min", "15 min", "30 min", "45 min"]).set_index(["subject", "day"])
    df_sequence.loc[(subject_info[-2], subject_info[-1]),["0 min", "15 min", "30 min", "45 min"]] = random_sequence_order
    img_array = []
    for index, windows_to_cut in enumerate(windows_to_cut_frame):
        cap = cv2.VideoCapture(path_to_video)
        for frame in range(number_frame_to_cut+1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,(frame+windows_to_cut)/frame_count)
            _, image = cap.read()
            img_array.append(image)
            path_folder_to_save = "data/stage_data_out/image_for_irba/"+video_name+"/sequence_"+str(random_sequence_order[index])
            path_img_to_save = path_folder_to_save +"/" +video_name+'_frame_'+str(frame)+'.png'
            if os.path.exists(path_folder_to_save) == False:
                os.makedirs(path_folder_to_save)
            #cv2.imwrite(path_img_to_save,image)
        height, width, layers = img_array[0].shape
        size = (width,height)
        print(size)
        out = cv2.VideoWriter(video_name+"_"+str(random_sequence_order[index])+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(img_array)):
            print(i)
            out.write(img_array[i])
        out.release()
        img_array = []
    if os.path.isfile("data/stage_data_out/image_for_irba/sequence_order.csv") == False:
        df_sequence.to_csv("data/stage_data_out/image_for_irba/sequence_order.csv",mode="w", header=True ) 
    else : df_sequence.to_csv("data/stage_data_out/image_for_irba/sequence_order.csv",mode="a", header=False )
    
    
    
    

subject_list = ["H90", "H91", "H95", "H98", "H103", "H105"]
subject_list_lundi = [ "DESFAM_F_"+subject+"_LUNDI.avi" for subject in subject_list ]
subject_list_vendredi = [ "DESFAM_F_"+subject+"_VENDREDI.avi" for subject in subject_list ]
file_path = ["/mnt/feb02e35-bf58-4dba-aec4-589661cff1a5/data/OneDrive/IRBA/Video_IRBA_40_min/"+video_path for video_path in subject_list_vendredi + subject_list_lundi if os.path.isfile("/mnt/feb02e35-bf58-4dba-aec4-589661cff1a5/data/OneDrive/IRBA/Video_IRBA_40_min/"+video_path) == True]
file_path = file_path + ["/media/simeon/Data/DESFAM-F/"+video_path for video_path in subject_list_vendredi + subject_list_lundi if os.path.isfile("/media/simeon/Data/DESFAM-F/"+video_path) == True]
for path_to_cut in file_path:
    video_cut(path_to_cut)
"""df_sequence = pd.read_csv("data/stage_data_out/image_for_irba/sequence_order.csv", index_col=["subject","day"]) 
df_sequence.sort_index().to_excel("data/stage_data_out/image_for_irba/sequence_order.xlsx")"""