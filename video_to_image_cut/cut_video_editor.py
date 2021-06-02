import numpy as np
import cv2
import os
cap = cv2.VideoCapture('linusi.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def video_cut(path_to_video):
    cap = cv2.VideoCapture(path_to_video)
    fps = 10#cap.get(cv2.CAP_PROP_FPS)
    frame_count = 28738 #cap.get(cv2.CAP_PROP_FRAME_COUNT)
    time_to_cut_sec = 10
    number_frame_to_cut = time_to_cut_sec  * fps
    windows_to_cut_frame = [0, 15*60*fps , 30*60*fps, frame_count - number_frame_to_cut ]
    video_name = path_to_video.split("/")[-1]
    for windows_to_cut in windows_to_cut_frame:
        for frame in range(number_frame_to_cut+1):
            #cap.set(2,(frame+windows_to_cut)/frame_count) 
            #success, image = cap.read()
            path_folder_to_save = "data/stage_data_out/image_for_irba/"+video_name
            path_img_to_save = path_folder_to_save + video_name+'_frame_'+str(frame+windows_to_cut)+'.png'
            print((frame+windows_to_cut)/frame_count)
        """
        if os.path.exists(path_folder_to_save) == False:
            os.makedirs(path_folder_to_save)
        cv2.imwrite(path_img_to_save,image)
        """
video_cut("")