from mtcnn import MTCNN
import cv2
import pandas as pd

class FaceRecognitionMtcnn:
    def __init__(self):
        self.detector = MTCNN()
        
    def place_landmarks(self, img, count):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        points = self.detector.detect_faces(img_gray)
        confidence_df = pd.DataFrame(columns=["confidence"])
        for landmarks in points:
            confidence_df.loc[count] = landmarks.get("confidence")
            confidence_df.to_csv("data/stage_data_out/confidence_mtcnn.csv", mode="a")
            #box = landmarks.get("box") 
            #cv2.rectangle(img, (box[0], box[1]+box[3]), (box[0]+box[2],box[1]),(255,0,0),-1,cv2.LINE_AA)
            
            #cv2.circle(img, (mark[0], mark[1]), 2, (0,255,0), -1, cv2.LINE_AA)
        #cv2.imwrite("data/stage_data_out/landmarks_pics/frame"+str(count)+".jpg", img)
