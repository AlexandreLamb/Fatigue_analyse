import numpy as np
import cv2
def draw_circle_mouse(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),10,(255,0,0),-1)
        mouseX,mouseY = x,y
        
        
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)


while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print(mouseX,mouseY)
        
        
def place_landmarks_on_img(path_to_landmarks_csv, frame):
    landmarks_coordinates = pd.read_csv(path_to_landmarks_csv)
    frame_landmarks = list(landmarks_coordinates.loc[frame])
    frame_landmarks = frame_landmarks.pop(0)
    landmarks_pair = list(zip(frame_l[::2],frame_l[1::2]))
    for landmarks in landmarks_pair:
        cv2.circle(img,(x,y),10,(255,0,0),-1)
     