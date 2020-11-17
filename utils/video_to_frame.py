import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
				help="chemin d'accès à la video d'entrée")
args = vars(ap.parse_args())

vidcap = cv2.VideoCapture(args["video"])

success,image = vidcap.read()


count = 0;
while success:
  success,image = vidcap.read()
  if success:
   	cv2.imwrite("data/images/frame%d.jpg" % count, image)
  if cv2.waitKey(10) == 27:
      break
  count += 1
