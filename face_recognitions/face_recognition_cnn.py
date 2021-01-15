import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from log import logging

class FaceRecognitionCNN:
    def __init__(self):
        self.model_face_detector= "data/data_in/models/res10_300x300_ssd_iter_140000.caffemodel"
        self.config_file = "data/data_in/models/deploy.prototxt"
        self.model_landmarks = "data/data_in/models/pose_model"
        self.face_detector =  cv2.dnn.readNetFromCaffe(self.config_file, self.model_face_detector)
        self.landmark_model = keras.models.load_model(self.model_landmarks)

    def face_detection_dnn(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        res = self.face_detector.forward()
        faces = []
        for i in range(res.shape[2]):
            confidence = res[0, 0, i, 2]
            if confidence > 0.5:
                box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                faces.append([x, y, x1, y1])
        return faces

    def move_box(self, box, offset):
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    def get_square_box(self, box):
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y
        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)
        if diff == 0:                   # Already a square.
            return box
        elif diff > 0:                  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:                           # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1
        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'
        return [left_x, top_y, right_x, bottom_y]

    def draw_marks(image, marks, color=(0, 255, 0)):
        for mark in marks:
            cv2.circle(np.array(image), (mark[0], mark[1]), 2, color, -1, cv2.LINE_AA)

    def detect_landmarks_dnn(self, img, face):
        offset_y = int(abs((face[3] - face[1]) * 0.1))
        box_moved = self.move_box(face, [0, offset_y])
        facebox = self.get_square_box(box_moved)
        face_img = img[facebox[1]: facebox[3], facebox[0]: facebox[2]]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # # Actual detection.
        predictions = self.landmark_model.signatures["predict"](tf.constant([face_img], dtype=tf.uint8))
        # Convert predictions to landmarks.
        marks = np.array(predictions['output']).flatten()[:136]
        marks = np.reshape(marks, (-1, 2))
        marks *= (facebox[2] - facebox[0])
        marks[:, 0] += facebox[0]
        marks[:, 1] += facebox[1]
        marks = marks.astype(np.uint)
        return marks

    def place_landmarks(self, img, count) :
        logging.info("place landmarks on image "+str(count))
        rects = self.face_detection_dnn(img)
        if len(rects) == 1 :
            marks = self.detect_landmarks_dnn(img,rects[0])
            return marks
        elif len(rects) >= 1:
            logging.info("Detect more than 1 face on img number " +str(count) + " get default first face detect")
            marks = self.detect_landmarks_dnn(img,rects[0])
            return marks.ravel()
        else :
            return []
