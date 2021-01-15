class FaceRecognitionHOG:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    def place_landmarks_hog(self, image, count):
        logging.info("place landmarks on image " + str(count))
        image = imutils.resize(image, width=600)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # détecter les visages
        rects = self.detector(gray, 1)
        #print(rects)
        # Pour chaque visage détecté, recherchez le repère.
        for rect in rects:
            # déterminer les repères du visage for the face region, then
            # convertir le repère du visage (x, y) en un array NumPy
            marks = self.predictor(gray, rect)
            marks = face_utils.shape_to_np(marks)
            return marks
