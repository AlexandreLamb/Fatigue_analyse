import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import dlib
import datetime
import cv2
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import date_id, make_landmarks_pair, make_box_from_landmarks

class DlibPredictorTrainer:
    def __init__(self, path_to_landmarks_csv, path_to_video, path_to_xml):
        self.options = dlib.shape_predictor_training_options()
        self.train_accuracy = None
        self.test_accuracy = None
        self.path_to_xml = path_to_xml
        self.path_to_landmarks_csv = path_to_landmarks_csv
        self.path_to_video = path_to_video
        self.path_to_dataset_folder = "dlib_trainer/dataset/"
        self.video_name = path_to_video.split("/")[-1]
        self.landmarks_csv_name = path_to_landmarks_csv.split("/")[-1]
        self.new_dataset_image_path = os.path.join(self.path_to_dataset_folder,self.landmarks_csv_name.split(".")[0])
        self.dataset_tree =  ET.parse(path_to_xml)
        self.dataset_root = self.dataset_tree.getroot()
        
    def append_image(self, image_path, box_position, landmarks_tuple_array):       
        new_image = ET.SubElement(self.dataset_root.find('images'), "image")
        new_image.set("file", image_path)
        
        new_box = ET.SubElement(new_image, 'box')
        new_box.set( "top",str(box_position.get("top")))
        new_box.set( "left",str(box_position.get("left")))
        new_box.set( "width",str(box_position.get("width")))
        new_box.set( "height",str(box_position.get("height")))
        
        for index, landmarks_tuple in enumerate(landmarks_tuple_array):
            new_parts = ET.SubElement(new_box, 'part')
            new_parts.set("name",str(index))
            new_parts.set("x",str(landmarks_tuple[0]))
            new_parts.set("y", str(landmarks_tuple[1]))
        
    def save_dataset_xml(self):
        path_to_new_dataset = os.path.join(self.path_to_dataset_folder,"dataset_dlib_predictor_"+date_id()+".xml")
        self.dataset_tree.write(path_to_new_dataset)  
    
    def make_train_test_xml(self, path_to_xml, path_to_train_xml, path_to_test_xml, train_size):
        tree =  ET.parse(path_to_xml)
        root = tree.getroot()
        path_to_template = path_to_template_xml

        images = root.find("images")
        images_data = images.findall("image")
        images_len = len(images_data)
            
        images_train, images_test = train_test_split(images_data, test_size=0.3, random_state = 42)

        self.save_train_test_xml(images_train, path_to_train_xml, "xml_to_save")
        self.save_train_test_xml(images_test, path_to_template_xml, "xml_to_save")

    def save_train_test_xml(self, list_xml_element, path_to_template_xml, path_xml_to_save):
        tree_template =  ET.parse(path_to_template_xml)
        root_template = tree_template.getroot()
        images_tempalte = root_template.find("images")
        for element in list_xml_element:
            images_tempalte.append(element)
        tree_template.write(path_xml_to_save)
    
    def set_option(self, oversampling_amount, nu, tree_depth, be_verbose):
        self.option.oversampling_amount = oversampling_amount
        self.options.nu = nu
        self.option.tree_depth = tree_depth
        self.option.be_verbose = be_verbose
    
    def train_predictor(self, model_name, training_xml_path, test_xml_path):
        dlib.train_shape_predictor(training_xml_path, model_name, self.options)
        self.train_accuracy = dlib.test_shape_predictor(training_xml_path, model_name)
        self.test_accuracy = dlib.test_shape_predictor(test_xml_path, model_name)
    
    def show_image(self, img, img_landmarks, count, row) : 
        cv2.imshow("frame_" +str(count), img_landmarks)
        key = cv2.waitKey(0)
        sucess = True
        if key == 27 :
            cv2.destroyAllWindows()
        elif key == 97 :
            image_name = self.video_name.split(".")[0] + "_frame_" + str(count) + "_" + date_id() + ".png"        
            img_path = os.path.join(self.new_dataset_image_path, image_name)
            if os.path.isdir(self.new_dataset_image_path) : 
                cv2.imwrite(img_path, img)
                print("image save")
                self.append_image(img_path, make_box_from_landmarks(row), make_landmarks_pair(row))
                print("image append")
            cv2.destroyAllWindows()
        elif key == 115 :
            cv2.destroyAllWindows()
            print("save")
            self.save_dataset_xml()
            sucess = False
        return sucess

    def place_landmarks_on_img(self, img, landmarks):
        print("start_place")
        marks_pair = make_landmarks_pair(landmarks)
        for mark in marks_pair:
            cv2.circle(img, (mark[0], mark[1]), 2, (0,255,0), -1, cv2.LINE_AA)
        print("finish place")
        return img

    def manual_classify_image(self) : 
        df_landmarks = pd.read_csv(self.path_to_landmarks_csv, index_col="frame")       
        cap = cv2.VideoCapture(self.path_to_video)
        sucess, image = cap.read()
        img_index = 0 
        while sucess:
            if img_index in df_landmarks.index :
                image_landmarks = self.place_landmarks_on_img(image, list(df_landmarks.loc[img_index]))
                sucess = self.show_image(image, image_landmarks, img_index, df_landmarks.loc[img_index])
            if sucess:
                sucess, image = cap.read()
                img_index += 1
            

        