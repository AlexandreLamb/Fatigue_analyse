# importing the module.
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split

path_to_xml = "dlib_trainer/ibug_300W_large_face_landmark_dataset/labels_ibug_300W.xml"
tree =  ET.parse(path_to_xml)
root = tree.getroot()

path_to_template = "dlib_trainer/label_template_empty.xml"
tree_template =  ET.parse(path_to_template)
root_template = tree_template.getroot()
images_tempalte = root_template.find("images")

images = root.find("images")
images_data = images.findall("image")
images_len = len(images_data)
      
images_train, images_test = train_test_split(images_data, test_size=0.3, random_state = 42)

#images_train.write('dlib_trainer/images_train.xml')
for element in images_train:
	images_tempalte.append(element)

tree_template.write("dlib_trainer/train.xml")