import pandas as pd
from datetime import datetime
import numpy as np
import re
import sys
import os
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from database_connector import SFTPConnector
from dotenv import load_dotenv
load_dotenv("env_file/.env_path")
PATH_TO_IRBA_DATA_PVT = os.environ.get("PATH_TO_IRBA_DATA_PVT")

"""
main.py
====================================
The core module of my example project
"""

def make_landmarks_header():
    """Function who generate a colcumns header for the landamrks dataframe

    :return: array of string with 68 landmarks_(number)_x and landmarks_(number)_y
    :rtype: Array (str)
    """
    csv_header = []
    for i in range(1,69):
        csv_header.append("landmarks_"+str(i)+"_x")
        csv_header.append("landmarks_"+str(i)+"_y")
    return csv_header

def parse_path_to_name(path):
    """function for parse a path to file and who split the "/" path and extension sfile for get just the name

    :param path: Path to file to parse
    :type path: str
    :return: name parsed without the extensions
    :rtype: str
    """
    name_with_extensions = path.split("/")[-1]
    name = name_with_extensions.split(".")[0]
    return name

def paths_to_df(csv_array):
    """Function who generate an array of dataframe from an array of csv path

    :param csv_array: Array of csv path to read
    :type csv_array: Array (str)
    :return: Array of Pandas dataframe with all the csv path read and append
    :rtype: Array of Pandas dataframe
    """
    sftp = SFTPConnector()
    df_array = []
    for path in csv_array:
        df_array.append(sftp.read_remote_df(path).rename(columns={"Unnamed: 0" : "frame"}))
    del sftp
    return df_array

def make_box_from_landmarks(row, threeshold_px = 20):
    """Function who generate an box with specific threeshold in pixels, use for get a correct box who contains the face

    :param row: An object who contains all landmarks coordinate for create the box.
    :type row: Object with landmarks
    :param threeshold_px: the value of space in pixels to augment the box, defaults to 20
    :type threeshold_px: int, optional
    :return: Object with "top", "left", "height" and "width" of the box
    :rtype: Object of int
    """
    left = int(row["landmarks_1_x"] - threeshold_px)
    top = int(np.mean([row["landmarks_20_y"] , row["landmarks_25_y"]]) - 2* threeshold_px)
    height = int( row["landmarks_9_y"] - top - threeshold_px)
    width = int(row["landmarks_17_x"] - left + threeshold_px)
    return {"top" : top, "left" : left, "height" : height, "width" : width} 


def parse_video_name(video_name_list):
    """Function to parse the list of bideo name into a list of video name with fomrat like that HXX_condition_periode

    :param video_name_list: List of video name to parse
    :type video_name_list: List (str)
    :return: List of video name parsed
    :rtype: List (str)
    """
    sftp = SFTPConnector()
    subject_condition = list(sftp.read_remote_df(os.path.join(PATH_TO_IRBA_DATA_PVT,"sujets_data_pvt_perf.csv"), sep=";", index_col = [0,1]).index)
    jour_1_to_parse = ["LUNDI", "lundi"]

    string_to_remove = ["DESFAM", "DESFAM-F", "PVT", "P1", "P2", "DEBUT", "FIN", "retard","min","de" , "avant PVT", "avant", "PVT", "F", "Semaine","1", "08", "8"]
    subject_list = []
    for subject_to_parse in video_name_list:
        subject_clean = []
        subject_split = re.split("_|\s+",subject_to_parse)
        #print(subject_split)
        for sub in subject_split:
            #print("str split :" +str(subject_split))
            if sub not in string_to_remove :
                subject_clean.append(sub)
        condition =  [cond[1] for cond  in subject_condition if cond[0] == subject_clean[0]][0]
        if subject_clean[1] in jour_1_to_parse:
            subject_list.append(subject_clean[0] +"_"+condition +  "_T1")
        else :
            subject_list.append(subject_clean[0] +"_"+condition + "_T2")
    del sftp
    return subject_list


date_id = lambda : datetime.now().strftime("%Y_%m_%d_%H_%M")

make_landmarks_pair  = lambda marks : list(zip(marks[::2],marks[1::2]))


def get_last_date_item(path_to_folder):
    sftp = SFTPConnector()
    dataset_array = sftp.list_dir_remote(path_to_folder)
    dataset_array.sort()
    del sftp
    return os.path.join(path_to_folder,dataset_array[-1])