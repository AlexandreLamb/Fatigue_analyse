import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import parse_path_to_name


import random
class Pannel(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, width=400, height=800, bg='green')
        tk.Frame.pack(self, side="left")
        tk.Frame.pack_propagate(self,0)
        
        self.pannel_title = tk.StringVar()
        self.pannel_title.set("Bibliothèque")
        self.pannel_label = tk.Label(self, textvariable=self.pannel_title)
        self.pannel_label.pack(side="top", pady=(10,0))
              
        self.frame_contenaire_to_analyse = tk.Frame(self, width=300, height=200)
        self.frame_contenaire_to_analyse.pack(pady=(50,0))
        
        self.contenaire_videos_to_analyse = tk.Canvas(self.frame_contenaire_to_analyse, width=300, height=200, bg='red', scrollregion=(0,0,0,500))
        
        self.scrollbar_to_analyse = tk.Scrollbar(self.frame_contenaire_to_analyse, orient=tk.VERTICAL)
        self.scrollbar_to_analyse.pack(side=tk.RIGHT, fill= Y)
        self.scrollbar_to_analyse.config(command=self.contenaire_videos_to_analyse.yview)
        
        self.contenaire_videos_to_analyse.config(yscrollcommand=self.scrollbar_to_analyse.set)
        self.contenaire_videos_to_analyse.pack()
        
        self.import_video_button = tk.Button(self, text ="Importer vidéos", command=self.add_item)
        self.import_video_button.pack(side="top", pady=(5,0), padx=(0,0))
        
        self.analyse_button = tk.Button(self, text="Transform", command= lambda: self.event_generate("<<TRANSFORM>>"))
        self.analyse_button.pack(side="top", pady=(5,0), padx =(0,0))
        
        self.frame_contenaire_analyse = tk.Frame(self, width=300, height=200)
        self.frame_contenaire_analyse.pack(pady=(100,0))
        
        self.contenaire_videos_analyse = tk.Canvas(self.frame_contenaire_analyse, width=300, height=200, bg='yellow', scrollregion=(0,0,0,500))
        
        self.scrollbar_analyse = tk.Scrollbar(self.frame_contenaire_analyse, orient=tk.VERTICAL)
        self.scrollbar_analyse.pack(side=tk.RIGHT, fill= Y)
        self.scrollbar_analyse.config(command=self.contenaire_videos_analyse.yview)
        
        self.contenaire_videos_analyse.config(yscrollcommand=self.scrollbar_analyse.set)
        self.contenaire_videos_analyse.pack()

        self.check_buttons_to_analyse_state =  [] 
        self.check_buttons_analyse_state =  []   
        self.path_select="" 
        self.video_analyse = pd.read_csv("data/data_out/videos_infos.csv")["video_name"]
        
        self.get_videos_analyse()
        
    def add_item(self):
        state = tk.IntVar()
        file_path = tk.filedialog.askopenfilename(title="Choose the file to open", filetypes=[("Only video", "*.mp4 *.avi *.mov")])
        
        if (parse_path_to_name(file_path) in list(self.video_analyse)) == False:
            self.check_buttons_to_analyse_state.append({"file_path" : file_path, "state" : state})     
        
            item_text = tk.StringVar()
            item_text.set(parse_path_to_name(file_path))
            item_label = tk.Checkbutton(self.contenaire_videos_to_analyse, variable= state, textvariable=item_text)
        
            self.contenaire_videos_to_analyse.create_window(150,25*len(self.check_buttons_to_analyse_state), window = item_label)
    
    def get_videos_analyse(self):
        data_path = "data/data_out/"
        for video_name in self.video_analyse:
            state = tk.IntVar()
            self.check_buttons_analyse_state.append({"file_path" : data_path + video_name, "state" : state})     
        
            item_text = tk.StringVar()
            item_text.set(video_name)
            item_label = tk.Checkbutton(self.contenaire_videos_analyse, variable= state, textvariable=item_text)
        
            self.contenaire_videos_analyse.create_window(150,25*len(self.check_buttons_analyse_state), window = item_label)
        
    def upadte_video_analyse(self):
        data_path = "/home/simeon/Documents/ING5/PFE/Fatigue_analyse/data/data_in/videos/"
        video_analyse_update = pd.read_csv("data/data_out/videos_infos.csv")["video_name"]

        for video_name in video_analyse_update:
            if (video_name in self.video_analyse) == False:
                state = tk.IntVar()
                self.check_buttons_analyse_state.append({"file_path" : data_path + video_name, "state" : state})     
            
                item_text = tk.StringVar()
                item_text.set(video_name)
                item_label = tk.Checkbutton(self.contenaire_videos_analyse, variable= state, textvariable=item_text)
            
                self.contenaire_videos_analyse.create_window(150,25*len(self.check_buttons_analyse_state), window = item_label)
                
    def delete_item_video_to_analyse(self, video_name):
        data_path = "/home/simeon/Documents/ING5/PFE/Fatigue_analyse/data/data_in/videos/"
        items = self.contenaire_videos_to_analyse.winfo_children()
        for index, video in enumerate(self.check_buttons_to_analyse_state):
            if video.get("file_path") == data_path + video_name:
                self.check_buttons_to_analyse_state.pop(index)
        for item in items:
            item.destroy()
    
