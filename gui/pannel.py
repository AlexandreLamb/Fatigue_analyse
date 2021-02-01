import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import parse_path_to_name
#from data_analyse import AnalyseData


import random
class Pannel(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, width=300, height=700, bg='gray80')
        tk.Frame.pack(self, side="left")
        tk.Frame.pack_propagate(self,0)
        
        self.pannel_title = tk.StringVar()
        self.pannel_title.set("Bibliothèque")
        self.pannel_label = tk.Label(self, textvariable=self.pannel_title, bd=5, height=2, width=18)
        self.pannel_label.pack(side="top", pady=(10,0), padx=(10,10))
              
        self.frame_contenaire_to_analyse = tk.Frame(self, width=300, height=200)
        self.frame_contenaire_to_analyse.pack(pady=(10,0), padx=(10,10))
        
        self.contenaire_videos_to_analyse = tk.Canvas(self.frame_contenaire_to_analyse, width=300, height=200, bg='gray94', scrollregion=(0,0,500,500))
        
        self.scrollbar_to_analyse = tk.Scrollbar(self.frame_contenaire_to_analyse, orient=tk.VERTICAL)
        self.scrollbar_to_analyse.pack(side=tk.RIGHT, fill= Y)
        self.scrollbar_to_analyse.config(command=self.contenaire_videos_to_analyse.yview)

        self.scrollbar_to_analyse2 = tk.Scrollbar(self.frame_contenaire_to_analyse, orient=tk.HORIZONTAL)
        self.scrollbar_to_analyse2.pack(side=tk.BOTTOM, fill= X)
        self.scrollbar_to_analyse2.config(command=self.contenaire_videos_to_analyse.xview)
        
        self.contenaire_videos_to_analyse.config(yscrollcommand=self.scrollbar_to_analyse.set, xscrollcommand=self.scrollbar_to_analyse2.set)
        self.contenaire_videos_to_analyse.pack(side=LEFT,expand=True,fill=BOTH)
        
        self.import_video_button = tk.Button(self, text ="Importer vidéos", bd=5, height=2, width=18, command=self.add_item)
        self.import_video_button.pack(side="top", pady=(5,0), padx=(0,0))
        
        self.analyse_button = tk.Button(self, text="Transformer", bd=5, height=2, width=18, command= lambda: self.event_generate("<<TRANSFORM>>"))
        self.analyse_button.pack(side="top", pady=(5,0), padx =(0,0))
        

        self.frame_contenaire_analyse = tk.Frame(self, width=300, height=200)
        self.frame_contenaire_analyse.pack(pady=(20,0), padx=(10,10))
        
        #videos déja présentes
        self.contenaire_videos_analyse = tk.Canvas(self.frame_contenaire_analyse, width=300, height=200, bg='gray94',scrollregion=(0,0,500,500))
        
        self.scrollbar_analyse = tk.Scrollbar(self.frame_contenaire_analyse, orient=tk.VERTICAL)
        self.scrollbar_analyse.pack(side=tk.RIGHT, fill= Y)
        self.scrollbar_analyse.config(command=self.contenaire_videos_analyse.yview)

        self.scrollbar_analyse2 = tk.Scrollbar(self.frame_contenaire_analyse, orient=tk.HORIZONTAL)
        self.scrollbar_analyse2.pack(side=tk.BOTTOM, fill= X)
        self.scrollbar_analyse2.config(command=self.contenaire_videos_analyse.xview)
   
        self.contenaire_videos_analyse.config(yscrollcommand=self.scrollbar_analyse.set, xscrollcommand=self.scrollbar_analyse2.set)
        self.contenaire_videos_analyse.pack(side=LEFT,expand=True,fill=BOTH)

        #fenêtre qui va afficher les graphiques 
        tk.Frame.__init__(self, parent, width=700, height=700, bg='gray70', highlightbackground="black", highlightthickness=2)
        tk.Frame.pack(self, side="left")
        tk.Frame.pack_propagate(self,0)
        
        self.pannel_title = tk.StringVar()
        self.pannel_title.set("Résultats:")
        self.pannel_label = tk.Label(self, textvariable=self.pannel_title, bd=8, height=1, width=18)
        self.pannel_label.pack(side="top", pady=(10,0), padx=(0,0))

        #self.get_graphs_analyse()
        

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

        if (parse_path_to_name(file_path) in list(self.video_analyse)) == True:
            messagebox.showinfo(title="Alerte", message="Vidéo déjà chargée")

    
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
    
    # def get_graphs_analyse(self):
    #     nose_wrinkles_graph = AnalyseData.nose_wrinkles()
    #     nose_wrinkles_graph = AnalyseData.plot_measure("eyebrow_eye")
        
