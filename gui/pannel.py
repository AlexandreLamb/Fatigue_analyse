import tkinter as tk
from tkinter import *
from tkinter import filedialog

##TODO: deal with import probleme with utils function 

def parse_path_to_name(path):
    name_with_extensions = path.split("/")[-1]
    name = name_with_extensions.split(".")[0]
    return name


import random
class Pannel(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, width=200, height=800, bg='green')
        tk.Frame.pack(self, side="left")
        tk.Frame.pack_propagate(self,0)
        self.pannel_title = tk.StringVar()
        self.pannel_title.set("Bibliothèque")
        self.pannel_label = tk.Label(self, textvariable=self.pannel_title)
        self.pannel_label.pack(side="top", pady=(10,0))
        self.import_video_button = tk.Button(self, text ="Importer vidéos", command=self.add_item)
        self.import_video_button.pack(side="bottom", pady=(0,10))
        self.contenaire_videos = tk.Canvas(self, width=150, height=500, bg='red')
        self.contenaire_videos.pack(pady=(100,0))
        self.number_item = 1
        
    def add_item(self):
        file_path = tk.filedialog.askopenfilename(title="Choose the file to open", filetypes=[("All files", ".*")])
        item_text = tk.StringVar()
        item_text.set(parse_path_to_name(file_path))
        item_label = tk.Label(self.contenaire_videos, textvariable=item_text)
        item_label_2 = tk.Label(self.contenaire_videos, textvariable=item_text)
        self.contenaire_videos.create_window(75,25*self.number_item, window = item_label)
        self.number_item +=1