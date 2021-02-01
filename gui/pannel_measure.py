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
class MeasurePannel(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, width=300, height=700, bg='gray80')
        tk.Frame.pack(self, side="right")
        tk.Frame.pack_propagate(self,0)
        
        self.pannel_title = tk.StringVar()
        self.pannel_title.set("Measure")
        self.pannel_label = tk.Label(self, textvariable=self.pannel_title, bd=5, height=2, width=18)
        self.pannel_label.pack(side="top", pady=(10,0), padx=(10,10))
        self.validate_button = TK.Button(self,)
        self.entries = []
        self.add_measure("test")
        self.add_measure("test")
        self.add_measure_input("test","tets_1","test_2")
        self.add_measure_input("test","tets_1")

    
    def add_measure(self, measure_name):
        item_frame = tk.Frame(self, width=300, height=50, bg="yellow")
        item_text = tk.StringVar()
        item_text.set(measure_name)
        state = tk.IntVar()
        item_label = tk.Checkbutton(item_frame, variable= state, textvariable=item_text)
        item_label.pack(pady=(12,0))
        item_frame.pack(pady=(10,0))
        item_frame.pack_propagate(0)

    
    def add_measure_input(self, measure_name, input_name_1, input_name_2 = None):
        
        item_frame = tk.Frame(self, width=300, height=50, bg="yellow")
        item_title = tk.StringVar()
        item_title.set(measure_name)
        state = tk.IntVar()
        item_label = tk.Checkbutton(item_frame, variable= state, textvariable=item_title)
        
        item_input = tk.StringVar()
        item_input.set(input_name_1 + " : ")
        label_input = tk.Label(item_frame, textvariable=item_input)
        
        ent = tk.Entry(item_frame, width = 5)    
        ent.insert(0, "0")
        self.entries.append(ent)
        
        if(input_name_2 != None):
            item_input_2 = tk.StringVar()
            item_input_2.set(input_name_2 + " : ")
            label_input_2 = tk.Label(item_frame, textvariable=item_input_2)
            
            ent_2 = tk.Entry(item_frame, width = 5)    
            ent_2.insert(0, "0")
            self.entries.append(ent_2)
            
        

     
    
        item_label.pack(side="top")
        label_input.pack(side="left", padx=(20,0))
        ent.pack(side="left", padx=(20,0))
        if(input_name_2 != None):
            label_input_2.pack(side="left", padx=(20,0))
            ent_2.pack(side="left", padx=(20,0))
        item_frame.pack(pady=(10,0))
        item_frame.pack_propagate(0)

        