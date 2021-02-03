import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import parse_path_to_name
#from data_analyse import AnalyseData


import random
class MeasurePannel(tk.Frame):
    def __init__(self, parent, video_name):
        tk.Frame.__init__(self, parent, width=450, height=700, highlightbackground="black", highlightthickness=3)
        tk.Frame.pack(self, side="right", padx=(50,0))
        tk.Frame.pack_propagate(self,0)

        self.pannel_title = tk.StringVar()
        self.pannel_title.set("Measure on the following video: \n"+video_name)
        self.pannel_label = tk.Label(self, textvariable=self.pannel_title, bd=5, height=4, width=50)
        self.pannel_label.pack(side="top", pady=(3,0), padx=(10,10))

        #frame for measures
        self.frame_measures = tk.Frame(self, width=450, height=600)
        self.frame_measures.pack()
         
        self.contenaire_measure = tk.Canvas(self.frame_measures, width=450, height=510)
        
    
        self.contenaire_measure.pack()
        

        self.frame_button = tk.Frame(self, width=450, height=200)
        self.frame_button.pack(pady=(10,0))

        self.contenaire_measure2 = tk.Canvas(self.frame_button, width=450, height=200, bg='blue')
        self.contenaire_measure2.pack()

        self.validate_button = tk.Button(self.contenaire_measure2, text="Compute", height=3, width=18, command = lambda: self.event_generate("<<COMPUTE>>"))
        self.validate_button.pack()
        self.check_buttons_measure_state = []

        self.add_measure("yawning_frequency", "threshold (sec)")
        self.add_measure("blinking_frequency", "threshold (sec)")
        self.add_measure("ear")
        self.add_measure("microsleep_measure","threshold (sec)")
        self.add_measure("eyebrow_nose")
        self.add_measure("perclos_measure","threshold (sec)", "percentage")
        self.add_measure("eyebrow_eye")
        self.add_measure("eye_area_mean_over","threshold (sec)")
        self.add_measure("eye_angle")


    def update_title(self, video_name):
         self.pannel_title.set("Measure on \n"+video_name)

    def add_measure(self, measure_name, input_name_1 = None, input_name_2 = None):
        
        item_frame = tk.Frame(self.contenaire_measure, width=350, height=45)
        item_title = tk.StringVar()
        item_title.set(measure_name)
        state = tk.IntVar()
        item_label = tk.Checkbutton(item_frame, variable= state, textvariable=item_title)
        
        if((input_name_1 == None) and (input_name_2 == None)):
            self.check_buttons_measure_state.append({"measure" : measure_name, "state" : state })

        elif((input_name_1 != None) and (input_name_2 == None)):
            item_input = tk.StringVar()
            item_input.set(input_name_1 + " : ")
            label_input_1 = tk.Label(item_frame, textvariable=item_input)
            
            ent_1 = tk.Entry(item_frame, width = 5)    
            ent_1.insert(0, "0")
            self.check_buttons_measure_state.append({"measure" : measure_name, "state" : state, "input_1": input_name_1,  "ent_1" :ent_1})
        
        elif((input_name_1 != None) and (input_name_2 != None)):
            item_frame.config(width=350, height=100)
            item_input = tk.StringVar()
            item_input.set(input_name_1 + " : ")
            label_input_1 = tk.Label(item_frame, textvariable=item_input)
            
            ent_1 = tk.Entry(item_frame, width = 5)    
            ent_1.insert(0, "0")
            
            item_input_2 = tk.StringVar()
            item_input_2.set(input_name_2 + " : ")
            label_input_2 = tk.Label(item_frame, textvariable=item_input_2)
            
            ent_2 = tk.Entry(item_frame, width = 5)    
            ent_2.insert(0, "0")
            self.check_buttons_measure_state.append({"measure" : measure_name, "state" : state, "input_1": input_name_1, "input_2" : input_name_2, "ent_1" :ent_1, "ent_2":ent_2})
       
        if((input_name_1 == None) and (input_name_2 == None)):
            item_label.pack(side="left", pady=(0,0))
        else:
            item_label.pack(side="left", pady=(0,0))
            
        if((input_name_1 != None) and (input_name_2 == None)):
            label_input_1.pack()
            ent_1.pack()
        if((input_name_1 != None) and (input_name_2 != None)):
            label_input_1.pack(padx=(35,0))
            ent_1.pack(padx=(45,0))
            label_input_2.pack(padx=(35,0))
            ent_2.pack(padx=(45,0))
        
        item_frame.pack(pady=(5,0))
        item_frame.pack_propagate(0)
        
        
    def plot_info(self):
        for measure in self.check_buttons_measure_state:
            
            print(measure.get("measure"))
            print(measure.get("state").get())
            if measure.get("ent_1") != None :
                print(measure.get("ent_1").get())

        