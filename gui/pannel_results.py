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
from data_analyse import AnalyseData
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure



class Pannel_results(tk.Frame):
    def __init__(self, parent):
        #fenêtre qui va afficher les graphiques 
        tk.Frame.__init__(self, parent, width=700, height=700, bg='gray70', highlightbackground="black", highlightthickness=2)
        tk.Frame.pack(self, side="left")
        tk.Frame.pack_propagate(self,0)

        self.canv = tk.Canvas(self, width=700, height=700, scrollregion=(0, 0, 0, 500))

        self.defilY = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.defilY.pack(side=tk.RIGHT, fill=Y)
        self.defilY.config(command=self.canv.yview)
        
        self.canv.config(yscrollcommand=self.defilY.set)

        self.pannel_title = tk.StringVar()
        self.pannel_title.set("Résultats:")
        self.pannel_label = tk.Label(self, textvariable=self.pannel_title, bd=8, height=1, width=18)
        self.pannel_label.pack(side="top", pady=(5,0), padx=(0,0))

        #self.get_graphs_analyse()
        self.plot_graphs_analyse("data/data_out/DESFAM_Semaine-2-Vendredi_PVT_H63_hog.csv")


    def plot_graphs_analyse(self, csv_path):
        data_analyse = AnalyseData(csv_path)

        ####
        data_analyse.nose_wrinkles()
        nose_wrinkles_graph = data_analyse.plot_measure("eyebrow_eye", "Nose wrinkles")
        canvas = FigureCanvasTkAgg(nose_wrinkles_graph, master=self)
        canvas.get_tk_widget().pack(pady=(5,0))
        canvas.draw()
        
        ####
        data_analyse.jaw_dropping()
        jaw_dropping_graph = data_analyse.plot_measure("jaw_dropping", "Jaw dropping")
        canvas = FigureCanvasTkAgg(jaw_dropping_graph, master=self)
        canvas.get_tk_widget().pack(pady=(5,0))
        canvas.draw()

        ####
        # data_analyse.measure_eyebrow_nose()
        # eyebrow_nose_graph = data_analyse.plot_measure("eyebrow_nose", "Distance between eyebrow and nose")
        # canvas = FigureCanvasTkAgg(eyebrow_nose_graph, master=self)
        # canvas.get_tk_widget().pack(pady=(5,0))
        # canvas.draw()
        
        ###
        # data_analyse.measure_yawning_frequency(1500)
        # yawning_frequency_graph = data_analyse.plot_measure("yawning_frequency", axis_x = "Time (in mins)")
        # canvas = FigureCanvasTkAgg(jaw_dropping_graph, master=self)
        # canvas.get_tk_widget().pack()
        # canvas.draw()

        ###
        # data_analyse.measure_ear()
        # ear_graph = data_analyse.plot_measure("ear", "Eye aspect ratio")
        # canvas = FigureCanvasTkAgg(ear_graph, master=self)
        # canvas.get_tk_widget().pack(pady=(5,0))
        # canvas.draw()

        # ###
        # data_analyse.measure_mean_eye_area(30)
        # ear_mean_graph = data_analyse.plot_measure("eye_area_mean_over_30_frame", "Eye aspect ratio mean" ,"eye_area_theshold")
        # canvas = FigureCanvasTkAgg(ear_mean_graph, master=self)
        # canvas.get_tk_widget().pack(pady=(5,0))
        # canvas.draw()

        ###
        # data_analyse.blinking_frequency(1500)
        # blinking_graph = data_analyse.plot_measure("blinking_frequency", "Blinking frequency per minute", axis_x = "Time (in mins)")
        # canvas = FigureCanvasTkAgg(blinking_graph, master=self)
        # canvas.get_tk_widget().pack(pady=(5,0))
        # canvas.draw()

        # ###
        # data_analyse.measure_perclos(1500, 80)
        # perclos_graph = data_analyse.plot_measure("perclos_measure", "Perclos measure", axis_x = "Time (in mins)")
        # canvas = FigureCanvasTkAgg(perclos_graph, master=self)
        # canvas.get_tk_widget().pack(pady=(5,0))
        # canvas.draw()

        # ###
        # data_analyse.eyes_angle()
        # left_eye_graph = data_analyse.plot_measure(["left_angle1","left_angle2"], "Angles measures for the left eye")
        # canvas = FigureCanvasTkAgg(left_eye_graph, master=self)
        # canvas.get_tk_widget().pack(pady=(5,0))
        # canvas.draw()
        # right_eye_graph = data_analyse.plot_measure(["right_angle1","right_angle2"], "Angles measures for the right eye")
        # canvas = FigureCanvasTkAgg(right_eye_graph, master=self)
        # canvas.get_tk_widget().pack(pady=(5,0))
        # canvas.draw()
        

        
   

    
        