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
from tkinter.filedialog import asksaveasfile 
from PIL import Image
import csv



class Pannel_results(tk.Frame):
    def __init__(self, parent):
        #fenêtre qui va afficher les graphiques 
        tk.Frame.__init__(self, parent, width=700, height=700, highlightbackground="black", highlightthickness=2)
        tk.Frame.pack(self, side = "left", padx=(50,0))
        tk.Frame.pack_propagate(self,0)

        #frame for title 
        self.frame_title = tk.Frame(self, width=700, height=10)
        self.frame_title.pack()

        self.pannel_title = tk.StringVar()
        self.pannel_title.set("Results:")
        self.pannel_label = tk.Label(self.frame_title, textvariable=self.pannel_title, bd=8, height=1, width=18)
        self.pannel_label.pack(side="top", pady=(5,0))
        
        #frame for graphs
        self.frame_graphs = tk.Frame(self, width=700, height=690)
        self.frame_graphs.pack(pady=(10,0))

        self.canvas_graphs = tk.Canvas(self.frame_graphs, width=700, height=600, scrollregion=(0, 0, 0, 7000))

        #scrollbar 
        self.graphs_scrollbar = tk.Scrollbar(self.frame_graphs, orient=tk.VERTICAL)
        self.graphs_scrollbar.pack(side=tk.RIGHT, fill=Y)
        self.graphs_scrollbar.config(command=self.canvas_graphs.yview)
        
        self.canvas_graphs.config(yscrollcommand=self.graphs_scrollbar.set)
        self.canvas_graphs.pack()

        #sself.plot_graphs_analyse("data/data_out/DESFAM_Semaine 2-Vendredi_PVT_H63_hog.csv")
    
    def save_plot(self, fig):
        image = fig
        #print(type(fig))
        #self.file_opt = options = {}
        formats = [('jpeg image', '*.jpg'), ('png image', '*.png')]
        result = tk.filedialog.asksaveasfile(parent=self, filetypes=formats, title="Save as...")
        if result is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        image.savefig(result.name)

    def save_csv(self, measures):
        formats = [('Comma Separated values', '*.csv')]
        file_name = tk.filedialog.asksaveasfilename(parent=self, filetypes=formats, title="Save as...")
        if file_name:
            with open(file_name, 'w') as fp:
                a = csv.writer(fp)
                # write row of header names
                a.writerow(measures)
                np.savetxt("foo.csv", measures, delimiter=",")
      
    def plot_graphs_analyse(self, csv_path, measure):

        data_analyse = AnalyseData(csv_path)

        #######################################################
        frame1 = tk.Frame(self.frame_graphs, width=700, height=700)
        frame1.pack()

        data_analyse.nose_wrinkles()
        nose_wrinkles_measures = data_analyse.nose_wrinkles()
        nose_wrinkles_graph = data_analyse.plot_measure("eyebrow_eye", "Nose wrinkles")
        canvas = FigureCanvasTkAgg(nose_wrinkles_graph, master=frame1)
        canvas.get_tk_widget().pack(pady=(5,0))
        canvas.draw()
        self.canvas_graphs.create_window(350, 200, window=frame1)

        b1 = Button(frame1, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(nose_wrinkles_graph))
        b1.pack(side=tk.LEFT, padx=(200,0))

        z1 = Button(frame1, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(nose_wrinkles_measures))
        z1.pack(side=tk.LEFT,padx=(5,0))

        ######################################################

        frame2 = tk.Frame(self.frame_graphs, width=700, height=700)
        frame2.pack()
        
        data_analyse.jaw_dropping()
        jaw_dropping_measures = data_analyse.jaw_dropping()
        jaw_dropping_graph = data_analyse.plot_measure("jaw_dropping", "Jaw dropping")
        canvas2 = FigureCanvasTkAgg(jaw_dropping_graph, master=frame2)
        canvas2.get_tk_widget().pack(pady=(5,0))
        canvas2.draw()
        self.canvas_graphs.create_window(350, 670, window=frame2)

        b2 = Button(frame2, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(jaw_dropping_graph))
        b2.pack(side=tk.LEFT, padx=(200,0))

        z2 = Button(frame2, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(jaw_dropping_measures))
        z2.pack(side=tk.LEFT,padx=(5,0))

        ###################################################

        frame3 = tk.Frame(self.frame_graphs, width=700, height=700)
        frame3.pack()

        data_analyse.measure_eyebrow_nose()
        eyebrow_nose_measures = data_analyse.measure_eyebrow_nose()
        eyebrow_nose_graph = data_analyse.plot_measure("eyebrow_nose", "Distance between eyebrow and nose")
        canvas3 = FigureCanvasTkAgg(eyebrow_nose_graph, master=frame3)
        canvas3.get_tk_widget().pack(pady=(5,0))
        canvas3.draw()
        self.canvas_graphs.create_window(350, 1100, window=frame3)

        b3 = Button(frame3, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(eyebrow_nose_graph))
        b3.pack(side=tk.LEFT, padx=(200,0))

        z3 = Button(frame3, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(eyebrow_nose_measures))
        z3.pack(side=tk.LEFT,padx=(5,0))
        
        ##########################################################

        frame4 = tk.Frame(self.frame_graphs, width=700, height=700)
        frame4.pack()
        
        data_analyse.measure_yawning_frequency(1500)
        yawning_frequency_measures = data_analyse.measure_yawning_frequency(1500)
        yawning_frequency_graph = data_analyse.plot_measure("yawning_frequency", "Yawning frequency", axis_x = "Time (in mins)")
        canvas4 = FigureCanvasTkAgg(yawning_frequency_graph, master=frame4)
        canvas4.get_tk_widget().pack(pady=(5,0))
        canvas4.draw()
        self.canvas_graphs.create_window(350, 1550, window=frame4)

        b4 = Button(frame4, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(yawning_frequency_graph))
        b4.pack(side=tk.LEFT, padx=(200,0))

        z4 = Button(frame4, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(yawning_frequency_measures))
        z4.pack(side=tk.LEFT,padx=(5,0))

        ################################################

        frame5 = tk.Frame(self.frame_graphs, width=700, height=700)
        frame5.pack()

        data_analyse.measure_ear()
        ear_measures = data_analyse.measure_ear()
        ear_graph = data_analyse.plot_measure("ear", "Eye aspect ratio")
        canvas5 = FigureCanvasTkAgg(ear_graph, master=frame5)
        canvas5.get_tk_widget().pack(pady=(5,0))
        canvas5.draw()
        self.canvas_graphs.create_window(350, 2000, window=frame5)

        b5 = Button(frame5, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(ear_graph))
        b5.pack(side=tk.LEFT, padx=(200,0))

        z5 = Button(frame5, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(ear_measures))
        z5.pack(side=tk.LEFT,padx=(5,0))

        ############################################################
        frame6 = tk.Frame(self.frame_graphs, width=700, height=700)
        frame6.pack()

        data_analyse.measure_mean_eye_area(30)
        ear_mean_measures = data_analyse.measure_mean_eye_area(30)
        ear_mean_graph = data_analyse.plot_measure("eye_area_mean_over_30_frame", "Eye aspect ratio mean" ,"eye_area_theshold")
        canvas6 = FigureCanvasTkAgg(ear_mean_graph, master=frame6)
        canvas6.get_tk_widget().pack(pady=(5,0))
        canvas6.draw()
        self.canvas_graphs.create_window(350, 2450, window=frame6)

        b6 = Button(frame6, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(ear_mean_graph))
        b6.pack(side=tk.LEFT, padx=(200,0))

        z6 = Button(frame6, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(ear_mean_measures))
        z6.pack(side=tk.LEFT,padx=(5,0))

        ##############################################################
        frame7 = tk.Frame(self.frame_graphs, width=700, height=700)
        frame7.pack()

        data_analyse.blinking_frequency(1500)
        blinking_frequency_measures = data_analyse.blinking_frequency(1500)
        blinking_graph = data_analyse.plot_measure("blinking_frequency", "Blinking frequency per minute", axis_x = "Time (in mins)")
        canvas7 = FigureCanvasTkAgg(blinking_graph, master=frame7)
        canvas7.get_tk_widget().pack(pady=(5,0))
        canvas7.draw()
        self.canvas_graphs.create_window(350, 3000, window=frame7)

        b7 = Button(frame7, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(blinking_graph))
        b7.pack(side=tk.LEFT, padx=(200,0))

        z7 = Button(frame7, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(blinking_frequency_measures))
        z7.pack(side=tk.LEFT,padx=(5,0))

        ################################################################
        frame8 = tk.Frame(self.frame_graphs, width=700, height=700)
        frame8.pack()

        data_analyse.measure_perclos(1500, 80)
        perclos_measures = data_analyse.measure_perclos(1500, 80)
        perclos_graph = data_analyse.plot_measure("perclos_measure", "Perclos measure", axis_x = "Time (in mins)")
        canvas8 = FigureCanvasTkAgg(perclos_graph, master=frame8)
        canvas8.get_tk_widget().pack(pady=(5,0))
        canvas8.draw()
        self.canvas_graphs.create_window(350, 3450, window=frame8)

        b8 = Button(frame8, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(perclos_graph))
        b8.pack(side=tk.LEFT, padx=(200,0))

        z8 = Button(frame8, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(perclos_measures))
        z8.pack(side=tk.LEFT,padx=(5,0))

        ###################################################################
        frame9 = tk.Frame(self.frame_graphs, width=700, height=700)
        frame9.pack()

        data_analyse.eyes_angle()

        left_eye_graph = data_analyse.plot_measure(["left_angle1","left_angle2"], "Angles measures for the left eye")
        canvas9 = FigureCanvasTkAgg(left_eye_graph, master=frame9)
        canvas9.get_tk_widget().pack(pady=(5,0))
        canvas9.draw()
        self.canvas_graphs.create_window(350, 3900, window=frame9)

        b9 = Button(frame9, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(left_eye_graph))
        b9.pack(side=tk.LEFT, padx=(200,0))


        frame10 = tk.Frame(self.frame_graphs, width=700, height=700)
        frame10.pack()

        right_eye_graph = data_analyse.plot_measure(["right_angle1","right_angle2"], "Angles measures for the right eye")
        canvas10 = FigureCanvasTkAgg(right_eye_graph, master=frame10)
        canvas10.get_tk_widget().pack(pady=(5,0))
        canvas10.draw()
        self.canvas_graphs.create_window(350, 4400, window=frame10)

        b10 = Button(frame10, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(right_eye_graph))
        b10.pack(side=tk.LEFT, padx=(200,0))


        

        
   

    
        