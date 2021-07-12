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
        tk.Frame.__init__(self, parent, width=1000, height=700, highlightbackground="black", highlightthickness=2)
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
        self.frame_graphs = tk.Frame(self, width=1000, height=690)
        self.frame_graphs.pack(pady=(10,0))

        self.canvas_graphs = tk.Canvas(self.frame_graphs, width=1000, height=600, scrollregion=(0, 0, 0, 7000))

        #scrollbar 
        self.graphs_scrollbar = tk.Scrollbar(self.frame_graphs, orient=tk.VERTICAL)
        self.graphs_scrollbar.pack(side=tk.RIGHT, fill=Y)
        self.graphs_scrollbar.config(command=self.canvas_graphs.yview)
        
        self.canvas_graphs.config(yscrollcommand=self.graphs_scrollbar.set)
        self.canvas_graphs.pack()
        self.graph_list = []

    
    def save_plot(self, fig):
        image = fig
        #print(type(fig))
        #self.file_opt = options = {}
        formats = [('jpeg image', '*.jpg'), ('png image', '*.png')]
        result = tk.filedialog.asksaveasfile(parent=self, filetypes=formats, title="Save as...")
        if result is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        image.savefig(result.name)

    def save_csv(self, data_analyse, measure, axis = None):
        formats = [('Comma Separated values', '*.csv')]
        file_name = tk.filedialog.asksaveasfilename(parent=self, filetypes=formats, title="Save as...")
        if axis == None:
            data_analyse.save_csv(measure, file_name)
        else : 
            data_analyse.save_csv(measure, file_name, axis)
       
      
    def plot_graphs_analyse(self, csv_path, measures_state):
        self.clean_graph_frame()
        df_videos_infos = pd.read_csv("data/data_out/videos_infos.csv")
        fps = list(df_videos_infos[df_videos_infos["video_name"]==parse_path_to_name(csv_path)]["fps"])[0]
        data_analyse = AnalyseData(csv_path+".csv")

        measure_eyebrow_eye         = [val for index, val in enumerate(measures_state) if (val.get("state").get()) and (val.get("measure") == "eyebrow_eye")]
        measure_jaw_dropping        = [val for index, val in enumerate(measures_state) if (val.get("state").get()) and (val.get("measure") == "jaw_dropping")]
        measure_yawning_frequency   = [val for index, val in enumerate(measures_state) if (val.get("state").get()) and (val.get("measure") == "yawning_frequency")]
        measure_ear                 = [val for index, val in enumerate(measures_state) if (val.get("state").get()) and (val.get("measure") == "ear")]
        measure_eye_area_mean_over  = [val for index, val in enumerate(measures_state) if (val.get("state").get()) and (val.get("measure") == "eye_area_mean_over")]
        measure_blinking_frequency  = [val for index, val in enumerate(measures_state) if (val.get("state").get()) and (val.get("measure") == "blinking_frequency")]
        measure_perclos             = [val for index, val in enumerate(measures_state) if (val.get("state").get()) and (val.get("measure") == "perclos_measure")]
        measure_eye_angle           = [val for index, val in enumerate(measures_state) if (val.get("state").get()) and (val.get("measure") == "eye_angle")]
        measure_eyebrow_nose        = [val for index, val in enumerate(measures_state) if (val.get("state").get()) and (val.get("measure") == "eyebrow_nose")]

        #######################################################
        if(len(measure_eyebrow_eye) == 1):
            frame1 = tk.Frame(self.frame_graphs, width=900, height=700)
            frame1.pack()

            data_analyse.nose_wrinkles()
            nose_wrinkles_graph = data_analyse.plot_measure("eyebrow_eye", "Nose wrinkles")
            canvas = FigureCanvasTkAgg(nose_wrinkles_graph, master=frame1)
            canvas.get_tk_widget().pack(pady=(5,0))
            canvas.draw()
            item_id= self.canvas_graphs.create_window(250, len(self.graph_list)*500 + 200, window=frame1)
            print(item_id)
            b1 = Button(frame1, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(nose_wrinkles_graph))
            b1.pack(side=tk.LEFT, padx=(200,0))

            z1 = Button(frame1, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(data_analyse, "eyebrow_eye"))
            z1.pack(side=tk.LEFT,padx=(5,0))
            self.graph_list.append(1)
        ######################################################
        if(len(measure_jaw_dropping) == 1):
            frame2 = tk.Frame(self.frame_graphs, width=900, height=700)
            frame2.pack()
            
            data_analyse.jaw_dropping()
            jaw_dropping_graph = data_analyse.plot_measure("jaw_dropping", "Jaw dropping")
            canvas2 = FigureCanvasTkAgg(jaw_dropping_graph, master=frame2)
            canvas2.get_tk_widget().pack(pady=(5,0))
            canvas2.draw()
            self.canvas_graphs.create_window(250, len(self.graph_list)*500 + 200, window=frame2)

            b2 = Button(frame2, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(jaw_dropping_graph))
            b2.pack(side=tk.LEFT, padx=(200,0))

            z2 = Button(frame2, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(data_analyse, "jaw_dropping"))
            z2.pack(side=tk.LEFT,padx=(5,0))
            self.graph_list.append(1)

        ###################################################
        if(len(measure_eyebrow_nose) == 1):
    
            frame3 = tk.Frame(self.frame_graphs, width=900, height=700)
            frame3.pack()

            data_analyse.measure_eyebrow_nose()
            eyebrow_nose_graph = data_analyse.plot_measure("eyebrow_nose", "Distance between eyebrow and nose")
            canvas3 = FigureCanvasTkAgg(eyebrow_nose_graph, master=frame3)
            canvas3.get_tk_widget().pack(pady=(5,0))
            canvas3.draw()
            self.canvas_graphs.create_window(250, len(self.graph_list)*500 + 200, window=frame3)

            b3 = Button(frame3, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(eyebrow_nose_graph))
            b3.pack(side=tk.LEFT, padx=(200,0))

            z3 = Button(frame3, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(data_analyse,"eyebrow_nose" ))
            z3.pack(side=tk.LEFT,padx=(5,0))
            self.graph_list.append(1)
        ##########################################################
        if(len(measure_yawning_frequency) == 1):
    
            frame4 = tk.Frame(self.frame_graphs, width=900, height=700)
            frame4.pack()
            treshold = int(int(measure_yawning_frequency[0].get("ent_1").get())*fps)
            ## TODO: add conversion frame sec
            
            data_analyse.measure_yawning_frequency(treshold)
            yawning_frequency_graph = data_analyse.plot_measure("yawning_frequency", "Yawning frequency (over "+measure_yawning_frequency[0].get("ent_1").get()+")", axis_x = "Time (in mins)")
            canvas4 = FigureCanvasTkAgg(yawning_frequency_graph, master=frame4)
            canvas4.get_tk_widget().pack(pady=(5,0))
            canvas4.draw()
            self.canvas_graphs.create_window(250, len(self.graph_list)*500 + 200, window=frame4)

            b4 = Button(frame4, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(yawning_frequency_graph))
            b4.pack(side=tk.LEFT, padx=(200,0))

            z4 = Button(frame4, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(data_analyse,"yawning_frequency","Time (in mins)"))
            z4.pack(side=tk.LEFT,padx=(5,0))
            self.graph_list.append(1)

        ################################################
        if(len(measure_ear) == 1):
    
            frame5 = tk.Frame(self.frame_graphs, width=900, height=700)
            frame5.pack()

            data_analyse.measure_ear()
            ear_graph = data_analyse.plot_measure("ear", "Eye aspect ratio")
            canvas5 = FigureCanvasTkAgg(ear_graph, master=frame5)
            canvas5.get_tk_widget().pack(pady=(5,0))
            canvas5.draw()
            self.canvas_graphs.create_window(250, len(self.graph_list)*500 + 200, window=frame5)

            b5 = Button(frame5, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(ear_graph))
            b5.pack(side=tk.LEFT, padx=(200,0))

            z5 = Button(frame5, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(data_analyse, "ear"))
            z5.pack(side=tk.LEFT,padx=(5,0))
            self.graph_list.append(1)

        ############################################################
        if(len(measure_eye_area_mean_over) == 1):
            frame6 = tk.Frame(self.frame_graphs, width=900, height=700)
            frame6.pack()
            
            treshold = int(int(measure_eye_area_mean_over[0].get("ent_1").get())*fps)
            data_analyse.measure_mean_eye_area(treshold)

            ear_mean_graph = data_analyse.plot_measure("eye_area_mean_over_"+str(treshold)+"_frame", "Eye area mean" ,"eye_area_theshold")
            canvas6 = FigureCanvasTkAgg(ear_mean_graph, master=frame6)
            canvas6.get_tk_widget().pack(pady=(5,0))
            canvas6.draw()
            self.canvas_graphs.create_window(250, len(self.graph_list)*500 + 200, window=frame6)

            b6 = Button(frame6, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(ear_mean_graph))
            b6.pack(side=tk.LEFT, padx=(200,0))

            z6 = Button(frame6, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(data_analyse, "eye_area_mean_over_"+str(treshold)+"_frame","eye_area_theshold"))
            z6.pack(side=tk.LEFT,padx=(5,0))
            self.graph_list.append(1)

        ##############################################################
        if(len(measure_blinking_frequency) == 1):
            frame7 = tk.Frame(self.frame_graphs, width=900, height=700)
            frame7.pack()
            
            treshold = int(int(measure_blinking_frequency[0].get("ent_1").get())*fps)
            data_analyse.blinking_frequency(treshold)
            
            blinking_graph = data_analyse.plot_measure("blinking_frequency", "Blinking frequency per "+measure_blinking_frequency[0].get("ent_1").get() + " s", axis_x = "Time (in mins)")
            canvas7 = FigureCanvasTkAgg(blinking_graph, master=frame7)
            canvas7.get_tk_widget().pack(pady=(5,0))
            canvas7.draw()
            self.canvas_graphs.create_window(250, len(self.graph_list)*500 + 200, window=frame7)

            b7 = Button(frame7, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(blinking_graph))
            b7.pack(side=tk.LEFT, padx=(200,0))

            z7 = Button(frame7, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(data_analyse, "blinking_frequency", "Time (in mins)"))
            z7.pack(side=tk.LEFT,padx=(5,0))
            self.graph_list.append(1)

        ################################################################
        if(len(measure_perclos) == 1):  
            frame8 = tk.Frame(self.frame_graphs, width=900, height=700)
            frame8.pack()
            
            treshold = int(int(measure_perclos[0].get("ent_1").get())*fps)
            percentage = int(measure_perclos[0].get("ent_2").get())
            data_analyse.measure_perclos(treshold, percentage)
            perclos_measures = data_analyse.measure_perclos(treshold, percentage)
            perclos_graph = data_analyse.plot_measure("perclos_measure", "Perclos measure at "+str(percentage)+"%", axis_x = "Time (in mins)")
            canvas8 = FigureCanvasTkAgg(perclos_graph, master=frame8)
            canvas8.get_tk_widget().pack(pady=(5,0))
            canvas8.draw()
            self.canvas_graphs.create_window(250, len(self.graph_list)*500 + 200, window=frame8)

            b8 = Button(frame8, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(perclos_graph))
            b8.pack(side=tk.LEFT, padx=(200,0))

            z8 = Button(frame8, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(data_analyse, "perclos_measure", "Time (in mins)"))
            z8.pack(side=tk.LEFT,padx=(5,0))
            self.graph_list.append(1)

        ###################################################################
        if(len(measure_eye_angle) == 1):   
            frame9 = tk.Frame(self.frame_graphs, width=900, height=700)
            frame9.pack()

            data_analyse.eyes_angle()

            left_eye_graph = data_analyse.plot_measure(["left_angle1","left_angle2"], "Angles measures for the left eye")
            canvas9 = FigureCanvasTkAgg(left_eye_graph, master=frame9)
            canvas9.get_tk_widget().pack(pady=(5,0))
            canvas9.draw()
            self.canvas_graphs.create_window(250, len(self.graph_list)*500 + 200, window=frame9)

            b9 = Button(frame9, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(left_eye_graph))
            b9.pack(side=tk.LEFT, padx=(200,0))

            z9 = Button(frame9, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(data_analyse, ["left_angle1", "left_angle2"]))
            z9.pack(side=tk.LEFT,padx=(5,0))
            
            self.graph_list.append(1)

            self.graph_list.append(1)
            

            frame10 = tk.Frame(self.frame_graphs, width=900, height=700)
            frame10.pack()

            right_eye_graph = data_analyse.plot_measure(["right_angle1","right_angle2"], "Angles measures for the right eye")
            canvas10 = FigureCanvasTkAgg(right_eye_graph, master=frame10)
            canvas10.get_tk_widget().pack(pady=(5,0))
            canvas10.draw()
            self.canvas_graphs.create_window(250, len(self.graph_list)*500 + 200, window=frame10)

            b10 = Button(frame10, text="Save plot", bg="gray70", fg = 'black', command = lambda : self.save_plot(right_eye_graph))
            b10.pack(side=tk.LEFT, padx=(200,0))

            z10 = Button(frame10, text="Save csv", bg="gray70", fg = 'black', command = lambda : self.save_csv(data_analyse, ["right_angle1", "right_angle2"]))
            z10.pack(side=tk.LEFT,padx=(5,0))
            self.graph_list.append(1)
    
    def clean_graph_frame(self):
        print("clean")
        self.canvas_graphs.destroy()
        self.graphs_scrollbar.destroy()
        self.canvas_graphs = tk.Canvas(self.frame_graphs, width=1000, height=600, scrollregion=(0, 0, 0, 7000))
        self.graphs_scrollbar = tk.Scrollbar(self.frame_graphs, orient=tk.VERTICAL)
        self.graphs_scrollbar.pack(side=tk.RIGHT, fill=Y)
        self.graphs_scrollbar.config(command=self.canvas_graphs.yview)
        self.canvas_graphs.config(yscrollcommand=self.graphs_scrollbar.set)
        self.canvas_graphs.pack()
        self.graph_list = []
        """ print(self.graph_list)
        print(self.canvas_graphs.winfo_children())
        for graph in self.graph_list:
            self.canvas_graphs.delete(str(graph))
        print(self.canvas_graphs.winfo_children())
        self.graph_list = []"""


        

        
   

    
        