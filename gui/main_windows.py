
import math
import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk
import tkinter.filedialog 
from tkinter import messagebox
from pannel_measure import MeasurePannel
from pannel_import import Pannel_import
from pannel_results import Pannel_results
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from video_transforme import VideoToLandmarks
from format_data import DataFormator
from utils import parse_path_to_name

class MainWindows(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.pannel_import = Pannel_import(parent)
        self.pannel_import.bind("<<TRANSFORM>>", lambda e: self.analyse_video(self.pannel_import))
        self.pannel_import.bind("<<ADDMEASURE>>", lambda e: self.add_measure_pannel(parent, self.pannel_import))
        self.pannel_measure =None
        self.pannel_result = None
        self.path_video_analyse = None
        
    def analyse_video(self, pannel):
        video_to_analyse = [val.get("file_path") for index, val in enumerate(pannel.check_buttons_to_analyse_state) if val.get("state").get()]
        for video in video_to_analyse:
            videoLandmarks = VideoToLandmarks(video)
            videoLandmarks.load_and_transform()
    
    def add_measure_pannel(self, parent, pannel):
        video_paths =  [val.get("file_path") for index, val in enumerate(pannel.check_buttons_analyse_state) if val.get("state").get()]
        
        if(len(video_paths) >=2):
            data_format = DataFormator()
            messagebox.showinfo(title="Alert", message="Fusion of all video, this can take time")
            print([path + ".csv" for path in video_paths])
            self.path_video_analyse = data_format.merge_csv([path + ".csv" for path in video_paths])
        else : 
            self.path_video_analyse = video_paths[0]
            
        if(self.pannel_measure == None):
            self.pannel_measure = MeasurePannel(parent, parse_path_to_name(self.path_video_analyse))
            self.pannel_measure.bind("<<COMPUTE>>", lambda e: self.plot_graphs(parent, self.pannel_measure))
        else : 
            self.pannel_measure.update_title(parse_path_to_name(self.path_video_analyse))
    
    def plot_graphs(self, parent, pannel):
        if(self.pannel_result == None):
            self.pannel_result = Pannel_results(parent)
        self.pannel_result.plot_graphs_analyse(self.path_video_analyse, pannel.check_buttons_measure_state)


root = tk.Tk()
window = MainWindows(root).pack(side="top", fill="both", expand=True)
root.geometry("1500x900")
root.mainloop()