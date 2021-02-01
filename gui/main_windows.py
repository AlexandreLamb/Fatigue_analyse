
import math
import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk
import tkinter.filedialog 

from pannel_measure import MeasurePannel
from pannel_import import Pannel_import
from pannel_results import Pannel_results
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from video_transforme import VideoToLandmarks


class MainWindows(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.pannel = Pannel_import(parent)
        self.pannel.bind("<<TRANSFORM>>", lambda e: self.analyse_video(self.pannel))
        self.pannel = Pannel_results(parent)
        self.video_reader = None
        
        self.measure_pannel = MeasurePannel(parent)
        
    def analyse_video(self, pannel):
        video_to_analyse = [val.get("file_path") for index, val in enumerate(pannel.check_buttons_to_analyse_state) if val.get("state").get()]
        for video in video_to_analyse:
            videoLandmarks = VideoToLandmarks(video)
            videoLandmarks.load_and_transform()
    

root = tk.Tk()
window = MainWindows(root).pack(side="top", fill="both", expand=True)
root.geometry("1500x900")
root.mainloop()