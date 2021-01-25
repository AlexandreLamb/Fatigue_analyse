
import math
import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk
import tkinter.filedialog 
from pannel import Pannel

class MainWindows(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.pannel = Pannel(parent)

root = tk.Tk()
window = MainWindows(root).pack(side="top", fill="both", expand=True)
root.geometry("1500x900")
root.mainloop()