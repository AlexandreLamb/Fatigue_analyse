
import math

from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
from tkinter import *
import tkinter as tk
from coordinator import Coordinatore

class MyWindow(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.createMenuBar()
        self.text = tk.StringVar()
        self.text.set("")
        self.label = tk.Label(self, textvariable=self.text)
        self.label.pack()
        self.button = tk.Button(self,text="Analyse", command=self.analyse)
        self.button.pack()
        # Fill the content of the window
        self.path = None
        self.geometry( "300x200" )
        self.title( "MyFirstMenu V1.0" )

    def createMenuBar(self):
        menuBar = Menu(self)

        menuFile = Menu(menuBar, tearoff=0)
        menuFile.add_command(label="Open from file", command=self.openFile)
        menuFile.add_command(label="Open from directory", command=self.openDirectory)
        menuFile.add_command(label="Save", command=self.doSomething)
        menuFile.add_separator()
        menuFile.add_command(label="Exit", command=self.quit)
        menuBar.add_cascade( label="File", menu=menuFile)

        menuEdit = Menu(menuBar, tearoff=0)
        menuEdit.add_command(label="Undo", command=self.doSomething)
        menuEdit.add_separator()
        menuEdit.add_command(label="Copy", command=self.doSomething)
        menuEdit.add_command(label="Cut", command=self.doSomething)
        menuEdit.add_command(label="Paste", command=self.doSomething)
        menuBar.add_cascade( label="Edit", menu=menuEdit)

        menuHelp = Menu(menuBar, tearoff=0)
        menuHelp.add_command(label="About", command=self.doAbout)
        menuBar.add_cascade( label="Help", menu=menuHelp)

        self.config(menu = menuBar)

    def openDirectory(self):
        directory = tk.filedialog.askdirectory()
        self.text.set("Directory load : " +  directory)
        self.path = directory

    def openFile(self):
        file = askopenfilename(title="Choose the file to open",
                filetypes=[("All files", ".*")])
        self.text.set("File load : " +  file)
        self.path = file

    def analyse(self):
        coordinator = Coordinatore(self.path)
        coordinator.analyze()
        print("")
    def doSomething(self):
        print("Menu clicked")

    def doAbout(self):
        messagebox.showinfo("My title", "My message")


window = MyWindow()
window.mainloop()
