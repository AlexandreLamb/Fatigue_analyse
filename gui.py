
import math
import tkinter as tk
import tkinter.ttk as ttk
from coordinator import Coordinatore

class MainWindows(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.createMenuBar()
        self.text = tk.StringVar()
        self.text.set("")
        self.label = tk.Label(self, textvariable=self.text)
        self.label.pack()
        self.button = tk.Button(self,text="Analyse", command=self.analyse)
        self.button.pack()
        self.progressBar = ttk.Progressbar(self,length = 100, mode = 'determinate' )
        self.progressBar.pack()
        # Fill the content of the window
        self.path = None
        self.attributes("-fullscreen", True)
        self.title( "MyFirstMenu V1.0" )

    def createMenuBar(self):
        menuBar = tk.Menu(self)
        menuFile = tk.Menu(menuBar, tearoff=0)
        menuFile.add_command(label="Open from file", command=self.openFile)
        menuFile.add_command(label="Open from directory", command=self.openDirectory)
        menuFile.add_separator()
        menuFile.add_command(label="Exit", command=self.quit)
        menuBar.add_cascade( label="File", menu=menuFile)
        self.config(menu = menuBar)

    def openDirectory(self):
        directory = tk.filedialog.askdirectory()
        self.text.set("Directory load : " +  directory)
        self.path = directory

    def openFile(self):
        file = tk.filedialog.askopenfilename(title="Choose the file to open",
                filetypes=[("All files", ".*")])
        self.text.set("File load : " +  file)
        self.path = file

    def analyse(self):
        coordinator = Coordinatore(self.path)
        coordinator.create_csv_landmarks()

window = MainWindows()
window.mainloop()
