from PhotoApp import ImageApp
from AudioApp import RecorderApp
import tkinter as tk

if __name__ == "__main__":
    ImageApp(tk.Tk(),'Take Happy Photo')
    RecorderApp(tk.Tk())
