from Windows.Audio_window import Audio_recorder
from Windows.Photo_window import Photo_taker
import tkinter as tk


if __name__ == "__main__":
    print("-----------------Loading photo application---------------------")
    Photo_taker(tk.Tk(),'Take Happy Photo', True)
    Audio_recorder(tk.Tk())