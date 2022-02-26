from Windows.Audio_window import Audio_recorder
from Windows.Photo_window import Photo_taker
import tkinter as tk


if __name__ == "__main__":
    # add begining explanation window
    print("-----------------Loading photo application---------------------")
    #Photo_taker(tk.Tk(),'Take Happy Photo 0/10', False)
    Audio_recorder(tk.Tk(), 'Audio Recorder', False)

    # design of both, add redo, add text to image and add test feature to audio
    # comment code