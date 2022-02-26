from Windows.Audio_window import Audio_recorder
from Windows.Photo_window import Photo_taker
import tkinter as tk


if __name__ == "__main__":
    # add begining explanation window
    print("-----------------Loading photo application---------------------")
    Photo_taker(tk.Tk(),'Take Happy Photo 0/10', False)
    #Audio_recorder(tk.Tk(), 'Audio Recorder', True)

    # design of both, add redo, add text to image and add test feature to audio
    # try other audio methode to gather wave files
    # add crop into photo window so that if a face isnt detected the photo can be taken again
    # if augmented data doesnt equal 110 tne limit all to lowest one
    # comment code