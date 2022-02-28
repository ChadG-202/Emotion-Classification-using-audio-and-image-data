from Windows.Result_window import Result
from Windows.Start_window import Start
from Windows.Audio_window import Audio_recorder
from Windows.Photo_window import Photo_taker
import tkinter as tk


if __name__ == "__main__":
    # Start(tk.Tk())
    # Photo_taker(tk.Tk(),'Take Happy Photo 1/10', False)
    # Audio_recorder(tk.Tk(), 'Audio Recorder', False)
    # Train()
    # Photo_taker(tk.Tk(),'Take Photo', True)
    Audio_recorder(tk.Tk(), 'Audio Recorder', True)
    # Predict()
    # Result(tk.Tk())
    # Reply_bot(tk.Tk())

    # if augmented data doesnt equal 110 then limit all to lowest one
    # comment code