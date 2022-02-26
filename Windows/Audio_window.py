import random
from turtle import position
import sounddevice as sound
from scipy.io.wavfile import write
import time
import tkinter as tk

class Audio_recorder:
    def __init__(self, window, window_title, test_set, pos=0, emotion='Happy'):
        self.root = window
        self.root.title(window_title)
        self.test_set = test_set
        self.pos = pos
        self.emotion = emotion

        self.root.geometry("640x600")
        self.root.resizable(False, False)
        self.root.title("Voice Recorder")
        self.root.configure(background="#4a4a4a")

        #Icon
        self.image_icon=tk.PhotoImage(file="App_Images/rec-button.png")
        self.root.iconphoto(False, self.image_icon)

        #Logo
        self.photo=tk.PhotoImage(file="App_Images/rec-button.png")
        self.myimage=tk.Label(image=self.photo,background="#4a4a4a")
        self.myimage.pack(pady=30)

        #Name
        tk.Label(text="Voice Recorder", font="arial 30 bold", background="#4a4a4a", fg="white").pack()

        #Button
        self.record = tk.Button(self.root, font="arial 20", text="Record",bg="#111111",fg="white",border=0,command=self.Record).pack(pady=30)

        self.sentence()

        self.root.mainloop()

    def Recording(self, type, path):
        freq=22050
        dur=4
        channel=2
        recording=sound.rec(dur*freq, samplerate=freq,channels=channel)

        #Timer
        timeLeft=dur
        while timeLeft>0:
            timeLeft-=1
            tk.Label(text=f"{str(timeLeft)}", font="arial 40",width=4,background="#4a4a4a").place(x=240, y=420)
            self.root.update()
            time.sleep(1)

        sound.wait()
        if self.test_set:
            write("App_Data/Test/Preprocessed/Audio/test.wav",freq,recording)
        else:
            write("App_Data/Training/Raw/Audio/"+type+"/"+path+".wav",freq,recording)
            write("App_Data/Training/Preprocessed/Audio/"+type+"/"+path+".wav",freq,recording)

        self.pos +=1
        self.sentence()

    def Record(self):
        self.Recording(self.emotion, str(self.pos))
    
    def sentence(self):
        questions = ["'Can you help me?'", "'Who are you?'",
        "'Where am I?'", "'Why is this happening?'", "'What time is it?'", 
        "'Take me home?'", "'Whats the weather today?'", "'Set a timer for 10 minutes?'",
        "'How do you spell tree?'", "'Whats 10 + 20?'"]

        tempPos = self.pos

        if self.test_set:
            if self.pos < 1:
                sen = random.randint(0, 9)
                text = "Say the phrase: "+questions[sen]+" in chosen emotion."
                tk.Label(text=f"{text}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=20, y=350)
            else:
                self.root.destroy()
        else:
            if self.pos > 9 and self.pos < 20:
                tempPos = self.pos - 10
                self.emotion = "Neutral"
            elif self.pos > 19 and self.pos < 30:
                tempPos = self.pos - 20
                self.emotion = "Sad"

            if self.pos < 30:
                ask = "Say the phrase: "+questions[tempPos]+" in a "+self.emotion+" voice."
                tk.Label(text=f"{ask}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=20, y=350)
                position = self.emotion+": "+str(tempPos)+"/10"
                tk.Label(text=f"{position}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=20, y=380)
            else:
                self.root.destroy()