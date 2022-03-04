import tkinter as tk
from PIL import ImageTk, Image
from pydub import AudioSegment
from pydub.playback import play


class Result():
    def __init__(self, window, audio_results, image_results, combined_results, pos=0, again=True):
        self.root = window
        self.root.geometry("640x600")
        self.root.resizable(False, False)
        self.root.title("Results")
        self.root.configure(background="#4a4a4a")

        self.audio_r = audio_results
        self.image_r = image_results
        self.com_r = combined_results
        self.pos = pos
        self.again = again

        # Image
        self.canv = tk.Canvas(master=self.root)
        self.img = ImageTk.PhotoImage(Image.open("App_Data/Test/Preprocessed/Image/test.jpg"))

        # Audio
        self.play = tk.Button(self.root, font="arial 20", text="Play",bg="#dc143c",fg="white",border=0,command=self.play_sample)

        # Combine
        scoresText = "Happy: "+str(int((self.com_r[0]*100)/2))+"% Neutral: "+str(int((self.com_r[1]*100)/2))+"% Sad: "+str(int((self.com_r[2]*100)/2))+"%"
        self.scores = tk.Label(text=f"{scoresText}", font="arial 20",width=38,background="#4a4a4a",fg="white")

        # Name
        self.title = tk.Label(text="Emotion Predicion Results", font="arial 30 bold", background="#4a4a4a", fg="white").pack()

        # Button
        self.nextB = tk.Button(self.root, font="arial 20", text="Next",bg="#111111",fg="white",border=0,command=self.next).place(x=510, y=430)
        self.backB = tk.Button(self.root, font="arial 20", text="Back",bg="#111111",fg="white",border=0,command=self.back).place(x=15, y=430)
        self.againB = tk.Button(self.root, font="arial 20", text="Again",bg="#C1E1C1",fg="black",border=0,command=self.again).place(x=260, y=430)
        self.answerB = tk.Button(self.root, font="arial 20", text="Answer Question",bg="#A7C7E7",fg="black",border=0,command=self.answer).place(x=195, y=365)

        self.content()

        self.root.mainloop()

    def next(self):
        if self.pos >= 0 and self.pos < 2:
            self.pos += 1
            self.content()

    def back(self):
        if self.pos > 0 and self.pos <= 2:
            self.pos -= 1
            self.content()

    def again(self):
        pass #! make it loop if true
    
    def play_sample(self):
        song = AudioSegment.from_wav("App_Data/Test/Preprocessed/Audio/test.wav")
        play(song)

    def answer(self):
        self.pos = -1
        self.content()

    def emotion(self, happy, neutral, sad):
        result = ""
        if happy > neutral and happy > sad:
            result += "Happy"
        elif neutral > happy and neutral > sad:
            result += "Neutral"
        elif sad > happy and sad > neutral:
            result += "Sad"
        elif happy > neutral and happy == sad:
            result += "Happy or Sad"
        elif happy > sad and happy == neutral:
            result += "Happy or Neutral"
        elif neutral > happy and neutral == sad:
            result += "Neutral or Sad"
        else:
            result += "Happy or Neutral or Sad"
        return result
    
    def content(self):
        type = ""
        if self.pos == 0:
            self.canv.place(x=280, y=250, width=48, height=48)
            type = "IMAGE"
            prediction = str(int(max(self.image_r)*100))+"%"+" "+self.emotion(self.image_r[0], self.image_r[1], self.image_r[2])
            self.canv.create_image(0, 0, image=self.img, anchor='nw')
            self.play.place(x=600, y=500)
        elif self.pos == 1:
            self.canv.place(x=600, y=500, width=48, height=48)
            type = "AUDIO"
            prediction = str(int(max(self.audio_r)*100))+"%"+" "+self.emotion(self.audio_r[0], self.audio_r[1], self.audio_r[2])
            self.scores.place(x=600, y=500)
            self.play.place(x=280, y=250)
        elif self.pos == 2:
            type = "COMBINED"
            prediction = str(int((max(self.com_r)*100)/2))+"%"+" "+self.emotion(self.com_r[0], self.com_r[1], self.com_r[2])
            self.scores.place(x=10, y=250)
            self.play.place(x=600, y=500)
        elif self.pos == -1:
            self.scores.place(x=600, y=500)
            self.canv.place(x=600, y=500, width=48, height=48)
            self.play.place(x=600, y=500)
      
        if self.pos >= 0 and self.pos <= 2:
            text = "The "+type+" emotion prediction is: "
            text1 = tk.Label(text=f"{text}", font="arial 15",width=50,background="#4a4a4a",fg="white")
            text1.place(x=30, y=100)
            text2 = tk.Label(text=f"{prediction}", font="arial 20",width=35,background="#4a4a4a",fg="#98FB98")
            text2.place(x=30, y=150)