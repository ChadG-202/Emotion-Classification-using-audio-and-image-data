from tkinter import *
from PIL import ImageTk, Image
import pygame
from pydub import AudioSegment


class ResultApp:
    def __init__(self, window, audio_results, image_results, combined_results, pos=0):
        self.root = window
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        self.root.title("Results")
        self.root.configure(background="#4a4a4a")

        self.audio_r = audio_results
        self.image_r = image_results
        self.com_r = combined_results
        # self.ATH = audio_training_history
        # self.ITH = image_training_history
        self.pos = pos
        self.again = False

        pygame.mixer.init()

        #Image
        self.canv = Canvas(master=self.root)
        self.img = ImageTk.PhotoImage(Image.open("App/PreprocessedTest/Image/test.jpg"))

        #audio
        self.audio_canv = Canvas(master=self.root, background="#4a4a4a")
        self.audio_img = ImageTk.PhotoImage(Image.open("App/AppImages/audio-waves.png"))

        #Combine
        scoresText = "Happy: "+str(int((self.com_r[0]*100)/2))+"% Neutral: "+str(int((self.com_r[1]*100)/2))+"% Sad: "+str(int((self.com_r[2]*100)/2))+"%"
        self.scores = Label(text=f"{scoresText}", font="arial 20",width=38,background="#4a4a4a",fg="white")

        #Name
        self.title = Label(text="Emotion Predicion Results", font="arial 30 bold", background="#4a4a4a", fg="white").pack()

        #Button
        self.nextB = Button(self.root, font="arial 20", text="Next",bg="#111111",fg="white",border=0,command=self.next).place(x=510, y=430)
        self.backB = Button(self.root, font="arial 20", text="Back",bg="#111111",fg="white",border=0,command=self.back).place(x=15, y=430)
        self.againB = Button(self.root, font="arial 20", text="Again",bg="#C1E1C1",fg="black",border=0,command=self.again).place(x=260, y=430)
        self.answerB = Button(self.root, font="arial 20", text="Answer Question",bg="#A7C7E7",fg="black",border=0,command=self.answer).place(x=195, y=365)

        self.content()

        self.root.mainloop()

    def next(self):
        if self.pos < 2:
            self.pos += 1
            self.content()

    def back(self):
        if self.pos > 0:
            self.pos -= 1
            self.content()

    def again(self):
        pass

    def answer(self):
        self.pos = 4
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
            self.audio_canv.place(x=600, y=500, width=128, height=128)
        elif self.pos == 1:
            self.audio_canv.place(x=236, y=216, width=128, height=128)
            self.audio_canv.create_image(0, 0, image=self.audio_img, anchor='nw')
            self.canv.place(x=600, y=500, width=48, height=48)
            type = "AUDIO"
            prediction = str(int(max(self.audio_r)*100))+"%"+" "+self.emotion(self.audio_r[0], self.audio_r[1], self.audio_r[2])
            self.scores.place(x=600, y=500)
        elif self.pos == 2:
            type = "COMBINED"
            prediction = str(int((max(self.com_r)*100)/2))+"%"+" "+self.emotion(self.com_r[0], self.com_r[1], self.com_r[2])
            self.scores.place(x=10, y=250)
            self.audio_canv.place(x=600, y=500, width=128, height=128)
        elif self.pos == 4:
            self.scores.place(x=600, y=500)
            self.canv.place(x=600, y=500, width=48, height=48)
            self.audio_canv.place(x=600, y=500, width=128, height=128)
      
        if self.pos < 3:
            text = "The "+type+" emotion prediction is: "
            text1 = Label(text=f"{text}", font="arial 15",width=50,background="#4a4a4a",fg="white")
            text1.place(x=30, y=100)
            text2 = Label(text=f"{prediction}", font="arial 20",width=35,background="#4a4a4a",fg="white")
            text2.place(x=30, y=150)

  