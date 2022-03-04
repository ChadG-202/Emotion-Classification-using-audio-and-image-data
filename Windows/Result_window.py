import tkinter as tk
from PIL import ImageTk, Image
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr


class Result():
    def __init__(self, window, audio_results, image_results, combined_results, pos=0, again="n"):
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
        self.scores = tk.Label(text=f"{scoresText}", font="arial 20",width=40,background="#4a4a4a",fg="white")

        # Name
        self.title = tk.Label(text="Emotion Predicion Results", font="arial 30 bold", background="#4a4a4a", fg="white").pack()

        # Button
        self.nextB = tk.Button(self.root, font="arial 20", text="Next",bg="#111111",fg="white",border=0,command=self.next)
        self.nextB.place(x=550, y=530)
        self.backB = tk.Button(self.root, font="arial 20", text="Back",bg="#111111",fg="white",border=0,command=self.back)
        self.backB.place(x=15, y=530)
        self.againB = tk.Button(self.root, font="arial 20", text="Again",bg="#C1E1C1",fg="black",border=0,command=self.test_again)
        self.againB.place(x=640, y=600)
        self.answerB = tk.Button(self.root, font="arial 20", text="Answer Question",bg="#A7C7E7",fg="black",border=0,command=self.answer)
        self.answerB.place(x=205, y=530)

        # Text
        self.text1 = tk.StringVar()
        self.label1 = tk.Label(textvariable=self.text1, font="arial 15",width=60,background="#4a4a4a",fg="white")
        self.label1.place(x=0, y=150)
        self.text2 = tk.StringVar()
        self.label2 = tk.Label(textvariable=self.text2, font="arial 20",width=40,background="#4a4a4a",fg="#98FB98")
        self.label2.place(x=0, y=200)

        self.content()

        self.root.mainloop()
    
    def __repr__(self):
        return self.again

    def next(self):
        if self.pos >= 0 and self.pos < 2:
            self.pos += 1
            self.content()

    def back(self):
        if self.pos > 0 and self.pos <= 2:
            self.pos -= 1
            self.content()

    def test_again(self):
        self.again = "y"
        self.root.destroy()
    
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

    def Get_reply(self):
        predicted_emotion = self.emotion(self.com_r[0], self.com_r[1], self.com_r[2])
        return "this would be the reply because you are "+predicted_emotion
    
    def Chatbot(self):
        r = sr.Recognizer()
        audio = False
        with sr.AudioFile("App_Data/Test/Preprocessed/Audio/test.wav") as source:
            audio = r.record(source)
        try:
            s = r.recognize_google(audio)
            tk.Label(text="You asked: "+s, font="arial 20",width=40,background="#4a4a4a",fg="white").place(x=0, y=150)
        except Exception as e:
            print("Exception: "+str(e))
        reply = self.Get_reply()
        tk.Label(text="Chatbot reply: "+reply, font="arial 12",width=60,background="#4a4a4a",fg="#98FB98").place(x=0, y=200)

    def Clear(self):
        self.canv.place(x=640, y=600)
        self.scores.place(x=640, y=600)
        self.play.place(x=640, y=600)
        self.backB.place(x=640, y=600)
        self.nextB.place(x=640, y=600)
    
    def content(self):
        type = ""
        if self.pos == 0:
            self.Clear()

            type = "IMAGE"
            prediction = str(int(max(self.image_r)*100))+"%"+" "+self.emotion(self.image_r[0], self.image_r[1], self.image_r[2])

            self.canv.place(x=296, y=300, width=48, height=48)
            self.canv.create_image(0, 0, image=self.img, anchor='nw')
            self.nextB.place(x=550, y=530)

        elif self.pos == 1:
            self.Clear()
            
            type = "AUDIO"
            prediction = str(int(max(self.audio_r)*100))+"%"+" "+self.emotion(self.audio_r[0], self.audio_r[1], self.audio_r[2])


            self.play.place(x=280, y=300)
            self.backB.place(x=15, y=530)
            self.nextB.place(x=550, y=530)

        elif self.pos == 2:
            self.Clear()

            type = "COMBINED"
            prediction = str(int((max(self.com_r)*100)/2))+"%"+" "+self.emotion(self.com_r[0], self.com_r[1], self.com_r[2])

            self.scores.place(x=10, y=300)
            self.backB.place(x=15, y=530)

        elif self.pos == -1:
            self.Clear()
            self.againB.place(x=280, y=530)
            self.answerB.place(x=640, y=600)
            self.label1.place(x=640, y=600)
            self.label2.place(x=640, y=600)
            self.Chatbot()
      
        if self.pos >= 0 and self.pos <= 2:
            text = "The "+type+" emotion prediction is: "
            self.text1.set(text)
            self.text2.set(prediction)
            
