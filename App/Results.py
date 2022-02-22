from tkinter import *
from PIL import ImageTk, Image
from playsound import playsound


class ResultApp:
    def __init__(self, window, audio_results, image_results, combined_results, pos=0):
        self.root = window
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        self.root.title("Voice Recorder")
        self.root.configure(background="#4a4a4a")

        self.audio_r = audio_results
        self.image_r = image_results
        self.com_r = combined_results
        # self.ATH = audio_training_history
        # self.ITH = image_training_history
        self.pos = pos

        #Image
        self.canv = Canvas(master=self.root)
        self.canv.place(x=276, y=250, width=48, height=48)
        self.img = ImageTk.PhotoImage(Image.open("App/PreprocessedTest/Image/test.jpg"))

        #Name
        Label(text="Emotion Predicion Results", font="arial 30 bold", background="#4a4a4a", fg="white").pack()

        #Button
        self.record = Button(self.root, font="arial 20", text="Next",bg="#111111",fg="white",border=0,command=self.slide).place(x=500, y=430)

        self.content()

        self.root.mainloop()

    def play(self):
        playsound('1.mp3')

    def slide(self):
        self.pos += 1
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
            type = "IMAGE"
            prediction = str(int(max(self.image_r)*100))+"%"+" "+self.emotion(self.image_r[0], self.image_r[1], self.image_r[2])
              
            self.canv.create_image(0, 0, image=self.img, anchor='nw')
        elif self.pos == 1:
            type = "AUDIO"
            prediction = str(int(max(self.audio_r)*100))+"%"+" "+self.emotion(self.audio_r[0], self.audio_r[1], self.audio_r[2])
        elif self.pos == 2:
            type = "COMBINED"
            prediction = str(int((max(self.com_r)*100)/2))+"%"+" "+self.emotion(self.com_r[0], self.com_r[1], self.com_r[2])
      
        if self.pos < 3:
            text = "The "+type+" emotion prediction is: "
            Label(text=f"{text}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=30, y=100)
            Label(text=f"{prediction}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=30, y=150)
        else:
            self.root.destroy()

  