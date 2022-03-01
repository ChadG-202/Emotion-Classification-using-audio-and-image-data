import tkinter as tk


class Start():
    def __init__(self, window, window_title, pos=0):
        self.root = window
        self.root.title(window_title)
        self.pos = pos

        self.root.geometry("640x600")
        self.root.resizable(False, False)
        self.root.configure(background="#4a4a4a")

        #Name
        tk.Label(text="Emotion Chatbot", font="arial 30 bold", background="#4a4a4a", fg="white").pack(pady=20)

        #Button
        self.nextB = tk.Button(self.root, font="arial 20", text="Next",bg="#C1E1C1",fg="black",border=0,command=self.next)
        self.nextB.place(x=280, y=450)

        self.sentence()
        self.root.mainloop()

    def next(self):
        self.pos += 1
        self.sentence()

    def sentence(self):
        text = ""
        if self.pos == 0:
            text1 = "This application uses image and audio data to predict your emotion."
            text2 = "It does this by gathering samples from you, these samples are then"
            text3 = "agumented to create a larger set of data. This data is then used to"
            text4 = "train two deep learing CNN models. The program will ask you to take"
            text5 = "one more photo and recording. This data will be passed through the"
            text6 = "models provding a prediction surrounding your emotion. A chatbot will"
            text7 = "reply to your question with consideration to the emotion predicted by"
            text8 = "the CNN model."
        elif self.pos == 1:
            text1 = "The photo application works by having you take 10 pictures of youself"
            text2 = "in a happy/neutral/sad expression. You will be told on screen which"
            text3 = "emotion to show. Each time you click 'snapshot' a photo will be"
            text4 = "taken, this will increment the counter, onces you have done 10 photos"
            text5 = "it will promote you with the next emotiont. Try keep your face in the"
            text6 = "center of the screen and move it very each timr slightly at different"
            text7 = "angles. Make sure both eyes are visible on screen. If you make a"
            text8 = "mistake you can always re-take by clicking the 're-take' button."
        elif self.pos == 2:
            text1 = "The recording application will get you to say 10 questions. These"
            text2 = "questions are the ones this chatbot knows. The CNN will use the change"
            text3 = "in your tone to determine your emotion later on so try to exagerate a"
            text4 = "change for each emotion. To take a sample press the 'record' button,"
            text5 = "this will then sample the following 4 seconds. It will count down the"
            text6 = "4 seconds on screen so try to say the question in time. If you say it"
            text7 = "wrong or not in time, you can press the 're-take' button. After each"
            text8 = "sample it will be played back to you so you can determine if its good."
        elif self.pos == 3:
            text1 = "The application will then process your data and train the models."
            text2 = "It will then promote you for one more picture and a question in"
            text3 = "the same emotion. This data will be predicted upon before shwoing a"
            text4 = "results window which will show you the emotion it has predicted and"
            text5 = "what percentage of confidence it has for the image, audio and combined"
            text6 = "results. Once this window is closed a reply from the chatbot will show."
            text7 = "This reply should not only answer the question, but should be directly"
            text8 = "responsive to your emotion."
        elif self.pos == 4:
            text1 = "If you have trained the models, you can then continuously try asking"
            text2 = "the chatbot new questions in different emotions by clicking the 'ask"
            text3 = "again' button, shown on the response page."
            text4 = ""
            text5 = ""
            text6 = ""
            text7 = ""
            text8 = ""
        else:
            self.root.destroy()

        if self.pos < 5:
            tk.Label(text=f"{text1}", font="arial 15",width=58,background="#4a4a4a",fg="white").place(x=0, y=150)
            tk.Label(text=f"{text2}", font="arial 15",width=58,background="#4a4a4a",fg="white").place(x=0, y=180)
            tk.Label(text=f"{text3}", font="arial 15",width=58,background="#4a4a4a",fg="white").place(x=0, y=210)
            tk.Label(text=f"{text4}", font="arial 15",width=58,background="#4a4a4a",fg="white").place(x=0, y=240)
            tk.Label(text=f"{text5}", font="arial 15",width=58,background="#4a4a4a",fg="white").place(x=0, y=270)
            tk.Label(text=f"{text6}", font="arial 15",width=58,background="#4a4a4a",fg="white").place(x=0, y=300)
            tk.Label(text=f"{text7}", font="arial 15",width=58,background="#4a4a4a",fg="white").place(x=0, y=330)
            tk.Label(text=f"{text8}", font="arial 15",width=58,background="#4a4a4a",fg="white").place(x=0, y=360)
