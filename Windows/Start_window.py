import tkinter as tk


class Start():
    def __init__(self, window, window_title, pos=0, samples=10):
        self.root = window
        self.root.title(window_title)
        self.pos = pos

        self.root.geometry("640x600")
        self.root.resizable(False, False)
        self.root.configure(background="#4a4a4a")

        self.samples = samples

        #Name
        tk.Label(text="Emotion Chatbot", font="arial 30 bold", background="#4a4a4a", fg="white").pack(pady=20)

        #Button
        self.nextB = tk.Button(self.root, font="arial 20", text="Next",bg="#C1E1C1",fg="black",border=0,command=self.next)
        self.nextB.place(x=280, y=450)
        self.submitB = tk.Button(self.root, font="arial 20", text="Submit",bg="#FFFF00",fg="black",border=0,command=self.submit)
        self.submitB.place(x=640, y=600)

        # TextBox Creation
        self.inputtxt = tk.Text(self.root, height = 1, width = 20)
        self.inputtxt.place(x=640, y=600)

        self.sentence()
        self.root.mainloop()
    
    def __repr__(self):
        return str(self.samples)

    def next(self):
        self.pos += 1
        self.sentence()

    def submit(self):
        try:
            self.samples = int(self.inputtxt.get(1.0, "end-1c"))
            if (self.samples > 1 and self.samples < 11):
                self.next()
            else:
                tk.Label(text="Make sure you enter a number between 2 and 10.", font="arial 15",width=58,background="#DC143C",fg="white").place(x=0, y=520)
        except:
            tk.Label(text="Make sure you enter a number", font="arial 15",width=58,background="#DC143C",fg="white").place(x=0, y=520)
            
    def sentence(self):
        text = ""
        if self.pos == 0:
            text += "This application uses image and audio data to predict your emotion.\n"
            text += "It does this by gathering samples from you, these samples are then\n"
            text += "agumented to create a larger set of data. This data is then used to\n"
            text += "train two deep learing CNN models. The program will ask you to take\n"
            text += "one more photo and recording. This data will be passed through the\n"
            text += "models provding a prediction surrounding your emotion. A chatbot will\n"
            text += "reply to your question with consideration to the emotion predicted by\n"
            text += "the CNN model.\n"
        elif self.pos == 1:
            text += "The photo application works by having you take 2-10 pictures of youself\n"
            text += "in a happy/neutral/sad expression. You will be told on screen which\n"
            text += "emotion to show. Each time you click 'snapshot' a photo will be\n"
            text += "taken, this will increment the counter, onces you have done 2-10 photos\n"
            text += "it will promote you with the next emotiont. Try keep your face in the\n"
            text += "center of the screen and move it very each timr slightly at different\n"
            text += "angles. Make sure both eyes are visible on screen. If you make a\n"
            text += "mistake you can always re-take by clicking the 're-take' button.\n"
        elif self.pos == 2:
            text += "The recording application will get you to say 2-10 questions. These\n"
            text += "questions are the ones this chatbot knows. The CNN will use the change\n"
            text += "in your tone to determine your emotion later on so try to exagerate a\n"
            text += "change for each emotion. To take a sample press the 'record' button,\n"
            text += "this will then sample the following 4 seconds. It will count down the\n"
            text += "4 seconds on screen so try to say the question in time. If you say it\n"
            text += "wrong or not in time, you can press the 're-take' button. After each\n"
            text += "sample it will be played back to you so you can determine if its good.\n"
        elif self.pos == 3:
            text += "The application will then process your data and train the models.\n"
            text += "It will then promote you for one more picture and a question in\n"
            text += "the same emotion. This data will be predicted upon before shwoing a\n"
            text += "results window which will show you the emotion it has predicted and\n"
            text += "what percentage of confidence it has for the image, audio and combined\n"
            text += "results. Once this window is closed a reply from the chatbot will show.\n"
            text += "This reply should not only answer the question, but should be directly\n"
            text += "responsive to your emotion.\n"
        elif self.pos == 4:
            text += "If you have trained the models, you can then continuously try asking\n"
            text += "the chatbot new questions in different emotions by clicking the 'ask\n"
            text += "again' button, shown on the response page.\n"
            text += "\n"
            text += "\n"
            text += "\n"
            text += "\n"
            text += "Enter the number of samples you want to take. (2-10)\n"
            self.nextB.place(x=640, y=600)
            self.submitB.place(x=270, y=450)
            self.inputtxt.place(x=240, y=380)
        else:
            self.root.destroy()

        if self.pos < 5:
            tk.Label(text=f"{text}", font="arial 15",width=58,background="#4a4a4a",fg="white").place(x=0, y=150)
