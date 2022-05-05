import tkinter as tk

from Windows.Structure_window import Structure

'''
Tkinter window that explains how the program works.
It also allows the user to input the number of samples
to be taken.
'''
class Start(Structure):
    def __init__(self, window, window_title, samples=10):
        Structure.__init__(self, window, window_title)

        self.samples = samples # Number of pictures/recordings to be taken

        # Name
        tk.Label(text="Emotion Chatbot", font="arial 30 bold", background="#4a4a4a", fg="white").pack(pady=20)

        # Buttons
        self.nextB = tk.Button(self.root, font="arial 20", text="Next",bg="#C1E1C1",fg="black",border=0,command=self.next)
        self.nextB.place(x=280, y=450)
        self.submitB = tk.Button(self.root, font="arial 20", text="Submit",bg="#FFFF00",fg="black",border=0,command=self.submit)
        self.submitB.place(x=640, y=600)
        self.testB = tk.Button(self.root, font="arial 20", text="Test",bg="#ADD8E6",fg="black",border=0,command=self.test)
        self.testB.place(x=640, y=600)

        # TextBox Creation
        self.inputtxt = tk.Text(self.root, height = 1, width = 20)
        self.inputtxt.place(x=640, y=600)

        self.sentence()
        self.root.mainloop()
    
    # Return sample number
    def __repr__(self):
        return str(self.samples)
    
    # Use test model
    def test(self):
        self.samples = -1 # set sample to -1 to tell program this is in test mode
        self.pos = 5      # finish window by setting to pos 5
        self.sentence()

    # Next page
    def next(self):
        self.pos += 1
        self.sentence()

    # Submitting sample number
    def submit(self):
        try: # validate for number
            self.samples = int(self.inputtxt.get(1.0, "end-1c"))
            if (self.samples > 1 and self.samples < 11): # validate that its in range 2-10
                self.next()
            else:
                tk.Label(text="Make sure you enter a number between 2 and 10.", font="arial 15",width=58,background="#DC143C",fg="white").place(x=0, y=520)
        except:
            tk.Label(text="Make sure you enter a number", font="arial 15",width=58,background="#DC143C",fg="white").place(x=0, y=520)

    # displayed  
    def sentence(self):
        text = ""
        if self.pos == 0:
            text += "This application uses image and audio data to predict emotion.\n"
            text += "It does this by gathering samples from you, these samples are\n"
            text += "agumented to create a larger set of data. This data can be used to\n"
            text += "train two deep learing CNN models. The program, will ask you to take\n"
            text += "one more photo and recording. This data will be passed through the\n"
            text += "models providing a emotion prediction. The chatbot will then\n"
            text += "reply to your question with consideration to the emotion found by\n"
            text += "the CNN model.\n"
            self.testB.place(x=280, y=525)
        elif self.pos == 1:
            text += "The photo application works by taking 2-10 pictures\n"
            text += "in a happy/neutral/sad expression. You will be told on screen which\n"
            text += "emotion to show. Each time you click 'snapshot' a photo will be\n"
            text += "taken. Onces you have done 2-10 photos it will prompt you with \n"
            text += "the next emotion. Make sure your face is in the center of the screen.\n"
            text += "Try moving your face slightly each time to provide a range of angles.\n"
            text += "Make sure both eyes are visible on screen. If you make a\n"
            text += "mistake you can always re-take by clicking the 're-take' button.\n"
            self.testB.place(x=640, y=600)
        elif self.pos == 2:
            text += "The recording application will get you to say 2-10 questions. These\n"
            text += "questions are the ones this chatbot knows. The CNN will use the change\n"
            text += "in your tone to determine your emotion. Try to exagerate your emotions\n"
            text += "to help the CNN predict. To take a sample press the 'record' button,\n"
            text += "this will then sample the following 4 seconds. It will count down on the\n"
            text += "screen so try to say the question in time. If you say it wrong\n"
            text += "or not in time, you can press the 're-take' button. After each\n"
            text += "sample is taken, the recording will be played back for you to check.\n"
        elif self.pos == 3:
            text += "The data will then be preprocessed and processed, before\n"
            text += "training. You will be promoted to take one more picture and one more\n"
            text += "recording in a chosen emotion. This data will be predicted upon\n"
            text += "before the results window shows the emotion it has predicted and\n"
            text += "what percentage of confidence it has for the image, audio and combined\n"
            text += "data. Pressing the 'Chatbot reply' button will allow you to\n"
            text += "play the reply the chatbot has found to be most fitting the situation,\n"
            text += "it should be related to your emotion and question.\n"
        elif self.pos == 4:
            text += "If you have trained the models, you can then continuously ask\n"
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

        # Display text if in range 0-4
        if self.pos < 5:
            tk.Label(text=f"{text}", font="arial 15",width=58,background="#4a4a4a",fg="white").place(x=0, y=150)