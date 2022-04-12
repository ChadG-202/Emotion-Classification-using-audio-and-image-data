import threading
from gtts import gTTS
from PIL import ImageTk, Image
import datetime
import os
import speech_recognition as sr
import tkinter as tk

from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
from Windows.Structure_window import Structure

'''
Tkinter window to display confidence 
results, and give access to chatbot reply.
'''
class Result(Structure):
    def __init__(self, window, window_title, audio_results, image_results, combined_results):
        Structure.__init__(self, window, window_title)

        self.audio_r = audio_results          # Array of confidence scores for the audio data
        self.image_r = image_results          # Array of confidence scores for the image data
        self.com_r = combined_results         # Array of confidence scores for the combined data
        self.again = "n"                      # Repeat tests?
        self.question = ""                    # Asked question

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
    
    # Return again status
    def __repr__(self):
        return self.again

    # Increment screen state
    def next(self):
        if self.pos >= 0 and self.pos < 2:
            self.pos += 1
            self.content()

    # Decrement screen state
    def back(self):
        if self.pos > 0 and self.pos <= 2:
            self.pos -= 1
            self.content()

    # Test again
    def test_again(self):
        self.again = "y"
        self.root.destroy()
    
    # Play audio
    def play_sample(self):
        def playing():
            sample = AudioSegment.from_wav("App_Data/Test/Preprocessed/Audio/test.wav")
            play(sample)
        t1 = threading.Thread(target=playing)
        t1.start()

    # Set answer question state
    def answer(self):
        self.pos = -1
        self.content()

    # Determine emotion from confidence scores
    def emotion(self, happy, neutral, sad):
        result = ""
        if happy > neutral and happy > sad:
            result += "Happy"
        elif neutral > happy and neutral > sad:
            result += "Neutral"
        elif sad > happy and sad > neutral:
            result += "Sad"
        elif neutral > happy and neutral == sad:
            result += "Neutral" #! Neutral biast
        else:
            result += "Happy" #! Happy biast
        return result

    # Chatbot reply
    def get_reply(self):
        question = self.question 
        emotion = self.emotion(self.com_r[0], self.com_r[1], self.com_r[2])
        reply = ""
        if question == "can you help me":
            if emotion == "Happy":
                reply = "Yes, i'm always in the mood to help you"
            elif emotion == "Sad":
                reply = "Are you alright, what do you need?"
            else:
                reply = "What do you need?"

        elif question == "what is the weather today":
            if emotion == "Happy":
                reply = "The weather is like your mood sunny with a high of 12 degrees"
            elif emotion == "Sad":
                reply = "Hopefully your mood will be lifted as its sunny todays with a high of 12 degrees."
            else:
                reply = "It looks to be sunny with a high of 12 degrees."

        elif question == "can you find me a route home":
            if emotion == "Happy":
                reply = "Of course finding the best possible routes for you to get home."
            elif emotion == "Sad":
                reply = "Working as quick as possible to find you routes home. Do you need emergency services?"
            else:
                reply = "Finding routes to home."

        elif question == "what is there to watch on Netflix":
            if emotion == "Happy":
                reply = "The comedy meet the parents is currently on netflix. Would you like me to play it?"
            elif emotion == "Sad":
                reply = "I suggest the fault in our stars. Would you like me to play it?"
            else:
                reply = "There is a range of movies from the hustle, king kong and justice league. Would you like me to open the app so you can decide?"

        elif question == "what time is it":
            currentDT = datetime.datetime.now()
            if emotion == "Happy":
                reply = "It's {} {}. I hope you remain this happy for the rest of the day.".format(str(currentDT.hour), str(currentDT.minute))
            elif emotion == "Sad":
                reply = "I'm sure you can make it on time its {} {}.".format(str(currentDT.hour), str(currentDT.minute))
            else:
                reply = "It is {} {}.".format(str(currentDT.hour), str(currentDT.minute))

        elif question == "can you turn the light on":
            if emotion == "Happy":
                reply = "Let match this room to your mood, turning the light on."
            elif emotion == "Sad":
                reply = "Of course, hopefully this can lighten your day."
            else:
                reply = "Turning the light on."

        elif question == "what song is this":
            if emotion == "Happy":
                reply = "This feature doesnt exist, we know you want this feature so we are working our best to get it to you."
            elif emotion == "Sad":
                reply = "I'm sorry but this feature cannot yet be used. We are working hard to get it implimented."
            else:
                reply = "This feature does not yet exist."

        elif question == "who am i":
            if emotion == "Happy":
                reply = "You are warm, pleasant human being"
            elif emotion == "Sad":
                reply = "Don't be upset, you are an amazing human being"
            else:
                reply = "you are a human being"
                
        elif question == "are unicorns real":
            if emotion == "Happy":
                reply = "Of course unicorns are real."
            elif emotion == "Sad":
                reply = "Don't be sad, there's no evidence to suggest unicorns dont exist."
            else:
                reply = "There is currently no evidence of the existence of unicorns."

        elif question == "how do you spell tree":
            if emotion == "Happy":
                reply = "Good question, tree is spelt t r e e."
            elif emotion == "Sad":
                reply = "It's okay im here to help with your spelling, tree is spelt t r e e."
            else:
                reply = "Tree is spelt t r e e"
        else:
            reply = "Unable to match your question. Try again."
        
        tk.Label(text=reply, font="arial 12",width=70,background="#4a4a4a",fg="red").place(x=0, y=450)

        try:
            os.remove("Chatbot/bot_reply.mp3")
        except:
            print("no chatbot reply to remove")
        # Convert text to speech
        bot_reply = gTTS(text=reply, lang="en", slow=False) 
        # Save speech
        bot_reply.save("Chatbot/bot_reply.mp3")
        # Play speech
        tk.Button(self.root, font="arial 20", text="Play Again",bg="#DC143C",fg="white",border=0,command=self.play_again).place(x=255, y=330)
        playsound("Chatbot/bot_reply.mp3")

    def play_again(self):
        def play():
            playsound("Chatbot/bot_reply.mp3")
        t1 = threading.Thread(target=play)
        t1.start()
    
    # Chatbot state
    def Chatbot(self):
        r = sr.Recognizer()
        
        def get_question_asked():
            with sr.AudioFile("App_Data/Test/Preprocessed/Audio/test.wav") as source:
                audio = r.record(source)
            try:
                self.question = r.recognize_google(audio)
                tk.Label(text="You asked: "+self.question, font="arial 20",width=40,background="#4a4a4a",fg="white").place(x=0, y=150)
            except Exception as e:
                print("Exception: "+str(e))

            self.get_reply()

        t1 = threading.Thread(target=get_question_asked)
        t1.start()

        tk.Label(text="Chatbot should reply based on your question and emotion.", font="arial 15",width=58,background="#4a4a4a",fg="#98FB98").place(x=0, y=230)
        tk.Label(text="Turn sound on to hear reply!", font="arial 12",width=71,background="#4a4a4a",fg="#98FB98").place(x=0, y=280)

        tk.Label(text="Loading...", font="arial 20",width=40,background="#4a4a4a",fg="white").place(x=0, y=330)
        tk.Label(text="Chatbot Reply:", font="arial 20",width=40,background="#4a4a4a",fg="red").place(x=0, y=400)
        
        tk.Button(self.root, font="arial 20", text="Again",bg="#C1E1C1",fg="black",border=0,command=self.test_again).place(x=280, y=530)

    # Clear window items
    def Clear(self):
        self.canv.place(x=640, y=600)
        self.scores.place(x=640, y=600)
        self.play.place(x=640, y=600)
        self.backB.place(x=640, y=600)
        self.nextB.place(x=640, y=600)
    
    # Main content to be displayed
    def content(self):
        type = ""
        prediction = ""
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

            self.answerB.place(x=640, y=600)
            self.label1.place(x=640, y=600)
            self.label2.place(x=640, y=600)
            self.Chatbot()
      
        if self.pos >= 0 and self.pos <= 2:
            text = "The "+type+" emotion prediction is: "
            self.text1.set(text)
            self.text2.set(prediction)