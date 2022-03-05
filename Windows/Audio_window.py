import random
import pyaudio
import wave
from pydub import AudioSegment
from pydub.playback import play
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
        self.record = tk.Button(self.root, font="arial 20", text="Record",bg="#C1E1C1",fg="black",border=0,command=self.Record).pack(pady=20)
        self.btn_retake=tk.Button(self.root, font="arial 20", text="Re-take", bg="#111111", fg="white", border=0, command=self.retake).pack(pady=10)

        self.sentence()

        self.root.mainloop()
    
    def retake(self):
        if self.pos > 0:
            if self.pos%10 == 0:
                if self.emotion == "Neutral":
                    self.emotion = "Happy"
                elif self.emotion == "Sad":
                    self.emotion == "Neutral"
            self.pos -= 1
            self.sentence()

    def Recording(self, type, pos):
        p = pyaudio.PyAudio()

        channels = 2
        rate = 22050
        chunk = 1024
        seconds = 4

        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

        frames = []

        for i in range(0, int(rate / chunk * seconds)):
            if i < 21:
                timeLeft = "3"
            elif i > 21 and i < 43:
                timeLeft = "2"
            elif i > 43 and i < 64:
                timeLeft = "1"
            elif i > 64:
                timeLeft = "0"
            tk.Label(text=f"{str(timeLeft)}", font="arial 40",width=4,background="#4a4a4a").place(x=260, y=520)
            self.root.update()
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
        
        if self.test_set:
            wf = wave.open("App_Data/Test/Preprocessed/Audio/test.wav", 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()
        else:
            wf = wave.open("App_Data/Training/Raw/Audio/"+type+"/"+pos+".wav", 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            wf2 = wave.open("App_Data/Training/Preprocessed/Audio/"+type+"/"+pos+".wav", 'wb')
            wf2.setnchannels(channels)
            wf2.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf2.setframerate(rate)
            wf2.writeframes(b''.join(frames))
            wf2.close()

            question = AudioSegment.from_wav("App_Data/Training/Raw/Audio/"+type+"/"+pos+".wav")
            play(question)

        self.pos += 1
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
                tk.Label(text=f"{text}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=45, y=450)
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
                tk.Label(text=f"{ask}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=45, y=450)
                position = self.emotion+": "+str(tempPos+1)+"/10"
                tk.Label(text=f"{position}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=45, y=480)
            else:
                self.root.destroy()