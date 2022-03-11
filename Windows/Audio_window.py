import pyaudio
import wave
from pydub import AudioSegment
from pydub.playback import play
import tkinter as tk
import os

class Audio_recorder:
    def __init__(self, window, window_title, samples_num, test_set, pos=0, emotion='Happy'):
        self.root = window
        self.root.title(window_title)
        self.test_set = test_set
        self.sample_num = samples_num
        self.pos = pos
        self.emotion = emotion

        self.root.geometry("640x600")
        self.root.resizable(False, False)
        self.root.title("Voice Recorder")
        self.root.configure(background="#4a4a4a")

        #Icon
        self.image_icon=tk.PhotoImage(file="App_Images/rec-button.png")
        self.root.iconphoto(False, self.image_icon)

        # path
        self.path = "App_Data/Training/Raw/Audio/"
        self.list_of_dir = ["Happy", "Neutral", "Sad"]

        #Logo
        self.photo=tk.PhotoImage(file="App_Images/rec-button.png")
        self.myimage=tk.Label(image=self.photo,background="#4a4a4a")
        self.myimage.pack(pady=30)

        #Name
        tk.Label(text="Voice Recorder", font="arial 30 bold", background="#4a4a4a", fg="white").pack()

        #Button
        self.record = tk.Button(self.root, font="arial 20", text="Record",bg="#C1E1C1",fg="black",border=0,command=self.Record).pack(pady=20)
        if not self.test_set:
            self.btn_retake=tk.Button(self.root, font="arial 20", text="Re-take", bg="#111111", fg="white", border=0, command=self.retake).pack(pady=10)

        # Initial clear
        self.clear(self.path)

        self.sentence()

        self.root.mainloop()

    # Clear old data
    def clear(self, path):
        for dir in self.list_of_dir:
            for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path+dir)):
                for f in filenames:
                    os.remove(os.path.join(dirpath, f))
    
    def retake(self):
        if self.pos > 0:
            if self.pos%self.sample_num == 0:
                if self.emotion == self.list_of_dir[1]:
                    self.emotion = self.list_of_dir[0]
                elif self.emotion == self.list_of_dir[2]:
                    self.emotion == self.list_of_dir[1]
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
            wf = wave.open(self.path+type+"/"+pos+".wav", 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            question = AudioSegment.from_wav(self.path+type+"/"+pos+".wav")
            play(question)

        self.pos += 1
        self.sentence()

    def Record(self):
        self.Recording(self.emotion, str(self.pos))
    
    def sentence(self):
        questions = ["'Can you help me?'", "'What is the weather today?'",
        "'Can you find me a route home?'", "'What time is it?'", 
        "'Can you turn the light on?'", "'What song is this?'", 
        "'Who am i?'", "'What is there to watch on Netflix?'",
        "'Are unicorns real?'", "'How do you spell tree?'"]

        tempPos = self.pos

        if self.test_set:
            if self.pos < 1:
                text = "Choose one question below to ask the bot.\n"
                for i in range(0, 9, 2):
                    text += questions[i]+", "+questions[i+1]+",\n"
                tk.Label(text=f"{text}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=45, y=340)
            else:
                self.root.destroy()
        else:
            if self.pos > self.sample_num-1 and self.pos < self.sample_num*2:
                tempPos = self.pos - self.sample_num
                self.emotion = self.list_of_dir[1]
            elif self.pos > (self.sample_num*2)-1 and self.pos < self.sample_num*3:
                tempPos = self.pos - self.sample_num*2
                self.emotion = self.list_of_dir[2]

            if self.pos < self.sample_num*3:
                ask = "Say the question: "+questions[tempPos]+" in a "+self.emotion+" voice."
                tk.Label(text=f"{ask}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=45, y=450)
                position = self.emotion+": "+str(tempPos+1)+"/"+str(self.sample_num)
                tk.Label(text=f"{position}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=45, y=480)
            else:
                self.root.destroy()