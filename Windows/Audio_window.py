from pydub import AudioSegment
from pydub.playback import play

import os
import pyaudio
import threading
import time
import tkinter as tk
import wave

'''
Tkinter window which records and 
saves wav audio files.
'''
class Audio_recorder:
    def __init__(self, window, window_title, samples_num, test_set, pos=0, emotion='Happy'):
        # Window global var
        self.root = window
        self.root.title(window_title)
        self.test_set = test_set        # Is it a test window?
        self.sample_num = samples_num   # Number of samples to be taken
        self.pos = pos                  # Current state
        self.emotion = emotion          # Current emotion
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
    
    # Retake recording
    def retake(self):
        if self.pos > 0:
            if self.pos%self.sample_num == 0:
                if self.emotion == self.list_of_dir[1]:
                    self.emotion = self.list_of_dir[0]
                elif self.emotion == self.list_of_dir[2]:
                    self.emotion == self.list_of_dir[1]
            self.pos -= 1
            self.sentence()

    # Record 4 second sample
    def Recording(self, type, pos):
        p = pyaudio.PyAudio()

        # Recording values
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

        # Count down
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
        
        # Save audio
        def Save():
            save_path = ""
            if self.test_set:
                save_path = "App_Data/Test/Preprocessed/Audio/test.wav"
            else:
                save_path  = self.path+type+"/"+pos+".wav"

            wf = wave.open(save_path, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            # Play audio
            if not self.test_set:
                try:
                    question = AudioSegment.from_wav(self.path+type+"/"+pos+".wav")
                    play(question)
                except Exception as e:
                    print(str(e)+" couldnt play")

        # Save in seperate thread
        t1 = threading.Thread(target=Save)
        t1.start()

        self.pos += 1
        self.sentence()

    # Call recording
    def Record(self):
        self.Recording(self.emotion, str(self.pos))
    
    # Window text
    def sentence(self):
        questions = ["'Can you help me?'", "'What is the weather today?'",
        "'Can you find me a route home?'", "'What time is it?'", 
        "'Can you turn the light on?'", "'What song is this?'", 
        "'Who am i?'", "'What is there to watch on Netflix?'",
        "'Are unicorns real?'", "'How do you spell tree?'"]

        tempPos = self.pos

        # Test
        if self.test_set:
            if self.pos < 1:
                text = "Choose one question below to ask the bot.\n"
                for i in range(0, 9, 2):
                    text += questions[i]+", "+questions[i+1]+"\n"
                tk.Label(text=f"{text}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=45, y=340)
            else:
                self.root.destroy()
        # train
        else:
            if self.pos > self.sample_num-1 and self.pos < self.sample_num*2:
                tempPos = self.pos - self.sample_num
                self.emotion = self.list_of_dir[1]
            elif self.pos > (self.sample_num*2)-1 and self.pos < self.sample_num*3:
                tempPos = self.pos - self.sample_num*2
                self.emotion = self.list_of_dir[2]

            if self.pos < self.sample_num*3:
                ask = "Say the question:\n"+questions[tempPos]+"\nin a "+self.emotion+" expression:\n"+str(tempPos+1)+"/"+str(self.sample_num)
                tk.Label(text=f"{ask}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=45, y=415)
            else:
                time.sleep(4.5)
                self.root.destroy()