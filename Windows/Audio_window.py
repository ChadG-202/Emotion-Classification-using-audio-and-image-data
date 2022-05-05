import pyaudio
import threading
import time
import tkinter as tk
import wave

from pydub import AudioSegment
from pydub.playback import play
from Windows.Source_window import Source
from Windows.Structure_window import Structure

'''
Tkinter window which records and 
saves wav audio files.
'''
class Audio_recorder(Structure, Source):
    def __init__(self, window, window_title, path, samples_num, test_set):
        Structure.__init__(self, window, window_title)
        Source.__init__(self, path, samples_num, test_set)
        
        self.done_recording = True

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
        if not self.test_set:
            self.btn_retake=tk.Button(self.root, font="arial 20", text="Re-take", bg="#111111", fg="white", border=0, command=self.retakeB).pack(pady=10)

            # Initial clear
            self.clear(self.path)

        self.sentence()

        self.root.mainloop()
    
    # Retake recording
    def retakeB(self):
        self.pos = self.retake(self.pos)
        self.sentence()

    # Record 4 second sample
    def Recording(self, type, pos):
        self.done_recording = False
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
        self.done_recording = True
        
        # Save audio
        def Save():
            save_path = ""
            if self.test_set:
                save_path = self.path+"test.wav"
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

        self.taken += 1
        self.sentence()

    # Call recording
    def Record(self):
        if self.done_recording:
            self.Recording(self.list_of_dir[self.pos], str(self.taken))
    
    # Window text
    def sentence(self):
        questions = ["'Can you help me?'", "'What is the weather today?'",
        "'Can you find me a route home?'", "'What time is it?'", 
        "'Can you turn the light on?'", "'What song is this?'", 
        "'Who am i?'", "'What is there to watch on Netflix?'",
        "'Are unicorns real?'", "'How do you spell tree?'"]

        tempTaken = self.taken

        # Test
        if self.test_set:
            if self.taken < 1:
                text = "Choose one question below to ask the bot.\n"
                for i in range(0, 9, 2):
                    text += questions[i]+", "+questions[i+1]+"\n"
                tk.Label(text=f"{text}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=45, y=340)
            else:
                self.root.destroy()
        # train
        else:
            if self.taken > self.sample_num-1 and self.taken < self.sample_num*2:
                tempTaken = self.taken - self.sample_num
                self.pos = 1
            elif self.taken > (self.sample_num*2)-1 and self.taken < self.sample_num*3:
                tempTaken = self.taken - self.sample_num*2
                self.pos = 2

            if self.taken < self.sample_num*3:
                ask = "Say the question:\n"+questions[tempTaken]+"\nin a "+self.list_of_dir[self.pos]+" expression:\n"+str(tempTaken+1)+"/"+str(self.sample_num)
                tk.Label(text=f"{ask}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=45, y=415)
            else:
                time.sleep(4.5)
                self.root.destroy()