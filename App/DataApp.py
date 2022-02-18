from tkinter import *
import sounddevice as sound
from scipy.io.wavfile import write
import time
import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import argparse

class RecorderApp:
    def __init__(self, window, pos=0, emotion="Happy"):
        self.pos = pos
        self.emotion = emotion
        self.root = window
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        self.root.title("Voice Recorder")
        self.root.configure(background="#4a4a4a")

        #Icon
        self.image_icon=PhotoImage(file="App/rec-button.png")
        self.root.iconphoto(False, self.image_icon)

        #Logo
        self.photo=PhotoImage(file="App/rec-button.png")
        self.myimage=Label(image=self.photo,background="#4a4a4a")
        self.myimage.pack(pady=30)

        #Name
        Label(text="Voice Recorder", font="arial 30 bold", background="#4a4a4a", fg="white").pack()

        #Button
        self.record = Button(self.root, font="arial 20", text="Record",bg="#111111",fg="white",border=0,command=self.Record).pack(pady=30)

        self.sentence()

        self.root.mainloop()

    def Recording(self, type, path):
        freq=22050
        dur=4
        channel=2
        recording=sound.rec(dur*freq, samplerate=freq,channels=channel)

        #Timer
        timeLeft=dur
        while timeLeft>0:
            timeLeft-=1
            Label(text=f"{str(timeLeft)}", font="arial 40",width=4,background="#4a4a4a").place(x=240, y=420)
            self.root.update()
            time.sleep(1)

        sound.wait()
        write("App/AppData/Audio/"+type+"/"+path+".wav",freq,recording)
        write("App/PreprocessedData/Audio/"+type+"/"+path+".wav",freq,recording)

        self.pos +=1
        self.sentence()

    def Record(self):
        self.Recording(self.emotion, str(self.pos))
    
    def sentence(self):
        questions = ["'Can you help me?'", "'Who are you?'",
        "'Where am I?'", "'Why is this happening?'", "'What time is it?'", 
        "'Take me home?'", "'Whats the weather today?'", "'Set a timer for 10 minutes?'",
        "'How do you spell tree?'", "'Whats 10 + 20?'"]

        tempPos = self.pos
        if self.pos > 9 and self.pos < 20:
            tempPos = self.pos - 10
            self.emotion = "Neutral"
        elif self.pos > 19 and self.pos < 30:
            tempPos = self.pos - 20
            self.emotion = "Sad"

        if self.pos < 30:
            ask = "Say the phrase: "+questions[tempPos]+" in a "+self.emotion+" voice."
            Label(text=f"{ask}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=20, y=350)
        else:
            self.root.destroy()

class ImageApp:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.video_source = video_source
        self.ok=False
        self.stage = 0
        self.photos_taken = 0
        self.window.title(window_title+str(self.photos_taken)+'/10')

        # open video source (by default this will try to open the computer webcam)
        self.vid = VideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tk.Button(window, text="Snapshot", command=self.snapshot)
        self.btn_snapshot.pack(side=tk.LEFT)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay=10
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret,frame=self.vid.get_frame()

        if ret:
            if self.photos_taken < 10:
                cv2.imwrite("App/AppData/Image/Happy/"+str(self.photos_taken)+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                cv2.imwrite("App/AugImageData/Image/Happy/"+str(self.photos_taken)+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                self.photos_taken +=1
                if self.photos_taken == 10:
                    self.stage =1
            elif self.photos_taken >= 10 and self.photos_taken < 20:
                cv2.imwrite("App/AppData/Image/Neutral/"+str(self.photos_taken-10)+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                cv2.imwrite("App/AugImageData/Image/Neutral/"+str(self.photos_taken-10)+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                self.photos_taken +=1
                if self.photos_taken == 20:
                    self.stage =2
            elif self.photos_taken >= 20 and self.photos_taken < 30:
                cv2.imwrite("App/AppData/Image/Sad/"+str(self.photos_taken-20)+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                cv2.imwrite("App/AugImageData/Image/Sad/"+str(self.photos_taken-20)+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                self.photos_taken +=1
                # Finished
                if self.photos_taken == 30:
                    self.stage =3
                    self.window.destroy()


        if self.stage == 0:
            self.window.title('Take Happy Photo'+str(self.photos_taken)+'/10')
        elif self.stage == 1:
            self.window.title('Take Neutral Photo'+str(self.photos_taken-10)+'/10')
        elif self.stage == 2:
            self.window.title('Take Sad Photo'+str(self.photos_taken-20)+'/10')
        elif self.stage == 3:
            self.window.title('Done press x to move on!')
       
    def update(self):

        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.window.after(self.delay,self.update)


class VideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Command Line Parser
        args=CommandLineParser().args

        
        #create videowriter

        # 1. Video Type
        VIDEO_TYPE = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            #'mp4': cv2.VideoWriter_fourcc(*'H264'),
            'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }

        self.fourcc=VIDEO_TYPE[args.type[0]]

        # 2. Video Dimension
        STD_DIMENSIONS =  {
            '480p': (640, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
            '4k': (3840, 2160),
        }
        res=STD_DIMENSIONS[args.res[0]]
        print(args.name,self.fourcc,res)

        #set video sourec width and height
        self.vid.set(3,res[0])
        self.vid.set(4,res[1])

        # Get video source width and height
        self.width,self.height=res


    # To get frames
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            cv2.destroyAllWindows()

class CommandLineParser:
    
    def __init__(self):

        # Create object of the Argument Parser
        parser=argparse.ArgumentParser(description='Script to take photo')

        # Only values is supporting for the tag --type. So nargs will be '1' to get
        parser.add_argument('--type', nargs=1, default=['avi'], type=str, help='Type of the video output: for now we have only AVI & MP4')

        # Only one values are going to accept for the tag --res. So nargs will be '1'
        parser.add_argument('--res', nargs=1, default=['480p'], type=str, help='Resolution of the video output: for now we have 480p, 720p, 1080p & 4k')

        # Only one values are going to accept for the tag --name. So nargs will be '1'
        parser.add_argument('--name', nargs=1, default=['output'], type=str, help='Enter Output video title/name')

        # Parse the arguments and get all the values in the form of namespace.
        # Here args is of namespace and values will be accessed through tag names
        self.args = parser.parse_args()