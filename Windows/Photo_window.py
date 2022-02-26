import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import argparse


class Photo_taker():
    def __init__(self, window, window_title, test_set, video_source=0, pos=0, taken=0, ok=False):
        self.root = window
        self.root.title(window_title)
        self.test_set = test_set
        self.video_source = video_source
        self.pos = pos
        self.taken = taken
        self.ok=ok

        self.root.configure(background="#4a4a4a")
        self.root.geometry("640x600")

        # open video source (by default this will try to open the computer webcam)
        self.vid = VideoCapture(self.video_source)

        # Label description
        self.photo_description=tk.Label(text="Take Picture", font="arial 20 bold", background="#4a4a4a", fg="white")

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self.root, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tk.Button(self.root, font="arial 20", text="Capture", bg="#111111", fg="white", border=0, command=self.snapshot)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay=10
        self.update()

        self.root.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret,frame=self.vid.get_frame()

        if ret:
            if self.test_set:
                if self.taken < 1:
                    cv2.imwrite("App_Data/Test/Raw/Image/test.jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    self.taken += 1
                    if self.taken > 0:
                        self.root.destroy()
            else:
                if self.taken < 10:
                    cv2.imwrite("App_Data/Training/Raw/Image/Happy/"+str(self.taken)+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    self.taken +=1
                    if self.taken == 10:
                        self.pos =1
                elif self.taken >= 10 and self.taken < 20:
                    cv2.imwrite("App_Data/Training/Raw/Image/Neutral/"+str(self.taken-10)+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    self.taken +=1
                    if self.taken == 20:
                        self.pos =2
                elif self.taken >= 20 and self.taken < 30:
                    cv2.imwrite("App_Data/Training/Raw/Image/Sad/"+str(self.taken-20)+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    self.taken +=1
                    # Finished
                    if self.taken == 30:
                        self.pos =3
                        self.root.destroy()

                if self.pos == 0:
                    self.root.title('Take Happy Photo '+str(self.taken)+'/10')
                elif self.pos == 1:
                    self.root.title('Take Neutral Photo '+str(self.taken-10)+'/10')
                elif self.pos == 2:
                    self.root.title('Take Sad Photo '+str(self.taken-20)+'/10')
                elif self.pos == 3:
                    self.root.title('Done press x to move on!')
       
    def update(self):

        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.photo_description.place(x=230, y=490)
            self.btn_snapshot.place(x=270, y=540)
        self.root.after(self.delay,self.update)


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
        
# make it look better