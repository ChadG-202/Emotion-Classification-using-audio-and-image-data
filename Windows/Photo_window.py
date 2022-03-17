import argparse
import cv2
import dlib
import os
import PIL.Image, PIL.ImageTk
import threading
import tkinter as tk

'''
Tkinter window which can be used to view,
take and store pictures.
'''
class Photo_taker():
    def __init__(self, window, window_title, samples_num, test_set, video_source=0, pos=0, taken=0, ok=False):
        # Window global var
        self.root = window
        self.root.title(window_title)
        self.sample_num = samples_num     # Number of samples to be taken
        self.test_set = test_set          # Is this a test widnow?
        self.video_source = video_source  
        self.pos = pos                    # State of window
        self.taken = taken                # Samples taken
        self.ok=ok
        self.root.configure(background="#4a4a4a")
        self.root.geometry("640x600")

        # path
        self.path = "App_Data/Training/Raw/Image/"     # Path to store images
        self.list_of_dir = ["Happy", "Neutral", "Sad"] # Directories avaliable

        # open video source (by default this will try to open the computer webcam)
        self.vid = VideoCapture(self.video_source)

        # Label description
        self.my_string_var = tk.StringVar()
        if self.test_set:
            self.my_string_var.set("Take a photo make sure your face is in the center")
        else:
            self.my_string_var.set("Take a HAPPY photo, make sure your face is in the center")
        # Screen text
        self.photo_description=tk.Label(textvariable=self.my_string_var, font="arial 12 bold", background="#4a4a4a", fg="white")
        self.photo_description2=tk.Label(text="of the screen and that both eyes can be seen.", font="arial 12 bold", background="#4a4a4a", fg="white")

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self.root, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tk.Button(self.root, font="arial 20", text="Snapshot", bg="#C1E1C1", fg="black", border=0, command=self.snapshot)

        # Redo button
        self.btn_retake=tk.Button(self.root, font="arial 20", text="Re-take", bg="#111111", fg="white", border=0, command=self.retake)

        # Clear data from path to make room for new data
        if not self.test_set:
            self.clear(self.path)

        # After it is called once, the update method will be automatically called every 10 milliseconds
        self.delay=10
        self.update()

        self.root.mainloop()
    
    # Clear old data
    def clear(self, path):
        for dir in self.list_of_dir:
            for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path+dir)):
                for f in filenames:
                    os.remove(os.path.join(dirpath, f))
    
    # Take photo again
    def retake(self):
        if self.taken > 0:
            if self.taken%self.sample_num == 0:
                self.pos -=1
            self.taken -= 1
            self.update_title()

    # Take a photo and store in relevant folder
    def snapshot(self):
        # Get a frame from the video source
        ret,frame=self.vid.get_frame()

        if ret:
            path = ""
            sample_total = 0
            # Save image
            def save_image():
                try:
                    cv2.imwrite(path,cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(str(e) + " Folder does not exist")

            # Make sure there is 1 face visible in photo
            def check_face():
                detector = dlib.get_frontal_face_detector()

                frame =cv2.imread(path)
                gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                if len(faces) == 0 or len(faces) > 1:
                    print("No face found or too many faces")
                    self.retake()

            # Test data
            if self.test_set:
                if self.taken < 1:
                    path = "App_Data/Test/Raw/Image/test.jpg"  
            # Training data
            else: 
                if self.taken < self.sample_num:
                    path = self.path+self.list_of_dir[0]+"/"+str(self.taken)+".jpg"
                    sample_total = self.sample_num
                elif self.taken >= self.sample_num and self.taken < self.sample_num*2:
                    path = self.path+self.list_of_dir[1]+"/"+str(self.taken-self.sample_num)+".jpg"
                    sample_total = self.sample_num*2
                elif self.taken >= self.sample_num*2 and self.taken < self.sample_num*3:
                    path = self.path+self.list_of_dir[2]+"/"+str(self.taken-self.sample_num*2)+".jpg"
                    sample_total = self.sample_num*3

            t1 = threading.Thread(target=save_image)
            t2 = threading.Thread(target=check_face)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            self.taken +=1

            if self.test_set:
                if self.taken > 0:
                    self.root.destroy()
            else:
                if self.taken == sample_total:
                    self.pos += 1
                    if self.pos == 3:
                        self.root.destroy() # Finished
                
                self.update_title()

    # Update title
    def update_title(self):
        title = 'Take Happy Photo '+str(self.taken+1)+'/'+str(self.sample_num)
        text = "Take a HAPPY photo, make sure your face is in the center"

        if self.pos == 1:
            title = 'Take Neutral Photo '+str(self.taken+1-self.sample_num)+'/'+str(self.sample_num)
            text = "Take NEUTRAL photo make sure face is in the center"
        elif self.pos == 2:
            title = 'Take Sad Photo '+str(self.taken+1-self.sample_num*2)+'/'+str(self.sample_num)
            text = "Take SAD photo make sure face is in the center"
        
        self.root.title(title)
        self.my_string_var.set(text)

    # Update tkinter window
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.photo_description.place(x=150, y=485)
            self.photo_description2.place(x=165, y=505)
            if self.test_set:
                self.btn_retake.place(x=640, y=600)
                self.btn_snapshot.place(x=265, y=540)
            else:
                self.btn_retake.place(x=200, y=540)
                self.btn_snapshot.place(x=330, y=540)
        self.root.after(self.delay,self.update)

'''
Display video source to tkinter window
'''
class VideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Command Line Parser
        args=CommandLineParser().args

        # Video Type
        VIDEO_TYPE = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            #'mp4': cv2.VideoWriter_fourcc(*'H264'),
            'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }

        self.fourcc=VIDEO_TYPE[args.type[0]]

        # Video Dimension
        STD_DIMENSIONS =  {
            '480p': (640, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
            '4k': (3840, 2160),
        }
        res=STD_DIMENSIONS[args.res[0]]
        print(args.name,self.fourcc,res)

        # Set video sourec width and height
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
