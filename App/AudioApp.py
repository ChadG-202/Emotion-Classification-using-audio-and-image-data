from tkinter import *
from tkinter import messagebox

root=Tk()
root.geometry("600x700+400+80")
root.resizable(False, False)
root.title("Voice Recorder")
root.configure(background="#4a4a4a")

# Logo
photo = "rec-button.png"
myimage=Label(image=photo,background="#4a4a4a")
myimage.pack(padx=5,pady=5)

root.mainloop()