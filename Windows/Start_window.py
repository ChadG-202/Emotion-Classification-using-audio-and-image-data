import tkinter as tk


class Start():
    def __init__(self, window, window_title, pos=0):
        self.root = window
        self.root.title(window_title)
        self.pos = pos

        self.root.geometry("640x600")
        self.root.resizable(False, False)
        self.root.title("Voice Recorder")
        self.root.configure(background="#4a4a4a")

        #Name
        tk.Label(text="Emotion Chatbot", font="arial 30 bold", background="#4a4a4a", fg="white").pack()

        #Button
        self.nextB = tk.Button(self.root, font="arial 20", text="Next",bg="#C1E1C1",fg="black",border=0,command=self.next).pack(pady=30)

        self.sentence()
        self.root.mainloop()

    def next(self):
        self.pos += 1
        self.sentence()

    def sentence(self):
        text = ""
        if self.pos == 0:
            text = "page 1"
        elif self.pos == 1:
            text = "page 2"
        elif self.pos == 2:
            text = "page 3"
        else:
            self.root.destroy()

        if self.pos < 3:
            tk.Label(text=f"{text}", font="arial 15",width=50,background="#4a4a4a",fg="white").place(x=45, y=450)
