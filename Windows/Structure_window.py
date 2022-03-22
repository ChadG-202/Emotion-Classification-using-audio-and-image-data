'''
Defines how each window will be structured.
'''
class Structure():
    def __init__(self, window, window_title):
        self.root = window
        self.root.title(window_title)
        self.root.geometry("640x600")
        self.root.resizable(False, False)
        self.root.configure(background="#4a4a4a")
        self.pos = 0