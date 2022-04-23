import os

'''
Defines some common functions 
shared between the windows that source data.
'''
class Source():
    def __init__(self, path, samples_num, test_set):
        self.path = path
        self.sample_num = samples_num
        self.test_set = test_set
        self.list_of_dir = ["Happy", "Neutral", "Sad"]
        self.taken = 0

    # Clear old data
    def clear(self, path): 
        for dir in self.list_of_dir:
            for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path+dir)):
                for f in filenames:
                    os.remove(os.path.join(dirpath, f))

    # Take data again
    def retake(self, pos): 
        if self.taken > 0:
            if self.taken%self.sample_num == 0:
                pos -=1
            self.taken -= 1
        return pos