# import unittest
# import tkinter as tk
# import shutil
# import os
# from Audio_window import Audio_recorder

# class TestAudioWindow(unittest.TestCase):

#     def test_clear(self):
#         src = 'Test_Data/Training/Raw/Image/Happy/'
#         trg = 'Test_Data/Fake_data/'
        
#         files=os.listdir(src)
        
#         for fname in files:
#             shutil.copy2(os.path.join(src,fname), trg)

#         amount = len(os.listdir(trg))
#         self.assertEqual(amount, 10)

#         test = Audio_recorder(tk.Tk(), 'test', trg, 1, True)
#         test.clear(test.path)

#         amount = len(os.listdir(trg))
#         self.assertEqual(amount, 0)

# if __name__ == '__main__':
#     unittest.main()