import unittest
import shutil
import os
from Source_window import Source

class TestSourceWindow(unittest.TestCase, Source):


    def test_clear(self):
        src = 'Test_Data/Training/Raw/Image/Happy/'
        trg = 'Test_Data/Fake_Data/'
        
        files=os.listdir(src)
        
        for fname in files:
            shutil.copy2(os.path.join(src,fname), trg)

        amount = len(os.listdir(trg))
        self.assertEqual(amount, 10)

        test = Source(trg, 1, True)
        test.clear(trg)

        amount = len(os.listdir(trg))
        self.assertEqual(amount, 0)

    def test_retake(self):
        trg = 'Test_Data/Fake_Data/'

        test = Source(trg, 2, True)
        num = test.retake(10)
        self.assertEqual(num, 9)

if __name__ == '__main__':
    unittest.main()