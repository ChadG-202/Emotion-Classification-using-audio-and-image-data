import unittest
from Main import find_smallest_dataset

class TestMainMethode(unittest.TestCase):

    def test_find_smallest_dataset(self):
        size = find_smallest_dataset("App_Data/Training/Preprocessed/Image")
        self.assertEqual(size, 55)


if __name__ == '__main__':
    unittest.main()