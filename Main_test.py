import unittest
from Main import find_smallest_dataset, Predictions

class TestMainMethode(unittest.TestCase):

    def test_find_smallest_dataset(self):
        size = find_smallest_dataset("Test_Data/Training/Preprocessed/Image")
        self.assertEqual(size, 204)

    def test_Predictions(self):
        audio, image, percentages = Predictions([0.68, 0.21, 0.11], [0.821, 0.079, 0.1])
        self.assertEqual(audio, [0.68, 0.21, 0.11])
        self.assertEqual(image, [0.821, 0.079, 0.1])
        self.assertEqual(percentages, [1.501, 0.289, 0.21000000000000002])

if __name__ == '__main__':
    unittest.main()