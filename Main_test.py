import unittest
from Main import *

class TestMainMethode(unittest.TestCase):

    # Test function finds smallest dataset
    def test_find_smallest_dataset(self):
        size = find_smallest_dataset("Testing_App_Data/Training/Preprocessed/Image")
        self.assertGreater(size, 190, "Expected 204 but found size: "+str(size))

    # Test correct predctions are being fused
    def test_Predictions(self):
        audio, image, percentages = Predictions([0.68, 0.21, 0.11], [0.821, 0.079, 0.1])
        self.assertEqual(audio, [0.68, 0.21, 0.11])
        self.assertEqual(image, [0.821, 0.079, 0.1])
        self.assertEqual(percentages, [1.501, 0.289, 0.21000000000000002])

    # Test prediction are being correclty made
    def test_Predict(self):
        audio, image, percentages = Predict(-1, "JSON_files/TrainingExample.json")
        happy_pred = str(percentages[0])
        neutral_pred = str(percentages[1])
        sad_pred = str(percentages[2])
        self.assertEqual(happy_pred, "1.6549125")
        self.assertEqual(neutral_pred, "0.0061103036")
        self.assertEqual(sad_pred, "0.33897704")

    # Test enough augmented images are generated
    def test_Aug_Image(self):
        augment_image_data("Testing_App_Data/Training/Raw/Image", "Testing_App_Data/Training/Augmented/")
        dir_list = ["/Happy", "/Neutral", "/Sad"]
        data_set_sizes = []
        for dir in dir_list:
            size = sum(len(files) for _, _, files in os.walk("Testing_App_Data/Training/Augmented/Image"+dir))
            data_set_sizes.append(size)
        total_size = sum(data_set_sizes)
        self.assertGreater(total_size , 580, "Too few image augmented samples")

    # Test enough augmented audio samples are generated
    def test_Aug_Audio(self):
        augment_audio_data("Testing_App_Data/Training/Raw/Audio", "Testing_App_Data/Training/Preprocessed/")
        dir_list = ["/Happy", "/Neutral", "/Sad"]
        data_set_sizes = []
        for dir in dir_list:
            size = sum(len(files) for _, _, files in os.walk("Testing_App_Data/Training/Preprocessed/Audio"+dir))
            data_set_sizes.append(size)
        total_size  = sum(data_set_sizes)
        self.assertGreater(total_size , 580, "Too few audio augmented samples")
    
    # Test images are cropped
    def test_Crop_Faces(self):
        crop_faces("Testing_App_Data/Training/Augmented/Image", "Testing_App_Data/Training/Preprocessed/", True)
        dir_list = ["/Happy", "/Neutral", "/Sad"]
        data_set_sizes = []
        correct_size = []
        for dir in dir_list:
            size = sum(len(files) for _, _, files in os.walk("Testing_App_Data/Training/Preprocessed/Image"+dir))
            data_set_sizes.append(size)
            size_c = sum(len(files) for _, _, files in os.walk("Testing_App_Data/Training/Augmented/Image"+dir))
            correct_size.append(size_c)
        total_size = sum(data_set_sizes)
        answer = sum(correct_size)
        self.assertEqual(total_size, answer, "Wrong number of cropped images")

    # Test JSON file is created
    def test_Process(self):
        os.remove("Test_JSON/test.json")
        Process("Testing_App_Data/Training/Preprocessed/Audio", "Testing_App_Data/Training/Preprocessed/Image", "Test_JSON/test.json", False)
        file_generated = sum(len(files) for _, _, files in os.walk("Test_JSON"))
        self.assertEqual(file_generated, 1, "File hasnt been made")

if __name__ == '__main__':
    unittest.main()