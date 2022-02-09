from signal import signal
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Shift
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import librosa
import soundfile as sf
import os

IMAGE_PATH = "RawEmotionData/Image"
AUG_IMAGE_PATH = "AugEmotionData"
AUDIO_PATH = "RawEmotionData/Audio"
AUG_AUDIO_PATH = "EmotionDataset/Train"

# Audio augmenter
audio_augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5)
])

# Image augmenter
image_augment = ImageDataGenerator(rescale=1./255,
    rotation_range=30,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

if __name__ == "__main__":
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(AUDIO_PATH)):
        for f in filenames:
            signal, sr = librosa.load(os.path.join(dirpath, f))
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            for count in range(0, 10):
                label = semantic_label + "/" + str(count) + "_" + f
                augmented_signal = audio_augment(signal, sr)
                sf.write(os.path.join(AUG_AUDIO_PATH, label), augmented_signal, sr)
                print("Saving: "+ AUG_AUDIO_PATH + label)
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(IMAGE_PATH)):
        for f in filenames:
            pic = load_img(os.path.join(dirpath, f))
            pic_array = img_to_array(pic)

            X = pic_array.reshape((1,) + pic_array.shape) 

            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
                
            f_s = f.split(".")
            
            count = 1
            for batch in image_augment.flow(X, batch_size=5,save_to_dir=os.path.join(AUG_IMAGE_PATH, semantic_label), save_prefix=f_s[0], save_format=f_s[1]):
                count += 1
                if count > 10:
                    break