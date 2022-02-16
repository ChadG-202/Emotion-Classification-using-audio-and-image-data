from PhotoApp import ImageApp
from AudioApp import RecorderApp
import tkinter as tk
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Shift
import librosa
import soundfile as sf
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import dlib
import cv2
import math
import json

def crop_faces():
    detector = dlib.get_frontal_face_detector()

    def save(img,name, bbox, i, width=48,height=48):
        x, y, w, h = bbox
        imgCrop = img[y:h, x: w]
        try:
            imgCrop = cv2.resize(imgCrop, (width, height))
            cv2.imwrite(os.path.join(name, i), imgCrop)
        except Exception as e:
            print(str(e) +"\n couldnt resize: "+ i)

    def faces(new_path, old_path):
        for root, dirs, files in os.walk(old_path):
            for i, file in enumerate(files):
                frame =cv2.imread(os.path.join(root, file))
                gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                dirpath_components = root.split("/")
                semantic_label = dirpath_components[-1]

                for face in faces:
                    x1, y1 = face.left(), face.top()
                    x2, y2 = face.right(), face.bottom()
                    save(gray,os.path.join(new_path, semantic_label),(x1,y1,x2,y2), file)
                
                if not faces:
                    print("No face found: "+os.path.join(new_path, semantic_label)+"/"+ file)

    faces('App/PreprocessedData/', 'App/AugImageData/Image')

def augment_data():
    def augment_audio_data():
        AUDIO_PATH = "App/AppData/Audio"
        AUG_AUDIO_PATH = "App/PreprocessedData/"

        augment = Compose([
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5)
        ])

        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(AUDIO_PATH)):
            for f in filenames:
                signal, sr = librosa.load(os.path.join(dirpath, f))
                dirpath_components = dirpath.split("/")
                semantic_label = dirpath_components[-1]
                for count in range(0, 10):
                    label = semantic_label + "/" + str(count) + "_" + f
                    augmented_signal = augment(signal, sr)
                    sf.write(os.path.join(AUG_AUDIO_PATH, label), augmented_signal, sr)

    def augment_image_data():
        IMAGE_PATH = "App/AppData/Image"
        AUG_IMAGE_PATH = "App/AugImageData/"

        datagen = ImageDataGenerator(rescale=1./255,
            rotation_range=15,
            zoom_range=0.1,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(IMAGE_PATH)):
            for f in filenames:
                pic = load_img(os.path.join(dirpath, f))
                pic_array = img_to_array(pic)

                X = pic_array.reshape((1,) + pic_array.shape) 

                dirpath_components = dirpath.split("/")
                semantic_label = dirpath_components[-1]
                    
                f_s = f.split(".")
                
                count = 1
                for batch in datagen.flow(X, batch_size=5,save_to_dir=os.path.join(AUG_IMAGE_PATH, semantic_label), save_prefix=f_s[0], save_format=f_s[1]):
                    count += 1
                    if count > 10:
                        break
    augment_image_data()
    augment_audio_data()
    crop_faces()

def process_data():
    # Paths
    DATASET_AUDIO_TRAIN = "App/PreprocessedData/Audio"
    DATASET_IMAGE_TRAIN = "App/PreprocessedData/Image"
    JSON_TRAIN = "App/data.json"

    # Audio Var
    SAMPLE_RATE = 22050
    DURATION = 4
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

    # Image Var
    IMG_SIZE = 48

    # Process audio and image data - store in data.json file
    def save_data(audio_path, image_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
        data = {
            "mapping": [],
            "mfcc": [],
            "image": [],
            "labels": []
        }
        
        # how many samples for each audio input
        num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments) 
        # number of mfccs if samples are made
        expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) 
        
        # walk through audio directories
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(audio_path)):
            if dirpath is not audio_path:
                dirpath_components = dirpath.split("/")
                semantic_label = dirpath_components[-1]
                # store the directories opened
                data["mapping"].append(semantic_label)
                
                print("\nProcessing {}".format(semantic_label))
                
                for f in filenames:
                    try:
                        file_path = os.path.join(dirpath, f)
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    except Exception as e:                                                    
                        print('Audio failed to process: ' + e)
                    
                    for s in range(num_segments):
                        # Process audio
                        start_sample = num_samples_per_segment * s
                        finish_sample = start_sample + num_samples_per_segment
                            
                        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)

                        mfcc = mfcc.T
                        
                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                            # store mfcc data
                            data["mfcc"].append(mfcc.tolist())
                            # store audio type
                            data["labels"].append(i-1)
                            print("{}, segment:{}".format(file_path, s+1))
        
        # walk through image directories
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(image_path)):
            if dirpath is not image_path:
                dirpath_components = dirpath.split("/")
                semantic_label = dirpath_components[-1]
                # store the directories opened
                data["mapping"].append(semantic_label)
                
                print("\nProcessing {}".format(semantic_label))
                
                for f in filenames:
                    file_path = os.path.join(dirpath, f)
                    try:
                        # process image
                        img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) 
                        sized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        # store image data
                        data["image"].append(sized_array.tolist())
                        print("{}".format(file_path))
                    except Exception as e:                                                    
                        print('Image failed to process: ' + e)
        
        # dump stored data into json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)
        

    save_data(DATASET_AUDIO_TRAIN, DATASET_IMAGE_TRAIN, JSON_TRAIN, num_segments=1)


if __name__ == "__main__":
    ImageApp(tk.Tk(),'Take Happy Photo')
    RecorderApp(tk.Tk())
    augment_data()
    process_data()

