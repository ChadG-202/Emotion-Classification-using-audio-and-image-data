import cv2
import dlib
import json
import librosa
import math
import numpy as np
import os
import soundfile as sf
import threading
import tensorflow as tf
import tkinter as tk
import tensorflow.keras as keras

from audiomentations import Compose, TimeStretch, Shift
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from Windows.Audio_window import Audio_recorder
from Windows.Photo_window import Photo_taker
from Windows.Result_window import Result
from Windows.Start_window import Start

# Clear directories
def clear_dir(path, type):
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path+type+"/")):
        for f in filenames:
            os.remove(os.path.join(dirpath, f))

# Save cropped faces
def save_face(img,name, bbox, i, width=48,height=48):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    try:
        imgCrop = cv2.resize(imgCrop, (width, height))
        cv2.imwrite(os.path.join(name, i), imgCrop)
    except Exception as e:
        print(str(e) +"\n couldnt resize: "+ i)

# Find cropped faces
def crop_faces(old_path, new_path, clear):
    detector = dlib.get_frontal_face_detector()

    if clear:
        clear_dir(new_path, "Image")

    for i, (root, dirs, files) in enumerate(os.walk(old_path)):
        for file in files:
            frame =cv2.imread(os.path.join(root, file))
            gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            dirpath_components = root.split("/")
            semantic_label = dirpath_components[-1]

            for face in faces:
                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()
                save_face(gray,os.path.join(new_path, semantic_label),(x1,y1,x2,y2), file)
            
            if not faces:
                print("No face found: "+ file)

# Augment the audio data
def augment_audio_data(path, aug_path): 
    augment = Compose([
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5)
    ])

    clear_dir(aug_path, "Audio")

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path)):
        for f in filenames:
            signal, sr = librosa.load(os.path.join(dirpath, f))
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            # Save original
            label = semantic_label + "/" + f
            sf.write(os.path.join(aug_path, label), signal, sr)
            # Save 10 aug
            for count in range(0, 20):
                label = semantic_label + "/" + str(count) + "_" + f
                augmented_signal = augment(signal, sr)
                sf.write(os.path.join(aug_path, label), augmented_signal, sr)

# Augment the image data
def augment_image_data(path, aug_path): 
    datagen = ImageDataGenerator(rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    clear_dir(aug_path, "Image")
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path)):
        for f in filenames:
            pic = load_img(os.path.join(dirpath, f))
            pic_array = img_to_array(pic)

            X = pic_array.reshape((1,) + pic_array.shape) 

            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
                
            f_s = f.split(".")
            
            for x, val in zip(datagen.flow(X, batch_size=2, save_to_dir=os.path.join(aug_path, semantic_label), save_prefix=f_s[0], save_format=f_s[1]),range(19)):     
                pass

# Determine the smallest dataset
def find_smallest_dataset(path):
    dir_list = ["/Happy", "/Neutral", "/Sad"]
    data_set_sizes = []
    for dir in dir_list:
        size = sum(len(files) for _, _, files in os.walk(path+dir))
        data_set_sizes.append(size)
    smallest_size = min(data_set_sizes)
    return smallest_size

# Process audio and image data - store in json file
def Process(audio_path, image_path, json_path, test, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
    SAMPLE_RATE = 22050
    DURATION = 4
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    IMG_SIZE = 48

    if test:
        data = {
            "mfcc": [],
            "image": []
        }
        
        num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

        try:
            signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(e)                                                    
        
        for s in range(num_segments):
            start_sample = num_samples_per_segment * s
            finish_sample = start_sample + num_samples_per_segment
                
            mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                        sr=sr,
                                        n_fft=n_fft,
                                        n_mfcc=n_mfcc,
                                        hop_length=hop_length)
            
            mfcc = mfcc.T
            
            if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                data["mfcc"].append(mfcc.tolist())
        
        img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
        sized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data["image"].append(sized_array.tolist())
    else:
        dataset_size = find_smallest_dataset(image_path)

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
                
                for f in filenames[:dataset_size]:
                    try:
                        file_path = os.path.join(dirpath, f)
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    except Exception as e:                                                    
                        print('Audio failed to process: ' + str(e))
                    
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
                
                for f in filenames[:dataset_size]:
                    file_path = os.path.join(dirpath, f)
                    try:
                        # process image
                        img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) 
                        sized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        # store image data
                        data["image"].append(sized_array.tolist())
                        print("{}".format(file_path))
                    except Exception as e:                                                    
                        print('Image failed to process: ' + str(e))
    
    # dump stored data into json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

# Train the models on the training data
def Train_models(DATA_PATH, IMG_SIZE):
    # Retrive audio data
    def load_audio_data(data_path):
        with open(data_path, "r") as fp:
            data = json.load(fp)
            
        X = np.array(data["mfcc"])
        y = np.array(data["labels"])
        return X, y
    # Retrive image data
    def load_image_data(data_path):
        with open(data_path, "r") as fp:
            data = json.load(fp)
            
        X = np.array(data["image"]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        y = np.array(data["labels"])
        return X, y

    # Split data
    def prepare_datasets(validation_size, X, y, A_type):
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size)

        if(A_type):
            X_train = X_train[..., np.newaxis]
            X_validation = X_validation[..., np.newaxis]
        
        return X_train, X_validation, y_train, y_validation
    
    # CNN image model
    def build_image_model(input_shape):
        model = keras.Sequential()
        
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))                                
        model.add(keras.layers.Dropout(0.05))  
        
        model.add(keras.layers.Conv2D(16, (3, 3), activation='relu')) 
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))   
        model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))                                
        model.add(keras.layers.Dropout(0.35))
        
        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(480, activation='relu'))
        model.add(keras.layers.Dropout(0.15))
        
        model.add(keras.layers.Dense(3, activation='softmax'))
        
        return model
    
    # CNN audio model
    def build_audio_model(input_shape):
        model = keras.Sequential()
        
        model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
        model.add(keras.layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))                                
        model.add(keras.layers.Dropout(0.1))
        
        model.add(keras.layers.Conv2D(16, (3, 3), activation='relu')) 
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))   
        model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))                                
        model.add(keras.layers.Dropout(0.4))
        
        model.add(keras.layers.Flatten())
        
        model.add(keras.layers.Dense(192, activation='relu'))
        model.add(keras.layers.Dropout(0.05))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.05))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(160, activation='relu'))
        model.add(keras.layers.Dropout(0.05))
        model.add(keras.layers.Dense(480, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        
        model.add(keras.layers.Dense(3, activation='softmax'))
        
        return model

    # Train/Test Split data
    validation_size = 0.2

    X_audio, y_audio = load_audio_data(DATA_PATH)
    X_image, y_image = load_image_data(DATA_PATH)

    X_audio_train, X_audio_validation, y_audio_train, y_audio_validation = prepare_datasets(validation_size, X_audio, y_audio, True)

    X_image_train, X_image_validation, y_image_train, y_image_validation = prepare_datasets(validation_size, X_image, y_image, False)

    def train_audio():
        # Audio train
        audio_input_shape = (X_audio_train.shape[1], X_audio_train.shape[2], X_audio_train.shape[3])
        audio_model = build_audio_model(audio_input_shape)

        audio_optimizer = keras.optimizers.Adam(learning_rate=0.0001)

        audio_model.compile(optimizer=audio_optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])

        # audio_model.summary()

        audio_history = audio_model.fit(X_audio_train, y_audio_train, batch_size=2, epochs=15, validation_data=(X_audio_validation, y_audio_validation), verbose=0)
        
        audio_model.save("Models/audioClassifier.model")

    t1 = threading.Thread(target=train_audio)
    t1.start()

    # Image train
    X_image_train = X_image_train.astype("float32")/255.0
    X_image_validation = X_image_validation.astype("float32")/255.0

    image_input_shape = (X_image_train.shape[1:])
    image_model = build_image_model(image_input_shape)

    image_optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    image_model.compile(optimizer=image_optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    # image_model.summary()

    image_history = image_model.fit(X_image_train, y_image_train, batch_size=2, epochs=15, validation_data=(X_image_validation, y_image_validation), verbose=0)

    image_model.save("Models/imageClassifier.model")

    # return audio_history, image_history

# Generate combined prediction
def Predictions(audio, image):
    percentages = list(map(lambda x, y: abs(x)+abs(y), audio, image))
    return audio, image, percentages

# Predict on test data
def Predict(test_mode, data):
    IMG_SIZE = 48

    # Load test Json data
    def load_test_data(data_path):
        with open(data_path, "r") as fp:
            data = json.load(fp)
            
        X_a = np.array(data["mfcc"])
        X_i = np.array(data["image"])
        return X_a, X_i

    # Classifier models
    if test_mode == "-1":
        image_model = tf.keras.models.load_model('Models/CNNimageClassifierV1.model')
        audio_model = tf.keras.models.load_model('Models/CNNaudioClassifierV1.model')
    else:
        image_model = tf.keras.models.load_model('Models/imageClassifier.model')
        audio_model = tf.keras.models.load_model('Models/audioClassifier.model')

    # Retrive data
    audio, image = load_test_data(data)

    # Fit audio data
    audio = audio[..., np.newaxis]
    s = (-1, audio.shape[1], audio.shape[2], 1)
    a = audio.reshape(s)

    # Predict audio
    audio_predictions = audio_model.predict(a)

    # Fit image data
    image = np.array(image).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    image = image.astype("float32")/255.0

    # Predict image
    image_predictions = image_model.predict(image)

    return Predictions(audio_predictions[0], image_predictions[0])

# Test model
def Test(test_mode):
    Photo_taker(tk.Tk(),'Take Photo', "App_Data/Test/Raw/Image/", 1, True)
    Audio_recorder(tk.Tk(), 'Audio Recorder', "App_Data/Test/Preprocessed/Audio/", 1, True)
    crop_faces('App_Data/Test/Raw/Image', 'App_Data/Test/Preprocessed/', False)
    Process("App_Data/Test/Preprocessed/Audio/test.wav", "App_Data/Test/Preprocessed/Image/test.jpg", "JSON_files/TestData.json", True)
    audio_result, image_result, combined_result = Predict(test_mode, "JSON_files/TestData.json")
    again = Result(tk.Tk(), 'Results', audio_result, image_result, combined_result)
    if str(again) == "y":
        Test(test_mode)

# Main
if __name__ == "__main__":
    sample_test = str(Start(tk.Tk(), 'Emotion Chatbot'))
    if not sample_test == "-1":
        Photo_taker(tk.Tk(),'Take Photo', "App_Data/Training/Raw/Image/", int(sample_test), False)
        Audio_recorder(tk.Tk(), 'Audio Recorder', "App_Data/Training/Raw/Audio/", int(sample_test), False)
        print("Processing...")
        def augment_audio():
            augment_audio_data("App_Data/Training/Raw/Audio", "App_Data/Training/Preprocessed/")
        t1 = threading.Thread(target=augment_audio)
        t1.start()
        augment_image_data("App_Data/Training/Raw/Image", "App_Data/Training/Augmented/")
        crop_faces('App_Data/Training/Augmented/Image', 'App_Data/Training/Preprocessed/', True)
        crop_faces('App_Data/Training/Raw/Image', 'App_Data/Training/Preprocessed/', False)
        Process("App_Data/Training/Preprocessed/Audio", "App_Data/Training/Preprocessed/Image", "JSON_files/TrainData.json", False)
        Train_models("JSON_files/TrainData.json", 48)
    Test(sample_test)