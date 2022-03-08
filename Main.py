import json
import math
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
from Windows.Result_window import Result
from Windows.Start_window import Start
from Windows.Audio_window import Audio_recorder
from Windows.Photo_window import Photo_taker
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Shift
import librosa
import os
import dlib
import cv2
import tkinter as tk
import soundfile as sf
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def save_face(img,name, bbox, i, width=48,height=48):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    try:
        imgCrop = cv2.resize(imgCrop, (width, height))
        cv2.imwrite(os.path.join(name, i), imgCrop)
    except Exception as e:
        print(str(e) +"\n couldnt resize: "+ i)

def crop_faces(old_path, new_path, clear):
    detector = dlib.get_frontal_face_detector()

    if clear:
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(new_path+"Image/")):
            for f in filenames:
                os.remove(os.path.join(dirpath, f))

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

def augment_audio_data(path, aug_path):
    augment = Compose([
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5)
    ])

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path)):
        for f in filenames:
            signal, sr = librosa.load(os.path.join(dirpath, f))
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            for count in range(0, 10):
                label = semantic_label + "/" + str(count) + "_" + f
                augmented_signal = augment(signal, sr)
                sf.write(os.path.join(aug_path, label), augmented_signal, sr)

def augment_image_data(path, aug_path):
    datagen = ImageDataGenerator(rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(aug_path+"Image/")):
        for f in filenames:
            os.remove(os.path.join(dirpath, f))
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path)):
        for f in filenames:
            pic = load_img(os.path.join(dirpath, f))
            pic_array = img_to_array(pic)

            X = pic_array.reshape((1,) + pic_array.shape) 

            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
                
            f_s = f.split(".")
            
            for x, val in zip(datagen.flow(X, batch_size=5, save_to_dir=os.path.join(aug_path, semantic_label), save_prefix=f_s[0], save_format=f_s[1]),range(9)):     
                pass

# Process audio and image data - store in data.json file
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

def Preprocess(test):
    if test:
        crop_faces('App_Data/Test/Raw/Image', 'App_Data/Test/Preprocessed/', False)
    else:
        augment_image_data("App_Data/Training/Raw/Image", "App_Data/Training/Augmented/")
        augment_audio_data("App_Data/Training/Raw/Audio", "App_Data/Training/Preprocessed/")
        crop_faces('App_Data/Training/Augmented/Image', 'App_Data/Training/Preprocessed/', True)
        crop_faces('App_Data/Training/Raw/Image', 'App_Data/Training/Preprocessed/', False)

def Train_models(DATA_PATH, IMG_SIZE):

    def load_audio_data(data_path):
        with open(data_path, "r") as fp:
            data = json.load(fp)
            
        X = np.array(data["mfcc"])
        y = np.array(data["labels"])
        return X, y

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
    
    # CNN model
    def build_model(input_shape):
        model = keras.Sequential()
        
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))                                
        model.add(keras.layers.Dropout(0.25))  
        
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))   
        model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))                                
        model.add(keras.layers.Dropout(0.25))
        
        model.add(keras.layers.Flatten())
            
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(480, activation='relu'))
        model.add(keras.layers.Dropout(0.05))
        
        model.add(keras.layers.Dense(3, activation='softmax'))
        
        return model

    # Train/Test Split data
    validation_size = 0.2

    X_audio, y_audio = load_audio_data(DATA_PATH)
    X_image, y_image = load_image_data(DATA_PATH)

    X_audio_train, X_audio_validation, y_audio_train, y_audio_validation = prepare_datasets(validation_size, X_audio, y_audio, True)

    X_image_train, X_image_validation, y_image_train, y_image_validation = prepare_datasets(validation_size, X_image, y_image, False)

    # Audio train
    audio_input_shape = (X_audio_train.shape[1], X_audio_train.shape[2], X_audio_train.shape[3])
    audio_model = build_model(audio_input_shape)

    audio_optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    audio_model.compile(optimizer=audio_optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    # audio_model.summary()

    audio_history = audio_model.fit(X_audio_train, y_audio_train, batch_size=2, epochs=20, validation_data=(X_audio_validation, y_audio_validation), verbose=0)

    # Image train
    X_image_train = X_image_train.astype("float32")/255.0
    X_image_validation = X_image_validation.astype("float32")/255.0

    image_input_shape = (X_image_train.shape[1:])
    image_model = build_model(image_input_shape)

    image_optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    image_model.compile(optimizer=image_optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    # image_model.summary()

    image_history = image_model.fit(X_image_train, y_image_train, batch_size=2, epochs=20, validation_data=(X_image_validation, y_image_validation), verbose=0)

    # Save models
    audio_model.save("Models/audioClassifier.model")
    image_model.save("Models/imageClassifier.model")

    # return audio_history, image_history

def Predictions(audio, image):
    percentages = []
    happy = float(abs(image[0])) + float(abs(audio[0]))
    percentages.append(happy)
    neutral = float(abs(image[1])) + float(abs(audio[1]))
    percentages.append(neutral)
    sad = float(abs(image[2])) + float(abs(audio[2]))
    percentages.append(sad)
    return audio, image, percentages

def Predict():
    IMG_SIZE = 48

    # Load Json data
    def load_test_data(data_path):
        with open(data_path, "r") as fp:
            data = json.load(fp)
            
        X_a = np.array(data["mfcc"])
        X_i = np.array(data["image"])
        return X_a, X_i
    
    # CLassifier models
    image_model = tf.keras.models.load_model('Models/imageClassifier.model')
    audio_model = tf.keras.models.load_model('Models/audioClassifier.model')

    # Retrive data
    audio, image = load_test_data("JSON_files/TestData.json")

    # Fit audio data
    audio = audio[..., np.newaxis]
    # Predict audio
    audio_predictions = audio_model.predict(audio)

    # Fit image data
    image = np.array(image).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    image = image.astype("float32")/255.0
    # Predict image
    image_predictions = image_model.predict(image)

    return Predictions(audio_predictions[0], image_predictions[0])

def get_num_of_samples():
    wrong = True

    while wrong:
        numberOfSamples = int(input("how many samples do you want to take out of 2-10: "))
        if (numberOfSamples > 1 and numberOfSamples < 11):
            wrong = False
    return numberOfSamples

def Test():
    # Photo_taker(tk.Tk(),'Take Photo', True)
    # Audio_recorder(tk.Tk(), 'Audio Recorder', True)
    # Preprocess(True)
    # Process("App_Data/Test/Preprocessed/Audio/test.wav", "App_Data/Test/Preprocessed/Image/test.jpg", "JSON_files/TestData.json", True)
    audio_result, image_result, combined_result = Predict()
    again = Result(tk.Tk(), audio_result, image_result, combined_result)
    if str(again) == "y":
        Test()

if __name__ == "__main__":
    sample_num = get_num_of_samples()
    # Start(tk.Tk(), 'Emotion Chatbot')
    # Photo_taker(tk.Tk(),'Take Happy Photo 1/10', False)
    # Audio_recorder(tk.Tk(), 'Audio Recorder', False)
    # Preprocess(False)
    # Process("App_Data/Training/Preprocessed/Audio", "App_Data/Training/Preprocessed/Image", "JSON_files/TrainData.json", False)
    # Train_models("JSON_files/TrainData.json", 48)
    # Test()


    #? if augmented data doesnt equal 110 then limit to smallest dataset
    #? Allow user to choose number of samples they take
    #? Have a test set mode
    #TODO create a training enviroment to train the model on around 300 samples find best model for task (Compare MLP, CNN, other)
    #TODO create junit tests
    #! add play bot reply button if clicked again dont reprocess just play audio clip
