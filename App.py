# Imports
import pygame
import json
import math
import time
import wave
import librosa, librosa.display
import numpy as np
import tensorflow as tf
import cv2
import dlib
import pyaudio
from random import randint
from pydub import AudioSegment
from pydub.playback import play
from tkinter import W

# Constant variables
# Pygame var
pygame.init()
RES = (400,600)
SCREEN = pygame.display.set_mode(RES)
WIDTH = SCREEN.get_width()
HEIGHT = SCREEN.get_height()
  
WHITE = (255, 255, 255)
RED = (200, 0, 0)
DARK_RED = (100, 0, 0)
GREY = (50, 50, 50)
BLACK = (0, 0, 0)
  
LARGEFONT = pygame.font.SysFont('Corbel',55)
MEDIUMFONT = pygame.font.SysFont('Corbel',35)
SMALLFONT = pygame.font.SysFont('Corbel',25)

# CLassifier models
IMAGE_MODEL = tf.keras.models.load_model('CNNimageClassifier.model')
AUDIO_MODEL = tf.keras.models.load_model('CNNaudioClassifier.model')
JSON_FILE = "json_storage/app.json"
#Image var
IMG_SIZE = 48
IMAGE_FILE_PATH = "AppStorage/face_image.jpg"
# Audio var
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 22050
RECORD_SECONDS = 2
SAMPLES_PER_TRACK = RATE * RECORD_SECONDS
AUDIO_FILE_PATH = "AppStorage/audio.wav"
WORDS = ["ant", "car", "bar", "date", "egg", "young", "white", "voice", 
        "thought", "sheep", "rain", "pain", "hug", "make", "lean", "keep", "jail", 
        "half", "gap", "fall", "cause", "bath", "mend", "kick", "tree", "lamp", 
        "pen", "sad", "happy", "neutral", "far", "dream", "great",  "fate"]

# Process the test data
def process(audio_path, image_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
    data = {
        "mfcc": [],
        "image": []
    }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    try:
        signal, sr = librosa.load(audio_path, sr=RATE)
    except Exception as e:                                                    
        print('Audio failed to process: ' + e)
    
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

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)    

# Load Json data
def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
        
    X_a = np.array(data["mfcc"])
    X_i = np.array(data["image"])
    return X_a, X_i

# Record 2 second audio
def record():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording")
    time.sleep(0.5)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        print(i)
        data = stream.read(CHUNK)
        frames.append(data)

    print("Done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(AUDIO_FILE_PATH, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Repeat record till correct
def recording(again=True):
    ran = randint(0, len(WORDS))
    while again:
        input("Say the phrase 'say the word "+WORDS[ran]+"' (in a happy, neutral or sad way)")
        record()

        song = AudioSegment.from_wav(AUDIO_FILE_PATH)
        play(song)

        a = input("REC AGAIN? (y/enter): ")
        if a == "y":
            again = True
        else:
            again = False

# Take picture
def take_pic():
    camera = cv2.VideoCapture(0)

    filename = "AppStorage/original_face_image.jpg"

    while True:
        return_value,image = camera.read()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',gray)
        if cv2.waitKey(1)& 0xFF == ord('s'):
            cv2.imwrite(filename,image)
            break

    camera.release()
    cv2.destroyAllWindows()
    crop_pic()

# Crop face
def crop_pic():
    detector = dlib.get_frontal_face_detector()

    oldfile = "AppStorage/original_face_image.jpg"

    def save(img, bbox, width=180,height=227):
        x, y, w, h = bbox
        imgCrop = img[y:h, x: w]
        imgCrop = cv2.resize(imgCrop, (width, height))
        cv2.imwrite(IMAGE_FILE_PATH, imgCrop)

    frame =cv2.imread(oldfile)
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        save(gray,(x1,y1,x2,y2))

    if not faces:
        print("No face found")
        take_pic()
    else:
        print("done saving")

# Predict emotion on data
def emotion_classifier():
    # Process data
    process(AUDIO_FILE_PATH, IMAGE_FILE_PATH, JSON_FILE, num_segments=1)

    # Retrive data
    audio, image = load_data(JSON_FILE)

    # Fit audio data
    audio = audio[..., np.newaxis]
    # Predict audio
    audio_predictions = AUDIO_MODEL.predict(audio)

    # Fit image data
    image = np.array(image).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    image = image.astype("float32")/255.0
    # Predict image
    image_predictions = IMAGE_MODEL.predict(image)

    return audio_predictions[0], image_predictions[0]

# Use both results to give final prediction
def combined_emotion_predicition():
    audio, image = emotion_classifier()
    print("Audio: "+str(audio))
    print("Image: "+str(image))
    result = ""
    happy = float(abs(image[0])) + float(abs(audio[0]))
    neutral = float(abs(image[1])) + float(abs(audio[1]))
    sad = float(abs(image[2])) + float(abs(audio[2]))

    if happy > neutral and happy > sad:
        result += "Happy"
    elif neutral > happy and neutral > sad:
        result += "Neutral"
    elif sad > happy and sad > neutral:
        result += "Sad"
    elif happy > neutral and happy == sad:
        result += "Happy or Sad"
    elif happy > sad and happy == neutral:
        result += "Happy or Neutral"
    elif neutral > happy and neutral == sad:
        result += "Neutral or Sad"
    else:
        result += "Happy or Neutral or Sad"

    return "Prediction: "+result+ " : "+str(happy)+", "+str(neutral)+", "+str(sad)

window = True
page = 0
record_button = False

quit_button = MEDIUMFONT.render('QUIT' , True , WHITE)
quit_height = 1.1
next_button = MEDIUMFONT.render('NEXT' , True , WHITE)
next_height = 1.25

while window:
      
    for ev in pygame.event.get():
          
        if ev.type == pygame.QUIT:
            window = False
            pygame.quit()
              
        #checks if a mouse is clicked
        if ev.type == pygame.MOUSEBUTTONDOWN:
              
            if WIDTH/2-70 <= mouse[0] <= WIDTH/2+70 and HEIGHT/quit_height-20 <= mouse[1] <= HEIGHT/quit_height+20:
                window = False
                pygame.quit()
            elif WIDTH/2-70 <= mouse[0] <= WIDTH/2+70 and HEIGHT/next_height-20 <= mouse[1] <= HEIGHT/next_height+20:
                page += 1
                  
    # fills the screen with a color
    SCREEN.fill((WHITE))
      
    mouse = pygame.mouse.get_pos()
      
    if WIDTH/2-70 <= mouse[0] <= WIDTH/2+70 and HEIGHT/quit_height-20 <= mouse[1] <= HEIGHT/quit_height+20:
        pygame.draw.rect(SCREEN,DARK_RED,[WIDTH/2-70,HEIGHT/quit_height-20,140,40])
    else:
        pygame.draw.rect(SCREEN,RED,[WIDTH/2-70,HEIGHT/quit_height-20,140,40])

    if WIDTH/2-70 <= mouse[0] <= WIDTH/2+70 and HEIGHT/next_height-20 <= mouse[1] <= HEIGHT/next_height+20:
        pygame.draw.rect(SCREEN,GREY,[WIDTH/2-70,HEIGHT/next_height-20,140,40])
    else:
        pygame.draw.rect(SCREEN,BLACK,[WIDTH/2-70,HEIGHT/next_height-20,140,40])

    def main():
        text_rect = quit_button.get_rect(center=(WIDTH/2, HEIGHT/quit_height))
        SCREEN.blit(quit_button, text_rect)
        text_rect = next_button.get_rect(center=(WIDTH/2, HEIGHT/next_height))
        SCREEN.blit(next_button, text_rect)


        if page == 0:
            inital_text_l1 = LARGEFONT.render('EMOTION', True, BLACK)
            inital_text_l2 = LARGEFONT.render('PREDICTION', True, BLACK)
            text_rect = inital_text_l1.get_rect(center=(WIDTH/2, HEIGHT/6))
            SCREEN.blit(inital_text_l1, text_rect)
            text_rect = inital_text_l2.get_rect(center=(WIDTH/2, HEIGHT/6+65))
            SCREEN.blit(inital_text_l2, text_rect)
        elif page == 1:
            explanation_l1 = SMALLFONT.render('This application uses a CNN', True, BLACK)
            explanation_l2 = SMALLFONT.render('deep learning model to predict', True, BLACK)
            explanation_l3 = SMALLFONT.render('human emotion through image', True, BLACK)
            explanation_l4 = SMALLFONT.render('and audio data', True, BLACK)
            text_rect = explanation_l1.get_rect(center=(WIDTH/2, HEIGHT/6))
            SCREEN.blit(explanation_l1, text_rect)
            text_rect = explanation_l2.get_rect(center=(WIDTH/2, HEIGHT/6+30))
            SCREEN.blit(explanation_l2, text_rect)
            text_rect = explanation_l3.get_rect(center=(WIDTH/2, HEIGHT/6+60))
            SCREEN.blit(explanation_l3, text_rect)
            text_rect = explanation_l4.get_rect(center=(WIDTH/2, HEIGHT/6+90))
            SCREEN.blit(explanation_l4, text_rect)
        elif page == 2:
            explanation_l1 = SMALLFONT.render('Please provide the AI with', True, BLACK)
            explanation_l2 = SMALLFONT.render('an image of your face. Choose', True, BLACK)
            explanation_l3 = SMALLFONT.render('a emotion (Happy/Neutral/sad)', True, BLACK)
            explanation_l4 = SMALLFONT.render('then press "s" on keyboard.', True, BLACK)
            text_rect = explanation_l1.get_rect(center=(WIDTH/2, HEIGHT/6))
            SCREEN.blit(explanation_l1, text_rect)
            text_rect = explanation_l2.get_rect(center=(WIDTH/2, HEIGHT/6+30))
            SCREEN.blit(explanation_l2, text_rect)
            text_rect = explanation_l3.get_rect(center=(WIDTH/2, HEIGHT/6+60))
            SCREEN.blit(explanation_l3, text_rect)
            text_rect = explanation_l4.get_rect(center=(WIDTH/2, HEIGHT/6+90))
            SCREEN.blit(explanation_l4, text_rect)
        elif page == 3:
            window = False
            pygame.quit()
            take_pic()
            # combined_emotion_predicition()

    
    main()

    pygame.display.update()