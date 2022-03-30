# Emotion-Classification-using-audio-and-image-data

## Introduction
This paper presents a framework for improving a chatbot's understanding of emotionally charged questions. Through facial expression (webcam) and verbalisation (microphone), a reply can be constructed with the asker’s respective emotion and question. This concept has been developed and tested in a proof-of-concept study. The main goal of this study was to investigate if image and audio data provided a better accuracy score together or individually. This concept was implemented into a product allowing for a chatbot to provide emotionally charged responses. The software was calibrated by taking ten images and recordings of each emotion. In this case, three emotions were used: happy, neutral and sad. This was validated by sampling either a happy, neutral or sad picture and question. This data would be used to predict the user’s emotion before being passed through a chatbot to establish a reply. The overall accuracy concluded that the combined prediction was ten per cent more accurate on average than the individual results. While most existing software only allows for a single data type to train on, this software allows for two. This paves the way for enhancing the response by providing a better chance at allocating the correct emotion.

## How to use
To use this application run the main.py file. Read the start window to understand application. Dont click test mode as this will not work for you, instead enter the number of samples you want to take and click submit. Now take your samples using the promted on screen emotions. Allow time for models to be trained, this can be for a few second to a few minuets depending on computers power. Fianlly, take test samples and wait for result window to show you how well it performed at predicting your emotion. To hear the chatbots reply click the reply button.

### Required Packages
cv2,
dlib,
json,
librosa,
math,
numpy,
os,
soundfile,
threading,
tensorflow,
keras,
tkinter,
audiomentations,
sklearn,
pyaudio,
time,
wave,
pydub,
PIL,
datetime,
gtts,
speech_recognition,
playsound,
