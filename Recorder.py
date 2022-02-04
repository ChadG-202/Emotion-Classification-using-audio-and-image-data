# Imports
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import tensorflow as tf
import pyaudio
import wave
import os
from pydub import AudioSegment
from pydub.playback import play

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 22050
RECORD_SECONDS = 2
SAMPLES_PER_TRACK = RATE * RECORD_SECONDS
PATH = "EmotionDataset/Audio/Sad"
OUTPUT_FILENAME = ["ant.wav", "ant2.wav", "ant3.wav", "car.wav",
 "car2.wav", "car3.wav", "bar.wav", "bar2.wav", "bar3.wav", "date.wav",
  "date2.wav", "date3.wav", "egg.wav", "egg2.wav", "egg3.wav", "young.wav",
   "young2.wav", "young3.wav", "white.wav", "white2.wav", "white3.wav",
    "voice.wav", "voice2.wav", "voice3.wav", "thought.wav", "thought2.wav",
     "thought3.wav", "sheep.wav", "sheep2.wav", "sheep3.wav", "rain.wav",
      "rain2.wav", "rain3.wav", "pain.wav", "pain2.wav", "pain3.wav",
       "hug.wav", "hug2.wav", "hug3.wav", "make.wav", "make2.wav", "make3.wav",
        "lean.wav", "lean2.wav", "lean3.wav", "keep.wav", "keep2.wav", "keep3.wav",
         "jail.wav", "jail2.wav", "jail3.wav", "half.wav", "half2.wav", "half3.wav",
          "gap.wav", "gap2.wav", "gap3.wav", "fall.wav", "fall2.wav", "fall3.wav",
            "cause.wav", "cause2.wav", "cause3.wav", "bath.wav", "bath2.wav", "bath3.wav",
             "mend.wav", "mend2.wav", "mend3.wav", "kick.wav", "kick2.wav", "kick3.wav",
              "tree.wav", "tree2.wav", "tree3.wav", "lamp.wav", "lamp2.wav", "lamp3.wav",
               "pen.wav", "pen2.wav", "pen3.wav", "sad.wav", "sad2.wav", "sad3.wav",
                "happy.wav", "happy2.wav", "happy3.wav", "neutral.wav", "neutral2.wav", "neutral3.wav",
                 "far.wav", "far2.wav", "far3.wav", "dream.wav", "dream2.wav", "dream3.wav",
                  "great.wav", "great2.wav", "great3.wav", "fate.wav", "fate2.wav", "fate3.wav"]

again = True
n = 0

print(len(OUTPUT_FILENAME))

for i in range(len(OUTPUT_FILENAME)-(n)):
    again = True
    while again:

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

            wf = wave.open(os.path.join(PATH, OUTPUT_FILENAME[n]), 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

        input(str(n)+": say the word: "+OUTPUT_FILENAME[n])

        record()

        song = AudioSegment.from_wav(os.path.join(PATH, OUTPUT_FILENAME[n]))
        play(song)

        a = input(OUTPUT_FILENAME[n]+": REC AGAIN? (y/enter): ")
        if a == "y":
            again = True
        else:
            again = False
            n +=1
        
