import os
import cv2

PATH = "RawEmotionData/Sad" 

camera = cv2.VideoCapture(0)

f = open("RawEmotionData/current_image_num.txt", "r")
image_num = int(f.read())
if image_num > 101:
    print("done")
filename = str(image_num)+".jpg"

while True:
    return_value,image = camera.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('image',gray)
    if cv2.waitKey(1)& 0xFF == ord('s'):
        cv2.imwrite(os.path.join(PATH, filename),image)
        break

file = open("RawEmotionData/current_image_num.txt", "w") 
file.write(str(image_num+1)) 
file.close() 

camera.release()
cv2.destroyAllWindows()