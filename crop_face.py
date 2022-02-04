import cv2
import dlib
import os

detector = dlib.get_frontal_face_detector()
new_path = 'EmotionDataset/Image/Sad/'
old_path = 'RawEmotionData/Sad'

def save(img,name, bbox, width=180,height=227):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    imgCrop = cv2.resize(imgCrop, (width, height))
    cv2.imwrite(name+".jpg", imgCrop)

def faces():
    for root, dirs, files in os.walk(old_path):
        for i, file in enumerate(files):
            frame =cv2.imread(os.path.join(old_path, file))
            gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()
                save(gray,new_path+str(i+1),(x1,y1,x2,y2))
            
            if not faces:
                print("No face found: "+ file)
            else:
                print("done saving: "+ file)
faces()