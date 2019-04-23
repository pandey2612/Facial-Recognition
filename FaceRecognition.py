import cv2
import numpy
import pickle

Webcam  = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

Recognizer  = cv2.face.LBPHFaceRecognizer_create()
Recognizer.read('Training.yml')

Labels = {}

with open('Label.pickle' , 'rb') as file:
    OriginalLabel = pickle.load(file)
    Labels = {v:k for k,v in OriginalLabel.items()}

while (True):
    Return ,Frame =Webcam.read()
    Frame = cv2.flip(Frame , 1)

    GrayScale = cv2.cvtColor(Frame , cv2.COLOR_BGR2GRAY)
    Faces = face_cascade.detectMultiScale(GrayScale , 1.3 ,5)
    for X,Y,W,H in Faces:
        ROI_GRAY = GrayScale[Y:Y+H , X:X+W]
        ROI_COLOR = Frame[Y:Y+H , X:X+W]

        ID , Confidence = Recognizer.predict(ROI_GRAY)
        if Confidence >= 45:
            print(Labels[ID])
            cv2.putText(Frame , Labels[ID],(X,Y) , cv2.FONT_HERSHEY_SIMPLEX , 1,(0,0,255),2)
        cv2.rectangle(Frame , (X,Y) , (X+W , Y+H) , (255 , 0 , 0) , 1)

    cv2.imshow('Image'  , Frame)
    if cv2.waitKey(1) == ord('q'):
        break
Webcam.release()
cv2.destroyAllWindows()
    