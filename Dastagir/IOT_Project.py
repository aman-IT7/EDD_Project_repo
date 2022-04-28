import cv2
import imutils
import numpy as np

from Machine_learning import HOGCV

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


webcam=cv2.VideoCapture(0)
print('Detecting People....')
while True:
    ret,frame = webcam.read()
    if ret:
        frame = imutils.resize(frame,width=min(400,frame.shape[1]))

    bounding_box_coordinates,weights = HOGCV.detectMultiScale(frame, winStride=(4,4), padding = (4,4), scale=1.05)

    person = 1
    for x,y,w,h in bounding_box_coordinates:
        cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,0),2)
        cv2.putText(frame, f'person {person}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        person+=1

    cv2.putText(frame, f'Status : Detecting',(40,40),cv2.FONT_HERSHEY_DUPLEX,0.8, (255,0,0),2)
    cv2.putText(frame, f'Total Persons: {person-1}',(40,70),cv2.FONT_HERSHEY_DUPLEX,0.8, (255,0,0),2)

    cv2.imshow('output',frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
