import os
import cv2
import numpy as np
import time
from imutils.video import VideoStream
import imutils

base_path = os.path.dirname(os.path.abspath(__file__))

prototxt_path = os.path.join(base_path + '/model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_path+ '/model_data/weights.caffemodel')

#read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path,caffemodel_path)

print("Starting video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame,width=400)

    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0, (h,w), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    for i in range(0,detections.shape[2]):
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype("int")
        face = frame[startY:endY,startX:endX]
        confidence = detections[0,0,i,2]

        if(confidence>0.5):
            cv2.rectangle(frame,(startX,startY),(endX,endY),(255,255,255)
            ,2)
    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)&0xFF

    if key==ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
vs.stream.release()
print("Video Stream stopped.")
        
