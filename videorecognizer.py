# from distutils.command.config import config
# from pyexpat import model
# from pyexpat import model

import cv2
import matplotlib.pyplot as plt 
import urllib.request as ureq
import numpy as np

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_file)

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

classLabels = []
file_names = 'labels.txt'
with open(file_names,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


video_url = 'https://youtu.be/K4NiaXmXIhE'
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

font_scale = 1
font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

ret,frame = cap.read()
ClassIndex,confidence,bbox = model.detect(frame,confThreshold=0.55)
print(ClassIndex, confidence, bbox)
if len(ClassIndex) != 0:
    for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
        if(ClassInd <= 80):
            cv2.rectangle(frame,boxes,(255,0,0),2)
            cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=2)
# cv2.imshow(frame)
plt.imshow(frame)
plt.show()
# while True:
#     ret,frame = cap.read()
#     ClassIndex,confidence,bbox = model.detect(frame,confThreshold=0.55)
#     print(ClassIndex, confidence, bbox)

#     if len(ClassIndex) != 0:
#         for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
#             if(ClassInd <= 80):
#                 cv2.rectangle(frame,boxes,(255,0,0),2)
#                 cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=2)
#     # cv2.imshow(frame)
#     plt.imshow(frame)
#     plt.show()
#     if cv2.waitKey(2) & 0xFF == ord('q'):
#         break

cap.release()
cv2.destroyAllWindows()