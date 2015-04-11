#!/usr/bin/env python

import cv2
import numpy as np 
# import matplotlib.pyplot as plt

# mouse callback function
def mouseCallback(event,x,y,flags,param):
    global sx, sy, ex, ey, drawing, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        sx, sy, ex, ey, drawing = x, y, x, y, True
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.namedWindow('Crop')
        if (sx != x and sy != y):
            if (sx > x): sx, x = x, sx
            if (sy > y): sy, y = y, sy
            cv2.imshow('Crop', frame[sy:y, sx:x])
    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            ex, ey = x,y

def drawImage(img):
    global sx, sy, ex, ey, drawing
    if drawing:
        img2 = np.copy(img)
        cv2.rectangle(img2, (sx,sy), (ex,ey), (255,0,0), 2)
        cv2.imshow('Movie', img2)
    else:   
        cv2.imshow('Movie', img);

drawing = False
cap = cv2.VideoCapture('../inputs/clip_test.m4v')

cv2.namedWindow('Movie')
cv2.setMouseCallback('Movie', mouseCallback)

ret = 1
while ret:
    ret, frame = cap.read()
    drawImage(frame)
    if (cv2.waitKey(30) >= 0): break;
cap.release()



# while(1):
#     cv2.imshow('image',img)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

