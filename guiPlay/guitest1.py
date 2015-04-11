#!/usr/bin/env python

import cv2
import numpy as np 
# import matplotlib.pyplot as plt

# mouse callback function
def mouseCallback(event,x,y,flags,param):
    global sx, sy, ex, ey, drawing, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        sx, sy, ex, ey, drawing = x, y, x, y, True
        print "button done: {}, {}".format(sx, sy)
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.namedWindow('Crop')
        if (sx != x and sy != y):
            if (sx > x): sx, x = x, sx
            if (sy > y): sy, y = y, sy
            print "buttonup: " + repr(sx) +"," + repr(sy) + " - " + repr(x) + "," + repr(y)
            # cv2.imshow('Crop', frame[sx:x, sy:y])
    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            print "mouse move: {}, {}".format(x,y)
            ex, ey = x,y

def drawImage(img):
    if drawing:
        img2 = np.copy(img)
        cv2.imshow('Movie', img2)
    else:   
        cv2.imshow('Movie', img);


# Create a black image, a window and bind the function to window
# img = np.zeros((512,512,3), np.uint8)
# cv2.namedWindow('image')

drawing = False
cap = cv2.VideoCapture('./clip_test.m4v')

cv2.namedWindow('Movie')
cv2.setMouseCallback('Movie', mouseCallback)



ret = 1
while ret:
    ret, frame = cap.read()
    cv2.imshow('Movie', frame);
    if (cv2.waitKey(30) >= 0): break;
cap.release()



# while(1):
#     cv2.imshow('image',img)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

