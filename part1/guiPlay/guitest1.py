#!/usr/bin/env python

import cv2
import numpy as np

class Selector:

	def __init__(self, selection_callback):
		self.draw_mode = False
		if selection_callback is None:
			selection_callback = lambda p1, p2: None
		self.callback = selection_callback

	def __call__(self, event, x, y, flags, param):

		if event == cv2.EVENT_LBUTTONDOWN:
			self.x0, self.y0 = self.x1, self.y1 = x, y
			self.draw_mode = True

		if event == cv2.EVENT_LBUTTONUP:
			self.draw_mode = False
			if self.x0 != x and self.y0 != y:
				if self.x0 > x: 
					self.x0, x = x, self.x0
				if self.y0 > y: 
					self.y0, y = y, self.y0
				self.x1, self.y1 = x, y

			print (self.x0, self.y0), (self.x1, self.y1)

		if event == cv2.EVENT_MOUSEMOVE:
			if self.draw_mode:
				self.x1, self.y1 = x, y

class SelectorContext:

	def __init__(self, winname):
		self.selector = Selector()
		self.winname = winname

	def imshow(self, img):
		if self.selector.draw_mode:
			p0 = (self.selector.x0, self.selector.y0)
			p1 = (self.selector.x1, self.selector.y1)
			cv2.rectangle(img, p0, p1, (255, 0, 0), 2)
		else:
			if self.selector.last_event == cv2.EVENT_LBUTTONUP:
				img_new = np.Cropy(img[self.selector.y0:self.selector.y1, self.selector.x0:self.selector.x1])
				cv2.imshow("Crop", img_new);
		cv2.imshow(self.winname, img)

	def __enter__(self):
		cv2.setMouseCallback(self.winname, self.selector)
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		pass

cap = cv2.VideoCapture('../inputs/clip_test.m4v')

cv2.namedWindow('Movie')
with SelectorContext('Movie') as c:
	ret = 1
	while ret:
		ret, frame = cap.read()
		c.imshow(frame)
		if (cv2.waitKey(30) >= 0): break;
	cap.release()

# while(1):
#     cv2.imshow('image',img)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

