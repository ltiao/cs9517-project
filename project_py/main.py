#!/usr/bin/env python

import logging, logging.config
import numpy as np
import yaml
import cv2
import os

from operator import attrgetter, itemgetter

# The FLANN Index enums aren't exposed in the OpenCV Python bindings. We create
# our own in accordance with:
# https://github.com/Itseez/opencv/blob/7d4d28605087ec2d3878f9467aea313a2acdfd49/
# modules/flann/include/opencv2/flann/defines.h#L81
FLANN_INDEX_LINEAR, FLANN_INDEX_KDTREE, FLANN_INDEX_KMEANS,	FLANN_INDEX_COMPOSITE, \
	FLANN_INDEX_KDTREE_SINGLE, FLANN_INDEX_HIERARCHICAL, FLANN_INDEX_LSH = range(7) 

NEON_GREEN = 60, 255, 20

BASE_DIR = os.path.dirname(__file__)

# Initialize logger from configuration file
with open(os.path.join(BASE_DIR, 'logging.yml')) as infile:
	logging.config.dictConfig(yaml.load(infile))

logger = logging.getLogger(__name__)

def corners(im):
	# Order matters when drawing closed polygon 
	# 0 --- 1
	# |     |
	# |     |
	# 3 --- 2
	h, w = im.shape[:2]
	return [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]

if __name__ == '__main__':

	import argparse

	# Argument Parsing

	parser = argparse.ArgumentParser(
		description = 'COMP9517 Project (Part 1)',
		version = '1.0'
	)

	# Positional (required) arguments
	parser.add_argument('input_video', 
						type=str,
						action="store", 
						help='Input video')

	# Optional arguments
	parser.add_argument('--image-filenames',
						'-i',
						default=[],
						nargs='*',
						action="store", 
						help='Filename of input images')
	parser.add_argument('--output-video-file',
						'-o',
						type=str,
						action="store", 
						help='Output video to specified file instead of displaying in window')

	# parse arguments, which by default is sys.argv
	args = parser.parse_args()
	
	logger.debug('Received arguments: {0}'.format(args))
	logger.debug('Reading training images: {0}'.format(args.image_filenames))

	# Initialize training images as those specified 
	# through command line arguments 
	train_imgs = map(cv2.imread, args.image_filenames)

	logger.debug('Read {0} training images of dimensions: {1}' \
		.format(len(train_imgs), map(lambda img: img.shape, train_imgs)))

	detector = cv2.SURF()
	# Generalized version
	# detector = cv2.FeatureDetector_create('SIFT')

	# The method `compute` requires the image and detected keypoints
	# and returns a pair consisting of keypoints and descriptors,
	# where descriptors is a len(keypoints)x128 array (a keypoint descriptor
	# is an 128 element vector.)
	# Note that the keypoints returned by `compute` may not be same
	# as the input keypoints:  "Keypoints for which a descriptor cannot 
	# be computed are removed and the remaining ones may be reordered. 
	# Sometimes new keypoints can be added, for example: SIFT duplicates 
	# a keypoint with several dominant orientations (for each orientation)."
	
	train_keypoints_lst = map(detector.detect, train_imgs)
	train_keypoints_lst, train_descriptors_lst = zip(*map(detector.compute, train_imgs, train_keypoints_lst))

	# equivalent to
	# zip(*map(detector.detectAndCompute, train_imgs, [None for _ in train_imgs]))

	# Note that `map(detector.compute, train_imgs, train_keypoints)` returns
	# a list of pairs consisting of keypoints and descriptors for each image,
	# whereas we actually want a separate list of keypoints and list of descriptors.
	# So we need to apply some sort of inverse `zip` function to the result of the
	# `map`. Recall that `zip` is actually its own inverse. E.g.
	# zip(['a', 'b', 'c'], [1, 4, 6, 8, 9]) -> [('a', 1), ('b', 4), ('c', 6)]
	# zip(('a', 1), ('b', 4), ('c', 6)) -> [('a', 'b', 'c'), (1, 4, 6, 8, 9)]

	logger.debug('Detected {0} keypoints (resp.) in each image'.format(map(len, train_keypoints_lst)))

	# The VideoCapture class initialization is overloaded
	# to accommodate video filenames or device numbers
	# If the string argument provided can be cast to an 
	# integer, we interpret it as device number, otherwise
	# it is a video filename.
	try:
		input_video = int(args.input_video)
	except ValueError:
		input_video = args.input_video

	logger.debug("Reading video file/device: {0}".format(input_video))

	cap = cv2.VideoCapture(input_video)
	
	if not cap.isOpened():
		logger.error("Couldn't read video file/device: {0}".format(input_video))
		exit(1)

	if args.output_video_file is not None:
		# TODO:
		# out = cv2.VideoWriter(filename=args.output_video_file)
		pass

	cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)

	# By default, uses L2-norm with no cross-checking
	matcher = cv2.BFMatcher()

	while True:
		
		ret, query_img = cap.read()
		
		if not ret: break

		query_keypoints, query_descriptors = detector.detectAndCompute(query_img, mask=None)

		# TODO: Loop over all train_keypoints here and decide 
		# whether and how to display all of them. Only working 
		# with the first one for right now
		train_img, train_keypoints, train_descriptors = train_imgs[0], train_keypoints_lst[0], train_descriptors_lst[0]
		
		# list of pairs of best and second best match
		top_matches = matcher.knnMatch(query_descriptors, train_descriptors, k=2)
		# filter matches
		matches = [a for a, b in top_matches if a.distance < 0.75*b.distance]

		src_pts = np.float32(map(lambda m: train_keypoints[m.trainIdx].pt, matches))
		dst_pts = np.float32(map(lambda m: query_keypoints[m.queryIdx].pt, matches))

		H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
		
		train_img_corners = np.float32(corners(train_img)).reshape(-1, 1, 2)

		transformed_train_img_corners = cv2.perspectiveTransform(train_img_corners, H)

		cv2.polylines(query_img, [np.int32(transformed_train_img_corners)], \
			isClosed=True, color=NEON_GREEN, thickness=2, lineType=cv2.CV_AA)

		# logger.debug('Detected {0} keypoints in frame'.format(len(query_keypoints)))
		# cv2.imshow("Tracking", cv2.drawKeypoints(query_img, query_keypoints))
		cv2.imshow("Tracking", query_img)
		if cv2.waitKey(1) >= 0: break

	cap.release()
