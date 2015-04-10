#!/usr/bin/env python

import logging, logging.config
import numpy as np
import yaml
import cv2
import os

BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, 'logging.yml')) as infile:
    logging.config.dictConfig(yaml.load(infile))

logger = logging.getLogger(__name__)

if __name__ == '__main__':

	import argparse

	# Argument Parsing

	parser = argparse.ArgumentParser(
		description = 'COMP9517 Project (Part 1)',
        version = '1.0'
	)

	# Positional arguments
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

	detector = cv2.SIFT()
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
	
	train_keypoints = map(detector.detect, train_imgs)
	train_keypoints, train_descriptors = zip(*map(detector.compute, train_imgs, train_keypoints))

	# equivalent to
	# zip(*map(detector.detectAndCompute, train_imgs, [None for _ in train_imgs]))

	# Note that `map(detector.compute, train_imgs, train_keypoints)` returns
	# a list of pairs consisting of keypoints and descriptors for each image,
	# whereas we actually want a separate list of keypoints and list of descriptors.
	# So we need to apply some sort of inverse `zip` function to the result of the
	# `map`. Recall that `zip` is actually its own inverse. E.g.
	# zip(['a', 'b', 'c'], [1, 4, 6, 8, 9]) -> [('a', 1), ('b', 4), ('c', 6)]
	# zip(('a', 1), ('b', 4), ('c', 6)) -> [('a', 'b', 'c'), (1, 4, 6, 8, 9)]

	print train_keypoints
	print train_descriptors

	exit(0)

	# The VideoCapture class initialization is overloaded
	# to accommodate video filename or device number
	# If the string argument provided can be cast to an 
	# integer, we interpret it as device number, otherwise
	# it is a video filename.
	try:
		input_video = int(args.input_video)
	except ValueError:
		input_video = args.input_video

	cap = cv2.VideoCapture(input_video)

	if args.output_video_file is not None:
		pass
		# TODO:
		# out = cv2.VideoWriter(filename=args.output_video_file)

	cv2.namedWindow("Movie", cv2.WINDOW_AUTOSIZE)

	while True:
	    ret, query_img = cap.read()
	    if not ret: break


	    cv2.imshow("Movie", query_img)
	    if cv2.waitKey(30) >= 0: break

	cap.release()
