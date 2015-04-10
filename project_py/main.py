#!/usr/bin/env python

import logging, logging.config
import numpy as np
import yaml
import cv2
import os

BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, 'logging.yml')) as infile:
    logging.config.dictConfig(yaml.load(infile))

if __name__ == '__main__':
