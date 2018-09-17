# import the packeages
import imutils
from imutils import face_utils
import numpy as np
import dlib
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-pred", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

