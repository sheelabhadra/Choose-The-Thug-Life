# import the packages
import imutils
from imutils import face_utils
import numpy as np
import dlib
import cv2
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# dlib's face detector (HOG + Linear SVM)
detector = dlib.get_frontal_face_detector()
# shape_predictor_68_face_landmarks.dat needs the absolute path
predictor = dlib.shape_predictor(os.path.abspath("models/shape_predictor_68_face_landmarks.dat"))

# input image
img = cv2.imread(args["image"])
img = imutils.resize(img, width=500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
faces = detector(gray, 1)

for (i,face) in enumerate(faces):
	# determine the facial landmarks for the face region, then convert the facial landmark 
	# (x, y)-coordinates to a NumPy array
	shape = predictor(gray, face)
	shape = face_utils.shape_to_np(shape)

	# convert dlib's bbox format to OpenCV format
	(x,y,w,h) = face_utils.rect_to_bb(face)
	cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

	# draw facial landmarks
	for (x,y) in shape:
		cv2.circle(img, (x,y), 1, (0,0,255), -1)

cv2.imshow("Faces", img)
cv2.waitKey(0)
