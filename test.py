#cython: language_level=3, boundscheck=False
# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

import argparse
from time import time ,sleep

import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import FileVideoStream, VideoStream
# import the necessary packages
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,#
	help="path to facial landmark predictor")
# ~ ap.add_argument("-v", "--video", type=str, default="",#
	# ~ help="path to input video file")
args ={"shape_predictor":"shape_predictor_68_face_landmarks.dat"}# vars(ap.parse_args())
 
 
print("#######################################################\n")
print(args)
print("\n#####################################################")
 
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 2

# initialize the frame counters and the total number of blinks
sleep(1.0)
StartTime=time()
lastTime=StartTime
timeRecorder=StartTime
timeDifference=0.0
frameCounter=0.0
FPS=0.0
lastEyeStatetime=0.0
fristTimeNoFace=True

COUNTER = 0
TOTAL = 0
lastEyeState =0



# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# ~ vs = FileVideoStream(args["video"]).start()
# ~ fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# ~ fileStream = False

# # # # # StartTime = currentTime = time()
# # # # # lastTime = time()
# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	# ~ if fileStream and not vs.more():
		# ~ break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	#print rects
	if len(rects) == 0 :
		pass
		if fristTimeNoFace :
			pass
			StartTime= time()
			fristTimeNoFace=False
			#lastTime=time() - StartTime
			print(lastTime)
			
		else:
			pass
			lastTime= time() - StartTime
			print(lastTime)
			print("after 7 second the car will stop !!!! \n\a")
			if  lastTime >=7 :#and lastTime <7
				pass
				print("Emergency mode activated !!!! \n\a")
				break
			# elif  lastTime >7:
			# 	pass
			# 	
				
				
			
		
		sleep(0.5)
		continue
			


	#else :
		#
		#	print("if you aren\'t okay activate self driving mode !!!!\n\a")
		#StartTime=time()
	fristTimeNoFace =True
	#elif
	# loop over the face detections
	#for rect in rects:
	#while(1):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rects[0])
	shape = face_utils.shape_to_np(shape)
	#print shape

	# extract the left and right eye coordinates, then use the
	# coordinates to compute the eye aspect ratio for both eyes
	if len(rects)>1:
		driverFaceNum=0
		currentLeftI=[]
		driverLeftI=[]
		driverIWidth=0
		currentIWidth=0
		for faceNum in shape:
			currentLeftI=shape[lStart:lEnd]
			currentIWidth=dist.euclidean(currentLeftI[0], currentLeftI[3])
			if driverIWidth < currentIWidth:
				driverFaceNum = faceNum

		leftEye = shape[driverFaceNum][lStart:lEnd]
		rightEye = shape[driverFaceNum][rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)		

	
	else:
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
	
	# ~ print(leftEye,rightEye)
	
	# average the eye aspect ratio together for both eyes
	ear = (leftEAR + rightEAR) / 2.0

	# compute the convex hull for the left and right eye, then
	# visualize each of the eyes
	leftEyeHull = cv2.convexHull(leftEye)
	rightEyeHull = cv2.convexHull(rightEye)
	cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
	cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

	# check to see if the eye aspect ratio is below the blink
	# threshold, and if so, increment the blink frame counter
	if ear < EYE_AR_THRESH:
		COUNTER += 1
		lastEyeState += 1
		if lastEyeState >80:
			lastEyeState=80
					


	# otherwise, the eye aspect ratio is not below the blink
	# threshold
	else:
		# if the eyes were closed for a sufficient number of
		# then increment the total number of blinks
		
		lastEyeState -= 3
		if lastEyeState<0:
			lastEyeState =0

		if COUNTER >= 0.1*FPS:#EYE_AR_CONSEC_FRAMES
			TOTAL += 1
			
		# reset the eye frame counter
		COUNTER = 0

				
	#lastEyeStatetime= 

	if lastEyeState >= 30  :#and lastEyeState < 30
		print("!!!Take care  please lock on the road now \n\a")
		#lastTime=time() - StartTime

		if lastEyeState >= 40 :

			if lastEyeState >= 50 :
				print("\a!!!!Emergency mode activated  \n\a")
			else:
				print("!!!!after 3 second the car will stop  \n\n \a")
			#fristTimeNoFace =True
			#	print("if you aren\'t okay activate self driving mode !!!!\n\a")

	timeDifference =time() - timeRecorder
	if timeDifference < 0.25:
		frameCounter +=1
		
	else:
		FPS=frameCounter/timeDifference
		timeRecorder=time()
		frameCounter=0
		


	# draw the total number of blinks on the frame along with
	# the computed eye aspect ratio for the frame
	cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	
	cv2.putText(frame, "FPS: {:.1f}".format(FPS), (155, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	#StartTime=time()
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
