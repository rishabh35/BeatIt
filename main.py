from collections import deque
import numpy as np
import argparse
import imutils
import threading
import time 
import sys

def nothing(x):
    pass
ar = []

myrecording = []
p = 1
lock = threading.Lock()

def mainloop():
		
		mod1 = 7
		mod2 = 8
		kc = 1

		import cv2
		# cv2.namedWindow('Threshold1')
		# mask = None
		# count = 0
		# cv2.createTrackbar('Hue       ', 'Threshold1',0,179,nothing)
		# cv2.createTrackbar('Saturation', 'Threshold1',0,255,nothing)
		# cv2.createTrackbar('Value     ', 'Threshold1',0,255,nothing)
		
		# while(True):
		#     ret, frame = camera.read() 
		#     if ret == True:
		#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
		#         frame = cv2.flip(frame,1) 
		#         hsv = cv2.flip(hsv,1) 
		#         h = cv2.getTrackbarPos('Hue       ','Threshold1')
		#         s = cv2.getTrackbarPos('Saturation','Threshold1')
		#         v = cv2.getTrackbarPos('Value     ','Threshold1')  
		#         lower = np.array([h-10,s,v], dtype = np.uint8)
		#         upper = np.array([h+10,255,255], dtype = np.uint8)
		#         mask = cv2.inRange(hsv, lower, upper)     
		#         mask = cv2.bitwise_and(hsv,hsv,mask = mask)     
		#         cv2.imshow("Threshold1", mask)
		#         key = cv2.waitKey(1) & 0xFF
		#         if key == ord("q"):
		#             break
		#     else:
		#         break
		# cv2.destroyAllWindows()

		# cv2.namedWindow('Threshold2')
		# cv2.createTrackbar('Hue       ', 'Threshold2',0,179,nothing)
		# cv2.createTrackbar('Saturation', 'Threshold2',0,255,nothing)
		# cv2.createTrackbar('Value     ', 'Threshold2',0,255,nothing)
		
		# while(True):
		#     ret, frame = camera.read() 
		#     if ret == True:
		#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
		#         frame = cv2.flip(frame,1) 
		#         hsv = cv2.flip(hsv,1) 
		#         h2 = cv2.getTrackbarPos('Hue       ','Threshold2')
		#         s2 = cv2.getTrackbarPos('Saturation','Threshold2')
		#         v2 = cv2.getTrackbarPos('Value     ','Threshold2')  
		#         lower = np.array([h2-10,s2,v2], dtype = np.uint8)
		#         upper = np.array([h2+10,255,255], dtype = np.uint8)
		#         mask = cv2.inRange(hsv, lower, upper)     
		#         mask = cv2.bitwise_and(hsv,hsv,mask = mask)     
		#         cv2.imshow("Threshold2", mask)
		#         key = cv2.waitKey(1) & 0xFF
		#         if key == ord("q"):
		#             break
		#     else:
		#         break
		# cv2.destroyAllWindows()

		p = 0
	
		cv2.namedWindow('Frame')
		thrc.start()
		thrc2.start()

		
		while True:
			if not kc:
				if sad !=0:
					ar.append(sad)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("p"):
				p = not p
				if p:
					print "Enabled"
				else:
					print "Disabled"

			if key == ord("q"):
				
				break

			if key == ord("r"):
				if kc:
					print "Recording ON"
				if not kc:
					print "Recorded"
					print ar
					thr2.start()
				kc = not kc

		cv2.destroyAllWindows()


def delaybeat():
	loopcheck = 0
	while(loopcheck<8):
		loopcheck = loopcheck + 1
	return

def backplay():


	print "I'm in thread : "
	#print len(myrecording)
	#sd.play(myrecording, fs)

	while True:
		i = 0
		for i in ar:
			time.sleep(0.25)
			print i
			if(i==1):
				sd.play(do1, fs1)
			elif(i==2):
				sd.play(do2, fs2)
			elif(i==3):
				sd.play(do3, fs3)
			elif(i==4):
				sd.play(do4, fs4)

def loop1(lock):
	print "yaya"
	import cv2
	import soundfile as sf
	import sounddevice as sd
	import pyglet
	s1 = pyglet.media.load('sounds/brush.wav', streaming = False)
	s2 = pyglet.media.load('sounds/hh.wav', streaming = False)
	s3 = pyglet.media.load('sounds/drum.wav', streaming = False)
	s4 = pyglet.media.load('sounds/snare.wav', streaming = False)



	# do1, fs1 = sf.read('sounds/brush.wav')
	# do2, fs2 = sf.read('sounds/hh.wav')
	# do3, fs3 = sf.read('sounds/drum.wav')
	# do4, fs4 = sf.read('sounds/snare.wav')
	# fs = 65535
	# duration = 10.5  # seconds
	camera = cv2.VideoCapture(0)
	h, s, v = 119, 97, 14
	gLower = (h-10,s,v)
	gUpper = (h+10, 255, 255)
	f = 0
	while(True):
		sad = 0
		(grabbed, frame) = camera.read()
		frame = cv2.flip(frame,1)

		frame = imutils.resize(frame, width=1200)
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		mask = cv2.inRange(hsv, gLower, gUpper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)


		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		center = None

		cv2.rectangle(frame,(50,150), (300, 300), (0,0,255), 3)
		cv2.rectangle(frame,(1150,150), (900, 300), (0,0,255), 3)
		cv2.rectangle(frame,(300,450), (550, 600), (0,0,255), 3)
		cv2.rectangle(frame,(900,450), (650, 600), (0,0,255), 3)
		
		if p:
			if len(cnts) > 0:
				c = max(cnts, key=cv2.contourArea)
				((x, y), radius) = cv2.minEnclosingCircle(c)
				M = cv2.moments(c)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				if radius > 10:
					cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
					cv2.circle(frame, center, 5, (0, 0, 255), -1) 
					radius = 10
				if (center[0] >= 100 and center[0] <= 250 and center[1] >= 170 and center[1] <= 280 and radius <= 300):
					if f == 0:	
						s1.play()
						# sd.play(do1,fs1)
						sad = 1
						f = 1
				elif (center[0] >= 950 and center[0] <= 1100 and center[1] >= 170 and center[1] <= 280 and radius <= 300):
					if f == 0:
						s2.play()
						# sd.play(do2, fs2)
						sad = 2
						f = 1
				elif (center[0] >= 350 and center[0] <= 500 and center[1] >= 470 and center[1] <= 580 and radius <= 300):
					if f == 0:
						s3.play()
						# sd.play(do3, fs3)
						sad = 3
						f = 1
				elif (center[0] >= 700 and center[0] <= 850 and center[1] >= 470 and center[1] <= 580 and radius <= 300):
					if f == 0:
						s4.play()
						# sd.play(do4, fs4)
						sad = 4
						f = 1
				else:
					f = 0
		lock.acquire()
		cv2.imshow("Frame", frame)
		lock.release()
# 11, 116, 151

def loop2(lock):
	print "yaya"
	import cv2
	import soundfile as sf
	import sounddevice as sd
	import pyglet
	s1 = pyglet.media.load('sounds/brush.wav', streaming = False)
	s2 = pyglet.media.load('sounds/hh.wav', streaming = False)
	s3 = pyglet.media.load('sounds/drum.wav', streaming = False)
	s4 = pyglet.media.load('sounds/snare.wav', streaming = False)




	# do1, fs1 = sf.read('sounds/brush.wav')
	# do2, fs2 = sf.read('sounds/hh.wav')
	# do3, fs3 = sf.read('sounds/drum.wav')
	# do4, fs4 = sf.read('sounds/snare.wav')
	# fs = 65535
	# duration = 10.5  # seconds
	camera = cv2.VideoCapture(0)
	h, s, v = 11, 116, 151
	gLower = (h-10,s,v)
	gUpper = (h+10, 255, 255)
	f = 0
	while(True):
		sad = 0
		(grabbed, frame) = camera.read()
		frame = cv2.flip(frame,1)

		frame = imutils.resize(frame, width=1200)
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		mask = cv2.inRange(hsv, gLower, gUpper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)


		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		center = None

		cv2.rectangle(frame,(50,150), (300, 300), (0,0,255), 3)
		cv2.rectangle(frame,(1150,150), (900, 300), (0,0,255), 3)
		cv2.rectangle(frame,(300,450), (550, 600), (0,0,255), 3)
		cv2.rectangle(frame,(900,450), (650, 600), (0,0,255), 3)
		
		if p:
			if len(cnts) > 0:
				c = max(cnts, key=cv2.contourArea)
				((x, y), radius) = cv2.minEnclosingCircle(c)
				M = cv2.moments(c)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				if radius > 10:
					cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
					cv2.circle(frame, center, 5, (0, 0, 255), -1) 
					radius = 10
				if (center[0] >= 100 and center[0] <= 250 and center[1] >= 170 and center[1] <= 280 and radius <= 300):
					if f == 0:	
						s1.play()
						# sd.play(do1,fs1)
						sad = 1
						f = 1
				elif (center[0] >= 950 and center[0] <= 1100 and center[1] >= 170 and center[1] <= 280 and radius <= 300):
					if f == 0:
						s2.play()
						# sd.play(do2, fs2)
						sad = 2
						f = 1
				elif (center[0] >= 350 and center[0] <= 500 and center[1] >= 470 and center[1] <= 580 and radius <= 300):
					if f == 0:
						s3.play()
						# sd.play(do3, fs3)
						sad = 3
						f = 1
				elif (center[0] >= 700 and center[0] <= 850 and center[1] >= 470 and center[1] <= 580 and radius <= 300):
					if f == 0:
						s4.play()
						# sd.play(do4, fs4)
						sad = 4
						f = 1
				else:
					f = 0
		lock.acquire()
		cv2.imshow("Frame", frame)
		lock.release()





thr = threading.Thread(target=mainloop, args=(), kwargs={})
thr2 = threading.Thread(target=backplay, args=(), kwargs={})
thrc = threading.Thread(target=loop1, args=(lock,), kwargs={})
thrc2 = threading.Thread(target=loop2, args=(lock,), kwargs={})

thr2.daemon = True
thrc.daemon = True
thrc2.daemon = True


mainloop()
