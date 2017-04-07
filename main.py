from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import soundfile as sf
import sounddevice as sd
import threading
import time 
import sys

def nothing(x):
    pass
ar = []
do1, fs1 = sf.read('sounds/brush.wav')
do2, fs2 = sf.read('sounds/hh.wav')
do3, fs3 = sf.read('sounds/drum.wav')
do4, fs4 = sf.read('sounds/snare.wav')
fs = 65535
duration = 10.5  # seconds
myrecording = []
# cflag=0
p = 1
# cnts = []
# cnts2 = []
# pts = deque(maxlen=10)
# pts2 = deque(maxlen=10)


# center = None
loopcheck1 = 0
loopcheck2 = 0
camera = cv2.VideoCapture(0)
ret, frame = camera.read() 
mask = frame
mask2 = frame
def mainloop():
		mod1 = 7
		mod2 = 8
		kc = 1

		
		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-v", "--video",
			help="path to the (optional) video file")
		ap.add_argument("-b", "--buffer", type=int, default=64,
			help="max buffer size")
		args = vars(ap.parse_args())
		center = (0,0)
		radius = 0
		
		loopcheck1 = 0
		loopcheck2 = 0

		cv2.namedWindow('Threshold1')
		mask = None
		count = 0
		h, s, v = 100, 100, 100
		h2, s2, v2 = 100, 100, 100
		cv2.createTrackbar('Hue       ', 'Threshold1',0,179,nothing)
		cv2.createTrackbar('Saturation', 'Threshold1',0,255,nothing)
		cv2.createTrackbar('Value     ', 'Threshold1',0,255,nothing)
		
		while(True):
		    ret, frame = camera.read() 
		    if ret == True:
		        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
		        frame = cv2.flip(frame,1) 
		        hsv = cv2.flip(hsv,1) 
		        h = cv2.getTrackbarPos('Hue       ','Threshold1')
		        s = cv2.getTrackbarPos('Saturation','Threshold1')
		        v = cv2.getTrackbarPos('Value     ','Threshold1')  
		        lower = np.array([h-10,s,v], dtype = np.uint8)
		        upper = np.array([h+10,255,255], dtype = np.uint8)
		        mask = cv2.inRange(hsv, lower, upper)     
		        mask = cv2.bitwise_and(hsv,hsv,mask = mask)     
		        cv2.imshow("Threshold1", mask)
		        key = cv2.waitKey(1) & 0xFF
		        if key == ord("q"):
		            break
		    else:
		        break
		cv2.destroyAllWindows()

		cv2.namedWindow('Threshold2')
		cv2.createTrackbar('Hue       ', 'Threshold2',0,179,nothing)
		cv2.createTrackbar('Saturation', 'Threshold2',0,255,nothing)
		cv2.createTrackbar('Value     ', 'Threshold2',0,255,nothing)
		
		while(True):
		    ret, frame = camera.read() 
		    if ret == True:
		        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
		        frame = cv2.flip(frame,1) 
		        hsv = cv2.flip(hsv,1) 
		        h2 = cv2.getTrackbarPos('Hue       ','Threshold2')
		        s2 = cv2.getTrackbarPos('Saturation','Threshold2')
		        v2 = cv2.getTrackbarPos('Value     ','Threshold2')  
		        lower = np.array([h2-10,s2,v2], dtype = np.uint8)
		        upper = np.array([h2+10,255,255], dtype = np.uint8)
		        mask = cv2.inRange(hsv, lower, upper)     
		        mask = cv2.bitwise_and(hsv,hsv,mask = mask)     
		        cv2.imshow("Threshold2", mask)
		        key = cv2.waitKey(1) & 0xFF
		        if key == ord("q"):
		            break
		    else:
		        break
		cv2.destroyAllWindows()

		p = 0

		thrc.start()
		thrc2.start()

		cv2.namedWindow('Frame')
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
			loopcheck1 = (loopcheck1 + 1)%mod1
			loopcheck2 = (loopcheck2 + 1)%mod2

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

def loop1():
	h, s, v = 100, 100, 100
	gLower = (h-10,s,v)
	gUpper = (h+10, 255, 255)
	f = 0
	while(True):
		sad = 0
		(grabbed, frame) = camera.read()
		frame = cv2.flip(frame,1)

		# if args.get("video") and not grabbed:
		# 	break
		frame = imutils.resize(frame, width=1200)
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		mask = cv2.inRange(hsv, gLower, gUpper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)


		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		center = None

		cv2.circle(frame,(447,263), 63, (0,0,255), -1)
		cv2.circle(frame,(447,400), 63, (0,255,0), -1)
		cv2.circle(frame,(227,263), 63, (255,0,0), -1)
		cv2.circle(frame,(227,400), 63, (255,0,255), -1)
		if p:
			if len(cnts) > 0:
				c = max(cnts, key=cv2.contourArea)
				((x, y), radius) = cv2.minEnclosingCircle(c)
				M = cv2.moments(c)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				if radius > 10:
					cv2.circle(frame, (int(x), int(y)), int(radius),
						(0, 255, 255), 2)
					cv2.circle(frame, center, 5, (0, 0, 255), -1)
					radius = 10

			# pts.appendleft(center)

			# for i in xrange(1, len(pts)):
			# 	if pts[i - 1] is None or pts[i] is None:
			# 		continue

			# 	thickness = int(np.sqrt(10 / float(i + 1)) * 2.5)
			# 	cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
		print "f: " + str(f)
		if(center==None):
			center = (0,0)
			radius = 0
		if p:
			if (380<=center[0] + radius<=510) and (200<=center[1] + radius<=326) and loopcheck1==0:
				if f == 0:
					print "Check"	
					sd.play(do1,fs1)
					sad = 1
					f = 1
			elif (380<=center[0] + radius<=510) and (337<=center[1] + radius<=463) and loopcheck1==0:
				if f == 0:
					print "Check2"
					sd.play(do2, fs2)
					sad = 2
					f = 1
			elif (180<=center[0] + radius<=306) and (200<=center[1] + radius<=326) and loopcheck1==0:
				if f == 0:
					print "Check3"
					sd.play(do3, fs3)
					sad = 3
					f = 1
			elif (180<=center[0] + radius<=306) and (337<=center[1] + radius<=463) and loopcheck1==0:
				if f == 0:
					print "Check4"
					sd.play(do4, fs4)
					sad = 4
					f = 1
			else:
				f = 0

		cv2.imshow("Frame", frame)


def loop2():
	h2, s2, v2 = 200, 100, 100
	gLower2 = (h2-10,s2,v2)
	gUpper2 = (h2+10, 255, 255)
	f = 0
	while(True):
		sad = 0
		(grabbed, frame) = camera.read()
		frame = cv2.flip(frame,1)

		# if args.get("video") and not grabbed:
		# 	break
		frame = imutils.resize(frame, width=1200)
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		mask2 = cv2.inRange(hsv, gLower2, gUpper2)
		mask2 = cv2.erode(mask2, None, iterations=2)
		mask2 = cv2.dilate(mask2, None, iterations=2)


		cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		center = None



		cv2.circle(frame,(447,263), 63, (0,0,255), -1)
		cv2.circle(frame,(447,400), 63, (0,255,0), -1)
		cv2.circle(frame,(227,263), 63, (255,0,0), -1)
		cv2.circle(frame,(227,400), 63, (255,0,255), -1)
		if p:
			print len(cnts2)
			if len(cnts2) > 0:
				c = max(cnts2, key=cv2.contourArea)
				((x, y), radius) = cv2.minEnclosingCircle(c)
				M = cv2.moments(c)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				if radius > 10:
					cv2.circle(frame, (int(x), int(y)), int(radius),
						(0, 255, 255), 2)
					cv2.circle(frame, center, 5, (0, 0, 255), -1)
					radius = 10

			pts2.appendleft(center)

			for i in xrange(1, len(pts2)):
				if pts2[i - 1] is None or pts2[i] is None:
					continue

				thickness = int(np.sqrt(10 / float(i + 1)) * 2.5)
				cv2.line(frame, pts2[i - 1], pts2[i], (0, 0, 255), thickness)
		if(center==None):
			center = (0,0)
			radius = 0
		if p:
			if (380<=center[0] + radius<=510) and (200<=center[1] + radius<=326) and loopcheck1==0:
				if f == 0:
					print "Check"	
					sd.play(do1,fs1)
					sad = 1
					f = 1
			elif (380<=center[0] + radius<=510) and (337<=center[1] + radius<=463) and loopcheck1==0:
				if f == 0:
					print "Check2"
					sd.play(do2, fs2)
					sad = 2
					f = 1 
			elif (180<=center[0] + radius<=306) and (200<=center[1] + radius<=326) and loopcheck1==0:
				if f == 0:
					print "Check3"
					sd.play(do3, fs3)
					sad = 3
					f = 1
			elif (180<=center[0] + radius<=306) and (337<=center[1] + radius<=463) and loopcheck1==0:
				if f == 0:
					print "Check4"
					sd.play(do4, fs4)
					sad = 4
					f = 1
			else:
				f = 0

		
		cv2.imshow("Frame", frame)






thr = threading.Thread(target=mainloop, args=(), kwargs={})
thr2 = threading.Thread(target=backplay, args=(), kwargs={})
thrc = threading.Thread(target=loop1, args=(), kwargs={})
thrc2 = threading.Thread(target=loop2, args=(), kwargs={})

thr2.daemon = True
thrc.daemon = True
thrc2.daemon = True


mainloop()
