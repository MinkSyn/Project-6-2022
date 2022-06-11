import os
import cv2
import imutils
import argparse

def extract(link):
	arr_error=[]
	current_directory=os.getcwd()

	#Delete file in folder Object
	os.chdir(current_directory+'/Object')
	arr=os.listdir()
	for name in arr:
		os.remove(name)
	os.chdir(current_directory)

	#Load image
	image=cv2.imread(link)
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	 
	#Load the face detector and detect faces in the image
	detector=cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
	 
	if imutils.is_cv2():
		faceRects=detector.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5,
			minSize=(30,30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
	else:
		faceRects=detector.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5,
			minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
	
	for i in range(len(faceRects)):
		try:
			x,y,w,h=faceRects[i]
			img=image[y:y+h+int(h*0.15),x:x+w,:]	 
			cv2.imwrite('Object/Object_'+str(i)+'.jpg',img)
		except:
			arr_error.append(i)
	return faceRects

if __name__=="__main__":
	pass