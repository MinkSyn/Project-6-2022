import numpy as np
import cv2

#Read and smoothing image
img=cv2.imread('Image_demo.jpg')
img=cv2.resize(img,(400,300))
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(img_gray,(5,5),0) #uses gaussian

edged=cv2.Canny(blurred,100,220) #2 parameters are threshold

#Count object and save loaction object
(counter,_)=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

#Draw edge
coins=img.copy()
cv2.drawContours(coins,counter,-1,(0, 0, 0), 2)

#Display object
for (i,c) in enumerate(counter):
	#Cut location of objects
	(x,y,w,h)=cv2.boundingRect(c)

	coin=img[y:y+h, x:x+w]

	mask=np.zeros(img.shape[:2], dtype = "uint8")
	((cX,cY),r)=cv2.minEnclosingCircle(c)
	cv2.circle(mask,(int(cX), int(cY)), int(r), 255, -1)
	mask=mask[y:y+h, x:x+w]
	cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask = mask))
	cv2.waitKey(0)