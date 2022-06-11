import cv2
import csv
import os
import numpy as np

def histogram(ori:list,mag:list)->list:
	#Use method 4
	ans=np.array([0]*9)
	for i in range(8):
		for j in range(8):
			x=int(ori[i][j]//20)
			if x==8 or x==9: 
				x=8
				ans[x]+=mag[i][j]
			else:
				ans[x]+=(((x+1)*20-ori[i][j])/20)*mag[i][j]
				ans[x+1]+=((ori[i][j]-x*20)/20)*mag[i][j]
	return ans

def hog_result(link:str,col=32,row=64)->list:
	im=cv2.imread(link,0)
	im=cv2.resize(im,(col,row))

	#Calculate gradient follow mask line
	x_mask=np.array([[-1,0,1]])
	y_mask=x_mask.T
	# x_mask=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
	# y_mask=x_mask.T
	fx=cv2.filter2D(im,cv2.CV_32F,x_mask)
	fy=cv2.filter2D(im,cv2.CV_32F,y_mask)

	#Calculate magnitude
	magnitude=np.sqrt(np.square(fx)+np.square(fy))
	orientation=np.arctan(np.divide(fy,fx+0.00001))
	#Translate to degrees(độ) because auto change -90 to 90 so plus 90 to output 0 to 180
	orientation=np.degrees(orientation)+90

	#Calculate cell and block
	cell_x=im.shape[1]//8
	cell_y=im.shape[0]//8

	#Create histogram of gradient hollow( 9 is shortened value degrees)
	hist_shorten=np.zeros([cell_y,cell_x,9]) #16x8x9
	#Calculate histogram of gradient
	for x in range(cell_x):
		for y in range(cell_y):
			#Take orientation and magnitude on a cell
			ori=orientation[y*8:y*8+8,x*8:x*8+8]
			mag=magnitude[y*8:y*8+8,x*8:x*8+8]
			#Create histogram for a cell
			hist_shorten[y,x,:]=histogram(ori,mag) #16,8,9

	#Create normalization matrix hollow
	feature_hog=np.zeros([cell_y-1,cell_x-1,36])
	#Normalization
	for x in range(cell_x-1):
		for y in range(cell_y-1):
			k=np.sqrt(np.sum(np.square(hist_shorten[y:y+2,x:x+2,:])))
			feature=hist_shorten[y:y+2,x:x+2,:]/k
			feature=feature.flatten()
			feature_hog[y,x,:]=feature

	return feature_hog.flatten()

def main():
	list_state=['Test','Train']
	list_name=['Correct','Incorrect','Mask','Not_Mask']
	curr_dict=os.getcwd()
	for name in list_state: 
		for i in range(4):
			link='C:/Users/minh dung/Desktop/Python/Computer Project/Dataset/'+name+'/'+list_name[i]
			os.chdir(link)
			arr=os.listdir()
			os.chdir(curr_dict)

			with open(name+'_'+list_name[i]+'.csv','w+') as file:
				writer=csv.writer(file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
				for word in arr:
					feature=hog_result(link+'/'+word)
					writer.writerow(feature)

if __name__=="__main__":
	main()