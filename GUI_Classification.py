import os
import cv2
import time
import pygame
import easygui
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Extract_Object import extract
from SVM_Classification import results_for_GUI

pygame.init()

WIDTH,HEIGHT=1200,680
WHITE=(255,255,255)
BLACK=(0,0,0)
LIGHT=(170,170,170)
DARK=(100,100,100)
BLUE=(14,174,223)
BLUE_DARK=(25,25,112)
BLUE_LIGHT=(0,178,191)

screen=pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption('Recognition Interface')
background=pygame.image.load('Images/background.jpg')
fpsClock=pygame.time.Clock()

#Create text for function draw_screen display
class Text_data:
	def __init__(self):
		self.state="Welcome"
		self.text=10.0

	def create_text(self)->str:
		if self.state=="Welcome":
			text=self.create_font("Times New Roman",50,"Project: Identify people wearing masks",BLACK)
			return text,310,20
		elif self.state=="Run":
			text=self.create_font("Times New Roman",50,"Press the Start button for identification",BLACK)
			return text,315,20
		elif self.state=="Error":
			text=self.create_font("Times New Roman",50,"Error loading: Please reload",BLACK)
			return text,440,20
		elif self.state=="Load":
			text=self.create_font("Times New Roman",50,"Please upload data",BLACK)
			return text,530,20
		elif self.state=="Display":
			text=self.create_font("Times New Roman",50,"Complete with time: %.4f s" %(time_model),BLACK)
			return text,440,20
		elif self.state=="HOG_C":
			text=self.create_font("Times New Roman",50,"Mask: "+ld.mask+", Size: "+ld.hog_size+", C: "+str(self.text),BLACK)
			return text,370,20

	def create_font(self ,font:str ,size:int, word:str, color:tuple)->str:
		smallfont=pygame.font.SysFont(font,size)
		text=smallfont.render(word,True,color)
		return text
tx=Text_data()

#Create button for function draw_screen display
class Button:
	def __init__(self):
		self.mouse=None
		self.start=pygame.image.load('Images/start.png')
		self.view=pygame.image.load('Images/view.jpg')
		self.left=pygame.image.load('Images/left.png')
		self.right=pygame.image.load('Images/right.png')

	def upload_button(self, x:int, y:int, width:int, height:int, mid:int, center:int, text:str)->None:
		if x<=self.mouse[0]<x+width and y<=self.mouse[1]<=y+height:
			pygame.draw.rect(screen,DARK,[x,y,width,height])
			screen.blit(text,(x+mid,y+center))
		else:
			pygame.draw.rect(screen,LIGHT,[x,y,width,height])
			screen.blit(text,(x+mid,y+center))

	#Create normal button for function draw_screen
	def start_button(self,name:list ,x:int, y:int, width:int, height:int)->None:
		if x<=self.mouse[0]<x+width and y<=self.mouse[1]<=y+height:
			button=pygame.transform.scale(name,(width-14,height-14)).convert_alpha()
			screen.blit(button,(x+7,y+7))
		else:
			button=pygame.transform.scale(name,(width,height)).convert_alpha()
			screen.blit(button,(x,y))

	#Create button can change color for function draw_screen, ex: button 16x32, button 32x64
	def button_change(self,x:int,y:int,width:int,height:int,mid:int,center:int,stack:bool,text:str)->None:
		if x<=self.mouse[0]<x+width and y<=self.mouse[1]<=y+height:
			pygame.draw.rect(screen,BLUE,[x,y,width,height])
			screen.blit(text,(x+mid,y+center))
		elif stack==True:
			pygame.draw.rect(screen,BLUE,[x,y,width,height])
			screen.blit(text,(x+mid,y+center))
		else:
			pygame.draw.rect(screen,LIGHT,[x,y,width,height])
			screen.blit(text,(x+mid,y+center))
bt=Button()

#Load image, confusion matrix, chart C for GUI display
class Load:
	def __init__(self):
		self.img=None
		self.link=None
		self.state_load=True
		self.hog16=False
		self.hog32=True
		self.confusion=None
		self.chartC=None
		self.mask='Xy'
		self.hog_size='32x64'
		self.labels=[['Incorrect','Correct'],['Mask','Not_Mask']]

	def load_image(self)->list:
		try:
			tx.state="Run"
			#Load image for display on GUI
			self.link=easygui.fileopenbox()
			self.img=pygame.image.load(self.link)
			self.img=pygame.transform.scale(self.img,(850,510)).convert_alpha()
			return self.img
		except:
			tx.state="Error"
			self.img=None

	def draw_chart_C(self)->list:
		#Load data for chart C
		value_labels=pd.read_csv('Results/Accuracy/'+self.mask+'_'+self.hog_size+'.csv',header=None)
		value_labels=np.array(value_labels)

		labels=['0.001','0.01','0.1','1.0','10.0','100.0']
		#Draw chart C
		plt.bar(labels,value_labels[0],width=0.4,color='blue')

		plt.title(self.mask+'_'+self.hog_size,fontweight="bold")
		plt.xlabel('Value C')
		plt.ylabel('Acuracy (%)')
		plt.yticks(np.arange(0,101,20))
		plt.tight_layout()
		plt.savefig('Images/chartC.jpg',bbox_inches='tight')
		plt.show()

		#Load data for display on GUI
		img=cv2.imread('Images/chartC.jpg')
		img=cv2.resize(img,(900,490))
		cv2.imwrite('Images/chartC.jpg',img)
		self.chartC=pygame.image.load('Images/chartC.jpg')

	def draw_confusion_matrix(self)->list:
		#Load data C
		data=pd.read_excel('Results/Confusion_Matrix/'+self.mask+'_'+self.hog_size+'.xlsx')
		data=np.array(data)
		value_c=[0.001,0.01,0.1,1,10,100]
		
		#Draw confusion matrix
		value=np.array([int(num) for num in data[value_c.index(tx.text),:9]])
		value=np.reshape(value,(3,3))
		ax=sns.heatmap(value,annot=True, cmap='Blues',cbar=False,linewidths=2,linecolor='blue',annot_kws={'size':16},fmt='d')

		ax.set_title("Mask: "+ld.mask+", Size: "+ld.hog_size+", C: "+str(tx.text),fontsize=14);
		ax.set_xlabel('Actual Values',fontsize=16)
		ax.set_ylabel('Predicted Values',fontsize=16)

		ax.xaxis.set_ticklabels(['Incorrect','Correct','Not Mask'],fontsize=14)
		ax.yaxis.set_ticklabels(['Incorrect','Correct','Not Mask'],fontsize=14)
		plt.tight_layout() #Display not error
		plt.savefig('Images/confusionMatrix.jpg',bbox_inches='tight')
		plt.show()

		#Load data for display on GUI
		img=cv2.imread('Images/confusionMatrix.jpg')
		img=cv2.resize(img,(900,490))
		cv2.imwrite('Images/confusionMatrix.jpg',img)
		self.confusion=pygame.image.load('Images/confusionMatrix.jpg')
ld=Load()

#Function run classification
def identify():
	global image,time_model
	if tx.state=="Run":		
		Code_color=[(0,0,255),(0,255,0),(0,255,255)]
		Code_text=['Not Mask','Correct','Incorrect']
		start_time=time.time()

		#Detach object and classification each object
		#try:
		faceRects=extract(ld.link)
		if tx.text>0.5: C=int(tx.text)
		else: C=tx.text
		result,percent1,percent2=results_for_GUI(ld.mask+'_Mask',ld.hog_size,C) #Sobel, 32x64, 1
		im=cv2.imread(ld.link)

		if im.shape[0]>im.shape[1]: size=im.shape[1]/500
		else: size=im.shape[0]/500

		#Draw rect for display on GUI
		for i in range(len(faceRects)):
			x,y,w,h=faceRects[i]
			color=Code_color[result[i]-1]
			text=Code_text[result[i]-1]
			cv2.putText(im,text,(x,y-20),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=size,color=color)
			cv2.putText(im,str(percent1[i])+'% ,'+str(percent2[i])+'%',(x,y+h+int(h*0.3)),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=size,color=color)
			cv2.rectangle(im,(x,y),(x+w,y+h),color,8)

		#Save image for display on GUI
		cv2.imwrite('Object/BG_image.jpg',im)
		image=pygame.image.load('Object/BG_image.jpg')
		image=pygame.transform.scale(image,(850,510)).convert_alpha()
		tx.state="Display"
		# except:
		# 	tx.state="Error"
		end_time=time.time()
		time_model=end_time-start_time
	else: 
		tx.state="Load"	

def draw_screen():
	screen.blit(background,(0,0))
	#Display text
	text,x,y=tx.create_text()
	screen.blit(text,(x,y))
	#Display button
	if ld.state_load==True:
		#Draw background for window button
		pygame.draw.rect(screen,BLUE_LIGHT,[1,208,201,500])
		#Button demo and chart
		bt.upload_button(12,220,180,60,26,6,tx.create_font('Corbel',50,"Chart",BLUE_DARK))
		#Button upload data
		bt.upload_button(12,390,180,60,13,6,tx.create_font('Corbel',50,"Upload",BLACK))
		#Button run classification
		bt.start_button(bt.start,-12,500,220,90)
	else:
		#Button demo and chart
		bt.upload_button(12,220,180,60,26,6,tx.create_font('Corbel',50,"Demo",BLUE_DARK))
		#Button mask
		pygame.draw.rect(screen,LIGHT,[32,290,140,45])
		if ld.mask=='Xy': temp=(82,287)
		else: temp=(60,290)
		screen.blit(tx.create_font('Arial',40,str(ld.mask),BLACK),temp)
		#Button size HOG
		bt.button_change(32,345,140,45,25,0,ld.hog16,tx.create_font('Arial',40,"16x32",BLACK))
		bt.button_change(32,400,140,45,25,0,ld.hog32,tx.create_font('Arial',40,"32x64",BLACK))
		#Button C
		bt.start_button(bt.left,4,465,45,45)
		bt.start_button(bt.right,154,465,45,45)
		pygame.draw.rect(screen,WHITE,[52,465,100,45])
		screen.blit(tx.create_font('Arial',40,str(tx.text),BLACK),(60,465))
		#Button watch confusion matrix
		bt.start_button(bt.view,12,520,180,70)	
		#Button draw chart C
		bt.upload_button(12,600,180,60,26,10,tx.create_font('Corbel',40,"Chart C",BLACK))		
		
	#Display image and results
	if tx.state=="Run": screen.blit(ld.img,(275,115))
	if tx.state=="Display": screen.blit(image,(275,115))
	if ld.confusion!=None: screen.blit(ld.confusion,(250,130))
	if ld.chartC!=None: screen.blit(ld.chartC,(250,130))

def main():
	running=True
	while running:
		for event in pygame.event.get():
			if event.type==pygame.QUIT: running=False
			if event.type==pygame.MOUSEBUTTONDOWN:
				#Button demo and chart
				if 12<=bt.mouse[0]<=12+180 and 220<=bt.mouse[1]<=220+60:
					if ld.state_load==False: 
						ld.state_load=True
						ld.confusion=None
						ld.chartC=None
					else: 
						ld.state_load=False
					tx.state="HOG_C"
				#Button upload data
				if 12<=bt.mouse[0]<=12+180 and 390<=bt.mouse[1]<=390+60 and ld.state_load==True:
					ld.load_image()
				#Button run classification
				if 0<=bt.mouse[0]<=210 and 500<=bt.mouse[1]<=500+90 and ld.state_load==True:
					identify()
				#Button mask
				if 32<=bt.mouse[0]<=172 and 290<=bt.mouse[1]<=290+45 and ld.state_load==False:
					if ld.mask=='Xy': ld.mask='Sobel'
					else: ld.mask='Xy'
				#Button size HOG
				if 32<=bt.mouse[0]<=172 and 345<=bt.mouse[1]<=345+45 and ld.state_load==False:	
					if ld.hog32==True: ld.hog32=False
					ld.hog16=True
					ld.hog_size='16x32'
				if 32<=bt.mouse[0]<=172 and 400<=bt.mouse[1]<=400+45 and ld.state_load==False:
					if ld.hog16==True: ld.hog16=False
					ld.hog32=True
					ld.hog_size='32x64'
				#Button C
				if 4<=bt.mouse[0]<=4+45 and 465<=bt.mouse[1]<=465+45 and ld.state_load==False:
					if tx.text>0.001: tx.text/=10
				if 154<=bt.mouse[0]<=154+45 and 465<=bt.mouse[1]<=465+45 and ld.state_load==False:
					if tx.text<100: tx.text*=10
				#Button watch confusion matrix
				if 12<=bt.mouse[0]<=12+180 and 520<=bt.mouse[1]<=520+70 and ld.state_load==False:
					if ld.chartC!=None: ld.chartC=None
					ld.draw_confusion_matrix()
				#Button draw chart C
				if 12<=bt.mouse[0]<=192 and 600<=bt.mouse[1]<=600+60 and ld.state_load==False:
					if ld.confusion!=None: ld.confusion=None
					ld.draw_chart_C()

		bt.mouse=pygame.mouse.get_pos()
		draw_screen()

		pygame.display.update()
		fpsClock.tick(60)
	pygame.quit()

if __name__=="__main__":
	main()