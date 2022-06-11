from sklearn.svm import SVC
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Features_Extraction import hog_result
#from Detach_Background import detach

dict_train={'Incorrect':3415,'Correct':3545,'Mask':6960,'Not_Mask':6857}
dict_test={'Incorrect':660,'Correct':710,'Mask':1370,'Not_Mask':1353}

def load_data(name:str,size:str)->list:
	result=pd.read_csv('HOG_Features/Sobel_Mask/'+size+'/Train/Train_'+name+'.csv',header=None)
	result=np.array(result)
	return result

def sum_data(name1:str,name2:str,num1:int,num2:int,size:str)->[list,list]:
	mat1,mat2=load_data(name1,size),load_data(name2,size)
	X_train=np.append(mat1,mat2,axis=0)
	y=np.concatenate((np.ones(num1),-np.ones(num2)),axis=0)
	return X_train,y

def training_models(name1:str,name2:str,size:str,num:int,c:float)->None:
	X_train,y=sum_data(name1,name2,dict_train[name1],dict_train[name2],size)
	model=SVC(kernel='linear',C=c)
	model.fit(X_train,y)
	weight=model.coef_
	bias=model.intercept_

	with open('Train_'+str(num)+'.csv','w+') as file:
		writer=csv.writer(file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
		writer.writerow(weight[0])
		writer.writerow(bias)
		writer.writerow([c])

def accuracy_percent(mask:str,size:str,num:int)->None:
	data=pd.read_excel('Results/Confusion_Matrix/'+mask+'_'+size+'.xlsx')
	data=np.array(data)

	percent=sum([data[num,i] for i in [0,4,8]])/2723		
	return percent*100

def output_accuracy(mask:str,size:str):
	value_c=[0.001,0.01,0.1,1,10,100]
	value=[0]*6

	for num in range(6):
		percent=accuracy_percent(mask,size,num)
		value[num]=percent

	with open(mask+'_'+size+'.csv','w+') as file:
		writer=csv.writer(file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
		writer.writerow(value)

def output_confusion_matrix(mask:str,size:str,name:str,c:int)->[int,int]:
	data_test=pd.read_csv('HOG_Features/'+mask+'_Mask/'+size+'/Test/Test_'+name+'.csv',header=None)
	data_test=np.array(data_test)

	train=pd.read_csv('Models/'+mask+'_Mask/'+size+'/'+str(c)+'/Train_1.csv',header=None)
	weight=np.array(train.values[0,:])
	bias=np.array(train.values[1,0])

	arr=[]
	for i in range(len(data_test)):
		temp=data_test[i].dot(weight.T)+bias
		if temp>0:
			train_0=pd.read_csv('Models/'+mask+'_Mask/'+size+'/'+str(c)+'/Train_2.csv',header=None)
			weight_0=np.array(train_0.values[0,:])
			bias_0=np.array(train_0.values[1,0])
			temp_0=data_test[i].dot(weight_0.T)+bias_0
			if temp_0>0: arr.append('Incorrect')
			else: arr.append('Correct')
		else: arr.append('Not_Mask')

	# 1: Not_Mask, 2: Correct, 3: Incorrect
	return arr.count('Incorrect'),arr.count('Correct'),arr.count('Not_Mask')

def results_for_GUI(mask:str,size:str,c:int)->list:
	data_test=[]
	#Get feature image
	curr_dict=os.getcwd()
	os.chdir(curr_dict+'/Object')
	arr=os.listdir()
	os.chdir(curr_dict)

	# Have detach background
	# detach(arr)
	# for name in arr:
	# 	data_test.append(hog_result('Object/BG_'+name))

	#Don't detach background
	if size=='16x32':
		for name in arr:
			data_test.append(hog_result('Object/'+name,16,32))
	else:
		for name in arr:
			data_test.append(hog_result('Object/'+name))

	result,percent1,percent2=np.array([0]*len(data_test)),np.array([0]*len(data_test)),np.array([0]*len(data_test))
	# 1: Not_Mask, 2: Correct, 3: Incorrect
	data_test=np.array(data_test)
	os.chdir(curr_dict)

	train=pd.read_csv('Models/'+mask+'/'+size+'/'+str(c)+'/Train_1.csv',header=None)
	weight=np.array(train.values[0,:])
	bias=np.array(train.values[1,0])

	for i in range(len(data_test)):
		temp=data_test[i].dot(weight.T)+bias
		if temp>0:
			train_0=pd.read_csv('Models/'+mask+'/'+size+'/'+str(c)+'/Train_2.csv',header=None)
			weight_0=np.array(train_0.values[0,:])
			bias_0=np.array(train_0.values[1,0])
			temp_0=data_test[i].dot(weight_0.T)+bias_0
			if temp_0>0:
				percent1[i]=100.0 if temp>=1 else abs(temp*100)
				percent2[i]=100.0 if temp_0>=1 else temp_0*100
				result[i]=3
			else: 
				percent1[i]=100.0 if temp>=1 else temp*100
				percent2[i]=100.0 if temp_0<=-1 else abs(temp_0*100)
				result[i]=2
		else: 
			percent1[i]=100.0 if temp<=-1 else abs(temp*100)
			percent2[i]=0
			result[i]=1

	return result,percent1,percent2

if __name__=="__main__":
	pass