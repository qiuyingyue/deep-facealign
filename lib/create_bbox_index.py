# coding:utf-8
import os
import cv2
import h5py
import numpy as np
import random
import math
imageroot="/home/qiuyy/workspace/deep-facealign/dataset/train"



#bbox:x1,x2,y1,y2
#landmark:leye,reye,nose,lmou,rmou
#point:x,y
def inbox(bbox,point):
	if point[0]>bbox[0] and point[0]<bbox[1] and point[1]>bbox[2] and point[1]<bbox[3]:
		return True
	else:
		return False
def showImage():
	filelist=open("/home/qiuyy/workspace/deep-facealign/dataset/train/valImageList.txt",'r')
	for l in filelist.readlines():
	
		items=l.split(' ')
		filename=items[0]
		
		bbox=(int(items[1]),int(items[2]),int(items[3]),int(items[4]))
		leye=(int(float(items[5])),int(float(items[6])))	
		reye=(int(float(items[7])),int(float(items[8])))
		nose=(int(float(items[9])),int(float(items[10])))
		lmou=(int(float(items[11])),int(float(items[12])))
		rmou=(int(float(items[13])),int(float(items[14])))
		
		filepath=imageroot+'/'+filename
		print filepath
		img=cv2.imread(filepath)
		if img is None:
			continue
		cv2.rectangle(img,(bbox[0], bbox[2]), (bbox[1],bbox[3]), (255, 0, 0), 2)
		cv2.circle(img,leye,1,(0,255,0),2)
		cv2.circle(img,reye,1,(0,255,0),2)
		cv2.circle(img,nose,1,(0,255,0),2)
		cv2.circle(img,lmou,1,(0,255,0),2)
		cv2.circle(img,rmou,1,(0,255,0),2)
		
		cv2.imshow("aaa",img)
		cv2.waitKey(0)	
	
def annoCele():
	lmks_file=open("/home/qiuyy/workspace/deep-facealign/dataset/Anno4cele/list_landmarks_celeba.txt",'r')
	trainfile=open("/home/qiuyy/workspace/deep-facealign/dataset/train/trainList.txt",'w')
	valfile=open("/home/qiuyy/workspace/deep-facealign/dataset/train/valList.txt",'w')
	num=0
	for l in lmks_file.readlines():
		landmarks=l.strip().split(" ")
		for item in reversed(landmarks):
			if item.strip()=='':
				landmarks.remove(item)
		
		
		filename="img_celeba/"+landmarks[0]
		
		
		leye=(int(landmarks[1]),int(landmarks[2]))
		reye=(int(landmarks[3]),int(landmarks[4]))
		nose=(int(landmarks[5]),int(landmarks[6]))	
		lmou=(int(landmarks[7]),int(landmarks[8]))
		rmou=(int(landmarks[9]),int(landmarks[10]))
		
		'''leye=(float(landmarks[1]),float(landmarks[2]))
		reye=(float(landmarks[3]),float(landmarks[4]))
		nose=(float(landmarks[5]),float(landmarks[6]))	
		lmou=(float(landmarks[7]),float(landmarks[8]))
		rmou=(float(landmarks[9]),float(landmarks[10]))'''
		
		xx=reye[0]-leye[0]
		yy=reye[1]-leye[1]
		unit=int(math.sqrt(xx*xx+yy*yy)/2)	

		c_x=(leye[0]+reye[0]+nose[0]+lmou[0]+rmou[0])/5.0
		c_y=(leye[1]+reye[1]+nose[1]+lmou[1]+rmou[1])/5.0
		center=(c_x,c_y)
		
		bbox=[]
		bbox.append(int(center[0]-2*unit))
		bbox.append(int(center[0]+2*unit))
		bbox.append(int(center[1]-2*unit))
		bbox.append(int(center[1]+2*unit))
		
		l=filename
		for i in range(4):
			l=l+" "+str(bbox[i])
		for i in range(10):
			l=l+" "+landmarks[i+1]
		l=l+'\n'
		
		#print l		
		#judge if the box is correct
		if bbox[0]>0 and bbox[2]>0 and inbox(bbox,leye) and inbox(bbox,reye) and inbox(bbox,nose) and inbox(bbox,lmou) and inbox(bbox,rmou):

			
			#draw image
			filepath=imageroot+'/'+filename
			img=cv2.imread(filepath)
			if img is None:
				continue
			'''cv2.rectangle(img,(bbox[0], bbox[2]), (bbox[1],bbox[3]), (255, 0, 0), 2)
			cv2.circle(img,leye,1,(0,255,0),2)
			cv2.circle(img,reye,1,(0,255,0),2)
			cv2.circle(img,nose,1,(0,255,0),2)
			cv2.circle(img,lmou,1,(0,255,0),2)
			cv2.circle(img,rmou,1,(0,255,0),2)
			
			cv2.imshow("aaa",img)
			cv2.waitKey(0)'''
			
			#write file
			if(num%8==0):
				valfile.write(l)
			else:
				trainfile.write(l)
			num=num+1
			#print num

annoCele()
	
	
	
	

	


