#!/usr/bin/env python2.7
# coding: utf-8
"""
    This file use Caffe model to predict data from http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
"""

import os, sys
from functools import partial
import numpy as np
import cv2
sys.path.append('/home/qiuyy/workspace/caffe_2016/python')
import caffe
from utils import getDataFromTxt, createDir, logger, drawLandmark,BBox,processImage
sys.path.append("/home/qiuyy/workspace/deep-facealign")
class CNN(object):
    """
        Generalized CNN for simple run forward with given Model
    """

    def __init__(self, net, model):
        self.net = net
        self.model = model
        try:
            self.cnn = caffe.Net(net,model, caffe.TEST)
        except:
            # silence
            print "Can not open %s, %s"%(net, model)
	
    def forward(self, data, layer='bbox_pred'):#bbox_pred
        fake = np.zeros((len(data), 1, 1, 1))
        self.cnn.set_input_arrays(data.astype(np.float32), fake.astype(np.float32))
        self.cnn.forward()
        result = self.cnn.blobs[layer].data[0]
        print "result",result
        # 2N --> Nx(2)
        t = lambda x: np.asarray([np.asarray([x[2*i], x[2*i+1]]) for i in range(len(x)/2)])
        result = t(result)
        return result
def init_net(network=''):
	m1="_iter_1000000.caffemodel"
	#m1="base_shift0.25+scale-0.08\\0.4_addvgg_38*38_small.caffemodel"
	#m1="base_shift0.25+scale-0.08\\0.4_38*38_small.caffemodel"
	F = CNN('prototxt/%s_deploy.prototxt'%(network), 'model/%s/%s'%(network,m1))#model/2small_bbox/_iter_1000000.caffemodel
	return F	
def level(img, F):
	"""
		img: gray image
		bbox: bounding box of face
	"""
	#m1="base_shift_scale.caffemodel"

	# F
	#f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
	#f_bbox = bbox #.subBBox(-0.0,0.0, -0.0, 0.0)
	sz=38

	f_face = img#[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]
	f_face = cv2.resize(f_face, (sz, sz))


	f_face = f_face.reshape((1, 1, sz, sz))
	f_face = processImage(f_face)
	f = F.forward(f_face)
	print "f",f

	return f

#TXT = 'dataset/test/lfpw_test_249_bbox.txt'
TXT='/home/qiuyy/workspace/MuDataSet/Mufilelist.txt'
def run_landmark():
	OUTPUT = 'dataset/test/out_landmark'
	createDir(OUTPUT)
	'''data = getDataFromTxt(TXT, with_landmark=False)
	for imgPath, bbox in data:
		img = cv2.imread(imgPath)
		assert(img is not None)
		imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		logger("process %s" % imgPath)

		landmark = P(imgGray, bbox)
		landmark = bbox.reprojectLandmark(landmark)
		print landmark
		drawLandmark(img, bbox, landmark)
		cv2.imwrite(os.path.join(OUTPUT, os.path.basename(imgPath)), img)
		cv2.imshow("asd",img)
		cv2.waitKey(0)'''
		
	with open(TXT, 'r') as fd:
		lines = fd.readlines()
	for line in lines:
		imgPath = line.strip()
		img = cv2.imread(imgPath)
		print 'imgPath',imgPath
		if img is not None:
			imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			logger("process %s" % imgPath)
			bbox=(0,imgGray.shape[0],0,imgGray.shape[1])
			bbox=BBox(bbox)
			
			landmark = level(imgGray, bbox,"2_landmark")
			
			landmark = bbox.reprojectLandmark(landmark)
			#print landmark
			drawLandmark(img, bbox, landmark)
			cv2.imwrite(os.path.join(OUTPUT, os.path.basename(imgPath)), img)
			cv2.imshow("asd",img)
			cv2.waitKey(0)
			

    
def run_bbox():
	OUTPUT = 'test/out_bbox_350K38*38'
	createDir(OUTPUT)
	#dataroot='/home/qiuyy/workspace/MuDataSet/MuFrameSave160120/MuFrameFace_ext10'
	dataroot='/home/qiuyy/workspace/MuDataSet/SmileFaceData/FaceTest0.3/'
	
	network=init_net("2_bbox")
	for path,dirs,files in os.walk(dataroot):
		if len(files)>0:
			for f in files:
				imgPath = path+'/'+f#line.strip()
				img = cv2.imread(imgPath)
				print 'imgPath',imgPath
				if img is not None:
					imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)			
					logger("process %s" % imgPath)
					newface=img
					x1=0
					x2=imgGray.shape[1]
					y1=0
					y2=imgGray.shape[0]
					for i in range(2):
						imgGray= cv2.cvtColor(newface, cv2.COLOR_BGR2GRAY)	
						result = level(imgGray, network)
						newbox=np.array(result)#-0.5
						x1+=-(newbox[0][0])*imgGray.shape[1]
						x2+=-(newbox[0][1])*imgGray.shape[1]
						y1+=-(newbox[1][0])*imgGray.shape[0]
						y2+=-(newbox[1][1])*imgGray.shape[0]						
						
						print 'newbox',newbox
						print x1,y1,x2,y2
						
						x1,x2,y1,y2=rerec(x1,x2,y1,y2)
						
						
						if x1<0:
							x1=0
						if y1<0:
							y1=0
						if x2>=img.shape[1]:
							x2=	img.shape[1]-1
						if y2>=img.shape[0]:
							y2=	img.shape[0]-1
							
						
						
						print x1,y1,x2,y2
						newface=img[y1:y2+1,x1:x2+1].copy()
						print 'shape',newface.shape
						
						if abs(newbox[0][0])<0.08 and abs(newbox[0][1])<0.08 and abs(newbox[1][0])<0.08 and abs(newbox[1][1])<0.08:
							break
						
					cv2.imshow("original",img)
					newface=cv2.resize(newface,(28,28))
					cv2.imwrite(OUTPUT+'/'+f,newface)
					cv2.imshow("newface",newface)
					
					#cv2.waitKey(0)    
def rerec(bboxA):
	
	# convert bboxA to square
	w = bboxA[1] - bboxA[0]
	h = bboxA[3] - bboxA[2]
	l = (w+h)/2#max(w,h)

	bboxA[0] = int(round(bboxA[0] + w*0.5 - l*0.5))#si she wu ru
	bboxA[1] = int(round( bboxA[0]+l))
	bboxA[2] = int(round(bboxA[2] + max(0, h*0.5 - l*0.5) ))# + h*0.5 - l*0.5 
	bboxA[3] = int(round(bboxA[2]+l))
	return bboxA[0],bboxA[1],bboxA[2],bboxA[3]					
def bbox_regression(img,initial_box,times=2):
	'''
	parameters: 
		img: original image,
		initial_box: initial bounding box (x1,x2,y1,y2)
		items: maximum times for regression
	return:
		result_box:resulting bounding box (x1,x2,y1,y2)
	'''
	x1=initial_box[0]
	x2=initial_box[1]
	y1=initial_box[2]
	y2=initial_box[3]
	updateface=img[y1:y2+1,x1:x2+1]		
	cv2.imshow("extend",updateface)
	
	for i in range(times):

		imgGray=cv2.cvtColor(updateface, cv2.COLOR_BGR2GRAY)				
		result = level(imgGray, network)# put in CNN		
		newbox=np.array(result)
		x1-=(newbox[0][0])*imgGray.shape[1]
		x2-=(newbox[0][1])*imgGray.shape[1]
		y1-=(newbox[1][0])*imgGray.shape[0]
		y2-=(newbox[1][1])*imgGray.shape[0]					
		
		print 'newbox',newbox
		print x1,x2,y1,y2
		x1,x2,y1,y2=rerec([x1,x2,y1,y2])  #convert to square box
				
		if x1<0:
			x1=0
		if y1<0:
			y1=0
		if x2>=img.shape[1]:
			x2=	img.shape[1]-1
		if y2>=img.shape[0]:
			y2=	img.shape[0]-1				
		
		print x1,x2,y1,y2
		updateface=img[y1:y2+1,x1:x2+1].copy()
		print 'shape',updateface.shape
		
		if abs(newbox[0][0])<0.08 and abs(newbox[0][1])<0.08 and abs(newbox[1][0])<0.08 and abs(newbox[1][1])<0.08:
			break
	result_box=[x1,x2,y1,y2]
	return result_box
		
	 				
	
	
def testMu_bbox():
	OUTPUT = 'test/out_bbox_38*38_Mu_610K_1bias'
	network=init_net("2small_bbox")
	createDir(OUTPUT)
	#dataroot='/home/qiuyy/workspace/MuDataSet/MuFrameSave160120/MuFrameFace_ext10'
	dataroot='/home/qiuyy/workspace/MuDataSet/SmileFaceData/'
	
	bbox='FaceDetGT.txt'
	
	bboxfile=open(os.path.join(dataroot,bbox),'r')
	
	for l in bboxfile.readlines():
		items=l.strip().split(' ')
		f=os.path.join(dataroot,'MuFrameImg80',items[0])
		img=cv2.imread(f)		
		print 'imgPath',f
		if img is not None and len(items)>3:		
			#img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			logger("process %s" % f)
			cv2.imshow("original",img)
						
			x1=int(items[2])
			y1=int(items[3])
			width=int(items[4])
			height=int(items[5])
			
			MuCrop=img[y1:y1+height+1,x1:x1+width+1]			
			cv2.imshow("MuCrop",MuCrop)				
			#expand
			scale=0.3
			x1=int(x1-scale*width)
			y1=int(y1-scale*height)
			x2=int(x1+width*(1+scale*2))
			y2=int(y1+height*(1+scale*2))
			
			if x1<0:
				x1=0
			if y1<0:
				y1=0
			if x2>=img.shape[1]:
				x2=img.shape[1]-1
			if y2>=img.shape[0]:
				y2=img.shape[0]-1
				
			updateface=img[y1:y2+1,x1:x2+1]
			
			cv2.imshow("extend",updateface)
			
			for i in range(2):

				imgGray=cv2.cvtColor(updateface, cv2.COLOR_BGR2GRAY)				
				result = level(imgGray, network)
				
				newbox=np.array(result)#-0.5
				x1-=(newbox[0][0])*imgGray.shape[1]
				x2-=(newbox[0][1])*imgGray.shape[1]
				y1-=(newbox[1][0])*imgGray.shape[0]
				y2-=(newbox[1][1])*imgGray.shape[0]						
				
				print 'newbox',newbox
				print x1,y1,x2,y2
				x1,x2,y1,y2=rerec([x1,x2,y1,y2])
				
				
				if x1<0:
					x1=0
				if y1<0:
					y1=0
				if x2>=img.shape[1]:
					x2=	img.shape[1]-1
				if y2>=img.shape[0]:
					y2=	img.shape[0]-1				
				
				print x1,y1,x2,y2
				updateface=img[y1:y2+1,x1:x2+1].copy()
				print 'shape',updateface.shape
				
				if abs(newbox[0][0])<0.08 and abs(newbox[0][1])<0.08 and abs(newbox[1][0])<0.08 and abs(newbox[1][1])<0.08:
					break
				
				
			updateface=cv2.resize(updateface,(28,28))
			cv2.imwrite(OUTPUT+'/'+items[0],updateface)
			cv2.imshow("newface",updateface)
			
			#cv2.waitKey(0)   				
if __name__ == '__main__':
	#run_landmark()
	#run_bbox()
	testMu_bbox()
