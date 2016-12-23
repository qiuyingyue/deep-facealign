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

#F = CNN('prototxt/%s_deploy.prototxt'%(network), 'model/%s/%s'%(network,m1))
def level(img, bbox,network=""):
	"""
		LEVEL-1
		img: gray image
		bbox: bounding box of face
	"""
	m1="_iter_600000.caffemodel"
	
	
	F = CNN('prototxt/%s_deploy.prototxt'%(network), 'model/%s/%s'%(network,m1))
	# F
	#f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
	f_bbox = bbox #.subBBox(-0.0,0.0, -0.0, 0.0)


	f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]
	f_face = cv2.resize(f_face, (39, 39))


	f_face = f_face.reshape((1, 1, 39, 39))
	f_face = processImage(f_face)
	f = F.forward(f_face)
	print "f",f

	return f

#TXT = 'dataset/test/lfpw_test_249_bbox.txt'
TXT='/home/qiuyy/workspace/MuDataSet/Mufilelist jpg.txt'
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
	OUTPUT = 'dataset/test/out_bbox'
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
			
			result = level(imgGray, bbox,"1_bbox")
			newbox=np.array(result)-0.5
			x1=-(newbox[0][0])*img.shape[0]
			x2=img.shape[0]-(newbox[0][1])*img.shape[0]
			y1=-(newbox[1][0])*img.shape[1]
			y2=img.shape[1]-(newbox[1][1])
			
			print newbox,x1,y1,x2,y2
			#print landmark
			#drawLandmark(img, bbox, landmark)
			cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),1)
			cv2.imwrite(os.path.join(OUTPUT, os.path.basename(imgPath)), img)
			cv2.imshow("asd",img)
			cv2.waitKey(0)       
if __name__ == '__main__':
	#run_landmark()
	run_bbox()
