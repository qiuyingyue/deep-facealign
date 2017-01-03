#!/usr/bin/env python2.7
# coding: utf-8
"""
    This file convert dataset from http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
    We convert data for LEVEL-1 training data.
    all data are formated as (data, landmark), and landmark is ((x1, y1), (x2, y2)...)
"""

import os
import time
import math
from os.path import join, exists
import cv2
import numpy as np
import h5py
from utils import getDataFromTxt,logger,createDir,processImage,shuffle_in_unison_scary,show_landmark
from utils import show_landmark, flip, rotate,shift,scale,shift_and_scale


TRAIN = 'dataset/train'
OUTPUT = 'train'
createDir(OUTPUT)
assert(exists(TRAIN) and exists(OUTPUT))

channel=1
sz=39

def appendlist(F_imgs,F_bboxs,F_landmarks,f_face,f_norm_bbox,f_landmark):
	#print f_face.shape,f_norm_bbox
	
	f_face = cv2.resize(f_face, (sz, sz))
	
	#cv2.imshow("asd",f_face)
	#cv2.waitKey(0)
	
	f_face = f_face.reshape((channel, sz, sz))		
	F_imgs.append(f_face)
	
	f_norm_bbox=np.array(f_norm_bbox)#+0.5
	#print f_norm_bbox
	F_bboxs.append(f_norm_bbox.reshape(4))
	if F_landmarks is not None or f_landmark is not None:
		F_landmarks.append(f_landmark.reshape(10))#f_landmark = landmarkGt.reshape(10)
		
def showBbox(img,bbox):
	cv2.rectangle(img,(int(bbox.left),int(bbox.top)),(int(bbox.right),int(bbox.bottom)),(255,0,0),2)
	cv2.imshow("a",img)
	cv2.waitKey(0)	
	
def generate_hdf5_bbox(ftxt, output, dname,fname, train=False,with_bbox=True,with_landmark=False):



	data = getDataFromTxt(ftxt,with_landmark=False)#data = getDataFromTxt(TXT, with_landmark=False)
	print "total amount:",len(data)
	F_imgs = []
	F_bboxs=[]
	
	num=0
	for (imgPath, bbox) in data:		
		img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		#img = cv2.imread(imgPath)
		print imgPath
		assert(img is not None)
		logger("process %s" % imgPath)
		# F
		f_bbox = bbox#.subBBox(-0.05, 0.05, -0.05, 0.05)
		f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]

		#showBbox(img,bbox)		
		
		if train:
			times=3
		else:
			times=1
		### random shift and scale
		for i in range(times): #if argument and np.random.rand() > -1:
			
			
			###shift 	
			face_shifted,bbox_shifted,landmark_shifted,IOU=shift(img,f_bbox,None,0.4)
			if IOU is not None:
				appendlist(F_imgs,F_bboxs,None,face_shifted,bbox_shifted.norm_box(),None)			
			
				### shift+flip
				face_flipped, landmark_flipped = flip(face_shifted, landmark_shifted)			
				appendlist(F_imgs,F_bboxs,None,face_flipped,bbox_shifted.flip_norm_box(),None)
						
			###scale
			face_scaled,bbox_scaled,landmark_scaled,IOU=scale(img,f_bbox,None)
			appendlist(F_imgs,F_bboxs,None,face_scaled,bbox_scaled.norm_box(),None)			

			### scale+flip
			face_flipped, landmark_flipped = flip(face_scaled, landmark_scaled)			
			appendlist(F_imgs,F_bboxs,None,face_flipped,bbox_scaled.flip_norm_box(),None)
			
								
			###shift and scale
			face_shifted_scaled,bbox_shifted_scaled,landmark_shifted_scaled,IOU=shift_and_scale(img,f_bbox,None,0.4)
			if IOU is not None:
				appendlist(F_imgs,F_bboxs,None,face_shifted_scaled,bbox_shifted_scaled.norm_box(),None)		
				#print bbox_shifted_scaled.norm_box()	
				#showBbox(img,bbox_shifted_scaled)		
								
				### flip
				face_flipped, landmark_flipped = flip(face_shifted_scaled, landmark_shifted_scaled)			
				appendlist(F_imgs,F_bboxs,None,face_flipped,bbox_shifted_scaled.flip_norm_box(),None)
				
				
			###shift and scale
			face_shifted_scaled,bbox_shifted_scaled,landmark_shifted_scaled,IOU=shift_and_scale(img,f_bbox,None,0.4)
			if IOU is not None:
				appendlist(F_imgs,F_bboxs,None,face_shifted_scaled,bbox_shifted_scaled.norm_box(),None)			
				#print bbox_shifted_scaled.norm_box()
				#showBbox(img,bbox_shifted_scaled)					
				
				### flip
				face_flipped, landmark_flipped = flip(face_shifted_scaled, landmark_shifted_scaled)			
				appendlist(F_imgs,F_bboxs,None,face_flipped,bbox_shifted_scaled.flip_norm_box(),None)

			###shift and scale
			face_shifted_scaled,bbox_shifted_scaled,landmark_shifted_scaled,IOU=shift_and_scale(img,f_bbox,None,0.4)
			if IOU is not None:
				appendlist(F_imgs,F_bboxs,None,face_shifted_scaled,bbox_shifted_scaled.norm_box(),None)			
				#print bbox_shifted_scaled.norm_box()
				#showBbox(img,bbox_shifted_scaled)					
				
				### flip
				face_flipped, landmark_flipped = flip(face_shifted_scaled, landmark_shifted_scaled)			
				appendlist(F_imgs,F_bboxs,None,face_flipped,bbox_shifted_scaled.flip_norm_box(),None)
					

				
		appendlist(F_imgs,F_bboxs,None,f_face,f_bbox.norm_box(),None)
		
		### flip
		face_flipped, landmark_flipped = flip(f_face, None)			
		appendlist(F_imgs,F_bboxs,None,face_flipped,f_bbox.flip_norm_box(),None)
				
		num=num+1
		
		if num>=15000 or num>=len(data):
			F_imgs, F_bboxs= np.asarray(F_imgs), np.asarray(F_bboxs)
			F_imgs = processImage(F_imgs)
			shuffle_in_unison_scary(F_imgs,F_bboxs)

			# full face
			base = join(OUTPUT, dname)
			createDir(base)
			output = join(base, fname)
			logger("generate %s" % output)
			with h5py.File(output, 'w') as h5:
				h5['data'] = F_imgs.astype(np.float32)
				if with_bbox:
					h5['bbox'] =F_bboxs.astype(np.float32)
				if(with_landmark):
					h5['landmark'] = F_landmarks.astype(np.float32)
			break	

def generate_hdf5(ftxt, output,dname, fname, train=False,with_bbox=True,with_landmark=True):



	data = getDataFromTxt(ftxt,with_landmark=True)#data = getDataFromTxt(TXT, with_landmark=False)
	print "total amount:",len(data)
	F_imgs = []
	F_bboxs=[]
	F_landmarks = []
	num=0
	for (imgPath, bbox, landmarkGt) in data:		
		img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		#if img.shape[0]+img.shape[1]>800:
			#continue
		assert(img is not None)
		logger("process %s" % imgPath)
		# F
		f_bbox = bbox#.subBBox(-0.05, 0.05, -0.05, 0.05)
		f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]
		
		#show_landmark(f_face,landmarkGt)
		### random shift and scale
		if train:
			times=5
		else:
			times=1
		for i in range(times): #if argument and np.random.rand() > -1:
			###shift 5 times		
			face_shifted,bbox_shifted,landmark_shifted,IOU=shift(img,f_bbox,landmarkGt,0.65)
			appendlist(F_imgs,F_bboxs,F_landmarks,face_shifted,bbox_shifted.norm_box(),landmark_shifted)			
			#show_landmark(face_shifted,landmark_shifted)
			
			### flip
			face_flipped, landmark_flipped = flip(face_shifted, landmark_shifted)			
			appendlist(F_imgs,F_bboxs,F_landmarks,face_flipped,bbox_shifted.flip_norm_box(),landmark_flipped)
			#show_landmark(face_flipped,landmark_flipped)

			#scale
			face_scaled,bbox_scaled,landmark_scaled,IOU=scale(img,f_bbox,landmarkGt,0.65)
			appendlist(F_imgs,F_bboxs,F_landmarks,face_scaled,bbox_scaled.norm_box(),landmark_scaled)			
			#show_landmark(face_scaled,landmark_scaled)
			### flip
			face_flipped, landmark_flipped = flip(face_scaled, landmark_scaled)			
			appendlist(F_imgs,F_bboxs,F_landmarks,face_flipped,bbox_scaled.flip_norm_box(),landmark_flipped)
			#show_landmark(face_flipped,landmark_flipped)		
			
			#shift and scale
			face_shifted_scaled,bbox_shifted_scaled,landmark_shifted_scaled,IOU=shift_and_scale(img,f_bbox,landmarkGt,0.65)
			appendlist(F_imgs,F_bboxs,F_landmarks,face_shifted_scaled,bbox_shifted_scaled.norm_box(),landmark_shifted_scaled)			
			#show_landmark(face_shifted_scaled,landmark_shifted_scaled)
						
			### flip
			face_flipped, landmark_flipped = flip(face_shifted_scaled, landmark_shifted_scaled)			
			appendlist(F_imgs,F_bboxs,F_landmarks,face_flipped,bbox_shifted_scaled.flip_norm_box(),landmark_flipped)
			#show_landmark(face_flipped,landmark_flipped)
			
			### rotation 5 degree
			'''if np.random.rand() > 0.5:
				face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox, bbox.reprojectLandmark(landmarkGt), 5)
				landmark_rotated = bbox.projectLandmark(landmark_rotated)				
				appendlist(F_imgs,F_bboxs,F_landmarks,face_rotated_by_alpha,f_bbox.norm_box(),landmark_rotated)

				### flip with rotation
				face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
				appendlist(F_imgs,F_bboxs,F_landmarks,face_flipped,f_bbox.norm_box(),landmark_flipped)

			### rotation -5 degree
			if np.random.rand() > 0.5:
				face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox, bbox.reprojectLandmark(landmarkGt), -5)
				landmark_rotated = bbox.projectLandmark(landmark_rotated)				
				appendlist(F_imgs,F_bboxs,F_landmarks,face_rotated_by_alpha,f_bbox.norm_box(),landmark_rotated)

				### flip with rotation
				face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
				appendlist(F_imgs,F_bboxs,F_landmarks,face_flipped,f_bbox.norm_box(),landmark_flipped)'''


		appendlist(F_imgs,F_bboxs,F_landmarks,f_face,f_bbox.norm_box(),landmarkGt)
		
		num=num+1
		
		if num>=13000 or num>=len(data):
			F_imgs, F_bboxs,F_landmarks = np.asarray(F_imgs), np.asarray(F_bboxs),np.asarray(F_landmarks)
			F_imgs = processImage(F_imgs)
			shuffle_in_unison_scary(F_imgs,F_bboxs,F_landmarks)

			# full face
			base = join(OUTPUT, dname)
			createDir(base)
			output = join(base, fname)
			logger("generate %s" % output)
			with h5py.File(output, 'w') as h5:
				h5['data'] = F_imgs.astype(np.float32)
				if with_bbox:
					h5['bbox'] =F_bboxs.astype(np.float32)
				if(with_landmark):
					h5['landmark'] = F_landmarks.astype(np.float32)
			break
#for landmark
def generate_hdf5_landmark(ftxt, output,dname, fname, argument=False,with_bbox=False,with_landmark=True):

	channel=1
	sz=39

	data = getDataFromTxt(ftxt)
	print "total amount:",len(data)
	F_imgs = []
	F_bboxs=[]
	F_landmarks = []
	num=0
	for (imgPath, bbox, landmarkGt) in data:
		img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		assert(img is not None)
		logger("process %s" % imgPath)
		# F
		f_bbox = bbox#.subBBox(-0.05, 1.05, -0.05, 1.05)
		f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]

		## data argument
		if argument and np.random.rand() > -1:
			### flip
			face_flipped, landmark_flipped = flip(f_face, landmarkGt)
			face_flipped = cv2.resize(face_flipped, (sz, sz))
			F_imgs.append(face_flipped.reshape((channel, sz, sz)))
			F_landmarks.append(landmark_flipped.reshape(10))
			### rotation
			if np.random.rand() > 0.5:
				face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox, bbox.reprojectLandmark(landmarkGt), 5)
				print "5",face_rotated_by_alpha.shape
				
				landmark_rotated = bbox.projectLandmark(landmark_rotated)
				
				face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (sz, sz))
				F_imgs.append(face_rotated_by_alpha.reshape((channel, sz, sz)))
				F_landmarks.append(landmark_rotated.reshape(10))
				### flip with rotation
				face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
				face_flipped = cv2.resize(face_flipped, (sz, sz))
				F_imgs.append(face_flipped.reshape((channel, sz, sz)))
				F_landmarks.append(landmark_flipped.reshape(10))
			### rotation
			if np.random.rand() > 0.5:
				face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox, bbox.reprojectLandmark(landmarkGt), -5)
				print "-5",face_rotated_by_alpha.shape
				landmark_rotated = bbox.projectLandmark(landmark_rotated)
				
				face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (sz, sz))				
				F_imgs.append(face_rotated_by_alpha.reshape((channel, sz, sz)))
				F_landmarks.append(landmark_rotated.reshape(10))
				### flip with rotation
				face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
				face_flipped = cv2.resize(face_flipped, (sz, sz))
				F_imgs.append(face_flipped.reshape((channel, sz, sz)))
				F_landmarks.append(landmark_flipped.reshape(10))

		f_face = cv2.resize(f_face, (sz, sz))
		f_face = f_face.reshape((channel, sz, sz))
		
		F_imgs.append(f_face)
		F_landmarks.append(landmarkGt.reshape(10))#f_landmark = landmarkGt.reshape(10)

		num=num+1
		
		if num>=20000 or num>=len(data):
			F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
			F_imgs = processImage(F_imgs)
			shuffle_in_unison_scary(F_imgs, F_landmarks)

			# full face
			base = join(OUTPUT, dname)
			createDir(base)
			output = join(base, fname)
			
			logger("generate %s" % output)
			print len(F_imgs), len(F_landmarks)
			with h5py.File(output, 'w') as h5:
				h5['data'] = F_imgs.astype(np.float32)
				if with_bbox:
					h5['bbox'] =F_bboxs.astype(np.float32)
				if(with_landmark):
					h5['landmark'] = F_landmarks.astype(np.float32)
			break




if __name__ == '__main__':
	# train data
	full_name="2small_bbox"
	tp_name="bbox"
	train_txt = join(TRAIN, 'trainBoxList.txt')#trainImageList
	generate_hdf5_bbox(train_txt, OUTPUT,full_name, 'train_%s_39.h5'%tp_name, train=True)

	test_txt = join(TRAIN, 'valBoxList.txt')#valImageList
	generate_hdf5_bbox(test_txt, OUTPUT, full_name,'test_%s_39.h5'%tp_name)

	#with open(join(OUTPUT, '%s/train.txt'%full_name), 'w') as fd:
	#	fd.write('train/%s/train_%s.h5'%(full_name,tp_name))
	#with open(join(OUTPUT, '%s/test.txt'%full_name), 'w') as fd:
	#	fd.write('train/%s/test_%s.h5'%(full_name,tp_name))
		


    # Done
    
#Regions whose the Intersec-tion-over-Union (IoU) 
#(i) Negatives: IoU ratio are less than 0.3 to any ground-truth faces; 
#(ii) Positives: IoU above 0.65 to a ground truth face; 
#(iii) Part faces: IoU between 0.4 and 0.65 to a ground truth face; and 
#(iv) Landmark faces: faces labeled 5 landmarks’positions. 
#There is an unclear gap between part faces and negatives, and there are variances among different face annotations.
'''So, we choose IoU gap between 0.3 to 0.4. 
Negatives and positives are used for face classification tasks, 
positives and part faces are used for bounding box regression, and 
landmark faces are used for facial landmark localization. Total training data are
composed of 3:1:1:2 (negatives/ positives/ part face/ landmark face) data.'''
