# coding: utf-8

import os
from os.path import join, exists
import time
import cv2
import numpy as np

class BBox(object):
	"""
		Bounding Box of face
	"""
	''' def __init__(self, bbox):
		self.left = bbox[0]
		self.right = bbox[1]
		self.top = bbox[2]
		self.bottom = bbox[3]
		self.x = bbox[0]
		self.y = bbox[2]
		self.w = bbox[1] - bbox[0]
		self.h = bbox[3] - bbox[2]
		self.norm_bbox=[0,0,0,0]'''
		
	def __init__(self, bbox,norm_bbox=[0,0,0,0]):
		self.left = bbox[0]
		self.right = bbox[1]
		self.top = bbox[2]
		self.bottom = bbox[3]
		self.x = bbox[0]
		self.y = bbox[2]
		self.w = bbox[1] - bbox[0]
		self.h = bbox[3] - bbox[2]
		self.bbox=bbox
		self.norm_bbox=norm_bbox    
	
	def norm_box(self):
		return self.norm_bbox
	def flip_norm_box(self):
		bb=self.norm_bbox
		return[-bb[1],-bb[0],bb[2],bb[3]]
		 
	def expand(self, scale=0.05):
		bbox = [self.left, self.right, self.top, self.bottom]
		bbox[0] -= int(self.w * scale)
		bbox[1] += int(self.w * scale)
		bbox[2] -= int(self.h * scale)
		bbox[3] += int(self.h * scale)
		return BBox(bbox)

	def project(self, point):
		x = (point[0]-self.x) / self.w
		y = (point[1]-self.y) / self.h
		return np.asarray([x, y])

	def reproject(self, point):
		x = self.x + self.w*point[0]
		y = self.y + self.h*point[1]
		return np.asarray([x, y])

	def reprojectLandmark(self, landmark):
		p = np.zeros((len(landmark), 2))
		for i in range(len(landmark)):
			p[i] = self.reproject(landmark[i])
		return p

	def projectLandmark(self, landmark):
		p = np.zeros((len(landmark), 2))
		for i in range(len(landmark)):
			p[i] = self.project(landmark[i])
		return p

	def subBBox(self, leftR, rightR, topR, bottomR):
		leftDelta = self.w * leftR
		rightDelta = self.w * rightR
		topDelta = self.h * topR
		bottomDelta = self.h * bottomR
		left = self.left + leftDelta
		right = self.right + rightDelta#self.left + rightDelta
		top = self.top + topDelta
		bottom = self.bottom + bottomDelta#self.top + bottomDelta
		return BBox([left, right, top, bottom],[leftR, rightR, topR, bottomR])
		
	def __repr__(self):
		return  str(self.bbox)+str(self.norm_bbox)
        

def logger(msg):
    """
        log message
    """
    now = time.ctime()
    print("[%s] %s" % (now, msg))

def createDir(p):
    if not os.path.exists(p):
        os.mkdir(p)

def shuffle_in_unison_scary(a, b,c=None):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    if c is not None:
		np.random.set_state(rng_state)
		np.random.shuffle(b)

'''def shuffle_in_unison_scary(a, b,c):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)   
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)'''
    
def drawLandmark(img, bbox, landmark):
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 1)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 1, (0,255,0), -1)
    return img
    
def show_landmark(face, landmark):
    """
        view face with landmark for visualization
    """
    face_copied = face.copy().astype(np.uint8)
    #print "landmark",landmark
    for (x, y) in landmark:
        xx = int(face.shape[0]*x)
        yy = int(face.shape[1]*y)
        cv2.circle(face_copied, (xx, yy), 2, (0,0,0), -1)
    cv2.imshow("face_rot", face_copied)
    cv2.waitKey(0)
    
def getDataFromTxt(txt, with_landmark=True):
    """
        Generate data from txt file
        return [(img_path, bbox, landmark)]
            bbox: [left, right, top, bottom]
            landmark: [(x1, y1), (x2, y2), ...]
        Original point: up-left corner of face
    """
    dirname = os.path.dirname(txt)
    with open(txt, 'r') as fd:
        lines = fd.readlines()

    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        img_path = os.path.join(dirname, components[0]) # file path
        # bounding box, (left, right, top, bottom)
        bbox = (components[1], components[2], components[3], components[4])
        bbox = [int(float(_)) for _ in bbox]
        # landmark
        if not with_landmark:
            result.append((img_path, BBox(bbox)))
            continue
        landmark = np.zeros((5, 2))
        for index in range(0, 5):
            rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = rv
        for index, one in enumerate(landmark):
            rv = ((one[0]-bbox[0])/(bbox[1]-bbox[0]), (one[1]-bbox[2])/(bbox[3]-bbox[2]))
            landmark[index] = rv
            assert(rv[0]>0 and rv[0]<1 and rv[1]>0 and rv[1]<1)
        result.append((img_path, BBox(bbox), landmark))
    return result

def getPatch(img, bbox, point, padding):
    """
        Get a patch image around the given point in bbox with padding
        point: relative_point in [0, 1] in bbox
    """
    point_x = bbox.x + point[0] * bbox.w
    point_y = bbox.y + point[1] * bbox.h
    patch_left = point_x - bbox.w * padding
    patch_right = point_x + bbox.w * padding
    patch_top = point_y - bbox.h * padding
    patch_bottom = point_y + bbox.h * padding
    patch = img[patch_top: patch_bottom+1, patch_left: patch_right+1]
    patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
    return patch, patch_bbox


def processImage(imgs):
    """
        preprocess images before feeding to CNNs
        imgs: N x 1 x W x H
    """
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
		
		m = img.mean()
		s = img.std()
		#print "m:",m,",s:",s
		imgs[i] = (img - m) / (s+1e-6)
    return imgs

def dataArgument(data):
    """
        dataArguments
        data:
            imgs: N x 1 x W x H
            bbox: N x BBox
            landmarks: N x 10
    """
    pass


def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    print "in rotate"
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1],img.shape[0]))
 
    
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
	
    face = img_rotated_by_alpha[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    #print img.shape,img_rotated_by_alpha.shape,bbox.top,bbox.bottom,bbox.left,bbox.right
    return (face, landmark_)


def flip(face, landmark):
	"""
		flip face
	"""
	face_flipped_by_x = cv2.flip(face, 1)
	if landmark is not None:
		landmark_ = np.asarray([(1-x, y) for (x, y) in landmark])
		landmark_[[0, 1]] = landmark_[[1, 0]]
		landmark_[[3, 4]] = landmark_[[4, 3]]
	else:
		landmark_=None
	return (face_flipped_by_x, landmark_)

def getIOU(bbox_gt,bbox_roi):
	"""
		calculate intersection of union
		box cordinate:(x1,x2,y1,y2)
	"""
	xlist=[bbox_gt.left,bbox_gt.right,bbox_roi.left,bbox_roi.right]
	ylist=[bbox_gt.top,bbox_gt.bottom,bbox_roi.top,bbox_roi.bottom]
	xlist.sort()
	ylist.sort()
	#print xlist,ylist
	inter=(xlist[2]-xlist[1])*(ylist[2]-ylist[1])
	union=(bbox_gt.right-bbox_gt.left)*(bbox_gt.bottom-bbox_gt.top)+(bbox_roi.right-bbox_roi.left)*(bbox_roi.bottom-bbox_roi.top)-inter
	#print "iou",float(inter)/union
	return float(inter)/union

def isLegalLandmark(landmark):
	if new_landmark[i][0]>0 and new_landmark[i][1]>0 and  new_landmark[i][0]<1 and new_landmark[i][1]<1:
		return True
	else:
		return False
					#print IOU,"Not a landmark face"
def shift(img,bbox,landmark,ioulimit=0.4):
	#Positives: IoU above 0.65 to a ground truth face;
	#Part faces: IoU between 0.4 and 0.65 to a ground truth face;
	#Landmark faces: faces labeled 5 landmarks’positions. 
	rnd=0
	while(True):
		
		#new_center=(np.random.rand(),np.random.rand())  #random number from (0, 1)
		x_shift=np.random.uniform(-0.25,0.25)
		y_shift=np.random.uniform(-0.25,0.25)
		new_box=bbox.subBBox(x_shift,x_shift,y_shift,y_shift)

		IOU=getIOU(bbox,new_box)
		

		if(IOU>ioulimit ):
			print "Shifting: IOU",IOU,"img shape:",img.shape
			if (new_box.left>0 and new_box.top>0 and new_box.bottom+1<img.shape[0] and new_box.right+1<img.shape[1] ):
				if landmark is not None:	
					new_landmark=np.zeros((5, 2))				
					for i in range(5):					
						new_landmark[i][0]=landmark[i][0]-x_shift								
						new_landmark[i][1]=landmark[i][1]-y_shift
				else:
					new_landmark=None
				
				new_face=img[new_box.top:new_box.bottom+1,new_box.left:new_box.right+1]			
				return new_face,new_box,new_landmark,IOU
			else:			
				print "old",bbox
				print "new",new_box	
				rnd+=1
				print '**********************',rnd
				if rnd>100:
					print 'failed'
					return   None,None,None,None

def scale(img,bbox,landmark,ioulimit=0.4):
	rnd=0
	while(True):
		x_scale=np.random.uniform(-0.08,0.4)	 #[-0.15,0.5) [0.7,2)
		y_scale=x_scale#np.random.uniform(-0.08,0.4)
		new_box=bbox.subBBox(-x_scale,x_scale,-y_scale,y_scale)
		IOU=getIOU(bbox,new_box)
		
		if(IOU>ioulimit ):
			print "Scaling:","IOU",IOU,"img shape:",img.shape
			if (new_box.left>0 and new_box.top>0 and new_box.bottom+1<img.shape[0] and new_box.right+1<img.shape[1] ):
				if landmark is not None:	
					new_landmark=np.zeros((5, 2))				
					for i in range(5):					
						new_landmark[i][0]=(landmark[i][0]-0.5)/(x_scale*2+1)+0.5							
						new_landmark[i][1]=(landmark[i][1]-0.5)/(y_scale*2+1)+0.5
				else:
					new_landmark=None
				
				new_face=img[new_box.top:new_box.bottom+1,new_box.left:new_box.right+1]			
				return new_face,new_box,new_landmark,IOU
			else:
				print "old",bbox
				print "new",new_box	,new_box.left>0	,new_box.top>0,new_box.bottom+1<img.shape[0],new_box.right+1<img.shape[1] 		
				rnd+=1
				print '**********************',rnd
				if rnd>100:
					print 'failed'
					return   None,None,None,None
		
	
def shift_and_scale(img,bbox,landmark,ioulimit=0.4):
	rnd=0
	while(True):
	
		x_shift=np.random.uniform(-0.25,0.25)#random number from (-0.4, 0.4)
		y_shift=np.random.uniform(-0.25,0.25)
		
		
		x_scale=np.random.uniform(-0.08,0.4)	 #[-0.3,0.8) [0.7,2)
		y_scale=x_scale#np.random.uniform(-0.08,0.4)
		
		new_box=bbox.subBBox(x_shift-x_scale,x_shift+x_scale,y_shift-y_scale,y_shift+y_scale)
		#print 'new_box', new_box
		IOU=getIOU(bbox,new_box)
		
		if(IOU>ioulimit ):
			print "Shift and Scale: IOU",IOU,"img shape:",img.shape
			if (new_box.left>0 and new_box.top>0 and new_box.bottom+1<img.shape[0] and new_box.right+1<img.shape[1] ):
				if landmark is not None:	
					new_landmark=np.zeros((5, 2))				
					for i in range(5):					
						new_landmark[i][0]=(landmark[i][0]-x_shift-0.5)/(x_scale*2+1)+0.5				
						new_landmark[i][1]=(landmark[i][1]-y_shift-0.5)/(y_scale*2+1)+0.5
				else:
					new_landmark=None
				
				new_face=img[new_box.top:new_box.bottom+1,new_box.left:new_box.right+1]	
				#print 'before return',new_box		
				return new_face,new_box,new_landmark,IOU
			else:				
				print "old",bbox
				print "new",new_box			
				rnd+=1
				print '**********************',rnd
				if rnd>80:
					print 'failed'
					return   None,None,None,None

'''def randomShift(landmarkGt, shift):
    """
        Random Shift one time
    """
    diff = np.random.rand(5, 2)
    diff = (2*diff - 1) * shift
    landmarkP = landmarkGt + diff
    return landmarkP

def randomShiftWithArgument(landmarkGt, shift,N = 2):
    """
        Random Shift more
    """
    
    landmarkPs = np.zeros((N, 5, 2))
    for i in range(N):
        landmarkPs[i] = randomShift(landmarkGt, shift)
    return landmarkPs'''
