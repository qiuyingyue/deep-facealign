
import h5py
import numpy as np
h5py_file="/home/qiuyy/workspace/deep-facealign/train/2small_bbox/train_bbox.h5"
max_x2=0
max_y2=0
max_x1=0
max_y1=0
min_x1=0
min_x2=0
min_y1=0
min_y2=0
with h5py.File(h5py_file, 'r') as f:
	data=f.get("data")
	print "data",data
	landmarks=f.get("landmark")
	print "landmarks",landmarks
	bboxes=f.get("bbox")
	for b in bboxes:
		if b[0]>0.5:
			max_x1=max_x1+1
		if b[0]<-0.5:
			min_x1=min_x1+1
		if b[1]>0.5:
			max_x2=max_x2+1
		if b[1]<-0.5:
			min_x2=min_x2+1
		if b[2]>0.5:
			max_y1=max_y1+1
		if b[2]<-0.5:
			min_y1=min_y1+1
		if b[3]>0.5:
			max_y2=max_y2+1
		if b[3]<-0.5:
			min_y2=min_y2+1
		#print b
	print 'min_x1',min_x1,'min_x2',min_x2,'min_y1',min_y1,'min_y2',min_y2
	print 'max_x1',max_x1,'max_x2',max_x2,'max_y1',max_y1,'max_y2',max_y2
	
	'''assert(len(landmarks)==len(bboxes))
	for i in range(len(landmarks)):
		
		flag=False
		for k in range(10):
			if landmarks[i][k]<-0.1 or  landmarks[i][k]>1.1:
				flag=True
		if flag:
			#print bboxes[i]
			print landmarks[i]'''
	'''for key in enumerate(f.keys()):
		#contents[key] = np.array(f.get(key))
		print "key",key
		print f.get(key)'''
