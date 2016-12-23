
import h5py
import numpy as np
h5py_file="/home/qiuyy/workspace/deep-facealign/train/1_bbox_landmark/train_bbox_landmark.h5"
with h5py.File(h5py_file, 'r') as f:
	data=f.get("data")
	print "data",data
	landmarks=f.get("landmark")
	print "landmarks",landmarks
	bboxes=f.get("bbox")
	assert(len(landmarks)==len(bboxes))
	for i in range(len(landmarks)):
		
		flag=False
		for k in range(10):
			if landmarks[i][k]<-0.1 or  landmarks[i][k]>1.1:
				flag=True
		if flag:
			#print bboxes[i]
			print landmarks[i]
	'''for key in enumerate(f.keys()):
		#contents[key] = np.array(f.get(key))
		print "key",key
		print f.get(key)'''
