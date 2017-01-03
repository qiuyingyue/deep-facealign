#!/usr/bin/env python2.7
# coding: utf-8
"""
    This file use Caffe model to predict landmarks and evaluate the mean error.
"""

import os, sys
import time
from functools import partial
import cv2
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from utils import getDataFromTxt, logger,processImage
sys.path.append('/home/qiuyy/workspace/caffe_2016/python')
import caffe
sys.path.append("/home/qiuyy/workspace/deep-facealign")
TXT = 'dataset/train/valImageList.txt'
#TXT='/home/qiuyy/workspace/deep-facealign/dataset/train/trainList.txt'
template = '''################## Summary #####################
Test Number: %d
Time Consume: %.03f s
FPS: %.03f
LEVEL - %d
Mean Error:
    Left Eye       = %f
    Right Eye      = %f
    Nose           = %f
    Left Mouth     = %f
    Right Mouth    = %f
Failure:
    Left Eye       = %f
    Right Eye      = %f
    Nose           = %f
    Left Mouth     = %f
    Right Mouth    = %f
'''

def evaluateError(landmarkGt, landmarkP, bbox):
    e = np.zeros(5)
    for i in range(5):
        e[i] = norm(landmarkGt[i] - landmarkP[i])
    e = e / bbox.w
    print 'landmarkGt'
    print landmarkGt
    print 'landmarkP'
    print landmarkP
    print 'error', e
    return e
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

    def forward(self, data, layer='fc2'):#bbox_pred
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
def levelP(img, bbox,F):
	"""
		LEVEL-1
		img: gray image
		bbox: bounding box of face
	"""

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

def E(level=1):
    if level == 0:
		m1="_iter_1000000.caffemodel"
		F = CNN('prototxt/2_landmark_deploy.prototxt', 'model/2_landmark/%s'%(m1))
    elif level == 1:
        from common import level1 as P
    elif level == 2:
        from common import level2 as P
    else:
        from common import level3 as P

    data = getDataFromTxt(TXT)
    error = np.zeros((len(data), 5))
    for i in range(len(data)):
        imgPath, bbox, landmarkGt = data[i]
        img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        logger("process %s" % imgPath)
        assert(img is not None )
        

        landmarkP = levelP(img, bbox,F)

        # real landmark
        landmarkP = bbox.reprojectLandmark(landmarkP)
        landmarkGt = bbox.reprojectLandmark(landmarkGt)
        error[i] = evaluateError(landmarkGt, landmarkP, bbox)
    return error

def plotError(e, name):
    # config global plot
    plt.rc('font', size=16)
    plt.rcParams["savefig.dpi"] = 240

    fig = plt.figure(figsize=(20, 15))
    binwidth = 0.001
    yCut = np.linspace(0, 70, 100)
    xCut = np.ones(100)*0.05
    # left eye
    ax = fig.add_subplot(321)
    data = e[:, 0]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('left eye')
    # right eye
    ax = fig.add_subplot(322)
    data = e[:, 1]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('right eye')
    # nose
    ax = fig.add_subplot(323)
    data = e[:, 2]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('nose')
    # left mouth
    ax = fig.add_subplot(325)
    data = e[:, 3]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('left mouth')
    # right mouth
    ax = fig.add_subplot(326)
    data = e[:, 4]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('right mouth')

    fig.suptitle('%s'%name)
    fig.savefig('log/%s.png'%name)


nameMapper = ['F_test', 'level1_test', 'level2_test', 'level3_test']

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    level = int(sys.argv[1])

    t = time.clock()
    error = E(level)
    t = time.clock() - t

    N = len(error)
    fps = N / t
    errorMean = error.mean(0)
    # failure
    failure = np.zeros(5)
    threshold = 0.05
    for i in range(5):
        failure[i] = float(sum(error[:, i] > threshold)) / N
    # log string
    s = template % (N, t, fps, level, errorMean[0], errorMean[1], errorMean[2], \
        errorMean[3], errorMean[4], failure[0], failure[1], failure[2], \
        failure[3], failure[4])
    print s

    logfile = 'log/{0}.log'.format(nameMapper[level])
    with open(logfile, 'w') as fd:
        fd.write(s)

    # plot error hist
    plotError(error, nameMapper[level])
