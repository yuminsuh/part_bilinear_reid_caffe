""" Set paths """
CAFFE_ROOT = ''
MARS_DATASET_ROOT = ''
SAVE_PATH = 'mars_feat.mat'

# """ Example """
# CAFFE_ROOT = '/home/yumin/codes/caffe_retrieval'
# MARS_DATASET_ROOT = '/media/yumin/8fd45556-46a9-4805-95f5-d2420f2833cb/home/yumin/dataset/MARS/'
# SAVE_PATH = 'mars_feat.mat'
#
# The dataset directory looks like
# MARS_DATASET_ROOT
# ㄴbbox_test
#   ㄴ0001
#   ㄴ...
# ㄴbbox_train

if CAFFE_ROOT=='':
    raise ValueError('Please set CAFFE_ROOT')
if MARS_DATASET_ROOT=='':
    raise ValueError('Please set MARS_DATASET_ROOT')

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe
import math
import scipy.io
import time
import copy

model_def = 'test.prototxt'
model_weight = 'model_iter_75000.caffemodel'

""" Parameters """
gpu_id = 0
batchsize = 100

""" MODEL """
caffe.set_mode_gpu()
caffe.set_device(gpu_id)
net = caffe.Net(model_def, model_weight, caffe.TEST)

## Set preprocessor
mu = np.array((104,117,123)) # order correct? RGB or BGR?
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data',mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

""" Image list """
imglist = [line.rstrip().split()[0] for line in open('MARS-evaluation/info/test_name.txt', 'r').readlines()]
num_test = len(imglist)

""" Extract features """
feat_dim = net.blobs['normed_feature'].data.shape[1]
print(feat_dim)
features = np.zeros((num_test, feat_dim), dtype=np.float32)
avg_time, cnt = 0, 0
for idx in range(0, num_test, batchsize):
    cnt = cnt + 1
    t0 = time.time()

    # Forward network
    batch = np.arange(idx, min(idx+batchsize, num_test))
    inputimage = np.zeros((len(batch),3,160,80), dtype=np.float32)

    for i, frame_idx in enumerate(batch):
        imagepath = os.path.join(MARS_DATASET_ROOT, 'bbox_test', imglist[frame_idx][:4], imglist[frame_idx])
        image = caffe.io.load_image(imagepath)
        image = transformer.preprocess('data', image)
        inputimage[i,:,:,:] = image
    net.blobs['data'].reshape(*inputimage.shape)
    net.blobs['data'].data[...] = inputimage
    net.forward()

    # Get feature
    features[batch,:] = net.blobs['normed_feature'].data[:len(batch),:].copy().squeeze()

    # time
    avg_time = (avg_time*(cnt-1) + (time.time()-t0))/cnt
    print("average time: ", avg_time)
    print("expected to finish after ", avg_time * ((num_test-idx)/batchsize/60), "min")

""" Save features """
scipy.io.savemat(SAVE_PATH, mdict={'features':features})
