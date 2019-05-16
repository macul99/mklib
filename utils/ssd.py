# evaluate and compare face-align models
from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import caffe
from os import listdir
from os.path import isfile,join
from os import walk
import cv2
import caffe
from caffe.proto import caffe_pb2
import cPickle as pickle

class ssd():
    def __init__(   self,
                    model_path='~/mymodels/caffe_ssd', 
                    prototxt_name='deploy', 
                    caffemodel_name='weights/SFD', 
                    mode='gpu0',
                    img_height=480,
                    img_width=640,
                    th=0.9,
                    resize_ratio=0.5):
        print('model_path', model_path)
        print('prototxt_name', prototxt_name)

        self.model_path = model_path
        self.prototxt_name = prototxt_name
        self.caffemodel_name = caffemodel_name
        self.th = th
        self.resize_ratio = resize_ratio

        self.mean_array = np.zeros([3,int(img_height*resize_ratio), int(img_width*resize_ratio)])
        self.mean_array[0,:,:] = 104.0
        self.mean_array[1,:,:] = 117.0
        self.mean_array[2,:,:] = 123.0

        if mode == 'cpu':
            caffe.set_mode_cpu()
        elif mode == 'gpu0':
            caffe.set_device(0)
            caffe.set_mode_gpu()
        elif mode == 'gpu1':
            caffe.set_device(1)
            caffe.set_mode_gpu()
        else:
            assert False, 'error: please specify mode'

        self.Net = caffe.Net(join(model_path, prototxt_name+".prototxt"), 
                                join(model_path, caffemodel_name+".caffemodel"), caffe.TEST)

    def getModelOutput(self, image): # image should be single color image
        image = cv2.resize(image, None, fx = self.resize_ratio, fy = self.resize_ratio)
        image = np.rollaxis(image,2)
        img_c, img_h, img_w = image.shape
        assert img_c==3, 'image shape is wrong'

        if image.shape != self.mean_array.shape:
            self.mean_array = np.zeros([3,img_h,img_w])
            self.mean_array[0,:,:] = 104.0
            self.mean_array[1,:,:] = 117.0
            self.mean_array[2,:,:] = 123.0

        image = image - self.mean_array

        self.Net.blobs['data'].reshape(1,img_c,img_h,img_w)
        self.Net.blobs['data'].data[...] = image
        out = self.Net.forward()
        out = out['detection_out'][0][0]
        #print(out)

        h,w = out.shape

        result = []

        for i in range(h):
            if (out[i][0]!=-1) & (out[i][2]>self.th):
                result += [[out[i][3]*img_w/self.resize_ratio, out[i][4]*img_h/self.resize_ratio, out[i][5]*img_w/self.resize_ratio, out[i][6]*img_h/self.resize_ratio]]

        return np.array(result).astype(np.int16)
