# see /mk_utils/mklib/notebooks/merge-caffe-bn-conv-layer.ipynb for how to use it

import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import caffe
from os import listdir
from os.path import isfile,join
from os import walk
from scipy.io import loadmat
import dlib
import cv2
import caffe
from caffe.proto import caffe_pb2
import cPickle as pickle
import scipy


class caffeMergeBn():
    def __init__(   self,
                    train_proto, 
                    train_model,
                    deploy_proto,
                    save_model,
                    conv_bn_list,
                    EPS=2e-5):

        self.train_proto = train_proto
        self.train_model = train_model
        self.deploy_proto = deploy_proto
        self.save_model = save_model
        self.conv_bn_list = conv_bn_list
        self.EPS = EPS
        self.conv_bn_dic = {}

        with open(self.conv_bn_list,'rb') as f:
            for line in f.readlines():
                name_conv, name_bn, name_scale = line.strip().split(',')
                self.conv_bn_dic[name_conv] = [name_bn, name_scale]

        self.net = caffe.Net(self.train_proto, self.train_model, caffe.TRAIN)  
        self.net_deploy = caffe.Net(self.deploy_proto, caffe.TEST)


    def merge_bn(self):
        '''
        merge the batchnorm, scale layer weights to the conv layer, to  improve the performance
        var = var / scaleFactor
        rstd = 1. / sqrt(var + eps)
        w = w * rstd * scale
        b = (b - mean) * rstd * scale + shift
        '''
        # copy params from net to net_deploy
        for key in self.net_deploy.params.iterkeys():
            conv = self.net.params[key]
            for i, w in enumerate(conv):
                self.net_deploy.params[key][i].data[...] = w.data

        for key in self.conv_bn_dic.keys():
            print(key,self.conv_bn_dic[key][0],self.conv_bn_dic[key][1])
            assert self.net.params.has_key(key)     
            assert self.net.params.has_key(self.conv_bn_dic[key][0])
            assert self.net.params.has_key(self.conv_bn_dic[key][1])
            assert type(self.net.params[key]) is caffe._caffe.BlobVec

            conv = self.net.params[key]
            bn = self.net.params[self.conv_bn_dic[key][0]]
            scale = self.net.params[self.conv_bn_dic[key][1]]

            wt = conv[0].data
            channels = wt.shape[0]
            bias = np.zeros(wt.shape[0])
            if len(conv) > 1:
                bias = conv[1].data
            mean = bn[0].data
            var = bn[1].data
            scalef = bn[2].data[0]

            scales = scale[0].data
            shift = scale[1].data
            if scalef==0.0:
                scalef = 1. / (scalef+2e-17)
            else:
                scalef = 1. / scalef
            mean = mean * scalef
            var = var * scalef
            rstd = 1. / np.sqrt(var + self.EPS)
            rstd1 = rstd.reshape((channels,1,1,1))
            scales1 = scales.reshape((channels,1,1,1))
            wt = wt * rstd1 * scales1
            bias = (bias - mean) * rstd * scales + shift
            
            self.net_deploy.params[key][0].data[...] = wt
            self.net_deploy.params[key][1].data[...] = bias

        self.net_deploy.save(self.save_model)

    def merge_bn_front(self):
        '''
        merge the batchnorm, scale layer which are in front of conv layer
        Layer should be in the order of: BN, Scale, Conv
        var = var / scaleFactor
        rstd = 1. / sqrt(var + eps)
        w = w * rstd * scale
        b = (b - mean) * rstd * scale + shift
        '''
        # copy params from net to net_deploy
        for key in self.net_deploy.params.iterkeys():
            conv = self.net.params[key]
            for i, w in enumerate(conv):
                self.net_deploy.params[key][i].data[...] = w.data

        for key in self.conv_bn_dic.keys():
            print(key,self.conv_bn_dic[key][0],self.conv_bn_dic[key][1])
            assert self.net.params.has_key(key)     
            assert self.net.params.has_key(self.conv_bn_dic[key][0])
            assert self.net.params.has_key(self.conv_bn_dic[key][1])
            assert type(self.net.params[key]) is caffe._caffe.BlobVec

            bn = self.net.params[key]
            scale = self.net.params[self.conv_bn_dic[key][0]]
            conv = self.net.params[self.conv_bn_dic[key][1]]

            wt = conv[0].data
            channels = wt.shape[0]
            bias = np.zeros(wt.shape[0])
            if len(conv) > 1:
                bias = conv[1].data
            mean = bn[0].data
            var = bn[1].data
            scalef = bn[2].data[0]

            scales = scale[0].data
            shift = scale[1].data
            if scalef==0.0:
                scalef = 1. / (scalef+2e-17)
            else:
                scalef = 1. / scalef
            mean = mean * scalef
            var = var * scalef

            rstd = 1. / np.sqrt(var + self.EPS)
            rstd1 = rstd.reshape((channels,1,1,1))
            scales1 = scales.reshape((channels,1,1,1))

            tmp = shift-scales*mean*rstd
            tmp1=tmp.reshape((channels,1,1,1))*np.ones(wt.shape)
            # bias_new = np.einsum('ijkl,ijkl->i', tmp1, wt) + bias
            bias_new = self.conv_weights(tmp1, wt)
            # bias_new = np.sum(np.sum(np.sum(wt, axis=3),axis=2),axis=1) * tmp + bias
            wt_new = wt * rstd1 * scales1

            self.net_deploy.params[self.conv_bn_dic[key][1]][0].data[...] = wt_new
            self.net_deploy.params[self.conv_bn_dic[key][1]][1].data[...] = bias_new

        self.net_deploy.save(self.save_model)

    def merge_fc_bn(self):
        '''
        merge the batchnorm, scale layer weights to the conv layer, to  improve the performance
        var = var / scaleFactor
        rstd = 1. / sqrt(var + eps)
        w = w * rstd * scale
        b = (b - mean) * rstd * scale + shift
        '''
        # copy params from net to net_deploy
        for key in self.net_deploy.params.iterkeys():
            fc = self.net.params[key]
            for i, w in enumerate(fc):
                self.net_deploy.params[key][i].data[...] = w.data

        for key in self.conv_bn_dic.keys():
            print(key,self.conv_bn_dic[key][0],self.conv_bn_dic[key][1])
            assert self.net.params.has_key(key)     
            assert self.net.params.has_key(self.conv_bn_dic[key][0])
            assert self.net.params.has_key(self.conv_bn_dic[key][1])
            assert type(self.net.params[key]) is caffe._caffe.BlobVec

            fc = self.net.params[key]
            bn = self.net.params[self.conv_bn_dic[key][0]]
            scale = self.net.params[self.conv_bn_dic[key][1]]

            wt = fc[0].data
            channels = wt.shape[0]
            bias = np.zeros(wt.shape[0])
            if len(fc) > 1:
                bias = fc[1].data
            mean = bn[0].data
            var = bn[1].data
            scalef = bn[2].data[0]

            scales = scale[0].data
            shift = scale[1].data
            if scalef==0.0:
                scalef = 1. / (scalef+2e-17)
            else:
                scalef = 1. / scalef
            mean = mean * scalef
            var = var * scalef
            rstd = 1. / np.sqrt(var + self.EPS)
            rstd1 = rstd.reshape((channels,1))
            scales1 = scales.reshape((channels,1))
            wt_new = wt * rstd1 * scales1
            bias_new = (bias - mean) * rstd * scales + shift
            
            self.net_deploy.params[key][0].data[...] = wt_new
            self.net_deploy.params[key][1].data[...] = bias_new

        self.net_deploy.save(self.save_model)


    def conv_weights(self, wt, data):
        h, w = wt.shape[0:2]

        tmp_new = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                tmp_new[i,j] = np.sum(scipy.signal.convolve2d(data[i,j,...],wt[i,j,...],mode='same'))

        return np.sum(tmp_new, axis=1)
'''
from caffeMergeBn import caffeMergeBn
cm = caffeMergeBn('/media/macul/black/mxnet_training/r50/dgx_train1/dgx_train1_31-caffe.prototxt', \
'/media/macul/black/mxnet_training/r50/dgx_train1/dgx_train1_31-caffe.caffemodel', \
'/media/macul/black/mxnet_training/r50/dgx_train1/dgx_train1_31-caffe-merge-bn.prototxt', \
'/media/macul/black/mxnet_training/r50/dgx_train1/dgx_train1_31-caffe-merge-bn.caffemodel', \
'/media/macul/black/mxnet_training/r50/dgx_train1/dgx_train1_conv_bn_list.txt', \
2e-5)
cm.merge_bn()

import numpy as np
import caffe
net = caffe.Net('/media/macul/black/mxnet_training/r50/dgx_train1/dgx_train1_31-caffe.prototxt', '/media/macul/black/mxnet_training/r50/dgx_train1/dgx_train1_31-caffe.caffemodel', caffe.TRAIN)

net.params['stem_conv1'][0].data

'''