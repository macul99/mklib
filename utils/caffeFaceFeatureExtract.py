# evaluate and compare face-align models
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

class caffeFaceFeatureExtract():
    def __init__(   self,
                    model_path, 
                    prototxt_name='model-caffe', 
                    caffemodel_name='model-caffe', 
                    mode='gpu0',
                    outputs_name={'embedding':'fc1'},
                    mean=[127.5, 127.5, 127.5], # in RGB order
                    img_height=112,
                    img_width=112):
        print 'model_path', model_path
        print 'prototxt_name', prototxt_name

        self.model_path = model_path
        self.prototxt_name = prototxt_name
        self.caffemodel_name = caffemodel_name
        self.outputs_name = outputs_name
        self.mean = mean
        self.img_height = img_height
        self.img_width = img_width

        self.mean_array = np.zeros([3,int(img_height), int(img_width)])
        self.mean_array[0,:,:] = self.mean[0]
        self.mean_array[1,:,:] = self.mean[1]
        self.mean_array[2,:,:] = self.mean[2]
        self.scale_value = 0.0078125

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
        self.Net.blobs['data'].reshape(1,3,self.img_height,self.img_width)


    def getImage(self, img_path, img_normalization=True):
        #print(img_path)
        assert isfile(img_path)
        img = cv2.imread(img_path)
        if self.img_width == 96:
            img = img[:,8:104,:]
        assert img.shape==(self.img_height,self.img_width,3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = np.rollaxis(img,2)
        if img_normalization:
            img = img - self.mean_array
            img = img * self.scale_value
        img = img[np.newaxis,...]     
        return img

    # img should be in RGB order
    def processing(self, img):
        assert img.shape==(1, 3, self.img_height,self.img_width)

        self.Net.blobs['data'].data[...] = img
        '''
        import time
        start_time = time.time()
        for i in range(1000):
            self.Net.forward()
        print("--- %s seconds ---" % (time.time() - start_time))
        '''
        out = self.Net.forward()
        out = out[self.outputs_name['embedding']][0]
        out = out/np.linalg.norm(out)
        #print(out)

        print(out.shape)

        output = {}
        output['embedding'] = out

        return output

    # img should be in RGB order
    def getOutput(self, img_path, img_normalization):
        img = self.getImage(img_path, img_normalization)
        return self.processing(img)

    # some model do image normalization inside the model (original insightface model), so set img_normalization flag to false when use it
    def getEmbedding(self, img_path):
        return self.getOutput(img_path, img_normalization=True)['embedding']

'''
import sys
import numpy as np
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from caffeFeatureExtract import caffeFeatureExtract
a=caffeFeatureExtract('/media/macul/black/mxnet_training/r50/insightface-r50-am-lfw',prototxt_name='model-caffe',caffemodel_name='model-caffe',outputs_name={'embedding':'fc1'})
embedding=a.getEmbedding('/media/macul/black/face_database_raw_data/mscelb_from_insightface/Data/0000000/0000.png')


import sys
import numpy as np
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from caffeFeatureExtract import caffeFeatureExtract
a=caffeFeatureExtract('/media/macul/black/mxnet_training/r50/dgx_train1',prototxt_name='dgx_train1_31-caffe-merge-bn',caffemodel_name='dgx_train1_31-caffe-merge-bn',outputs_name={'embedding':'out_embedding'})
embedding=a.getEmbedding('/media/macul/black/face_database_raw_data/mscelb_from_insightface/Data/0000000/0000.png')
a1=caffeFeatureExtract('/media/macul/black/mxnet_training/r50/dgx_train1',prototxt_name='dgx_train1_31-caffe',caffemodel_name='dgx_train1_31-caffe',outputs_name={'embedding':'out_embedding'})
embedding1=a1.getEmbedding('/media/macul/black/face_database_raw_data/mscelb_from_insightface/Data/0000000/0000.png')
np.sum(embedding-embedding1)

import sys
import numpy as np
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from caffeFeatureExtract import caffeFeatureExtract
a=caffeFeatureExtract('/home/macul/Projects/ego_mk_op/ego/recognition/models',prototxt_name='sphereface_deploy',caffemodel_name='sphereface20_ms1m',outputs_name={'embedding':'fc5'},img_width=96)
embedding=a.getEmbedding('/media/macul/black/face_database_raw_data/mscelb_from_insightface/Data/0000000/0000.png')

import time
import sys
sys.path.append('/home/macul/Documents/macul/mklib/utils/')
from caffeFaceFeatureExtract import caffeFaceFeatureExtract
#a1=caffeFaceFeatureExtract('/home/macul/Documents/macul/FaceRecog_ResNet_MX/output/dgx_train7',prototxt_name='resnet20',caffemodel_name='resnet20',outputs_name={'embedding':'out_embedding'})
a1=caffeFaceFeatureExtract('/home/macul/Downloads',prototxt_name='sphereface_deploy',caffemodel_name='sphereface20_ms1m',outputs_name={'embedding':'fc5'},img_width=96)
img=a1.getImage('/media/macul/black/face_database_raw_data/mscelb_from_insightface/Data/0000000/0000.png')
a1.Net.blobs['data'].data[...] = img
def test_time(repeat=1000):
    start_time = time.time()
    for i in range(repeat):
        a1.Net.forward()
    print("--- %s seconds ---" % (time.time() - start_time))
'''