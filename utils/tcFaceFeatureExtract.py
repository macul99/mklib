# for PyTorch
import sys
from os.path import isdir, isfile, join
from os import mkdir
import os
#import cv2
import numpy as np
import torch
import argparse
#from imutils import paths
#import progressbar
from collections import namedtuple
import json
import imp
from PIL import Image

class tcFeatureExtract():
    def __init__(self, model_net, model_weight=None, model_path=None, outputs_name={'embedding':'out_embedding'}, 
                input_shape=(112,112,3), out_dim=512, ctx='gpu', mean_value=None, mmdnn_convert=False): # mmdnn_convert is used for model convertion
        self.model_path = model_path
        self.model_net = model_net
        self.model_weight = model_weight
        self.ctx = ctx
        assert isfile(model_net)
        if type(model_path)==type(None):
            assert type(model_weight)!=type(None)
            assert isfile(model_weight)
            self.MainModel = imp.load_source('MainModel', model_net)
            self.model = self.MainModel.KitModel(model_weight)
        elif type(model_weight)==type(None):
            assert type(model_path)!=type(None)
            assert isfile(model_path)
            self.MainModel = imp.load_source('MainModel', model_net)
            self.model = torch.load(model_path)

        assert len(input_shape)==3
        assert input_shape[2]==3

        if ctx=='cpu':
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.eval() # set model as eval mode          
        self.input_shape = input_shape
        self.out_dim = out_dim
        self.data_shape = (1, self.input_shape[2], self.input_shape[0], self.input_shape[1])
        if mean_value==None:
            self.mean_value = [138.0485, 110.2243, 96.73112] # R,G,B
        else:
            self.mean_value = json.loads(open(mean_value).read())
            self.mean_value = [self.mean_value['R'],self.mean_value['G'],self.mean_value['B']]
        self.mean_value = np.array(self.mean_value, dtype=np.float32).reshape(1,1,3)
        #self.mean_value = torch.from_numpy(self.mean_value)
        self.scale_value = 0.0078125        

    def getImage(self, img_path, img_normalization=True):
        #print(img_path)
        assert isfile(img_path)
        img = Image.open(img_path)
        assert img.mode=='RGB'
        assert img.size==(self.input_shape[0],self.input_shape[1])
        img = np.array(img, dtype=np.float32)

        if img_normalization:
            img = img - self.mean_value
            img = img * self.scale_value
        img = img.transpose((2,0,1))
        img = img[np.newaxis,...]        
        return img

    # img should be in RGB order
    def processing(self, img):
        assert img.shape==self.data_shape

        data = torch.from_numpy(img)
        if self.ctx=='cpu':
            data = torch.autograd.Variable(torch.FloatTensor(data).cpu(), requires_grad = False)
        else:
            data = torch.autograd.Variable(torch.FloatTensor(data).cuda(), requires_grad = False)
	        
        embedding = self.model(data)
        embedding = embedding.cpu().data.numpy()
        embedding = np.squeeze(embedding)
        embedding = embedding/np.linalg.norm(embedding)

        return embedding

    # img should be in RGB order
    def getOutput(self, img_path, img_normalization):
        img = self.getImage(img_path, img_normalization)
        return self.processing(img)

    # some model do image normalization inside the model (original insightface model), so set img_normalization flag to false when use it
    def getEmbedding(self, img_path):
        return self.getOutput(img_path, img_normalization=True)

    # some model do image normalization inside the model (original insightface model), so set img_normalization flag to false when use it
    def getEmbeddingNoNorm(self, img_path):
        return self.getOutput(img_path, img_normalization=False)

'''
# call convertIR2Pytorch.py to convert mxnet model to pytorch model first

import sys
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from PIL import Image
import numpy as np
import torch
from tcFaceFeatureExtract import tcFeatureExtract
extractor_gpu=tcFeatureExtract(model_net='net_pytorch_gpu.py',model_weight='wt_pytorch_gpu.npy', ctx='gpu')
extractor_cpu=tcFeatureExtract(model_net='net_pytorch_cpu.py',model_weight='wt_pytorch_cpu.npy', ctx='cpu')
emb_gpu=extractor_gpu.getEmbedding('seagull112.jpg')
emb_cpu=extractor_cpu.getEmbedding('seagull112.jpg')

extractor_gpu1=tcFeatureExtract(model_net='net_pytorch_gpu.py',model_path='pytorch_gpu.pth', ctx='gpu')
extractor_cpu1=tcFeatureExtract(model_net='net_pytorch_cpu.py',model_path='pytorch_cpu.pth', ctx='cpu')
emb_gpu1=extractor_gpu1.getEmbedding('seagull112.jpg')
emb_cpu1=extractor_cpu1.getEmbedding('seagull112.jpg')
'''
