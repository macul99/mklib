# convert rec file to image
import sys
from os.path import isdir, isfile, join
from os import mkdir
import os
import cv2
import numpy as np
from mklib.nn.mxiter.mxiter import MyImageIter
import mxnet as mx
import argparse
from imutils import paths
import progressbar
from collections import namedtuple
import json

class mxFaceFeatureExtract():
    def __init__(self, model_path, model_prefix, model_epoch, outputs_name={'embedding':'embedding_output','landmark':'lro_output'}, 
                input_shape=(112,112,3), out_dim=512, ctx='gpu', mean_value=None, mmdnn_convert=False): # mmdnn_convert is used for model convertion
        print(os.path.sep.join([model_path, model_prefix+'-{0:04d}.params'.format(model_epoch)]))
        assert isfile(os.path.sep.join([model_path, model_prefix+'-{0:04d}.params'.format(model_epoch)]))
        assert isfile(os.path.sep.join([model_path, model_prefix+'-symbol.json']))
        assert len(input_shape)==3
        assert input_shape[2]==3

        self.Batch = namedtuple('Batch',['data'])
        self.model_path = model_path
        self.model_prefix = model_prefix
        self.model_epoch = model_epoch
        self.outputs_name = outputs_name
        self.input_shape = input_shape
        self.out_dim = out_dim
        self.data_shape = (1, self.input_shape[2], self.input_shape[0], self.input_shape[1])
        if mean_value==None:
            self.mean_value = [127.5]*3 # R,G,B
        else:
            self.mean_value = json.loads(open(mean_value).read())
            self.mean_value = [self.mean_value['R'],self.mean_value['G'],self.mean_value['B']]
        self.mean_value = np.array(self.mean_value, dtype=np.float32).reshape(1,1,3)
        self.mean_value = mx.nd.array(self.mean_value).reshape((1,1,3))
        self.scale_value = 0.0078125

        if ctx=='cpu':
            self.ctx = mx.cpu()
        else:
            self.ctx = mx.gpu(1)
        sym_tmp, self.arg_params, self.aux_params = mx.model.load_checkpoint(os.path.sep.join([model_path, model_prefix]), model_epoch)
        all_layers = sym_tmp.get_internals()
        output_list = all_layers.list_outputs()
        
        try:            
            if mmdnn_convert:
                sym_list = [all_layers[self.outputs_name[nm]] for nm in ['embedding']]
            else:
                sym_list = [all_layers[self.outputs_name[nm]] for nm in self.outputs_name.keys()]
            self.sym = mx.sym.Group(sym_list)
        except:
            print(output_list)
            assert False, 'outputs not valid!!!'

        self.model = mx.mod.Module(symbol=self.sym, context=self.ctx, label_names=None)
        if mmdnn_convert:
            self.model.bind(for_training=False, data_shapes=[('data',self.data_shape)])
        else:            
            self.model.bind(for_training=False, data_shapes=[('data',self.data_shape)], label_shapes=self.model._label_shapes)        
        self.model.set_params(self.arg_params, self.aux_params, allow_missing=True, allow_extra=True)

        self.landmark_ref = np.array([  [30.2946 + 8, 51.6963],
                                        [65.5318 + 8, 51.5014],
                                        [48.0252 + 8, 71.7366],
                                        [33.5493 + 8, 92.3655],
                                        [62.7299 + 8, 92.2041] ])
        self.landmark_w_h = 112.0

    def saveModel(self, model_path, name):
        self.model.save_checkpoint(join(model_path, name), 0)

    def getImage(self, img_path, img_normalization=True):
        #print(img_path)
        assert isfile(img_path)
        img = mx.image.imread(img_path)
        assert self.input_shape == img.shape
        img = img.astype('float32')
        if img_normalization:
            img = img - self.mean_value
            img = img * self.scale_value
        img = img.transpose((2,0,1))
        img = img.expand_dims(axis=0)        
        return img

    # img should be in RGB order
    def processing(self, img):
        assert img.shape==self.data_shape        

        self.model.forward(self.Batch([img]))
        output = {}
        for i, out_name in enumerate(self.outputs_name.keys()):
            if out_name == 'embedding':
                embedding = self.model.get_outputs()[i].asnumpy()
                embedding = np.squeeze(embedding)
                embedding = embedding/np.linalg.norm(embedding)
                output[out_name] = embedding
            elif out_name == 'landmark': # landmark regression
                landmark = self.model.get_outputs()[i].asnumpy()
                landmark = np.squeeze(landmark)
                for j in range(self.landmark_ref.shape[0]):
                    landmark[j*2] = landmark[j*2]*self.landmark_w_h+self.landmark_ref[j,0]
                    landmark[j*2+1] = landmark[j*2+1]*self.landmark_w_h+self.landmark_ref[j,1]
                output[out_name] = landmark
        return output

    # img should be in RGB order
    def getOutput(self, img_path, img_normalization):
        img = self.getImage(img_path, img_normalization)
        return self.processing(img)

    # some model do image normalization inside the model (original insightface model), so set img_normalization flag to false when use it
    def getEmbedding(self, img_path):
        return self.getOutput(img_path, img_normalization=True)['embedding']

    # some model do image normalization inside the model (original insightface model), so set img_normalization flag to false when use it
    def getEmbeddingNoNorm(self, img_path):
        return self.getOutput(img_path, img_normalization=False)['embedding']

    def getLandmark(self, img_path):
        return self.getOutput(img_path, img_normalization=True)['landmark']

    # don't do image normalization
    def getLandmarkNoNorm(self, img_path):
        return self.getOutput(img_path, img_normalization=False)['landmark']

    def net_forward(self, img):
        self.model.forward(self.Batch([img]))


'''
import sys
import numpy as np
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from mxFeatureExtract import mxFeatureExtract
a=mxFeatureExtract('/media/macul/black/mxnet_training/r50/insightface-r50-am-lfw','model',0,outputs_name={'embedding':'fc1_output'}, mean_value=None)
embedding=a.getEmbeddingNoNorm('/media/macul/black/face_database_raw_data/mscelb_from_insightface/Data/0000000/0000.png')


import sys
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from mxFeatureExtract import mxFeatureExtract
a=mxFeatureExtract('/media/macul/black/mxnet_training/r50/Baseline_insightDatabase_single_loss','train_11',31,outputs_name={'embedding':'out_embedding_output'}, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
#a=mxFeatureExtract('/media/macul/black/mxnet_training/r50/Baseline_insightDatabase_single_loss','train_11-deploy',31,outputs_name={'embedding':'out_embedding_output'}, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
embedding=a.getEmbeddingNoNorm('/media/macul/black/face_database_raw_data/mscelb_from_insightface/Data/0000000/0000.png')


import sys
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from mxFeatureExtract import mxFeatureExtract
#a1=mxFeatureExtract('/media/macul/black/mxnet_training/r50/train_19','train_19',1, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
a1=mxFeatureExtract('/media/macul/black/mxnet_training/r50/Baseline_insightDB_landmarkloss_before_bugfix','train_14',33, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
embedding_1=a1.getEmbedding('/media/macul/black/face_database_raw_data/mscelb_from_insightface/Data/0000000/0000.png')


from mxFeatureExtract import mxFeatureExtract
import time
a1=mxFeatureExtract('/media/macul/black/mxnet_training/r50/server_train16','train_16',3942, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
img=a1.getImage('/media/macul/black/face_database_raw_data/mscelb_from_insightface/Data/0000000/0000.png')
def test_time(repeat=1000):
    start_time = time.time()
    for i in range(repeat):
        a1.net_forward(img)
    print("--- %s seconds ---" % (time.time() - start_time))


sym, arg_params, aux_params = mx.model.load_checkpoint('/media/macul/black/mxnet_training/r50/Baseline_insightDatabase_single_loss/train_11-deploy', 31)
'''