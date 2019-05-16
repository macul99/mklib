# evaluate and compare face-align models
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from PIL import Image
from os import listdir
from os.path import isfile,join
from os import walk
from scipy.io import loadmat
import pickle

class tfFaceFeatureExtract():
    def __init__(   self,
                    model_path,
                    outputs_name={'embedding':'prefix'},
                    mean=[127.5, 127.5, 127.5], # in RGB order
                    img_height=112,
                    img_width=112):
        self.model_path = model_path
        self.outputs_name = outputs_name
        self.img_height = img_height
        self.img_width = img_width
        self.mean = mean
        self.mean_array = np.zeros([int(img_height), int(img_width), 3])
        self.mean_array[:,:,0] = self.mean[0]
        self.mean_array[:,:,1] = self.mean[1]
        self.mean_array[:,:,2] = self.mean[2]
        self.scale_value = 0.0078125

        with tf.gfile.GFile(self.model_path, "rb") as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
    
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(self.graph_def, name=self.outputs_name['embedding'])
        self.graph = graph

        self.graph_x = self.graph.get_tensor_by_name('{}/img_inputs:0'.format(self.outputs_name['embedding']))
        self.graph_y = self.graph.get_tensor_by_name('{}/embeddings:0'.format(self.outputs_name['embedding']))

        self.sess = tf.Session(graph=self.graph)

    def getImage(self, img_path):
        #print(img_path)        
        assert isfile(img_path)
        img = Image.open(img_path)
        assert img.shape==(self.img_height,self.img_width,3)
        img = img - self.mean_array
        img = img * self.scale_value
          
        return np.reshape(img,[1,self.img_height,self.img_width,3])

    # img should be in RGB order
    def processing(self, img):
        assert img.shape==(1,self.img_height,self.img_width,3)

        out = self.sess.run(self.graph_y, feed_dict={self.graph_x: img})[0]

        print(out.shape)

        out = out/np.linalg.norm(out)
        #print(out)
        output = {}
        output['embedding'] = out

        return output

    # img should be in RGB order
    def getOutput(self, img_path):
        img = self.getImage(img_path)
        return self.processing(img)

    # some model do image normalization inside the model (original insightface model), so set img_normalization flag to false when use it
    def getEmbedding(self, img_path):
        return self.getOutput(img_path)['embedding']

'''
import sys
import numpy as np
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from tfFeatureExtract import tfFeatureExtract
a=tfFeatureExtract('/media/macul/black/MobileFaceNet_TF/MobileFaceNet_9925_9680.pb',outputs_name={'embedding':'prefix'})
embedding=a.getEmbedding('/media/macul/black/face_database_raw_data/mscelb_from_insightface/Data/0000000/0000.png')
'''