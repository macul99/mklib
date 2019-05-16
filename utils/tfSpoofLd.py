import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import argparse
import logging
import json
from os.path import isdir
from os import mkdir
from shutil import copyfile
import time
from tensorflow.core.protobuf import config_pb2
from PIL import Image
import pickle
from tensorflow.python.platform import gfile
#import sys
#sys.path.append('/home/macul/libraries/mk_utils/mklib/nn/tfnet/')
#sys.path.append('/home/macul/libraries/mk_utils/mklib/nn/tfloss/')
#sys.path.append('/home/macul/libraries/mk_utils/tf_spoof/')
#sys.path.append('/home/macul/libraries/mk_utils/spoofing_lbp/')
from .LineDetect import LineDetect


class TfSpoofLd():
    def __init__(self):
        self.feature_size = 80
        self.lineDetect = LineDetect()

    # img should be in RGB order
    def eval(self, sess, input_tensor, output_tensor, img=None, bbox=None):
        if type(img) == type(None):
            lines = np.random.rand(80)
        else:
            lines = self.lineDetect.get_lines(img, bbox)

        if type(lines) == type(None):
            prediction = np.array([[[[1.0, 0.0]]]])
        else:
            #features_tensor = self.feature_extraction(img, self.ft_extractor)
            prediction = sess.run(output_tensor, feed_dict={input_tensor: lines.reshape([-1,1,1,self.feature_size])})

        return prediction

    def load_pb(self, pb_path):           
        print("load graph")
        f = gfile.FastGFile(pb_path,'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph_nodes=[n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='prefix')
                
            lines_in = graph.get_tensor_by_name('prefix/clf_data:0')
            pred_out = graph.get_tensor_by_name('prefix/clf_prob:0')
        
            sess = tf.Session(graph=graph)
        return sess, lines_in, pred_out, names, graph_nodes

        

'''
import tensorflow as tf
import numpy as np
import os
from os.path import isdir
from os import mkdir
from shutil import copyfile
import time
from PIL import Image
import pickle
from spoofing_ld.TfSpoofLd import TfSpoofLd

pb_path = '~/mymodels/tf_spoof_ld/clf.pb'
tfSpoofLd = TfSpoofLd()
sess, lines_in, pred_out, names, graph_nodes = tfSpoofLd.load_pb(pb_path)
prediction = tfSpoofLd.eval(sess, lines_in, pred_out)
print(prediction)

'''
