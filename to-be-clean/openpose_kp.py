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
from openpose import *


class openpose_kp():
    def __init__(   self, 
                    model_path, 
                    input_height, 
                    model_name="COCO",
                    mode='gpu0'):
        print 'model_path', model_path
        #self.greyscale = False
        #self.boundingBoxIdx = 0 # either 0 or 1 since two are given in the dataset
        #self.nmsTreshold = [0.5, 0.7, 0.7, 0.7]

        self.model_path = model_path
        self.model_name = model_name
        
        #self.displayImg = display_img
        #self.displayInfo = False

        self.kp_name = ['Nose',
                        'Neck',
                        'RShoulder',
                        'RElbow',
                        'RWrist',
                        'LShoulder',
                        'LElbow',
                        'LWrist',
                        'RHip',
                        'RKnee',
                        'RAnkle',
                        'LHip',
                        'LKnee',
                        'LAnkle',
                        'REye',
                        'LEye',
                        'REar',
                        'LEar']

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

        #self.PNet = caffe.Net(join(model_path, "det1.prototxt"), 
        #                        join(model_path, "det1.caffemodel"), caffe.TEST)
        #self.RNet = caffe.Net(join(model_path, "det2.prototxt"), 
        #                        join(model_path, "det2.caffemodel"), caffe.TEST)
        self.target_input_height = input_height
        self.params = dict()
        self.params["logging_level"] = 3
        self.params["output_resolution"] = "-1x-1"
        self.params["net_resolution"] = "-1x{}".format(self.target_input_height)
        self.params["model_pose"] = model_name
        self.params["alpha_pose"] = 0.6
        self.params["scale_gap"] = 0.3
        self.params["scale_number"] = 1
        self.params["render_pose"] = 0
        self.params["render_threshold"] = 0.05
        # If GPU version is built, and multiple GPUs are available, set the ID here
        self.params["num_gpu_start"] = 1
        self.params["disable_blending"] = True
        # Ensure you point to the correct path where models are located
        self.params["default_model_folder"] = model_path
        # Construct OpenPose object allocates GPU memory
        self.op_net = OpenPose(self.params)

    def detectLandmark(self, img_url):
        image = cv2.imread(img_url)
        #print('image shape: ', image.shape)
        net_resolution = "-1x"+str(int(round(1.0*image.shape[0]/16)*16))
        #print('net_resolution', net_resolution)
        self.op_net.resizeInput(self.params["output_resolution"], net_resolution, self.params["scale_gap"], self.params["scale_number"])
        keypoints, output_image = self.op_net.forward(image, True)
        #print(keypoints)
        '''
        points = [] 

        for i in range(keypoints.shape[0]):
            if self.isInsideBoundingBox(keypoints[i][self.nose_idx][0:2], boundingBox):
                tmp = [keypoints[i][self.r_eye_idx][0], keypoints[i][self.l_eye_idx][0], keypoints[i][self.nose_idx][0], \
                        keypoints[i][self.r_eye_idx][1], keypoints[i][self.l_eye_idx][1], keypoints[i][self.nose_idx][1]]
                points += [tmp]
                break        

        #print('points [2]: ', points)
        '''
        return keypoints #np.array(points)

#op = openpose_kp('/home/macul/libraries/openpose/models/', 240)
#kps = op.detectLandmark('/home/macul/2018-10-15.png')
