from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
import cPickle as pkl
import time
import os
from os import mkdir, makedirs
from os.path import join, exists, relpath, abspath
import argparse
from imutils import paths

from dan import dan
from ssd import ssd

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--src", required=True, help="source folder")
ap.add_argument("-d", "--dst", required=True, help="destination folder")
ap.add_argument("-danp", "--dan-path", required=False, default='../../mymodels/caffe_dan', help="dan model path")
ap.add_argument("-ssdp", "--ssd-path", required=False, default='../../mymodels/caffe_ssd', help="ssd model path")
ap.add_argument("-ih", "--tgt-img-height", required=False, default=112, type=int, help="target img height")
ap.add_argument("-iw", "--tgt-img-width", required=False, default=112, type=int, help="target img width")
ap.add_argument("-mpb", "--model-pb", required=False, default="", help="pb model")

sim_tr_dst_pts = np.array([[30.29+8,51.69],[65.53+8,51.5],[48.02+8,71.73],[33.55+8,92.37],[62.73+8,92.2]])

def similar_transform_matrix(src_points, dst_points):
    num_points = src_points.shape[0]
    #print('num_points: ',num_points)
    X = np.zeros([num_points*2, 4]).astype(np.float64)
    U = np.zeros([num_points*2, 1]).astype(np.float64)
    for i in range(num_points):
        U[i, 0] = src_points[i,0]
        U[i+num_points, 0] = src_points[i,1]
        X[i, 0] = dst_points[i,0]
        X[i, 1] = dst_points[i,1]
        X[i, 2] = 1.0
        X[i, 3] = 0.0
        X[i+num_points, 0] = dst_points[i,1]
        X[i+num_points, 1] = -dst_points[i,0]
        X[i+num_points, 2] = 0.0
        X[i+num_points, 3] = 1.0
    X_t = np.transpose(X)
    XX = np.matmul(X_t, X)
    r = np.matmul(np.matmul(np.linalg.inv(XX), X_t), U)
    Tinv = np.zeros([3,3])
    Tinv[0,0] = r[0]
    Tinv[0,1] = -r[1]
    Tinv[0,2] = 0.0
    Tinv[1,0] = r[1]
    Tinv[1,1] = r[0]
    Tinv[1,2] = 0.0
    Tinv[2,0] = r[2]
    Tinv[2,1] = r[3]
    Tinv[2,2] = 1.0
    Tinv = np.transpose(Tinv)
    T = np.linalg.inv(Tinv)
    return T[0:2,:]


def process(color_bgr):
    #print(color_array.shape)
    #raw_input()
    faces = ssd_det.getModelOutput(color_bgr)
    if len(faces): # only process first face only
        color_bgr = color_bgr.astype(np.uint8)
        landmarks,score = dan_det.detectLandmark(color_bgr, num_points=5, boundingBox=faces[0])        
        #p1 = (faces[0][0],faces[0][1])
        #p2 = (faces[0][2],faces[0][3])
        STM = similar_transform_matrix(landmarks, sim_tr_dst_pts)
        crop_color = cv2.warpAffine(color_bgr, STM, (args['tgt_img_height'], args['tgt_img_width']))

        return crop_color

    return None


if __name__ == '__main__': 

    args = vars(ap.parse_args())
    args['src'] = abspath(args['src'])
    args['dst'] = abspath(args['dst'])

    assert exists(args['src']), 'source folder does not exist'

    if not exists(args['dst']):
        makedirs(args['dst'])

    ssd_det = ssd(args['ssd_path'])
    dan_det = dan(args['dan_path'])

    imagePaths = list(paths.list_images(args['src']))

    for i, imp in enumerate(imagePaths):
        imp_dst = args['dst']+'/'+relpath(imp, args['src'])
        imp_fd = imp_dst[0:imp_dst.rfind('/')]
        if not exists(imp_fd):
            makedirs(imp_fd)

        im = cv2.imread(imp)
        print(i, im.shape)

        im_align = process(im)

        if type(im_align) != type(None):
            cv2.imwrite(imp_dst,im_align)


    

#python -m crop_align_ssd_dan -s /media/macul/hdd/Projects/NIR_FR_PTH/CASIA/NIR -d /media/macul/hdd/Projects/NIR_FR_PTH/CASIA/NIR_ALIGN