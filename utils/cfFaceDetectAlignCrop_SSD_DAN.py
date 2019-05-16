# convert rec file to image
import sys
from os.path import isdir, isfile, join, exists
from os import mkdir
from os import listdir, makedirs
import os
import cv2
import numpy as np
from imutils import paths
import progressbar
from PIL import Image
import pickle as pkl
from dan_caffe.dan import dan
from ssd_caffe.ssd import ssd
import time



class cfFaceDetectAlignCrop_SSD_DAN():
    def __init__(self,  ssd_model_path='../../dan_caffe/model', 
                        dan_model_path='../../ssd_caffe/model', 
                        tgt_face_size=[112,112], single_face=True, ctx='gpu0'):
        assert tgt_face_size in [[96,112],[112,112]]
        ssd_model_path = os.path.abspath(ssd_model_path)
        dan_model_path = os.path.abspath(dan_model_path)
        self.tgt_face_size = tgt_face_size
        self.single_face = single_face
        self.ssd_det = ssd(ssd_model_path, mode=ctx, resize_ratio=1.0)
        self.dan_det = dan(model_path=dan_model_path, mode=ctx)

        if self.tgt_face_size == [96,112]:
            self.landmark_ref = np.array([  [30.2946, 51.6963],
                                            [65.5318, 51.5014],
                                            [48.0252, 71.7366],
                                            [33.5493, 92.3655],
                                            [62.7299, 92.2041] ])
        elif self.tgt_face_size == [112,112]:
            self.landmark_ref = np.array([  [30.2946 + 8, 51.6963],
                                            [65.5318 + 8, 51.5014],
                                            [48.0252 + 8, 71.7366],
                                            [33.5493 + 8, 92.3655],
                                            [62.7299 + 8, 92.2041] ])

    def round_int(self, x):
        return int(round(x))


    def similar_transform_matrix(self, src_points, dst_points):
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

    def processSingleImage(self, img_path):
        assert isfile(img_path)
        img = cv2.imread(img_path)
        (h, w) = img.shape[:2]
        if w>1443: # limitation by HDF data layer
            r = 1443.0 / float(w)
            dim = (1443, int(h * r))
            img = cv2.resize(img, dim)

        faces = self.ssd_det.getModelOutput(img)
        outputFace = []
        for i in range(len(faces)):
            landmarks,score = self.dan_det.detectLandmark(img, num_points=5, boundingBox=faces[i])
            STM = self.similar_transform_matrix(landmarks, self.landmark_ref)
            outputFace.append(cv2.warpAffine(img, STM, tuple(self.tgt_face_size)))
            if self.single_face:
                break
        return outputFace

    def processFolder(self, src_folder, dst_folder):
        assert isdir(src_folder)
        if not exists(dst_folder):
            mkdir(dst_folder)

        for i, img_path in enumerate(list(paths.list_images(src_folder))):
            print(i, img_path)
            faces = self.processSingleImage(img_path)          
            dst_img_path = img_path.replace(src_folder,dst_folder)  
            for j in range(len(faces)):
                if not exists(dst_img_path[0:dst_img_path.rfind('/')]):
                    makedirs(dst_img_path[0:dst_img_path.rfind('/')])
                if self.single_face:
                    cv2.imwrite(dst_img_path, faces[j])
                else:
                    tmp_path = dst_img_path[0:dst_img_path.rfind('.')]
                    tmp_path1 = dst_img_path[dst_img_path.rfind('.'):]
                    tmp_path = tmp_path+'_{}'.format(j)+tmp_path1
                    cv2.imwrite(tmp_path, faces[j])


'''
from faceDetectAlignCrop_SSD_DAN import faceDetectAlignCrop_SSD_DAN
fd=faceDetectAlignCrop_SSD_DAN(tgt_face_size=[112,112])
fd.processFolder('/media/macul/black/face_database_raw_data/faceRecog_out','/media/macul/black/face_database_raw_data/faceRecog_out_crop')

fd.processFolder('/media/macul/black/face_database_raw_data/faceRecog_out/database/var/data/ego_java_server_data/IIM_Images','/media/macul/black/face_database_raw_data/faceRecog_out/database/var/data/ego_java_server_data/IIM_Images_Crop')

out=fd.processSingleImage('/media/macul/black/face_database_raw_data/faceRecog_out/database/var/data/ego_java_server_data/IIM_Images/0a1d329811fd4b8db9b84ee5d1f1081f.jpg')
'''