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
import cPickle as pickle
from scipy import ndimage

class dan():
    def __init__(   self, 
                    model_path='~/mymodels/caffe_dan', 
                    prototxt_name='dan', 
                    caffemodel_name='dan', 
                    mode='gpu0', 
                    th=0.9):
        print ('model_path', model_path)
        print ('prototxt_name', prototxt_name)

        self.model_path = model_path
        self.prototxt_name = prototxt_name
        self.caffemodel_name = caffemodel_name
        self.th = th

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

        f=np.load(join(model_path, 'DAN.npz'),'rb')
        self.meanInitLandmarks = f['initLandmarks']
        self.meanImg=f['meanImg']
        self.stdDevImg=f['stdDevImg']

        self.Net = caffe.Net(join(model_path, prototxt_name+".prototxt"), 
                                join(model_path, caffemodel_name+".caffemodel"), caffe.TEST)

    def getModelOutput(self, image): # image should be single color image
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
        print(out)

        h,w = out.shape

        result = []

        for i in range(h):
            if (out[i][0]!=-1) & (out[i][2]>self.th):
                result += [[out[i][3]*img_w, out[i][4]*img_h, out[i][5]*img_w, out[i][6]*img_h]]

        return np.array(result).astype(np.int16)

    def detectLandmark(self, image, num_points=68, boundingBox=None): # image should be bgr with shape [h,w,c]
        # change from RGB to BGR
        img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)    
        h, w = img.shape

        if type(boundingBox) != type(None):
            total_boxes = np.array(boundingBox).astype(long)
        else:
            total_boxes = np.array([0, 0, w-1, h-1])

        initLandmarks = self.bestFitRect(None, self.meanInitLandmarks, total_boxes)

        A, t = self.bestFit(self.meanInitLandmarks, initLandmarks, True)
        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)

        inputImg = np.zeros((112, 112), dtype=np.float32)
        inputImg = ndimage.interpolation.affine_transform(img, A2, t2[[1, 0]], output_shape=(112, 112))
        #cv2.imshow("image", inputImg)
        #key = cv2.waitKey(0)
        #cv2.destroyAllWindows()

        inputImg = inputImg[np.newaxis]
        inputImg = inputImg - self.meanImg
        inputImg = inputImg / self.stdDevImg
        inputImg = inputImg[np.newaxis]

        points = []               

        self.Net.blobs['data'].reshape(1, 1, 112, 112)
        self.Net.blobs['data'].data[...] = inputImg
        out = self.Net.forward()
        
        # mymtcnn net has no prob1 output, set score to 1
        landmarks = out['s1_output'][0]
        landmarks = landmarks.reshape((-1, 2))
        score = out['s1_confidence'][0][1]

        points=np.dot((landmarks+self.meanInitLandmarks) - t, np.linalg.inv(A))

        if num_points==5:
            points_5 = np.zeros([5,2])
            points_5[0,0] = np.mean(points[36:42,0])
            points_5[0,1] = np.mean(points[36:42,1])
            points_5[1,0] = np.mean(points[42:48,0])
            points_5[1,1] = np.mean(points[42:48,1])
            points_5[2,0] = np.mean(points[30,0])
            points_5[2,1] = np.mean(points[30,1])
            points_5[3,0] = np.mean(points[48,0])
            points_5[3,1] = np.mean(points[48,1])
            points_5[4,0] = np.mean(points[54,0])
            points_5[4,1] = np.mean(points[54,1])
            points = points_5

        #print('landmarks: ', points.shape, points)
        #print('score: ', score)
        return points, score

    def bestFitRect(self, points, meanS, box=None):
        if box is None:
            box = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])
        boxCenter = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ])

        boxWidth = box[2] - box[0]
        boxHeight = box[3] - box[1]

        meanShapeWidth = meanS[:, 0].max() - meanS[:, 0].min()
        meanShapeHeight = meanS[:, 1].max() - meanS[:, 1].min()

        scaleWidth = boxWidth / meanShapeWidth
        scaleHeight = boxHeight / meanShapeHeight
        scale = (scaleWidth + scaleHeight) / 2

        S0 = meanS * scale

        S0Center = [(S0[:, 0].min() + S0[:, 0].max()) / 2, (S0[:, 1].min() + S0[:, 1].max()) / 2]    
        S0 += boxCenter - S0Center

        return S0

    def bestFit(self, destination, source, returnTransform=False):
        destMean = np.mean(destination, axis=0)
        srcMean = np.mean(source, axis=0)

        srcVec = (source - srcMean).flatten()
        destVec = (destination - destMean).flatten()

        a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec)**2
        b = 0
        for i in range(destination.shape[0]):
            b += srcVec[2*i] * destVec[2*i+1] - srcVec[2*i+1] * destVec[2*i] 
        b = b / np.linalg.norm(srcVec)**2
        
        T = np.array([[a, b], [-b, a]])
        srcMean = np.dot(srcMean, T)

        if returnTransform:
            return T, destMean - srcMean
        else:
            return np.dot(srcVec.reshape((-1, 2)), T) + destMean

    def drawBoxes(self, im, boxes):
        x1 = boxes[0]
        y1 = boxes[1]
        x2 = boxes[2]
        y2 = boxes[3]

        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
        return im

    def drawPoints(self, im, points):
        for i in range(points.shape[0]):
            cv2.circle(im, (int(points[i,0]), int(points[i, 1])), 3, (0,255,0), 2)
        return im


class dan1():
    def __init__(self, dataset, model_path, prototxt_name, caffemodel_name, display_img=False, threshold=0.7, mode='gpu'):
        print ('model_path', model_path)
        print ('prototxt_name', prototxt_name)

        self.boundingBoxIdx = 0 # either 0 or 1 since two are given in the dataset

        self.dataset = dataset
        self.model_path = model_path
        self.prototxt_name = prototxt_name
        self.caffemodel_name = caffemodel_name
        self.threshold = threshold
        
        self.displayImg = display_img
        self.displayInfo = False

        if mode == 'cpu':
            caffe.set_mode_cpu()
        elif mode == 'gpu':
            caffe.set_device(0)
            caffe.set_mode_gpu()
        else:
            assert False, 'error: please specify mode'

        f=np.load(join(model_path, 'DAN.npz'),'rb')
        self.meanInitLandmarks = f['initLandmarks']
        self.meanImg=f['meanImg']
        self.stdDevImg=f['stdDevImg']

        #self.PNet = caffe.Net(join(model_path, "det1.prototxt"), 
        #                        join(model_path, "det1.caffemodel"), caffe.TEST)
        #self.RNet = caffe.Net(join(model_path, "det2.prototxt"), 
        #                        join(model_path, "det2.caffemodel"), caffe.TEST)
        self.ONet = caffe.Net(join(model_path, prototxt_name+".prototxt"), 
                                join(model_path, caffemodel_name+".caffemodel"), caffe.TEST)

    def getModelOutput(self, num_points=5):
        assert num_points in [3,5,68], 'only 3, 5 or 68 points are supported'
        outputDic = {}
        keys = self.dataset.dataDic.keys()
        keys.sort()

        for k in keys:
            if self.dataset.preprocessed:
                img_orig = self.dataset.dataDic[k]['image']
            else:
                img_orig = cv2.imread(join(self.dataset.path, k))
            
            img = cv2.cvtColor(img_orig,cv2.COLOR_BGR2GRAY)            

            #if self.dataset.dataDic[k]['boundingBox'].any() != None:
            #    leftP,topP,rightP,bottomP = np.array(self.dataset.dataDic[k]['boundingBox'][self.boundingBoxIdx]).astype(long)
            ##   convert boundingBox to square
            #    img = img[topP:bottomP, leftP:rightP, :]            
                
            #_, points = self.detectFace(img)

            if type(self.dataset.dataDic[k]['boundingBox']) == type(None):
                points, score = self.detectLandmark(img, None)
            else:                
                points, score = self.detectLandmark(img, self.dataset.dataDic[k]['boundingBox'][self.boundingBoxIdx])
                if self.displayImg: img = self.drawBoxes(img, self.dataset.dataDic[k]['boundingBox'][self.boundingBoxIdx])

            if score>self.threshold:
                if num_points == 68:
                    outputDic[k] = points
                elif num_points == 5:
                    tmpPoints = np.zeros((5,2))
                    tmpPoints[0] = np.mean(points[self.dataset.fivePointsIdx[0],], 0)
                    tmpPoints[1] = np.mean(points[self.dataset.fivePointsIdx[1],], 0)
                    tmpPoints[2] = np.mean(points[self.dataset.fivePointsIdx[2],], 0)
                    tmpPoints[3] = np.mean(points[self.dataset.fivePointsIdx[3],], 0)
                    tmpPoints[4] = np.mean(points[self.dataset.fivePointsIdx[4],], 0)

                    outputDic[k] = tmpPoints
                elif num_points == 3:
                    tmpPoints = np.zeros((3,2))
                    tmpPoints[0] = np.mean(points[self.dataset.fivePointsIdx[0],], 0)
                    tmpPoints[1] = np.mean(points[self.dataset.fivePointsIdx[1],], 0)
                    tmpPoints[2] = np.mean(points[self.dataset.fivePointsIdx[2],], 0)

                    outputDic[k] = tmpPoints
                else:
                    assert False, 'Error, only 68, 5 or 3 points are supported!'
            else:
                print ('error found, points is: ', points)
                print ('error file name: ', k)
                outputDic[k] = []
                continue
            
            if self.displayImg:
                #img = self.drawPoints(img, outputDic[k])
                cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                #cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow('img', img)
                kVal=cv2.waitKey()
                if kVal == 93:
                    cv2.destroyAllWindows()
                    break
                else:
                    continue

            #break

        return outputDic


    def detectLandmark(self, image, boundingBox): # image should be bgr with shape [h,w,c]
        # change from RGB to BGR
        img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)    
        h, w = img.shape

        if type(boundingBox) != type(None):
            total_boxes = np.array(boundingBox).astype(long)
        else:
            total_boxes = np.array([0, 0, w-1, h-1])

        initLandmarks = self.bestFitRect(None, self.meanInitLandmarks, total_boxes)

        A, t = self.bestFit(self.meanInitLandmarks, initLandmarks, True)
        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)

        inputImg = np.zeros((112, 112), dtype=np.float32)
        inputImg = ndimage.interpolation.affine_transform(img, A2, t2[[1, 0]], output_shape=(112, 112))
        #cv2.imshow("image", inputImg)
        #key = cv2.waitKey(0)
        #cv2.destroyAllWindows()

        inputImg = inputImg[np.newaxis]
        inputImg = inputImg - self.meanImg
        inputImg = inputImg / self.stdDevImg
        inputImg = inputImg[np.newaxis]

        points = []               

        self.ONet.blobs['data'].reshape(1, 1, 112, 112)
        self.ONet.blobs['data'].data[...] = inputImg
        out = self.ONet.forward()
        
        # mymtcnn net has no prob1 output, set score to 1
        landmarks = out['s1_output'][0]
        landmarks = landmarks.reshape((-1, 2))
        score = out['s1_confidence'][0][1]

        points=np.dot((landmarks+self.meanInitLandmarks) - t, np.linalg.inv(A))

        #print('landmarks: ', points.shape, points)
        #print('score: ', score)
        return points, score

    def bestFitRect(self, points, meanS, box=None):
        if box is None:
            box = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])
        boxCenter = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ])

        boxWidth = box[2] - box[0]
        boxHeight = box[3] - box[1]

        meanShapeWidth = meanS[:, 0].max() - meanS[:, 0].min()
        meanShapeHeight = meanS[:, 1].max() - meanS[:, 1].min()

        scaleWidth = boxWidth / meanShapeWidth
        scaleHeight = boxHeight / meanShapeHeight
        scale = (scaleWidth + scaleHeight) / 2

        S0 = meanS * scale

        S0Center = [(S0[:, 0].min() + S0[:, 0].max()) / 2, (S0[:, 1].min() + S0[:, 1].max()) / 2]    
        S0 += boxCenter - S0Center

        return S0

    def bestFit(self, destination, source, returnTransform=False):
        destMean = np.mean(destination, axis=0)
        srcMean = np.mean(source, axis=0)

        srcVec = (source - srcMean).flatten()
        destVec = (destination - destMean).flatten()

        a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec)**2
        b = 0
        for i in range(destination.shape[0]):
            b += srcVec[2*i] * destVec[2*i+1] - srcVec[2*i+1] * destVec[2*i] 
        b = b / np.linalg.norm(srcVec)**2
        
        T = np.array([[a, b], [-b, a]])
        srcMean = np.dot(srcMean, T)

        if returnTransform:
            return T, destMean - srcMean
        else:
            return np.dot(srcVec.reshape((-1, 2)), T) + destMean


    def drawBoxes(self, im, boxes):
        x1 = boxes[0]
        y1 = boxes[1]
        x2 = boxes[2]
        y2 = boxes[3]

        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
        return im

    def drawPoints(self, im, points):
        for i in range(points.shape[0]):
            cv2.circle(im, (int(points[i,0]), int(points[i, 1])), 3, (0,255,0), 2)
        return im
