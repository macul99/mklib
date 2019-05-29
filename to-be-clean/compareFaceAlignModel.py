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
import cPickle as pickle
from openpose import *
from scipy import ndimage

# create dataset with label points, bounding box info
class Dataset:
    def __init__(self, path, bound_box_file='bounding_boxes.mat', img_ext=['jpg','png'], lbl_ext='pts', openpose_bbox=False, op_model_path='/home/macul/libraries/openpose/models/', op_fh=48):

        self.path = path # put img, face points and bounding boxes file (if any) in the same folder
        self.boundBoxFile = bound_box_file
        self.imgExt = img_ext
        self.lblExt = lbl_ext
        self.dataDic = {}
        self.preprocessed = 0
        #self.fivePointsIdx = [range(37-1,43-1), range(43-1,49-1), range(32-1,37-1), [49-1], [55-1]]
        self.fivePointsIdx = [range(37-1,43-1), range(43-1,49-1), [31-1], [49-1], [55-1]]
        self.openpose_bbox = openpose_bbox
        if self.openpose_bbox:
            self.op_kp = OpenPoseKeyPoints(None, op_model_path, op_fh)

        # load bounding box infoc
        if type(self.boundBoxFile) != type(None):
            boundingBoxDic, boundingBoxDicOpenpose = self.getBoundingBox()

        # get img file names 
        for (_, _, imgList) in walk(self.path):
            imgList = [imgList[i] 
                            for i in range(len(imgList)) if imgList[i].split('.')[1] in self.imgExt]
            break # break the first time it yields to get the filename for the top directory only

        # update self.dataDic        
        for i in range(len(imgList)):
            tmpDic = {}
            tmpDic['lblPoints'], tmpDic['lbl5Points'] = self.getLabelPoints(imgList[i])

            # only processing 68 points data
            if tmpDic['lblPoints'].shape[0] != 68:
                continue

            if self.boundBoxFile:
                if self.openpose_bbox:
                    tmpDic['boundingBox'] = boundingBoxDicOpenpose[imgList[i]]
                else:
                    tmpDic['boundingBox'] = boundingBoxDic[imgList[i]]

                if  (tmpDic['lbl5Points'][:,0] < tmpDic['boundingBox'][0][0]).any() or \
                    (tmpDic['lbl5Points'][:,0] > tmpDic['boundingBox'][0][2]).any() or \
                    (tmpDic['lbl5Points'][:,1] < tmpDic['boundingBox'][0][1]).any() or \
                    (tmpDic['lbl5Points'][:,1] > tmpDic['boundingBox'][0][3]).any():

                    #print tmpDic['lblPoints']
                    #print tmpDic['boundingBox'][0]
                    continue
            else:
                if self.openpose_bbox:
                    tmpBbox = self.getBoundingBoxOpenpose(imgList[i])
                    if type(tmpBbox)!=type(None):
                        tmpDic['boundingBox'] = np.array([tmpBbox,tmpBbox])
                    else:
                        tmpDic['boundingBox'] = self.estimateBoundingBox(tmpDic['lbl5Points'])
                else:
                    tmpDic['boundingBox'] = self.estimateBoundingBox(tmpDic['lbl5Points'])
        
            self.dataDic[imgList[i]] = tmpDic
               
    def getLabelPoints(self, imgFileName):
        fileName = imgFileName.split('.')[0] + '.' + self.lblExt
        with open(join(self.path, fileName), 'r') as f:
            lblLines = f.readlines()

        if '}' in lblLines:
            lblLines = lblLines[lblLines.index('{\n')+1:lblLines.index('}')]
        else:
            lblLines = lblLines[lblLines.index('{\n')+1:lblLines.index('}\n')]
        lblPoints = np.zeros((len(lblLines), len(lblLines[0].split())))
        lbl5Points = np.zeros((5, len(lblLines[0].split())))

        # only process 68 points data
        if len(lblLines) != 68:
            return lblPoints, lbl5Points

        for i in range(len(lblLines)):
            #print(lblLines[i])
            for j, s in enumerate(lblLines[i].split()):
                #print(s)
                lblPoints[i,j] = float(s)

        for i in range(5):
            lbl5Points[i,:] = np.mean(lblPoints[self.fivePointsIdx[i],:], axis=0)

        return lblPoints, lbl5Points

    def getBoundingBox(self):
        boundingBox = loadmat(join(self.path, self.boundBoxFile))
        boundingBox = boundingBox['bounding_boxes']
        boundingBox = boundingBox[0,:]

        boundingBox = [boundingBox[i][0][0] for i in range(len(boundingBox))]

        boundingBoxDic = {}
        boundingBoxDicOpenpose = {}

        for i in range(len(boundingBox)):
            boundingBoxDic[boundingBox[i][0][0]] = np.array([boundingBox[i][1][0], boundingBox[i][2][0]])

            if self.openpose_bbox:
                tmpBbox = self.getBoundingBoxOpenpose(boundingBox[i][0][0], boundingBox[i][1][0])
                if type(tmpBbox) != type(None):
                    boundingBoxDicOpenpose[boundingBox[i][0][0]] = np.array([tmpBbox, tmpBbox])
                else:
                    print('cannot detect boundingbox by openpose model!!!')
                    boundingBoxDicOpenpose[boundingBox[i][0][0]] = np.array([boundingBox[i][1][0], boundingBox[i][2][0]])
        return boundingBoxDic, boundingBoxDicOpenpose

    def getBoundingBoxOpenpose(self, image_name, bbox_groundtruth=None):
        print(image_name)
        img = cv2.imread(join(self.path, image_name))  

        tmpBboxes = self.op_kp.detectBoundingBox(img, bbox_groundtruth)
        tmpBbox = None
        for bbx in tmpBboxes:
            print('bbx: ', bbx)
            if type(bbox_groundtruth) == type(None):
                bbox_groundtruth = [0,0,img.shape[1],img.shape[0]]
            if self.op_kp.isInsideBoundingBox([np.mean([bbx[0],bbx[2]]),np.mean([bbx[1],bbx[3]])], bbox_groundtruth):
                tmpBbox = bbx
                break
        return tmpBbox

    def estimateBoundingBox(self, points):
        min_x = min(points[:,0])
        max_x = max(points[:,0])
        min_y = min(points[:,1])
        max_y = max(points[:,1])

        x1 = min_x - 0.5 * (max_x - min_x)
        x2 = max_x + 0.5 * (max_x - min_x)
        y1 = min_y - 0.9 * (max_y - min_y)
        y2 = max_y + 0.9 * (max_y - min_y)

        x1 = max(x1, 0)
        y1 = max(y1, 0)

        return np.array([[x1, y1, x2, y2],[x1, y1, x2, y2]])

    def computeError(self, predictDic, num_points=68):
        keys = self.dataDic.keys()
        keys.sort()
        errorDic = {}

        for k in keys:
            if predictDic[k] == []:
                errorDic[k] = 0
                continue

            interocular_distance = np.linalg.norm(self.dataDic[k]['lbl5Points'][0] - self.dataDic[k]['lbl5Points'][1])

            if num_points == 68:
                errorDic[k] = np.mean(np.linalg.norm(self.dataDic[k]['lblPoints'] - predictDic[k],axis=1))/interocular_distance
            elif num_points == 5:
                errorDic[k] = np.mean(np.linalg.norm(self.dataDic[k]['lbl5Points'] - predictDic[k],axis=1))/interocular_distance
            elif num_points == 3:
                errorDic[k] = np.mean(np.linalg.norm(self.dataDic[k]['lbl5Points'][0:num_points,:] - predictDic[k][0:num_points,:],axis=1))/interocular_distance
            else:
                assert False, 'Error, only 68, 5 or 3 points are supported!'

        return errorDic

    def showPoints(self, num_points=68, predicted=None):
        keys = self.dataDic.keys()
        keys.sort()

        for k in keys:
            if type(predicted) != type(None) and predicted[k]==[]:
                continue

            interocular_distance = np.linalg.norm(self.dataDic[k]['lbl5Points'][0] - self.dataDic[k]['lbl5Points'][1])

            if self.preprocessed:
                img = self.dataDic[k]['image']
            else:
                img = cv2.imread(join(self.path, k))

            if type(self.dataDic[k]['boundingBox']) == type(None):
                height, width = img.shape[:2]
                bBox = (0, 0, width-1, height-1)
            else:
                bBox = np.array(self.dataDic[k]['boundingBox'][0]).astype(long)
            
            img = self.drawBoxes(img, bBox)

            if num_points==68:
                img = self.drawPoints(img, self.dataDic[k]['lblPoints'], interocular_distance=interocular_distance)
            elif num_points==5:
                img = self.drawPoints(img, self.dataDic[k]['lbl5Points'], interocular_distance=interocular_distance)
            elif num_points==3:
                img = self.drawPoints(img, self.dataDic[k]['lbl5Points'][0:num_points,:], interocular_distance=interocular_distance)
            else:
                assert False, 'only 68 ,5 or 3 points are supported'

            if type(predicted) != type(None):
                img = self.drawPoints(img, predicted[k], shape='rect', color=(0,0,255), interocular_distance=interocular_distance)

            img = img[bBox[1]:bBox[3],bBox[0]:bBox[2]]
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            #cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            print k
            cv2.imshow('img', img)
            kVal=cv2.waitKey()
            if kVal == 93:
                cv2.destroyAllWindows()
                break
            else:
                continue

    def drawBoxes(self, im, boxes):
        x1 = boxes[0]
        y1 = boxes[1]
        x2 = boxes[2]
        y2 = boxes[3]

        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
        return im

    def drawPoints(self, im, points, shape='circle',color=(0,255,0), interocular_distance=None):
        for i in range(points.shape[0]):
            if shape == 'circle':
                if interocular_distance:
                    lineWidth = int(interocular_distance/25)
                    radius = int(lineWidth * 1.5)
                else:
                    radius = 3
                    lineWidth = 2
                cv2.circle(im, (int(points[i,0]), int(points[i, 1])), radius, color, lineWidth)
            elif shape == 'rect':
                if interocular_distance:
                    lineWidth = int(interocular_distance/25)
                    boxWidth = int(lineWidth * 3)
                else:
                    boxWidth = 6
                    lineWidth = 2
                cv2.rectangle(im, (int(points[i,0]-boxWidth), int(points[i, 1]-boxWidth)), (int(points[i,0]+boxWidth), int(points[i, 1]+boxWidth)), color, lineWidth)
            else:
                assert False, 'this shape is not supported'
        return im

'''
import cv2
from scipy import ndimage
from compareFaceAlignModel import Dataset, DlibShapePredictor, mtcnn, OpenPoseKeyPoints, dan
#ds= Dataset('/home/macul/Projects/300W/labeled_data/helen/testset')
ds= Dataset('/home/macul/Projects/300W/labeled_data/lfpw/testset',openpose_bbox=True)

danPred = dan(ds,'/home/macul/Projects/DAN_landmark/models','dan','dan')
outputDic = danPred.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic)

rst=ds.getBoundingBoxOpenpose('296814969_3.jpg',[296.48719152068236, 289.5227870143133, 568.0989672690732, 554.1701582563351])
'''

# create dataset with label points, bounding box info
class Dataset_orig:
    def __init__(self, path, bound_box_file='bounding_boxes.mat', img_ext=['jpg','png'], lbl_ext='pts'):

        self.path = path # put img, face points and bounding boxes file (if any) in the same folder
        self.boundBoxFile = bound_box_file
        self.imgExt = img_ext
        self.lblExt = lbl_ext
        self.dataDic = {}
        self.preprocessed = 0
        #self.fivePointsIdx = [range(37-1,43-1), range(43-1,49-1), range(32-1,37-1), [49-1], [55-1]]
        self.fivePointsIdx = [range(37-1,43-1), range(43-1,49-1), [31-1], [49-1], [55-1]]

        # load bounding box infoc
        if type(self.boundBoxFile) != type(None):
            boundingBoxDic = self.getBoundingBox()

        # get img file names 
        for (_, _, imgList) in walk(self.path):
            imgList = [imgList[i] 
                            for i in range(len(imgList)) if imgList[i].split('.')[1] in self.imgExt]
            break # break the first time it yields to get the filename for the top directory only

        # update self.dataDic        
        for i in range(len(imgList)):
            tmpDic = {}
            tmpDic['lblPoints'], tmpDic['lbl5Points'] = self.getLabelPoints(imgList[i])

            # only processing 68 points data
            if tmpDic['lblPoints'].shape[0] != 68:
                continue

            if self.boundBoxFile:
                tmpDic['boundingBox'] = boundingBoxDic[imgList[i]]

                if  (tmpDic['lbl5Points'][:,0] < tmpDic['boundingBox'][0][0]).any() or \
                    (tmpDic['lbl5Points'][:,0] > tmpDic['boundingBox'][0][2]).any() or \
                    (tmpDic['lbl5Points'][:,1] < tmpDic['boundingBox'][0][1]).any() or \
                    (tmpDic['lbl5Points'][:,1] > tmpDic['boundingBox'][0][3]).any():

                    #print tmpDic['lblPoints']
                    #print tmpDic['boundingBox'][0]
                    continue
            else:
                tmpDic['boundingBox'] = self.estimateBoundingBox(tmpDic['lbl5Points'])
        
            self.dataDic[imgList[i]] = tmpDic
               
    def getLabelPoints(self, imgFileName):
        fileName = imgFileName.split('.')[0] + '.' + self.lblExt
        with open(join(self.path, fileName), 'r') as f:
            lblLines = f.readlines()

        if '}' in lblLines:
            lblLines = lblLines[lblLines.index('{\n')+1:lblLines.index('}')]
        else:
            lblLines = lblLines[lblLines.index('{\n')+1:lblLines.index('}\n')]
        lblPoints = np.zeros((len(lblLines), len(lblLines[0].split())))
        lbl5Points = np.zeros((5, len(lblLines[0].split())))

        # only process 68 points data
        if len(lblLines) != 68:
            return lblPoints, lbl5Points

        for i in range(len(lblLines)):
            #print(lblLines[i])
            for j, s in enumerate(lblLines[i].split()):
                #print(s)
                lblPoints[i,j] = float(s)

        for i in range(5):
            lbl5Points[i,:] = np.mean(lblPoints[self.fivePointsIdx[i],:], axis=0)

        return lblPoints, lbl5Points

    def getBoundingBox(self):
        boundingBox = loadmat(join(self.path, self.boundBoxFile))
        boundingBox = boundingBox['bounding_boxes']
        boundingBox = boundingBox[0,:]

        boundingBox = [boundingBox[i][0][0] for i in range(len(boundingBox))]

        boundingBoxDic = {}
        for i in range(len(boundingBox)):
            boundingBoxDic[boundingBox[i][0][0]] = np.array([boundingBox[i][1][0], boundingBox[i][2][0]])


        return boundingBoxDic

    def estimateBoundingBox(self, points):
        min_x = min(points[:,0])
        max_x = max(points[:,0])
        min_y = min(points[:,1])
        max_y = max(points[:,1])

        x1 = min_x - 0.5 * (max_x - min_x)
        x2 = max_x + 0.5 * (max_x - min_x)
        y1 = min_y - 0.9 * (max_y - min_y)
        y2 = max_y + 0.9 * (max_y - min_y)

        x1 = max(x1, 0)
        y1 = max(y1, 0)

        return np.array([[x1, y1, x2, y2],[x1, y1, x2, y2]])

    def computeError(self, predictDic, num_points=68):
        keys = self.dataDic.keys()
        keys.sort()
        errorDic = {}

        for k in keys:
            if predictDic[k] == []:
                errorDic[k] = 0
                continue

            interocular_distance = np.linalg.norm(self.dataDic[k]['lbl5Points'][0] - self.dataDic[k]['lbl5Points'][1])

            if num_points == 68:
                errorDic[k] = np.mean(np.linalg.norm(self.dataDic[k]['lblPoints'] - predictDic[k],axis=1))/interocular_distance
            elif num_points == 5:
                errorDic[k] = np.mean(np.linalg.norm(self.dataDic[k]['lbl5Points'] - predictDic[k],axis=1))/interocular_distance
            elif num_points == 3:
                errorDic[k] = np.mean(np.linalg.norm(self.dataDic[k]['lbl5Points'][0:num_points,:] - predictDic[k][0:num_points,:],axis=1))/interocular_distance
            else:
                assert False, 'Error, only 68, 5 or 3 points are supported!'

        return errorDic

    def showPoints(self, num_points=68, predicted=None):
        keys = self.dataDic.keys()
        keys.sort()

        for k in keys:
            if type(predicted) != type(None) and predicted[k]==[]:
                continue

            interocular_distance = np.linalg.norm(self.dataDic[k]['lbl5Points'][0] - self.dataDic[k]['lbl5Points'][1])

            if self.preprocessed:
                img = self.dataDic[k]['image']
            else:
                img = cv2.imread(join(self.path, k))

            if type(self.dataDic[k]['boundingBox']) == type(None):
                height, width = img.shape[:2]
                bBox = (0, 0, width-1, height-1)
            else:
                bBox = np.array(self.dataDic[k]['boundingBox'][0]).astype(long)
            
            img = self.drawBoxes(img, bBox)

            if num_points==68:
                img = self.drawPoints(img, self.dataDic[k]['lblPoints'], interocular_distance=interocular_distance)
            elif num_points==5:
                img = self.drawPoints(img, self.dataDic[k]['lbl5Points'], interocular_distance=interocular_distance)
            elif num_points==3:
                img = self.drawPoints(img, self.dataDic[k]['lbl5Points'][0:num_points,:], interocular_distance=interocular_distance)
            else:
                assert False, 'only 68 ,5 or 3 points are supported'

            if type(predicted) != type(None):
                img = self.drawPoints(img, predicted[k], shape='rect', color=(0,0,255), interocular_distance=interocular_distance)

            img = img[bBox[1]:bBox[3],bBox[0]:bBox[2]]
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            #cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            print k
            cv2.imshow('img', img)
            kVal=cv2.waitKey()
            if kVal == 93:
                cv2.destroyAllWindows()
                break
            else:
                continue

    def drawBoxes(self, im, boxes):
        x1 = boxes[0]
        y1 = boxes[1]
        x2 = boxes[2]
        y2 = boxes[3]

        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
        return im

    def drawPoints(self, im, points, shape='circle',color=(0,255,0), interocular_distance=None):
        for i in range(points.shape[0]):
            if shape == 'circle':
                if interocular_distance:
                    lineWidth = int(interocular_distance/25)
                    radius = int(lineWidth * 1.5)
                else:
                    radius = 3
                    lineWidth = 2
                cv2.circle(im, (int(points[i,0]), int(points[i, 1])), radius, color, lineWidth)
            elif shape == 'rect':
                if interocular_distance:
                    lineWidth = int(interocular_distance/25)
                    boxWidth = int(lineWidth * 3)
                else:
                    boxWidth = 6
                    lineWidth = 2
                cv2.rectangle(im, (int(points[i,0]-boxWidth), int(points[i, 1]-boxWidth)), (int(points[i,0]+boxWidth), int(points[i, 1]+boxWidth)), color, lineWidth)
            else:
                assert False, 'this shape is not supported'
        return im

# CelebA dataset
class DatasetCelebA(Dataset):
    def __init__(self, path, lbl_file='list_landmarks_celeba.txt', 
                 bound_box_file='list_bbox_celeba.txt', partition_file='list_eval_partition.txt', 
                 img_ext=['jpg','png'], openpose_bbox=False, op_model_path='/home/macul/libraries/openpose/models/', op_fh=48):

        self.path = path.split()[0]
        self.imgFolder = path.split()[1]
        self.partition = path.split()[2] # 0-train, 1-val, 2-test
        self.lblFile = lbl_file
        self.boundBoxFile = bound_box_file
        self.partitionFile = partition_file
        self.imgExt = img_ext        
        self.dataDic = {}
        self.preprocessed = 0

        self.fivePointsIdx = [range(37-1,43-1), range(43-1,49-1), [31-1], [49-1], [55-1]] # used by dlib shapepredictor
        self.openpose_bbox = openpose_bbox
        if self.openpose_bbox:
            self.op_kp = OpenPoseKeyPoints(None, op_model_path, op_fh)


        # load partition info
        with open(join(self.path, self.partitionFile), 'r') as f:
            partitionLines = f.readlines()

        self.partitionDic={}

        for pline in partitionLines:
            words = pline.split()
            self.partitionDic[words[0].split('.')[0]] = words[1]            

        # load bounding box infoc
        if self.boundBoxFile:
            boundingBoxDic, boundingBoxDicOpenpose = self.getBoundingBox()

        lbl5PointsDic = self.getLabelPoints()

        # update self.dataDic   
        for (_, _, imgList) in walk(join(self.path, self.imgFolder)):
            imgList = [imgList[i] 
                            for i in range(len(imgList)) if imgList[i].split('.')[1] in self.imgExt]
            break # break the first time it yields to get the filename for the top directory only     
        for i, k in enumerate(imgList):
            #if i>10000:
            #    break

            if self.partitionDic[k.split('.')[0]] == self.partition:
                tmpDic = {}
                
                tmpDic['lbl5Points'] = lbl5PointsDic[k.split('.')[0]]

                if self.boundBoxFile:
                    if self.openpose_bbox:
                        if k.split('.')[0] not in boundingBoxDicOpenpose.keys():
                            continue
                        tmpDic['boundingBox'] = boundingBoxDicOpenpose[k.split('.')[0]]
                    else:
                        if k.split('.')[0] not in boundingBoxDic.keys():
                            continue
                        tmpDic['boundingBox'] = boundingBoxDic[k.split('.')[0]]
                    if (tmpDic['lbl5Points'][:,0] < tmpDic['boundingBox'][0][0]).any() or \
                        (tmpDic['lbl5Points'][:,0] > tmpDic['boundingBox'][0][2]).any() or \
                        (tmpDic['lbl5Points'][:,1] < tmpDic['boundingBox'][0][1]).any() or \
                        (tmpDic['lbl5Points'][:,1] > tmpDic['boundingBox'][0][3]).any():
                        continue
                else:
                    if self.openpose_bbox:
                        tmpBbox = self.getBoundingBoxOpenpose(k.split('.')[0])
                        if type(tmpBbox)!=type(None):
                            tmpDic['boundingBox'] = np.array([tmpBbox,tmpBbox])
                        else:
                            tmpDic['boundingBox'] = self.estimateBoundingBox(tmpDic['lbl5Points'])
                    else:
                        tmpDic['boundingBox'] = self.estimateBoundingBox(tmpDic['lbl5Points'])
            
                self.dataDic[self.imgFolder+"/"+k] = tmpDic
               
    def getLabelPoints(self):
        with open(join(self.path, self.lblFile), 'r') as f:
            labelLines = f.readlines()

        lbl5Points = {}
        for i in range(2,len(labelLines)):
            words = labelLines[i].split()

            if self.partitionDic[words[0].split('.')[0]] == self.partition:
                leftEye = [float(words[1]), float(words[2])]
                rightEye = [float(words[3]), float(words[4])]
                nose = [float(words[5]), float(words[6])]
                leftMouth = [float(words[7]), float(words[8])]
                rightMouth = [float(words[9]), float(words[10])]

                lbl5Points[words[0].split('.')[0]] = np.array([leftEye, rightEye, nose, leftMouth, rightMouth])

        return lbl5Points

    def getBoundingBox(self):
        with open(join(self.path, self.boundBoxFile), 'r') as f:
            boundBoxLines = f.readlines()

        boundingBoxDic = {}
        boundingBoxDicOpenpose = {}
        for i in range(2,len(boundBoxLines)):
            words = boundBoxLines[i].split()
            if self.partitionDic[words[0].split('.')[0]] == self.partition:
                x1 = float(words[1])
                y1 = float(words[2])
                x2 = x1 + float(words[3]) - 1
                y2 = y1 + float(words[4]) - 1
                boundingBoxDic[words[0].split('.')[0]] = np.array([[x1, y1, x2, y2],[x1, y1, x2, y2]])

                if self.openpose_bbox:
                    tmpBbox = self.getBoundingBoxOpenpose(words[0], [x1, y1, x2, y2])
                    if type(tmpBbox) != type(None):
                        boundingBoxDicOpenpose[words[0].split('.')[0]] = np.array([tmpBbox, tmpBbox])
                    else:
                        print('cannot detect boundingbox by openpose model!!!')
                        boundingBoxDicOpenpose[words[0].split('.')[0]] = np.array([[x1, y1, x2, y2],[x1, y1, x2, y2]])

        return boundingBoxDic, boundingBoxDicOpenpose

    def getBoundingBoxOpenpose(self, image_name, bbox_groundtruth=None):
        print(image_name)
        img = cv2.imread(join(self.path, self.imgFolder, image_name))  

        tmpBboxes = self.op_kp.detectBoundingBox(img, bbox_groundtruth)
        tmpBbox = None
        for bbx in tmpBboxes:
            print('bbx: ', bbx)
            if type(bbox_groundtruth) == type(None):
                bbox_groundtruth = [0,0,img.shape[1],img.shape[0]]
            if self.op_kp.isInsideBoundingBox([np.mean([bbx[0],bbx[2]]),np.mean([bbx[1],bbx[3]])], bbox_groundtruth):
                tmpBbox = bbx
                break
        return tmpBbox


# CelebA dataset
class DatasetCelebA_orig(Dataset):
    def __init__(self, path, lbl_file='list_landmarks_celeba.txt', 
                 bound_box_file='list_bbox_celeba.txt', partition_file='list_eval_partition.txt', 
                 img_ext=['jpg','png']):

        self.path = path.split()[0]
        self.imgFolder = path.split()[1]
        self.partition = path.split()[2] # 0-train, 1-val, 2-test
        self.lblFile = lbl_file
        self.boundBoxFile = bound_box_file
        self.partitionFile = partition_file
        self.imgExt = img_ext        
        self.dataDic = {}
        self.preprocessed = 0

        self.fivePointsIdx = [range(37-1,43-1), range(43-1,49-1), [31-1], [49-1], [55-1]] # used by dlib shapepredictor



        # load partition info
        with open(join(self.path, self.partitionFile), 'r') as f:
            partitionLines = f.readlines()

        self.partitionDic={}

        for pline in partitionLines:
            words = pline.split()
            self.partitionDic[words[0].split('.')[0]] = words[1]            

        # load bounding box infoc
        if self.boundBoxFile:
            boundingBoxDic = self.getBoundingBox()

        lbl5PointsDic = self.getLabelPoints()

        # update self.dataDic   
        for (_, _, imgList) in walk(join(self.path, self.imgFolder)):
            imgList = [imgList[i] 
                            for i in range(len(imgList)) if imgList[i].split('.')[1] in self.imgExt]
            break # break the first time it yields to get the filename for the top directory only     
        for i, k in enumerate(imgList):
            #if i>10000:
            #    break

            if self.partitionDic[k.split('.')[0]] == self.partition:
                tmpDic = {}
                
                tmpDic['lbl5Points'] = lbl5PointsDic[k.split('.')[0]]

                if self.boundBoxFile:
                    if k.split('.')[0] not in boundingBoxDic.keys():
                        continue

                    tmpDic['boundingBox'] = boundingBoxDic[k.split('.')[0]]
                    if (tmpDic['lbl5Points'][:,0] < tmpDic['boundingBox'][0][0]).any() or \
                        (tmpDic['lbl5Points'][:,0] > tmpDic['boundingBox'][0][2]).any() or \
                        (tmpDic['lbl5Points'][:,1] < tmpDic['boundingBox'][0][1]).any() or \
                        (tmpDic['lbl5Points'][:,1] > tmpDic['boundingBox'][0][3]).any():
                        continue
                else:
                    tmpDic['boundingBox'] = self.estimateBoundingBox(tmpDic['lbl5Points'])
            
                self.dataDic[self.imgFolder+"/"+k] = tmpDic
               
    def getLabelPoints(self):
        with open(join(self.path, self.lblFile), 'r') as f:
            labelLines = f.readlines()

        lbl5Points = {}
        for i in range(2,len(labelLines)):
            words = labelLines[i].split()

            if self.partitionDic[words[0].split('.')[0]] == self.partition:
                leftEye = [float(words[1]), float(words[2])]
                rightEye = [float(words[3]), float(words[4])]
                nose = [float(words[5]), float(words[6])]
                leftMouth = [float(words[7]), float(words[8])]
                rightMouth = [float(words[9]), float(words[10])]

                lbl5Points[words[0].split('.')[0]] = np.array([leftEye, rightEye, nose, leftMouth, rightMouth])

        return lbl5Points

    def getBoundingBox(self):
        with open(join(self.path, self.boundBoxFile), 'r') as f:
            boundBoxLines = f.readlines()

        boundingBoxDic = {}
        for i in range(2,len(boundBoxLines)):
            words = boundBoxLines[i].split()
            if self.partitionDic[words[0].split('.')[0]] == self.partition:
                x1 = float(words[1])
                y1 = float(words[2])
                x2 = x1 + float(words[3]) - 1
                y2 = y1 + float(words[4]) - 1
                boundingBoxDic[words[0].split('.')[0]] = np.array([[x1, y1, x2, y2],[x1, y1, x2, y2]])

        return boundingBoxDic

class DatasetProcessed(Dataset):
    def __init__(self, path):

        self.path = path # put img, face points and bounding boxes file (if any) in the same folder
        self.dataDic = {}
        self.preprocessed = 1

        with open(self.path, 'rb') as f:
            result = pickle.load(f)

        imgList = result.keys()

        # update self.dataDic        
        for i in range(len(imgList)):
            tmpDic = {}
            tmpDic['image'] = np.swapaxes(result[imgList[i]]['image'], 0, 2)
            tmpDic['image'] = (tmpDic['image']*128+127.5).astype(np.uint8)
            tmpDic['lblPoints'] = None
            tmpDic['lbl5Points'] = result[imgList[i]]['lbl5Points'].reshape((2,-1)).T

            if type(tmpDic['lbl5Points']) != type(None):
                tmpDic['lbl5Points'] *= tmpDic['image'].shape[1]

            tmpDic['boundingBox'] = None
        
            self.dataDic[imgList[i]] = tmpDic


# dlib shape_predictor model
class DlibShapePredictor():
    def __init__(self, dataset, display_img=False, face_landmark_path='/home/macul/Projects/shape_predictor_68_face_landmarks.dat'):
        self.greyscale = False
        self.boundingBoxIdx = 0 # either 0 or 1 since two are given in the dataset

        self.dataset = dataset
        self.predictor = dlib.shape_predictor(face_landmark_path)

        self.displayImg = display_img


    def getModelOutput(self, num_points=68):
        # num_points = 68 or 5

        outputDic = {}
        keys = self.dataset.dataDic.keys()
        keys.sort()

        for k in keys:
            if self.dataset.preprocessed:
                img = self.dataset.dataDic[k]['image']                
            else:
                if self.greyscale:
                    img = cv2.imread(join(self.dataset.path, k), cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(join(self.dataset.path, k))

            if type(self.dataset.dataDic[k]['boundingBox']) == type(None):
                height, width = img.shape[:2]
                det = dlib.rectangle(0, 0, width-1, height-1)
            else:
                det = dlib.rectangle(*np.array(self.dataset.dataDic[k]['boundingBox'][self.boundingBoxIdx]).astype(long))

            try:
                shape = self.predictor(img, det)
            except:
                print k
                assert False
            points = map(lambda i: shape.part(i), range(shape.num_parts))
            points = np.array(map(lambda p: [p.x, p.y], points))

            #print(points.shape, points)

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

            if self.displayImg:
                img = self.drawPoints(img, outputDic[k])
                cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                #cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow('img', img)
                kVal=cv2.waitKey()

                if kVal == 93:
                    cv2.destroyAllWindows()
                    break
                else:
                    continue

        return outputDic

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


# mtcnn model
class mtcnn():
    def __init__(self, dataset, model_path, prototxt_name, caffemodel_name, display_img=False, threshold=[0.6,0.7,0.0001], factor=0.709, minsize=20, mode='gpu'):
        print 'model_path', model_path
        print 'prototxt_name', prototxt_name
        self.greyscale = False
        self.boundingBoxIdx = 0 # either 0 or 1 since two are given in the dataset
        self.nmsTreshold = [0.5, 0.7, 0.7, 0.7]

        self.dataset = dataset
        self.model_path = model_path
        self.prototxt_name = prototxt_name
        self.caffemodel_name = caffemodel_name
        self.threshold = threshold
        self.factor = factor
        self.minsize = minsize
        
        self.displayImg = display_img
        self.displayInfo = False

        if mode == 'cpu':
            caffe.set_mode_cpu()
        elif mode == 'gpu':
            caffe.set_device(0)
            caffe.set_mode_gpu()
        else:
            assert False, 'error: please specify mode'

        #self.PNet = caffe.Net(join(model_path, "det1.prototxt"), 
        #                        join(model_path, "det1.caffemodel"), caffe.TEST)
        #self.RNet = caffe.Net(join(model_path, "det2.prototxt"), 
        #                        join(model_path, "det2.caffemodel"), caffe.TEST)
        self.ONet = caffe.Net(join(model_path, prototxt_name+".prototxt"), 
                                join(model_path, caffemodel_name+".caffemodel"), caffe.TEST)

    def getModelOutput(self, num_points=5):
        assert num_points in [3,5], 'only 3 or 5 points are supported'
        outputDic = {}
        keys = self.dataset.dataDic.keys()
        keys.sort()

        for k in keys:
            if self.dataset.preprocessed:
                img = self.dataset.dataDic[k]['image']
            else:
                if self.greyscale:
                    img = cv2.imread(join(self.dataset.path, k), cv2.IMREAD_GRAYSCALE)
                    img = img[:,:,np.newaxis]
                else:
                    img = cv2.imread(join(self.dataset.path, k))
            
            #if self.dataset.dataDic[k]['boundingBox'].any() != None:
            #    leftP,topP,rightP,bottomP = np.array(self.dataset.dataDic[k]['boundingBox'][self.boundingBoxIdx]).astype(long)
            ##   convert boundingBox to square
            #    img = img[topP:bottomP, leftP:rightP, :]            
                
            #_, points = self.detectFace(img)

            if type(self.dataset.dataDic[k]['boundingBox']) == type(None):
                points = self.detectLandmark(img, self.dataset.dataDic[k]['boundingBox'])
            else:                
                points = self.detectLandmark(img, self.dataset.dataDic[k]['boundingBox'][self.boundingBoxIdx])
                if self.displayImg: img = self.drawBoxes(img, self.dataset.dataDic[k]['boundingBox'][self.boundingBoxIdx])
            try:
                if points.shape[0]>1:
                    print 'more than one face detected: ', points.shape[0]
                    print 'more face detected on: ', k
                    assert False, 'more than one face detected'
                outputDic[k] = np.array(points[0].reshape(2,5).T)[0:num_points,:] # only pick up the first one if multiple detected
            except:
                print 'error found, points is: ', points
                print 'error file name: ', k
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

    def detectFace(self, image):
        img = image.copy()
        tmp = img[:,:,2].copy()
        img[:,:,2] = img[:,:,0]
        img[:,:,0] = tmp

        h, w = img.shape[:2]

        total_boxes = np.zeros((0,9), np.float)
        points = []
        minl = min(h, w)
        img = img.astype(float)
        m = 12.0/self.minsize
        minl = minl*m

        scales = []
        factor_count = 0
        while minl >= 12:
            scales.append(m * pow(self.factor, factor_count))
            minl *= self.factor
            factor_count += 1

        # first stage
        for scale in scales:
            hs = int(np.ceil(h*scale))
            ws = int(np.ceil(w*scale))

            im_data = cv2.resize(img, (ws,hs)) # default is bilinear
            im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]


            im_data = np.swapaxes(im_data, 0, 2)
            im_data = np.array([im_data], dtype = np.float)
            self.PNet.blobs['data'].reshape(1, im_data.shape[1], ws, hs)
            self.PNet.blobs['data'].data[...] = im_data

            out = self.PNet.forward()
        
            boxes = self.generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, self.threshold[0])
            if boxes.shape[0] != 0:
                #print boxes[4:9]
                #print 'im_data', im_data[0:5, 0:5, 0], '\n'
                #print 'prob1', out['prob1'][0,0,0:3,0:3]

                pick = self.nms(boxes, self.nmsTreshold[0], 'Union')

                if len(pick) > 0 :
                    boxes = boxes[pick, :]

            if boxes.shape[0] != 0:
                total_boxes = np.concatenate((total_boxes, boxes), axis=0)

        if self.displayInfo: print "[1]:",total_boxes.shape[0]

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # nms
            pick = self.nms(total_boxes, self.nmsTreshold[1], 'Union')
            total_boxes = total_boxes[pick, :]
            if self.displayInfo: print "[2]:",total_boxes.shape[0]
            
            # revise and convert to square
            regh = total_boxes[:,3] - total_boxes[:,1]
            regw = total_boxes[:,2] - total_boxes[:,0]
            t1 = total_boxes[:,0] + total_boxes[:,5]*regw
            t2 = total_boxes[:,1] + total_boxes[:,6]*regh
            t3 = total_boxes[:,2] + total_boxes[:,7]*regw
            t4 = total_boxes[:,3] + total_boxes[:,8]*regh
            t5 = total_boxes[:,4]
            total_boxes = np.array([t1,t2,t3,t4,t5]).T

            total_boxes = self.rerec(total_boxes) # convert box to square
            if self.displayInfo: print "[4]:",total_boxes.shape[0]
            
            total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
            if self.displayInfo: print "[4.5]:",total_boxes.shape[0]
            #print total_boxes
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, w, h)


        numbox = total_boxes.shape[0]
        if numbox > 0:
            # second stage

            #print 'tmph', tmph
            #print 'tmpw', tmpw
            #print "y,ey,x,ex", y, ey, x, ex, 
            #print "edy", edy

            #tempimg = np.load('tempimg.npy')

            # construct input for RNet
            tempimg = np.zeros((numbox, 24, 24, im_data.shape[1])) # (24, 24, 3, numbox)
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,im_data.shape[1]))
              
                #print "dx[k], edx[k]:", dx[k], edx[k]
                #print "dy[k], edy[k]:", dy[k], edy[k]
                #print "img.shape", img[y[k]:ey[k]+1, x[k]:ex[k]+1].shape
                #print "tmp.shape", tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1].shape

                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                #print "y,ey,x,ex", y[k], ey[k], x[k], ex[k]
                #print "tmp", tmp.shape
                
                tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))
                #tempimg[k,:,:,:] = imResample(tmp, 24, 24)
                #print 'tempimg', tempimg[k,:,:,:].shape
                #print tempimg[k,0:5,0:5,0] 
                #print tempimg[k,0:5,0:5,1] 
                #print tempimg[k,0:5,0:5,2] 
                #print k
        
            #print tempimg.shape
            #print tempimg[0,0,0,:]
            tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python

            #np.save('tempimg.npy', tempimg)

            # RNet

            tempimg = np.swapaxes(tempimg, 1, 3)
            #print tempimg[0,:,0,0]
            
            self.RNet.blobs['data'].reshape(numbox, im_data.shape[1], 24, 24)
            self.RNet.blobs['data'].data[...] = tempimg
            out = self.RNet.forward()

            #print out['conv5-2'].shape
            #print out['prob1'].shape

            score = out['prob1'][:,1]
            #print 'score', score
            pass_t = np.where(score>self.threshold[1])[0]
            #print 'pass_t', pass_t
            
            score =  np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
            if self.displayInfo: print "[5]:",total_boxes.shape[0]
            #print total_boxes

            #print "1.5:",total_boxes.shape
            
            mv = out['conv5-2'][pass_t, :].T
            #print "mv", mv
            if total_boxes.shape[0] > 0:
                pick = self.nms(total_boxes, self.nmsTreshold[2], 'Union')
                #print 'pick', pick
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    if self.displayInfo: print "[6]:",total_boxes.shape[0]
                    total_boxes = self.bbreg(total_boxes, mv[:, pick])
                    if self.displayInfo: print "[7]:",total_boxes.shape[0]
                    total_boxes = self.rerec(total_boxes)
                    if self.displayInfo: print "[8]:",total_boxes.shape[0]
                
            #####
            # 2 #
            #####
            if self.displayInfo: print "2:",total_boxes.shape
            if self.displayInfo: print "2 boxes data:", total_boxes

            numbox = total_boxes.shape[0]
            if numbox > 0:
                # third stage
                
                total_boxes = np.fix(total_boxes)
                [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, w, h)
               
                #print 'tmpw', tmpw
                #print 'tmph', tmph
                #print 'y ', y
                #print 'ey', ey
                #print 'x ', x
                #print 'ex', ex
            

                tempimg = np.zeros((numbox, 48, 48, im_data.shape[1]))
                for k in range(numbox):
                    tmp = np.zeros((int(tmph[k]), int(tmpw[k]),im_data.shape[1]))
                    tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                    tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
                tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                    
                # ONet
                tempimg = np.swapaxes(tempimg, 1, 3)
                self.ONet.blobs['data'].reshape(numbox, im_data.shape[1], 48, 48)
                self.ONet.blobs['data'].data[...] = tempimg
                out = self.ONet.forward()
                
                score = out['prob1'][:,1]
                points = out['conv6-3']
                pass_t = np.where(score>self.threshold[2])[0]
                points = points[pass_t, :]
                score = np.array([score[pass_t]]).T
                total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
                if self.displayInfo: print "[9]:",total_boxes.shape
                if self.displayInfo: print "[9] boxes data:", total_boxes
                
                mv = out['conv6-2'][pass_t, :].T
                w = total_boxes[:,3] - total_boxes[:,1] + 1
                h = total_boxes[:,2] - total_boxes[:,0] + 1

                points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
                points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

                if total_boxes.shape[0] > 0:
                    if self.displayInfo: print "mv[]:",mv
                    total_boxes = self.bbreg(total_boxes, mv[:,:])
                    if self.displayInfo: print "[10]:",total_boxes.shape
                    if self.displayInfo: print "[10] boxes data:", total_boxes
                    pick = self.nms(total_boxes, self.nmsTreshold[3], 'Min')
                    
                    #print pick
                    if len(pick) > 0 :
                        total_boxes = total_boxes[pick, :]
                        if self.displayInfo: print "[11]:",total_boxes.shape
                        points = points[pick, :]

        #####
        # 3 #
        #####
        if self.displayInfo: print "3:",total_boxes.shape
        if self.displayInfo: print "3 boxes data:", total_boxes

        return total_boxes, points


    def detectLandmark(self, image, boundingBox):
        # change from RGB to BGR
        img = image.copy()        
        tmp = img[:,:,2].copy()
        img[:,:,2] = img[:,:,0]
        img[:,:,0] = tmp

        h, w = img.shape[:2]

        if type(boundingBox) != type(None):
            total_boxes = np.array(boundingBox).astype(long)
        else:
            total_boxes = np.array([0, 0, w-1, h-1])

        # print('total_boxes [0]: ', total_boxes)
        total_boxes = total_boxes[np.newaxis,:]
        total_boxes = self.rerec(total_boxes) # convert box to square
        total_boxes = np.fix(total_boxes)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, w, h)

        points = []        

        tempimg = np.zeros((1, 48, 48, img.shape[2]))

        k = 0
        tmp = np.zeros((int(tmph[k]), int(tmpw[k]),img.shape[2]))
        tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
        tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))

        tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
            
        # ONet
        tempimg = np.swapaxes(tempimg, 1, 3)
        self.ONet.blobs['data'].reshape(1, img.shape[2], 48, 48)
        self.ONet.blobs['data'].data[...] = tempimg
        out = self.ONet.forward()
        
        # mymtcnn net has no prob1 output, set score to 1
        points = out['conv6-3']
        try:
            score = out['prob1'][:,1]
        except:
            score = np.array([1.0])
        #print('score: ', score)
        #print('points [1]: ',points)
        pass_t = np.where(score>self.threshold[2])[0]
        points = points[pass_t, :]
        score = np.array([score[pass_t]]).T

        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
        if self.displayInfo: print "[9]:",total_boxes.shape
        if self.displayInfo: print "[9] boxes data:", total_boxes
        
        #mv = out['conv6-2'][pass_t, :].T
        w = total_boxes[:,3] - total_boxes[:,1] + 1
        h = total_boxes[:,2] - total_boxes[:,0] + 1

        points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 0.5
        points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T - 0.5

        #print('points [2]: ', points)
        return points


    def generateBoundingBox(self, map, reg, scale, t):
        stride = 2
        cellsize = 12
        map = map.T
        dx1 = reg[0,:,:].T
        dy1 = reg[1,:,:].T
        dx2 = reg[2,:,:].T
        dy2 = reg[3,:,:].T
        (x, y) = np.where(map >= t)

        yy = y
        xx = x
        
        score = map[x,y]
        reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])

        if reg.shape[0] == 0:
            pass
        boundingbox = np.array([yy, xx]).T

        bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
        bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
        score = np.array([score])

        boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

        #print '(x,y)',x,y
        #print 'score', score
        #print 'reg', reg

        return boundingbox_out.T

    def nms(self, boxes, threshold, type):
        """nms
        :boxes: [:,0:5]
        :threshold: 0.5 like
        :type: 'Min' or others
        :returns: TODO
        """
        if boxes.shape[0] == 0:
            return np.array([])
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        s = boxes[:,4]
        area = np.multiply(x2-x1+1, y2-y1+1)
        I = np.array(s.argsort()) # read s using I
        
        pick = [];
        while len(I) > 0:
            xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
            yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
            xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
            yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if type == 'Min':
                o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
            else:
                o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
            pick.append(I[-1])
            I = I[np.where( o <= threshold)[0]]
        return pick

    def pad(self, boxesA, w, h):
        boxes = boxesA.copy() # shit, value parameter!!!
        #print '#################'
        #print 'boxes', boxes
        #print 'w,h', w, h
        
        tmph = boxes[:,3] - boxes[:,1] + 1
        tmpw = boxes[:,2] - boxes[:,0] + 1
        numbox = boxes.shape[0]

        #print 'tmph', tmph
        #print 'tmpw', tmpw

        dx = np.ones(numbox)
        dy = np.ones(numbox)
        edx = tmpw 
        edy = tmph

        x = boxes[:,0:1][:,0]
        y = boxes[:,1:2][:,0]
        ex = boxes[:,2:3][:,0]
        ey = boxes[:,3:4][:,0]
       
       
        tmp = np.where(ex > w)[0]
        if tmp.shape[0] != 0:
            edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
            ex[tmp] = w-1

        tmp = np.where(ey > h)[0]
        if tmp.shape[0] != 0:
            edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
            ey[tmp] = h-1

        tmp = np.where(x < 1)[0]
        if tmp.shape[0] != 0:
            dx[tmp] = 2 - x[tmp]
            x[tmp] = np.ones_like(x[tmp])

        tmp = np.where(y < 1)[0]
        if tmp.shape[0] != 0:
            dy[tmp] = 2 - y[tmp]
            y[tmp] = np.ones_like(y[tmp])
        
        # for python index from 0, while matlab from 1
        dy = np.maximum(0, dy-1)
        dx = np.maximum(0, dx-1)
        y = np.maximum(0, y-1)
        x = np.maximum(0, x-1)
        edy = np.maximum(0, edy-1)
        edx = np.maximum(0, edx-1)
        ey = np.maximum(0, ey-1)
        ex = np.maximum(0, ex-1)

        #print 'boxes', boxes
        return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]

    def rerec(self, bboxA):
        # convert bboxA to square
        w = bboxA[:,2] - bboxA[:,0]
        h = bboxA[:,3] - bboxA[:,1]
        l = np.maximum(w,h).T
        
        #print 'bboxA', bboxA
        #print 'w', w
        #print 'h', h
        #print 'l', l
        bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
        bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
        bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
        return bboxA


    def bbreg(self, boundingbox, reg):
        reg = reg.T 
        
        # calibrate bouding boxes
        if reg.shape[1] == 1:
            print "reshape of reg"
            pass # reshape of reg
        w = boundingbox[:,2] - boundingbox[:,0] + 1
        h = boundingbox[:,3] - boundingbox[:,1] + 1

        bb0 = boundingbox[:,0] + reg[:,0]*w
        bb1 = boundingbox[:,1] + reg[:,1]*h
        bb2 = boundingbox[:,2] + reg[:,2]*w
        bb3 = boundingbox[:,3] + reg[:,3]*h
        
        boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
        #print "bb", boundingbox
        return boundingbox

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


# dan model
class dan():
    def __init__(self, dataset, model_path, prototxt_name, caffemodel_name, display_img=False, threshold=0.7, mode='gpu'):
        print 'model_path', model_path
        print 'prototxt_name', prototxt_name

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
                print 'error found, points is: ', points
                print 'error file name: ', k
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


    def detectLandmark(self, image, boundingBox):
        # change from RGB to BGR
        img = image.copy()        
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


# dan model
class dan_test():
    def __init__(self, dataset, model_path, prototxt_name, caffemodel_name, display_img=False, threshold=0.7, mode='gpu', openpose_bbox=False, op_model_path='/home/macul/libraries/openpose/models/', op_fh=16):
        print 'model_path', model_path
        print 'prototxt_name', prototxt_name
        self.openpose_bbox = openpose_bbox
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
        if self.openpose_bbox:
            self.op_kp = OpenPoseKeyPoints(self.dataset, op_model_path, op_fh)

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
                if self.openpose_bbox:
                    tmpBboxes = self.op_kp.detectBoundingBox(img_orig, None)
                    if len(tmpBboxes):
                        points, score = self.detectLandmark(img, tmpBboxes[0])
                    else:
                        points, score = self.detectLandmark(img, None)
                else:
                    points, score = self.detectLandmark(img, None)
            else:                
                if self.openpose_bbox:
                    tmpBboxes = self.op_kp.detectBoundingBox(img_orig, self.dataset.dataDic[k]['boundingBox'][self.boundingBoxIdx])
                    tmpBbox = []
                    for bbx in tmpBboxes:
                        print('bbx: ', bbx)
                        if self.op_kp.isInsideBoundingBox([np.mean([bbx[0],bbx[2]]),np.mean([bbx[1],bbx[3]])], self.dataset.dataDic[k]['boundingBox'][self.boundingBoxIdx]):
                            tmpBbox = bbx
                            break
                    if len(tmpBbox):
                        points, score = self.detectLandmark(img, tmpBbox)
                        if self.displayImg: img = self.drawBoxes(img, tmpBbox)
                    else:
                        print 'error to found bbox by openpose!!'
                        print 'error file name: ', k
                        outputDic[k] = []
                        continue
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
                print 'error found, points is: ', points
                print 'error file name: ', k
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


    def detectLandmark(self, image, boundingBox):
        # change from RGB to BGR
        img = image.copy()        
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
'''
import cv2
from scipy import ndimage
from compareFaceAlignModel import Dataset, DlibShapePredictor, mtcnn, OpenPoseKeyPoints, dan
ds= Dataset('/home/macul/Projects/300W/labeled_data/helen/testset')
danPred = dan(ds,'/home/macul/Projects/DAN_landmark/models','dan','dan')
outputDic = danPred.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic)


import cv2
from scipy import ndimage
from compareFaceAlignModel import Dataset, DlibShapePredictor, mtcnn, OpenPoseKeyPoints, dan
ds= Dataset('/home/macul/Projects/300W/labeled_data/helen/testset')
danPred1 = dan(ds,'/home/macul/Projects/DAN_landmark/models','dan','dan',openpose_bbox=True)
outputDic1 = danPred1.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic1)



shapePred = DlibShapePredictor(ds)
outputDic = shapePred.getModelOutput()
'''

'''
import cv2
import numpy as np
from scipy import ndimage
import utils

color_img = cv2.imread("../test1.jpg")
bbox=[126, 81, 166+126, 172+81]

#color_img = cv2.imread("../test1.jpg")[81:253,126:292,:]
#bbox=[0, 0, 165, 171]
cv2.rectangle(color_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0))
cv2.imshow("image", color_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()


color_img = cv2.imread("../4560029166_1.jpg")
#bbox1=[249, 143, 386, 299]
bbox=[241, 129, 386, 294]
cv2.rectangle(color_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0))
#cv2.rectangle(color_img, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), (255, 255, 0))
#cv2.imshow("image", color_img)
#key = cv2.waitKey(0)
#cv2.destroyAllWindows()

if len(color_img.shape) > 2:
    gray_img = np.mean(color_img, axis=2).astype(np.uint8)
else:
    gray_img = color_img.astype(np.uint8)


f=np.load('../DAN.npz','rb')
meanInitLandmarks = f['initLandmarks']
#meanImg=f['meanImg']
#stdDevImg=f['stdDevImg']
meanImg=cv2.imread('../mean.jpg')[:,:,0]
meanImg = meanImg[np.newaxis].astype(np.float32)
stdDevImg=cv2.imread('../dev.jpg')[:,:,0]
stdDevImg = stdDevImg[np.newaxis].astype(np.float32)

initLandmarks = utils.bestFitRect(None, meanInitLandmarks, bbox)

A, t = utils.bestFit(meanInitLandmarks, initLandmarks, True)
A2 = np.linalg.inv(A)
t2 = np.dot(-t, A2)

inputImg = np.zeros((112, 112), dtype=np.float32)
inputImg = ndimage.interpolation.affine_transform(gray_img, A2, t2[[1, 0]], output_shape=(112, 112))
cv2.imshow("image", inputImg)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

inputImg = inputImg[np.newaxis]
inputImg = inputImg - meanImg
inputImg = inputImg / stdDevImg
inputImg = inputImg[np.newaxis]
#inputImg = np.swapaxes(inputImg, 2, 3)

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
ONet = caffe.Net('/home/macul/Projects/DAN_landmark/models/dan.prototxt', 
                    '/home/macul/Projects/DAN_landmark/models/dan.caffemodel', caffe.TEST)
ONet.blobs['data'].reshape(1, 1, 112, 112)
ONet.blobs['data'].data[...] = inputImg
out = ONet.forward()

# mymtcnn net has no prob1 output, set score to 1
landmarks = out['s1_output'][0]
landmarks = landmarks.reshape((-1, 2))

points=np.dot((landmarks+meanInitLandmarks) - t, np.linalg.inv(A))
score = out['s1_confidence'][0][0]

landmarks, confidence = model.processImg(gray_img[np.newaxis], initLandmarks)

inputImg, transform = self.CropResizeRotate(img, inputLandmarks)
        inputImg = inputImg - self.meanImg
        inputImg = inputImg / self.stdDevImg

        output = self.generate_network_output([inputImg])[0][0]


def CropResizeRotate(self, img, inputShape):
        A, t = utils.bestFit(self.initLandmarks, inputShape, True)
    
        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)

        outImg = np.zeros((self.nChannels, self.imageHeight, self.imageWidth), dtype=np.float32)
        for i in range(img.shape[0]):
            outImg[i] = ndimage.interpolation.affine_transform(img[i], A2, t2[[1, 0]], output_shape=(self.imageHeight, self.imageWidth))

        return outImg, [A, t]   
'''

# openpose model
class OpenPoseKeyPoints():
    def __init__(self, dataset, model_path, face_height, model_name="COCO",display_img=False, mode='gpu'):
        print 'model_path', model_path
        self.greyscale = False
        self.boundingBoxIdx = 0 # either 0 or 1 since two are given in the dataset
        self.nmsTreshold = [0.5, 0.7, 0.7, 0.7]

        self.dataset = dataset
        self.model_path = model_path
        self.model_name = model_name
        
        self.displayImg = display_img
        self.displayInfo = False

        self.nose_idx = 0
        self.neck_idx = 1
        self.r_eye_idx = 14
        self.l_eye_idx = 15
        self.r_ear_idx = 16
        self.l_ear_idx = 17

        #if mode == 'cpu':
        #    caffe.set_mode_cpu()
        #elif mode == 'gpu':
        #    caffe.set_device(0)
        #    caffe.set_mode_gpu()
        #else:
        #    assert False, 'error: please specify mode'

        #self.PNet = caffe.Net(join(model_path, "det1.prototxt"), 
        #                        join(model_path, "det1.caffemodel"), caffe.TEST)
        #self.RNet = caffe.Net(join(model_path, "det2.prototxt"), 
        #                        join(model_path, "det2.caffemodel"), caffe.TEST)
        self.target_face_height = face_height
        self.params = dict()
        self.params["logging_level"] = 3
        self.params["output_resolution"] = "-1x-1"
        self.params["net_resolution"] = "-1x160"
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
        self.openpose = OpenPose(self.params)

    def getModelOutput(self):
        outputDic = {}
        keys = self.dataset.dataDic.keys()
        keys.sort()        

        for k in keys:
            print k
            if self.dataset.preprocessed:
                img = self.dataset.dataDic[k]['image']
            else:
                if self.greyscale:
                    img = cv2.imread(join(self.dataset.path, k), cv2.IMREAD_GRAYSCALE)
                    img = img[:,:,np.newaxis]
                else:
                    img = cv2.imread(join(self.dataset.path, k))                    

            #if self.dataset.dataDic[k]['boundingBox'].any() != None:
            #    leftP,topP,rightP,bottomP = np.array(self.dataset.dataDic[k]['boundingBox'][self.boundingBoxIdx]).astype(long)
            ##   convert boundingBox to square
            #    img = img[topP:bottomP, leftP:rightP, :]            
                
            #_, points = self.detectFace(img)

            if type(self.dataset.dataDic[k]['boundingBox']) == type(None):
                points = self.detectLandmark(img, self.dataset.dataDic[k]['boundingBox'])
            else:                
                points = self.detectLandmark(img, self.dataset.dataDic[k]['boundingBox'][self.boundingBoxIdx])
                if self.displayImg: img = self.drawBoxes(img, self.dataset.dataDic[k]['boundingBox'][self.boundingBoxIdx])
            try:
                #print 'points.shape: ', points.shape
                if points.shape[0]>1:
                    print 'more than one face detected: ', points.shape[0]
                    print 'more face detected on: ', k
                    assert False, 'more than one face detected'
                outputDic[k] = np.array(points[0].reshape(2,3).T) # only pick up the first one if multiple detected
            except:
                print 'error found, points is: ', points
                print 'error file name: ', k
                outputDic[k] = []
                continue
            
            if self.displayImg:
                #img = self.drawPoints(img, outputDic[k])
                cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                #cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow('img', output_image)
                kVal=cv2.waitKey()
                if kVal == 93:
                    cv2.destroyAllWindows()
                    break
                else:
                    continue

            #break

        return outputDic


    def detectLandmark(self, image, boundingBox):
        #print('image shape: ', image.shape)
        net_resolution = "-1x"+str(int(round(1.0*self.target_face_height/(boundingBox[3]-boundingBox[1])*image.shape[0]/16)*16))
        print('net_resolution', net_resolution)
        self.openpose.resizeInput(self.params["output_resolution"], net_resolution, self.params["scale_gap"], self.params["scale_number"])
        keypoints, output_image = self.openpose.forward(image, True)
        #print(keypoints)

        points = [] 

        for i in range(keypoints.shape[0]):
            if self.isInsideBoundingBox(keypoints[i][self.nose_idx][0:2], boundingBox):
                tmp = [keypoints[i][self.r_eye_idx][0], keypoints[i][self.l_eye_idx][0], keypoints[i][self.nose_idx][0], \
                        keypoints[i][self.r_eye_idx][1], keypoints[i][self.l_eye_idx][1], keypoints[i][self.nose_idx][1]]
                points += [tmp]
                break        

        #print('points [2]: ', points)
        return np.array(points)

    def detectBoundingBox(self, image, boundingBox=None):
        if type(boundingBox==type(None)):
            net_resolution = "-1x"+str(int(round(1.0*self.target_face_height/image.shape[0]*image.shape[0]/16)*16))
        else:
            net_resolution = "-1x"+str(int(round(1.0*self.target_face_height/(boundingBox[3]-boundingBox[1])*image.shape[0]/16)*16))
        print('net_resolution', net_resolution)
        self.openpose.resizeInput(self.params["output_resolution"], net_resolution, self.params["scale_gap"], self.params["scale_number"])
        keypoints, output_image = self.openpose.forward(image, True)
        #print(keypoints)

        bboxes = []

        for i in range(keypoints.shape[0]):

            if keypoints[i][self.r_eye_idx][2]<0.5 or keypoints[i][self.l_eye_idx][2]<0.5 or keypoints[i][self.nose_idx][2]<0.5:
                continue

            x_center = np.mean([keypoints[i][self.r_eye_idx][0], keypoints[i][self.l_eye_idx][0], keypoints[i][self.nose_idx][0]])
            y_center = keypoints[i][self.nose_idx][1]
            if keypoints[i][self.neck_idx][2] > 0.8 and keypoints[i][self.neck_idx][1]>keypoints[i][self.nose_idx][1]:
                y_neck = keypoints[i][self.neck_idx][1]
                bbox_half_height = min((y_neck-y_center)*0.6, y_center)
                bbox_half_width = min(bbox_half_height*0.48, x_center, image.shape[1]-x_center)
                #bbox_half_width = min((y_neck-y_center)*0.6, x_center, image.shape[1]-x_center)
                if (bbox_half_height<=0 or bbox_half_width<=0):
                    print('keypoints: ', keypoints)
                assert (bbox_half_height>0 and bbox_half_width>0), 'bbox_half_height and bbox_half_width must be greater than 0!!!'
            else:
                bbox_half_height = (keypoints[i][self.nose_idx][1] - np.mean([keypoints[i][self.r_eye_idx][1], keypoints[i][self.l_eye_idx][1]]))*1.5
                bbox_half_width  = bbox_half_height*0.99
                bbox_half_height = min(bbox_half_height, y_center, image.shape[0]-y_center)
                bbox_half_width  = min(bbox_half_width,  x_center, image.shape[1]-x_center)
            x1 = max(0, x_center - bbox_half_width)
            y1 = max(0, y_center - bbox_half_height)
            x2 = min(image.shape[1], x_center + bbox_half_width)
            y2 = min(image.shape[0], y_center + bbox_half_height)
            bboxes += [ [x1,y1,x2,y2] ]  

            #print(keypoints[i])
            #cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            #cv2.imshow('img', image)
            #kVal=cv2.waitKey()
            #if kVal:
            #    cv2.destroyAllWindows()
        #print('bboxes: ', bboxes)
        return bboxes

    def detectLandmark_test(self, image, boundingBox):
        keypoints, output_image = self.openpose.forward(image[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2], :], True)
        print(keypoints)
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        #cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('img', image[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2], :])
        kVal=cv2.waitKey()
        if kVal==93:
            cv2.destroyAllWindows()
            exit(1)

        points = [] 

        for i in range(keypoints.shape[0]):
            tmp = [keypoints[i][self.r_eye_idx][0]+bbox_x1, keypoints[i][self.l_eye_idx][0]+bbox_x1, keypoints[i][self.nose_idx][0]+bbox_x1, \
                    keypoints[i][self.r_eye_idx][1]+bbox_y1, keypoints[i][self.l_eye_idx][1]+bbox_y1, keypoints[i][self.nose_idx][1]+bbox_y1]
            points += [tmp]
            break        

        #print('points [2]: ', points)
        return np.array(points)

    def isInsideBoundingBox(self, point, boundingbox):
        if (point[0] > boundingbox[0]) and (point[0] < boundingbox[2]) and (point[1] > boundingbox[1]) and (point[1] < boundingbox[3]):
            return True
        else:
            return False

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

'''
import cv2
from compareFaceAlignModel import Dataset, DlibShapePredictor, mtcnn, OpenPoseKeyPoints
ds= Dataset('/home/macul/Projects/300W/labeled_data/ibug')
#ds= Dataset('/home/macul/Projects/300W/labeled_data/helen/testset')
#ds= Dataset('/home/macul/Projects/300W/labeled_data/aflw', bound_box_file=None)
op_kp=OpenPoseKeyPoints(ds, '/home/macul/libraries/openpose/models/', 16)
outputDic = op_kp.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic)

img = cv2.imread("/home/macul/Projects/ego/spoofing_kp/00251.jpg")
keypoints, output_image = op_kp.openpose.forward(img, True)

mtcnnPred = mtcnn(ds,'/home/macul/Projects/300W','48net','48net')
outputDic1 = mtcnnPred.getModelOutput(num_points=3)
ds.showPoints(num_points=3, predicted=outputDic1)
'''
'''
fig = plt.figure()
plt.plot(a,np.cumsum(a))
plt.show()
'''

'''
import compareFaceAlignModel as cfam
ds= cfam.Dataset('/home/lin/Projects/300W/labeled_data/afw')
#ds= cfam.Dataset('/home/lin/Projects/300W/labeled_data/300W_Cropped/01_Indoor', bound_box_file=None)
shapePred=cfam.DlibShapePredictor(ds, face_landmark_path='/home/lin/Projects/shape_predictor_68_face_landmarks.dat')
outputDic=shapePred.getModelOutput(num_points=5)
errorDic=ds.computeError(outputDic)
outputDic[outputDic.keys()[0]].shape

import compareFaceAlignModel as cfam
ds1= cfam.Dataset('/home/lin/Projects/300W/labeled_data/small', bound_box_file=None)
mtcnnPred = cfam.mtcnn(ds1,'/home/lin/Projects/mtcnn/model')
outputDic1 = mtcnnPred.getModelOutput()
outputDic1[outputDic1.keys()[0]].shape

import compareFaceAlignModel as cfam
ds= cfam.Dataset('/home/lin/Projects/300W/labeled_data/afw')
mtcnnPred1 = cfam.mtcnn(ds,'/home/lin/Projects/mtcnn/model')
outputDic2 = mtcnnPred1.getModelOutput()
errorDic2=ds.computeError(outputDic2)
outputDic2[outputDic2.keys()[0]].shape

from compareFaceAlignModel import Dataset, DlibShapePredictor, mtcnn
ds= Dataset('/home/macul/Projects/300W/labeled_data/aflw', bound_box_file=None)
mtcnnPred = mtcnn(ds,'/home/macul/Projects/mtcnn/model')
outputDic1 = mtcnnPred.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic1)
shapePred = DlibShapePredictor(ds)
outputDic = shapePred.getModelOutput(num_points=5)
ds.showPoints(num_points=5, predicted=outputDic)


from compareFaceAlignModel import Dataset, DlibShapePredictor, mtcnn
ds= Dataset('/home/macul/Projects/300W/labeled_data/aflw', bound_box_file=None)
mtcnnPred = mtcnn(ds,'/home/macul/Projects/300W','myMtcnnTest1','_iter_5460000')
outputDic1 = mtcnnPred.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic1)

from compareFaceAlignModel import Dataset, DatasetCelebA, DlibShapePredictor, mtcnn
ds= DatasetCelebA(  '/home/macul/Projects/300W/labeled_data/CelebA_Align img_align_celeba_png 1', lbl_file='list_landmarks_align_celeba.txt', bound_box_file=None)
mtcnnPred = mtcnn(ds,'/home/macul/Projects/300W','myMtcnnTest1','_iter_5460000')
outputDic1 = mtcnnPred.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic1)

from compareFaceAlignModel import Dataset, DatasetCelebA, DlibShapePredictor, mtcnn
ds= DatasetCelebA(  '/home/macul/Projects/300W/labeled_data/CelebA img_celeba 1', lbl_file='list_landmarks_celeba.txt', bound_box_file='list_bbox_celeba.txt')
mtcnnPred = mtcnn(ds,'/home/macul/Projects/300W','myMtcnnTest1','_iter_5460000')
outputDic1 = mtcnnPred.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic1)

from compareFaceAlignModel import Dataset, DlibShapePredictor, mtcnn
ds= Dataset('/home/macul/Projects/300W/labeled_data/afw')
mtcnnPred = mtcnn(ds,'/home/macul/Projects/300W','myMtcnnTest1','_iter_5460000')
outputDic1 = mtcnnPred.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic1)


from compareFaceAlignModel import Dataset, DatasetCelebA, DatasetProcessed, DlibShapePredictor, mtcnn
ds= DatasetProcessed('/home/macul/Projects/300W/valDataset_1_1_aflw.pkl')
mtcnnPred = mtcnn(ds,'/home/macul/Projects/mtcnn/model','det3','det3')
outputDic = mtcnnPred.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic)

ds= DatasetProcessed('/home/macul/Projects/300W/valDataset_1_1.pkl')
myMtcnnPred = mtcnn(ds,'/home/macul/Projects/300W','myMtcnnTest1','_iter_120000')
myOutputDic = myMtcnnPred.getModelOutput()
ds.showPoints(num_points=5, predicted=myOutputDic)

from compareFaceAlignModel import Dataset, DatasetCelebA, DlibShapePredictor, mtcnn
ds= DatasetCelebA(  '/home/macul/Projects/300W/labeled_data/CelebA img_celeba 1', lbl_file='list_landmarks_celeba.txt', bound_box_file='list_bbox_celeba.txt')
mtcnnPred = mtcnn(ds,'/home/macul/Projects/300W','snapshot_2/fc1024_layer_freeze_test','snapshot_2/_iter_120000')
outputDic1 = mtcnnPred.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic1)

from compareFaceAlignModel import Dataset, DlibShapePredictor, mtcnn
ds= Dataset('/home/macul/Projects/300W/labeled_data/helen/testset')
mtcnnPred = mtcnn(ds,'/home/macul/Projects/300W','snapshot_2/fc1024_layer_freeze_test','snapshot_2/_iter_120000')
outputDic1 = mtcnnPred.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic1)


from compareFaceAlignModel import Dataset, DatasetCelebA, DatasetProcessed, DlibShapePredictor, mtcnn
ds= DatasetCelebA('/home/macul/Projects/300W/labeled_data/CelebA img_celeba 2', lbl_file='list_landmarks_celeba.txt', bound_box_file='ZF_bbox_list.txt')
shapePred = DlibShapePredictor(ds, display_img=True)
outputDic = shapePred.getModelOutput(num_points=5)
ds.showPoints(num_points=5, predicted=outputDic)
#ds.showPoints(num_points=5)

ds1= Dataset('/home/macul/Projects/300W/labeled_data/helen/testset', lbl_file='bounding_boxes.mat', bound_box_file='bounding_boxes.mat')
ds1.showPoints(num_points=5)

from compareFaceAlignModel import Dataset, DatasetCelebA, DatasetProcessed, DlibShapePredictor, mtcnn
ds= DatasetCelebA('/home/macul/Projects/300W/labeled_data/CelebA img_celeba 2', lbl_file='list_landmarks_celeba.txt', bound_box_file='ZF_bbox_list.txt')
mtcnnPred = mtcnn(ds,'/home/macul/Projects/300W','48net','48net')
outputDic = mtcnnPred.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic)



from compareFaceAlignModel import Dataset, DlibShapePredictor, mtcnn
ds= Dataset('/home/macul/Projects/300W/labeled_data/afw')
mtcnnPred = mtcnn(ds,'/home/macul/Projects/300W','48net','48net')
outputDic1 = mtcnnPred.getModelOutput()
ds.showPoints(num_points=5, predicted=outputDic1)
'''