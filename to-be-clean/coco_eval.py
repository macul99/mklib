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
from openpose_kp import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tqdm

def round_int(x):
    return int(round(x))

class cocoeval():
    def __init__(   self, 
                    output_file,
                    coco_path = '/media/macul/black/cocodata/dataset/COCO/',
                    op_path = '/media/macul/black/open_pose_train/models/',
                    op_input_height = 240,
                    mode='gpu0' ):
        print 'coco_path', coco_path
        #self.greyscale = False
        #self.boundingBoxIdx = 0 # either 0 or 1 since two are given in the dataset
        #self.nmsTreshold = [0.5, 0.7, 0.7, 0.7]

        self.coco_path = coco_path
        self.image_dir = join(coco_path, 'images/val2014')
        self.anno_dir = join(coco_path, 'annotations/person_keypoints_val2014.json')
        self.cocoGt = COCO(self.anno_dir)
        self.catIds = self.cocoGt.getCatIds(catNms=['person'])
        self.keys = self.cocoGt.getImgIds(catIds=self.catIds)
        self.op_path = op_path
        self.op_input_height = op_input_height
        self.mode = mode
        self.op = openpose_kp(self.op_path, self.op_input_height, 'COCO', self.mode)
        self.output_file = output_file
        self.coco_ids = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]        

    def process(self):
        result = []
        total_image = len(self.keys)
        for i, k in enumerate(self.keys):
            print('image {} out of {}: '.format(i+1, total_image))
            img_meta = self.cocoGt.loadImgs(k)[0]
            img_idx = img_meta['id']
            humans = self.op.detectLandmark(join(self.image_dir, img_meta['file_name']))
            #ann_idx = self.cocoGt.getAnnIds(imgIds=[img_idx], catIds=[1])
            #anns = self.cocoGt.loadAnns(ann_idx)
            for hm in humans:
                kp = []
                score = []
                for ki in self.coco_ids:
                    kp += [round_int(hm[ki,0]),round_int(hm[ki,1]),1]
                    if hm[ki,2]>0:
                        score += [hm[ki,2]]
                item = {
                    'image_id': img_idx,
                    'category_id': 1,
                    'keypoints': kp,
                    'score': np.mean(score).astype(np.float64)
                }
                result.append(item)
        fp = open(self.output_file,'w')
        json.dump(result, fp)
        fp.close()

    def eval(self):
        cocoDt = self.cocoGt.loadRes(self.output_file)
        cocoEval = COCOeval(self.cocoGt, cocoDt, 'keypoints')
        cocoEval.params.imgIds = self.keys
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print(''.join(["%11.4f |" % x for x in cocoEval.stats]))
'''
from coco_eval import *
work_dir = '/media/macul/black/open_pose_train/resnet101_3a_4stage'
cocoeval = cocoeval(output_file=work_dir+'/result.json', op_path=work_dir+'/models/', mode='gpu0')
cocoeval.process()
cocoeval.eval()
#op = openpose_kp('/home/macul/libraries/openpose/models/', 240)
#kps = op.detectLandmark('/home/macul/2018-10-15.png')
'''