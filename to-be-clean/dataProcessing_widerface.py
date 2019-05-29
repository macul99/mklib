#wider face database processing
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
from os import listdir
from os.path import isfile,join
from os import walk
import timeit
import itertools
from scipy.io import loadmat
import cv2
import cPickle as pickle

# yRange and xRange are an array of 2 items
def proposingBoxes(numBoxes, maxBoxSize, minBoxSize, xRange, yRange):
	pboxes = np.zeros([numBoxes, 4])
	pboxes[:,0:2] = np.random.rand(numBoxes,2)*np.array([xRange[1]-xRange[0]+1, yRange[1]-yRange[0]+1]) + np.array([xRange[0], yRange[0]])
	wh = (np.random.rand(numBoxes,2)*(maxBoxSize-minBoxSize)+minBoxSize) * np.abs(np.random.normal(1, 0.1, size=[numBoxes,2]))
	pboxes[:,2:4] = pboxes[:,0:2] + wh

	# make sure the boxes are inside the image
	pboxes[:,[0,1]] = np.maximum(pboxes[:,[0,1]], np.ones([numBoxes,2])*np.array([xRange[0], yRange[0]]))
	pboxes[:,[2,3]] = np.minimum(pboxes[:,[2,3]], np.ones([numBoxes,2])*np.array([xRange[1], yRange[1]]))

	# make sure the boxes are bigger than minBoxSize
	pick=np.where(np.min(wh, axis=1)>=minBoxSize)[0]

	return pboxes[pick,:]


# return True when overlap bigger than threshold
def checkOverlap(box, boxes, threshold):
	return np.max(IoU(box, boxes)) > threshold


def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes
    Parameters:
    ----------
    box: numpy array , shape (4, ): x1, y1, x2, y2
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes
    Returns:
    -------
    ovl: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovl = inter / np.minimum(area, box_area)
    return ovl

# bboxes: array of [x1,y1,x2,y2]
# threshold:
# 	pos: ovl >threshold[0]
# 	part: threshold[1]<ovl<=threshold[0]
# 	neg: ovl < threshold[2]
# 	two boxex are considered redundant if ovl>threshold[3]
# ratio: number of boxes generated for each category for each gt box
def generateBox(bboxes, minisize, img_h, img_w, ratio=[5, 5, 15], threshold=[0.65, 0.4, 0.3, 0.7]):
	pick=np.min(bboxes[:,2:4]-bboxes[:,0:2], axis=1)>=minisize
	non_pick=np.min(bboxes[:,2:4]-bboxes[:,0:2], axis=1)<minisize
	#print('pick',pick)
	if pick.any():
		box_gt = bboxes[pick,:]

		sz_pos = box_gt.shape[0]*ratio[0]
		sz_part = box_gt.shape[0]*ratio[1]
		sz_neg = box_gt.shape[0]*ratio[2]

		box_pos  = np.zeros([sz_pos, 5]) # the last column stores the index of corresponding gt box
		box_part = np.zeros([sz_part, 5]) # the last column stores the index of corresponding gt box
		box_neg  = np.zeros([sz_neg, 4])

		cnt_pos = 0
		cnt_part = 0
		cnt_neg = 0

		# get neg boxex
		num_try = 0
		while cnt_neg<sz_neg:

			pboxes = proposingBoxes(sz_neg, np.max(bboxes[:,2:4]), minisize, [0,img_w-1], [0,img_h-1])	

			num_try += 1
			#print('neg: num_try ', num_try)
			if num_try > 400:					
				if cnt_neg:
					box_neg = box_neg[0:cnt_neg,:]
				else:
					box_neg = None

				break

			for i in range(pboxes.shape[0]):
				ovl = np.max(IoU(pboxes[i,:], bboxes))
				'''
				print('ovl', ovl)
				print('cnt_neg', cnt_neg)
				'''
				if ovl<threshold[2]:
					# skip if overlap with existing pos boxes
					if cnt_neg and checkOverlap(pboxes[i,:], box_neg[0:cnt_neg,:], threshold[3]):
						continue

					box_neg[cnt_neg,:] = pboxes[i,:]
					cnt_neg += 1

					if cnt_neg >= sz_neg:
						break

		
		# get pos boxex
		num_try = 0
		while (cnt_pos<sz_pos) or (cnt_part<sz_part):
						
			for j in range(box_gt.shape[0]):

				tmp_h = box_gt[j,3]-box_gt[j,1]+1
				tmp_w = box_gt[j,2]-box_gt[j,0]+1
				tmp_max_sz = int(np.max([tmp_h, tmp_w])*2)
				tmp_min_sz = int(np.min([tmp_h, tmp_w])*0.5)
				tmp_x_range = [np.maximum(box_gt[j,0]-np.max([tmp_h, tmp_w]),0), np.minimum(box_gt[j,2]+np.max([tmp_h, tmp_w]),img_w-1)]
				tmp_y_range = [np.maximum(box_gt[j,1]-np.max([tmp_h, tmp_w]),0), np.minimum(box_gt[j,3]+np.max([tmp_h, tmp_w]),img_h-1)]

				pboxes = proposingBoxes(sz_pos*10, tmp_max_sz, tmp_min_sz, tmp_x_range, tmp_y_range)

				num_try += 1
				#print('pos: num_try ', num_try)
				if num_try > max(1000,10*box_gt.shape[0]):					
					if cnt_pos:
						box_pos = box_pos[0:cnt_pos,:]
					else:
						box_pos = None

					if cnt_part:
						box_part = box_part[0:cnt_part,:]
					else:
						box_part = None

					print('pos/part: num_try ', num_try)
					return box_gt, box_pos, box_part, box_neg			

				for i in range(pboxes.shape[0]):
					#ovl = np.max(IoU(pboxes[i,:], bboxes))
					ovl_gt = IoU(pboxes[i,:], box_gt)
					ovl_gt_max = np.max(ovl_gt)
					tmp_idx = np.argmax(ovl_gt)

					# skip if the max_ovl is not with current gt box
					if j!=tmp_idx:
						continue
					
					'''
					print('pick',pick)
					print('non_pick',non_pick)
					print('ovl', ovl)
					print('cnt_pos', cnt_pos)
					print('cnt_neg', cnt_neg)
					print('cnt_part', cnt_part)
					'''
					if ovl_gt_max>threshold[1]:
						# skip if there is another gt_box has big overlap with current box
						ovl_gt_1 = np.delete(ovl_gt, tmp_idx)
						if len(ovl_gt_1) and np.max(ovl_gt_1)>threshold[1]:
							continue

						# skip if overlap with non_pick boxes with the lowest threshold, in this case is 0.3 or threshold[2]
						if non_pick.any() and checkOverlap(pboxes[i,:], bboxes[non_pick,:], threshold[2]):
							continue

						# skip if overlap with existing pos boxes
						if cnt_pos and checkOverlap(pboxes[i,:], box_pos[0:cnt_pos,0:4], threshold[3]):
							continue

						# skip if overlap with existing part boxes
						if cnt_part and checkOverlap(pboxes[i,:], box_part[0:cnt_part,0:4], threshold[3]):
							continue

						# part
						if ovl_gt_max<=threshold[0]:
							if cnt_part and len(np.where(box_part[0:cnt_part,4]==j)[0])>=ratio[1]:
								pass # skip if enough boxes obtained
							else:
								box_part[cnt_part,0:4] = pboxes[i,:]					
								box_part[cnt_part,4] = j
								cnt_part += 1
						# pos
						else:
							if cnt_pos and len(np.where(box_pos[0:cnt_pos,4]==j)[0])>=ratio[0]:
								pass # skip if enough boxes obtained
							else:
								box_pos[cnt_pos,0:4] = pboxes[i,:]					
								box_pos[cnt_pos,4] = tmp_idx
								cnt_pos += 1
					else:
						pass

					if cnt_pos and cnt_part and len(np.where(box_pos[0:cnt_pos,4]==j)[0])>=ratio[0] and len(np.where(box_part[0:cnt_part,4]==j)[0])>=ratio[1]:
						break

		print('pos/part: num_try ', num_try)
		return box_gt, box_pos, box_part, box_neg
	else:
		return None, None, None, None


dataLoc = { 'train': '/home/macul/Projects/300W/labeled_data/WiderFace/WIDER_train',
			'val': '/home/macul/Projects/300W/labeled_data/WiderFace/WIDER_val',
			#'test': '/home/macul/Projects/300W/labeled_data/WiderFace/WIDER_test' 
		  }
minisize = 12 # the min(w,h) of the bbox

for dk in dataLoc.keys():
	tmpDataDic={}

	matData = loadmat(join(dataLoc[dk],'wider_face_'+dk+'.mat'))

	fileList = matData['file_list']
	bbxList = matData['face_bbx_list']
	dirList = listdir(join(dataLoc[dk],'images'))
	dirList.sort()
	numFolder = len(dirList)

	for i in range(numFolder):
		numFile = fileList[i,0].shape[0]
		
		for j in range(numFile):
			bbox = bbxList[i,0][j,0]
			bbox[:,2:4]=bbox[:,0:2]+bbox[:,2:4]

			filePath=join(dataLoc[dk],'images',dirList[i],fileList[i,0][j,0][0]+'.jpg')
			print('filePath', filePath)
			img=cv2.imread(filePath)
			img_h, img_w, _ = img.shape
			#print('img.shape', img.shape)

			tmp_gt,tmp_pos,tmp_part,tmp_neg = generateBox(bbox, minisize, img_h, img_w)

			if type(tmp_gt) != type(None):
				tmpDataDic[filePath] = {}
				tmpDataDic[filePath]['gt']   = tmp_gt
				tmpDataDic[filePath]['pos']  = tmp_pos
				tmpDataDic[filePath]['part'] = tmp_part
				tmpDataDic[filePath]['neg']  = tmp_neg

		try:
			with open('/home/macul/Projects/300W/widerface_'+dk+'.pkl', 'wb') as f:
				pickle.dump(tmpDataDic, f, protocol=pickle.HIGHEST_PROTOCOL)
		except:
			pass

'''
import dataProcessing_widerface
import numpy as np
a,b,c,d=dataProcessing_widerface.generateBox(np.array([[5,6, 10, 12],[10,15,16,20]]),3,30,30)


with open('/home/macul/Projects/300W/widerface_train.pkl', 'rb') as f:
    result=pickle.load(f)
'''
