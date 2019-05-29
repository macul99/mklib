# Combine Face Align Dataset
# Be in the ~/Projects/300W folder
import numpy as np
from os.path import isfile,join
import sys
#sys.path.append('/home/cmcc/caffe-master/python')
import cv2
import random
import cPickle as pickle
from os import listdir
from imgArgument import showPointsBoxes,box_cropOnly,box_translation,box_zoom,box_vFlip,box_rotate

def pad(boxesA, w, h):
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

def rerec(bboxA):
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

def preprocessing(fileName, dataDic):
	net_side = 48

	image = cv2.imread(fileName)

	if type(image)==type(None):
		print 'image name: ', fileName

	i_w = image.shape[1]
	i_h = image.shape[0]
	i_c = image.shape[2]
	
	# convert RGB to BGR
	img = image.copy()
	tmp = img[:,:,2].copy()
	img[:,:,2] = img[:,:,0]
	img[:,:,0] = tmp


	cnt_neg = 0
	cnt_pos = 0
	cnt_part = 0

	gt_data = dataDic['gt']
	neg_data = dataDic['neg']
	pos_data = dataDic['pos']
	part_data = dataDic['part']

	neg_result = {}
	pos_result = {}
	part_result = {}

	# process gt bbox
	for i in range(gt_data.shape[0]):
		tmpDic = {}	
		tmpDic['label'] = 1

		gt_w = gt_data[i][2]-gt_data[i][0]+1
		gt_h = gt_data[i][3]-gt_data[i][1]+1
		rec_bbox = rerec(gt_data[i][np.newaxis,:])[0].astype(float) #[x1,y1,x2,y2]	
		rec_bbox = np.fix(rec_bbox)	
		tmpDic['bbox'] = np.array([ (gt_data[i][0]-rec_bbox[0])/gt_w, 
									(gt_data[i][1]-rec_bbox[1])/gt_h, 
									(gt_data[i][2]-rec_bbox[2])/gt_w, 
									(gt_data[i][3]-rec_bbox[3])/gt_h	])

		[dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(rec_bbox[np.newaxis,:], i_w, i_h)

		tempimg = np.zeros((net_side, net_side, i_c))
		tmp = np.zeros((int(tmph[0]), int(tmpw[0]), i_c))
		tmp[int(dy[0]):int(edy[0])+1, int(dx[0]):int(edx[0])+1] = img[int(y[0]):int(ey[0])+1, int(x[0]):int(ex[0])+1]
		tempimg[:,:,:] = cv2.resize(tmp, (net_side, net_side))
		tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]

		#cv2.namedWindow("gt", cv2.WINDOW_NORMAL)
		#cv2.imshow('gt', tempimg.astype(np.uint8))
		#kVal=cv2.waitKey()
		
		tempimg = np.swapaxes(tempimg, 0, 2)
		tmpDic['image'] = tempimg

		kstr = fileName.split('/')[-1].split('.')[0]+'_'+str(cnt_pos)
		cnt_pos += 1
		pos_result[kstr]=tmpDic

	# process pos bbox
	if type(pos_data) != type(None):
		for i in range(pos_data.shape[0]):
			tmpDic = {}	
			tmpDic['label'] = 1

			ref_gt_bbox_num = int(pos_data[i][4])
			gt_w = gt_data[ref_gt_bbox_num][2]-gt_data[ref_gt_bbox_num][0]+1
			gt_h = gt_data[ref_gt_bbox_num][3]-gt_data[ref_gt_bbox_num][1]+1
			rec_bbox = rerec(pos_data[i][np.newaxis,0:4])[0].astype(float) #[x1,y1,x2,y2]
			rec_bbox = np.fix(rec_bbox)
			tmpDic['bbox'] = np.array([ (gt_data[ref_gt_bbox_num][0]-rec_bbox[0])/gt_w, 
										(gt_data[ref_gt_bbox_num][1]-rec_bbox[1])/gt_h, 
										(gt_data[ref_gt_bbox_num][2]-rec_bbox[2])/gt_w, 
										(gt_data[ref_gt_bbox_num][3]-rec_bbox[3])/gt_h	])

			[dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(rec_bbox[np.newaxis,:], i_w, i_h)

			tempimg = np.zeros((net_side, net_side, i_c))
			tmp = np.zeros((int(tmph[0]), int(tmpw[0]), i_c))
			tmp[int(dy[0]):int(edy[0])+1, int(dx[0]):int(edx[0])+1] = img[int(y[0]):int(ey[0])+1, int(x[0]):int(ex[0])+1]
			tempimg[:,:,:] = cv2.resize(tmp, (net_side, net_side))
			tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]

			#cv2.namedWindow("pos", cv2.WINDOW_NORMAL)
			#cv2.imshow('pos', tempimg.astype(np.uint8))
			#kVal=cv2.waitKey()

			tempimg = np.swapaxes(tempimg, 0, 2)
			tmpDic['image'] = tempimg

			kstr = fileName.split('/')[-1].split('.')[0]+'_'+str(cnt_pos)
			cnt_pos += 1
			pos_result[kstr]=tmpDic

	# process part bbox
	if type(part_data) != type(None):
		for i in range(part_data.shape[0]):
			tmpDic = {}	
			tmpDic['label'] = 1

			ref_gt_bbox_num = int(part_data[i][4])
			gt_w = gt_data[ref_gt_bbox_num][2]-gt_data[ref_gt_bbox_num][0]+1
			gt_h = gt_data[ref_gt_bbox_num][3]-gt_data[ref_gt_bbox_num][1]+1
			rec_bbox = rerec(part_data[i][np.newaxis,0:4])[0].astype(float) #[x1,y1,x2,y2]
			rec_bbox = np.fix(rec_bbox)
			tmpDic['bbox'] = np.array([ (gt_data[ref_gt_bbox_num][0]-rec_bbox[0])/gt_w, 
										(gt_data[ref_gt_bbox_num][1]-rec_bbox[1])/gt_h, 
										(gt_data[ref_gt_bbox_num][2]-rec_bbox[2])/gt_w, 
										(gt_data[ref_gt_bbox_num][3]-rec_bbox[3])/gt_h	])

			[dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(rec_bbox[np.newaxis,:], i_w, i_h)

			tempimg = np.zeros((net_side, net_side, i_c))
			tmp = np.zeros((int(tmph[0]), int(tmpw[0]), i_c))
			tmp[int(dy[0]):int(edy[0])+1, int(dx[0]):int(edx[0])+1] = img[int(y[0]):int(ey[0])+1, int(x[0]):int(ex[0])+1]
			tempimg[:,:,:] = cv2.resize(tmp, (net_side, net_side))
			tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]

			#cv2.namedWindow("part", cv2.WINDOW_NORMAL)
			#cv2.imshow('part', tempimg.astype(np.uint8))
			#kVal=cv2.waitKey()

			tempimg = np.swapaxes(tempimg, 0, 2)
			tmpDic['image'] = tempimg

			kstr = fileName.split('/')[-1].split('.')[0]+'_'+str(cnt_part)
			cnt_part += 1
			part_result[kstr]=tmpDic

	# process neg bbox
	if type(neg_data) != type(None):
		for i in range(neg_data.shape[0]):
			tmpDic = {}	
			tmpDic['label'] = 0

			rec_bbox = rerec(neg_data[i][np.newaxis,0:4])[0].astype(float) #[x1,y1,x2,y2]
			rec_bbox = np.fix(rec_bbox)

			[dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(rec_bbox[np.newaxis,:], i_w, i_h)

			tempimg = np.zeros((net_side, net_side, i_c))
			tmp = np.zeros((int(tmph[0]), int(tmpw[0]), i_c))
			tmp[int(dy[0]):int(edy[0])+1, int(dx[0]):int(edx[0])+1] = img[int(y[0]):int(ey[0])+1, int(x[0]):int(ex[0])+1]
			tempimg[:,:,:] = cv2.resize(tmp, (net_side, net_side))
			tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]

			#cv2.namedWindow("neg", cv2.WINDOW_NORMAL)
			#cv2.imshow('neg', tempimg.astype(np.uint8))
			#kVal=cv2.waitKey()

			tempimg = np.swapaxes(tempimg, 0, 2)
			tmpDic['image'] = tempimg

			kstr = fileName.split('/')[-1].split('.')[0]+'_'+str(cnt_neg)
			cnt_neg += 1
			neg_result[kstr]=tmpDic

	return neg_result, pos_result, part_result


with open('/home/macul/Projects/300W/widerface_train.pkl', 'rb') as f:
	dataDic = pickle.load(f)

filePostfix = 'train_'
dsKeys = dataDic.keys()
dsKeys.sort()

negDic = {}
posDic = {}
partDic = {}

countMax=20000
storeIdx=[1,1,1]

for i, dk in enumerate(dsKeys):
	print(dk)
	tmp_negDic, tmp_posDic, tmp_partDic = preprocessing(dk, dataDic[dk])

	if len(tmp_negDic):
		negDic.update(tmp_negDic)

		if len(negDic) >= countMax:
			print "Writing to pickle file: ", '/home/macul/Projects/300W/widerFace_'+filePostfix+'neg_'+str(storeIdx[0])+'.pkl'
			with open('/home/macul/Projects/300W/widerFace_'+filePostfix+'neg_'+str(storeIdx[0])+'.pkl', 'wb') as f:
				pickle.dump(negDic, f, protocol=pickle.HIGHEST_PROTOCOL)
			negDic={}
			storeIdx[0]+=1

	if len(tmp_posDic):
		posDic.update(tmp_posDic)

		if len(posDic) >= countMax:
			print "Writing to pickle file: ", '/home/macul/Projects/300W/widerFace_'+filePostfix+'pos_'+str(storeIdx[1])+'.pkl'
			with open('/home/macul/Projects/300W/widerFace_'+filePostfix+'pos_'+str(storeIdx[1])+'.pkl', 'wb') as f:
				pickle.dump(posDic, f, protocol=pickle.HIGHEST_PROTOCOL)
			posDic={}
			storeIdx[1]+=1

	if len(tmp_partDic):
		partDic.update(tmp_partDic)

		if len(partDic) >= countMax:
			print "Writing to pickle file: ", '/home/macul/Projects/300W/widerFace_'+filePostfix+'part_'+str(storeIdx[2])+'.pkl'
			with open('/home/macul/Projects/300W/widerFace_'+filePostfix+'part_'+str(storeIdx[2])+'.pkl', 'wb') as f:
				pickle.dump(partDic, f, protocol=pickle.HIGHEST_PROTOCOL)
			partDic={}
			storeIdx[2]+=1

if len(negDic):
	print "Writing to pickle file: ", '/home/macul/Projects/300W/widerFace_'+filePostfix+'neg_'+str(storeIdx[0])+'.pkl'
	with open('/home/macul/Projects/300W/widerFace_'+filePostfix+'neg_'+str(storeIdx[0])+'.pkl', 'wb') as f:
		pickle.dump(negDic, f, protocol=pickle.HIGHEST_PROTOCOL)

if len(posDic):
	print "Writing to pickle file: ", '/home/macul/Projects/300W/widerFace_'+filePostfix+'pos_'+str(storeIdx[1])+'.pkl'
	with open('/home/macul/Projects/300W/widerFace_'+filePostfix+'pos_'+str(storeIdx[1])+'.pkl', 'wb') as f:
		pickle.dump(posDic, f, protocol=pickle.HIGHEST_PROTOCOL)

if len(partDic) >= countMax:
	print "Writing to pickle file: ", '/home/macul/Projects/300W/widerFace_'+filePostfix+'part_'+str(storeIdx[2])+'.pkl'
	with open('/home/macul/Projects/300W/widerFace_'+filePostfix+'part_'+str(storeIdx[2])+'.pkl', 'wb') as f:
		pickle.dump(partDic, f, protocol=pickle.HIGHEST_PROTOCOL)