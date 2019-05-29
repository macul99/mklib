# generate non-face picture from existing dataset
# Be in the ~/Projects/300W folder
import pickle
import numpy as np
from os import listdir
from os.path import isfile,join
from os import walk
import cv2
import sys
import random

def view_bar(num, total):
	rate = float(num) / total
	rate_num = int(rate * 80)
	r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(80-rate_num), rate_num*100/80, )
	sys.stdout.write(r)
	sys.stdout.flush()


nonFaceFolder = '/home/macul/Projects/300W/labeled_data/nonFace'

with open('results.pkl', 'rb') as f:
	resultDic = pickle.load(f)

datasetLoc = resultDic['datasetLoc']
datasetDic = resultDic['datasetDic']
dsKeys = datasetDic.keys()
dsKeys.sort()

cropPosition = range(8)
counter=80000

dsKeys=[dsKeys[0]]
for i, k in enumerate(dsKeys):

	figNames = datasetDic[k].dataDic.keys()
	totalLen = len(figNames)

	for j, name in enumerate(figNames):
		view_bar(j, totalLen)

		random.shuffle(cropPosition)
		img = cv2.imread(join(datasetLoc[k], name))
		height, width, _ = img.shape
		boundingBox = datasetDic[k].dataDic[name]['boundingBox'][0]
		x1= int(boundingBox[0])
		y1= int(boundingBox[1])
		x2= int(boundingBox[2])
		y2= int(boundingBox[3])

		if x1 == 0:
			x1 += 1

		if y1 == 0:
			y1 += 1

		if x2 == width:
			x2 -= 1

		if y2 == height:
			y2 -= 1

		for c in cropPosition:
			if c==0:
				if (x1/y1<2) and (y1/x1<2):
					cv2.imwrite(join(nonFaceFolder, str(counter)+'.jpg'), img[0:y1,0:x1]) # top-left
					counter += 1
					break;
			elif c==1:
				if ((x2-x1)/y1<2) and (y1/(x2-x1)<2):
					cv2.imwrite(join(nonFaceFolder, str(counter)+'.jpg'), img[0:y1,x1:x2]) # top
					counter += 1
					break;
			elif c==2:
				if ((width-x2)/y1<2) and (y1/(width-x2)<2):
					cv2.imwrite(join(nonFaceFolder, str(counter)+'.jpg'), img[0:y1,x2:width]) # top-right
					counter += 1
					break;
			elif c==3:
				if (x1/(y2-y1)<2) and ((y2-y1)/x1<2):
					cv2.imwrite(join(nonFaceFolder, str(counter)+'.jpg'), img[y1:y2,0:x1]) # left
					counter += 1
					break;
			elif c==4:
				if ((width-x2)/(y2-y1)<2) and ((y2-y1)/(width-x2)<2):
					cv2.imwrite(join(nonFaceFolder, str(counter)+'.jpg'), img[y1:y2,x2:width]) # right
					counter += 1
					break;
			elif c==5:
				if (x1/(height-y2)<2) and ((height-y2)/x1<2):
					cv2.imwrite(join(nonFaceFolder, str(counter)+'.jpg'), img[y2:height,0:x1]) # btm-left
					counter += 1
					break;
			elif c==6:
				if ((x2-x1)/(height-y2)<2) and ((height-y2)/(x2-x1)<2):
					cv2.imwrite(join(nonFaceFolder, str(counter)+'.jpg'), img[y2:height,x1:x2]) # btm
					counter += 1
					break;
			elif c==7:
				if ((width-x2)/(height-y2)<2) and ((height-y2)/(width-x2)<2):
					cv2.imwrite(join(nonFaceFolder, str(counter)+'.jpg'), img[y2:height,x2:width]) # btm-right
					counter += 1
					break;


