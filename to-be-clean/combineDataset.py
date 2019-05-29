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

def preprocessing(fileName, label, fun_callback, bounding_box=None, points=None):	
	#print ('fileName', fileName)
	net_side=48

	dataDic={	'label': label,
				'boundingBox':bounding_box   }

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

	if type(bounding_box)!=type(None):
		bBox = bounding_box.copy().astype(long)				
		#bBox -= 1 # data from matlab is started from index 1			
	else:
		bBox = np.array([0,0,i_w-1,i_h-1])

	if type(points)!=type(None):
		pts = points.copy().astype(float)
		#pts -= 1 # data from matlab is started from index 1
	else:
		pts = None

	cropImg, newPts, scalePts = fun_callback(img, bounding_box=bBox, points=pts, make_square=True)
	#showPointsBoxes(cropImg,points=newPts)

	if type(cropImg)==type(None):
		return None

	if cropImg.shape[0]!=cropImg.shape[1]:
		assert False, 'Image size is not square!'

	#scalingFactor = float(net_side)/cropImg.shape[1]
	cropImg = cropImg.astype(float)
	cropImg = (cropImg-127.5)*0.0078125 # image normalization
	cropImg = cv2.resize(cropImg,(int(net_side),int(net_side))) # resize, do this before swapaxes
	cropImg = np.swapaxes(cropImg, 0, 2) # change (h,w,c) to (c,w,h)

	dataDic['image']=cropImg
	#print "cropImg Size: ", cropImg.shape
	if type(scalePts)!=type(None):
		#newPts *= scalingFactor # scale according to image resize ratio
		#print newPts
		scalePts=scalePts.astype(float)
		dataDic['lbl5Points']=scalePts.T.reshape((scalePts.size,))
		#print dataDic['lbl5Points']
	else:
		dataDic['lbl5Points']=None

	#showPointsBoxes(np.swapaxes(cropImg,0,2),points=newPts)

	return dataDic

### process data with face
trainValRatio = 0.75
preProcDic={'cropOnly': box_cropOnly,
			#'translation': box_translation,
			#'zoom': box_zoom,
			#'vFlip': box_vFlip,
			#'rotate': box_rotate
			}

with open('/home/macul/Projects/300W/results_celebA_zf.pkl', 'rb') as f:
	resultDic = pickle.load(f)

datasetDic = resultDic['datasetDic']
datasetLoc = resultDic['datasetLoc']
dsKeys = datasetDic.keys()
dsKeys.sort()

'''
for i, k in enumerate(dsKeys):
	if i == 0:
		totalDatasetDic = datasetDic[k].dataDic.copy()
	else:
		totalDatasetDic.update(datasetDic[k].dataDic)
'''

trainDatasetDic = {}
valDatasetDic = {}
testDatasetDic = {}
entryCounter=[0,0,0]
countMax=20000
storeIdx=[1,1,1]
filePostfix='_celebA_zf_cropOnly'

for i, k in enumerate(dsKeys):
	print "Processing dataset: ",k
	counter=0
	totalNum=len(datasetDic[k].dataDic.keys())
	for dk in datasetDic[k].dataDic.keys():
		counter+=1
		if counter%100 == 0:
			print "Image counter: ", counter, " out of ", totalNum		

		fileName=join(datasetLoc[k].split()[0],dk) # celebA dataset location is special and need to split

		#print 'fileName', fileName

		if k in ['helen_test','helen_train']:
			#partitionFlag='test'
			partitionFlag='pass'
		elif k in ['celeba_test','celeba_zf_test','celeba_align_test']: # goes to test dataset
			partitionFlag='test'
			#partitionFlag='pass'
		elif k in ['celeba_val','celeba_zf_val','celeba_align_val']: # goes to val dataset
			partitionFlag='val'
			#partitionFlag='pass'
		elif k in ['celeba_train','celeba_zf_train','celeba_align_train']: # goes to train dataset
			partitionFlag='train'
			#partitionFlag='pass'
		elif k in ['']: # goes to train dataset
			partitionFlag='pass'
		else:
			if random.random() < trainValRatio: # goes to train dataset
				partitionFlag='train'
			else: # goes to validation dataset
				partitionFlag='val'			

			partitionFlag='pass'

		if partitionFlag=='pass':
			continue

		for pk in preProcDic.keys():

			#print 'preprocessing: ', pk

			tmpDic=preprocessing(fileName, 1, preProcDic[pk], 
								 bounding_box=datasetDic[k].dataDic[dk]['boundingBox'][0], 
								 points=datasetDic[k].dataDic[dk]['lbl5Points'])

			if type(tmpDic)==type(None):
				continue

			entryName=k+"-"+dk+'-'+pk

			if partitionFlag=='test': # goes to test dataset
				testDatasetDic[entryName]=tmpDic
				entryCounter[2]+=1
				if entryCounter[2]>=countMax:
					entryCounter[2]=0
					print "Writing to pickle file: ", '/home/macul/Projects/300W/testDataset_1_'+str(storeIdx[2])+filePostfix+'.pkl'
					with open('/home/macul/Projects/300W/testDataset_1_'+str(storeIdx[2])+filePostfix+'.pkl', 'wb') as f:
						pickle.dump(testDatasetDic, f, protocol=pickle.HIGHEST_PROTOCOL)
					testDatasetDic={}
					storeIdx[2]+=1
			elif partitionFlag=='val': # goes to val dataset
				valDatasetDic[entryName]=tmpDic
				entryCounter[1]+=1
				if entryCounter[1]>=countMax:
					entryCounter[1]=0
					print "Writing to pickle file: ", '/home/macul/Projects/300W/valDataset_1_'+str(storeIdx[1])+filePostfix+'.pkl'
					with open('/home/macul/Projects/300W/valDataset_1_'+str(storeIdx[1])+filePostfix+'.pkl', 'wb') as f:
						pickle.dump(valDatasetDic, f, protocol=pickle.HIGHEST_PROTOCOL)
					valDatasetDic={}
					storeIdx[1]+=1
			elif partitionFlag=='train': # goes to train dataset
				trainDatasetDic[entryName]=tmpDic
				entryCounter[0]+=1
				if entryCounter[0]>=countMax:
					entryCounter[0]=0
					print "Writing to pickle file: ", '/home/macul/Projects/300W/trainDataset_1_'+str(storeIdx[0])+filePostfix+'.pkl'
					with open('/home/macul/Projects/300W/trainDataset_1_'+str(storeIdx[0])+filePostfix+'.pkl', 'wb') as f:
						pickle.dump(trainDatasetDic, f, protocol=pickle.HIGHEST_PROTOCOL)
					trainDatasetDic={}
					storeIdx[0]+=1
			else:
				pass

if testDatasetDic!={}:
	print "Writing to pickle file: ", '/home/macul/Projects/300W/testDataset_1_'+str(storeIdx[2])+filePostfix+'.pkl'
	with open('/home/macul/Projects/300W/testDataset_1_'+str(storeIdx[2])+filePostfix+'.pkl', 'wb') as f:
		pickle.dump(testDatasetDic, f, protocol=pickle.HIGHEST_PROTOCOL)

if valDatasetDic!={}:
	print "Writing to pickle file: ", '/home/macul/Projects/300W/valDataset_1_'+str(storeIdx[1])+filePostfix+'.pkl'
	with open('/home/macul/Projects/300W/valDataset_1_'+str(storeIdx[1])+filePostfix+'.pkl', 'wb') as f:
		pickle.dump(valDatasetDic, f, protocol=pickle.HIGHEST_PROTOCOL)
		
if trainDatasetDic!={}:
	print "Writing to pickle file: ", '/home/macul/Projects/300W/trainDataset_1_'+str(storeIdx[0])+filePostfix+'.pkl'
	with open('/home/macul/Projects/300W/trainDataset_1_'+str(storeIdx[0])+filePostfix+'.pkl', 'wb') as f:
		pickle.dump(trainDatasetDic, f, protocol=pickle.HIGHEST_PROTOCOL)
				
'''
### verify the dataset
with open('/home/macul/Projects/300W/valDataset_t.pkl', 'rb') as f:
    result=pickle.load(f)
    
keys=result.keys()
data=result[keys[0]]
image=(np.swapaxes(data['image'],0,2)*128+127.5).astype(np.uint8)
pts=(data['lbl5Points'].reshape((2,-1)).T*48).astype(int)
showPointsBoxes(image, points=pts)
'''

'''
### process nonFace data
trainRatio = 0.7
valRatio = 0.9
dirPath='/home/macul/Projects/300W/labeled_data/nonFace'
fileList = sorted(listdir(dirPath))

trainDatasetDic_nonFace = {}
valDatasetDic_nonFace = {}
testDatasetDic_nonFace = {}

for file in fileList:
	fileName=join(dirPath,file)
	tmpDic=preprocessing(fileName, 0)

	rdRatio=random.random()

	if rdRatio<trainRatio: # goes to train dataset
		trainDatasetDic_nonFace[fileName]=tmpDic
	elif rdRatio<valRatio: # goes to val dataset
		valDatasetDic_nonFace[fileName]=tmpDic
	else: # goes to test dataset
		testDatasetDic_nonFace[fileName]=tmpDic

with open('/home/macul/Projects/300W/trainDataset_0.pkl', 'wb') as f:
	pickle.dump(trainDatasetDic_nonFace, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/macul/Projects/300W/valDataset_0.pkl', 'wb') as f:
	pickle.dump(valDatasetDic_nonFace, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/macul/Projects/300W/testDataset_0.pkl', 'wb') as f:
	pickle.dump(testDatasetDic_nonFace, f, protocol=pickle.HIGHEST_PROTOCOL)



with open('/home/macul/Projects/300W/trainDataset_1_1_celebA_zf_cropOnly.pkl', 'rb') as f:
    result=pickle.load(f)
'''