# plot the 5 points face landmard for 300-W dataset
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import caffe
from os import listdir
from os.path import isfile,join
from os import walk
import pickle
import timeit
import itertools
from scipy import ndimage
import imp
import compareFaceAlignModel
imp.reload(compareFaceAlignModel) # reload the module if anything inside this module has changed
from compareFaceAlignModel import Dataset, DatasetCelebA, DlibShapePredictor, mtcnn, OpenPoseKeyPoints, dan


datasetLoc = 	{ #'celeba_align_train': '/home/macul/Projects/300W/labeled_data/CelebA_Align img_align_celeba_png 0',
				  #'celeba_align_val': '/home/macul/Projects/300W/labeled_data/CelebA_Align img_align_celeba_png 1',
				  #'celeba_align_test': '/home/macul/Projects/300W/labeled_data/CelebA_Align img_align_celeba_png 2',
				  #'celeba_train': '/home/macul/Projects/300W/labeled_data/CelebA img_celeba 0',
				  #'celeba_val': '/home/macul/Projects/300W/labeled_data/CelebA img_celeba 1',
				  'celeba_test': '/home/macul/Projects/300W/labeled_data/CelebA img_celeba 2',
				  #'celeba_zf_train': '/home/macul/Projects/300W/labeled_data/CelebA img_celeba 0',
				  #'celeba_zf_val': '/home/macul/Projects/300W/labeled_data/CelebA img_celeba 1',
				  #'celeba_zf_test': '/home/macul/Projects/300W/labeled_data/CelebA img_celeba 2',
				  'afw': '/home/macul/Projects/300W/labeled_data/afw',
				  'helen_test': '/home/macul/Projects/300W/labeled_data/helen/testset',
				  #'helen_train': '/home/macul/Projects/300W/labeled_data/helen/trainset',
				  #'ibug': '/home/macul/Projects/300W/labeled_data/ibug',
				  'lfpw_test': '/home/macul/Projects/300W/labeled_data/lfpw/testset',
				  #'lfpw_train': '/home/macul/Projects/300W/labeled_data/lfpw/trainset',
				  'fddb': '/home/macul/Projects/300W/labeled_data/fddb',
				  'aflw': '/home/macul/Projects/300W/labeled_data/aflw'
				}


boundBoxFile = 	{ 'celeba_align_train': None,
				  'celeba_align_val': None,
				  'celeba_align_test': None,
				  'celeba_train': 'list_bbox_celeba.txt',
				  'celeba_val': 'list_bbox_celeba.txt',
				  'celeba_test': 'list_bbox_celeba.txt',
				  'celeba_zf_train': 'ZF_bbox_list.txt',
				  'celeba_zf_val': 'ZF_bbox_list.txt',
				  'celeba_zf_test': 'ZF_bbox_list.txt',
				  'afw': 'bounding_boxes.mat',
				  'helen_test': 'bounding_boxes.mat',
				  'helen_train': 'bounding_boxes.mat',
				  'ibug': 'bounding_boxes.mat',
				  'lfpw_test': 'bounding_boxes.mat',
				  'lfpw_train': 'bounding_boxes.mat',
				  'fddb': None,
				  'aflw': None
				}

labelFile = 	{ 'celeba_align_train': 'list_landmarks_align_celeba.txt',
				  'celeba_align_val': 'list_landmarks_align_celeba.txt',
				  'celeba_align_test': 'list_landmarks_align_celeba.txt',
				  'celeba_train': 'list_landmarks_celeba.txt',
				  'celeba_val': 'list_landmarks_celeba.txt',
				  'celeba_test': 'list_landmarks_celeba.txt',
				  'celeba_zf_train': 'list_landmarks_celeba.txt',
				  'celeba_zf_val': 'list_landmarks_celeba.txt',
				  'celeba_zf_test': 'list_landmarks_celeba.txt',
				  'afw': None,
				  'helen_test': 'bounding_boxes.mat',
				  'helen_train': 'bounding_boxes.mat',
				  'ibug': None,
				  'lfpw_test': None,
				  'lfpw_train': None,
				  'fddb': None,
				  'aflw': None
				}


datasetKeys = datasetLoc.keys()
datasetKeys.sort()

resultDic = {}
resultDic['datasetLoc'] = datasetLoc
resultDic['boundBoxFile'] = boundBoxFile
resultDic['modelList'] = ['dlib68', 'mtcnn', 'dan', 'op_dan']#['dlib68', 'openpose_fh32', 'openpose_fh64', 'mtcnn', 'dan', 'op_dan']
resultFile='results_test.pkl'
datasetDic = {}
datasetDicOpenpose = {} #dataset using openpose to estimate the bbox
errorDic = {}
allOutputDic = {}
num_of_points = 3

colorList = ["r", "b", "g", "k", "m", "c"]
colors = itertools.cycle(colorList[0:len(resultDic['modelList'])])

for k in datasetKeys:
	print ('dataset: ', k)

	if 'celeba' in k:
		datasetDic[k] = DatasetCelebA(datasetLoc[k], lbl_file=labelFile[k], bound_box_file=boundBoxFile[k])
		if 'op_dan' in resultDic['modelList']:
			datasetDicOpenpose[k] = DatasetCelebA(datasetLoc[k], lbl_file=labelFile[k], bound_box_file=boundBoxFile[k], openpose_bbox=True)
	else:
		datasetDic[k] = Dataset(datasetLoc[k], bound_box_file=boundBoxFile[k])
		if 'op_dan' in resultDic['modelList']:
			datasetDicOpenpose[k] = Dataset(datasetLoc[k], bound_box_file=boundBoxFile[k], openpose_bbox=True)

	tmpError = []
	tmpOutput = []
	for i, m in enumerate(resultDic['modelList']):		
		print ('....model: ', m)
		
		if m == 'dlib68':
			shapePred=DlibShapePredictor(datasetDic[k])
			start_time=timeit.default_timer()
			outputDic=shapePred.getModelOutput(num_points=num_of_points)
		elif m == 'mtcnn':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/mtcnn/model', 'det3', 'det3')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput(num_points=num_of_points)
		elif m == 'openpose_fh32':
			assert num_of_points==3, 'openpose only support 3 points'
			opPred = OpenPoseKeyPoints(datasetDic[k],'/home/macul/libraries/openpose/models/', 32)
			start_time=timeit.default_timer()
			outputDic = opPred.getModelOutput()
		elif m == 'openpose_fh48':
			assert num_of_points==3, 'openpose only support 3 points'
			opPred = OpenPoseKeyPoints(datasetDic[k],'/home/macul/libraries/openpose/models/', 48)
			start_time=timeit.default_timer()
			outputDic = opPred.getModelOutput()
		elif m == 'openpose_fh64':
			assert num_of_points==3, 'openpose only support 3 points'
			opPred = OpenPoseKeyPoints(datasetDic[k],'/home/macul/libraries/openpose/models/', 64)
			start_time=timeit.default_timer()
			outputDic = opPred.getModelOutput()
		elif m == 'dan':
			danPred = dan(datasetDic[k],'/home/macul/Projects/DAN_landmark/models','dan','dan')
			start_time=timeit.default_timer()
			outputDic = danPred.getModelOutput(num_points=num_of_points)
		elif m == 'op_dan':
			danPred = dan(datasetDicOpenpose[k],'/home/macul/Projects/DAN_landmark/models','dan','dan')
			start_time=timeit.default_timer()
			outputDic = danPred.getModelOutput(num_points=num_of_points)
		elif m == 'mymtcnn40':			
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40', 'fc1024_layer_freeze_test', '_iter_40000')			
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()			
		elif m == 'mymtcnn40_1':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_1', 'fc1024_layer_freeze_test', '_iter_40000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn40_2':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_2', 'fc1024_layer_freeze_test', '_iter_490000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn40_3':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_3', 'fc1024_layer_freeze_test', '_iter_40000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn40_4':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_4', 'fc1024_layer_freeze_test', '_iter_180000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn40_5':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_5', 'fc1024_layer_freeze_test', '_iter_230000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn40_6':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_6', 'fc256_layer_freeze_test', '_iter_260000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn40_7':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_7', 'fc256_layer_learnall_test', '_iter_230000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mtcnn_opti':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_8', 'fc256_layer_learnall_test', '_iter_30000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn40_8_1':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_8_1', 'fc256_layer_learnall_test', '_iter_30000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn40_8_2':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_8_2', 'fc256_layer_learnall_test', '_iter_300000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn40_8_3':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_8_3', 'fc256_org_relpos_test', '_iter_300000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn40_8_4':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_8_4', 'fc256_org_relpos_test', '_iter_700000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn40_9':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_40_9', 'fc256_layer_learnall_test', '_iter_430000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn41':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_41', 'fc1024_layer_learnall_test', '_iter_50000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn42':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_42', 'fc256x2_test', '_iter_70000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn43':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_43', 'fc1024_relpos_freeze_test', '_iter_260000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn43_1':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_43_1', 'fc1024_relpos_freeze_test', '_iter_120000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn44':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_44', 'fc1024_relpos_freeze_test', '_iter_290000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn44_1':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_44_1', 'fc1024_relpos_freeze_test', '_iter_120000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn32':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_32', '3loss_fc1024_freeze_test', '_iter_740000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn31':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_31', '3loss_org_freeze_test', '_iter_330000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn30':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_30', '3loss_3cls_freeze_test', '_iter_60000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn23':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_23', 'fc256newx2_relpos1_freeze_test', '_iter_80000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn22':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_22', 'fc256newx2_relpos1_freeze_test', '_iter_210000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn21':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_21', 'fc256newx2_relpos1_freeze_test', '_iter_300000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn20':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_20', 'fc256newx2_relpos1_freeze_test', '_iter_50000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn16':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_16', 'fc256newx2_test', '_iter_200000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn15':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_15', 'fc256newx2_test', '_iter_450000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn13':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_13', 'fc256newx2_test', '_iter_210000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn14':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_14', 'fc256newx2_test', '_iter_210000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn12':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_12', 'fc256newx2_test', '_iter_200000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn1':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_1', 'fc1024_layer_freeze_test', '_iter_340000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn1_1':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_1_1', 'fc1024_layer_freeze_test', '_iter_300000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn1_2':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_1_2', 'fc1024_layer_freeze_test', '_iter_220000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn2':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_2', 'fc1024_layer_freeze_test', '_iter_1000000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn3':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_3', 'fc1024_layer_learnall_test', '_iter_350000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn4':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_4', 'fc1024_layer_freeze_test', '_iter_240000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn5':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_5', 'fc1024_ohem_freeze_test', '_iter_340000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn7':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_7', 'fc1024_relpos_freeze_test', '_iter_240000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn8':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_8', 'fc256x2_frz1st256fc_test', '_iter_70000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn9':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_9', 'fc256x2_test', '_iter_130000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn10':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_10', 'fc256newx2_test', '_iter_120000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		elif m == 'mymtcnn11':
			mtcnnPred = mtcnn(datasetDic[k],'/home/macul/Projects/300W/snapshot_11', 'fc256newx2_test', '_iter_210000')
			start_time=timeit.default_timer()
			outputDic = mtcnnPred.getModelOutput()
		else:
			assert False, 'Model: '+ m +' is not supported!'

		print ('Dataset-'+k+',Model-'+m+', time elapsed: ', timeit.default_timer() - start_time)

		if m == 'op_dan':
			tmpError += [datasetDicOpenpose[k].computeError(outputDic, num_points=num_of_points)]
		else:
			tmpError += [datasetDic[k].computeError(outputDic, num_points=num_of_points)]
		tmpOutput += [outputDic]

	errorDic[k] = tmpError
	allOutputDic[k] = tmpOutput

resultDic['datasetDic'] = datasetDic
resultDic['errorDic'] = errorDic
resultDic['outputDic'] = allOutputDic

# Store data (serialize)
with open(resultFile, 'wb') as f:
    pickle.dump(resultDic, f, protocol=pickle.HIGHEST_PROTOCOL)

# Load data (deserialize)
#with open('results.pkl', 'rb') as f:
#    resultDic1 = pickle.load(f)

if resultDic['modelList']!=[]:
	errorDic = resultDic['errorDic']
	modelList = resultDic['modelList']
	dsKeys = errorDic.keys()
	dsKeys.sort()

	plt.figure(figsize=(20,10))

	for i, k in enumerate(dsKeys):
		
		if i == 0:
			plt.subplot(231)
		elif i == 1:
			plt.subplot(232)
		elif i == 2:
			plt.subplot(233)
		elif i == 3:
			plt.subplot(234)
		elif i == 4:
			plt.subplot(235)
		elif i == 5:
			plt.subplot(236)
		elif i == 6:
			plt.subplot(247)
		elif i == 7:
			plt.subplot(248)
		else:
			assert False, 'Not supported!'
		

		imgKeys = errorDic[k][0].keys()
		imgKeys.sort()

		print('imgKeys: ', len(imgKeys))

		# ignore the cases where error=0 (happend when no landmark are detected)
		tmpKeys=[]
		for key in imgKeys:
			tmpErr=[]
			for mIdx in range(len(modelList)):
				try:
					tmp = errorDic[k][mIdx][key] # the key may not exist for openpose dataset
				except:
					tmp = 0
				tmpErr += [tmp]

			if 0 not in tmpErr:
				tmpKeys += [key]

		imgKeys = tmpKeys

		errorBuf = np.zeros((len(imgKeys), len(modelList)))

		legendTxt = []
		for j, m in enumerate(modelList):
			for l, ik in enumerate(imgKeys):
				errorBuf[l, j] = errorDic[k][j][ik]

			sortedErr = sorted(errorBuf[:,j])

			print('sortedErr/imgKeys: ', len(sortedErr),len(imgKeys))

			plt.plot(sortedErr, np.array(range(len(sortedErr))).astype(np.float32)/len(sortedErr), label=m, color=next(colors), linewidth=3)
			#legendTxt += [modelList[j]]
			legendTxt += [modelList[j] + '-avgErr: ' + str(round(np.mean(errorBuf[:,j]),3)) + ', -stdev: ' + str(round(np.std(errorBuf[:,j]),3))]

		plt.legend(legendTxt,loc='best',fontsize=10)

		#plt.xlim( (0, 0.8) )

		if i in [0,3]:
			plt.ylabel('Fraction of the number of img')

		if i in [3,4,5]:
			plt.xlabel('Pt-Pt error normalized')
		plt.title(k)

	plt.suptitle('Face Alignment Comparison btw DLIB and MTCNN')
	plt.show()

	plt.figure(figsize=(20,10))

	for i, k in enumerate(dsKeys):
		
		if i == 0:
			plt.subplot(231)
		elif i == 1:
			plt.subplot(232)
		elif i == 2:
			plt.subplot(233)
		elif i == 3:
			plt.subplot(234)
		elif i == 4:
			plt.subplot(235)
		elif i == 5:
			plt.subplot(236)
		elif i == 6:
			plt.subplot(247)
		elif i == 7:
			plt.subplot(248)
		else:
			assert False, 'Not supported!'
		

		imgKeys = errorDic[k][0].keys()
		imgKeys.sort()

		# ignore the cases where error=0 (happend when no landmark are detected)
		tmpKeys=[]
		for key in imgKeys:
			tmpErr=[]
			for mIdx in range(len(modelList)):
				try:
					tmp = errorDic[k][mIdx][key] # the key may not exist for openpose dataset
				except:
					tmp = 0
				tmpErr += [tmp]

			if 0 not in tmpErr:
				tmpKeys += [key]

		imgKeys = tmpKeys

		errorBuf = np.zeros((len(imgKeys), len(modelList)))

		legendTxt = []
		for j, m in enumerate(modelList):
			for l, ik in enumerate(imgKeys):
				errorBuf[l, j] = errorDic[k][j][ik]

			plt.plot(range(len(imgKeys)), np.cumsum(errorBuf[:,j]), label=m, color=next(colors), linewidth=3)
			legendTxt += [modelList[j] + '-avgErr: ' + str(round(np.mean(errorBuf[:,j]),3))]

		plt.legend(legendTxt,loc='best',fontsize=10)

		if i in [0,3]:
			plt.ylabel('cumsum(Normalized_Error)')

		if i in [3,4,5]:
			plt.xlabel('Img Idx')
		plt.title(k)

	plt.suptitle('Face Alignment Comparison btw DLIB and MTCNN')
	plt.show()


	plt.figure(figsize=(20,10))

	for i, k in enumerate(dsKeys):
		
		if i == 0:
			plt.subplot(231)
		elif i == 1:
			plt.subplot(232)
		elif i == 2:
			plt.subplot(233)
		elif i == 3:
			plt.subplot(234)
		elif i == 4:
			plt.subplot(235)
		elif i == 5:
			plt.subplot(236)
		elif i == 6:
			plt.subplot(247)
		elif i == 7:
			plt.subplot(248)
		else:
			assert False, 'Not supported!'
		

		imgKeys = errorDic[k][0].keys()
		imgKeys.sort()

		# ignore the cases where error=0 (happend when no landmark are detected)
		tmpKeys=[]
		for key in imgKeys:
			tmpErr=[]
			for mIdx in range(len(modelList)):
				try:
					tmp = errorDic[k][mIdx][key] # the key may not exist for openpose dataset
				except:
					tmp = 0
				tmpErr += [tmp]

			if 0 not in tmpErr:
				tmpKeys += [key]

		imgKeys = tmpKeys

		errorBuf = np.zeros((len(imgKeys), len(modelList)))

		legendTxt = []
		for j, m in enumerate(modelList):
			for l, ik in enumerate(imgKeys):
				errorBuf[l, j] = errorDic[k][j][ik]

			plt.scatter(range(len(imgKeys)), errorBuf[:,j], label=m, alpha=1, marker=r'x', color=next(colors))
			legendTxt += [modelList[j] + '-avgErr: ' + str(round(np.mean(errorBuf[:,j]),3))]

		plt.legend(legendTxt,loc='best',fontsize=10)

		if i in [0,3]:
			plt.ylabel('Normalized_Error')

		if i in [3,4,5]:
			plt.xlabel('Img Idx')
		plt.title(k)

	plt.suptitle('Face Alignment Comparison btw DLIB and MTCNN')
	plt.show()