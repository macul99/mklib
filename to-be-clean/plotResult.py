# plot result
import pickle
import matplotlib.pyplot as plt
import numpy as np
import itertools

with open('results.pkl', 'rb') as f:
	resultDic = pickle.load(f)

errorDic = resultDic['errorDic']
modelList = resultDic['modelList']
dsKeys = errorDic.keys()
dsKeys.sort()
colors = itertools.cycle(["r", "b"])

plt.figure(figsize=(20,10))

for i, k in enumerate(dsKeys):

	if i == 0:
		plt.subplot(241)
	elif i == 1:
		plt.subplot(242)
	elif i == 2:
		plt.subplot(243)
	elif i == 3:
		plt.subplot(244)
	elif i == 4:
		plt.subplot(245)
	elif i == 5:
		plt.subplot(246)
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
			tmpErr += [errorDic[k][mIdx][key]]

		if 0 not in tmpErr:
			tmpKeys += [key]

	imgKeys = tmpKeys

	errorBuf = np.zeros((len(imgKeys), len(modelList)))

	legendTxt = []
	for j, m in enumerate(modelList):
		for l, ik in enumerate(imgKeys):
			errorBuf[l, j] = errorDic[k][j][ik]

		sortedErr = sorted(errorBuf[:,j])

		plt.plot(sortedErr, np.array(range(len(sortedErr))).astype(np.float32)/len(sortedErr), label=m, color=next(colors))
		legendTxt += [modelList[j]]

	plt.legend(legendTxt)

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
		plt.subplot(241)
	elif i == 1:
		plt.subplot(242)
	elif i == 2:
		plt.subplot(243)
	elif i == 3:
		plt.subplot(244)
	elif i == 4:
		plt.subplot(245)
	elif i == 5:
		plt.subplot(246)
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
			tmpErr += [errorDic[k][mIdx][key]]

		if 0 not in tmpErr:
			tmpKeys += [key]

	imgKeys = tmpKeys

	errorBuf = np.zeros((len(imgKeys), len(modelList)))

	legendTxt = []
	for j, m in enumerate(modelList):
		for l, ik in enumerate(imgKeys):
			errorBuf[l, j] = errorDic[k][j][ik]

		plt.plot(range(len(imgKeys)), np.cumsum(errorBuf[:,j]), label=m, color=next(colors))
		legendTxt += [modelList[j] + '-avgErr: ' + str(round(np.mean(errorBuf[:,j]),3))]

	plt.legend(legendTxt)

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
		plt.subplot(241)
	elif i == 1:
		plt.subplot(242)
	elif i == 2:
		plt.subplot(243)
	elif i == 3:
		plt.subplot(244)
	elif i == 4:
		plt.subplot(245)
	elif i == 5:
		plt.subplot(246)
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
			tmpErr += [errorDic[k][mIdx][key]]

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

	plt.legend(legendTxt)

	if i in [0,3]:
		plt.ylabel('Normalized_Error')

	if i in [3,4,5]:
		plt.xlabel('Img Idx')
	plt.title(k)

plt.suptitle('Face Alignment Comparison btw DLIB and MTCNN')
plt.show()


