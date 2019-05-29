# compare auto exposure algorithm
import numpy as np
import random
import cPickle as pickle
from os import listdir
from os.path import isfile,join
import matplotlib.pyplot as plt

dataPath = '/home/macul/Projects/Auto-exposure'
dataDic={'log_30-01-2018-08-08-42_auto_exposure_true.txt': True,
		 'log_31-01-2018-08-40-14_auto_exposure_false.txt': False}
resultDic={}
for k in dataDic.keys():
	print k

	with open(join(dataPath,k),'rb') as f:
		lines = f.readlines()
		print "Total number of lines:", len(lines)

		startIdx = lines.index('}\n') + 1

		tmp = lines[startIdx].split()[1]
		if tmp == 'full' or tmp == 'face':
			numEntry = 4
		else:			
			numEntry = 5

		counter = 0
		curSec = ''
		fullData = []
		faceData = []

		for i in range(startIdx, len(lines)):
			records = lines[i].split()
			if curSec != records[numEntry-4]:
				curSec = records[numEntry-4]
				counter += 1

				if records[numEntry-3] == 'full':
					fullData.append([counter, int(records[numEntry-2]), int(records[numEntry-1])])
				elif records[numEntry-3] == 'face':
					faceData.append([counter, int(records[numEntry-2])])

		resultDic[k]={'fullData': np.array(fullData), 'faceData': np.array(faceData)}

with open(join(dataPath,'result.pkl'),'wb') as f:
	pickle.dump(resultDic, f, protocol=pickle.HIGHEST_PROTOCOL)

keys=resultDic.keys()

plt.figure(figsize=(20,10))
for key in resultDic.keys():
	if dataDic[key]:
		plt.subplot(211)
		plt.plot(resultDic[key]['fullData'][:,0], resultDic[key]['fullData'][:,1], label='curFull', color='r')
		plt.plot(resultDic[key]['fullData'][:,0], resultDic[key]['fullData'][:,2], label='refFull', color='g')
		plt.scatter(resultDic[key]['faceData'][:,0], resultDic[key]['faceData'][:,1], label='curFace(ref=130)', color='b', marker=r'x')
		plt.xlim([0, 100000])
		plt.ylim([0, 250])
		legendTxt = ['curFull', 'refFull', 'curFace(ref=130), RMSE='+str(int(np.sqrt(np.mean((resultDic[key]['faceData'][:,1]-130)**2))))]
		plt.legend(legendTxt)
		plt.ylabel('Intensity')
		plt.xlabel('Time (sec)')
		plt.title('New Auto-exposure')
	else:
		tmpCount=resultDic[key]['faceData'][:,0]
		#tmpIdx=(tmpCount<50000) | (tmpCount>70000)
		tmpIdx=range(0,len(tmpCount))
		tmpCount=tmpCount[tmpIdx]
		tmpArray=resultDic[key]['faceData'][:,1][tmpIdx]
		plt.subplot(212)
		plt.plot(resultDic[key]['fullData'][:,0], resultDic[key]['fullData'][:,1], label='curIntensity', color='r')
		plt.plot(resultDic[key]['fullData'][:,0], resultDic[key]['fullData'][:,2], label='refIntensity', color='g')
		plt.scatter(tmpCount, tmpArray, label='faceIntensity', color='b', marker=r'x')
		plt.xlim([0, 100000])
		plt.ylim([0, 250])
		legendTxt = ['curFull', 'refFull', 'curFace(ref=130), RMSE='+str(int(np.sqrt(np.mean((resultDic[key]['faceData'][:,1]-130)**2))))]
		plt.legend(legendTxt)
		plt.ylabel('Intensity')
		plt.xlabel('Time (sec)')
		plt.title('Original Auto-exposure')

plt.suptitle('Auto-exposure Comparison')
plt.show()
#plt.scatter(range(len(imgKeys)), errorBuf[:,j], label=m, alpha=1, marker=r'x', color=next(colors))
#plt.subplot(212)
'''
with open('result.pkl', 'rb') as f:
	resultDic = pickle.load(f)
'''