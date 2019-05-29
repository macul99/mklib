# show points
import pickle
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys

with open('results.pkl', 'rb') as f:
	resultDic = pickle.load(f)

datasetDic = resultDic['datasetDic']
outputDic = resultDic['outputDic']
modelList = resultDic['modelList']
dsKeys = datasetDic.keys()
dsKeys.sort()

modelName = ''
while modelName not in modelList:
	print "Please select the model: ", modelList
	print "Key in 'q' to exit!"

	modelName = raw_input()

	if modelName == 'q':
		raise Exception('exit')

	if modelName in modelList:
		print modelName, ' is selected!'
		break
	else:
		print "model selected is invalid, please try again, or press 'q' to quit"

dsName = ''
while dsName not in dsKeys:
	print "Please select the dataset: ", dsKeys
	print "Key in 'q' to exit!"

	dsName = raw_input()

	if dsName == 'q':
		raise Exception('exit')

	if dsName in dsKeys:
		print dsName, ' is selected!'
		break
	else:
		print "dataset selected is invalid, please try again, or press 'q' to quit"

selDataDic = datasetDic[dsName]
selOutputDic = outputDic[dsName][modelList.index(modelName)]

print "Key in ']' to exit!"
selDataDic.showPoints(num_points=5, predicted=selOutputDic)