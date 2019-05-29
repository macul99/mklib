# train myMtcnn net
import numpy as np 
import caffe
import myNet
import os
from os.path import isfile,isdir,join
from shutil import copyfile


def trainMyNet(testCase='1'):
	solverFile='mySolver.prototxt'
	if testCase == '1':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=10000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=9
		configDic['filePostfix']='"_celebA_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as testCase 1 except using celebA_cropOnly1 data. For this data, image is changed from RGB to BGR
	if testCase == '1_1':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=10000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=9
		configDic['filePostfix']='"_celebA_cropOnly1"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as testCase 1_2, retry to double confirm the performance
	if testCase == '1_2':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=10000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=9
		configDic['filePostfix']='"_celebA_cropOnly1"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# use batch size 256 instead of 64 compared with testcase 1
	if testCase == '2':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=10000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=9
		configDic['filePostfix']='"_celebA_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# learn all parameter based on tained weightings from testcase 2
	if testCase == '3':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_learnall.prototxt'
		solverDic['base_lr']=1e-9
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=9
		configDic['filePostfix']='"_celebA_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# use celebA_all as training set
	if testCase == '4':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# use celebA_all as training set and regLoss layer with ohem
	if testCase == '5':	
		solverDic={}
		solverDic['train_net']='fc1024_ohem_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_hard_Layer"
		configDic['ohemRatio']=0.7

		myNet.createNet(configDic, myNet.net_fc1024_ohem)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024_ohem)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# use celebA_all as training set and regLoss layer with ohem and relpos input layer
	if testCase == '6':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc1024_ohem_relpos_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data1_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_hard_Layer"
		configDic['ohemRatio']=0.7
		net_fc1024_ohem_relpos=1

		myNet.createNet(configDic, myNet.net_fc1024_ohem_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024_ohem_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# use celebA_all as training set and regLoss layer and relpos input layer
	if testCase == '7':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc1024_relpos_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data1_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7
		configDic['lossweight1']=0.2

		myNet.createNet(configDic, myNet.net_fc1024_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# use celebA_all as training set and add one more 256fc layer, freeze original 256fc layer
	if testCase == '8':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256x2_frz1st256fc.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7

		myNet.createNet(configDic, myNet.net_fc256x2_frz1) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256x2_frz1) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# use celebA_all as training set and add one more 256fc layer, don't freeze both 256fc layer
	if testCase == '9':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256x2.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7

		myNet.createNet(configDic, myNet.net_fc256x2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256x2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# use celebA_all as training set and make two new 256fc layer, don't freeze both 256fc layer
	if testCase == '10':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256newx2.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7

		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as testcase 10 except change stepsize to 30000
	if testCase == '11':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256newx2.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=30000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7

		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as testcase 11 except change base_lr to 1e-5
	if testCase == '12':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256newx2.prototxt'
		solverDic['base_lr']=1e-5
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=30000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7

		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as testcase 11 except change base_lr to 1e-4
	if testCase == '13':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256newx2.prototxt'
		solverDic['base_lr']=1e-4
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=30000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=10000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7

		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as testcase 11 except change base_lr to 1e-3
	if testCase == '14':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256newx2.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=30000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=10000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7

		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as testcase 11 except change base_lr to 1e-2
	if testCase == '15':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256newx2.prototxt'
		solverDic['base_lr']=1e-2
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=30000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7

		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# continue with testcase 14 iter 200000 with base_lr of 1e-8 and learnAll
	if testCase == '16':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256newx2.prototxt'
		solverDic['base_lr']=1e-8
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=30000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7

		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256newx2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# use celebA_all as training set and regLoss layer and relpos1 input layer and fc256newx2
	if testCase == '20':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256newx2_relpos1_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=30000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=10000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data2_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7
		configDic['lossweight1']=0.2

		myNet.createNet(configDic, myNet.net_fc256newx2_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256newx2_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as case 20 but use lossweight1 of 1
	if testCase == '21':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256newx2_relpos1_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=30000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=10000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data2_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7
		configDic['lossweight1']=1

		myNet.createNet(configDic, myNet.net_fc256newx2_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256newx2_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# based on case 20 iter 800000, learnAll, base_lr=1e-3
	if testCase == '22':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256newx2_relpos1_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=30000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=10000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data2_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7
		configDic['lossweight1']=0.2

		myNet.createNet(configDic, myNet.net_fc256newx2_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256newx2_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# based on case 20 iter 800000, learnAll, base_lr=1e-6
	if testCase == '23':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256newx2_relpos1_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=30000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=10000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data2_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7
		configDic['lossweight1']=0.2

		myNet.createNet(configDic, myNet.net_fc256newx2_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256newx2_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# train 3 loss net, use 3 class net, the last class indicate the instances with the last class will not take part in loss calculation
	if testCase == '30':	
		solverDic={}
		solverDic['train_net']='3loss_3cls_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data4_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=9
		configDic['filePostfix']='"_celebA_cropOnly1"'

		configDic['sampleRatio'] = '"[3,1,1,2]"'
		configDic['wideFaceDataFile'] = '"/home/macul/Projects/300W/widerFace_train_"'
		configDic['numOfWideFaceFile'] = '"[65, 30, 24]"'
		
		myNet.createNet(configDic, myNet.net_3loss_3cls) # modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_3loss_3cls) # modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# train 3 loss net same as original net with freeze most of the layers
	if testCase == '31':	
		solverDic={}
		solverDic['train_net']='3loss_org_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data5_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=9
		configDic['filePostfix']='"_celebA_cropOnly1"'

		configDic['sampleRatio'] = '"[0,0,0,2]"'
		configDic['wideFaceDataFile'] = '"/home/macul/Projects/300W/widerFace_train_"'
		configDic['numOfWideFaceFile'] = '"[65, 30, 24]"'

		configDic['filterLayer'] = "clsFilter_Layer"
		
		myNet.createNet(configDic, myNet.net_3loss_org) # modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_3loss_org) # modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as tc 31, but change fc5 to 1024
	if testCase == '32':	
		solverDic={}
		solverDic['train_net']='3loss_fc1024_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=100000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data6_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=9
		configDic['filePostfix']='"_celebA_cropOnly1"'

		configDic['sampleRatio'] = '"[3,1,1,2]"'
		configDic['wideFaceDataFile'] = '"/home/macul/Projects/300W/widerFace_train_"'
		configDic['numOfWideFaceFile'] = '"[65, 30, 24]"'

		configDic['filterLayer'] = "clsFilter_Layer"
		
		myNet.createNet(configDic, myNet.net_3loss_fc1024) # modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_3loss_fc1024) # modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as testcase 2 but use base_lr=1e-3 and use _celebA_zf_cropOnly dataset
	if testCase == '40':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as testcase 40 and use newly generated _celebA_zf_cropOnly dataset (bug fixed)
	if testCase == '40_1':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# same as testcase 40_1 and use base_lr=1e-6
	if testCase == '40_2':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_freeze.prototxt'
		solverDic['base_lr']=1e-6
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# same as testcase 40_1 but use Data_Flip_Layer_train
	if testCase == '40_3':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Flip_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# same as testcase 40_1 and use newly generated _celebA_zf_cropOnly dataset with 0.5 points adjustment
	if testCase == '40_4':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# same as testcase 40_4 but use Data_Flip_Layer_train
	if testCase == '40_5':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Flip_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# same as testcase 40_4 but use fc256 layer and train from scratch
	if testCase == '40_6':	
		solverDic={}
		solverDic['train_net']='fc256_layer_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc256)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# fine tune original net, use Adam, base_lr=1e-7
	if testCase == '40_7':	
		solverDic={}
		solverDic['train_net']='fc256_layer_learnall.prototxt'
		solverDic['base_lr']=1e-7
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc256_org)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256_org)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# fine tune original net, use Adam, base_lr=1e-8
	if testCase == '40_8':	
		solverDic={}
		solverDic['train_net']='fc256_layer_learnall.prototxt'
		solverDic['base_lr']=1e-8
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc256_org)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256_org)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# same as tc 40_8, use SGD, base_lr=1e-9
	if testCase == '40_8_1':	
		solverDic={}
		solverDic['train_net']='fc256_layer_learnall.prototxt'
		solverDic['base_lr']=1e-9
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="SGD" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc256_org)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256_org)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# continue with tc 40_8 with Data_Flip_Layer_train, use Adam, base_lr=1e-10
	if testCase == '40_8_2':	
		solverDic={}
		solverDic['train_net']='fc256_layer_learnall.prototxt'
		solverDic['base_lr']=1e-10
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Flip_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc256_org)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256_org)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# continue with tc 40_8 with relpos feature with learnall, use Adam, base_lr=1e-8
	if testCase == '40_8_3':	
		solverDic={}
		solverDic['train_net']='fc256_org_relpos.prototxt'
		solverDic['base_lr']=1e-8
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data1_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'
		configDic['lossweight1']=0.2

		myNet.createNet(configDic, myNet.net_fc256_org_relpos)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256_org_relpos)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# same as tc 40_8_3 with relpos feature with freeze, use Adam, base_lr=1e-3, lossweight1=1
	if testCase == '40_8_4':	
		solverDic={}
		solverDic['train_net']='fc256_org_relpos.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data1_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'
		configDic['lossweight1']=1

		myNet.createNet(configDic, myNet.net_fc256_org_relpos)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256_org_relpos)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# fine tune original net, use Adam, base_lr=1e-9
	if testCase == '40_9':	
		solverDic={}
		solverDic['train_net']='fc256_layer_learnall.prototxt'
		solverDic['base_lr']=1e-9
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=3000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc256_org)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256_org)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))		

	# learn all parameter based on tained weightings from testcase 40 and use base_lr=1e-9
	if testCase == '41':	
		solverDic={}
		solverDic['train_net']='fc1024_layer_learnall.prototxt'
		solverDic['base_lr']=1e-9
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024)
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# add one more 256fc layer, don't freeze both 256fc layer
	if testCase == '42':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc256x2.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'

		myNet.createNet(configDic, myNet.net_fc256x2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc256x2) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as testcase 7 but use celebA_zf_cropOnly as training and base_lr=1e-3
	if testCase == '43':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc1024_relpos_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data1_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7
		configDic['lossweight1']=0.2

		myNet.createNet(configDic, myNet.net_fc1024_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as testcase 7 but use celebA_zf_cropOnly as training and base_lr=1e-3
	if testCase == '43_1':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc1024_relpos_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="Adam" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data1_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7
		configDic['lossweight1']=0.5

		myNet.createNet(configDic, myNet.net_fc1024_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as tc 43 but use SGD solver
	if testCase == '44':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc1024_relpos_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="SGD" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data1_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7
		configDic['lossweight1']=0.2

		myNet.createNet(configDic, myNet.net_fc1024_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

	# same as tc 43 but use SGD solver
	if testCase == '44_1':
		### modify here	
		solverDic={}
		solverDic['train_net']='fc1024_relpos_freeze.prototxt'
		solverDic['base_lr']=1e-3
		solverDic['momentum']=0.9
		solverDic['weight_decay']=0.004
		solverDic['lr_policy']="step"
		solverDic['stepsize']=10000
		solverDic['gamma']=0.8
		solverDic['display']=500
		solverDic['max_iter']=1000000
		solverDic['snapshot']=10000
		solverDic['snapshot_prefix']='./snapshot_'+testCase+'/'
		solverDic['type']="SGD" # SGD, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp

		if isdir(solverDic['snapshot_prefix']):
			assert False, "The folder: " + solverDic['snapshot_prefix'] + ", already exists!"
		else:
			os.mkdir(solverDic['snapshot_prefix'])

		with open(solverFile,'w') as f:
			for k in solverDic.keys():
				if type(solverDic[k])==type(''):
					f.write(k+':\"'+solverDic[k]+'\"\n')
				else:
					f.write(k+':'+str(solverDic[k])+'\n')

		copyfile(solverFile, join(solverDic['snapshot_prefix'],solverFile))

		### modify here
		configDic={}
		configDic['netName']=solverDic['train_net'].split('.')[0]		
		configDic['phase']='train'
		configDic['learnAll']=False
		configDic['batchSize']=256
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data1_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=8
		configDic['filePostfix']='"_celebA_zf_cropOnly"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7
		configDic['lossweight1']=0.5

		myNet.createNet(configDic, myNet.net_fc1024_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

		configDic['netName']=configDic['netName']+'_test'
		configDic['phase']='test'
		myNet.createNet(configDic, myNet.net_fc1024_relpos) ### modify here
		copyfile(configDic['netName']+'.prototxt', join(solverDic['snapshot_prefix'],configDic['netName']+'.prototxt'))

if __name__ == '__main__':
	import sys
	#print(sys.argv) # Note the first argument is always the script filename.

	# OpenCL may be enabled by default in OpenCV3; disable it because it's not
	# thread safe and causes unwanted GPU memory allocations.
	#import cv2
	#cv2.ocl.setUseOpenCL(False)

	if len(sys.argv)==2:
		trainMyNet(testCase=sys.argv[1])
	