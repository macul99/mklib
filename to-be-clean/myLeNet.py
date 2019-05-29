# define my lenet()
from pylab import * # use pylab import for numpy
import caffe
import string

caffe_root = '/home/macul/Libraries/py-faster-rcnn/caffe-fast-rcnn/'

from caffe import layers as L, params as P 

def createNet_MyDet3(netName='myMtcnn', phase='train', learnAll='true', configDic=None):
	file = netName+'.prototxt'
	if string.lower(phase)=='train':
		if string.lower(learnAll)=='true':
			file = 'myMtcnnTrain6_learnall.prototxt'
			netName = 'myMtcnn_1024fc_learnall'
		elif string.lower(learnAll)=='false':
			file = 'myMtcnnTrain6_freeze.prototxt'
			netName = 'myMtcnn_1024fc_freeze'
		else:
			assert False, "The learnAll can only be 'true' or 'false'"
	elif string.lower(phase)=='test':
		file = 'myMtcnnTest6.prototxt'
		netName = 'myMtcnnTest_1024fc'
	else:
		assert False, "The phase can only be 'train' or 'test'"

	with open(file, 'w') as f:
		f.write('name: \"'+netName+'\"\n')
			#f.write('####################################\n')
		f.write(str(lenet(phase=phase,learnAll=string.lower(learnAll)=='true')))
    	print(str(lenet(phase=phase,learnAll=string.lower(learnAll)=='true')))

#change 256fc layer to 1024fc	
def lenet(phase='train',learnAll=True,configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']="/home/macul/Projects/300W/trainDataset_1_"
		configDic['numOfFile']=41
		configDic['filePostfix']="_celebA_all"


	if learnAll:
		paramDic = [dict(lr_mult=1, decay_mult=1), 
					dict(lr_mult=2, decay_mult=1)]
	else:
		paramDic = [dict(lr_mult=0, decay_mult=0), 
					dict(lr_mult=0, decay_mult=0)]

	n = caffe.NetSpec()

	#n.data, n.label = L.Data(ntop=2)
	if phase=='train':
		n.data, n.points = L.Python( 	ntop=2,
										python_param=dict(module="myPythonLayer", 
														  layer="Data_Layer_train",
														  param_str='{' 
														  				+ '"batchSize":' + str(configDic['batchSize']) + ','
																		+ '"faceDataFile":' + configDic['faceDataFile'] + ','
																		+ '"numOfFile":' + str(configDic['numOfFile']) + ','
																		+ '"filePostfix":' + configDic['filePostfix'] +
																	 '}'
														 ),
										include=dict(phase=caffe.TRAIN) )
	else:
		n.data = L.Input(shape=dict(dim=[configDic['batchSize'], 3, 48, 48]))
	n.conv1   = L.Convolution(	n.data, 
								name = 'conv1',
								param=paramDic, 
								num_output=32, 
								kernel_size=3, 								
								stride=1,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv1   = L.PReLU(		n.conv1, name = 'prelu1', in_place=True)
	n.pool1   = L.Pooling(		n.conv1, 
								kernel_size=3, 
								stride=2, 
								pool=P.Pooling.MAX )
	n.conv2   = L.Convolution(	n.pool1, 
								name = 'conv2',
								param=paramDic, 
								num_output=64, 
								kernel_size=3, 								
								stride=1,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv2   = L.PReLU(		n.conv2, name='prelu2', in_place=True)
	n.pool2   = L.Pooling(		n.conv2, 
								kernel_size=3, 
								stride=2, 
								pool=P.Pooling.MAX )
	n.conv3   = L.Convolution(	n.pool2, 
								name = 'conv3',
								param=paramDic, 
								num_output=64,
								kernel_size=3, 								
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv3   = L.PReLU(		n.conv3, name='prelu3', in_place=True)
	n.pool3   = L.Pooling(		n.conv3, 
								kernel_size=2, 
								stride=2, 
								pool=P.Pooling.MAX )
	n.conv4   = L.Convolution(	n.pool3, 
								name='conv4',
								param=paramDic, 
								num_output=128,
								kernel_size=2, 								
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))
	n.conv4   = L.PReLU(		n.conv4, name='prelu4', in_place=True)
	n.conv5   = L.InnerProduct(	n.conv4, 
								name='conv5-1',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=1024,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv5   = L.Dropout(		n.conv5, 
								name='drop5',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv5   = L.PReLU(		n.conv5, name='prelu5-1', in_place=True)
	n.conv6_3 = L.InnerProduct(	n.conv5, 
								name='conv6-31',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=10,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	if phase=='train':
		n.regLoss = L.EuclideanLoss(n.conv6_3, n.points)

	#n.__setattr__('conv1', n.conv1) # no use, since there are multiple n.conv1 in the net
	#n.__setattr__('conv2', n.conv2) # no use, since there are multiple n.conv2 in the net
	#n.__setattr__('conv3', n.conv3) # no use, since there are multiple n.conv3 in the net
	#n.__setattr__('conv4', n.conv4) # no use, since there are multiple n.conv4 in the net
	#n.__setattr__('conv5', n.conv5) # no use, since there are multiple n.conv5 in the net
	n.__setattr__('conv6-3', n.conv6_3) # change top name of conv6_3 to conv6-3
#	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
#	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#	n.fc1 = L.InnerProduct(n.pool2, num_output=500,weight_filler=dict(type='xavier'))
#	n.relu1 = L.ReLU(n.fc1, in_place=True)
#	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
#	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()

# add new layer after conv1
def lenet5(phase='train',learnAll=True):

	defaultParamDic = [dict(lr_mult=1, decay_mult=1), 
					   dict(lr_mult=2, decay_mult=1)]
	if learnAll:
		paramDic = defaultParamDic
	else:
		paramDic = [dict(lr_mult=0, decay_mult=0), 
					dict(lr_mult=0, decay_mult=0)]

	n = caffe.NetSpec()

	#n.data, n.label = L.Data(ntop=2)
	if phase=='train':
		n.data, n.points = L.Python( 	ntop=2,
										python_param=dict(module="myPythonLayer", 
														  layer="Data_Layer_train",
														  param_str='{ \
														    			"batchSize": 64, \
																		"faceDataFile":"/home/macul/Projects/300W/trainDataset_1_", \
																		"numOfFile":41, \
																		"filePostfix":"_celebA_all" \
																	 }'),
										include=dict(phase=caffe.TRAIN) )
	else:
		n.data = L.Input(shape=dict(dim=[64, 3, 48, 48]))
	n.conv1   = L.Convolution(	n.data, 
								name = 'conv1',
								param=paramDic, 
								num_output=32, 
								kernel_size=3, 								
								stride=1,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv1   = L.PReLU(		n.conv1, name = 'prelu1', in_place=True)
	n.conv1_1 = L.Convolution(	n.conv1, 
								name = 'conv1-1',
								param=defaultParamDic, 
								num_output=32, 
								kernel_size=3, 								
								stride=1,
								pad=1,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv1_1 = L.PReLU(		n.conv1_1, name = 'prelu1-1', in_place=True)
	n.pool1   = L.Pooling(		n.conv1_1, 
								kernel_size=3, 
								stride=2, 
								pool=P.Pooling.MAX )
	n.conv2   = L.Convolution(	n.pool1, 
								name = 'conv2',
								param=defaultParamDic, 
								num_output=64, 
								kernel_size=3, 								
								stride=1,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv2   = L.PReLU(		n.conv2, name='prelu2', in_place=True)
	n.pool2   = L.Pooling(		n.conv2, 
								kernel_size=3, 
								stride=2, 
								pool=P.Pooling.MAX )
	n.conv3   = L.Convolution(	n.pool2, 
								name = 'conv3',
								param=defaultParamDic, 
								num_output=64,
								kernel_size=3, 								
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv3   = L.PReLU(		n.conv3, name='prelu3', in_place=True)
	n.pool3   = L.Pooling(		n.conv3, 
								kernel_size=2, 
								stride=2, 
								pool=P.Pooling.MAX )
	n.conv4   = L.Convolution(	n.pool3, 
								name='conv4',
								param=defaultParamDic, 
								num_output=128,
								kernel_size=2, 								
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))
	n.conv4   = L.PReLU(		n.conv4, name='prelu4', in_place=True)
	n.conv5   = L.InnerProduct(	n.conv4, 
								name='conv5',
								param=defaultParamDic, 
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv5   = L.Dropout(		n.conv5, 
								name='drop5',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv5   = L.PReLU(		n.conv5, name='prelu5', in_place=True)
	n.conv6_3 = L.InnerProduct(	n.conv5, 
								name='conv6-3',
								param=defaultParamDic,
								num_output=10,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	if phase=='train':
		n.regLoss = L.EuclideanLoss(n.conv6_3, n.points)

	#n.__setattr__('conv1', n.conv1) # no use, since there are multiple n.conv1 in the net
	#n.__setattr__('conv2', n.conv2) # no use, since there are multiple n.conv2 in the net
	#n.__setattr__('conv3', n.conv3) # no use, since there are multiple n.conv3 in the net
	#n.__setattr__('conv4', n.conv4) # no use, since there are multiple n.conv4 in the net
	#n.__setattr__('conv5', n.conv5) # no use, since there are multiple n.conv5 in the net
	n.__setattr__('conv6-3', n.conv6_3) # change top name of conv6_3 to conv6-3
#	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
#	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#	n.fc1 = L.InnerProduct(n.pool2, num_output=500,weight_filler=dict(type='xavier'))
#	n.relu1 = L.ReLU(n.fc1, in_place=True)
#	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
#	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()

#original net without clssification and bbox regression	
def lenet4(phase='train',learnAll=True):

	if learnAll:
		paramDic = [dict(lr_mult=1, decay_mult=1), 
					dict(lr_mult=2, decay_mult=1)]
	else:
		paramDic = [dict(lr_mult=0, decay_mult=0), 
					dict(lr_mult=0, decay_mult=0)]

	n = caffe.NetSpec()

	#n.data, n.label = L.Data(ntop=2)
	if phase=='train':
		n.data, n.points = L.Python( 	ntop=2,
										python_param=dict(module="myPythonLayer", 
														  layer="Data_Layer_train",
														  param_str='{ \
														    			"batchSize": 64, \
																		"faceDataFile":"/home/macul/Projects/300W/trainDataset_1_", \
																		"numOfFile":42, \
																		"filePostfix":"_celebA_cropOnly" \
																	 }'),
										include=dict(phase=caffe.TRAIN) )
	else:
		n.data = L.Input(shape=dict(dim=[64, 3, 48, 48]))
	n.conv1   = L.Convolution(	n.data, 
								name = 'conv1',
								param=paramDic, 
								num_output=32, 
								kernel_size=3, 								
								stride=1,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv1   = L.PReLU(		n.conv1, name = 'prelu1', in_place=True)
	n.pool1   = L.Pooling(		n.conv1, 
								kernel_size=3, 
								stride=2, 
								pool=P.Pooling.MAX )
	n.conv2   = L.Convolution(	n.pool1, 
								name = 'conv2',
								param=paramDic, 
								num_output=64, 
								kernel_size=3, 								
								stride=1,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv2   = L.PReLU(		n.conv2, name='prelu2', in_place=True)
	n.pool2   = L.Pooling(		n.conv2, 
								kernel_size=3, 
								stride=2, 
								pool=P.Pooling.MAX )
	n.conv3   = L.Convolution(	n.pool2, 
								name = 'conv3',
								param=paramDic, 
								num_output=64,
								kernel_size=3, 								
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv3   = L.PReLU(		n.conv3, name='prelu3', in_place=True)
	n.pool3   = L.Pooling(		n.conv3, 
								kernel_size=2, 
								stride=2, 
								pool=P.Pooling.MAX )
	n.conv4   = L.Convolution(	n.pool3, 
								name='conv4',
								param=paramDic, 
								num_output=128,
								kernel_size=2, 								
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))
	n.conv4   = L.PReLU(		n.conv4, name='prelu4', in_place=True)
	n.conv5   = L.InnerProduct(	n.conv4, 
								name='conv5',
								param=paramDic, 
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv5   = L.Dropout(		n.conv5, 
								name='drop5',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv5   = L.PReLU(		n.conv5, name='prelu5', in_place=True)
	n.conv6_3 = L.InnerProduct(	n.conv5, 
								name='conv6-3',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=10,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	if phase=='train':
		n.regLoss = L.EuclideanLoss(n.conv6_3, n.points)

	#n.__setattr__('conv1', n.conv1) # no use, since there are multiple n.conv1 in the net
	#n.__setattr__('conv2', n.conv2) # no use, since there are multiple n.conv2 in the net
	#n.__setattr__('conv3', n.conv3) # no use, since there are multiple n.conv3 in the net
	#n.__setattr__('conv4', n.conv4) # no use, since there are multiple n.conv4 in the net
	#n.__setattr__('conv5', n.conv5) # no use, since there are multiple n.conv5 in the net
	n.__setattr__('conv6-3', n.conv6_3) # change top name of conv6_3 to conv6-3
#	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
#	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#	n.fc1 = L.InnerProduct(n.pool2, num_output=500,weight_filler=dict(type='xavier'))
#	n.relu1 = L.ReLU(n.fc1, in_place=True)
#	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
#	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()

#original net without bbox regression	
def lenet1(phase='train',learnAll=True):

	if learnAll:
		paramDic = [dict(lr_mult=1, decay_mult=1), 
					dict(lr_mult=2, decay_mult=1)]
	else:
		paramDic = [dict(lr_mult=0, decay_mult=0), 
					dict(lr_mult=0, decay_mult=0)]

	n = caffe.NetSpec()

	#n.data, n.label = L.Data(ntop=2)
	if phase=='train':
		n.data, n.label, n.points = L.Python( 	ntop=3,
												python_param=dict(module="myPythonLayer", 
																  layer="Data3_Layer_train",
																  param_str='{ \
																    			"batchSize": 64, \
																				"posRatio": 0.7, \
																				"faceDataFile":"/home/macul/Projects/300W/trainDataset_1.pkl", \
																				"nonFaceDataFile":"/home/macul/Projects/300W/trainDataset_0.pkl" \
																			 }'),
												include=dict(phase=caffe.TRAIN) )
	else:
		n.data = L.Input(shape=dict(dim=[64, 3, 48, 48]))
	n.conv1   = L.Convolution(	n.data, 
								name = 'conv1',
								param=paramDic, 
								num_output=32, 
								kernel_size=3, 								
								stride=1,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv1   = L.PReLU(		n.conv1, name = 'prelu1', in_place=True)
	n.pool1   = L.Pooling(		n.conv1, 
								kernel_size=3, 
								stride=2, 
								pool=P.Pooling.MAX )
	n.conv2   = L.Convolution(	n.pool1, 
								name = 'conv2',
								param=paramDic, 
								num_output=64, 
								kernel_size=3, 								
								stride=1,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv2   = L.PReLU(		n.conv2, name='prelu2', in_place=True)
	n.pool2   = L.Pooling(		n.conv2, 
								kernel_size=3, 
								stride=2, 
								pool=P.Pooling.MAX )
	n.conv3   = L.Convolution(	n.pool2, 
								name = 'conv3',
								param=paramDic, 
								num_output=64,
								kernel_size=3, 								
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0) )
	n.conv3   = L.PReLU(		n.conv3, name='prelu3', in_place=True)
	n.pool3   = L.Pooling(		n.conv3, 
								kernel_size=2, 
								stride=2, 
								pool=P.Pooling.MAX )
	n.conv4   = L.Convolution(	n.pool3, 
								name='conv4',
								param=paramDic, 
								num_output=128,
								kernel_size=2, 								
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))
	n.conv4   = L.PReLU(		n.conv4, name='prelu4', in_place=True)
	n.conv5   = L.InnerProduct(	n.conv4, 
								name='conv5',
								param=paramDic, 
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv5   = L.Dropout(		n.conv5, 
								name='drop5',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv5   = L.PReLU(		n.conv5, name='prelu5', in_place=True)
	n.conv6_1 = L.InnerProduct(	n.conv5, 
								name='fc6-1',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=2,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))
	n.conv6_3 = L.InnerProduct(	n.conv5, 
								name='fc6-3',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=10,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.prob1   = L.Softmax(n.conv6_1)
	if phase=='train':
		n.zclsLoss = L.SoftmaxWithLoss(n.conv6_1, n.label, loss_weight=1)
		n.regLoss = L.Python(		n.conv6_3, n.points, n.label,
									ntop=1,
									propagate_down=[1,0,0],
									loss_weight=1,
									python_param=dict(module="myPythonLayer", 
													  layer="regression3_Layer") )

	#n.__setattr__('conv1', n.conv1) # no use, since there are multiple n.conv1 in the net
	#n.__setattr__('conv2', n.conv2) # no use, since there are multiple n.conv2 in the net
	#n.__setattr__('conv3', n.conv3) # no use, since there are multiple n.conv3 in the net
	#n.__setattr__('conv4', n.conv4) # no use, since there are multiple n.conv4 in the net
	#n.__setattr__('conv5', n.conv5) # no use, since there are multiple n.conv5 in the net
	n.__setattr__('conv6-1', n.conv6_1) # change top name of conv6_1 to conv6-1
	n.__setattr__('conv6-3', n.conv6_3) # change top name of conv6_3 to conv6-3
#	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
#	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#	n.fc1 = L.InnerProduct(n.pool2, num_output=500,weight_filler=dict(type='xavier'))
#	n.relu1 = L.ReLU(n.fc1, in_place=True)
#	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
#	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()

if __name__ == '__main__':
	import sys
	#print(sys.argv) # Note the first argument is always the script filename.
	if len(sys.argv)==2:
		createNet_MyDet3(phase=sys.argv[1])
	elif len(sys.argv)==3:
		createNet_MyDet3(phase=sys.argv[1], learnAll=sys.argv[2])