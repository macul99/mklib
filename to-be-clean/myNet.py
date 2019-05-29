# define my lenet()
from pylab import * # use pylab import for numpy
import caffe
import string

caffe_root = '/home/macul/Libraries/py-faster-rcnn/caffe-fast-rcnn/'

from caffe import layers as L, params as P 

def createNet(configDic, funCallback):
	file = configDic['netName']+'.prototxt'
	if string.lower(configDic['phase'])=='train':
		pass
	elif string.lower(configDic['phase'])=='test':
		pass
	else:
		assert False, "The phase can only be 'train' or 'test'"

	with open(file, 'w') as f:
		f.write('name: \"'+configDic['netName']+'\"\n')
			#f.write('####################################\n')
		f.write(str(funCallback(configDic)))
    	print(str(funCallback(configDic)))


#original net with label, bbox and landmark training
def net_3loss_3cls(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data4_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'

		configDic['sampleRatio'] = '"[3,1,1,2]"'

		configDic['wideFaceDataFile'] = '"/home/macul/Projects/300W/widerFace_train_"'
		configDic['numOfWideFaceFile'] = '"[13, 5, 4]"'

	phase = configDic['phase']
	learnAll = configDic['learnAll']

	if learnAll:
		paramDic = [dict(lr_mult=1, decay_mult=1), 
					dict(lr_mult=2, decay_mult=1)]
	else:
		paramDic = [dict(lr_mult=0, decay_mult=0), 
					dict(lr_mult=0, decay_mult=0)]

	n = caffe.NetSpec()

	#n.data, n.label = L.Data(ntop=2)
	if phase=='train':
		n.data, n.points, n.bbox, n.label, n.pts_mask, n.bbox_mask = L.Python( 	ntop=6,
																				python_param=dict(	module=configDic['module'], 
																  									layer=configDic['dataLayer'],
																								  	param_str='{' 
																								  				+ '"batchSize":' + str(configDic['batchSize']) + ','
																												+ '"faceDataFile":' + configDic['faceDataFile'] + ','
																												+ '"numOfFile":' + str(configDic['numOfFile']) + ','
																												+ '"filePostfix":' + configDic['filePostfix'] + ','
																												+ '"sampleRatio":' + configDic['sampleRatio'] + ','
																												+ '"wideFaceDataFile":' + configDic['wideFaceDataFile'] + ','
																												+ '"numOfWideFaceFile":' + configDic['numOfWideFaceFile'] +
																											  '}' ),
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
								name='conv5',
								param=paramDic,
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv5   = L.Dropout(		n.conv5, 
								name='drop5',
								dropout_ratio=0.5,
								in_place=True )	
	n.conv5   = L.PReLU(		n.conv5, name='prelu5', in_place=True)
	n.conv6_1 = L.InnerProduct(	n.conv5, 
								name='conv6-11',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=3,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv6_2 = L.InnerProduct(	n.conv5, 
								name='conv6-2',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=4,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))
	n.conv6_3 = L.InnerProduct(	n.conv5, 
								name='conv6-3',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=10,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	
	if phase=='train':
		#n.conv6_1m = L.Eltwise(		n.conv6_1, n.lbl_mask,
		#						name='conv6-1m',
		#						operation=P.Eltwise.PROD )
		n.clsLoss = L.SoftmaxWithLoss(	n.conv6_1, n.label, 
										loss_weight=1, 
										loss_param=dict(ignore_label=2) ) # will not calculate loss if the label equals to the value specified by ignore_label (class number)

		n.conv6_2m = L.Eltwise(		n.conv6_2, n.bbox_mask,
								name='conv6-2m',
								operation=P.Eltwise.PROD )
		n.bboxLoss = L.EuclideanLoss(n.conv6_2m, n.bbox, loss_weight=0.5)

		n.conv6_3m = L.Eltwise(		n.conv6_3, n.pts_mask,
								name='conv6-3m',
								operation=P.Eltwise.PROD )
		n.ptsLoss = L.EuclideanLoss(n.conv6_3m, n.points, loss_weight=1)
	else:
		n.prob1   = L.Softmax(n.conv6_1)

	#n.__setattr__('conv1', n.conv1) # no use, since there are multiple n.conv1 in the net
	#n.__setattr__('conv2', n.conv2) # no use, since there are multiple n.conv2 in the net
	#n.__setattr__('conv3', n.conv3) # no use, since there are multiple n.conv3 in the net
	#n.__setattr__('conv4', n.conv4) # no use, since there are multiple n.conv4 in the net
	#n.__setattr__('conv5', n.conv5) # no use, since there are multiple n.conv5 in the net
	n.__setattr__('conv6-1', n.conv6_1) # change top name of conv6_1 to conv6-1
	n.__setattr__('conv6-2', n.conv6_2) # change top name of conv6_2 to conv6-2
	n.__setattr__('conv6-3', n.conv6_3) # change top name of conv6_3 to conv6-3
#	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
#	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#	n.fc1 = L.InnerProduct(n.pool2, num_output=500,weight_filler=dict(type='xavier'))
#	n.relu1 = L.ReLU(n.fc1, in_place=True)
#	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
#	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()


#original net with label, bbox and landmark training
def net_3loss_org(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data5_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'

		configDic['sampleRatio'] = '"[3,1,1,2]"'

		configDic['wideFaceDataFile'] = '"/home/macul/Projects/300W/widerFace_train_"'
		configDic['numOfWideFaceFile'] = '"[13, 5, 4]"'

		configDic['filterLayer'] = "clsFilter_Layer"

	phase = configDic['phase']
	learnAll = configDic['learnAll']

	if learnAll:
		paramDic = [dict(lr_mult=1, decay_mult=1), 
					dict(lr_mult=2, decay_mult=1)]
	else:
		paramDic = [dict(lr_mult=0, decay_mult=0), 
					dict(lr_mult=0, decay_mult=0)]

	n = caffe.NetSpec()

	#n.data, n.label = L.Data(ntop=2)
	if phase=='train':
		n.data, n.points, n.bbox, n.label, n.pts_mask, n.bbox_mask, n.lbl_mask = L.Python( 	ntop=7,
																							python_param=dict(	module=configDic['module'], 
																			  									layer=configDic['dataLayer'],
																											  	param_str='{' 
																											  				+ '"batchSize":' + str(configDic['batchSize']) + ','
																															+ '"faceDataFile":' + configDic['faceDataFile'] + ','
																															+ '"numOfFile":' + str(configDic['numOfFile']) + ','
																															+ '"filePostfix":' + configDic['filePostfix'] + ','
																															+ '"sampleRatio":' + configDic['sampleRatio'] + ','
																															+ '"wideFaceDataFile":' + configDic['wideFaceDataFile'] + ','
																															+ '"numOfWideFaceFile":' + configDic['numOfWideFaceFile'] +
																														  '}' ),
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
								name='conv5',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv5   = L.Dropout(		n.conv5, 
								name='drop5',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv5   = L.PReLU(		n.conv5, name='prelu5', in_place=True)
	n.conv6_1 = L.InnerProduct(	n.conv5, 
								name='conv6-1',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=2,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv6_2 = L.InnerProduct(	n.conv5, 
								name='conv6-2',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=4,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))
	n.conv6_3 = L.InnerProduct(	n.conv5, 
								name='conv6-3',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=10,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	
	if phase=='train':
		#n.conv6_1m = L.Eltwise(		n.conv6_1, n.lbl_mask,
		#						name='conv6-1m',
		#						operation=P.Eltwise.PROD )
		n.conv6_1m = L.Python( 	n.conv6_1, n.lbl_mask,
								ntop=1,
								python_param=dict(  module=configDic['module'], 
													layer=configDic['filterLayer']
														 ) )
		n.clsLoss = L.SoftmaxWithLoss(	n.conv6_1m, n.label, 
										loss_weight=1, 
										loss_param=dict(ignore_label=2) ) # will not calculate loss if the label equals to the value specified by ignore_label (class number)

		n.conv6_2m = L.Eltwise(		n.conv6_2, n.bbox_mask,
								name='conv6-2m',
								operation=P.Eltwise.PROD )
		n.bboxLoss = L.EuclideanLoss(n.conv6_2m, n.bbox, loss_weight=1)

		#n.conv6_3m = L.Eltwise(		n.conv6_3, n.pts_mask,
		#						name='conv6-3m',
		#						operation=P.Eltwise.PROD )
		n.ptsLoss = L.EuclideanLoss(n.conv6_3, n.points, loss_weight=1)
	else:
		n.prob1   = L.Softmax(n.conv6_1)

	#n.__setattr__('conv1', n.conv1) # no use, since there are multiple n.conv1 in the net
	#n.__setattr__('conv2', n.conv2) # no use, since there are multiple n.conv2 in the net
	#n.__setattr__('conv3', n.conv3) # no use, since there are multiple n.conv3 in the net
	#n.__setattr__('conv4', n.conv4) # no use, since there are multiple n.conv4 in the net
	#n.__setattr__('conv5', n.conv5) # no use, since there are multiple n.conv5 in the net
	n.__setattr__('conv6-1', n.conv6_1) # change top name of conv6_1 to conv6-1
	n.__setattr__('conv6-2', n.conv6_2) # change top name of conv6_2 to conv6-2
	n.__setattr__('conv6-3', n.conv6_3) # change top name of conv6_3 to conv6-3
#	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
#	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#	n.fc1 = L.InnerProduct(n.pool2, num_output=500,weight_filler=dict(type='xavier'))
#	n.relu1 = L.ReLU(n.fc1, in_place=True)
#	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
#	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()

#original net with label, bbox and landmark training
def net_3loss_fc1024(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data6_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'

		configDic['sampleRatio'] = '"[3,1,1,2]"'

		configDic['wideFaceDataFile'] = '"/home/macul/Projects/300W/widerFace_train_"'
		configDic['numOfWideFaceFile'] = '"[13, 5, 4]"'

		configDic['filterLayer'] = "clsFilter_Layer"

	phase = configDic['phase']
	learnAll = configDic['learnAll']

	if learnAll:
		paramDic = [dict(lr_mult=1, decay_mult=1), 
					dict(lr_mult=2, decay_mult=1)]
	else:
		paramDic = [dict(lr_mult=0, decay_mult=0), 
					dict(lr_mult=0, decay_mult=0)]

	n = caffe.NetSpec()

	#n.data, n.label = L.Data(ntop=2)
	if phase=='train':
		n.data, n.points, n.bbox, n.label, n.pts_mask, n.bbox_mask, n.lbl_mask = L.Python( 	ntop=7,
																							python_param=dict(	module=configDic['module'], 
																			  									layer=configDic['dataLayer'],
																											  	param_str='{' 
																											  				+ '"batchSize":' + str(configDic['batchSize']) + ','
																															+ '"faceDataFile":' + configDic['faceDataFile'] + ','
																															+ '"numOfFile":' + str(configDic['numOfFile']) + ','
																															+ '"filePostfix":' + configDic['filePostfix'] + ','
																															+ '"sampleRatio":' + configDic['sampleRatio'] + ','
																															+ '"wideFaceDataFile":' + configDic['wideFaceDataFile'] + ','
																															+ '"numOfWideFaceFile":' + configDic['numOfWideFaceFile'] +
																														  '}' ),
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
								name='drop5-1',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv5   = L.PReLU(		n.conv5, name='prelu5-1', in_place=True)
	n.conv6_1 = L.InnerProduct(	n.conv5, 
								name='conv6-11',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=2,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv6_2 = L.InnerProduct(	n.conv5, 
								name='conv6-21',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=4,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))
	n.conv6_3 = L.InnerProduct(	n.conv5, 
								name='conv6-31',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=10,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	
	if phase=='train':
		#n.conv6_1m = L.Eltwise(		n.conv6_1, n.lbl_mask,
		#						name='conv6-1m',
		#						operation=P.Eltwise.PROD )
		n.conv6_1m = L.Python( 	n.conv6_1, n.lbl_mask,
								ntop=1,
								python_param=dict(  module=configDic['module'], 
													layer=configDic['filterLayer']
														 ) )
		n.clsLoss = L.SoftmaxWithLoss(	n.conv6_1m, n.label, 
										loss_weight=1 )

		n.conv6_2m = L.Eltwise(	n.conv6_2, n.bbox_mask,
								name='conv6-2m',
								operation=P.Eltwise.PROD )
		n.bboxLoss = L.EuclideanLoss(n.conv6_2m, n.bbox, loss_weight=0.5)

		n.conv6_3m = L.Eltwise(	n.conv6_3, n.pts_mask,
								name='conv6-3m',
								operation=P.Eltwise.PROD )
		n.ptsLoss = L.EuclideanLoss(n.conv6_3m, n.points, loss_weight=1)
	else:
		n.prob1   = L.Softmax(n.conv6_1)

	#n.__setattr__('conv1', n.conv1) # no use, since there are multiple n.conv1 in the net
	#n.__setattr__('conv2', n.conv2) # no use, since there are multiple n.conv2 in the net
	#n.__setattr__('conv3', n.conv3) # no use, since there are multiple n.conv3 in the net
	#n.__setattr__('conv4', n.conv4) # no use, since there are multiple n.conv4 in the net
	#n.__setattr__('conv5', n.conv5) # no use, since there are multiple n.conv5 in the net
	n.__setattr__('conv6-1', n.conv6_1) # change top name of conv6_1 to conv6-1
	n.__setattr__('conv6-2', n.conv6_2) # change top name of conv6_2 to conv6-2
	n.__setattr__('conv6-3', n.conv6_3) # change top name of conv6_3 to conv6-3
#	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
#	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#	n.fc1 = L.InnerProduct(n.pool2, num_output=500,weight_filler=dict(type='xavier'))
#	n.relu1 = L.ReLU(n.fc1, in_place=True)
#	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
#	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()


#change 256fc layer to 1024fc	
def net_fc1024(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'

	phase = configDic['phase']
	learnAll = configDic['learnAll']

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
										python_param=dict(module=configDic['module'], 
														  layer=configDic['dataLayer'],
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

#original net	
def net_fc256_org(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'

	phase = configDic['phase']
	learnAll = configDic['learnAll']

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
										python_param=dict(module=configDic['module'], 
														  layer=configDic['dataLayer'],
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
								param=paramDic,
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

def net_fc256_org_relpos(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data1_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossweight1']=0.2

	phase = configDic['phase']
	learnAll = configDic['learnAll']

	if learnAll:
		paramDic = [dict(lr_mult=1, decay_mult=1), 
					dict(lr_mult=2, decay_mult=1)]
	else:
		paramDic = [dict(lr_mult=0, decay_mult=0), 
					dict(lr_mult=0, decay_mult=0)]

	n = caffe.NetSpec()

	#n.data, n.label = L.Data(ntop=2)
	if phase=='train':
		n.data, n.points, n.relativePos = L.Python( 	ntop=3,
														python_param=dict(module=configDic['module'], 
														layer=configDic['dataLayer'],
														param_str='{' 
														  			+ '"batchSize":' + str(configDic['batchSize']) + ','
																	+ '"faceDataFile":' + configDic['faceDataFile'] + ','
																	+ '"numOfFile":' + str(configDic['numOfFile']) + ','
																	+ '"filePostfix":' + configDic['filePostfix'] +
																'}' ),
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
								param=paramDic,
								num_output=10,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv6_4 = L.InnerProduct(	n.conv5, 
								name='conv6-4',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=8,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	

	if phase=='train':
		n.regLoss = L.EuclideanLoss(n.conv6_3, n.points)
		
		n.regLoss1 = L.EuclideanLoss(	n.conv6_4, n.relativePos,
										loss_weight=configDic['lossweight1'])

	#n.__setattr__('conv1', n.conv1) # no use, since there are multiple n.conv1 in the net
	#n.__setattr__('conv2', n.conv2) # no use, since there are multiple n.conv2 in the net
	#n.__setattr__('conv3', n.conv3) # no use, since there are multiple n.conv3 in the net
	#n.__setattr__('conv4', n.conv4) # no use, since there are multiple n.conv4 in the net
	#n.__setattr__('conv5', n.conv5) # no use, since there are multiple n.conv5 in the net
	n.__setattr__('conv6-3', n.conv6_3) # change top name of conv6_3 to conv6-3
	n.__setattr__('conv6-4', n.conv6_4) # change top name of conv6_3 to conv6-3
#	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
#	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#	n.fc1 = L.InnerProduct(n.pool2, num_output=500,weight_filler=dict(type='xavier'))
#	n.relu1 = L.ReLU(n.fc1, in_place=True)
#	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
#	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()

#train fc256 layer from scratch	
def net_fc256(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'

	phase = configDic['phase']
	learnAll = configDic['learnAll']

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
										python_param=dict(module=configDic['module'], 
														  layer=configDic['dataLayer'],
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
								num_output=256,
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

#add one more 256fc layer
#if freeze, the first 256fc is freezed as well
def net_fc256x2_frz1(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'

	phase = configDic['phase']
	learnAll = configDic['learnAll']

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
										python_param=dict(module=configDic['module'], 
														  layer=configDic['dataLayer'],
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
								param=paramDic,
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv5   = L.Dropout(		n.conv5, 
								name='drop5',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv5   = L.PReLU(		n.conv5, name='prelu5-1', in_place=True)
	n.conv51  = L.InnerProduct(	n.conv5, 
								name='conv5-11',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv51  = L.Dropout(		n.conv51, 
								name='drop5-1',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv51  = L.PReLU(		n.conv51, name='prelu5-11', in_place=True)
	n.conv6_3 = L.InnerProduct(	n.conv51, 
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

#add one more 256fc layer
#if freeze, both 256fc are not freezed
def net_fc256x2(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'

	phase = configDic['phase']
	learnAll = configDic['learnAll']

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
										python_param=dict(module=configDic['module'], 
														  layer=configDic['dataLayer'],
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
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv5   = L.Dropout(		n.conv5, 
								name='drop5',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv5   = L.PReLU(		n.conv5, name='prelu5-1', in_place=True)
	n.conv51  = L.InnerProduct(	n.conv5, 
								name='conv5-11',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv51  = L.Dropout(		n.conv51, 
								name='drop5-1',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv51  = L.PReLU(		n.conv51, name='prelu5-11', in_place=True)
	n.conv6_3 = L.InnerProduct(	n.conv51, 
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

#make two new 256fc layer
#both 256fc are not freezed
def net_fc256newx2(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'

	phase = configDic['phase']
	learnAll = configDic['learnAll']

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
										python_param=dict(module=configDic['module'], 
														  layer=configDic['dataLayer'],
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
								name='conv5-10',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv5   = L.Dropout(		n.conv5, 
								name='drop5-0',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv5   = L.PReLU(		n.conv5, name='prelu5-10', in_place=True)
	n.conv51  = L.InnerProduct(	n.conv5, 
								name='conv5-11',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv51  = L.Dropout(		n.conv51, 
								name='drop5-1',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv51  = L.PReLU(		n.conv51, name='prelu5-11', in_place=True)
	n.conv6_3 = L.InnerProduct(	n.conv51, 
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

#make two new 256fc layer
#change input data layer to add relative position
def net_fc256newx2_relpos(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data1_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7
		configDic['lossweight1']=0.2

	phase = configDic['phase']
	learnAll = configDic['learnAll']

	if learnAll:
		paramDic = [dict(lr_mult=1, decay_mult=1), 
					dict(lr_mult=2, decay_mult=1)]
	else:
		paramDic = [dict(lr_mult=0, decay_mult=0), 
					dict(lr_mult=0, decay_mult=0)]

	n = caffe.NetSpec()

	#n.data, n.label = L.Data(ntop=2)
	if phase=='train':
		n.data, n.points, n.relativePos = L.Python( 	ntop=3,
														python_param=dict(module=configDic['module'], 
														layer=configDic['dataLayer'],
														param_str='{' 
														  			+ '"batchSize":' + str(configDic['batchSize']) + ','
																	+ '"faceDataFile":' + configDic['faceDataFile'] + ','
																	+ '"numOfFile":' + str(configDic['numOfFile']) + ','
																	+ '"filePostfix":' + configDic['filePostfix'] +
																'}' ),
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
								name='conv5-10',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv5   = L.Dropout(		n.conv5, 
								name='drop5-0',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv5   = L.PReLU(		n.conv5, name='prelu5-10', in_place=True)
	n.conv51  = L.InnerProduct(	n.conv5, 
								name='conv5-11',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=256,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv51  = L.Dropout(		n.conv51, 
								name='drop5-1',
								dropout_ratio=0.25,
								in_place=True )	
	n.conv51  = L.PReLU(		n.conv51, name='prelu5-11', in_place=True)
	n.conv6_3 = L.InnerProduct(	n.conv51, 
								name='conv6-31',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=10,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	n.conv6_4 = L.InnerProduct(	n.conv51, 
								name='conv6-4',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=8,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	

	if phase=='train':
		n.regLoss = L.Python( 	n.conv6_3, n.points,
								ntop=1,
								propagate_down=[1,0],
								loss_weight=1,
								python_param=dict(  module=configDic['module'], 
													layer=configDic['lossLayer']
														 ) )
		n.regLoss1 = L.Python( 	n.conv6_4, n.relativePos,
								ntop=1,
								propagate_down=[1,0],
								loss_weight=configDic['lossweight1'],
								python_param=dict(  module=configDic['module'], 
													layer=configDic['lossLayer']
														 ) )

	#n.__setattr__('conv1', n.conv1) # no use, since there are multiple n.conv1 in the net
	#n.__setattr__('conv2', n.conv2) # no use, since there are multiple n.conv2 in the net
	#n.__setattr__('conv3', n.conv3) # no use, since there are multiple n.conv3 in the net
	#n.__setattr__('conv4', n.conv4) # no use, since there are multiple n.conv4 in the net
	#n.__setattr__('conv5', n.conv5) # no use, since there are multiple n.conv5 in the net
	n.__setattr__('conv6-3', n.conv6_3) # change top name of conv6_3 to conv6-3
	n.__setattr__('conv6-4', n.conv6_4) # change top name of conv6_3 to conv6-3
#	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
#	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#	n.fc1 = L.InnerProduct(n.pool2, num_output=500,weight_filler=dict(type='xavier'))
#	n.relu1 = L.ReLU(n.fc1, in_place=True)
#	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
#	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()

#change 256fc layer to 1024fc	
#change loss layer to regression_hard_Layer
def net_fc1024_ohem(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_hard_Layer"
		configDic['ohemRatio']=0.7

	phase = configDic['phase']
	learnAll = configDic['learnAll']

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
										python_param=dict(module=configDic['module'], 
														  layer=configDic['dataLayer'],
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
		n.regLoss = L.Python( 	n.conv6_3, n.points,
								ntop=1,
								propagate_down=[1,0],
								loss_weight=1,
								python_param=dict(  module=configDic['module'], 
													layer=configDic['lossLayer'],
													param_str='{'+ '"ratio":' + str(configDic['ohemRatio']) + '}'
														 ) )

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

#change 256fc layer to 1024fc	
#change input data layer to add relative position
def net_fc1024_relpos(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data1_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_Layer"
		configDic['ohemRatio']=0.7
		configDic['lossweight1']=0.2

	phase = configDic['phase']
	learnAll = configDic['learnAll']

	if learnAll:
		paramDic = [dict(lr_mult=1, decay_mult=1), 
					dict(lr_mult=2, decay_mult=1)]
	else:
		paramDic = [dict(lr_mult=0, decay_mult=0), 
					dict(lr_mult=0, decay_mult=0)]

	n = caffe.NetSpec()

	#n.data, n.label = L.Data(ntop=2)
	if phase=='train':
		n.data, n.points, n.relativePos = L.Python( 	ntop=3,
														python_param=dict(module=configDic['module'], 
														layer=configDic['dataLayer'],
														param_str='{' 
														  			+ '"batchSize":' + str(configDic['batchSize']) + ','
																	+ '"faceDataFile":' + configDic['faceDataFile'] + ','
																	+ '"numOfFile":' + str(configDic['numOfFile']) + ','
																	+ '"filePostfix":' + configDic['filePostfix'] +
																'}' ),
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
	n.conv6_4 = L.InnerProduct(	n.conv5, 
								name='conv6-4',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=8,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	if phase=='train':
		n.regLoss = L.Python( 	n.conv6_3, n.points,
								ntop=1,
								propagate_down=[1,0],
								loss_weight=1,
								python_param=dict(  module=configDic['module'], 
													layer=configDic['lossLayer']
														 ) )
		n.regLoss1 = L.Python( 	n.conv6_4, n.relativePos,
								ntop=1,
								propagate_down=[1,0],
								loss_weight=configDic['lossweight1'],
								python_param=dict(  module=configDic['module'], 
													layer=configDic['lossLayer']
														 ) )

	#n.__setattr__('conv1', n.conv1) # no use, since there are multiple n.conv1 in the net
	#n.__setattr__('conv2', n.conv2) # no use, since there are multiple n.conv2 in the net
	#n.__setattr__('conv3', n.conv3) # no use, since there are multiple n.conv3 in the net
	#n.__setattr__('conv4', n.conv4) # no use, since there are multiple n.conv4 in the net
	#n.__setattr__('conv5', n.conv5) # no use, since there are multiple n.conv5 in the net
	n.__setattr__('conv6-3', n.conv6_3) # change top name of conv6_3 to conv6-3
	n.__setattr__('conv6-4', n.conv6_4) # change top name of conv6_3 to conv6-3
#	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
#	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#	n.fc1 = L.InnerProduct(n.pool2, num_output=500,weight_filler=dict(type='xavier'))
#	n.relu1 = L.ReLU(n.fc1, in_place=True)
#	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
#	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()

#change 256fc layer to 1024fc	
#change loss layer to regression_hard_Layer
#change input data layer to add relative position
def net_fc1024_ohem_relpos(configDic=None):

	if type(configDic)==type(None):
		configDic={}
		configDic['phase']='train'
		configDic['learnAll']=True
		configDic['batchSize']=64
		configDic['module']="myPythonLayer"
		configDic['dataLayer']="Data1_Layer_train"
		configDic['faceDataFile']='"/home/macul/Projects/300W/trainDataset_1_"'
		configDic['numOfFile']=41
		configDic['filePostfix']='"_celebA_all"'
		configDic['lossLayer']="regression_hard_Layer"
		configDic['ohemRatio']=0.7
		configDic['lossweight1']=0.2

	phase = configDic['phase']
	learnAll = configDic['learnAll']

	if learnAll:
		paramDic = [dict(lr_mult=1, decay_mult=1), 
					dict(lr_mult=2, decay_mult=1)]
	else:
		paramDic = [dict(lr_mult=0, decay_mult=0), 
					dict(lr_mult=0, decay_mult=0)]

	n = caffe.NetSpec()

	#n.data, n.label = L.Data(ntop=2)
	if phase=='train':
		n.data, n.points, n.relativePos = L.Python( 	ntop=3,
														python_param=dict(module=configDic['module'], 
														layer=configDic['dataLayer'],
														param_str='{' 
														  			+ '"batchSize":' + str(configDic['batchSize']) + ','
																	+ '"faceDataFile":' + configDic['faceDataFile'] + ','
																	+ '"numOfFile":' + str(configDic['numOfFile']) + ','
																	+ '"filePostfix":' + configDic['filePostfix'] +
																'}' ),
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
	n.conv6_4 = L.InnerProduct(	n.conv5, 
								name='conv6-4',
								param=[dict(lr_mult=1, decay_mult=1), 
									   dict(lr_mult=2, decay_mult=1)],
								num_output=8,
								weight_filler=dict(type='xavier'), 
								bias_filler=dict(type='constant', value=0))	
	if phase=='train':
		n.regLoss = L.Python( 	n.conv6_3, n.points,
								ntop=1,
								propagate_down=[1,0],
								loss_weight=1,
								python_param=dict(  module=configDic['module'], 
													layer=configDic['lossLayer'],
													param_str='{'+ '"ratio":' + str(configDic['ohemRatio']) + '}'
														 ) )
		n.regLoss1 = L.Python( 	n.conv6_4, n.relativePos,
								ntop=1,
								propagate_down=[1,0],
								loss_weight=configDic['lossweight1'],
								python_param=dict(  module=configDic['module'], 
													layer=configDic['lossLayer'],
													param_str='{'+ '"ratio":' + str(configDic['ohemRatio']) + '}'
														 ) )

	#n.__setattr__('conv1', n.conv1) # no use, since there are multiple n.conv1 in the net
	#n.__setattr__('conv2', n.conv2) # no use, since there are multiple n.conv2 in the net
	#n.__setattr__('conv3', n.conv3) # no use, since there are multiple n.conv3 in the net
	#n.__setattr__('conv4', n.conv4) # no use, since there are multiple n.conv4 in the net
	#n.__setattr__('conv5', n.conv5) # no use, since there are multiple n.conv5 in the net
	n.__setattr__('conv6-31', n.conv6_3) # change top name of conv6_3 to conv6-3
	n.__setattr__('conv6-4', n.conv6_4) # change top name of conv6_3 to conv6-3
#	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
#	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#	n.fc1 = L.InnerProduct(n.pool2, num_output=500,weight_filler=dict(type='xavier'))
#	n.relu1 = L.ReLU(n.fc1, in_place=True)
#	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
#	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()