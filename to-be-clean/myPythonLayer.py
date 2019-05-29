import sys
#sys.path.append('/home/cmcc/caffe-master/python')
import cv2
import caffe
import numpy as np
import random
import cPickle as pickle
from os import listdir
from os.path import isfile,join
from os import walk

'''
def view_bar(num, total):
	rate = float(num) / total
	rate_num = int(rate * 100)
	r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
	sys.stdout.write(r)
	sys.stdout.flush()
'''

################################################################################
#########################Data Layer By Python###################################
################################################################################
'''
layer {
	name: "DataTrain"
	type: "Python"
	top: "data"
	top: "label"
	top: "pts"
	python_param {
	    module: "myPythonLayer"
	    layer: "Data_Layer_train"
	    param_str: '{ \
	    			 "batchSize": 64, \
	    			 "posRatio": 0.7, \
	    			 "posDataFile":"/home/macul/Projects/300W/totalDataset.pkl", \
	    			 "negDataFolder":"/home/macul/Projects/300W/labeled_data/nonFace" \
	    			}'
	}
	include {
	    phase: TRAIN
	}
}
# no bottom for input data layer
'''

# data layer for spoofing_ld
class Data_Layer_spoofing_line_detect(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.posDataFile = params["posDataFile"]
		self.negDataFile = params["negDataFile"]
		self.ft_size 	= params["ftSize"]

		np.random.seed(46)
		random.seed(100)
		with open(self.posDataFile, 'rb') as f:
			self.posData = pickle.load(f)
			np.random.shuffle(self.posData)
		with open(self.negDataFile, 'rb') as f:
			self.negData = pickle.load(f)
			np.random.shuffle(self.negData)
		self.posDataCounter = 0
		self.posDataLen = self.posData.shape[0]
		self.negDataCounter = 0
		self.negDataLen = self.negData.shape[0]

		top[0].reshape(self.batch_size*2, self.ft_size)
		top[1].reshape(self.batch_size*2)
		for itt in range(self.batch_size): 
			top[1].data[itt] = 0
			top[1].data[itt+self.batch_size] = 1

	def reshape(self, bottom, top):
		for itt in range(self.batch_size): 
			top[1].data[itt] = 0
			top[1].data[itt+self.batch_size] = 1

	def forward(self, bottom, top):
		for itt in range(self.batch_size):
			top[0].data[itt, ...] = self.posData[self.posDataCounter, ...]			
			top[0].data[itt+self.batch_size, ...] = self.negData[self.negDataCounter, ...]
			self.posDataCounter += 1
			self.negDataCounter += 1

			if self.posDataCounter >= self.posDataLen:
				np.random.shuffle(self.posData)
				self.posDataCounter = 0
			if self.negDataCounter >= self.negDataLen:
				np.random.shuffle(self.negData)
				self.negDataCounter = 0

	def backward(self, top, propagate_down, bottom):
		pass


# used to chagne loss_weight on the fly
# Note: need to set the loss_weight of the loss layers which connected to this layer to 0
# the first n inputs are loss, the next n inputs are coeff for the losses
class Loss_Layer_dynamic_loss_weight(caffe.Layer):
	def setup(self,bottom,top):
		self.n = len(bottom)
		assert self.n>1, 'at least two inputs are needed'
		assert self.n%2==0, 'number of inputs must be even, one coeff for each loss'
		assert len(top)==1, 'only one output is allowed'
		self.n /= 2
		for i in range(self.n):
			print(bottom[i].data.shape)
			assert bottom[i+self.n].data.shape == (1,), 'the input shape must be 1'
			bottom[i+self.n].diff[...] = 0
		top[0].reshape(1)

	def reshape(self,bottom,top):
		pass

	def forward(self,bottom,top):
		top[0].data[...] = sum([bottom[i].data[...]*bottom[i+self.n].data[...] for i in range(self.n)])

	def backward(self,top,propagate_down,bottom):
		for i in range(self.n):
			bottom[i].diff[...] = bottom[i+self.n].data[...]


class Data_Layer_online_classify_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.ftDataFile = params["ftDataFile"]
		self.smooth 	= params["smooth"]

		self.ft_size = 512

		#print "Data_Layer_online_classify_train, setup"

		np.random.seed(46)
		random.seed(100)
		with open(self.ftDataFile, 'rb') as f:
			self.ft_dic = pickle.load(f)

		self.ft_dic_keys = sorted(self.ft_dic.keys())
		random.shuffle(self.ft_dic_keys)
		self.dataCounter = 0
		self.dataLen = len(self.ft_dic_keys)

		top[0].reshape(self.batch_size, self.ft_size)
		top[1].reshape(self.batch_size, self.ft_size)
		top[2].reshape(self.batch_size)
		top[3].reshape(self.batch_size)
		top[4].reshape(self.batch_size)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		#print "Data_Layer_online_classify_train, forward"
		for itt in range(self.batch_size):
			# do padding, resizing and normalization, pts transformation and .reshape((10,))
			top[0].data[itt, ...] = self.ft_dic[self.ft_dic_keys[self.dataCounter]]['ft']
			tmp = np.random.random(self.ft_size).astype(np.float32)-0.5
			top[1].data[itt, ...] = tmp/np.linalg.norm(tmp)
			top[2].data[itt] = self.ft_dic[self.ft_dic_keys[self.dataCounter]]['label']
			top[3].data[itt] = 1.0-self.smooth
			top[4].data[itt] = 0.0

			self.dataCounter += 1

			if self.dataCounter >= self.dataLen:
				random.shuffle(self.ft_dic_keys)
				self.dataCounter = 0
		#print "Data_Layer_online_classify_train, forward #2"

	def backward(self, top, propagate_down, bottom):
		pass

# 3 data sources: real_ft inside database, real_ft outside database and fake_ft
# both of real_ft outside database and fake_ft are considered fake class for the classifer
# both of inside and outside real_ft are against fake_ft to train generator net
# generator does not generate image but the face_ft directly
class Data_Layer_online_classify_no_caffeface_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.ft_size = params["featureSize"]
		self.ftInDataFile = params["ftInDataFile"]
		self.ftOutDataFile = params["ftOutDataFile"]
		self.smooth 	= params["smooth"]
		

		#print "Data_Layer_online_classify_train, setup"

		np.random.seed(46)
		random.seed(100)
		with open(self.ftInDataFile, 'rb') as f:
			self.ft_in_dic = pickle.load(f)
		with open(self.ftOutDataFile, 'rb') as f:
			self.ft_out_dic = pickle.load(f)
		self.ft_in_dic_keys = sorted(self.ft_in_dic.keys())
		self.ft_out_dic_keys = sorted(self.ft_out_dic.keys())
		random.shuffle(self.ft_in_dic_keys)
		random.shuffle(self.ft_out_dic_keys)
		self.dataInCounter = 0
		self.dataOutCounter = 0
		self.dataInLen = len(self.ft_in_dic_keys)
		self.dataOutLen = len(self.ft_out_dic_keys)

		top[0].reshape(self.batch_size, self.ft_size)
		top[1].reshape(self.batch_size, self.ft_size)
		top[2].reshape(self.batch_size, self.ft_size)
		top[3].reshape(self.batch_size)
		top[4].reshape(self.batch_size)
		top[5].reshape(self.batch_size)
		top[6].reshape(self.batch_size)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		#print "Data_Layer_online_classify_train, forward"
		for itt in range(self.batch_size):
			# do padding, resizing and normalization, pts transformation and .reshape((10,))
			top[0].data[itt, ...] = self.ft_in_dic[self.ft_in_dic_keys[self.dataInCounter]]['ft']
			top[1].data[itt, ...] = self.ft_out_dic[self.ft_out_dic_keys[self.dataOutCounter]]
			tmp = np.random.random(self.ft_size).astype(np.float32)-0.5
			top[2].data[itt, ...] = tmp/np.linalg.norm(tmp)
			top[3].data[itt] = self.ft_in_dic[self.ft_in_dic_keys[self.dataInCounter]]['label']
			top[4].data[itt] = 1.0-self.smooth
			top[5].data[itt] = 0.0
			top[6].data[itt] = 0.0

			self.dataInCounter += 1
			self.dataOutCounter += 1

			if self.dataInCounter >= self.dataInLen:
				random.shuffle(self.ft_in_dic_keys)
				self.dataInCounter = 0
			if self.dataOutCounter >= self.dataOutLen:
				random.shuffle(self.ft_out_dic_keys)
				self.dataOutCounter = 0
		#print "Data_Layer_online_classify_train, forward #2"

	def backward(self, top, propagate_down, bottom):
		pass


# the gen_data is generated using auto-encoder with the real_in image, which will be fed into gan net
class Data_Layer_online_classify_conditional_gan_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.ft_size = params["featureSize"]
		self.ftInDataFile = params["ftInDataFile"]
		self.ftOutDataFile = params["ftOutDataFile"]
		self.smooth 	= params["smooth"]
		

		#print "Data_Layer_online_classify_train, setup"

		np.random.seed(46)
		random.seed(100)
		with open(self.ftInDataFile, 'rb') as f:
			self.ft_in_dic = pickle.load(f)
		with open(self.ftOutDataFile, 'rb') as f:
			self.ft_out_dic = pickle.load(f)
		self.ft_in_dic_keys = sorted(self.ft_in_dic.keys())
		self.ft_out_dic_keys = sorted(self.ft_out_dic.keys())
		random.shuffle(self.ft_in_dic_keys)
		random.shuffle(self.ft_out_dic_keys)
		self.dataInCounter = 0
		self.dataOutCounter = 0
		self.dataInLen = len(self.ft_in_dic_keys)
		self.dataOutLen = len(self.ft_out_dic_keys)

		top[0].reshape(self.batch_size, self.ft_size)
		top[1].reshape(self.batch_size, self.ft_size)
		top[2].reshape(self.batch_size, self.ft_size/2)
		top[3].reshape(self.batch_size)
		top[4].reshape(self.batch_size)
		top[5].reshape(self.batch_size)
		top[6].reshape(self.batch_size)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		#print "Data_Layer_online_classify_train, forward"
		for itt in range(self.batch_size):
			# do padding, resizing and normalization, pts transformation and .reshape((10,))
			top[0].data[itt, ...] = self.ft_in_dic[self.ft_in_dic_keys[self.dataInCounter]]['ft']
			top[1].data[itt, ...] = self.ft_out_dic[self.ft_out_dic_keys[self.dataOutCounter]]
			top[2].data[itt, ...] = self.ft_in_dic[self.ft_in_dic_keys[self.dataInCounter]]['ae_ft']
			top[3].data[itt] = self.ft_in_dic[self.ft_in_dic_keys[self.dataInCounter]]['label']
			top[4].data[itt] = 1.0-self.smooth
			top[5].data[itt] = 0.0
			top[6].data[itt] = 0.0

			self.dataInCounter += 1
			self.dataOutCounter += 1

			if self.dataInCounter >= self.dataInLen:
				random.shuffle(self.ft_in_dic_keys)
				self.dataInCounter = 0
			if self.dataOutCounter >= self.dataOutLen:
				random.shuffle(self.ft_out_dic_keys)
				self.dataOutCounter = 0
		#print "Data_Layer_online_classify_train, forward #2"

	def backward(self, top, propagate_down, bottom):
		pass


# auto encoder and classifier share the same weights
class Data_Layer_online_classify_conditional_gan_ae_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.ft_size = params["featureSize"]
		self.ftInDataFile = params["ftInDataFile"]
		self.ftOutDataFile = params["ftOutDataFile"]
		self.smooth 	= params["smooth"]
		

		#print "Data_Layer_online_classify_train, setup"

		np.random.seed(46)
		random.seed(100)
		with open(self.ftInDataFile, 'rb') as f:
			self.ft_in_dic = pickle.load(f)
		with open(self.ftOutDataFile, 'rb') as f:
			self.ft_out_dic = pickle.load(f)
		self.ft_in_dic_keys = sorted(self.ft_in_dic.keys())
		self.ft_out_dic_keys = sorted(self.ft_out_dic.keys())
		random.shuffle(self.ft_in_dic_keys)
		random.shuffle(self.ft_out_dic_keys)
		self.dataInCounter = 0
		self.dataOutCounter = 0
		self.dataInLen = len(self.ft_in_dic_keys)
		self.dataOutLen = len(self.ft_out_dic_keys)

		top[0].reshape(self.batch_size, self.ft_size)
		top[1].reshape(self.batch_size, self.ft_size)
		top[2].reshape(self.batch_size)
		top[3].reshape(self.batch_size)
		top[4].reshape(self.batch_size)
		top[5].reshape(self.batch_size)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		#print "Data_Layer_online_classify_train, forward"
		for itt in range(self.batch_size):
			# do padding, resizing and normalization, pts transformation and .reshape((10,))
			top[0].data[itt, ...] = self.ft_in_dic[self.ft_in_dic_keys[self.dataInCounter]]['ft']
			top[1].data[itt, ...] = self.ft_out_dic[self.ft_out_dic_keys[self.dataOutCounter]]
			top[2].data[itt] = self.ft_in_dic[self.ft_in_dic_keys[self.dataInCounter]]['label']
			top[3].data[itt] = 1.0-self.smooth
			top[4].data[itt] = 0.0
			top[5].data[itt] = 0.0

			self.dataInCounter += 1
			self.dataOutCounter += 1

			if self.dataInCounter >= self.dataInLen:
				random.shuffle(self.ft_in_dic_keys)
				self.dataInCounter = 0
			if self.dataOutCounter >= self.dataOutLen:
				random.shuffle(self.ft_out_dic_keys)
				self.dataOutCounter = 0
		#print "Data_Layer_online_classify_train, forward #2"

	def backward(self, top, propagate_down, bottom):
		pass


# same as Data_Layer_online_classify_conditional_gan_ae_train, but add a noise output which will be used for gan input
class Data_Layer_online_classify_conditional_gan_ae_noise_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.ft_size = params["featureSize"]
		self.ftInDataFile = params["ftInDataFile"]
		self.ftOutDataFile = params["ftOutDataFile"]
		self.smooth 	= params["smooth"]
		self.noiseFactor 	= params["noiseFactor"]
		

		#print "Data_Layer_online_classify_train, setup"

		np.random.seed(46)
		random.seed(100)
		with open(self.ftInDataFile, 'rb') as f:
			self.ft_in_dic = pickle.load(f)
		with open(self.ftOutDataFile, 'rb') as f:
			self.ft_out_dic = pickle.load(f)
		self.ft_in_dic_keys = sorted(self.ft_in_dic.keys())
		self.ft_out_dic_keys = sorted(self.ft_out_dic.keys())
		random.shuffle(self.ft_in_dic_keys)
		random.shuffle(self.ft_out_dic_keys)
		self.dataInCounter = 0
		self.dataOutCounter = 0
		self.dataInLen = len(self.ft_in_dic_keys)
		self.dataOutLen = len(self.ft_out_dic_keys)

		top[0].reshape(self.batch_size, self.ft_size)
		top[1].reshape(self.batch_size, self.ft_size)
		top[2].reshape(self.batch_size)
		top[3].reshape(self.batch_size)
		top[4].reshape(self.batch_size)
		top[5].reshape(self.batch_size)
		top[6].reshape(self.batch_size, self.ft_size/2)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		#print "Data_Layer_online_classify_train, forward"
		for itt in range(self.batch_size):
			# do padding, resizing and normalization, pts transformation and .reshape((10,))
			top[0].data[itt, ...] = self.ft_in_dic[self.ft_in_dic_keys[self.dataInCounter]]['ft']
			top[1].data[itt, ...] = self.ft_out_dic[self.ft_out_dic_keys[self.dataOutCounter]]
			top[2].data[itt] = self.ft_in_dic[self.ft_in_dic_keys[self.dataInCounter]]['label']
			top[3].data[itt] = 1.0-self.smooth
			top[4].data[itt] = 0.0
			top[5].data[itt] = 0.0
			tmp = np.random.random(self.ft_size)
			random.shuffle(tmp)
			top[6].data[itt, ...] = (tmp[0: self.ft_size/2]-0.5) * self.noiseFactor

			self.dataInCounter += 1
			self.dataOutCounter += 1

			if self.dataInCounter >= self.dataInLen:
				random.shuffle(self.ft_in_dic_keys)
				self.dataInCounter = 0
			if self.dataOutCounter >= self.dataOutLen:
				random.shuffle(self.ft_out_dic_keys)
				self.dataOutCounter = 0
		#print "Data_Layer_online_classify_train, forward #2"

	def backward(self, top, propagate_down, bottom):
		pass


class No_Backward_Layer(caffe.Layer):
	def setup(self,bottom,top):
		if len(bottom) != 1:
			raise Exception("Need 1 Inputs Only!")
		if len(top) != 1:
			raise Exception("Need 1 Outputs Only!")

	def reshape(self,bottom,top):
		top[0].reshape(*bottom[0].data.shape)

	def forward(self,bottom,top):
		top[0].data[...] = bottom[0].data[...]

	def backward(self,top,propagate_down,bottom):
		bottom[0].diff[...] = 0


class Data_Layer_online_classify_only_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.ftInDataFile = params["ftInDataFile"]
		self.ftOutDataFile = params["ftOutDataFile"]
		self.smooth 	= params["smooth"]

		self.ft_size = 512

		if self.phase==caffe.TRAIN:
			print("CAFFE TRAIN")
		else:
			print("CAFFE TEST")

		#print "Data_Layer_online_classify_train, setup"

		np.random.seed(46)
		random.seed(100)
		with open(self.ftInDataFile, 'rb') as f:
			self.ft_in_dic = pickle.load(f)

		self.ft_in_dic_keys = sorted(self.ft_in_dic.keys())
		random.shuffle(self.ft_in_dic_keys)
		self.dataInCounter = 0
		self.dataInLen = len(self.ft_in_dic_keys)

		with open(self.ftOutDataFile, 'rb') as f:
			self.ft_out_dic = pickle.load(f)

		self.ft_out_dic_keys = sorted(self.ft_out_dic.keys())
		random.shuffle(self.ft_out_dic_keys)
		self.dataOutCounter = 0
		self.dataOutLen = len(self.ft_out_dic_keys)

		top[0].reshape(self.batch_size, self.ft_size)
		top[1].reshape(self.batch_size, self.ft_size)
		top[2].reshape(self.batch_size)
		top[3].reshape(self.batch_size)
		top[4].reshape(self.batch_size)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		#print "Data_Layer_online_classify_train, forward"
		for itt in range(self.batch_size):
			# do padding, resizing and normalization, pts transformation and .reshape((10,))
			top[0].data[itt, ...] = self.ft_in_dic[self.ft_in_dic_keys[self.dataInCounter]]['ft']
			top[1].data[itt, ...] = self.ft_out_dic[self.ft_out_dic_keys[self.dataOutCounter]]
			top[2].data[itt] = self.ft_in_dic[self.ft_in_dic_keys[self.dataInCounter]]['label']
			top[3].data[itt] = 1.0-self.smooth
			top[4].data[itt] = 0.0

			self.dataInCounter += 1

			if self.dataInCounter >= self.dataInLen:
				random.shuffle(self.ft_in_dic_keys)
				self.dataInCounter = 0

			self.dataOutCounter += 1

			if self.dataOutCounter >= self.dataOutLen:
				random.shuffle(self.ft_out_dic_keys)
				self.dataOutCounter = 0
		#print "Data_Layer_online_classify_train, forward #2"

	def backward(self, top, propagate_down, bottom):
		pass


class Data_Layer_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.faceDataFile = params["faceDataFile"]
		self.fileNumList = range(1,params["numOfFile"]+1)
		self.filePostfix = params["filePostfix"]
		self.curFileIdx = 0

		self.net_side = 48

		print "Data_Layer_train, setup"

		random.seed(100)
		random.shuffle(self.fileNumList)
		with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
			self.posDataset = pickle.load(f)

		self.posDataList = sorted(self.posDataset.keys())
		random.shuffle(self.posDataList)
		self.posLen = len(self.posDataList)
		self.posCounter = 0	

		top[0].reshape(self.batch_size, 3, self.net_side, self.net_side)
		top[1].reshape(self.batch_size, 10)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):

		for itt in range(self.batch_size):			

			# do padding, resizing and normalization, pts transformation and .reshape((10,))
			top[0].data[itt, ...] = self.posDataset[self.posDataList[self.posCounter]]['image']				
			top[1].data[itt, ...] = self.posDataset[self.posDataList[self.posCounter]]['lbl5Points']	

			self.posCounter += 1

			if self.posCounter >= self.posLen:
				self.curFileIdx+=1
				if self.curFileIdx>=len(self.fileNumList):
					self.curFileIdx=0
					random.shuffle(self.fileNumList)

				with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
					self.posDataset = pickle.load(f)

				self.posDataList = sorted(self.posDataset.keys())
				random.shuffle(self.posDataList)
				self.posLen = len(self.posDataList)
				self.posCounter = 0	


	def backward(self, top, propagate_down, bottom):
		pass

# same as Data_Layer_train, but add flip feature
class Data_Flip_Layer_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.faceDataFile = params["faceDataFile"]
		self.fileNumList = range(1,params["numOfFile"]+1)
		self.filePostfix = params["filePostfix"]
		self.curFileIdx = 0

		self.net_side = 48

		print "Data_Layer_train, setup"

		random.seed(100)
		random.shuffle(self.fileNumList)
		with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
			self.posDataset = pickle.load(f)

		self.posDataList = sorted(self.posDataset.keys())
		random.shuffle(self.posDataList)
		self.posLen = len(self.posDataList)
		self.posCounter = 0	

		top[0].reshape(self.batch_size, 3, self.net_side, self.net_side)
		top[1].reshape(self.batch_size, 10)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):

		for itt in range(self.batch_size):			
			k = self.posDataList[self.posCounter]

			flipFlag = np.random.random(1)>=0.5 #################################################
			#flipFlag = 0

			if flipFlag:
				tmpImg = self.posDataset[k]['image'].copy()
				tmpImg = np.swapaxes(tmpImg, 0, 2)
				tmpImg = cv2.flip(tmpImg, 1)
				tmpImg = np.swapaxes(tmpImg, 0, 2)
				top[0].data[itt, ...] = tmpImg
			else:
				top[0].data[itt, ...] = self.posDataset[k]['image']

			top[1].data[itt, ...] = self.posDataset[k]['lbl5Points']
			if flipFlag:
				top[1].data[itt, 0:5] = 1.0-top[1].data[itt, [1,0,2,4,3]]

			self.posCounter += 1

			if self.posCounter >= self.posLen:
				self.curFileIdx+=1
				if self.curFileIdx>=len(self.fileNumList):
					self.curFileIdx=0
					random.shuffle(self.fileNumList)

				with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
					self.posDataset = pickle.load(f)

				self.posDataList = sorted(self.posDataset.keys())
				random.shuffle(self.posDataList)
				self.posLen = len(self.posDataList)
				self.posCounter = 0	


	def backward(self, top, propagate_down, bottom):
		pass

# output the relative distance to nose as well
class Data1_Layer_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.faceDataFile = params["faceDataFile"]
		self.fileNumList = range(1,params["numOfFile"]+1)
		self.filePostfix = params["filePostfix"]
		self.curFileIdx = 0

		self.net_side = 48

		print "Data_Layer_train, setup"

		random.seed(100)
		random.shuffle(self.fileNumList)
		with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
			self.posDataset = pickle.load(f)

		self.posDataList = sorted(self.posDataset.keys())
		random.shuffle(self.posDataList)
		self.posLen = len(self.posDataList)
		self.posCounter = 0	

		top[0].reshape(self.batch_size, 3, self.net_side, self.net_side)
		top[1].reshape(self.batch_size, 10)
		top[2].reshape(self.batch_size, 8)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):

		for itt in range(self.batch_size):			

			# do padding, resizing and normalization, pts transformation and .reshape((10,))
			top[0].data[itt, ...] = self.posDataset[self.posDataList[self.posCounter]]['image']				
			top[1].data[itt, ...] = self.posDataset[self.posDataList[self.posCounter]]['lbl5Points']
			top[2].data[itt, 0] = top[1].data[itt, 0] - top[1].data[itt, 2]
			top[2].data[itt, 1] = top[1].data[itt, 1] - top[1].data[itt, 2]
			top[2].data[itt, 2] = top[1].data[itt, 3] - top[1].data[itt, 2]
			top[2].data[itt, 3] = top[1].data[itt, 4] - top[1].data[itt, 2]
			top[2].data[itt, 4] = top[1].data[itt, 5] - top[1].data[itt, 7]
			top[2].data[itt, 5] = top[1].data[itt, 6] - top[1].data[itt, 7]
			top[2].data[itt, 6] = top[1].data[itt, 8] - top[1].data[itt, 7]
			top[2].data[itt, 7] = top[1].data[itt, 9] - top[1].data[itt, 7]

			self.posCounter += 1

			if self.posCounter >= self.posLen:
				self.curFileIdx+=1
				if self.curFileIdx>=len(self.fileNumList):
					self.curFileIdx=0
					random.shuffle(self.fileNumList)

				with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
					self.posDataset = pickle.load(f)

				self.posDataList = sorted(self.posDataset.keys())
				random.shuffle(self.posDataList)
				self.posLen = len(self.posDataList)
				self.posCounter = 0	


	def backward(self, top, propagate_down, bottom):
		pass

# output normalized relative distance to nose as well
class Data2_Layer_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.faceDataFile = params["faceDataFile"]
		self.fileNumList = range(1,params["numOfFile"]+1)
		self.filePostfix = params["filePostfix"]
		self.curFileIdx = 0

		self.net_side = 48

		print "Data_Layer_train, setup"

		random.seed(100)
		random.shuffle(self.fileNumList)
		with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
			self.posDataset = pickle.load(f)

		self.posDataList = sorted(self.posDataset.keys())
		random.shuffle(self.posDataList)
		self.posLen = len(self.posDataList)
		self.posCounter = 0	

		top[0].reshape(self.batch_size, 3, self.net_side, self.net_side)
		top[1].reshape(self.batch_size, 10)
		top[2].reshape(self.batch_size, 8)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):

		for itt in range(self.batch_size):			

			# do padding, resizing and normalization, pts transformation and .reshape((10,))
			top[0].data[itt, ...] = self.posDataset[self.posDataList[self.posCounter]]['image']				
			top[1].data[itt, ...] = self.posDataset[self.posDataList[self.posCounter]]['lbl5Points']

			max_x=max(top[1].data[itt, 0:5])
			min_x=min(top[1].data[itt, 0:5])
			max_y=max(top[1].data[itt, 5:10])
			min_y=min(top[1].data[itt, 5:10])
			max_dist = max([max_x-min_x, max_y-min_y])

			top[2].data[itt, 0] = (top[1].data[itt, 0] - top[1].data[itt, 2])/max_dist
			top[2].data[itt, 1] = (top[1].data[itt, 1] - top[1].data[itt, 2])/max_dist
			top[2].data[itt, 2] = (top[1].data[itt, 3] - top[1].data[itt, 2])/max_dist
			top[2].data[itt, 3] = (top[1].data[itt, 4] - top[1].data[itt, 2])/max_dist
			top[2].data[itt, 4] = (top[1].data[itt, 5] - top[1].data[itt, 7])/max_dist
			top[2].data[itt, 5] = (top[1].data[itt, 6] - top[1].data[itt, 7])/max_dist
			top[2].data[itt, 6] = (top[1].data[itt, 8] - top[1].data[itt, 7])/max_dist
			top[2].data[itt, 7] = (top[1].data[itt, 9] - top[1].data[itt, 7])/max_dist

			self.posCounter += 1

			if self.posCounter >= self.posLen:
				self.curFileIdx+=1
				if self.curFileIdx>=len(self.fileNumList):
					self.curFileIdx=0
					random.shuffle(self.fileNumList)

				with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
					self.posDataset = pickle.load(f)

				self.posDataList = sorted(self.posDataset.keys())
				random.shuffle(self.posDataList)
				self.posLen = len(self.posDataList)
				self.posCounter = 0	


	def backward(self, top, propagate_down, bottom):
		pass

# output img, cls, bbox, landmark and the mask for bbox and landmark, add class#2 which indicate that the instance with class#2 will be ignored for loss calculation
class Data4_Layer_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.faceDataFile = params["faceDataFile"]
		self.fileNumList = range(1,params["numOfFile"]+1)
		self.filePostfix = params["filePostfix"]		
		self.curFileIdx = 0 # for lardmark dataset

		self.sampleRatio = eval(params["sampleRatio"]) # ratio among [neg,pos,part,lardmark] such as [3,1,1,2]
		self.sampleRatio = np.cumsum(self.sampleRatio)/np.sum(self.sampleRatio).astype(float)
		
		self.wideFaceDataFile = params["wideFaceDataFile"]
		self.numOfWideFaceFile = eval(params["numOfWideFaceFile"]) # number of files for neg, pos and part, such as [13,5,4]
		self.negDataFileList = range(1, self.numOfWideFaceFile[0]+1)
		self.posDataFileList = range(1, self.numOfWideFaceFile[1]+1)
		self.partDataFileList = range(1, self.numOfWideFaceFile[2]+1)
		self.curWideFileIdx = [0,0,0] # for wider face dataset neg, pos and part

		self.net_side = 48

		print "Data_Layer_train, setup"

		random.seed(100)

		# process landmark dataset
		random.shuffle(self.fileNumList)		
		with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
			self.posDataset = pickle.load(f) # this is actually landmark dataset
		self.posDataList = sorted(self.posDataset.keys())
		random.shuffle(self.posDataList)
		self.posLen = len(self.posDataList)
		self.posCounter = 0

		# process wider face neg dataset
		random.shuffle(self.negDataFileList)
		with open(self.wideFaceDataFile+'neg_'+str(self.negDataFileList[self.curWideFileIdx[0]])+'.pkl', 'rb') as f:
			self.wfNegDataset = pickle.load(f) # this is wider face dataset
		self.wfNegDataList = sorted(self.wfNegDataset.keys())
		random.shuffle(self.wfNegDataList)
		self.wfNegLen = len(self.wfNegDataList)
		self.wfNegCounter = 0

		# process wider face pos dataset
		random.shuffle(self.posDataFileList)
		with open(self.wideFaceDataFile+'pos_'+str(self.posDataFileList[self.curWideFileIdx[1]])+'.pkl', 'rb') as f:
			self.wfPosDataset = pickle.load(f) # this is wider face dataset
		self.wfPosDataList = sorted(self.wfPosDataset.keys())
		random.shuffle(self.wfPosDataList)
		self.wfPosLen = len(self.wfPosDataList)
		self.wfPosCounter = 0

		# process wider face part dataset
		random.shuffle(self.partDataFileList)
		with open(self.wideFaceDataFile+'part_'+str(self.partDataFileList[self.curWideFileIdx[2]])+'.pkl', 'rb') as f:
			self.wfPartDataset = pickle.load(f) # this is wider face dataset
		self.wfPartDataList = sorted(self.wfPartDataset.keys())
		random.shuffle(self.wfPartDataList)
		self.wfPartLen = len(self.wfPartDataList)
		self.wfPartCounter = 0


		top[0].reshape(self.batch_size, 3, self.net_side, self.net_side) #image
		top[1].reshape(self.batch_size, 10) # landmark
		top[2].reshape(self.batch_size, 4) # bbox
		top[3].reshape(self.batch_size, 1) # label
		top[4].reshape(self.batch_size, 10) # landmark mask
		top[5].reshape(self.batch_size, 4) # bbox mask
		#top[6].reshape(self.batch_size, 2) # class mask

	def reshape(self, bottom, top):
		top[1].data[...] = 0
		top[2].data[...] = 0
		top[3].data[...] = 2 # class 2 will not take part in classification loss calculation
		top[4].data[...] = 0
		top[5].data[...] = 0
		#top[6].data[...] = 0

	def forward(self, bottom, top):

		category = np.random.random(self.batch_size)

		top[4].data[category>=self.sampleRatio[2], ...] = 1
		top[5].data[np.logical_and(category>=self.sampleRatio[0], category<self.sampleRatio[2]), ...] = 1
		#top[6].data[category<self.sampleRatio[1], ...] = 1

		for itt in range(self.batch_size):						

			if category[itt]>=self.sampleRatio[2]: # landmark dataset
				k=self.posDataList[self.posCounter]			
				flipFlag = np.random.random(1)>=0.5

				if flipFlag:
					tmpImg = self.posDataset[k]['image'].copy()
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					tmpImg = cv2.flip(tmpImg, 1)
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					top[0].data[itt, ...] = tmpImg
				else:
					top[0].data[itt, ...] = self.posDataset[k]['image']

				top[1].data[itt, ...] = self.posDataset[k]['lbl5Points']
				if flipFlag:
					top[1].data[itt, 0:5] = 1.0-top[1].data[itt, 0:5]

				self.posCounter += 1

				if self.posCounter >= self.posLen:
					self.curFileIdx+=1
					if self.curFileIdx>=len(self.fileNumList):
						self.curFileIdx=0
						random.shuffle(self.fileNumList)

					with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
						self.posDataset = pickle.load(f)

					self.posDataList = sorted(self.posDataset.keys())
					random.shuffle(self.posDataList)
					self.posLen = len(self.posDataList)
					self.posCounter = 0	
			elif category[itt]>=self.sampleRatio[1]: # wf part dataset
				k=self.wfPartDataList[self.wfPartCounter]
				flipFlag = np.random.random(1)>=0.5

				if flipFlag:
					tmpImg = self.wfPartDataset[k]['image'].copy()
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					tmpImg = cv2.flip(tmpImg, 1)
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					top[0].data[itt, ...] = tmpImg
				else:
					top[0].data[itt, ...] = self.wfPartDataset[k]['image']

				top[2].data[itt, ...] = self.wfPartDataset[k]['bbox']
				if flipFlag:
					tmpVal = top[2].data[itt, 2]
					top[2].data[itt, 2] = -1.0*top[2].data[itt, 0]
					top[2].data[itt, 0] = -1.0*tmpVal

				self.wfPartCounter += 1

				if self.wfPartCounter >= self.wfPartLen:
					self.curWideFileIdx[2]+=1
					if self.curWideFileIdx[2]>=len(self.partDataFileList):
						self.curWideFileIdx[2]=0
						random.shuffle(self.partDataFileList)

					with open(self.wideFaceDataFile+'part_'+str(self.partDataFileList[self.curWideFileIdx[2]])+'.pkl', 'rb') as f:
						self.wfPartDataset = pickle.load(f)

					self.wfPartDataList = sorted(self.wfPartDataset.keys())
					random.shuffle(self.wfPartDataList)
					self.wfPartLen = len(self.wfPartDataList)
					self.wfPartCounter = 0
			elif category[itt]>=self.sampleRatio[0]: # wf pos dataset
				k=self.wfPosDataList[self.wfPosCounter]
				flipFlag = np.random.random(1)>=0.5

				if flipFlag:
					tmpImg = self.wfPosDataset[k]['image'].copy()
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					tmpImg = cv2.flip(tmpImg, 1)
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					top[0].data[itt, ...] = tmpImg
				else:
					top[0].data[itt, ...] = self.wfPosDataset[k]['image']

				top[2].data[itt, ...] = self.wfPosDataset[k]['bbox']
				if flipFlag:
					tmpVal = top[2].data[itt, 2]
					top[2].data[itt, 2] = -1.0*top[2].data[itt, 0]
					top[2].data[itt, 0] = -1.0*tmpVal

				top[3].data[itt, ...] = 1

				self.wfPosCounter += 1

				if self.wfPosCounter >= self.wfPosLen:
					self.curWideFileIdx[1]+=1
					if self.curWideFileIdx[1]>=len(self.posDataFileList):
						self.curWideFileIdx[1]=0
						random.shuffle(self.posDataFileList)

					with open(self.wideFaceDataFile+'pos_'+str(self.posDataFileList[self.curWideFileIdx[1]])+'.pkl', 'rb') as f:
						self.wfPosDataset = pickle.load(f)

					self.wfPosDataList = sorted(self.wfPosDataset.keys())
					random.shuffle(self.wfPosDataList)
					self.wfPosLen = len(self.wfPosDataList)
					self.wfPosCounter = 0
			else: # wf neg dataset
				k=self.wfNegDataList[self.wfNegCounter]
				flipFlag = np.random.random(1)>=0.5

				if flipFlag:
					tmpImg = self.wfNegDataset[k]['image'].copy()
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					tmpImg = cv2.flip(tmpImg, 1)
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					top[0].data[itt, ...] = tmpImg
				else:
					top[0].data[itt, ...] = self.wfNegDataset[k]['image']

				top[3].data[itt, ...] = 0

				self.wfNegCounter += 1

				if self.wfNegCounter >= self.wfNegLen:
					self.curWideFileIdx[0]+=1
					if self.curWideFileIdx[0]>=len(self.negDataFileList):
						self.curWideFileIdx[0]=0
						random.shuffle(self.negDataFileList)

					with open(self.wideFaceDataFile+'neg_'+str(self.negDataFileList[self.curWideFileIdx[0]])+'.pkl', 'rb') as f:
						self.wfNegDataset = pickle.load(f)

					self.wfNegDataList = sorted(self.wfNegDataset.keys())
					random.shuffle(self.wfNegDataList)
					self.wfNegLen = len(self.wfNegDataList)
					self.wfNegCounter = 0

	def backward(self, top, propagate_down, bottom):
		pass

# output img, cls, bbox, landmark and the mask for cls, bbox and landmark
class Data5_Layer_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.faceDataFile = params["faceDataFile"]
		self.fileNumList = range(1,params["numOfFile"]+1)
		self.filePostfix = params["filePostfix"]		
		self.curFileIdx = 0 # for lardmark dataset

		self.sampleRatio = eval(params["sampleRatio"]) # ratio among [neg,pos,part,lardmark] such as [3,1,1,2]
		self.sampleRatio = np.cumsum(self.sampleRatio)/np.sum(self.sampleRatio).astype(float)
		
		self.wideFaceDataFile = params["wideFaceDataFile"]
		self.numOfWideFaceFile = eval(params["numOfWideFaceFile"]) # number of files for neg, pos and part, such as [13,5,4]
		self.negDataFileList = range(1, self.numOfWideFaceFile[0]+1)
		self.posDataFileList = range(1, self.numOfWideFaceFile[1]+1)
		self.partDataFileList = range(1, self.numOfWideFaceFile[2]+1)
		self.curWideFileIdx = [0,0,0] # for wider face dataset neg, pos and part

		self.net_side = 48

		print "Data_Layer_train, setup"

		random.seed(100)

		# process landmark dataset
		random.shuffle(self.fileNumList)		
		with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
			self.posDataset = pickle.load(f) # this is actually landmark dataset
		self.posDataList = sorted(self.posDataset.keys())
		random.shuffle(self.posDataList)
		self.posLen = len(self.posDataList)
		self.posCounter = 0

		# process wider face neg dataset
		random.shuffle(self.negDataFileList)
		with open(self.wideFaceDataFile+'neg_'+str(self.negDataFileList[self.curWideFileIdx[0]])+'.pkl', 'rb') as f:
			self.wfNegDataset = pickle.load(f) # this is wider face dataset
		self.wfNegDataList = sorted(self.wfNegDataset.keys())
		random.shuffle(self.wfNegDataList)
		self.wfNegLen = len(self.wfNegDataList)
		self.wfNegCounter = 0

		# process wider face pos dataset
		random.shuffle(self.posDataFileList)
		with open(self.wideFaceDataFile+'pos_'+str(self.posDataFileList[self.curWideFileIdx[1]])+'.pkl', 'rb') as f:
			self.wfPosDataset = pickle.load(f) # this is wider face dataset
		self.wfPosDataList = sorted(self.wfPosDataset.keys())
		random.shuffle(self.wfPosDataList)
		self.wfPosLen = len(self.wfPosDataList)
		self.wfPosCounter = 0

		# process wider face part dataset
		random.shuffle(self.partDataFileList)
		with open(self.wideFaceDataFile+'part_'+str(self.partDataFileList[self.curWideFileIdx[2]])+'.pkl', 'rb') as f:
			self.wfPartDataset = pickle.load(f) # this is wider face dataset
		self.wfPartDataList = sorted(self.wfPartDataset.keys())
		random.shuffle(self.wfPartDataList)
		self.wfPartLen = len(self.wfPartDataList)
		self.wfPartCounter = 0


		top[0].reshape(self.batch_size, 3, self.net_side, self.net_side) #image
		top[1].reshape(self.batch_size, 10) # landmark
		top[2].reshape(self.batch_size, 4) # bbox
		top[3].reshape(self.batch_size, 1) # label
		top[4].reshape(self.batch_size, 10) # landmark mask
		top[5].reshape(self.batch_size, 4) # bbox mask
		top[6].reshape(self.batch_size, 2) # class mask

	def reshape(self, bottom, top):
		top[1].data[...] = 0
		top[2].data[...] = 0
		top[3].data[...] = -1
		top[4].data[...] = 1 ################################################################
		top[5].data[...] = 0
		top[6].data[...] = 0

	def forward(self, bottom, top):

		category = np.random.random(self.batch_size)

		top[4].data[category>=self.sampleRatio[2], ...] = 1
		#top[5].data[np.logical_and(category>=self.sampleRatio[0], category<self.sampleRatio[2]), ...] = 1  ################################################################
		#top[6].data[category<self.sampleRatio[1], ...] = 1  ################################################################

		for itt in range(self.batch_size):						

			if category[itt]>=self.sampleRatio[2]: # landmark dataset
				k=self.posDataList[self.posCounter]			
				flipFlag = np.random.random(1)>=0.5

				if flipFlag:
					tmpImg = self.posDataset[k]['image'].copy()
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					tmpImg = cv2.flip(tmpImg, 1)
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					top[0].data[itt, ...] = tmpImg
				else:
					top[0].data[itt, ...] = self.posDataset[k]['image']

				top[1].data[itt, ...] = self.posDataset[k]['lbl5Points']
				if flipFlag:
					top[1].data[itt, 0:5] = 1.0-top[1].data[itt, 0:5]

				self.posCounter += 1

				if self.posCounter >= self.posLen:
					self.curFileIdx+=1
					if self.curFileIdx>=len(self.fileNumList):
						self.curFileIdx=0
						random.shuffle(self.fileNumList)

					with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
						self.posDataset = pickle.load(f)

					self.posDataList = sorted(self.posDataset.keys())
					random.shuffle(self.posDataList)
					self.posLen = len(self.posDataList)
					self.posCounter = 0	
			elif category[itt]>=self.sampleRatio[1]: # wf part dataset
				k=self.wfPartDataList[self.wfPartCounter]
				flipFlag = np.random.random(1)>=0.5

				if flipFlag:
					tmpImg = self.wfPartDataset[k]['image'].copy()
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					tmpImg = cv2.flip(tmpImg, 1)
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					top[0].data[itt, ...] = tmpImg
				else:
					top[0].data[itt, ...] = self.wfPartDataset[k]['image']

				top[2].data[itt, ...] = self.wfPartDataset[k]['bbox']
				if flipFlag:
					tmpVal = top[2].data[itt, 2]
					top[2].data[itt, 2] = -1.0*top[2].data[itt, 0]
					top[2].data[itt, 0] = -1.0*tmpVal

				self.wfPartCounter += 1

				if self.wfPartCounter >= self.wfPartLen:
					self.curWideFileIdx[2]+=1
					if self.curWideFileIdx[2]>=len(self.partDataFileList):
						self.curWideFileIdx[2]=0
						random.shuffle(self.partDataFileList)

					with open(self.wideFaceDataFile+'part_'+str(self.partDataFileList[self.curWideFileIdx[2]])+'.pkl', 'rb') as f:
						self.wfPartDataset = pickle.load(f)

					self.wfPartDataList = sorted(self.wfPartDataset.keys())
					random.shuffle(self.wfPartDataList)
					self.wfPartLen = len(self.wfPartDataList)
					self.wfPartCounter = 0
			elif category[itt]>=self.sampleRatio[0]: # wf pos dataset
				k=self.wfPosDataList[self.wfPosCounter]
				flipFlag = np.random.random(1)>=0.5

				if flipFlag:
					tmpImg = self.wfPosDataset[k]['image'].copy()
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					tmpImg = cv2.flip(tmpImg, 1)
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					top[0].data[itt, ...] = tmpImg
				else:
					top[0].data[itt, ...] = self.wfPosDataset[k]['image']

				top[2].data[itt, ...] = self.wfPosDataset[k]['bbox']
				if flipFlag:
					tmpVal = top[2].data[itt, 2]
					top[2].data[itt, 2] = -1.0*top[2].data[itt, 0]
					top[2].data[itt, 0] = -1.0*tmpVal

				top[3].data[itt, ...] = 1

				self.wfPosCounter += 1

				if self.wfPosCounter >= self.wfPosLen:
					self.curWideFileIdx[1]+=1
					if self.curWideFileIdx[1]>=len(self.posDataFileList):
						self.curWideFileIdx[1]=0
						random.shuffle(self.posDataFileList)

					with open(self.wideFaceDataFile+'pos_'+str(self.posDataFileList[self.curWideFileIdx[1]])+'.pkl', 'rb') as f:
						self.wfPosDataset = pickle.load(f)

					self.wfPosDataList = sorted(self.wfPosDataset.keys())
					random.shuffle(self.wfPosDataList)
					self.wfPosLen = len(self.wfPosDataList)
					self.wfPosCounter = 0
			else: # wf neg dataset
				k=self.wfNegDataList[self.wfNegCounter]
				flipFlag = np.random.random(1)>=0.5

				if flipFlag:
					tmpImg = self.wfNegDataset[k]['image'].copy()
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					tmpImg = cv2.flip(tmpImg, 1)
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					top[0].data[itt, ...] = tmpImg
				else:
					top[0].data[itt, ...] = self.wfNegDataset[k]['image']

				top[3].data[itt, ...] = 0

				self.wfNegCounter += 1

				if self.wfNegCounter >= self.wfNegLen:
					self.curWideFileIdx[0]+=1
					if self.curWideFileIdx[0]>=len(self.negDataFileList):
						self.curWideFileIdx[0]=0
						random.shuffle(self.negDataFileList)

					with open(self.wideFaceDataFile+'neg_'+str(self.negDataFileList[self.curWideFileIdx[0]])+'.pkl', 'rb') as f:
						self.wfNegDataset = pickle.load(f)

					self.wfNegDataList = sorted(self.wfNegDataset.keys())
					random.shuffle(self.wfNegDataList)
					self.wfNegLen = len(self.wfNegDataList)
					self.wfNegCounter = 0

	def backward(self, top, propagate_down, bottom):
		pass

# output img, cls, bbox, landmark and the mask for cls, bbox and landmark, change fc5 to 1024 output
class Data6_Layer_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.faceDataFile = params["faceDataFile"]
		self.fileNumList = range(1,params["numOfFile"]+1)
		self.filePostfix = params["filePostfix"]		
		self.curFileIdx = 0 # for lardmark dataset

		self.sampleRatio = eval(params["sampleRatio"]) # ratio among [neg,pos,part,lardmark] such as [3,1,1,2]
		self.sampleRatio = np.cumsum(self.sampleRatio)/np.sum(self.sampleRatio).astype(float)
		
		self.wideFaceDataFile = params["wideFaceDataFile"]
		self.numOfWideFaceFile = eval(params["numOfWideFaceFile"]) # number of files for neg, pos and part, such as [13,5,4]
		self.negDataFileList = range(1, self.numOfWideFaceFile[0]+1)
		self.posDataFileList = range(1, self.numOfWideFaceFile[1]+1)
		self.partDataFileList = range(1, self.numOfWideFaceFile[2]+1)
		self.curWideFileIdx = [0,0,0] # for wider face dataset neg, pos and part

		self.net_side = 48

		print "Data_Layer_train, setup"

		random.seed(100)

		# process landmark dataset
		random.shuffle(self.fileNumList)		
		with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
			self.posDataset = pickle.load(f) # this is actually landmark dataset
		self.posDataList = sorted(self.posDataset.keys())
		random.shuffle(self.posDataList)
		self.posLen = len(self.posDataList)
		self.posCounter = 0

		# process wider face neg dataset
		random.shuffle(self.negDataFileList)
		with open(self.wideFaceDataFile+'neg_'+str(self.negDataFileList[self.curWideFileIdx[0]])+'.pkl', 'rb') as f:
			self.wfNegDataset = pickle.load(f) # this is wider face dataset
		self.wfNegDataList = sorted(self.wfNegDataset.keys())
		random.shuffle(self.wfNegDataList)
		self.wfNegLen = len(self.wfNegDataList)
		self.wfNegCounter = 0

		# process wider face pos dataset
		random.shuffle(self.posDataFileList)
		with open(self.wideFaceDataFile+'pos_'+str(self.posDataFileList[self.curWideFileIdx[1]])+'.pkl', 'rb') as f:
			self.wfPosDataset = pickle.load(f) # this is wider face dataset
		self.wfPosDataList = sorted(self.wfPosDataset.keys())
		random.shuffle(self.wfPosDataList)
		self.wfPosLen = len(self.wfPosDataList)
		self.wfPosCounter = 0

		# process wider face part dataset
		random.shuffle(self.partDataFileList)
		with open(self.wideFaceDataFile+'part_'+str(self.partDataFileList[self.curWideFileIdx[2]])+'.pkl', 'rb') as f:
			self.wfPartDataset = pickle.load(f) # this is wider face dataset
		self.wfPartDataList = sorted(self.wfPartDataset.keys())
		random.shuffle(self.wfPartDataList)
		self.wfPartLen = len(self.wfPartDataList)
		self.wfPartCounter = 0


		top[0].reshape(self.batch_size, 3, self.net_side, self.net_side) #image
		top[1].reshape(self.batch_size, 10) # landmark
		top[2].reshape(self.batch_size, 4) # bbox
		top[3].reshape(self.batch_size, 1) # label
		top[4].reshape(self.batch_size, 10) # landmark mask
		top[5].reshape(self.batch_size, 4) # bbox mask
		top[6].reshape(self.batch_size, 2) # class mask

	def reshape(self, bottom, top):
		top[1].data[...] = 0
		top[2].data[...] = 0
		top[3].data[...] = 0
		top[4].data[...] = 0 ################################################################
		top[5].data[...] = 0
		top[6].data[...] = 0

	def forward(self, bottom, top):

		category = np.random.random(self.batch_size)

		top[4].data[category>=self.sampleRatio[2], ...] = 1
		top[5].data[np.logical_and(category>=self.sampleRatio[0], category<self.sampleRatio[2]), ...] = 1  ################################################################
		top[6].data[category<self.sampleRatio[1], ...] = 1  ################################################################

		for itt in range(self.batch_size):						

			if category[itt]>=self.sampleRatio[2]: # landmark dataset
				k=self.posDataList[self.posCounter]			
				#flipFlag = np.random.random(1)>=0.5 #################################################
				flipFlag = 0

				if flipFlag:
					tmpImg = self.posDataset[k]['image'].copy()
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					tmpImg = cv2.flip(tmpImg, 1)
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					top[0].data[itt, ...] = tmpImg
				else:
					top[0].data[itt, ...] = self.posDataset[k]['image']

				top[1].data[itt, ...] = self.posDataset[k]['lbl5Points']
				if flipFlag:
					top[1].data[itt, 0:5] = 1.0-top[1].data[itt, [1,0,2,4,3]]

				self.posCounter += 1

				if self.posCounter >= self.posLen:
					self.curFileIdx+=1
					if self.curFileIdx>=len(self.fileNumList):
						self.curFileIdx=0
						random.shuffle(self.fileNumList)

					with open(self.faceDataFile+str(self.fileNumList[self.curFileIdx])+self.filePostfix+'.pkl', 'rb') as f:
						self.posDataset = pickle.load(f)

					self.posDataList = sorted(self.posDataset.keys())
					random.shuffle(self.posDataList)
					self.posLen = len(self.posDataList)
					self.posCounter = 0	
			elif category[itt]>=self.sampleRatio[1]: # wf part dataset
				k=self.wfPartDataList[self.wfPartCounter]
				#flipFlag = np.random.random(1)>=0.5 ####################################################
				flipFlag = 0

				if flipFlag:
					tmpImg = self.wfPartDataset[k]['image'].copy()
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					tmpImg = cv2.flip(tmpImg, 1)
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					top[0].data[itt, ...] = tmpImg
				else:
					top[0].data[itt, ...] = self.wfPartDataset[k]['image']

				top[2].data[itt, ...] = self.wfPartDataset[k]['bbox']
				if flipFlag:
					tmpVal = top[2].data[itt, 2]
					top[2].data[itt, 2] = -1.0*top[2].data[itt, 0]
					top[2].data[itt, 0] = -1.0*tmpVal

				self.wfPartCounter += 1

				if self.wfPartCounter >= self.wfPartLen:
					self.curWideFileIdx[2]+=1
					if self.curWideFileIdx[2]>=len(self.partDataFileList):
						self.curWideFileIdx[2]=0
						random.shuffle(self.partDataFileList)

					with open(self.wideFaceDataFile+'part_'+str(self.partDataFileList[self.curWideFileIdx[2]])+'.pkl', 'rb') as f:
						self.wfPartDataset = pickle.load(f)

					self.wfPartDataList = sorted(self.wfPartDataset.keys())
					random.shuffle(self.wfPartDataList)
					self.wfPartLen = len(self.wfPartDataList)
					self.wfPartCounter = 0
			elif category[itt]>=self.sampleRatio[0]: # wf pos dataset
				k=self.wfPosDataList[self.wfPosCounter]
				#flipFlag = np.random.random(1)>=0.5 #####################################################
				flipFlag = 0

				if flipFlag:
					tmpImg = self.wfPosDataset[k]['image'].copy()
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					tmpImg = cv2.flip(tmpImg, 1)
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					top[0].data[itt, ...] = tmpImg
				else:
					top[0].data[itt, ...] = self.wfPosDataset[k]['image']

				top[2].data[itt, ...] = self.wfPosDataset[k]['bbox']
				if flipFlag:
					tmpVal = top[2].data[itt, 2]
					top[2].data[itt, 2] = -1.0*top[2].data[itt, 0]
					top[2].data[itt, 0] = -1.0*tmpVal

				top[3].data[itt, ...] = 1

				self.wfPosCounter += 1

				if self.wfPosCounter >= self.wfPosLen:
					self.curWideFileIdx[1]+=1
					if self.curWideFileIdx[1]>=len(self.posDataFileList):
						self.curWideFileIdx[1]=0
						random.shuffle(self.posDataFileList)

					with open(self.wideFaceDataFile+'pos_'+str(self.posDataFileList[self.curWideFileIdx[1]])+'.pkl', 'rb') as f:
						self.wfPosDataset = pickle.load(f)

					self.wfPosDataList = sorted(self.wfPosDataset.keys())
					random.shuffle(self.wfPosDataList)
					self.wfPosLen = len(self.wfPosDataList)
					self.wfPosCounter = 0
			else: # wf neg dataset
				k=self.wfNegDataList[self.wfNegCounter]
				#flipFlag = np.random.random(1)>=0.5 #######################################################
				flipFlag = 0

				if flipFlag:
					tmpImg = self.wfNegDataset[k]['image'].copy()
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					tmpImg = cv2.flip(tmpImg, 1)
					tmpImg = np.swapaxes(tmpImg, 0, 2)
					top[0].data[itt, ...] = tmpImg
				else:
					top[0].data[itt, ...] = self.wfNegDataset[k]['image']

				top[3].data[itt, ...] = 0

				self.wfNegCounter += 1

				if self.wfNegCounter >= self.wfNegLen:
					self.curWideFileIdx[0]+=1
					if self.curWideFileIdx[0]>=len(self.negDataFileList):
						self.curWideFileIdx[0]=0
						random.shuffle(self.negDataFileList)

					with open(self.wideFaceDataFile+'neg_'+str(self.negDataFileList[self.curWideFileIdx[0]])+'.pkl', 'rb') as f:
						self.wfNegDataset = pickle.load(f)

					self.wfNegDataList = sorted(self.wfNegDataset.keys())
					random.shuffle(self.wfNegDataList)
					self.wfNegLen = len(self.wfNegDataList)
					self.wfNegCounter = 0

	def backward(self, top, propagate_down, bottom):
		pass

class Data3_Layer_train(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str_) # use param_str_ here according to the python_layer.hpp used in the system
    	#param2 = params.get('param2', False) #I usually use this when fetching a bool

		self.batch_size = params["batchSize"]
		self.faceDataFile = params["faceDataFile"]
		self.nonFaceDataFile = params["nonFaceDataFile"]
		self.posRatio = params["posRatio"]

		self.net_side = 48

		print "Data_Layer_train, setup"

		with open(self.faceDataFile, 'rb') as f:
			self.posDataset = pickle.load(f)

		with open(self.nonFaceDataFile, 'rb') as f:
			self.negDataset = pickle.load(f)

		self.posDataList = sorted(self.posDataset.keys())
		self.negDataList = sorted(self.negDataset.keys())

		self.posLen = len(self.posDataList)
		self.negLen = len(self.negDataList)
		self.posCounter = 0
		self.negCounter = 0		

		random.seed(100)
		random.shuffle(self.posDataList)
		random.shuffle(self.negDataList)		

		top[0].reshape(self.batch_size, 3, self.net_side, self.net_side)
		top[1].reshape(self.batch_size, 1)
		top[2].reshape(self.batch_size, 10)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):

		for itt in range(self.batch_size):			

			if random.random() < self.posRatio:

				# do padding, resizing and normalization, pts transformation and .reshape((10,))
				top[0].data[itt, ...] = self.posDataset[self.posDataList[self.posCounter]]['image']				
				top[1].data[itt, ...] = 1
				top[2].data[itt, ...] = self.posDataset[self.posDataList[self.posCounter]]['lbl5Points']	

				self.posCounter += 1

				if self.posCounter >= self.posLen:
					self.posCounter = 0
					random.shuffle(self.posDataList)
			else:				

				# do padding, resizing and normalization, pts transformation
				top[0].data[itt, ...] = self.negDataset[self.negDataList[self.negCounter]]['image']	
				top[1].data[itt, ...] = 0
				top[2].data[itt, ...] = 0

				self.negCounter += 1

				if self.negCounter >= self.negLen:
					self.negCounter = 0
					random.shuffle(self.negDataList)

	def backward(self, top, propagate_down, bottom):
		pass


	# obsolete
	def preprocessing(self, image, boundingBox=None, points=None):		
		i_w = image.shape[1]
		i_h = image.shape[0]
		i_c = image.shape[2]

		if boundingBox!=None:
			bBox = boundingBox.copy().astype(int)				
			bBox -= 1 # data from matlab is started from index 1			
		else:
			bBox = np.array([0,0,i_w-1,i_h-1])

		b_w = bBox[2] - bBox[0] + 1 # calculate width and height of boundingBox
		b_h = bBox[3] - bBox[1] + 1
		b_max = np.maximum(b_w, b_h)

		bBox[0] = bBox[0] - (b_max - b_w) / 2 # change boundingBox to square shape
		bBox[1] = bBox[1] - (b_max - b_h) / 2
		bBox[2] = bBox[0] + b_max - 1
		bBox[3] = bBox[1] + b_max - 1
		
		i_x1 = np.maximum(0, bBox[0])# find the area from image used to fill cropImg
		i_y1 = np.maximum(0, bBox[1])
		i_x2 = np.minimum(i_w-1, bBox[2])
		i_y2 = np.minimum(i_h-1, bBox[3])

		cropImg = np.ones((b_max, b_max, i_c)) * 255
		c_x1 = np.maximum(0-bBox[0], 0) # find the area from cropImg to load data from image
		c_y1 = np.maximum(0-bBox[1], 0)
		c_x2 = c_x1 + (i_x2 - i_x1)
		c_y2 = c_y1 + (i_y2 - i_y1)

		cropImg[c_y1:c_y2+1,c_x1:c_x2+1,:] = image[i_y1:i_y2+1,i_x1:i_x2+1,:]
		'''
		cv2.namedWindow("img", cv2.WINDOW_NORMAL)
		cv2.imshow("img",cropImg.astype(np.uint8))
		kVal=cv2.waitKey()
		if kVal == 32: # 32 is space
			cv2.destroyAllWindows()
		'''
		cropImg -= 128 # image normalization
		cropImg /= 255.0
		cropImg = cv2.resize(cropImg,(int(self.net_side),int(self.net_side))) # resize, do this before swapaxes
		cropImg = np.swapaxes(cropImg, 0, 2) # change (h,w,c) to (c,w,h)

		#print "cropImg Size: ", cropImg.shape
		if points!=None:
			pts = points.copy().astype(float)
			pts -= 1 # data from matlab is started from index 1
			pts -= bBox[0:2] # now the pts reference to the top-left conner of square bounding box
			pts *= float(self.net_side)/b_max # scale according to image resize ratio

			return cropImg, pts.reshape((pts.size,))
		else:
			return cropImg


# posDataset['aflw__face_46593.jpg']['lbl5Points']-posDataset['aflw__face_46593.jpg']['boundingBox'][0][0:2]

################################################################################
######################Regression Loss Layer By Python###########################
################################################################################
'''
layer {
  name: "RegressionLoss"
  type: "Python"
  bottom: "fc6-3"
  bottom: "pts"
  bottom: "label"
  top: "PtsLoss"
  propagate_down: 1
  propagate_down: 0
  propagate_down: 0
  loss_weight: 1
}
'''

class clsFilter_Layer(caffe.Layer):
	def setup(self,bottom,top):
		if len(bottom) != 2:
			raise Exception("Need 2 Inputs")

	def reshape(self,bottom,top):
		if bottom[0].count != bottom[1].count:
			raise Exception("Input predict and groundTruth should have same dimension")
		#pts = bottom[1].data
		#self.valid_index = np.where(roi[:,0] != -1)[0]
		#self.N = len(self.valid_index)
		top[0].reshape(len(bottom[0].data), 2)
		#top[0].reshape(bottom[0].data.shape)

	def forward(self,bottom,top):
		top[0].data[...] = bottom[0].data[...]

		'''
		print 'bottom[0].data shape', bottom[0].data.shape
		print 'bottom[1].data shape', bottom[1].data.shape
		print 'top[0].data shape', top[0].data.shape
		
		assert False
		'''

	def backward(self,top,propagate_down,bottom):
		#pass		
		bottom[0].diff[...] = top[0].diff[...] * bottom[1].data[...]
		#print top[0].diff[...]
		#print bottom[0].diff[...]
		#print bottom[1].data[...]

class regression_Layer(caffe.Layer):
	def setup(self,bottom,top):
		if len(bottom) != 2:
			raise Exception("Need 2 Inputs")

	def reshape(self,bottom,top):
		if bottom[0].count != bottom[1].count:
			raise Exception("Input predict and groundTruth should have same dimension")
		#pts = bottom[1].data
		#self.valid_index = np.where(roi[:,0] != -1)[0]
		#self.N = len(self.valid_index)
		self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
		top[0].reshape(1)

	def forward(self,bottom,top):
		self.diff[...] = 0
		top[0].data[...] = 0

		self.diff[...] = bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape) 
		top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

		'''
		print 'bottom[2].data shape', bottom[2].data
		print 'self.diff shape', self.diff
		print 'top[0].data', top[0].data[...]

		assert False
		'''

	def backward(self,top,propagate_down,bottom):
		for i in range(2):
			if not propagate_down[i]:
				continue
			if i == 0:
				sign = 1
			else:
				sign = -1
			bottom[i].diff[...] = sign * self.diff / bottom[i].num


# only cal loss for worst ?% cases
class regression_hard_Layer(caffe.Layer):
	def setup(self,bottom,top):
		if len(bottom) != 2:
			raise Exception("Need 2 Inputs")

		params = eval(self.param_str_)
		self.ratio = params["ratio"]

		if self.ratio <=0 or self.ratio>1:
			raise Exception("ratio need to be in range (0,1]")

		self.hardSampleNum=int((1-self.ratio)*bottom[0].num)

	def reshape(self,bottom,top):
		if bottom[0].count != bottom[1].count:
			raise Exception("Input predict and groundTruth should have same dimension")
		#pts = bottom[1].data
		#self.valid_index = np.where(roi[:,0] != -1)[0]
		#self.N = len(self.valid_index)
		self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
		top[0].reshape(1)

	def forward(self,bottom,top):
		self.diff[...] = 0
		top[0].data[...] = 0

		self.diff[...] = bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape) 
		top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

		ind = np.argpartition(np.sum(self.diff[...]**2,axis=1), self.hardSampleNum)[0:self.hardSampleNum]
		self.diff[ind,...] = 0

		'''
		print 'bottom[2].data shape', bottom[2].data
		print 'self.diff shape', self.diff
		print 'top[0].data', top[0].data[...]

		assert False
		'''

	def backward(self,top,propagate_down,bottom):
		for i in range(2):
			if not propagate_down[i]:
				continue
			if i == 0:
				sign = 1
			else:
				sign = -1
			bottom[i].diff[...] = sign * self.diff / bottom[i].num

class regression3_Layer(caffe.Layer):
	def setup(self,bottom,top):
		if len(bottom) != 3:
			raise Exception("Need 3 Inputs")

	def reshape(self,bottom,top):
		if bottom[0].count != bottom[1].count:
			raise Exception("Input predict and groundTruth should have same dimension")
		#pts = bottom[1].data
		#self.valid_index = np.where(roi[:,0] != -1)[0]
		#self.N = len(self.valid_index)
		self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
		top[0].reshape(1)

	def forward(self,bottom,top):
		self.diff[...] = 0
		top[0].data[...] = 0

		self.diff[...] = (bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape)) * bottom[2].data
		top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

		'''
		print 'bottom[2].data shape', bottom[2].data
		print 'self.diff shape', self.diff
		print 'top[0].data', top[0].data[...]

		assert False
		'''

	def backward(self,top,propagate_down,bottom):
		for i in range(2):
			if not propagate_down[i]:
				continue
			if i == 0:
				sign = 1
			else:
				sign = -1
			bottom[i].diff[...] = sign * self.diff / bottom[i].num