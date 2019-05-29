# train my net
import caffe
from os import listdir
import cv2
import numpy as np
import cPickle as pickle


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')


caffe.set_device(1)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

model_def='/home/macul/Projects/300W/myMtcnnTest.prototxt'
model_weights='/home/macul/Projects/300W/mymtcnn_sgd_m10p10_lr10_newloss.caffemodel'
#model_def='/home/macul/Projects/300W/48net.prototxt'
#model_weights='/home/macul/Projects/300W/48net.caffemodel'
net=caffe.Net(model_def,model_weights,caffe.TEST)

with open('/home/macul/Projects/300W/testDataset_1.pkl', 'rb') as f:
  testDataDic = pickle.load(f)

threshold = 0.7
keys = testDataDic.keys()
errorDic = {}

for k in keys:
  net.blobs['data'].reshape(1, 3, 48, 48)
  net.blobs['data'].data[...] = testDataDic[k]['image']
  out = net.forward()

  score = out['prob1'][:,1]
  maxScoreIdx = score.argmax()

  if score[maxScoreIdx] > threshold:
    points = out['conv6-3'][maxScoreIdx]
    interocular_distance = np.linalg.norm(np.array([testDataDic[k]['lbl5Points'][0],testDataDic[k]['lbl5Points'][5]]) - np.array([testDataDic[k]['lbl5Points'][1],testDataDic[k]['lbl5Points'][6]]))
    errorDic[k] = np.linalg.norm(testDataDic[k]['lbl5Points'] - points)/interocular_distance
  else:
    errorDic[k] = -1 # no face detected


print np.mean(errorDic.values())

'''

solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver('./mydet32Solver.prototxt')
solver.net.copy_from(model_weights)

solver.net.forward()  # train net
#solver.test_nets[0].forward()  # test net (there can be more than one)
'''
'''
model_def1='/home/macul/Projects/mtcnn-caffe/48net/48net.prototxt'
model_weights1='/home/macul/Projects/mtcnn-caffe/48net/48net.caffemodel'
net1=caffe.Net(model_def1,model_weights1,caffe.TEST)
'''

'''
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)

net.blobs['data'].data[...] = transformed_image

'''
'''
fn=listdir("/home/macul/Projects/300W/labeled_data/nonFace")
for i in range(len(fn)):
	img=cv2.imread("/home/macul/Projects/300W/labeled_data/nonFace/"+fn[i])
	print fn[i]
	print img.shape


with open('/home/macul/Projects/300W/trainDataset_1_1_celebA_zf_cropOnly.pkl', 'rb') as f:
	resultDic = pickle.load(f)

### run the following in the command line
# in myMtcnnTrain_solver.prototxt change to: train_net : "myMtcnnTrain_freeze.prototxt"
caffe train -solver /home/macul/Projects/300W/myMtcnnTrain_solver.prototxt -weights /home/macul/Projects/300W/48net.caffemodel -gpu 1
# save the last model as 48net_freeze.caffemodel
# in myMtcnnTrain_solver.prototxt change to: train_net : "myMtcnnTrain_learnall.prototxt"
# in myMtcnnTrain_solver.prototxt change the solver type as desired
caffe train -solver /home/macul/Projects/300W/myMtcnnTrain_solver.prototxt -weights /home/macul/Projects/300W/48net_freeze.caffemodel -gpu 1
# rename the last model as mymtcnn_sgd.caffemodel if solver type is SGD
'''
from compareFaceAlignModel import Dataset, DatasetCelebA, DlibShapePredictor, mtcnn
dataset=DatasetCelebA('/home/macul/Projects/300W/labeled_data/CelebA_Align img_align_celeba_png 2', 'list_landmarks_align_celeba.txt',None)


import caffe
model_def='/home/macul/Projects/300W/48net.prototxt'
model_weights='/home/macul/Projects/300W/48net.caffemodel'
net=caffe.Net(model_def,model_weights,caffe.TEST)

import caffe
model_def='/home/macul/Projects/mtcnn-caffe/48net/train48.prototxt'
model_weights='/home/macul/Projects/300W/48net.caffemodel'
net=caffe.Net(model_def,model_weights,caffe.TRAIN)

'''
import caffe
model_def1='/home/macul/Projects/300W/48net.prototxt'
model_weights1='/home/macul/Projects/300W/sp_40_8_iter_30000.caffemodel'
net1=caffe.Net(model_def1,model_weights1,caffe.TRAIN)

model_def2='/home/macul/Projects/300W/snapshot_40_8_3/fc256_org_relpos_freeze.prototxt'
model_weights2='/home/macul/Projects/300W/snapshot_40_8_3/_iter_180000.caffemodel'
net2=caffe.Net(model_def2,model_weights2,caffe.TRAIN)

'''
import caffe
model_def='/home/macul/Projects/mtcnn_ego/ego/landmark_mtcnn/models/det3.prototxt'
model_weights='/home/macul/Projects/mtcnn_ego/ego/landmark_mtcnn/models/det3.caffemodel'
net=caffe.Net(model_def,model_weights,caffe.TEST)

model_def1='/home/macul/Projects/mtcnn_ego/ego/landmark_mtcnn/models/mk.prototxt'
model_weights1='/home/macul/Projects/mtcnn_ego/ego/landmark_mtcnn/models/mk.caffemodel'
net1=caffe.Net(model_def1,model_weights1,caffe.TEST)

for nm in net1.params.items():  
  print(nm[0])
  net.params[nm[0]][0].data[...] = net1.params[nm[0]][0].data[...]
  if 'conv' in nm[0]:
    net.params[nm[0]][1].data[...] = net1.params[nm[0]][1].data[...]

net.save('/home/macul/Projects/mtcnn_ego/ego/landmark_mtcnn/models/det3_zf.caffemodel')