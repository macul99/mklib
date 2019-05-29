# change and save net
import caffe

model_def = '/home/macul/Projects/300W/48net.prototxt'
model_def1 = '/home/macul/Projects/300W/snapshot_30/3loss_3cls_freeze_test.prototxt'
model_weights = '/home/macul/Projects/300W/48net.caffemodel'

net = caffe.Net(model_def, model_weights,caffe.TEST)
net1= caffe.Net(model_def1,model_weights,caffe.TEST)

net1.params['conv6-11'][0].data[0:2,:]=net.params['conv6-1'][0].data
net1.params['conv6-11'][1].data[0:2]=net.params['conv6-1'][1].data

net1.save('/home/macul/Projects/300W/48net_ignoreLabel.caffemodel')

'''
exec(open('updateNet.py','rb').read())
'''