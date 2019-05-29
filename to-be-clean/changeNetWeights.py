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