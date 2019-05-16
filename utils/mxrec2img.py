# convert rec file to image
import sys
from os.path import isdir, isfile, join
from os import mkdir
import cv2
import numpy as np
from mklib.nn.mxiter.mxiter import MyImageIter
import mxnet as mx
import argparse
from imutils import paths
import progressbar

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--src", required=True, help="path to source rec file")
ap.add_argument("-d", "--dest", required=True, help="path to destination folder")
args = vars(ap.parse_args())


if not isfile(args['src']):
    assert False, 'source rec file not exist!!!'

if not isdir(args['dest']):
    mkdir(args['dest'])

trainIter = mx.io.ImageRecordIter(
                                    path_imgrec=args['src'],
                                    data_shape=(3,112,112),
                                    batch_size=1,
                                    preprocess_threads=2)

record = mx.recordio.MXIndexedRecordIO(args['src'].split('.')[0]+'.idx', args['src'], 'r')
widgets = ["Building List: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(record.keys), widgets=widgets).start()

counter = 0
while True:
    try:
        batch = trainIter.next()
    except StopIteration:
        break

    counter += 1
    pbar.update(counter)

    if int(batch.label[0].asnumpy()[0]) <= 85717:
        continue
    label_name = '{0:07d}'.format(int(batch.label[0].asnumpy()[0]))
    img = mx.nd.transpose(batch.data[0][0,:,:,:],(1,2,0)).asnumpy().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(label_name)
    #print(img.shape)
    folder_name = join(args['dest'],label_name)
    try:        
        #print(folder_name)
        if not isdir(folder_name):
            mkdir(folder_name)
            cv2.imwrite(join(folder_name, '0000.png'),img)
        else:
            cv2.imwrite(join(folder_name, '{0:04d}'.format(len(list(paths.list_images(folder_name))))+'.png'),img)
    except:
        pass


    

pbar.finish()