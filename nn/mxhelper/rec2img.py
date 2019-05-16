from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import multiprocessing


class Rec2Img():
    @staticmethod
    def parse_args(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-s", "--src", required=True, help="path to source rec file")
        ap.add_argument("-d", "--dest", required=True, help="path to destination folder")
        ap.add_argument("-c", "--channel", type=int, default=3, help="image channel number")
        ap.add_argument("-h", "--height", type=int, default=112, help="image height")
        ap.add_argument("-w", "--width", type=int, default=112, help="image width")
        ap.add_argument("-t", "--thread", type=int, default=2, help="number of processing threads")
        ap.add_argument("-f", "--function", default='rec2img', help="function call")
        args = vars(ap.parse_args())
        args.src = os.path.abspath(args.src)
        args.dest = os.path.abspath(args.dest)
        return args

    @staticmethod
    def rec2img(self, src, dest, img_ch, img_height, img_width, threads=2, max_img_cnt=None):
        if not isfile(src):
            assert False, 'source rec file not exist!!!'

        if not isdir(dest):
            mkdir(dest)

        trainIter = mx.io.ImageRecordIter(
                                            path_imgrec=src,
                                            data_shape=(img_ch,img_height,img_width),
                                            batch_size=1,
                                            preprocess_threads=threads)

        record = mx.recordio.MXIndexedRecordIO(src.split('.')[0]+'.idx', src, 'r')
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

            if type(max_img_cnt) != type(None):
                if int(batch.label[0].asnumpy()[0]) <= max_img_cnt: # for insightface rec file, cannot get images more than this number
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
                    assert len(list(paths.list_images(folder_name))) > 9990
                    cv2.imwrite(join(folder_name, '{0:04d}'.format(len(list(paths.list_images(folder_name))))+'.png'),img)
            except:
                pass

        pbar.finish()

    # for insightface rec file
    @staticmethod
    def rec2img_insightface(self, src, dest, img_ch=3, img_height=112, img_width=112, threads=2):
        Rec2Img.Rec2Img.rec2img(src, dest, channel, height, width, thread, max_img_cnt=85717)

if __name__ == '__main__':
    args = Rec2Img.parse_args()

    if args.function == 'rec2img':
        # python -m rec2img -s /src -d /dst -c 3 -h 112 -w 112
        Rec2Img.rec2img(args.src, args.dest, args.channel, args.height, args.width, args.thread)
    elif args.function == 'rec2img_insightface':
        # python -m rec2img -s /media/macul/black/face_database_raw_data/faces_emore/train.rec -d /dst -f rec2img_insightface
        Rec2Img.rec2img_insightface(args.src, args.dest)   