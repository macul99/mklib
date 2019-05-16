from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mxnet import io
import numpy as np
import logging
import mxnet as mx
import numbers
from mxnet import ndarray as nd
import random
import multiprocessing
import cv2

class ImageMultiLabelIter(io.DataIter):
    def __init__(self, rc_iter, label_len=[1,10], label_name=['softmax_label', 'landmark_gt'], **kwargs):
        super(ImageMultiLabelIter, self).__init__()
        assert len(label_len)==len(label_name)        
        self.iter = rc_iter
        self.label_len = label_len
        self.label_name = label_name
    def next(self):
        batch = self.iter.next()
        assert batch.label[0].shape[1]==np.sum(self.label_len)
        data = []
        for dt in batch.data:
            data.append(dt)
        label = []
        idx = 0
        for ll in self.label_len:
            for lb in batch.label:
                label.append(lb[:,idx:idx+ll])
            idx += ll
        return io.DataBatch(data=data, label=label)
    def reset(self):
        self.iter.reset()
    @property
    def provide_data(self):
        return self.iter.provide_data
    @property
    def provide_label(self):
        label = self.iter.provide_label[0]
        new_label = []
        for i, lb_nm in enumerate(self.label_name):
            new_label.append(io.DataDesc(name=lb_nm, shape=(label.shape[0], long(self.label_len[i])), 
                                         dtype=label.dtype, layout=label.layout))
        return new_label

class ImageMultiLabelIter_IndividualLabel(io.DataIter):
    def __init__(self, rc_iter, label_idx=0, label_len=[1,10], label_name=['softmax_label', 'landmark_gt'], **kwargs):
        super(ImageMultiLabelIter_IndividualLabel, self).__init__()
        assert len(label_len)==len(label_name)
        assert label_idx>=0
        assert label_idx<len(label_len)        
        self.iter = rc_iter
        self.label_len = label_len
        self.label_name = label_name
        self.label_idx = label_idx
    def next(self):
        batch = self.iter.next()
        assert batch.label[0].shape[1]==np.sum(self.label_len)
        data = []
        for dt in batch.data:
            data.append(dt)
        label = []
        idx = 0
        for i in range(self.label_idx):
            idx += self.label_len[i]
        ll = self.label_len[self.label_idx]
        for lb in batch.label:
            label.append(lb[:,idx:idx+ll])
        return io.DataBatch(data=data, label=label)
    def reset(self):
        self.iter.reset()
    @property
    def provide_data(self):
        return self.iter.provide_data
    @property
    def provide_label(self):
        label = self.iter.provide_label[0]
        new_label = []
        lb_nm = self.label_name[self.label_idx]
        new_label.append(io.DataDesc(name=lb_nm, shape=(label.shape[0], long(self.label_len[self.label_idx])), 
                                     dtype=label.dtype, layout=label.layout))
        return new_label

class MyImageIter(io.DataIter):

    def __init__(self, batch_size, data_shape,
                 path_imgrec,
                 shuffle=True, 
                 mean = [127.5, 127.5, 127.5], # in order of R,G,B
                 data_name='data',                  
                 label_name=['softmax_label', 'landmark_gt'],
                 label_len=[1,10],
                 images_filter = 0,
                 rand_mirror = False,
                 cutoff = 0,
                 p_glasses = 0.0, # value in [0,1]
                 aug_list = ['brightness','saturation','contrast','color','pca_noise'], # 'brightness','saturation','contrast','color','pca_noise'
                 **kwargs):
        super(MyImageIter, self).__init__()
        assert path_imgrec
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.path_imgrec = path_imgrec
        self.shuffle = shuffle
        self.mean = mean
        self.rand_mirror = rand_mirror
        self.images_filter = images_filter
        self.cutoff = cutoff
        self.p_glasses = p_glasses
        self.glasses_cx_range = (-3, 3)
        self.glasses_cy_range = (0, 5)
        self.glasses_height_range = (10,20)
        self.glasses_width_range = (20,30)
        self.data_name = data_name
        self.label_name = label_name
        self.label_len = label_len
        if 'landmark_gt' in self.label_name:
            self.leye_lbl_pos = int(np.sum(self.label_len[0:self.label_name.index('landmark_gt')]))
            self.reye_lbl_pos = self.leye_lbl_pos + 2

        logging.info('loading recordio %s...', path_imgrec)
        path_imgidx = path_imgrec[0:-4]+".idx"
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if not isinstance(header.label, numbers.Number) and len(header.label)!=np.sum(self.label_len):
            print('header0 label', header.label)
            self.imgidx, self.id2range = self.build_idx(header)
            print('id2range', len(self.id2range))
        else:
            self.imgidx = list(self.imgrec.keys)
        self.seq = self.imgidx
        
        self.nd_mean = None
        if self.mean:
            self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
            self.nd_mean = mx.nd.array(self.mean).reshape((1,1,3))

        self.check_data_shape(self.data_shape)
        
        self.image_size = '%d,%d'%(self.data_shape[1],self.data_shape[2])
        self.HFA = mx.image.HorizontalFlipAug(p=1)
        self.ColorJA = mx.image.ColorJitterAug(0.5, 0.5, 0.5)
        self.BJA = mx.image.BrightnessJitterAug(brightness=0.5)
        self.ContrastJA = mx.image.ContrastJitterAug(contrast=0.5)
        self.SJA = mx.image.SaturationJitterAug(saturation=0.5)
        self.LA = mx.image.LightingAug(alphastd=0.5, eigval=np.asarray([1,1,1]), eigvec=np.ones((3,3)))
        self.aug_list = []
        for aug in aug_list:
            if aug == 'color':
                self.aug_list.append(self.ColorJA)
            elif aug == 'brightness':
                self.aug_list.append(self.BJA)
            elif aug == 'saturation':
                self.aug_list.append(self.SJA)
            elif aug == 'contrast':
                self.aug_list.append(self.ContrastJA)
            elif aug == 'pca_noise':
                self.aug_list.append(self.LA)
            else:
                assert False, 'aug list not supported!!!'
        self.augmentor = mx.image.RandomOrderAug(self.aug_list)
        self.cur = 0
        self.nbatch = 0
        self.is_init = False        

        self.landmark_ref = np.array([  [30.2946 + 8, 51.6963],
                                        [65.5318 + 8, 51.5014],
                                        [48.0252 + 8, 71.7366],
                                        [33.5493 + 8, 92.3655],
                                        [62.7299 + 8, 92.2041] ])

    def build_idx(self, header):
        imgidx = []
        id2range = {}
        for identity in range(int(header.label[0]), int(header.label[1])):
            s = self.imgrec.read_idx(identity)
            header, _ = mx.recordio.unpack(s)
            a,b = int(header.label[0]), int(header.label[1])
            count = b-a
            if count<self.images_filter:
                continue
            id2range[identity] = (a,b)
            imgidx += range(a, b)
        return imgidx, id2range

    @property
    def provide_data(self):
        return [(self.data_name, (self.batch_size,) + self.data_shape)]

    @property
    def provide_label(self):
        new_label = []
        for i, lb_nm in enumerate(self.label_name):
            new_label.append(io.DataDesc(name=lb_nm, shape=(self.batch_size, long(self.label_len[i]))))
        return new_label

    @property
    def num_samples(self):
        return len(self.seq)

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = 0
        if self.shuffle:
          random.shuffle(self.seq)
        if self.seq is None and self.imgrec is not None:
            self.imgrec.reset()    

    def next_sample(self):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        while True:
            if self.cur >= len(self.seq):
                print('len seq, ', len(self.seq))
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            s = self.imgrec.read_idx(idx)
            header, img = mx.recordio.unpack(s)
            img = mx.image.imdecode(img)
            label = header.label
            if isinstance(label, numbers.Number):
                label = [label]
            #print('label: ', label)
            return label, img

    def next(self):
        if not self.is_init:
            self.reset()
            self.is_init = True
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        self.nbatch+=1
        c, h, w = self.data_shape
        batch_data = nd.empty((self.batch_size, c, h, w))
        batch_label = []
        batch_buf = {}
        for ix, ln in enumerate(self.label_name):
            batch_buf[ln] = nd.empty((self.batch_size, self.label_len[ix]))
        i = 0
        try:
            while i < self.batch_size:
                _label, _data = self.next_sample()
                _data, _label = self.image_augmentation(_data, _label)
                try:
                    self.check_valid_image(_data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                #print('aa',data[0].shape)
                #data = self.augmentation_transform(data)
                #print('bb',data[0].shape)
                batch_data[i][:] = self.postprocess_data(_data)

                idx = 0
                
                for j, ll in enumerate(self.label_len):
                    batch_buf[self.label_name[j]][i][:] = _label[idx:idx+ll]
                    idx += ll
                i += 1
            
            for ln in self.label_name:
                batch_label.append(batch_buf[ln])
        except StopIteration:
            if i<self.batch_size:
                raise StopIteration
        #print(batch_label)
        return io.DataBatch([batch_data], batch_label, self.batch_size - i)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data.shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def image_augmentation(self, img, label=None):
        if img.shape[0]!=self.data_shape[1]:
            img = mx.image.resize_short(img, self.data_shape[1])
        if random.random() < self.p_glasses:
            if type(label) != type(None) and 'landmark_gt' in self.label_name:
                leye_pos = [label[self.leye_lbl_pos]*self.data_shape[2]+self.landmark_ref[0,0], label[self.leye_lbl_pos+1]*self.data_shape[1]+self.landmark_ref[0,1]]
                reye_pos = [label[self.reye_lbl_pos]*self.data_shape[2]+self.landmark_ref[1,0], label[self.reye_lbl_pos+1]*self.data_shape[1]+self.landmark_ref[1,1]]
                img = self.draw_glasses(img,leye_pos,reye_pos)
            else:
                img = self.draw_glasses(img,self.landmark_ref[0],self.landmark_ref[1])
        if self.rand_mirror:
            _rd = random.randint(0,1)
            if _rd==1:
                img = self.HFA(img)
                if type(label) != type(None) and 'landmark_gt' in self.label_name:
                    label = label.copy()
                    lb_idx = self.label_name.index('landmark_gt')
                    lb_idx = np.sum(self.label_len[0:lb_idx])
                    for i in range(self.landmark_ref.shape[0]):
                        label[lb_idx+i*2] = -1.0 *  label[lb_idx+i*2]
        if self.cutoff>0:
            _rd = random.randint(0,1)
            if _rd==1:
                #print('do cutoff aug', self.cutoff)
                centerh = random.randint(0, img.shape[0]-1)
                centerw = random.randint(0, img.shape[1]-1)
                half = self.cutoff//2
                starth = max(0, centerh-half)
                endh = min(img.shape[0], centerh+half)
                startw = max(0, centerw-half)
                endw = min(img.shape[1], centerw+half)
                #print(starth, endh, startw, endw, _data.shape)
                img[starth:endh, startw:endw, :] = 128
        if len(self.aug_list):
            img = self.augmentor(img.astype("float32"))
        if self.nd_mean is not None:
            img = img.astype('float32', copy=False)
            img -= self.nd_mean
            img *= 0.0078125                
        return img, label

    def draw_glasses(self, img, leye, reye):        
        leye = np.array(leye)
        reye = np.array(reye)
        img1 = img.asnumpy()
        img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
        glass_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))        
        glass_type = random.randint(0,2)
        if glass_type == 0:
            glass_width = random.randint(1,2)
            x_delta = random.randint(*self.glasses_cx_range)
            y_delta = random.randint(*self.glasses_cy_range)
            x_len = random.randint(*self.glasses_width_range)
            y_len = random.randint(*self.glasses_height_range)
            ul_x = np.max([leye[0]+x_delta-0.5*x_len, 0]).astype(np.uint8)
            ul_y = np.max([leye[1]+y_delta-0.5*y_len, 0]).astype(np.uint8)
            br_x = np.min([leye[0]+x_delta+0.5*x_len, self.data_shape[2]-1]).astype(np.uint8)
            br_y = np.min([leye[1]+y_delta+0.5*y_len, self.data_shape[1]-1]).astype(np.uint8)
            cv2.rectangle(img1, tuple([ul_x,ul_y]), tuple([br_x,br_y]), glass_color, glass_width)
            ul_x = np.max([reye[0]+x_delta-0.5*x_len, 0]).astype(np.uint8)
            ul_y = np.max([reye[1]+y_delta-0.5*y_len, 0]).astype(np.uint8)
            br_x = np.min([reye[0]+x_delta+0.5*x_len, self.data_shape[2]-1]).astype(np.uint8)
            br_y = np.min([reye[1]+y_delta+0.5*y_len, self.data_shape[1]-1]).astype(np.uint8)
            cv2.rectangle(img1, tuple([ul_x,ul_y]), tuple([br_x,br_y]), glass_color, glass_width)
        elif glass_type == 1:
            glass_width = random.randint(2,3)
            x_delta = random.randint(*self.glasses_cx_range)
            y_delta = random.randint(*self.glasses_cy_range)
            r_len = int(np.mean([random.randint(*self.glasses_width_range),random.randint(*self.glasses_height_range)])*0.7)
            cv2.circle(img1, tuple([int(leye[0]+x_delta),int(leye[1]+y_delta)]), r_len, glass_color, glass_width)
            cv2.circle(img1, tuple([int(reye[0]+x_delta),int(reye[1]+y_delta)]), r_len, glass_color, glass_width)
        elif glass_type == 2:
            glass_width = random.randint(1,2)
            x_delta = random.randint(*self.glasses_cx_range)
            y_delta = random.randint(*self.glasses_cy_range)-2
            x_len = int(random.randint(*self.glasses_width_range)*0.7)
            y_len = int(random.randint(*self.glasses_height_range)*0.75)
            cv2.ellipse(img1, tuple([int(leye[0]+x_delta),int(leye[1]+y_delta)]), tuple([x_len, y_len]), 0, 0, 360, glass_color, glass_width)
            cv2.ellipse(img1, tuple([int(reye[0]+x_delta),int(reye[1]+y_delta)]), tuple([x_len, y_len]), 0, 0, 360, glass_color, glass_width)
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        return mx.nd.array(img1)

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))


class MyImageIter_IndividualLabel(MyImageIter):
    def __init__(self, batch_size, data_shape,
                 path_imgrec,
                 shuffle=True, 
                 mean = [127.5, 127.5, 127.5], # in order of R,G,B
                 data_name='data',                  
                 label_name=['softmax_label', 'landmark_gt'],
                 label_len=[1,10],
                 images_filter = 0,
                 rand_mirror = False,
                 cutoff = 0,
                 p_glasses = 0.0, # value in [0,1]
                 aug_list = ['brightness','saturation','contrast','color','pca_noise'], # 'brightness','saturation','contrast','color','pca_noise'
                 label_idx=0,
                 **kwargs):
        super(MyImageIter_IndividualLabel, self).__init__( batch_size, 
                                                             data_shape,
                                                             path_imgrec,
                                                             shuffle=shuffle, 
                                                             mean=mean,
                                                             data_name=data_name,                  
                                                             label_name=label_name,
                                                             label_len=label_len,
                                                             images_filter=images_filter,                                                             
                                                             rand_mirror=rand_mirror, 
                                                             cutoff=cutoff,
                                                             p_glasses=p_glasses,
                                                             aug_list=aug_list
                                                             )
        assert label_idx>=0
        assert label_idx<len(label_len)  

        self.label_idx = label_idx

    @property
    def provide_label(self):
        new_label = []
        lb_nm = self.label_name[self.label_idx]
        new_label.append(io.DataDesc(name=lb_nm, shape=(self.batch_size, long(self.label_len[self.label_idx]))))
        return new_label

    def next(self):
        if not self.is_init:
            self.reset()
            self.is_init = True
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        self.nbatch+=1
        c, h, w = self.data_shape
        batch_data = nd.empty((self.batch_size, c, h, w))
        batch_label = nd.empty((self.batch_size, self.label_len[self.label_idx]))

        idx = 0
        for ii in range(self.label_idx):
            idx += self.label_len[ii]
        ll = self.label_len[self.label_idx]
        i = 0
        try:
            while i < self.batch_size:
                _label, _data = self.next_sample()
                _data, _label = self.image_augmentation(_data, _label)
                try:
                    self.check_valid_image(_data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                #print('aa',data[0].shape)
                #data = self.augmentation_transform(data)
                #print('bb',data[0].shape)
                batch_data[i][:] = self.postprocess_data(_data)
                
                batch_label[i][:] = _label[idx:idx+ll]
                i += 1
        except StopIteration:
            if i<self.batch_size:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], self.batch_size - i)

'''
# test image augmentation
import sys
sys.path.append('/home/macul/libraries/mk_utils/mklib/nn/')
import mxnet as mx
from mxiter.mxiter import ImageMultiLabelIter, MyImageIter, MyImageIter_IndividualLabel
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
def plot_mx_array(array):
    """
    Array expected to be height x width x 3 (channels), and values are floats between 0 and 255.
    """
    assert array.shape[2] == 3, "RGB Channel should be last"
    imshow((array.clip(0, 255)/255).asnumpy())
    plt.show()

trainIter = MyImageIter_IndividualLabel(
                        batch_size=1, 
                        data_shape=(3, 112, 112),
                        path_imgrec='/media/macul/black/face_database_raw_data/mscelb_from_insightface/rec/test.rec',
                        shuffle=False, 
                        mean = None,
                        data_name='data',                  
                        label_name=['softmax_label', 'landmark_gt'],
                        label_len=[1,10],
                        p_glasses=1.0,
                        aug_list = ['brightness','saturation','contrast','color','pca_noise'],
                        cutoff = 28,
                        rand_mirror = False)
batch=trainIter.next()
example_image = batch.data[0][0,:,:,:]
example_image = mx.nd.transpose(example_image, axes=(1,2,0))
#example_image = mx.image.imread("/home/macul/2018-10-15.png")
#example_image = example_image.astype("float32")
plot_mx_array(example_image)

trainIter = MyImageIter_IndividualLabel(
                        batch_size=1, 
                        data_shape=(3, 112, 112),
                        path_imgrec='/media/macul/black/face_database_raw_data/mscelb_from_insightface/rec/test.rec',
                        shuffle=False, 
                        mean = None,
                        data_name='data',                  
                        label_name=['softmax_label', 'landmark_gt'],
                        label_len=[1,10],
                        p_glasses=0.0,
                        aug_list = ['brightness','saturation','contrast','color','pca_noise'],
                        cutoff = 0,
                        rand_mirror = False)
batch=trainIter.next()
example_image = batch.data[0][0,:,:,:]
example_image = mx.nd.transpose(example_image, axes=(1,2,0))
plot_mx_array(example_image)
'''