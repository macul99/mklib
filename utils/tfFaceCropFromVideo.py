from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits import mplot3d
#from sklearn.cluster import KMeans
#from scipy.interpolate import Rbf, interp2d
#from scipy import fftpack, ndimage
from scipy.optimize import minimize
from tfFaceDet import tfFaceDet
from tfMtcnnFaceDet import tfMtcnnFaceDet
import time
import os
from os.path import isdir
from os import mkdir, listdir
import argparse
from spoofing_lbp.SpoofDspTf import SpoofDspTf
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0,"

def prepare_img(img, bbox, crop_scale_to_bbox=2.2, crop_square=False):
    img_h, img_w, _ = img.shape
    bbx_x, bbx_y, x2, y2 = bbox
    bbx_w = x2 - bbx_x + 1
    bbx_h = y2 - bbx_y + 1
    
    #print('orig_img_shape: ', img.shape)

    crp_w = int(bbx_w*crop_scale_to_bbox)
    crp_h = int(bbx_h*crop_scale_to_bbox)

    if crop_square:
        crp_w = max(crp_w, crp_h)
        crp_h = crp_w

    img_crop = np.zeros([crp_h, crp_w, 3]).astype(np.uint8)
    img_crop[:,:,1] = 255 # make empty portion green to differentiate with black

    
    crp_x1 = max(0, int(bbx_x-(crp_w-bbx_w)/2.0))
    crp_y1 = max(0, int(bbx_y-(crp_h-bbx_h)/2.0))
    crp_x2 = min(int(bbx_x-(crp_w-bbx_w)/2.0)+crp_w-1, img_w-1)
    crp_y2 = min(int(bbx_y-(crp_h-bbx_h)/2.0)+crp_h-1, img_h-1)

    delta_x1 = -min(0, int(bbx_x-(crp_w-bbx_w)/2.0))
    delta_y1 = -min(0, int(bbx_y-(crp_h-bbx_h)/2.0))

    img_crop[delta_y1:delta_y1+crp_y2-crp_y1+1, delta_x1:delta_x1+crp_x2-crp_x1+1] = img[crp_y1:crp_y2+1,crp_x1:crp_x2+1,:].copy()

    bbx_x1 = bbx_x - crp_x1 + delta_x1
    bbx_y1 = bbx_y - crp_y1 + delta_y1
    bbx_x2 = bbx_x1 + bbx_w -1
    bbx_y2 = bbx_y1 + bbx_h -1

    #img_crop=cv2.rectangle(img_crop, (bbx_x1, bbx_y1), (bbx_x2, bbx_y2), (0,255,0), 3)
    
    return img_crop, [bbx_x1, bbx_y1, bbx_x2, bbx_y2]

ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--prefix", required=True, help="prefix of saved image name")
ap.add_argument("-v", "--video-path", required=True, help="input video full path") # v can be a folder or a video file
ap.add_argument("-o", "--output-path", required=True, help="output folder path")
ap.add_argument("-sc", "--crop-scale", required=False, type=float, default=1.0, help="crop scale to bbox")
ap.add_argument("-sz", "--min-size", required=False, type=int, default=100, help="threshold")
ap.add_argument("-sq", "--crop-square", required=False, type=int, default=0, help="make square crop")
args = vars(ap.parse_args())

min_size = args['min_size']

if not isdir(args['output_path']):
    mkdir(args['output_path'])

square_flag = args['crop_square']>0
crop_scale = args['crop_scale']

useMtcnn = True
if useMtcnn:
    faceDet = tfMtcnnFaceDet()
else:
    faceDet = tfFaceDet()
#f_log = open('/home/macul/Projects/realsense/distance_face_size.log','wb')

#cap = cv2.VideoCapture('/home/macul/test1.avi')
#cap = cv2.VideoCapture('/media/macul/black/spoof_db/record_2018_08_13_17_31_27.avi')
#cap = cv2.VideoCapture('/home/macul/iim_sz_02_05_mov.avi')
#cap = cv2.VideoCapture('/media/macul/black/spoof_db/spoofing_data_Mar_2019/picture/record_2019_02_21_10_03_03.avi')
#cap = cv2.VideoCapture('/media/macul/hdd/video_sz/group0/output0.avi')


if isdir(args['video_path']):
    video_list = listdir(args['video_path'])
else:
    video_list = [args['video_path']]

for video in video_list:
    fn = video[video.rfind('/')+1:]
    fname = fn[0:fn.rfind('.')]
    ext = fn[fn.rfind('.')+1:]
    print(fname, ext)

    if ext not in ['avi']:
        continue

    if not isdir(args['output_path']+'/'+fname):
        mkdir(args['output_path']+'/'+fname)

    #print(args['video_path']+'/'+video)
    cap = cv2.VideoCapture(args['video_path']+'/'+video)

    try:
        counter = 0
        while True:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            ret, frame = cap.read()            
            [h, w] = frame.shape[:2]

            if useMtcnn:
                faces = faceDet.getModelOutput(frame)

                for face in faces:
                    # face: [ymin, xmin, ymax, xmax]
                    f_h = face[2]-face[0]
                    f_w = face[3]-face[1]
                    

                    if f_h>min_size and f_w>min_size:
                        
                        bbox = [face[1], face[0], face[3], face[2]]
                        print(face, bbox)

                        face_crop, bbox_new = prepare_img(frame, bbox, crop_scale_to_bbox=crop_scale, crop_square=square_flag)

                        cv2.imwrite(args['output_path']+'/{}/{}_{}.png'.format(fname, fname, counter), face_crop)
                        counter += 1

                        #cv2.imshow('frame',face_crop)
                        #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #    break
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
                faces = faceDet.getModelOutput(frame_rgb)

                for face in faces:
                    # face: [ymin, xmin, ymax, xmax]
                    f_h = int((face[2]-face[0])*h)
                    f_w = int((face[3]-face[1])*w)
                    

                    if f_h>min_size and f_w>min_size:
                        
                        bbox = [int(face[1]*w), int(face[0]*h), int(face[3]*w), int(face[2]*h)]
                        print(face, bbox)

                        face_crop, bbox_new = prepare_img(frame, bbox, crop_scale_to_bbox=crop_scale, crop_square=square_flag)

                        cv2.imwrite(args['output_path']+'/{}/{}_{}.png'.format(fname, fname, counter), face_crop)
                        counter += 1

                        #cv2.imshow('frame',face_crop)
                        #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #    break
    except:
        pass

    cap.release()

#cv2.destroyAllWindows()
    #plt.subplot(122)
    #plt.imshow(crop_color)
    #plt.show()

# python -m tfFaceCropFromVideo -v /media/macul/black/spoof_db/spoofing_data_Mar_2019/video_new -o /media/macul/black/spoof_db/spoofing_data_Mar_2019/video_new_crop