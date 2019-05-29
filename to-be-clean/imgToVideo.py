#conver image to video
import cv2
import numpy as np
import os
from os import listdir
from os.path import isdir,isfile,join
import time

def frames_to_video(input_loc, output_loc):

    if not isdir(input_loc):
        assert False, "Input directory does not exist!"

    output_dir = join(*output_loc.split('/')[0:-1])
    print(output_dir)
    try:
        os.mkdir(output_dir)
    except OSError:
        pass
    # Log the time
    time_start = time.time()

    fname=listdir(input_loc)
    fname.sort()
    nIMAGES = len(fname)

    print ("Number of frames: ", nIMAGES)

    imgShape=cv2.imread(join(input_loc,fname[0])).shape

    MOV = cv2.VideoWriter(filename=output_loc, fourcc=cv2.VideoWriter_fourcc(*"MJPG"), fps=15, frameSize=(imgShape[1], imgShape[0]))

    for i in np.arange(0, nIMAGES):
        print 'Working on image: ', i
        image = cv2.imread(join(input_loc,fname[i]))
        #crop_image = image[50:550, 252:472] #crop y:h, x:w

        # now let's create the movie:
        #crop_image = cv2.applyColorMap(crop_image, cv2.COLORMAP_JET)
        print image.shape
        MOV.write(cv2.merge([image[:,:,0],image[:,:,1],image[:,:,2]]))

    MOV.release()


output_loc = '/home/macul/Screencast_2019-05-06_11-02-09.avi'
input_loc = '/home/macul/Screencast_2019-05-06_11-02-09'
frames_to_video(input_loc, output_loc)
#scp -P 31625 iim@iim.ltd:~/catkin_ws/src/face_recognition/lib/2018_02_05* ~/zhaoshang
#scp -P 31624 iim@iim.ltd:~/catkin_ws/src/face_recognition/lib/2018_02_05* ~/iim_sz
