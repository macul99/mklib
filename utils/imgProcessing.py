import cv2
import numpy as np
import os
from os import listdir
from os.path import isdir,isfile,join
import time


def video_to_frames(input_file, output_dir):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_file: Input video file.
        output_dir: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_dir)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_file)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_dir + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        print ("Converting video..{} out of {} frames".format(count, video_length))
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break


def frames_to_video(input_dir, output_file):

    if not isdir(input_dir):
        assert False, "Input directory does not exist!"

    output_dir = "/"+join(*output_file.split('/')[0:-1])
    print(output_dir)
    try:
        os.mkdir(output_dir)
    except OSError:
        pass
    # Log the time
    time_start = time.time()

    fname=listdir(input_dir)
    fname.sort()
    nIMAGES = len(fname)

    print ("Number of frames: ", nIMAGES)

    imgShape=cv2.imread(join(input_dir,fname[0])).shape

    MOV = cv2.VideoWriter(filename=output_file, fourcc=cv2.VideoWriter_fourcc(*"MJPG"), fps=10, frameSize=(imgShape[1], imgShape[0]))

    for i in np.arange(0, nIMAGES):
        if fname[i].split('.')[-1] in ['jpg','png']:
            print 'Working on image: {}, {}'.format(i, fname[i])
            image = cv2.imread(join(output_dir,fname[i]))
            #crop_image = image[50:550, 252:472] #crop y:h, x:w

            # now let's create the movie:
            #crop_image = cv2.applyColorMap(crop_image, cv2.COLORMAP_JET)
            print image.shape
            MOV.write(cv2.merge([image[:,:,0],image[:,:,1],image[:,:,2]]))

    MOV.release()



def hisEq(img, plot=False):
    #img = cv2.imread(imgPathName)

    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)

    # Applying equalize Hist operation on Y channel.
    y_eq = cv2.equalizeHist(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

    if plot:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow('img', img_rgb_eq)
        kVal=cv2.waitKey()
        if kVal == 93:
            cv2.destroyAllWindows()
            
    return img_rgb_eq


def intensity(img, change, plot=False):
    #img = cv2.imread(imgPathName)

    img1 = img.copy().astype(int)+change
    img1[img1>255] = 255
    img1[img1<0] = 0

    img1=img1.astype(np.uint8)

    if plot:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow('img', img1)
        kVal=cv2.waitKey()
        if kVal == 93:
            cv2.destroyAllWindows()

    return img1

def contrast(img, change, plot=False):
    #img = cv2.imread(imgPathName)

    img1 = img.copy().astype(int)*change
    img1[img1>255] = 255
    img1[img1<0] = 0

    img1=img1.astype(np.uint8)

    if plot:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow('img', img1)
        kVal=cv2.waitKey()
        if kVal == 93:
            cv2.destroyAllWindows()

    return img1


def gamma(img, g, plot=False):
    #img = cv2.imread(imgPathName)

    img1 = (img.copy()/255.0)**g * 255.0

    img1=img1.astype(np.uint8)

    if plot:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow('img', img1)
        kVal=cv2.waitKey()
        if kVal == 93:
            cv2.destroyAllWindows()

    return img1


def denoising(img, param=[10,10,7,21], plot=False):
    #img = cv2.imread(imgPathName)

    img1 = cv2.fastNlMeansDenoisingColored(img,None,*param)

    if plot:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow('img', img1)
        kVal=cv2.waitKey()
        if kVal == 93:
            cv2.destroyAllWindows()

    return img1
