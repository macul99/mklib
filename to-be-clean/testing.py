import numpy as np
import cv2
import cPickle as pickle 
import imgArgument
from os.path import isfile,join



with open('results_aflw.pkl','rb') as f:
	result=pickle.load(f)

path = result['datasetLoc']['aflw']
dataDic=result['datasetDic']['aflw'].dataDic
keys=dataDic.keys()

fname='aflw__face_39872.jpg'

img = cv2.imread(join(path, fname))
points = dataDic[fname]['lbl5Points']
bbox = dataDic[fname]['boundingBox'][0]

imgArgument.showPointsBoxes(img, points=points, boxes=bbox, color=(0,255,0), make_copy=True)

img_r, pts_r = imgArgument.box_rotate(img, bbox, points=points, make_square=True)
imgArgument.showPointsBoxes(img_r, points=pts_r, boxes=None, color=(255,0,0), make_copy=True)

net_side=48
cropImg=img_r.copy()
scalingFactor = float(net_side)/cropImg.shape[1]
cropImg = cropImg.astype(float)
cropImg -= 127.5 # image normalization
cropImg /= 128.0
cropImg = cv2.resize(cropImg,(int(net_side),int(net_side))) # resize, do this before swapaxes

newPts=pts_r.astype(float).copy()
newPts *= scalingFactor
imgArgument.showPointsBoxes(cropImg, points=newPts, boxes=None, color=(0,0,255), make_copy=True)

def preprocessing(image, boundingBox=None, points=None):		
	i_w = image.shape[1]
	i_h = image.shape[0]
	i_c = image.shape[2]

	net_size = 48

	if boundingBox!=None:
		bBox = boundingBox.copy().astype(int)				
		bBox -= 1 # data from matlab is started from index 1			
	else:
		bBox = np.array([0,0,i_w-1,i_h-1])			

	print 'bBox: ', bBox

	b_w = bBox[2] - bBox[0] + 1 # calculate width and height of boundingBox
	b_h = bBox[3] - bBox[1] + 1
	b_max = np.maximum(b_w, b_h)

	print 'b_w,b_h,b_max: ', b_w, b_h, b_max

	bBox[0] = bBox[0] - (b_max - b_w) / 2 # change boundingBox to square shape
	bBox[1] = bBox[1] - (b_max - b_h) / 2
	bBox[2] = bBox[0] + b_max - 1
	bBox[3] = bBox[1] + b_max - 1

	print 'bBox: ', bBox
	
	i_x1 = np.maximum(0, bBox[0])# find the area from image used to fill cropImg
	i_y1 = np.maximum(0, bBox[1])
	i_x2 = np.minimum(i_w-1, bBox[2])
	i_y2 = np.minimum(i_h-1, bBox[3])

	print 'i_idx: ', [i_x1, i_y1, i_x2, i_y2]

	cropImg = np.ones((b_max, b_max, i_c)) * 255
	c_x1 = np.maximum(0-bBox[0], 0) # find the area from cropImg to load data from image
	c_y1 = np.maximum(0-bBox[1], 0)
	c_x2 = c_x1 + (i_x2 - i_x1)
	c_y2 = c_y1 + (i_y2 - i_y1)

	print 'c_idx: ', [c_x1, c_y1, c_x2, c_y2]

	cropImg[c_y1:c_y2+1,c_x1:c_x2+1,:] = image[i_y1:i_y2+1,i_x1:i_x2+1,:]
	#cropImg -= 128 # image normalization
	#cropImg /= 255.0
	#cropImg = cv2.resize(cropImg,(int(48),int(48))) # resize, do this before swapaxes
	#cropImg = np.swapaxes(cropImg, 0, 2) # change (h,w,c) to (c,w,h)

	cv2.namedWindow("img", cv2.WINDOW_NORMAL)
	cv2.imshow("img",cropImg.astype(np.uint8))
	kVal=cv2.waitKey()
	if kVal == 32:
		cv2.destroyAllWindows()

	if points!=None:
		pts = points.copy().astype(float)
		pts -= 1 # data from matlab is started from index 1
		pts -= bBox[0:2] # now the pts reference to the top-left conner of square bounding box
		pts *= float(net_size)/b_max # scale according to image resize ratio

		return cropImg, pts.reshape((pts.size,))
	else:
		return cropImg

'''
import cv2
import numpy as np
img1=cv2.imread('1.jpg')
from testing import preprocessing as pp
img2=pp(img1,np.array([1,1,100,50]))
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img",img2)
kVal=cv2.waitKey()
if kVal == 32:
	cv2.destroyAllWindows()

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img",img1)
kVal=cv2.waitKey()
if kVal == 32:
	cv2.destroyAllWindows()

'''

