import numpy as np
import random
import cv2
import cPickle as pickle
from os.path import isfile,join


def showPointsBoxes(image, points=None, boxes=None, color=(0,255,0), make_copy=True):	
	if make_copy:
		img = image.copy()
	else:
		img = image

	if type(boxes) != type(None):
		img = drawBoxes(img, boxes, color)

	if type(points) != type(None):
		interocular_distance = np.linalg.norm(points[0] - points[1])
		img = drawPoints(img, points, color=color, interocular_distance=interocular_distance)

	cv2.namedWindow("img", cv2.WINDOW_NORMAL)
	cv2.imshow('img', img)
	kVal=cv2.waitKey()
	cv2.destroyAllWindows()

def drawBoxes(im, boxes, color=(0,255,0)):
	x1 = boxes[0]
	y1 = boxes[1]
	x2 = boxes[2]
	y2 = boxes[3]

	cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
	return im

def drawPoints(im, points, shape='circle',color=(0,255,0), interocular_distance=None):
	for i in range(points.shape[0]):
		if shape == 'circle':
			if interocular_distance:
				lineWidth = int(interocular_distance/25)
				radius = int(lineWidth * 1.5)
			else:
				radius = 3
				lineWidth = 2
			cv2.circle(im, (int(points[i,0]), int(points[i, 1])), radius, color, lineWidth)
		elif shape == 'rect':
			if interocular_distance:
				lineWidth = int(interocular_distance/25)
				boxWidth = int(lineWidth * 3)
			else:
				boxWidth = 6
				lineWidth = 2
			cv2.rectangle(im, (int(points[i,0]-boxWidth), int(points[i, 1]-boxWidth)), (int(points[i,0]+boxWidth), int(points[i, 1]+boxWidth)), color, lineWidth)
		else:
			assert False, 'this shape is not supported'
	return im
		
def square_padding(im, points):
	imgSize=im.shape
	imgHeight = imgSize[0]
	imgWidth = imgSize[1]
	i_c = imgSize[2]

	if i_c!=3 and i_c!=1:
		assert False, "Wrong image shape"	

	if imgHeight == imgWidth:
		return im, points # already a square

	b_max = np.maximum(imgWidth, imgHeight)

	#sqrImg = np.ones((b_max, b_max, i_c), dtype = "uint8")*255 ### declare dtype is crutial
	sqrImg = np.zeros((b_max, b_max, i_c), dtype = "uint8") ### declare dtype is crutial

	i_x1 = int((b_max-imgWidth)/2)
	i_y1 = int((b_max-imgHeight)/2)
	i_x2 = i_x1 + imgWidth
	i_y2 = i_y1 + imgHeight

	sqrImg[i_y1:i_y2, i_x1:i_x2,:] = im

	points += [i_x1, i_y1]

	return sqrImg, points

def pad(boxesA, w, h):
    boxes = boxesA.copy() # shit, value parameter!!!
    #print '#################'
    #print 'boxes', boxes
    #print 'w,h', w, h
    
    tmph = boxes[3] - boxes[1] + 1
    tmpw = boxes[2] - boxes[0] + 1
    #numbox = boxes.shape[0]

    #print 'tmph', tmph
    #print 'tmpw', tmpw

    dx = 1
    dy = 1
    edx = tmpw 
    edy = tmph

    x = boxes[0]
    y = boxes[1]
    ex = boxes[2]
    ey = boxes[3]
   
   
    #tmp = np.where(ex > w)[0]
    if ex > w:
        edx = -ex + w-1 + tmpw
        ex = w-1

    #tmp = np.where(ey > h)[0]
    if ey > h:
        edy = -ey + h-1 + tmph
        ey = h-1

    #tmp = np.where(x < 1)[0]
    if x < 1:
        dx = 2 - x
        x = 1

    #tmp = np.where(y < 1)[0]
    if y < 1:
        dy = 2 - y
        y = 1
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)

    #print 'boxes', boxes
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]

def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[2] - bboxA[0]
    h = bboxA[3] - bboxA[1]
    l = np.maximum(w,h).T
    
    #print 'bboxA', bboxA
    #print 'w', w
    #print 'h', h
    #print 'l', l
    bboxA[0] = bboxA[0] + w*0.5 - l*0.5
    bboxA[1] = bboxA[1] + h*0.5 - l*0.5 
    bboxA[2:4] = bboxA[0:2] + l 
    return bboxA


def box_cropOnly(image, bounding_box, points=None, make_square=True):
	#print img[tuple(points[0])].shape
	#print img[points[0]].shape	

	img = image.copy()
	imgSize=img.shape
	imgHeight = imgSize[0]
	imgWidth = imgSize[1]
	i_c = imgSize[2]

	if i_c!=3 and i_c!=1:
		assert False, "Wrong image shape"	

	bBox = np.array(bounding_box).astype(long).copy()
	#print('bBox', bBox)
	#bBox[2] = np.minimum(bBox[2],imgWidth)
	#bBox[3] = np.minimum(bBox[3],imgHeight)

	if type(points) != type(None):
		if (np.max(points[:,0])>imgWidth) or (np.max(points[:,1])>imgHeight):
			return None, None, None

	if make_square:		
		'''
		w = bBox[2] - bBox[0]
		h = bBox[3] - bBox[1]

		b_max = np.maximum(w, h)

		bBox[0] = bBox[0] + w*0.5 - b_max*0.5 # change boundingBox to square shape
		bBox[1] = bBox[1] + h*0.5 - b_max*0.5
		bBox[2] = bBox[0] + b_max
		bBox[3] = bBox[1] + b_max
		'''
		bBox = rerec(bBox)
		bBox = np.fix(bBox)

		#print('bBox squared', bBox)

		[dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(bBox, imgWidth, imgHeight)
		cropImg = np.zeros((int(tmph), int(tmpw), i_c)) ### declare dtype is crutial
		#cropImg = np.zeros((int(tmph[k]), int(tmpw[k]), i_c), dtype = "uint8") ### declare dtype to show image using cv
		cropImg[int(dy):int(edy)+1, int(dx):int(edx)+1] = img[int(y):int(ey)+1, int(x):int(ex)+1]
	
		'''
		i_x1 = int(np.maximum(0, bBox[0]))
		i_y1 = int(np.maximum(0, bBox[1]))
		i_x2 = int(np.minimum(imgWidth-1, bBox[2]))
		i_y2 = int(np.minimum(imgHeight-1, bBox[3]))

		#cropImg = np.ones((b_max, b_max, i_c), dtype = "uint8")*255 ### declare dtype is crutial
		
		#print cropImg[:,:,0]
		c_x1 = int(np.maximum(0-bBox[0], 0)) # find the area from cropImg to load data from image
		c_y1 = int(np.maximum(0-bBox[1], 0))
		c_x2 = c_x1 + (i_x2 - i_x1)
		c_y2 = c_y1 + (i_y2 - i_y1)

		cropImg[c_y1:c_y2+1,c_x1:c_x2+1,:] = img[i_y1:i_y2+1,i_x1:i_x2+1,:]
		#print cropImg.shape
		#print bBox
		#print i_x1,i_y1,i_x2,i_y2
		#print c_x1,c_y1,c_x2,c_y2
		'''

	else:
		cropImg = img[bBox[1]:bBox[3]+1,bBox[0]:bBox[2]+1,:].copy()

	if type(points) != type(None):				
		new_points = np.array(points).astype(float).copy()
		#print('new_points', new_points)
		new_points[:,0] = new_points[:,0] - bBox[0] + 0.5 # use 0.5 instead of 1 to make the point symetric to the center of image
		new_points[:,1] = new_points[:,1] - bBox[1] + 0.5

		#print('new_points relative to bbox corner', new_points)

		w = bBox[3] - bBox[1] + 1
		h = bBox[2] - bBox[0] + 1
		scaled_points = new_points.astype(float).copy()
		scaled_points[:,0] = scaled_points[:,0] / float(w)
		scaled_points[:,1] = scaled_points[:,1] / float(h)

		#print('scaled_points', scaled_points)

		#assert False
		#new_points=new_points.astype(int)
	else:
		new_points=None
		scaled_points=None

	#print new_points

	'''
	if make_square:
		return square_padding(cropImg, new_points)
	else:
		return cropImg, new_points
	'''

	return cropImg, new_points, scaled_points

def box_translation(img, bounding_box, points=None, make_square=True):
	#print img[tuple(points[0])].shape
	#print img[points[0]].shape

	imgSize=img.shape
	imgHeight = imgSize[0]-1
	imgWidth = imgSize[1]-1
	i_c = imgSize[2]

	if i_c!=3 and i_c!=1:
		assert False, "Wrong image shape"

	bBox = np.array(bounding_box).astype(int).copy()
	bBox[2] = np.minimum(bBox[2],imgWidth)
	bBox[3] = np.minimum(bBox[3],imgHeight)

	if type(points) != type(None):
		if (np.max(points[:,0])>imgWidth) or (np.max(points[:,1])>imgHeight):
			return None, None

	w = bBox[2] - bBox[0] + 1
	h = bBox[3] - bBox[1] + 1

	if make_square:
		b_max = np.maximum(w, h)

		bBox[0] = bBox[0] - (b_max - w) / 2 # change boundingBox to square shape
		bBox[1] = bBox[1] - (b_max - h) / 2
		bBox[2] = bBox[0] + b_max - 1
		bBox[3] = bBox[1] + b_max - 1
	
		bBox[0] = int(np.maximum(0, bBox[0]))
		bBox[1] = int(np.maximum(0, bBox[1]))
		bBox[2] = int(np.minimum(imgWidth-1, bBox[2]))
		bBox[3] = int(np.minimum(imgHeight-1, bBox[3]))

		w = bBox[2] - bBox[0] + 1
		h = bBox[3] - bBox[1] + 1

	while True:

		dx = int((random.random()-0.5)*2*w*0.2) # upto 20% of image size
		dy = int((random.random()-0.5)*2*h*0.2) # upto 20% of image size

		new_bbox = [0,0,0,0]
		new_bbox[0] = 0 if bBox[0]+dx<0 else bBox[0]+dx
		new_bbox[1] = 0 if bBox[1]+dy<0 else bBox[1]+dy
		new_bbox[2] = imgWidth if bBox[2]+dx>imgWidth else bBox[2]+dx
		new_bbox[3] = imgHeight if bBox[3]+dy>imgHeight else bBox[3]+dy	
		new_bbox = np.array(new_bbox).astype(int)

		if type(points) != type(None):
			new_points = np.array(points).astype(int).copy()
			new_points[:,0] = new_points[:,0] - new_bbox[0]
			new_points[:,1] = new_points[:,1] - new_bbox[1]
			new_points=new_points.astype(int)

			if ( (new_points[:,0]>=0).all() and (new_points[:,0]<(new_bbox[2]-new_bbox[0]+1)).all() ) and \
				( (new_points[:,1]>=0).all() and (new_points[:,1]<(new_bbox[3]-new_bbox[1]+1)).all() ):	
				if make_square:
					return square_padding(img[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(), new_points)
				else:
					return img[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(), new_points
		else:
			if make_square:
				return square_padding(img[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(), new_points)
			else:
				return img[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(), None


def box_zoom(img, bounding_box, points=None, make_square=True):
	#print img[tuple(points[0])].shape
	#print img[points[0]].shape

	imgSize=img.shape
	imgHeight = imgSize[0]-1
	imgWidth = imgSize[1]-1
	i_c = imgSize[2]

	if i_c!=3 and i_c!=1:
		assert False, "Wrong image shape"

	bBox = np.array(bounding_box).astype(int).copy()
	bBox[2] = np.minimum(bBox[2],imgWidth)
	bBox[3] = np.minimum(bBox[3],imgHeight)

	if type(points) != type(None):
		if (np.max(points[:,0])>imgWidth) or (np.max(points[:,1])>imgHeight):
			return None, None

	w = bBox[2] - bBox[0] + 1
	h = bBox[3] - bBox[1] + 1

	if make_square:
		b_max = np.maximum(w, h)

		bBox[0] = bBox[0] - (b_max - w) / 2 # change boundingBox to square shape
		bBox[1] = bBox[1] - (b_max - h) / 2
		bBox[2] = bBox[0] + b_max - 1
		bBox[3] = bBox[1] + b_max - 1
	
		bBox[0] = int(np.maximum(0, bBox[0]))
		bBox[1] = int(np.maximum(0, bBox[1]))
		bBox[2] = int(np.minimum(imgWidth-1, bBox[2]))
		bBox[3] = int(np.minimum(imgHeight-1, bBox[3]))

		w = bBox[2] - bBox[0] + 1
		h = bBox[3] - bBox[1] + 1

	while True:
		zoomFactor = 1.1 + (random.random()-0.5)*2*0.3 # zoom in the range (0.8, 1.4)
		dx = int( (zoomFactor-1.0)*0.5*w )
		dy = int( (zoomFactor-1.0)*0.5*h )

		new_bbox = [0,0,0,0]
		new_bbox[0] = 0 if bBox[0]-dx<0 else bBox[0]-dx
		new_bbox[1] = 0 if bBox[1]-dy<0 else bBox[1]-dy
		new_bbox[2] = imgWidth if bBox[2]+dx>imgWidth else bBox[2]+dx
		new_bbox[3] = imgHeight if bBox[3]+dy>imgHeight else bBox[3]+dy		
		new_bbox = np.array(new_bbox).astype(int)

		if type(points) != type(None):
			new_points = np.array(points).astype(int).copy()
			new_points[:,0] = new_points[:,0] - new_bbox[0]
			new_points[:,1] = new_points[:,1] - new_bbox[1]
			new_points=new_points.astype(int)

			#print 'dx: ', dx
			#print 'dy: ', dy
			#print 'new_bbox: ', new_bbox
			#print 'new_points', new_points

			if ( (new_points[:,0]>=0).all() and (new_points[:,0]<(new_bbox[2]-new_bbox[0]+1)).all() ) and \
				( (new_points[:,1]>=0).all() and (new_points[:,1]<(new_bbox[3]-new_bbox[1]+1)).all() ):		
				if make_square:
					return square_padding(img[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(), new_points)
				else:	
					return img[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(), new_points
		else:
			if make_square:
				return square_padding(img[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(), new_points)
			else:
				return img[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(), None

def box_vFlip(img, bounding_box, points=None, make_square=True):

	imgSize=img.shape
	imgHeight = imgSize[0]-1
	imgWidth = imgSize[1]-1
	i_c = imgSize[2]

	if i_c!=3 and i_c!=1:
		assert False, "Wrong image shape"

	bBox = np.array(bounding_box).astype(int).copy()
	bBox[2] = np.minimum(bBox[2],imgWidth)
	bBox[3] = np.minimum(bBox[3],imgHeight)

	if type(points) != type(None):
		if (np.max(points[:,0])>imgWidth) or (np.max(points[:,1])>imgHeight):
			return None, None

	w = bBox[2] - bBox[0] + 1
	h = bBox[3] - bBox[1] + 1

	if make_square:
		b_max = np.maximum(w, h)

		bBox[0] = bBox[0] - (b_max - w) / 2 # change boundingBox to square shape
		bBox[1] = bBox[1] - (b_max - h) / 2
		bBox[2] = bBox[0] + b_max - 1
		bBox[3] = bBox[1] + b_max - 1
	
		bBox[0] = int(np.maximum(0, bBox[0]))
		bBox[1] = int(np.maximum(0, bBox[1]))
		bBox[2] = int(np.minimum(imgWidth-1, bBox[2]))
		bBox[3] = int(np.minimum(imgHeight-1, bBox[3]))

	new_bbox = np.array(bBox).astype(int)
	newRefX1 = new_bbox[2]

	if type(points) != type(None):
		new_points = np.array(points).astype(int).copy()
		new_points[:,0] = newRefX1 - new_points[:,0]
		new_points[:,1] = new_points[:,1] - new_bbox[1]
		new_points=new_points.astype(int)
	else:
		new_points=None

	if make_square:
		return square_padding(cv2.flip(img[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(),1), new_points)
	else:
		return cv2.flip(img[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(),1), new_points


def box_rotate(img, bounding_box, points=None, make_square=True):
	# rotate based on the nose which is the third point

	imgSize=img.shape
	imgHeight = imgSize[0]-1
	imgWidth = imgSize[1]-1
	i_c = imgSize[2]

	if i_c!=3 and i_c!=1:
		assert False, "Wrong image shape"

	bBox = np.array(bounding_box).astype(int).copy()
	bBox[2] = np.minimum(bBox[2],imgWidth)
	bBox[3] = np.minimum(bBox[3],imgHeight)

	if type(points) != type(None):
		if (np.max(points[:,0])>imgWidth) or (np.max(points[:,1])>imgHeight):
			return None, None

	w = bBox[2] - bBox[0] + 1
	h = bBox[3] - bBox[1] + 1

	if make_square:
		b_max = np.maximum(w, h)

		bBox[0] = bBox[0] - (b_max - w) / 2 # change boundingBox to square shape
		bBox[1] = bBox[1] - (b_max - h) / 2
		bBox[2] = bBox[0] + b_max - 1
		bBox[3] = bBox[1] + b_max - 1
	
		bBox[0] = int(np.maximum(0, bBox[0]))
		bBox[1] = int(np.maximum(0, bBox[1]))
		bBox[2] = int(np.minimum(imgWidth-1, bBox[2]))
		bBox[3] = int(np.minimum(imgHeight-1, bBox[3]))


	while True:
		rotate_degree = (random.random()-0.5)*2*15 # rotate in range of (-15, 15) degree

		M = cv2.getRotationMatrix2D((imgWidth/2,imgHeight/2),rotate_degree,1)
		dst = cv2.warpAffine(img,M,(imgWidth,imgHeight))

		tmp_bbox = np.dot(np.array([[bBox[0], bBox[1], 1],
									[bBox[2], bBox[3], 1],
									[bBox[2], bBox[1], 1],
									[bBox[0], bBox[3], 1]]), M.T)

		new_bbox = [max(np.min(tmp_bbox[:,0],0),0),
					max(np.min(tmp_bbox[:,1],0),0),
					min(np.max(tmp_bbox[:,0],0),imgWidth),
					min(np.max(tmp_bbox[:,1],0),imgHeight)]
		new_bbox = np.array(new_bbox).astype(int)

		if type(points) != type(None):
			points=np.array(points).astype(float)

			new_points = np.round( np.dot(np.append(points, np.ones([points.shape[0],1]),1), M.T) ).astype(int).copy()
			new_points[:,0] = new_points[:,0] - new_bbox[0]
			new_points[:,1] = new_points[:,1] - new_bbox[1]
			new_points=new_points.astype(int)

		#print 'rotate_degree', rotate_degree
		#print 'tmp_bbox', tmp_bbox
		#print 'new_bbox', new_bbox
		#print 'new_points', new_points

			if ( (new_points[:,0]>=0).all() and (new_points[:,0]<(new_bbox[2]-new_bbox[0]+1)).all() ) and \
				( (new_points[:,1]>=0).all() and (new_points[:,1]<(new_bbox[3]-new_bbox[1]+1)).all() ):	
				if make_square:
					return square_padding(dst[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(), new_points)
				else:		
					return dst[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(), new_points
		else:
			if make_square:
				return square_padding(dst[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(), new_points)
			else:
				return dst[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1].copy(), None

def unitTest():
	
	with open('/home/macul/Projects/300W/results1.pkl', 'rb') as f:
	    resultDic = pickle.load(f)

	datasetDic = resultDic['datasetDic']
	datasetLoc = resultDic['datasetLoc']
	dsKeys = datasetDic.keys()
	dsKeys.sort()

	imgPath=datasetLoc['celeba_train'].split()[0]
	imgList=datasetDic['celeba_train'].dataDic.keys()
	index = 0
	imgName=imgList[index]
	img = cv2.imread(join(imgPath,imgName))
	points = datasetDic['celeba_train'].dataDic[imgName]['lbl5Points']
	bbox = datasetDic['celeba_train'].dataDic[imgName]['boundingBox'][0]

	print "Show orig image, bbox and points, make_copy=True"
	showPointsBoxes(img, points, boxes=bbox)

	print "Show cropped image and points, make_square=True"
	img0, points0 = box_cropOnly(img, bbox, points=points)
	showPointsBoxes(img0, points0, color=(255,0,0))
	print "Show cropped image and points, make_square=False"
	img0, points0 = box_cropOnly(img, bbox, points=points, make_square=False)
	showPointsBoxes(img0, points0, color=(255,0,0))
	print "Show cropped image without points, make_square=True"
	img0, points0 = box_cropOnly(img, bbox,)
	showPointsBoxes(img0, points0, color=(255,0,0))

	print "Show orig image, bbox and points, make_copy=False"
	showPointsBoxes(img, points, boxes=bbox, make_copy=False)

	print "Show translaion image and points, make_square=True"
	img1, points1 = box_translation(img, bbox, points=points)
	showPointsBoxes(img1, points1, color=(255,0,0))
	print "Show translaion image and points, make_square=False"
	img1, points1 = box_translation(img, bbox, points=points, make_square=False)
	showPointsBoxes(img1, points1, color=(255,0,0))
	print "Show translaion image without points, make_square=True"
	img1, points1 = box_translation(img, bbox)
	showPointsBoxes(img1, points1, color=(255,0,0))

	print "Show zoomed image and points, make_square=True"
	img2, points2 = box_zoom(img, bbox, points=points)
	showPointsBoxes(img2, points2, color=(255,0,0))
	print "Show zoomed image and points, make_square=False"
	img2, points2 = box_zoom(img, bbox, points=points, make_square=False)
	showPointsBoxes(img2, points2, color=(255,0,0))
	print "Show zoomed image without points, make_square=True"
	img2, points2 = box_zoom(img, bbox)
	showPointsBoxes(img2, points2, color=(255,0,0))

	print "Show vFlipped image and points, make_square=True"
	img3, points3 = box_vFlip(img, bbox, points=points)
	showPointsBoxes(img3, points3, color=(255,0,0))
	print "Show vFlipped image and points, make_square=False"
	img3, points3 = box_vFlip(img, bbox, points=points, make_square=False)
	showPointsBoxes(img3, points3, color=(255,0,0))
	print "Show vFlipped image without points, make_square=True"
	img3, points3 = box_vFlip(img, bbox)
	showPointsBoxes(img3, points3, color=(255,0,0))

	print "Show rotated image and points, make_square=True"
	img4, points4 = box_rotate(img, bbox, points=points)
	showPointsBoxes(img4, points4, color=(255,0,0))
	print "Show rotated image and points, make_square=False"
	img4, points4 = box_rotate(img, bbox, points=points, make_square=False)
	showPointsBoxes(img4, points4, color=(255,0,0))
	print "Show rotated image without points, make_square=True"
	img4, points4 = box_rotate(img, bbox)
	showPointsBoxes(img4, points4, color=(255,0,0))