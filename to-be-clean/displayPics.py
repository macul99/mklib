#display image in fixed interval
#exec(open('/home/macul/libraries/mk_utils/displayPics.py').read())
# screen resolution: 1920x1080
import numpy as np 
import cv2
from os import listdir, walk
from os.path import isfile,join

screen_w = 1920
screen_h = 1080

path='/home/iim/spoofing/img_0'
counter = 0
for f in listdir(path):
	tgt_h = int((screen_h - 100)*(1.0-0.5*np.random.random()))

	#print(f)
	im = cv2.imread(join(path,f))

	try:
		im_h, im_w, _ = im.shape
	except:
		break

	print("im_w: ", im_w)
	print("im_h: ", im_h)

	im = cv2.resize(im, (int(1.0*im_w*tgt_h/im_h), tgt_h))

	im_h, im_w, _ = im.shape
	print("im_w: ", im_w)
	print("im_h: ", im_h)

	cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow(f, (screen_w-im_w)/2, 0)
	cv2.imshow(f,im)

	


	# here it should be the pause
	k = cv2.waitKey(2500)
	print('key: ', k)
	if k == 27:         # wait for ESC key to exit
		cv2.destroyAllWindows()
		break
	else:
	    cv2.destroyAllWindows()

	
	'''
	counter+=1
	if counter==10:
		break
	'''