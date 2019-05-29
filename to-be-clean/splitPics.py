#display image in fixed interval
#exec(open('/home/macul/libraries/mk_utils/displayPics.py').read())
# screen resolution: 1920x1080
import numpy as np 
import cv2
from os import listdir, walk, makedirs
from os.path import isfile,join,isdir,exists
import shutil

path='/home/iim/spoofing/'
source_folder = 'img_align_celeba_png'

folder_index = 29
max_num_of_file = 5000
counter = 0

for f in listdir(join(path, source_folder)):

	if counter%max_num_of_file==0:
		new_folder = join(path,source_folder+'_'+str(folder_index))
		if exists(new_folder):
			assert False, 'the folder already exists!'

		makedirs(new_folder)
		folder_index+=1

	counter+=1

	print f
	shutil.move(join(path, source_folder, f), join(new_folder,f))