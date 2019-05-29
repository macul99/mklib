from imgProcessing import hisEq, frames_to_video, video_to_frames, intensity, gamma, denoising, contrast
from os import listdir,mkdir
from os.path import isdir,isfile,join
import cv2

'''
input_dir = '/home/macul/video/new/record_2018_03_14_12_44_33'
output_dir = '/home/macul/video/new/record_2018_03_14_12_44_33/hiseq/'
output_file = join(output_dir, 'hiseq.avi')

try:
    mkdir(output_dir)
except OSError:
    pass

fname=listdir(input_dir)
fname.sort()
nIMAGES = len(fname)

for i in range(nIMAGES):
	print ('Histogram equalization: {} out of {} files'.format(i, nIMAGES))

	if isfile(join(input_dir,fname[i])):
		if not cv2.imwrite(join(output_dir,fname[i]),
							hisEq(join(input_dir, fname[i]))):
			print('write image to file error!')

frames_to_video(output_dir, output_file)


input_dir = '/home/macul/video/new/record_2018_03_14_12_44_33'
output_dir = '/home/macul/video/new/record_2018_03_14_12_44_33/intensity_10/'
output_file = join(output_dir, 'intensity_10.avi')

try:
    mkdir(output_dir)
except OSError:
    pass

fname=listdir(input_dir)
fname.sort()
nIMAGES = len(fname)

for i in range(nIMAGES):
	print ('intensity_10: {} out of {} files'.format(i, nIMAGES))

	if isfile(join(input_dir,fname[i])):
		if not cv2.imwrite(join(output_dir,fname[i]),
							intensity(join(input_dir, fname[i]), 10)):
			print('write image to file error!')

frames_to_video(output_dir, output_file)


input_dir = '/home/macul/video/new/record_2018_03_14_12_44_33'
output_dir = '/home/macul/video/new/record_2018_03_14_12_44_33/intensity_20/'
output_file = join(output_dir, 'intensity_10.avi')

try:
    mkdir(output_dir)
except OSError:
    pass

fname=listdir(input_dir)
fname.sort()
nIMAGES = len(fname)

for i in range(nIMAGES):
	print ('intensity_20: {} out of {} files'.format(i, nIMAGES))

	if isfile(join(input_dir,fname[i])):
		if not cv2.imwrite(join(output_dir,fname[i]),
							intensity(join(input_dir, fname[i]), 20)):
			print('write image to file error!')

frames_to_video(output_dir, output_file)



input_dir = '/home/macul/video/new/record_2018_03_14_12_44_33'
output_dir = '/home/macul/video/new/record_2018_03_14_12_44_33/gamma_0_85/'
output_file = join(output_dir, 'gamma_0_85.avi')

try:
    mkdir(output_dir)
except OSError:
    pass

fname=listdir(input_dir)
fname.sort()
nIMAGES = len(fname)

for i in range(nIMAGES):
	print ('gamma_0_85: {} out of {} files'.format(i, nIMAGES))

	if isfile(join(input_dir,fname[i])):
		if not cv2.imwrite(join(output_dir,fname[i]),
							gamma(join(input_dir, fname[i]), 0.85)):
			print('write image to file error!')

frames_to_video(output_dir, output_file)


input_dir = '/home/macul/video/new/record_2018_03_14_12_44_33'
output_dir = '/home/macul/video/new/record_2018_03_14_12_44_33/denoising_10/'
output_file = join(output_dir, 'denoising_10.avi')

try:
    mkdir(output_dir)
except OSError:
    pass

fname=listdir(input_dir)
fname.sort()
nIMAGES = len(fname)

for i in range(nIMAGES):
	print ('denoising_10: {} out of {} files'.format(i, nIMAGES))

	if isfile(join(input_dir,fname[i])):
		if not cv2.imwrite(join(output_dir,fname[i]),
							denoising(join(input_dir, fname[i]), [10,10,7,21])):
			print('write image to file error!')

frames_to_video(output_dir, output_file)




input_dir = '/home/macul/video/new/record_2018_03_14_12_44_33/intensity_10'
output_dir = '/home/macul/video/new/record_2018_03_14_12_44_33/denoising_2/'
output_file = join(output_dir, 'int_10_denoising_2.avi')

try:
    mkdir(output_dir)
except OSError:
    pass

fname=listdir(input_dir)
fname.sort()
nIMAGES = len(fname)

for i in range(nIMAGES):
	print ('int_10_denoising_2: {} out of {} files'.format(i, nIMAGES))

	#if isfile(join(input_dir,fname[i])):
	if fname[i].split('.')[-1] in ['jpg','png']:
		if not cv2.imwrite(join(output_dir,fname[i]),
							denoising(join(input_dir, fname[i]), [2,2,7,21])):
			print('write image to file error!')

frames_to_video(output_dir, output_file)




input_file = '/home/macul/video/new/record_2018_03_14_10_46_02.avi'
output_dir = '/home/macul/video/new/record_2018_03_14_10_46_02/'

video_to_frames(input_file, output_dir)




input_dir = '/home/macul/video/new/record_2018_03_14_10_46_02'
output_dir = '/home/macul/video/new/record_2018_03_14_10_46_02/denoising_4/'
output_file = join(output_dir, 'denoising_4.avi')

try:
    mkdir(output_dir)
except OSError:
    pass

fname=listdir(input_dir)
fname.sort()
nIMAGES = len(fname)

for i in range(nIMAGES):
	print ('denoising_4: {} out of {} files'.format(i, nIMAGES))

	if isfile(join(input_dir,fname[i])):
		if not cv2.imwrite(join(output_dir,fname[i]),
							denoising(join(input_dir, fname[i]), [4,4,7,21])):
			print('write image to file error!')

frames_to_video(output_dir, output_file)
'''


input_dir = '/home/macul/video/new/record_2018_03_14_12_44_33'
output_dir = '/home/macul/video/new/record_2018_03_14_12_44_33/orig/'
output_file = join(output_dir, 'orig.avi')

try:
    mkdir(output_dir)
except OSError:
    pass

tmpFileName=listdir(input_dir)
fname=[]
for fn in tmpFileName:
	if fn.split('.')[-1] in ['jpg','png']:
		fname+=[fn]

fname.sort()
nIMAGES = len(fname)

for i in range(nIMAGES):
	print ('gamma_0_85_dn_4: {} out of {} files'.format(i, nIMAGES))

	img = cv2.imread(join(input_dir, fname[i]))
	#img = contrast(img, 1.5)
	#img = denoising(img, [4,4,7,21])

	if not cv2.imwrite(join(output_dir,fname[i]), img):
		print('write image to file error!')

frames_to_video(output_dir, output_file)

#exec(open('imgConversion.py','rb').read())