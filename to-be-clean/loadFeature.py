import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import caffe
from os import listdir
from os.path import isfile,join
from os import walk

ft_dir='/home/macul/Projects/registration/Features'

dir_list = listdir(ft_dir)

ft_dict = {}

for di in dir_list:
	files = listdir(join(ft_dir, di))

	for fi in files:
		if '.txt' in fi:
			with open(join(join(ft_dir, di),fi), 'rb') as f:
				print(join(join(ft_dir, di),fi))
				name, uid, fid, feature = f.readline().strip().split(',')
				feature = [float(x) for x in feature.split()]

				if name in ft_dict:
					ft_dict[name]['feature'].update({fid: { 'ft': feature, 
									  						'img': join(join(ft_dir, di),'image.jpg')} })
				else:
					ft_dict[name] = { 'uid': uid, 
									  'feature': {fid: {'ft': feature, 
									  					'img': join(join(ft_dir, di),'image.jpg')} } }
		else:
			pass