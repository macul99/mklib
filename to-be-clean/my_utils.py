from os import listdir, walk
from os.path import join

def genFileList(dir_path, out_file):
	with open(out_file,'wb') as f:
		for fn in listdir(dir_path):
			f.write(join(dir_path,fn)+"\n")


def searchfiles(dir_path, out_file, extension='.avi'):
	with open(out_file, "w") as filewrite:
		for r, d, f in walk(dir_path):
			for file in f:
				if file.endswith(extension):
					filewrite.write(join(r,file)+'\n')
