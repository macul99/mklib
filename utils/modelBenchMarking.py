import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
from os import mkdir, makedirs, rename, listdir
from os.path import join, exists, relpath, abspath
from imutils import paths
import pandas as pd
from sklearn.metrics import roc_curve, auc
from progressbar import *

widgets = ['Test: ', Percentage(), ' ', Bar(marker='|',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options

def rename_subfolder(src):
	fdl = listdir(src)
	for i, fd in enumerate(fdl):
		os.rename(join(src,fd),join(src,str(i)))

def extract_feature(extractor, src, save_path=None):
	df = pd.DataFrame(columns=['person_name', 'img_name', 'feature'])
	imagePaths = list(paths.list_images(src))
	total_num = len(imagePaths)
	pbar = ProgressBar(widgets=widgets, maxval=total_num)
	pbar.start()
	for i, img_path in enumerate(imagePaths):
		pbar.update(i)
		person_name, _, img_name = img_path.rpartition('/')
		person_name = person_name.rpartition('/')[2]
		feature = extractor(img_path)
		df = df.append({'person_name':person_name,
						'img_name':img_name,
						'feature':feature},
					   ignore_index=True)
	pbar.finish()
	if save_path is not None:
		pickle.dump(df, open(abspath(save_path), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	return df

def rank_folder(df_reg, df_test, top=5, save_path=None):
	assert df_reg['feature'].loc[0].shape[0]==df_test['feature'].loc[0].shape[0]
	ft_size = df_reg['feature'].loc[0].shape[0]	
	all_reg = list(set(df_reg['person_name'].values))
	all_test = list(set(df_test['person_name'].values))
	df = pd.DataFrame(columns=['person_name','score_top_n','reg_name_top_n','reg_img_top_n','test_img_top_n'])
	ft_reg = np.array(list(df_reg['feature']))
	if len(ft_reg.shape)==1:
		ft_reg = ft_reg[np.newaxis,:]
	person_name_reg = np.array(list(df_reg['person_name']))
	img_name_reg = np.array(list(df_reg['img_name']))
	assert ft_reg.shape[0]==len(img_name_reg)
	total_num = len(all_test)
	pbar = ProgressBar(widgets=widgets, maxval=total_num)
	pbar.start()
	for i, p_test in enumerate(all_test):
		#print('{}/{}'.format(i,total_num))
		pbar.update(i)
		img_name_test = np.array(list(df_test[df_test['person_name']==p_test]['img_name']))
		ft_test = np.array(list(df_test[df_test['person_name']==p_test]['feature']))
		tmp_score = (np.dot(ft_test, ft_reg.T)+1.0)*0.5
		score_max_axis0 = np.max(tmp_score,0)
		top_idx = score_max_axis0.argsort()[-top:][::-1]
		best_img_idx = []
		for ti in top_idx:			
			best_img_idx += [np.argmax(tmp_score[:,ti])]
		df = df.append({'person_name':p_test,
						'score_top_n':score_max_axis0[top_idx],
						'reg_name_top_n':person_name_reg[top_idx],
						'reg_img_top_n':img_name_reg[top_idx],
						'test_img_top_n':img_name_test[best_img_idx]},
					   ignore_index=True)
	pbar.finish()
	if save_path is not None:
		pickle.dump(df, open(abspath(save_path), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	return df

def label_folder(test_path, reg_path, df_in, save_path=None):
	df_out = pd.DataFrame(columns=['person_name','true_label'])
	fdl = listdir(test_path)
	total_num = len(fdl)
	top_n = df_in.loc[0]['reg_img_top_n'].shape[0]
	for i, fd in enumerate(fdl):
		print('{}/{}'.format(i,total_num))
		df = df_in[df_in['person_name']==fd]
		reg_names = list(df['reg_name_top_n'].values[0])
		img_r = []
		img_t = []
		for j, (rn, r, t) in enumerate(zip(reg_names, list(df['reg_img_top_n'].values[0]), list(df['test_img_top_n'].values[0]))):
			#print(join(reg_path,fd,r))
			#print(join(test_path,fd,t))
			img_r += [cv2.putText(cv2.imread(join(reg_path,rn,r)),str(j),(5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),3)]
			img_t += [cv2.imread(join(test_path,fd,t))]
		#print([tmp.shape for tmp in img_r])
		#print([tmp.shape for tmp in img_t])
		img_r = np.vstack(img_r)
		img_t = np.vstack(img_t)
		img_final = np.hstack((img_r, img_t))
		cv2.imshow('Main', img_final)
		key = cv2.waitKey() & 0xFF - ord('0')
		if key>=0 and key<top_n:
			print(reg_names[key])
			df_out = df_out.append({'person_name':fd, 'true_label':reg_names[key]}, ignore_index=True)
		else:
			print('N/A')
			df_out = df_out.append({'person_name':fd, 'true_label':None}, ignore_index=True)
	cv2.destroyAllWindows()
	if save_path is not None:
		pickle.dump(df_out, open(abspath(save_path), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)		
	return df_out

# df_ft_reg 	: df contains all the feature of register images, output of extract_feature()
# df_label_test	: df specify the true label of test images, output of label_folder()
# df_ft_test 	: df contains all the feature of test images, output of extract_feature()
def predict_test_data(df_ft_reg, df_label_test, df_ft_test, batchsize=1000, save_path=None):
	assert set(df_label_test['person_name'].values)==set(df_ft_test['person_name'].values)
	df_ft_test = df_ft_test.reset_index()
	if 'true_label' not in df_ft_test.keys():
		df_ft_test=df_ft_test.join(df_label_test.set_index('person_name'),on='person_name')
	pred = []
	score = []
	total_num = len(df_ft_test)
	ft_reg = np.array(list(df_ft_reg['feature']))
	if len(ft_reg.shape)==1:
		ft_reg = ft_reg[np.newaxis,:]
	pbar = ProgressBar(widgets=widgets, maxval=total_num)
	pbar.start()
	for i in range(0, total_num, batchsize):
		print('{}/{}'.format(i,total_num))
		pbar.update(i)
		ft_test = np.array(list(df_ft_test.loc[i:i+batchsize-1]['feature']))
		if len(ft_test.shape)==1:
			ft_test = ft_test[np.newaxis,:]
		tmp_score = (np.dot(ft_test, ft_reg.T)+1.0)*0.5
		idx_max = np.argmax(tmp_score,1).tolist()
		score += np.max(tmp_score,1).tolist()
		pred += df_ft_reg['person_name'].values[idx_max].tolist()
	pbar.finish()
	assert len(score) == total_num
	assert len(pred) == total_num
	df_ft_test['pred_label'] = pred
	df_ft_test['pred_score'] = score
	if save_path is not None:
		pickle.dump(df_ft_test, open(abspath(save_path), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)	
	return df_ft_test

# df_pred: output of predict_test_data(), it can be a dataframe of list of dataframe
def eval_pred_result(df_pred):
	assert isinstance(df_pred, pd.DataFrame) or isinstance(df_pred, list)
	if isinstance(df_pred, pd.DataFrame):
		df_pred = [df_pred]
	label = []
	score = []
	for df in df_pred:
		label += (df['true_label']==df['pred_label']).values.tolist()
		score += df['pred_score'].values.tolist()
	fpr, tpr, thresholds = roc_curve(label, score)
	roc_auc = auc(fpr, tpr)
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()
	return fpr,tpr,thresholds,roc_auc

'''
## to label the test folder, only need to be done once with best model
import sys
import pickle
sys.path.append('/home/macul/Documents/macul')
sys.path.append('/home/macul/Documents/macul/mklib/utils')
from mxFaceFeatureExtract import mxFaceFeatureExtract
from modelBenchMarking import *
extractor=mxFaceFeatureExtract('/media/macul/black/mxnet_training/r50/server_train18','train_18',62, outputs_name={'embedding':'embedding_output'},mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
df_reg=extract_feature(extractor.getEmbedding, './regis_112x112', './reg_df.pkl')
df_test=extract_feature(extractor.getEmbedding, './test', './test_df.pkl')
df_rank=rank_folder(df_reg,df_test)
df_label=label_folder('./test', './regis_112x112', df_rank, './label_df.pkl')


## evaluate model
import sys
import pickle
sys.path.append('/home/macul/Documents/macul')
sys.path.append('/home/macul/Documents/macul/mklib/utils')
from mxFaceFeatureExtract import mxFaceFeatureExtract
from modelBenchMarking import *
extractor=mxFaceFeatureExtract('/media/macul/black/mxnet_training/r50/server_train18','train_18',62, outputs_name={'embedding':'embedding_output'},mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
df_reg=extract_feature(extractor.getEmbedding, './regis_112x112', './reg_df.pkl')
#df_reg = pickle.load(open('./reg_df.pkl','rb'))
df_test=extract_feature(extractor.getEmbedding, './test', './test_df.pkl')
#df_test = pickle.load(open('./test_df.pkl','rb'))
df_label=pickle.load(open('./label_df.pkl','rb'))
df_pred=predict_test_data(df_reg, df_label, df_test, save_path='./pred_df.pkl')
fpr,tpr,thresholds,roc_auc=eval_pred_result(df_pred)
'''