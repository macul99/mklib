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
import shutil
#from unionfind import UnionFind
import argparse

widgets = ['Test: ', Percentage(), ' ', Bar(marker='|',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options

args = argparse.ArgumentParser()
args.add_argument("-rrt", "--reg-root", required=True, help="path to reg root folder")
args.add_argument("-trt", "--test-root", required=True, help="path to test root folder")
args.add_argument("-tlb", "--test-label", required=True, help="path to test label file")
args.add_argument("-rft", "--reg-ft", required=True, help="path to reg featrue dataframe file")
args.add_argument("-tft", "--test-ft", required=True, help="path to test featrue dataframe file")
args.add_argument("-rst", "--result", required=True, help="path to result pickle file file")


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

def label_folder(test_path, reg_path, df_in, th_l=0.75, th_h=0.8, save_interval=50, save_path=None):
    if save_path is not None:
        f_w = open(abspath(save_path), 'wb')
    df_out = pd.DataFrame(columns=['person_name','true_label'])
    fdl = listdir(test_path)
    fdl.sort()
    total_num = len(fdl)
    top_n = df_in.loc[0]['reg_img_top_n'].shape[0]
    for i, fd in enumerate(fdl):        
        if save_path is not None and i%save_interval==0:
            print('Saving data...')
            pickle.dump(df_out, f_w, protocol=pickle.HIGHEST_PROTOCOL)
        df = df_in[df_in['person_name']==fd]
        reg_names = list(df['reg_name_top_n'].values[0])
        top_scores = list(df['score_top_n'].values[0])
        if top_scores[0]<th_l:
            df_out = df_out.append({'person_name':fd, 'true_label':None}, ignore_index=True)
            continue
        elif top_scores[0]>=th_h:
            df_out = df_out.append({'person_name':fd, 'true_label':reg_names[0]}, ignore_index=True)
            continue
        print('{}/{}'.format(i,total_num), reg_names)
        img_r = []
        img_t = []
        img_s = []
        for j, (rn, r, t) in enumerate(zip(reg_names, list(df['reg_img_top_n'].values[0]), list(df['test_img_top_n'].values[0]))):
            #print(join(reg_path,fd,r))
            #print(join(test_path,fd,t))
            img_r += [cv2.putText(cv2.imread(join(reg_path,rn,r)),str(j),(5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),3)]
            img_t += [cv2.imread(join(test_path,fd,t))]
            img_s += [cv2.putText(np.ones([112,112,3]).astype(np.uint8)*255,str(top_scores[j]),(5,65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,0),3)]
        #print([tmp.shape for tmp in img_r])
        #print([tmp.shape for tmp in img_t])
        img_r = np.vstack(img_r)
        img_t = np.vstack(img_t)
        img_s = np.vstack(img_s)
        img_final = np.hstack((img_r, img_t, img_s))
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
        pickle.dump(df_out, f_w, protocol=pickle.HIGHEST_PROTOCOL)      
    return df_out

# df_ft_reg     : df contains all the feature of register images, output of extract_feature()
# df_label_test : df specify the true label of test images, output of label_folder()
# df_ft_test    : df contains all the feature of test images, output of extract_feature()
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

def clean_reg(df_rank, reg_root, th=0.75, save_path=None):
    df_rank = df_rank.reset_index()
    uf = UnionFind(df_rank['person_name'].tolist())
    total_num = len(df_rank)
    for i in range(len(df_rank)):
        print('{}/{}'.format(i, total_num))
        test_name = df_rank.loc[i]['person_name']
        idx_th = df_rank.loc[i]['score_top_n']>th
        idx_th[0] = False # don't count itself
        reg_name = df_rank.loc[i]['reg_name_top_n'][idx_th].tolist()
        reg_img_name = df_rank.loc[i]['reg_img_top_n'][idx_th].tolist()
        test_img_name = df_rank.loc[i]['test_img_top_n'][idx_th].tolist()
        for rn, rin, tin in zip(reg_name, reg_img_name, test_img_name):
            if uf.connected(test_name, rn): continue
            #print(test_name)
            #print(rn)          
            img_t = cv2.imread(join(reg_root,test_name,tin))
            img_r = cv2.imread(join(reg_root,rn,rin))
            img_final = np.hstack((img_r, img_t))
            cv2.imshow('Main', img_final)
            key = cv2.waitKey() & 0xFF - ord('0')
            if key==0:
                uf.union(test_name, rn)
    cv2.destroyAllWindows()
    rst = uf.components()
    if save_path is not None:
        pickle.dump(rst, open(abspath(save_path), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return rst

def df_reg_to_dict(df_reg, save_path=None):
    rst = {}
    for i in range(len(df_reg)):
        rst[df_reg.loc[i]['img_name']] = df_reg.loc[i]['feature']
    if save_path is not None:
        pickle.dump(rst, open(abspath(save_path), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)  
    return rst

def combine_reg_folder(list_clean, src, dst):
    if not exists(dst):
        makedirs(dst)
    total_num = len(list_clean)
    for i, k in enumerate(list_clean):
        k = list(k)
        print('{}/{}'.format(i,total_num))
        print(k)
        shutil.copytree(join(src,k[0]),join(dst,k[0]))
        for j in range(1, len(k)):
            for file in listdir(join(src,k[j])):
                shutil.copy(join(src,k[j],file),join(dst,k[0]))


if __name__ == '__main__': 

    ## define feature extractor function
    import sys
    sys.path.append('/home/macul/Documents/macul/mklib/utils')
    from mxFaceFeatureExtract import mxFaceFeatureExtract
    extractor=mxFaceFeatureExtract('/media/macul/black/mxnet_training/r50/server_train18','train_18',62, outputs_name={'embedding':'embedding_output'},mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')

    args = vars(args.parse_args())
    args['reg_root'] = abspath(args['reg_root'])
    args['test_root'] = abspath(args['test_root'])
    args['reg_ft'] = abspath(args['reg_ft'])
    args['test_ft'] = abspath(args['test_ft'])
    args['test_label'] = abspath(args['test_label'])
    args['result'] = abspath(args['result'])

    assert exists(args['reg_root'])
    assert exists(args['test_root'])
    assert exists(args['test_label'])

    fw = open(args['result'],'wb')
    fw.close()

    df_label=pickle.load(open(args['test_label'],'rb'))

    if exists(args['reg_ft']):
        print('Load reg features ...')
        df_reg = pickle.load(open(args['reg_ft'],'rb'))
    else:
        print('Extracting reg features ...')
        df_reg=extract_feature(extractor.getEmbedding, args['reg_root'], args['reg_ft'])

    if exists(args['test_ft']):
        print('Load test features ...')
        df_test = pickle.load(open(args['test_ft'],'rb'))
    else:
        print('Extracting test features ...')
        df_test=extract_feature(extractor.getEmbedding, args['test_root'], args['test_ft'])

    print('Verifacation in progress ...')
    df_pred=predict_test_data(df_reg, df_label, df_test)
    fpr,tpr,thresholds,roc_auc=eval_pred_result(df_pred)

    rst = {}
    rst['pred'] = df_pred
    rst['fpr'] = fpr
    rst['tpr'] = tpr
    rst['thresholds'] = thresholds
    rst['roc_auc'] = roc_auc

    pickle.dump(rst, open(args['result'], 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


# python -m modelBenchMarking -rrt ./dataset -trt ./iim_dataset_test_112x112 -tlb ./label_df.pkl -rft ./reg_resnet50.pkl -tft ./test_resnet50.pkl -rst ./rst_resnet50.pkl

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
df_rank=rank_folder(df_reg,df_test,save_path='./rank_df.pkl')
#df_rank=pickle.load(open('./rank_df.pkl','rb'))
df_label=label_folder('./iim_dataset_test_112x112', './dataset', df_rank, save_path='./label_df.pkl')


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



# clean reg folder
df_rank=pickle.load(open('./rank_df.pkl','rb'))
dic_clean=clean_reg(df_rank, './regis_112x112', th=0.75, save_path='./clean_dict_itr1.pkl')
#dic_clean=pickle.load(open('./clean_dict_itr1.pkl','rb'))
combine_reg_folder(dic_clean,'./regis_112x112','./itr1')
'''