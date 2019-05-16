# convert rec file to image
import sys
from os.path import isdir, isfile, join, exists
from os import mkdir
from os import listdir, makedirs
import os
import cv2
import numpy as np
from imutils import paths
import progressbar
from PIL import Image
import pickle
#from dan import dan
#from ssd import ssd
import time
import pandas as pd



class faceVerifyIdentify():
    def __init__(self,  src_folder, 
                        ref_folder, 
                        df_embedding_path):
        assert isdir(src_folder)
        assert isdir(ref_folder)
        assert isfile(df_embedding_path)
        self.src_folder = src_folder
        self.ref_folder = ref_folder
        self.df_embedding_path = df_embedding_path
        with open(self.df_embedding_path,'rb') as f:
            self.df_emb = pickle.load(f)

    # the source folder name should be user_id
    # the des_folder image name should match database image_url
    def verify_face(self):
        src_img_list = list(paths.list_images(src_folder))
        src_img_list.sort()

        for src_img_path in src_img_list:
            usr_id = src_img_path.split('/')[-2]
            dst_image_path = self.df_emb[self.df_emb['user_id']==usr_id]['image_path']
            assert isfile(src_img_path)
            assert isfile(dst_img_path)
            print('src_path: ', src_img_path)
            print('dst_path: ', dst_img_path)
            img_src = cv2.imread(src_img_path)
            img_dst = cv2.imread(dst_img_path)
            combine = np.concatenate((img_src,img_dst),axis=1)

            cv2.imshow('frame',combine)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return


    # df_src should have columns ['image_path', 'embedding']
    def identify_face(self, df_src_path):
        assert isfile(df_src_path)
        with open(df_src_path,'rb') as f:
            df_src = pickle.load(f)
        df_rst = pd.DataFrame(columns=['src_path', 'dst_path', 'user_id', 'score'])
        user_id_list = self.df_emb['user_id'].tolist()
        image_path_list = self.df_emb['image_path'].tolist()
        embedding_list = self.df_emb['embedding'].tolist()
        totalLen=len(df_src)
        for idx, row in df_src.iterrows():
            print('{}/{}'.format(idx,totalLen))
            score = np.array([])
            for emb in embedding_list:
                score = np.append(score, np.dot(row['embedding'], emb))            
            top_5 = score.argsort()[::-1][0:5]
            #print(np.max(score),np.min(score), score[top_5])
            df_rst = df_rst.append(pd.Series([  row['image_path'], 
                                                [image_path_list[n] for n in top_5],
                                                [user_id_list[n] for n in top_5],
                                                [score[n] for n in top_5],
                                                ],
                                             index=['src_path', 'dst_path', 'user_id', 'score']), ignore_index=True)
            #break
        return df_rst

'''
from faceVerifyIdentify import faceVerifyIdentify
fvi=faceVerifyIdentify('/media/macul/black/face_database_raw_data/faceRecog_out_crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/db_caffe_sphereface20_ms1m.pkl')
df_rst=fvi.identify_face('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_caffe_sphereface20_ms1m.pkl')
#fvi=faceVerifyIdentify('/media/macul/black/face_database_raw_data/faceRecog_out_crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/db_server_train16.pkl')
#df_rst=fvi.identify_face('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_server_train16.pkl')
#fvi=faceVerifyIdentify('/media/macul/black/face_database_raw_data/faceRecog_out_crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/db_caffe_sphereface_deepglint.pkl')
#df_rst=fvi.identify_face('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_caffe_sphereface_deepglint.pkl')
#fvi=faceVerifyIdentify('/media/macul/black/face_database_raw_data/faceRecog_out_crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/db_dgx_train1.pkl')
#df_rst=fvi.identify_face('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_dgx_train1.pkl')
#fvi=faceVerifyIdentify('/media/macul/black/face_database_raw_data/faceRecog_out_crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/db_tf_mobilenet.pkl')
#df_rst=fvi.identify_face('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_tf_mobilenet.pkl')
#fvi=faceVerifyIdentify('/media/macul/black/face_database_raw_data/faceRecog_out_crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/db_caffe_sphereface_asianface.pkl')
#df_rst=fvi.identify_face('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_caffe_sphereface_asianface.pkl')
#fvi=faceVerifyIdentify('/media/macul/black/face_database_raw_data/faceRecog_out_crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/db_insight_original.pkl')
#df_rst=fvi.identify_face('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_insight_original.pkl')
#fvi=faceVerifyIdentify('/media/macul/black/face_database_raw_data/faceRecog_out_crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/db_insight_baseline.pkl')
#df_rst=fvi.identify_face('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_insight_baseline.pkl')
#fvi=faceVerifyIdentify('/media/macul/black/face_database_raw_data/faceRecog_out_crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/db_landmarkloss_bfBugFix.pkl')
#df_rst=fvi.identify_face('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_landmarkloss_bfBugFix.pkl')
#fvi=faceVerifyIdentify('/media/macul/black/face_database_raw_data/faceRecog_out_crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/db_caffe_sphereface.pkl')
#df_rst=fvi.identify_face('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_caffe_sphereface.pkl')
#fvi=faceVerifyIdentify('/media/macul/black/face_database_raw_data/faceRecog_out_crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop','/media/macul/black/face_database_raw_data/faceRecog_out_database/db_caffe_sphereface_asianface_glass_pp.pkl')
#df_rst=fvi.identify_face('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_caffe_sphereface_asianface_glass_pp.pkl')

df_rst['user_id_top0'] = df_rst['user_id'].apply(lambda x: x[0])
df_rst['score_top0'] = df_rst['score'].apply(lambda x: x[0])
df_rst['score_top1'] = df_rst['score'].apply(lambda x: x[1])
df_rst['angle_top0']=np.arccos(df_rst['score_top0'])
df_rst['angle_top1']=np.arccos(df_rst['score_top1'])
df_rst['dst_path_top0'] = df_rst['dst_path'].apply(lambda x: x[0])
df_rst['dst_path_top0_name']=df_rst['dst_path_top0'].apply(lambda x: x.split('/')[-1].split('.')[0])
df_rst['src_path_name']=df_rst['src_path'].apply(lambda x: x.split('/')[-2])


df50 = df_rst[df_rst['score_top0']>0.5]
np.mean(df50['angle_top1']-df50['angle_top0'])


def process_df(df_rst):
    df_rst['user_id_top0'] = df_rst['user_id'].apply(lambda x: x[0])
    df_rst['score_top0'] = df_rst['score'].apply(lambda x: x[0])
    df_rst['score_top1'] = df_rst['score'].apply(lambda x: x[1])
    df_rst['angle_top0']=np.arccos(df_rst['score_top0'])
    df_rst['angle_top1']=np.arccos(df_rst['score_top1'])
    df_rst['dst_path_top0'] = df_rst['dst_path'].apply(lambda x: x[0])
    df_rst['dst_path_top0_name']=df_rst['dst_path_top0'].apply(lambda x: x.split('/')[-1].split('.')[0])
    df_rst['src_path_name']=df_rst['src_path'].apply(lambda x: x.split('/')[-2])
    return df_rst


import numpy as np
import pandas as pd
import pickle
from PIL import Image
import psutil
def process_df(df_rst):
    df_rst['user_id_top0'] = df_rst['user_id'].apply(lambda x: x[0])
    df_rst['score_top0'] = df_rst['score'].apply(lambda x: x[0])
    df_rst['score_top1'] = df_rst['score'].apply(lambda x: x[1])
    df_rst['angle_top0']=np.arccos(df_rst['score_top0'])
    df_rst['angle_top1']=np.arccos(df_rst['score_top1'])
    df_rst['dst_path_top0'] = df_rst['dst_path'].apply(lambda x: x[0])
    df_rst['dst_path_top0_name']=df_rst['dst_path_top0'].apply(lambda x: x.split('/')[-1].split('.')[0])
    df_rst['src_path_name']=df_rst['src_path'].apply(lambda x: x.split('/')[-2])
    return df_rst

df1=pd.read_pickle('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_match_db_insight_original.pkl')
#df2=pd.read_pickle('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_match_db_tf_mobilenet.pkl')
#df2=pd.read_pickle('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_match_db_insight_baseline.pkl')
df2=pd.read_pickle('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_match_db_caffe_sphereface_deepglint.pkl')
#df3=pd.read_pickle('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_match_db_dgx_train1.pkl')
#df3=pd.read_pickle('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_match_db_landmarkloss_bfBugFix.pkl')
df3=pd.read_pickle('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_match_db_caffe_sphereface.pkl')
#df3=pd.read_pickle('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_match_db_caffe_sphereface_asianface_glass_pp.pkl')
#df3=pd.read_pickle('/media/macul/black/face_database_raw_data/faceRecog_out_crop/faceRecogOut_match_db_caffe_sphereface_asianface.pkl')
df1=process_df(df1)
df2=process_df(df2)
df3=process_df(df3)
df12=df1.merge(df2, on=['src_path', 'dst_path_top0_name'])
df13=df1.merge(df3, on=['src_path', 'dst_path_top0_name'])
df23=df2.merge(df3, on=['src_path', 'dst_path_top0_name'])
df123=df12.merge(df3, on=['src_path', 'dst_path_top0_name'])

df12_05=df12[(df12['score_top0_x']>0.5)&(df12['score_top0_y']>0.5)]
df12_05=df12_05.sort_values(['dst_path_top0_name'])
df12_05=df12_05.reset_index()

df13_05=df13[(df13['score_top0_x']>0.5)&(df13['score_top0_y']>0.5)]
df13_05=df13_05.sort_values(['dst_path_top0_name'])
df13_05=df13_05.reset_index()

df23_05=df23[(df23['score_top0_x']>0.5)&(df23['score_top0_y']>0.5)]
df23_05=df23_05.sort_values(['dst_path_top0_name'])
df23_05=df23_05.reset_index()

df123_05=df123[(df123['score_top0_x']>0.5)&(df123['score_top0_y']>0.5)&(df123['score_top0']>0.5)]
df123_05=df123_05.sort_values(['dst_path_top0_name'])
df123_05=df123_05.reset_index()

a=df12_05[df12_05['score_top0_x']<0.501][['src_path','dst_path_top0_name','score_top0_x','score_top0_y']].values
for ai in a:
    print(ai)
    img1=Image.open(ai[0])
    img1.show()
    img2=Image.open('/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop/{}.jpg'.format(ai[1]))
    img2.show()
    raw_input()
    for proc in psutil.process_iter():
        if proc.name()=='display':
            proc.kill()

a=df12_05[df12_05['score_top0_y']<0.501][['src_path','dst_path_top0_name','score_top0_x','score_top0_y']].values
for ai in a:
    print(ai)
    img1=Image.open(ai[0])
    img1.show()
    img2=Image.open('/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop/{}.jpg'.format(ai[1]))
    img2.show()
    raw_input()
    for proc in psutil.process_iter():
        if proc.name()=='display':
            proc.kill()


a=df13_05[df13_05['score_top0_x']<0.501][['src_path','dst_path_top0_name','score_top0_x','score_top0_y']].values
for ai in a:
    print(ai)
    img1=Image.open(ai[0])
    img1.show()
    img2=Image.open('/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop/{}.jpg'.format(ai[1]))
    img2.show()
    raw_input()
    for proc in psutil.process_iter():
        if proc.name()=='display':
            proc.kill()

a=df13_05[df13_05['score_top0_y']<0.501][['src_path','dst_path_top0_name','score_top0_x','score_top0_y']].values
for ai in a:
    print(ai)
    img1=Image.open(ai[0])
    img1.show()
    img2=Image.open('/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop/{}.jpg'.format(ai[1]))
    img2.show()
    raw_input()
    for proc in psutil.process_iter():
        if proc.name()=='display':
            proc.kill()


a=df123_05[df123_05['score_top0_x']<0.501][['src_path','dst_path_top0_name','score_top0_x','score_top0_y','score_top0']].values
for ai in a:
    print(ai)
    img1=Image.open(ai[0])
    img1.show()
    img2=Image.open('/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop/{}.jpg'.format(ai[1]))
    img2.show()
    raw_input()
    for proc in psutil.process_iter():
        if proc.name()=='display':
            proc.kill()

df1_50=df1[df1['score_top0']>0.50]
df3_46=df3[df3['score_top0']>0.46]
a=df13_m[df13_m['dst_path_top0_name_x']!=df13_m['dst_path_top0_name_y']][['src_path','dst_path_top0_name_x','dst_path_top0_name_y','score_top0_x','score_top0_y']].values
for ai in a:
    print(ai)
    img1=Image.open(ai[0])
    img1.show()
    img2=Image.open('/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop/{}.jpg'.format(ai[1]))
    img2.show()
    img3=Image.open('/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop/{}.jpg'.format(ai[2]))
    img3.show()
    raw_input()
    for proc in psutil.process_iter():
        if proc.name()=='display':
            proc.kill()
'''