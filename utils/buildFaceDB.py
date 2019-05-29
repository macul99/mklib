from __future__ import print_function
import sys
#sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
import mysql.connector # sudo pip install mysql-connector-python
# conda install -c anaconda mysql-connector-python
from mysql.connector import errorcode
import re
import numpy as np 
import pandas as pd
from datetime import date, datetime, timedelta
import os
from os.path import isdir, isfile, join, exists
from imutils import paths
from os import listdir, makedirs

class buildFaceDB():
    def __init__(self, db_usr_name='root', db_pwd='iim@1234', db_name='ZhaoGuang'):
        self.db_usr_name = db_usr_name
        self.db_pwd = db_pwd
        self.db_name = db_name
        self.connect_database()

    def connect_database(self):
        try:
            self.cnx = mysql.connector.connect(  user=self.db_usr_name, password=self.db_pwd, database=self.db_name)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print(self.db_name, ' does not exist!')
                return
            else:
                print(err)
                return
        else:
            self.cursor = self.cnx.cursor()
        return

    def get_userid_imgurl(self):
        query = (   "select user_id,image_url from user_feature" )
        self.cursor.execute(query, ())
        df=pd.DataFrame(self.cursor.fetchall(),columns=self.cursor.column_names)
        return df


    # dir_path should points to the folder of registration pictures which match the database, the folder can be different from database, but image name must match
    def get_embedding_from_db(self, extractor, df, dir_path='/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop'):
        df_rst = pd.DataFrame(columns=['user_id', 'image_path', 'embedding'])
        for idx, row in df.iterrows():
            img_path = join(dir_path,str(row['image_url'].split('/')[-1]))
            if isfile(img_path):
                df_rst = df_rst.append(pd.Series([row['user_id'], img_path, extractor(img_path)], index=['user_id', 'image_path', 'embedding']),ignore_index=True)
            print(img_path)
        return df_rst

    def get_embedding(self, extractor, dir_path='/media/macul/black/face_database_raw_data/faceRecog_out/database/var/data/ego_java_server_data/IIM_Images_Crop'):
        df_rst = pd.DataFrame(columns=['image_path', 'embedding'])
        for i, img_path in enumerate(list(paths.list_images(dir_path))):
            df_rst = df_rst.append(pd.Series([img_path, extractor(img_path)], index=['image_path', 'embedding']),ignore_index=True)
            print(img_path)
        return df_rst


    def get_embedding_to_txt(self, extractor, dir_path='/media/macul/black/face_database_raw_data/faceRecog_out/database/var/data/ego_java_server_data/IIM_Images_Crop', 
                             dst_dir='/media/macul/black/face_database_raw_data/faceRecog_out_database/var/data/ego_java_server_data/IIM_Images_Crop_txt'):
        if dir_path[-1]=='/':
            dir_path=dir_path[0:-1]
        if dst_dir[-1]=='/':
            dst_dir=dst_dir[0:-1]
        dir_path_len = len(dir_path)
        for i, img_path in enumerate(list(paths.list_images(dir_path))):
            dst_txt_path = join(dst_dir,img_path[dir_path_len+1:])
            dst_txt_path = dst_txt_path[0:dst_txt_path.rfind('.')]+'.txt'
            if not exists(dst_txt_path[0:dst_txt_path.rfind('/')]):
                makedirs(dst_txt_path[0:dst_txt_path.rfind('/')])
            embedding = extractor(img_path)
            print(img_path)
            np.savetxt(dst_txt_path, embedding[np.newaxis,:], delimiter=' ')
        return


'''
import sys
import pickle
from buildFaceDB import buildFaceDB
from mxFaceFeatureExtract import mxFaceFeatureExtract
buildDB = buildFaceDB()
extractor=mxFaceFeatureExtract('/media/macul/black/mxnet_training/mobilefacenet/server_train3','train_3',34, outputs_name={'embedding':'embedding_output'},mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json',out_dim=128)
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_mf_server_train3_34')

import sys
import pickle
from buildFaceDB import buildFaceDB
from mxFaceFeatureExtract import mxFaceFeatureExtract
buildDB = buildFaceDB()
extractor=mxFaceFeatureExtract('/media/macul/black/mxnet_training/mobilefacenet/server_train2','train_2',63, outputs_name={'embedding':'embedding_output'},mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json',out_dim=128)
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_mf_server_train2_63')

import sys
import pickle
from buildFaceDB import buildFaceDB
from mxFaceFeatureExtract import mxFaceFeatureExtract
buildDB = buildFaceDB()
extractor=mxFaceFeatureExtract('/media/macul/black/mxnet_training/mobilefacenet/dgx_train2','train_2',32, outputs_name={'embedding':'embedding_output'},mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json',out_dim=128)
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_mf_dgx_train2_32')

import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/insightface-r50-am-lfw','model',0,outputs_name={'embedding':'fc1_output'}, mean_value=None)
buildDB.get_embedding_to_txt(extractor.getEmbeddingNoNorm, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_insight_original')


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/server_train16','train_16',41, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_server_train16')


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/server_train17','train_17',19, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_server_train17')


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/server_train18','train_18',6062, outputs_name={'embedding':'embedding_output'},mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_server_train18')


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/server_train18','train_18',69, outputs_name={'embedding':'embedding_output'},mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_server_train18_69')

import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/server_train16','train_16',3942, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_server_train16_3942')


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/server_train19','train_19',92, outputs_name={'embedding':'embedding_output'},mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_server_train19')

import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/dgx_train1','train_1',31, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_dgx_train1')


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/dgx_train2','train_2',23, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_dgx_train2')


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/dgx_train3','train_3',55, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_dgx_train3')


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/dgx_train4','train_4',68, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_dgx_train4')


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/dgx_train6','train_6',53, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_dgx_train6')

import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from tfFeatureExtract import tfFeatureExtract
buildDB = buildFaceDB()
extractor=tfFeatureExtract('/media/macul/black/MobileFaceNet_TF/MobileFaceNet_9925_9680.pb',outputs_name={'embedding':'prefix'})
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_tf_mobilenet')


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/insightface-r50-am-lfw','model',0,outputs_name={'embedding':'fc1_output'}, mean_value=None)
df_db=buildDB.get_userid_imgurl()
df=buildDB.get_embedding_from_db(extractor.getEmbeddingNoNorm, df_db)
pickle.dump(df, open('db_insight_original.pkl','wb'))
df1=buildDB.get_embedding(extractor.getEmbeddingNoNorm, '/media/macul/black/face_database_raw_data/faceRecog_out_crop')
pickle.dump(df1, open('faceRecogOut_insight_original.pkl','wb'))


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/Baseline_insightDatabase_single_loss','train_11',31,outputs_name={'embedding':'out_embedding_output'}, mean_value=None)
df_db=buildDB.get_userid_imgurl()
df=buildDB.get_embedding_from_db(extractor.getEmbeddingNoNorm, df_db)
pickle.dump(df, open('db_insight_baseline.pkl','wb'))
df1=buildDB.get_embedding(extractor.getEmbeddingNoNorm, '/media/macul/black/face_database_raw_data/faceRecog_out_crop')
pickle.dump(df1, open('faceRecogOut_insight_baseline.pkl','wb'))


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/Baseline_insightDB_landmarkloss_before_bugfix','train_14',33, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
df_db=buildDB.get_userid_imgurl()
df=buildDB.get_embedding_from_db(extractor.getEmbedding, df_db)
pickle.dump(df, open('db_landmarkloss_bfBugFix.pkl','wb'))
df1=buildDB.get_embedding(extractor.getEmbedding, '/media/macul/black/face_database_raw_data/faceRecog_out_crop')
pickle.dump(df1, open('faceRecogOut_landmarkloss_bfBugFix.pkl','wb'))


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/dgx_train1','train_1',31, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
df_db=buildDB.get_userid_imgurl()
df=buildDB.get_embedding_from_db(extractor.getEmbedding, df_db)
pickle.dump(df, open('db_dgx_train1.pkl','wb'))
df1=buildDB.get_embedding(extractor.getEmbedding, '/media/macul/black/face_database_raw_data/faceRecog_out_crop')
pickle.dump(df1, open('faceRecogOut_dgx_train1.pkl','wb'))


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/server_train16','train_16',41, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
df_db=buildDB.get_userid_imgurl()
df=buildDB.get_embedding_from_db(extractor.getEmbedding, df_db)
pickle.dump(df, open('db_server_train16.pkl','wb'))
df1=buildDB.get_embedding(extractor.getEmbedding, '/media/macul/black/face_database_raw_data/faceRecog_out_crop')
pickle.dump(df1, open('faceRecogOut_server_train16.pkl','wb'))


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from caffeFeatureExtract import caffeFeatureExtract
buildDB = buildFaceDB()
extractor=caffeFeatureExtract('/media/macul/black/caffe_recognition_models',prototxt_name='sphereface_deploy',caffemodel_name='sphereface20_ms1m',outputs_name={'embedding':'fc5'}, img_width=96)
df_db=buildDB.get_userid_imgurl()
df=buildDB.get_embedding_from_db(extractor.getEmbedding, df_db)
pickle.dump(df, open('db_caffe_sphereface.pkl','wb'))
df1=buildDB.get_embedding(extractor.getEmbedding, '/media/macul/black/face_database_raw_data/faceRecog_out_crop')
pickle.dump(df1, open('faceRecogOut_caffe_sphereface.pkl','wb'))


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from caffeFeatureExtract import caffeFeatureExtract
buildDB = buildFaceDB()
extractor=caffeFeatureExtract('/media/macul/black/caffe_recognition_models',prototxt_name='sphereface_deploy',caffemodel_name='sphereface_asianface_glass_pp',outputs_name={'embedding':'fc5'}, img_width=96)
df_db=buildDB.get_userid_imgurl()
df=buildDB.get_embedding_from_db(extractor.getEmbedding, df_db)
pickle.dump(df, open('db_caffe_sphereface_asianface_glass_pp.pkl','wb'))
df1=buildDB.get_embedding(extractor.getEmbedding, '/media/macul/black/face_database_raw_data/faceRecog_out_crop')
pickle.dump(df1, open('faceRecogOut_caffe_sphereface_asianface_glass_pp.pkl','wb'))

import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from caffeFeatureExtract import caffeFeatureExtract
buildDB = buildFaceDB()
extractor=caffeFeatureExtract('/media/macul/black/caffe_recognition_models',prototxt_name='sphereface_deploy',caffemodel_name='sphereface_asianface',outputs_name={'embedding':'fc5'}, img_width=96)
df_db=buildDB.get_userid_imgurl()
df=buildDB.get_embedding_from_db(extractor.getEmbedding, df_db)
pickle.dump(df, open('db_caffe_sphereface_asianface.pkl','wb'))
df1=buildDB.get_embedding(extractor.getEmbedding, '/media/macul/black/face_database_raw_data/faceRecog_out_crop')
pickle.dump(df1, open('faceRecogOut_caffe_sphereface_asianface.pkl','wb'))

import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from caffeFeatureExtract import caffeFeatureExtract
buildDB = buildFaceDB()
extractor=caffeFeatureExtract('/media/macul/black/caffe_recognition_models',prototxt_name='sphereface_deploy',caffemodel_name='sphereface_deepglint',outputs_name={'embedding':'fc5'}, img_width=96)
df_db=buildDB.get_userid_imgurl()
df=buildDB.get_embedding_from_db(extractor.getEmbedding, df_db)
pickle.dump(df, open('db_caffe_sphereface_deepglint.pkl','wb'))
df1=buildDB.get_embedding(extractor.getEmbedding, '/media/macul/black/face_database_raw_data/faceRecog_out_crop')
pickle.dump(df1, open('faceRecogOut_caffe_sphereface_deepglint.pkl','wb'))


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from caffeFeatureExtract import caffeFeatureExtract
buildDB = buildFaceDB()
extractor=caffeFeatureExtract('/home/macul/Projects/ego_mk_op/ego/recognition/models',prototxt_name='sphereface_deploy',caffemodel_name='sphereface20_ms1m',outputs_name={'embedding':'fc5'}, img_width=96)
df_db=buildDB.get_userid_imgurl()
df=buildDB.get_embedding_from_db(extractor.getEmbedding, df_db)
pickle.dump(df, open('db_caffe_sphereface20_ms1m.pkl','wb'))
df1=buildDB.get_embedding(extractor.getEmbedding, '/media/macul/black/face_database_raw_data/faceRecog_out_crop')
pickle.dump(df1, open('faceRecogOut_caffe_sphereface20_ms1m.pkl','wb'))


import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from tfFeatureExtract import tfFeatureExtract
buildDB = buildFaceDB()
extractor=tfFeatureExtract('/media/macul/black/MobileFaceNet_TF/MobileFaceNet_9925_9680.pb',outputs_name={'embedding':'prefix'})
df_db=buildDB.get_userid_imgurl()
df=buildDB.get_embedding_from_db(extractor.getEmbedding, df_db)
pickle.dump(df, open('db_tf_mobilenet.pkl','wb'))
df1=buildDB.get_embedding(extractor.getEmbedding, '/media/macul/black/face_database_raw_data/faceRecog_out_crop')
pickle.dump(df1, open('faceRecogOut_tf_mobilenet.pkl','wb'))





'''