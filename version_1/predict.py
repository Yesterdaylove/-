# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import dlib
from face_net import Facenet
import cv2
import os
import pandas as pd
def debug(model,cut=1):
    while True:
#1
        image_1 = input('Input image_1 filename:')
        print(str(image_1))
        image_1 = detect_face(image_1,cut)
        if image_1=="error":
            print('Image_1 Open Error! Try again!')
            continue
        else:
            print(image_1.size)
#2
        image_2 = input('Input image_2 filename:')
        image_2 = detect_face(image_2,cut)
        if image_2=="error":
            print('Image_2 Open Error! Try again!')
            continue
        else:
            print(image_2.size)
        model.detect_image(image_1,image_2)

def first_processing(model,face_data_dir:str):
    """
    return None

    model : facenet
    face_data : folder_path
    ------
    人脸文件夹下图片命名格式:
        {姓名1}_1.jpg
        {姓名1}_2.jpg
        {姓名2}_1.jpg
        {姓名3}_1.jpg
    useage:
        Saving config embeddings to embedding_save
    notice:
        第一次运行模型，将对现有人脸数据进行处理并统一保存
        再次运行将根据现有人脸数据重置特征信息
    note:
        对于一人有多张人脸数据的进行特征mean处理
    """
    for root, dirs, files in os.walk(face_data_dir):
        break
    feature_embedding = []
    names = []
    for file in files:
        image_path = os.path.join(root,file)
        feature_embedding.append(model.get_embedding(Image.open(image_path)))
        names.append(file.split('_')[0])
    feature_embedding = np.array(feature_embedding).reshape(len(names),-1)
    feature_col = ['f'+str(i) for i in range(128)]
    Save_feature = pd.DataFrame(names)
    Save_feature.columns=['name']
    embedding_df = pd.DataFrame(feature_embedding,columns=feature_col)
    Save_feature = pd.concat((Save_feature,embedding_df),1)
    Save_feature[feature_col] = Save_feature.groupby('name')[feature_col].transform('mean')
    Save_feature = Save_feature.drop_duplicates()
    Save_feature.to_pickle('./embedding_save/person_feature.pkl')
    
    return names,feature_embedding,Save_feature

def update_embedding(model,image_path,name):
    """
    Returns None
    model : facenet
    face_data : folder_path
    -------
    useage:
        Saving update people embeddings into dir named embedding_sace.
    notice:
        新增人脸更新embedding
        一定要是新增（未曾出现）
    note:
        
    """
    embedding_df = pd.read_pickle('./embedding_save/person_feature.pkl')
    feature_embedding_add = model.get_embedding(Image.open(image_path))
    feature_embedding_add = np.array(feature_embedding_add).reshape(-1)
    feature_columns = ['f'+str(i) for i in range(128)]
    update_feature = pd.DataFrame(name)
    update_feature.columns =['name']
    update_embedding_df = pd.DataFrame(feature_embedding_add,columns=feature_columns)
    update_feature  = pd.concat((update_feature,update_embedding_df),1)
    embedding_df = pd.concat((embedding_df,update_feature))
    embedding_df.to_pickle('./embedding_save/person_feature.pkl')
    return "update down"

def detect_face(image_path,cut=True):
    """
    Parameters
    ----------
    image_path : TYPE
        DESCRIPTION.
    cut : TYPE, optional
        采用人脸检测，对人脸进行划分. default is True.
    Returns
    -------
    TYPE
        Image
    """
    if cut:
        image = cv2.imread(image_path)
        #对图片灰度化处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 检测人脸，探测图片中的人脸
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)
        if len(faces)!=1:
            print("Error!")
            return "error"
        else:
            # 分割人脸，方便获取人脸特征
            x1 = faces[0].left() 
            y1 = faces[0].top() 
            x2 = faces[0].right() 
            y2 = faces[0].bottom() 
        return Image.fromarray(image[y1:y2,x1:x2,:])
    else:
        return Image.open(image_path)

class recongnizer():
    def __init__(self,model,feature_base:pd.DataFrame,feature_names):
        """
        Parameters
        ----------
        model : facemodel define
            传入模型.
        feature_base : pd.DataFrame
            已经有记录的人脸数据.
        Returns
        -------
        None.

        """
        self.model = model
        self.base_feature = feature_base
        self.feature_names = feature_names
    def distance_l2(self,embedding_1,embedding_2):
        """
        Parameters
        ----------
        embedding_1 : TYPE
            DESCRIPTION.
        embedding_2 : TYPE
            DESCRIPTION.
        Returns
        -------
        out : TYPE
            l2 distance =={sum((x1-x2)^2)}.
        """
        out = np.linalg.norm(embedding_1-embedding_2, axis=1)
        return out
    
    def find_min(self,wait_recongnize_face_path:str):
        """
        Parameters
        ----------
        wait_recongnize_face_path : str
            path of wait recongnize.
        Returns
        -------
        None.
        """
        feature_embedding = self.model.get_embedding(Image.open(wait_recongnize_face_path)).reshape(-1)
        distance_distribute = self.distance_l2(self.base_feature[self.feature_names], feature_embedding)
        min_distance_index = np.argmin(distance_distribute)
        name = self.base_feature.iloc[min_distance_index]['name']
        print("识别结果为_: {}".format(name))
        return name
    

if __name__ == "__main__":
    
    model = Facenet()
    ###
    # 第一次需要重置现有人脸数据的embedding信息
    First_use = 0
    if First_use:
        name,embed,save = first_processing(model,'/home/item_spyder/Qian_ru/face_data')
    ###
    
    feature_names = ['f'+str(i) for i in range(128)]
    feature_base = pd.read_pickle('./embedding_save/person_feature.pkl')
    reconge_fun = recongnizer(model, feature_base,feature_names)
    # test
    reconge_fun.find_min('/home/item_spyder/Qian_ru/face_data/Abdullah_1.jpg')
    
    exit()
    
    debug(model,cut=1)