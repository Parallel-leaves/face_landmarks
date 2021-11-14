# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 21:00:09 2021

@author: hp
"""

from cv2 import cv2 as cv2
import os
import dlib
from skimage import io
import csv
import numpy as np
 
# 要读取人脸图像文件的路径
path_images_from_camera = "D:/myworkspace/JupyterNotebook/People/person/"

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib 人脸预测器
predictor = dlib.shape_predictor("D:/shape_predictor_68_face_landmarks.dat")

# Dlib 人脸识别模型
# Face recognition model, the object maps human faces into 128D vectors
face_rec = dlib.face_recognition_model_v1("D:/dlib_face_recognition_resnet_model_v1.dat")

# 返回单张图像的 128D 特征
def return_128d_features(path_img,i):
    img_rd = io.imread(path_img)
    s=path_img
    a=s[16:17]
    i1=str(a)
    a1=s[17:]
    str1="/"
    b=a1[a1.index(str1):-4]
    b1=b[0:]
    i2=str(i)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)
    for i in range(len(faces)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd,faces[i]).parts()])  
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            add="D:/myworkspace/JupyterNotebook/People/feature2/face_feature"+i2+".csv"
            with open(add, "a", newline="") as csvfile:
                writer1 = csv.writer(csvfile)
                writer1.writerow((idx,pos))
        print(add)
    print("%-40s %-20s" % ("检测到人脸的图像 / image with faces detected:", path_img), '\n')

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了
    # 所以要确保是 检测到人脸的人脸图像 拿去算特征
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = face_rec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")

    return face_descriptor


# 将文件夹中照片特征提取出来, 写入 CSV
def return_features_mean_personX(path_faces_personX):
    features_list_personX = []
    photos_list = os.listdir(path_faces_personX)
    if photos_list:
        for i in range(len(photos_list)):
            # 调用return_128d_features()得到128d特征
            print("%-40s %-20s" % ("正在读的人脸图像 / image to read:", path_faces_personX + "/" + photos_list[i]))
            features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i],i)
            #  print(features_128d)
            # 遇到没有检测出人脸的图片跳过
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)

    else:
        print("文件夹内图像文件为空 / Warning: No images in " + path_faces_personX + '/', '\n')

    # 计算 128D 特征的均值
    # N x 128D -> 1 x 128D
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean_personX = '0'

    return features_mean_personX


# 读取某人所有的人脸图像的数据
people = os.listdir(path_images_from_camera)
people.sort()
with open("D:/myworkspace/JupyterNotebook/People/feature/features2_all.csv", "w", newline="") as csvfile: #程序会新建一个表格文件来保存特征值，方便以后比对
    writer = csv.writer(csvfile)
    for person in people:
        print("##### " + person + " #####")
        # Get the mean/average features of face/personX, it will be a list with a length of 128D
        features_mean_personX = return_features_mean_personX(path_images_from_camera + person)
        writer.writerow(features_mean_personX)
        print("特征均值 / The mean of features:", list(features_mean_personX))
        print('\n')
    print("所有录入人脸数据存入 / Save all the features of faces registered into: D:/myworkspace/JupyterNotebook/People/feature/features_all2.csv")

