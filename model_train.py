import os
import cv2 as cv
from PIL import Image
import numpy as np

train = cv.face.LBPHFaceRecognizer_create()#建立训练模型
path = "facedata"
all_image_path = [os.path.join(path, i) for i in os.listdir(path)]#读取收集到的人脸图片
face = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
names = []#创建列表用于储存id
face_data = []#创建列表储存人脸数据
for each_img in all_image_path:
    id = int(os.path.split(each_img)[1][0:4])
    PIL_img = Image.open(each_img).convert('L')#将读取到的人脸数据转化为灰度数据
    np_img = np.array(PIL_img, np.uint8)#将人脸数据转化为uint8格式
    #检测人脸
    faces = face.detectMultiScale(np_img)
    for (x, y, h, w) in faces:
        face_data.append(np_img[y:y+w, x:x+h])
        names.append(id)

train.train(face_data, np.array(names))#开始训练
train.write("facedata2.xml")#将训练好的数据用xml文件保存起来

