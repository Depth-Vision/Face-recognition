#用于收集数据集

import cv2 as cv

id=eval(input("请输入你的id："))#为了方便处理统一id名为4位数字1001

count=1
font=cv.FONT_HERSHEY_SIMPLEX
video = cv.VideoCapture(0)
face = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')#加载人脸检测模块
while True:
    ok ,image = video.read()
    #转化为灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #中值滤波去噪
    gray=cv.medianBlur(gray,5)
    #检测人脸的位置并返回位置列表
    faces = face.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5,minSize=(30, 30) )

    for (x, y, h, w) in faces:
        cv.imwrite('./facedata/'+str(id)+'.'+str(count)+'.jpg', gray[y:y+w,x:x+h])#保存读取到的脸部信息（直接保存灰度）
        count+=1
        cv.rectangle(image,(x,y),(x+h, y+w), (0, 0, 255), 2)
        cv.putText(image, str(count), (x + 5, y - 5), font, 1, (0, 0, 255), 1)
    if count==1001:
        break
    cv.imshow('img', image)
    if cv.waitKey(1) == 27:
        break

