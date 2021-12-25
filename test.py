import cv2 as cv
from PIL import ImageDraw,ImageFont,Image
import numpy as np

"""def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):

    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本

    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
"""
video = cv.VideoCapture(0)
font = cv.FONT_HERSHEY_SIMPLEX
train = cv.face.LBPHFaceRecognizer_create() # 加载模型
face = cv.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
train.read("./facedata2.xml")
dicts = {1101:"wang",1102:"liu",1103:"xiao",1104:"chen",1105:"qi",1106:"yuan"}


while True:
    ok ,image = video.read()
    #转化为灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #检测人脸的位置并返回位置列表
    faces = face.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4,minSize=(30, 30) )
    for (x, y, h, w) in faces:
        cv.rectangle(image,(x,y),(x+h, y+w), (0, 0, 255), 2)
        #开始识别
        id,con=train.predict(gray[y:y+w,x:x+h])
        if int(con)>85:
            cv.putText(image,str(dicts[id]),(x+5,y-25),font,1,(0,0,255),1)
            #image = cv2AddChineseText(image," ",(x+5,y-25),(0,0,255),1)
            cv.putText(image,str(con),(x+5,y-5),font,1,(0,0,255),1)
        else:
            cv.putText(image,"son",(x+5,y-25),font,1,(0,0,255),1)
    cv.imshow('img', image)
    if cv.waitKey(1) == 27:
        break



