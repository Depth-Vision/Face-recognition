# Face-recognition
test
# Face-recognition

### 基于opencv的人脸识别

###### 需求分析

​		使用opencv和数字图像处理技术实现人脸信息识别，不得使用深度学习框架和机器学习框架

###### 开发环境

```
#python=3.6 opencv=3.4.1.15  opencv-contrib-python=3.4.3.18 pillow=4.2.1
#开发工具 pycharm2020.1.5 Anaconda3

```

###### 环境配置

```
conda install opencv==3.4.1.15
conda install opencv-contrib-python==3.4.3.18
conda install Pillow==4.2.1
```

###### 数据采集

​		使用opencv自己的分类器，将人脸检测出来，直接在文件夹中写入检测到的灰度人脸图片，注意每一张图片的文件名，为方便处理，我们给每个人的照片进行4位编号，方便在读取图片时进行处理。

###### 模型建立与训练

​	在该项目中，我们采用LBPH模型（局部二值模式直方图），该模型在opencv中可由

cv2.face.LBPHFaceRecognizer_create()接口生成LBPH识别器模型,然后使用.train()方法进行训练，在传入训练数据之前，一定要将数据灰度化。

###### 测试

​	最后在测试文件中使用.predict()方法进行预测；该方法会返回一个id和一个置信度评分，在显示图象时可以将id打印即可，但是opencv无法显示中文，需要使用pillow显示。

```python
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):

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

```

###### 总结

​	该方法虽然成功的识别出录入的人脸信息，但是由于数据集制作的不好，对识别精度有很大的影响，而且由于数据集较大，导致在测试时会出现卡顿等现象。



**文件说明**

***face_collection.py***

用于人脸图像数据收集

***model_train.py***

用于模型训练

***test.py***

用于人脸识别功能测试
