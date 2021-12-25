# Face-recognition

![image](https://github.com/Depth-Vision/Face-recognition/blob/main/Show.gif)

### 基于opencv的人脸识别

#### 项目要求

使用opencv和数字图像处理技术实现人脸信息识别，不使用深度学习框架和机器学习框架

#### 环境配置

```
conda install opencv==3.4.1.15
conda install opencv-contrib-python==3.4.3.18
conda install Pillow==4.2.1
```

#### 数据采集

使用opencv自己的分类器，将人脸检测出来，直接在文件夹中写入检测到的人脸灰度图片，为方便处理，我们给每个人的照片进行4位编号，方便在读取图片时进行处理。

#### 模型建立与训练

在该项目中，我们采用LBPH模型（局部二值模式直方图），该模型在opencv中可由 : `cv2.face.LBPHFaceRecognizer_create()` 接口生成LBPH识别器模型,然后使用 `.train()` 方法进行训练。

#### 测试

最后在测试文件中使用.predict()方法进行预测；该方法会返回一个id和一个置信度评分，在显示图象时可以将id打印即可，但是opencv无法显示中文，需要使用pillow显示。

#### 注

这个人脸识别项目最终的识别准确度与前期录入人脸数据时的摄像头的清晰度和识别所用的摄像头清晰度有关。


# 文件说明

#### face_collection.py 用于人脸图像数据收集

#### model_train.py 用于模型训练

#### test.py 用于人脸识别功能测试
