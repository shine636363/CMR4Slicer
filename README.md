# 多病种情况下跨影像设备的心脏磁共振智能分割算法
将原始的MICCAI 2020公开的M&Ms Challenge心脏数据集，切片得出标记收缩舒张末期的时序的标签图片，分割区域为左心室、右心室和左心室心肌此数据集包含多中心，多供应商和多疾病类型的MRI心脏序列。
代码引入SS-Loss，改进trainer.py中的Loss函数来提高模型的泛化力，该案例代码基于TransUnet官方代码(https://github.com/Beckschen/TransUNet)

## 使用

### 1. 环境

在已安装3DSlicer的基础上，双击install.bat，根据提示选择版本进行安装。
![image](https://user-images.githubusercontent.com/12916146/231329587-341a84f9-d8be-4d96-8079-bea50c6a477c.png)

### 2. 使用
1.打开3DSlicer前, 将准备好的模型文件放入安装路径里的model文件夹内(默认路径如下：)
C:\Users\{username}\AppData\Local\NA-MIC\Slicer x.x.x\lib\Slicer-x.x\qt-scripted-modules\SegmentCalcDir\model

2.打开3DSlicer, 选择Add Data加载数据或者打开已有的DICOM数据库均可
![image](https://user-images.githubusercontent.com/12916146/231329807-fbc966c0-c6ad-48c4-8947-a8e59be66b7f.png)

3. 在Data页面中，右键选中需要测试的数据项，点击"segCMR by {xxx.pth}"({xxx.pth}为model文件夹中的模型文件名)
![image](https://user-images.githubusercontent.com/12916146/231329657-558c5c92-0a90-4685-8eac-93f14e740f2d.png)

4. 预测结果会自动加载为分割模块
![image](https://user-images.githubusercontent.com/12916146/231329923-d8bf7d74-4895-4e80-ba4f-56bfbad8f433.png)

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [TransUnet](https://github.com/Beckschen/TransUNet)
