# 案例6 心脏超声分割
代码辅助用于《医学影像深度学习》（清华大学出版社）案例6，该案例代码基于TransUnet官方代码(https://github.com/Beckschen/TransUNet)

## 使用

### 1. 环境

在已安装3DSlicer的基础上，双击install.bat，根据提示选择版本进行安装。
![image](https://user-images.githubusercontent.com/12916146/231329587-341a84f9-d8be-4d96-8079-bea50c6a477c.png)

### 2. 使用
1. 选择Add Data加载数据或者打开已有的DICOM数据库均可
![image](https://user-images.githubusercontent.com/12916146/231329807-fbc966c0-c6ad-48c4-8947-a8e59be66b7f.png)

2. 在Data页面中，右键选中需要测试的数据项，点击"SegmentCalc this..."
![image](https://user-images.githubusercontent.com/12916146/231329657-558c5c92-0a90-4685-8eac-93f14e740f2d.png)

3. 预测结果会自动加载为分割模块
![image](https://user-images.githubusercontent.com/12916146/231329923-d8bf7d74-4895-4e80-ba4f-56bfbad8f433.png)

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [TransUnet](https://github.com/Beckschen/TransUNet)
