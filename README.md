# License-Plate-Detection-and-Recognition
### Pipeline：
  使用颜色分割和轮廓检测进行车牌定位->利用积分法进行单个字符的分割->利用CNN进行车牌字符识别
  
### Guide：
* Main:主程序，进行车牌定位与识别
* Train_number:训练数字、字母识别模型
* Train_province:训练省份代号识别模型
* dataload：自定义Pytorch的Dataloader类
* ycut、xcut：积分分割
* modeling：CNN模型建立
* License_plate:车牌定位与分割

