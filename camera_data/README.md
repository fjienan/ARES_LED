# R2 相机训练数据

每个摄像头独立一个目录。不同摄像头的曝光、白平衡、镜头和驱动都会改变颜色观测，
因此不能混用训练图片或 detector 参数。

```text
camera_data/
  usb_rgb_1/ 第一台 USB RGB 摄像头数据
  usb_rgb_2/ 第二台 USB RGB 摄像头数据
```

默认参与标定和评估的类别目录为：

```text
RED/
GREEN/
BLUE/
CYAN/
PURPLE/
NONE/
```

暂时采集但不参与训练的图片放在 `unused/` 下，例如 `unused/YELLOW/`。
