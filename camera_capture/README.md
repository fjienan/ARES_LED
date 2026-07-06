# 摄像头图片采集

该工具不依赖 ROS 2。运行时实时显示摄像头画面，并默认每 3 秒保存一张未经
绘制的原始图片，供后续手工标注。

```bash
cd ~/Desktop/LED/camera_capture
/usr/bin/python3 capture_images.py
```

图片默认保存在当前目录的 `images/` 中。按 `q`、`Esc` 或 `Ctrl+C` 停止。
自动选择会硬性排除设备名称中标有 `Integrated`、`Internal` 或 `Built-in` 的
笔记本集成摄像头。

常用参数：

```bash
/usr/bin/python3 capture_images.py \
  --device auto \
  --output ~/Desktop/LED/training_images \
  --interval 3 \
  --width 1280 --height 720 --fps 30
```

如果自动选择了错误的摄像头，使用 `v4l2-ctl --list-devices` 找到设备后，通过
`--device /dev/videoX` 明确指定。不要同时运行占用同一摄像头的 R2 接收节点。
