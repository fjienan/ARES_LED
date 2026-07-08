# USB 摄像头图片采集

本目录只保留普通 USB RGB 摄像头采集脚本：

- `capture_usb_rgb.py`：自动寻找外接 USB 摄像头，实时显示画面，并按固定间隔保存 JPG 原图。
- `detect_usb_rgb.py`：自动寻找外接 USB 摄像头，运行当前 R2 单色检测算法，实时显示候选框，
  并在检测到阳性时保存原图。
- `detect_usb_rgb_2_three_segment_live.py`：自动寻找外接 USB 摄像头，运行 `usb_rgb_2`
  三段四色灯带检测算法，实时显示候选框，并在检测到三段阳性时保存原图。

默认每 2 秒保存一张未经绘制的 JPG。按 `q`、`Esc` 或 `Ctrl+C` 停止。

当前采集阶段假定同一时间最多只接一台外接 USB 摄像头。脚本会排除常见笔记本内置摄像头；
如果发现多台非内置摄像头，会报错并列出候选设备，避免静默选错。

## 第一台 USB 摄像头

```bash
cd <仓库根目录>/camera_capture
/usr/bin/python3 ./capture_usb_rgb.py --camera 1
```

默认保存到：

```text
<仓库根目录>/camera_data/usb_rgb_1/raw
```

等价的显式写法：

```bash
/usr/bin/python3 ./capture_usb_rgb.py \
  --camera 1 \
  --interval 2 \
  --width 1280 --height 720 --fps 30
```

## 第二台 USB 摄像头

第二台摄像头也用同一个脚本，但输出到独立数据目录：

```bash
/usr/bin/python3 ./capture_usb_rgb.py \
  --camera 2 \
  --interval 2 \
  --width 1280 --height 720 --fps 30
```

`--camera 1` 和 `--camera 2` 只决定保存到哪组训练数据目录，不代表 `/dev/video1`
或 `/dev/video2`。实际摄像头默认由脚本自动选择。

如果自动选择失败，先用下面命令查看设备：

```bash
v4l2-ctl --list-devices
```

然后临时指定设备：

```bash
/usr/bin/python3 ./capture_usb_rgb.py --camera 1 --device /dev/videoX
```

## 实时单色检测并保存阳性图片

该脚本不启动 ROS，也不依赖两段通信协议，只使用当前对应摄像头 profile 的
`detector.yaml` 和 classifier。

摄像头 1：

```bash
cd <仓库根目录>/camera_capture
/usr/bin/python3 ./detect_usb_rgb.py --camera 1
```

摄像头 2：

```bash
cd <仓库根目录>/camera_capture
/usr/bin/python3 ./detect_usb_rgb.py --camera 2
```

默认保存目录：

```text
<仓库根目录>/camera_capture_positive_usb_rgb_1
<仓库根目录>/camera_capture_positive_usb_rgb_2
```

连续阳性时默认最多每 1 秒保存一张。需要改间隔：

```bash
/usr/bin/python3 ./detect_usb_rgb.py --camera 2 --interval 0.5
```

如果只想看检测效果，不保存图片：

```bash
/usr/bin/python3 ./detect_usb_rgb.py --camera 2 --no-save
```

## 实时三段四色灯带检测并保存阳性图片

该脚本不启动 ROS，也不依赖 shared 协议包。它只针对 `usb_rgb_2`，先识别单段灯带，
再组合成“三段、基本共线、相邻不同色”的候选。

```bash
cd <仓库根目录>/camera_capture
/usr/bin/python3 ./detect_usb_rgb_2_three_segment_live.py
```

默认保存目录：

```text
<仓库根目录>/camera_capture_positive_usb_rgb_2_combined
```

连续识别到同一三段编码时默认最多每 1 秒保存一张；如果三段编码变化，会立即保存：

```bash
/usr/bin/python3 ./detect_usb_rgb_2_three_segment_live.py --interval 0.5
```

只显示不保存：

```bash
/usr/bin/python3 ./detect_usb_rgb_2_three_segment_live.py --no-save
```
