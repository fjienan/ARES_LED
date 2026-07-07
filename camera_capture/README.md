# USB 摄像头图片采集

本目录只保留普通 USB RGB 摄像头采集脚本：

- `capture_usb_rgb.py`：直接读取 `/dev/video*`，实时显示画面，并按固定间隔保存 JPG 原图。

默认每 2 秒保存一张未经绘制的 JPG。按 `q`、`Esc` 或 `Ctrl+C` 停止。

## 第一台 USB 摄像头

```bash
cd <仓库根目录>/camera_capture
/usr/bin/python3 ./capture_usb_rgb.py
```

默认保存到：

```text
<仓库根目录>/camera_data/usb_rgb_1/raw
```

等价的显式写法：

```bash
/usr/bin/python3 ./capture_usb_rgb.py \
  --device auto \
  --output ../camera_data/usb_rgb_1/raw \
  --prefix usb_rgb_1 \
  --interval 2 \
  --width 1280 --height 720 --fps 30
```

## 第二台 USB 摄像头

第二台摄像头也用同一个脚本，但输出到独立数据目录：

```bash
/usr/bin/python3 ./capture_usb_rgb.py \
  --device /dev/videoX \
  --output ../camera_data/usb_rgb_2/raw \
  --prefix usb_rgb_2 \
  --interval 2 \
  --width 1280 --height 720 --fps 30
```

自动选择会排除设备名称中含有 `Integrated`、`Internal` 或 `Built-in` 的笔记本集成摄像头。
如果自动选择错误，先用 `v4l2-ctl --list-devices` 查看，再用 `--device /dev/videoX` 指定。
