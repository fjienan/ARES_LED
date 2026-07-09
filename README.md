# R1/R2 LED 光学通信

本仓库用 LED 灯带替代原 ArUco 屏幕通信。R1 负责显示三段颜色命令；R2 负责识别
RED/GREEN/BLUE/PURPLE 单色灯带候选，并将三段候选按 shared 协议解码为命令 ID。

## 目录

```text
shared/src/   公共 RGB 协议
r1_ws/src/    R1 编码发送节点；默认通过 USB 串口发送 WLED JSON
r2_ws/src/    R2 摄像头实时识别和离线数据集评测
camera_data/  R2 训练图片；按摄像头 profile 隔离
```

`shared/src/rgb_comm_protocol/config/rgb_protocol.yaml` 只保存命令与符号序列。
R1 的物理 RGB 值位于 `r1_ws/src/rgb_led_sender/config/colors.yaml`；R2 的相机观测
颜色模型位于 `r2_ws/src/rgb_camera_receiver/config/cameras/<profile>/detector.yaml`，
两者不得共享数值。

当前 profile：

| profile | 状态 | 数据目录 | detector |
|---|---|---|---|
| `usb_rgb_1` | 第一台 USB 摄像头 | `camera_data/usb_rgb_1` | 已有 |
| `usb_rgb_2` | 第二台 USB 摄像头 | `camera_data/usb_rgb_2` | 待采集、待标定 |

当前 R1 发送节点订阅 `/aruco_comm/tx_id`，通过 WLED 控制器的 USB CDC 串口直接发送
JSON 状态帧。

普通克隆即可获得构建所需的全部源码：

```bash
git clone <本仓库地址>
```

## R1/R2 三段四色通信协议

R1 使用 6 个物理段显示同一个三段协议码：`0,1,2` 为低亮度组，`3,4,5` 为高亮度组。
每段只使用 `RED/GREEN/BLUE/PURPLE` 四种颜色；同一命令的三段颜色必须全部不同。
协议只认正向顺序，不做正反等价。

| 符号 | 颜色 | RGB |
|---|---|---:|
| RED | 红 | R1 本地配置 |
| GREEN | 绿 | R1 本地配置 |
| BLUE | 蓝 | R1 本地配置 |
| PURPLE | 紫 | R1 本地配置 |

| 命令 ID | 三段颜色 |
|---:|:---:|
| 0 | BLUE, PURPLE, RED |
| 1 | BLUE, RED, GREEN |
| 2 | BLUE, GREEN, PURPLE |
| 3 | RED, GREEN, BLUE |
| 4 | PURPLE, BLUE, GREEN |
| 5 | RED, BLUE, PURPLE |
| 6 | GREEN, PURPLE, BLUE |
| 7 | GREEN, RED, PURPLE |
| 8 | RED, PURPLE, GREEN |

命令 `0` 是内部重置命令。R2 确认 `0` 后只清除去重状态，不向 `/aruco_comm/rx_id`
发布；它用于让相邻两个相同动作 ID 可以再次触发。

## R1 构建与启动

环境为 Ubuntu 22.04 + ROS 2 Humble。Conda 会污染 ROS Python，构建前清理：

```bash
cd ~/Desktop/LED/r1_ws
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH
source /opt/ros/humble/setup.bash
export PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
colcon build --base-paths src ../shared/src \
  --packages-select rgb_comm_protocol rgb_led_sender \
  --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ros2 launch rgb_led_sender r1_rgb_comm.launch.py
```

部署到 R1 远端主机时，当前路径为 `librgod@100.123.45.4:~/ARES_LED`：

```bash
ssh librgod@100.123.45.4
cd ~/ARES_LED/r1_ws
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH
source /opt/ros/humble/setup.bash
export PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
colcon build --base-paths src ../shared/src \
  --packages-select rgb_comm_protocol rgb_led_sender \
  --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ros2 launch rgb_led_sender r1_rgb_comm.launch.py
```

启动前确认 WLED 控制器 USB 已连接并枚举为 CDC 串口：

```bash
ls -l /dev/serial/by-id/* /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

应能看到类似：

```text
/dev/serial/by-id/usb-WEMOS.CC_LOLIN-S2-MINI_0-if00 -> ../../ttyACM0
```

注意不要选到 DAPLink 烧录器串口；`auto` 会尽量优先选择 WLED/LOLIN/WEMOS 设备：

```text
usb-ARM_DAPLink_CMSIS-DAP_... -> ../../ttyACM0
```

默认 `r1_ws/src/rgb_led_sender/config/sender.yaml` 使用：

```yaml
transport: serial
serial_device: auto
pixel_count: 6
initial_command_id: -1
idle_effect_enabled: true
idle_effect_color: [220, 0, 120]
```

如自动选择错误，可把 `serial_device` 改为上面的
`/dev/serial/by-id/usb-WEMOS.CC_LOLIN-S2-MINI_0-if00` 固定路径。

WLED 灯带连接：24V 电源正极接灯带 `+`，电源负极接灯带 `GND`，WLED `GND` 与灯带
`GND` 共地，WLED `DO` 接灯带 `DIN`。不要把 24V 接到 WLED 板的 5V。
默认总段数为 6。启动后、收到第一条真实命令前，R1 会发送 WLED 内置跑马灯效果；
当前 idle 颜色 `[220, 0, 120]` 在这条灯带上视觉接近橙黄色，不属于 R2 协议色。
收到第一条合法命令后，跑马灯立即停止，6 个物理段会一直保持对应协议颜色，直到收到下一条命令。

六段显示顺序在 `r1_ws/src/rgb_led_sender/config/sender.yaml` 中配置：

```yaml
low_segments: [0, 1, 2]
low_brightness: 6.0
low_reverse_order: false
high_segments: [3, 4, 5]
high_brightness: 60.0
high_reverse_order: false
```

如某一组物理接线方向相反，只改对应的 `*_reverse_order: true`。

发送测试命令：

```bash
ros2 topic pub --once /aruco_comm/tx_id std_msgs/msg/Int32 '{data: 1}'
```

发送另一条命令：

```bash
ros2 topic pub --once /aruco_comm/tx_id std_msgs/msg/Int32 '{data: 8}'
```

也可以绕过 ROS，直接测试 WLED 串口。建议在同一个终端保持串口长连接：

```bash
PORT=/dev/serial/by-id/usb-WEMOS.CC_LOLIN-S2-MINI_0-if00
stty -F "$PORT" 115200 raw -echo -hupcl
exec 3<>"$PORT"
sleep 1
```

整条测试红色：

```bash
printf '{"on":true,"bri":40,"seg":[{"id":0,"start":0,"stop":6,"col":[[255,0,0]],"fx":0}]}\n' >&3
```

关闭：

```bash
printf '{"on":false}\n' >&3
```

## R2 构建与启动

```bash
cd ~/Desktop/LED/r2_ws
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH
source /opt/ros/humble/setup.bash
export PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
colcon build --base-paths src ../shared/src \
  --packages-select rgb_comm_protocol rgb_camera_receiver \
  --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ros2 launch rgb_camera_receiver r2_dual_led_vision.launch.py
```

`colcon build` 会把 launch 和 config 安装到 `install/`。修改 `config/`、`launch/`
或 Python 节点后，需要重新构建并重新 `source install/setup.bash` 再启动。

双摄像头配置在 `r2_ws/src/rgb_camera_receiver/config/dual_receiver.yaml`。双摄像头
launch 会先读取 `config/cameras/<profile>/receiver.yaml`，再用 `dual_receiver.yaml`
中对应 slot 的字段覆盖；因此即使节点名变成 `rgb_camera_receiver_camera_1`，profile
默认参数也会生效。
不要在双摄像头配置中使用 `auto`；先查清楚摄像头枚举结果和稳定设备路径：

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id/
ls -l /dev/v4l/by-path/
```

再逐个测试哪个 `/dev/videoX` 真能出图。无预览测试：

```bash
/usr/bin/python3 ~/Desktop/LED/camera_capture/capture_usb_rgb.py \
  --camera 1 \
  --device /dev/video4 \
  --no-preview
```

实时预览测试：

```bash
/usr/bin/python3 ~/Desktop/LED/camera_capture/capture_usb_rgb.py \
  --camera 1 \
  --device /dev/video4
```

`/dev/video4` 只用于临时验证，最终配置不要写这种可能变化的名字。确认哪一路能出图后，
把对应的稳定路径填入 `dual_receiver.yaml`，优先使用 `/dev/v4l/by-id/...video-index0`；
如果两台摄像头型号相同或 by-id 不够区分，使用和 USB 口绑定的 `/dev/v4l/by-path/...`：

```yaml
camera_slots:
  camera_1:
    enabled: true
    profile: usb_rgb_1
    device: /dev/v4l/by-id/usb-xxx-camera-A-video-index0
    required: false
    frame_width: 2560
    frame_height: 1440
    camera_fps: 30.0
    scan_rate_hz: 30.0
    camera_fourcc: MJPG
    camera_buffer_size: 1
    show_preview: true
    preview_scale: 0.45
    processing_scale: 0.0

  camera_2:
    enabled: true
    profile: usb_rgb_2
    device: /dev/v4l/by-id/usb-yyy-camera-B-video-index0
    required: false
    camera_fps: 30.0
    scan_rate_hz: 20.0
    show_preview: true
```

`required: false` 表示只连接一台摄像头也允许启动测试。
启动后可以确认第一台相机的关键参数是否和
`camera_capture/detect_usb_rgb_1_three_segment_live.py` 一致：

```bash
ros2 param get /rgb_camera_receiver_camera_1 frame_width
ros2 param get /rgb_camera_receiver_camera_1 frame_height
ros2 param get /rgb_camera_receiver_camera_1 camera_fps
ros2 param get /rgb_camera_receiver_camera_1 scan_rate_hz
ros2 param get /rgb_camera_receiver_camera_1 camera_fourcc
ros2 param get /rgb_camera_receiver_camera_1 camera_buffer_size
ros2 param get /rgb_camera_receiver_camera_1 preview_scale
```

期望值分别为 `2560`、`1440`、`30.0`、`30.0`、`MJPG`、`1`、`0.45`。

如需单独调试某一个 profile：

```bash
ros2 launch rgb_camera_receiver r2_led_vision.launch.py camera_profile:=usb_rgb_1
```

第二台 USB 摄像头使用独立 profile：

```bash
ros2 launch rgb_camera_receiver r2_led_vision.launch.py camera_profile:=usb_rgb_2
```

每台相机的运行参数在 `config/cameras/<profile>/receiver.yaml`，R2 专用颜色及几何模型在
`config/cameras/<profile>/detector.yaml`；这些参数与 R1 输出 RGB 完全独立。
R2 按“处理帧”确认命令，当前默认连续 2 次处理结果一致才确认；
`usb_rgb_1` 最长确认窗口 0.30 秒，`usb_rgb_2` 为 0.25 秒。

离线处理全部标注数据并生成逐图结果：

```bash
ros2 run rgb_camera_receiver evaluate_led_dataset \
  --camera-profile usb_rgb_1
```

结果目录包含每张图片的全部候选框、颜色、置信度、排名和分差，以及 `results.csv`、
`results.json`。退出码非零表示至少一张图片未满足验收条件。

USB 摄像头重新标定：

```bash
ros2 run rgb_camera_receiver calibrate_led_colors --camera-profile usb_rgb_1
```

`camera_data/<profile>/unused/YELLOW` 中的黄色样本当前不参与默认标定和评估。

## 测试

```bash
cd ~/Desktop/LED
source /opt/ros/humble/setup.bash
/usr/bin/python3 -m pytest \
  shared/src/rgb_comm_protocol/test \
  r1_ws/src/rgb_led_sender/test \
  r2_ws/src/rgb_camera_receiver/test
```

当前硬性回归集为单色图像数据集：有效颜色必须全部识别正确，NONE 必须全部零候选。
R2 另有三段协议解码单元测试。
