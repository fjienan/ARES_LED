# led_controller

这是一个 ROS 2 Jazzy 节点，用 FT232H USB 转 SPI 控制 WS2811 LED 灯带。当前版本不使用预设灯效，只提供一个 action：按组控制颜色、亮度、闪烁频率和持续时间。

## 1. 先改路径

下面所有命令都用 `REPO_DIR` 表示仓库路径。别人拿到仓库后，只需要把这一行改成自己电脑上的实际路径：

```bash
export REPO_DIR=/home/gsp/led_controller
```

如果仓库放在 `~/ros2_ws/src/led_controller`，就改成：

```bash
export REPO_DIR=~/ros2_ws/src/led_controller
```

后面的命令都从这个路径进入：

```bash
cd $REPO_DIR
```

README 里的 `/home/gsp/led_controller` 只是示例路径。给别人用时，先让对方把所有 `export REPO_DIR=...` 改成自己的仓库路径。

## 2. 需要安装什么

系统需要 Ubuntu + ROS 2 Jazzy，并且已经安装好 `colcon`。如果没有，可以先装：

```bash
sudo apt update
sudo apt install python3-colcon-common-extensions python3-pip
```

这个包本身用到的 ROS 依赖在 `package.xml` 里声明了，主要是：

- `rclpy`
- `action_msgs`
- `std_msgs`
- `std_srvs`
- `builtin_interfaces`
- `rosidl_default_generators`
- `rosidl_default_runtime`

如果你的 ROS Jazzy 是完整安装，一般已经有这些依赖。

也可以在仓库目录里用 `rosdep` 检查并安装 ROS 依赖：

```bash
export REPO_DIR=/home/gsp/led_controller
cd $REPO_DIR
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths . --ignore-src -r -y
```

硬件模式还需要 FT232H/NeoPixel Python 库：

```bash
/usr/bin/python3 -m pip install --user adafruit-blinka adafruit-circuitpython-neopixel-spi
```

如果系统提示 Python 环境受保护，用：

```bash
/usr/bin/python3 -m pip install --user --break-system-packages adafruit-blinka adafruit-circuitpython-neopixel-spi
```

检查库是否可用：

```bash
BLINKA_FT232H=1 /usr/bin/python3 -c "import board; import neopixel_spi; print('ok')"
```

如果这里报 USB 权限错误，可以临时用 `sudo` 验证是不是权限问题：

```bash
BLINKA_FT232H=1 sudo -E /usr/bin/python3 -c "import board; import neopixel_spi; print('ok')"
```

长期使用建议添加 FT232H udev 规则。新建文件：

```bash
sudo nano /etc/udev/rules.d/99-ft232h.rules
```

写入：

```text
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6014", MODE="0666"
```

然后重新加载规则并重新插拔 FT232H：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## 3. 需要改哪些配置

配置文件在：

```text
config/led_controller.yaml
```

当前默认配置：

```yaml
led_action_server:
  ros__parameters:
    led_count: 60
    active_group_count: 4
    backend: "hardware"
    pixel_order: "BRG"
    default_brightness: 0.4
    action_name: "led_pattern"
    cancel_service_name: "cancel_led_pattern"
    frame_rate: 40.0
```

一般只需要改这几个：

- `led_count`: 可寻址 LED 段总数。你的灯带每个可寻址段本身包含 6 个相邻同色灯珠；长灯带接入时应写整条灯带的可寻址段数量，例如 `60`。
- `active_group_count`: 允许 action 控制的通信段数量。当前只用前 4 个可寻址段，所以是 `4`；后面的可寻址段每帧保持熄灭。
- `backend`: 接硬件时用 `"hardware"`；不接灯带、只测试 action 通信时用 `"simulate"`。
- `pixel_order`: 颜色顺序。当前灯带是 `"BRG"`。如果红绿蓝显示不对，优先改这里。
- `default_brightness`: 默认亮度，范围 `0.0-1.0`。

## 4. 构建

如果当前终端启用了 conda，先退出 conda，并清掉 conda 留下的动态库路径：

```bash
conda deactivate
unset LD_LIBRARY_PATH
```

然后构建：

```bash
export REPO_DIR=/home/gsp/led_controller
cd $REPO_DIR
source /opt/ros/jazzy/setup.bash
colcon build --packages-select led_controller --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

如果改过 action 文件，建议干净重建：

```bash
cd $REPO_DIR
source /opt/ros/jazzy/setup.bash
rm -rf build install log
colcon build --packages-select led_controller --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

## 5. 启动节点

硬件模式启动：

```bash
export REPO_DIR=/home/gsp/led_controller
cd $REPO_DIR
conda deactivate
unset LD_LIBRARY_PATH
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export BLINKA_FT232H=1
ros2 launch led_controller led_controller.launch.py
```

看到类似日志说明启动成功：

```text
LED group controller ready: action=/led_pattern, cancel_service=/cancel_led_pattern, backend=hardware, leds=60, active_groups=4
```

模拟模式启动前，先把 `config/led_controller.yaml` 里的 `backend` 改成 `"simulate"`，然后：

```bash
cd $REPO_DIR
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch led_controller led_controller.launch.py
```

## 6. 硬件接线

- FT232H `SCK` 接 WS2811 灯带 `DIN`
- `DIN` 前串一个 330-500 欧姆电阻
- FT232H `GND`、灯带电源 `GND`、外部电源 `GND` 必须共地
- 灯带电源根据你的灯带选择 5V 或 12V
- 不要直接从 FT232H 给整条灯带供电

如果 Linux 能看到 FT232H：

```bash
lsusb
```

正常会看到类似：

```text
0403:6014 Future Technology Devices International, Ltd FT232H
```

## 7. Action 接口

action 名称：

```text
/led_pattern
```

action 类型：

```text
led_controller/action/LedPattern
```

Goal 字段：

```text
int32[] groups
int32[] colors
float32[] brightnesses
float32[] blink_hz
float32 duration
```

字段含义：

- `groups`: 要控制的组号。`groups: [0]` 是第 1 组，`groups: [1]` 是第 2 组。空数组 `[]` 表示全部组。
- `colors`: RGB 数组。每 3 个数是一组颜色，按 `groups` 顺序对应。
- `brightnesses`: 亮度数组，范围 `0.0-1.0`。给 1 个值就是所有组共用；给多个值就是逐组对应。
- `blink_hz`: 闪烁频率数组。给 1 个值就是所有组共用；给多个值就是逐组对应。`0.0` 表示常亮。
- `duration`: 持续时间。`duration > 0` 到时间自动结束并关灯；`duration = 0.0` 一直运行，直到取消。

## 8. 发送命令

另开一个终端，先加载环境：

```bash
export REPO_DIR=/home/gsp/led_controller
cd $REPO_DIR
conda deactivate
unset LD_LIBRARY_PATH
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

第 0 组红色常亮 5 秒：

```bash
ros2 action send_goal /led_pattern led_controller/action/LedPattern \
"{groups: [0], colors: [255, 0, 0], brightnesses: [0.5], blink_hz: [0.0], duration: 5.0}" \
--feedback
```

第 0、1、2 组全部绿色，2Hz 闪烁，一直运行：

```bash
ros2 action send_goal /led_pattern led_controller/action/LedPattern \
"{groups: [0, 1, 2], colors: [0, 255, 0], brightnesses: [0.5], blink_hz: [2.0], duration: 0.0}" \
--feedback
```

第 0、1、2 组分别红、绿、蓝，亮度不同，常亮：

```bash
ros2 action send_goal /led_pattern led_controller/action/LedPattern \
"{groups: [0, 1, 2], colors: [255, 0, 0, 0, 255, 0, 0, 0, 255], brightnesses: [0.3, 0.5, 0.8], blink_hz: [0.0], duration: 0.0}" \
--feedback
```

第 0、1、2 组分别红、绿、蓝，频率不同：

```bash
ros2 action send_goal /led_pattern led_controller/action/LedPattern \
"{groups: [0, 1, 2], colors: [255, 0, 0, 0, 255, 0, 0, 0, 255], brightnesses: [0.5, 0.5, 0.5], blink_hz: [0.0, 1.0, 3.0], duration: 0.0}" \
--feedback
```

控制全部组，白色低亮度常亮 3 秒：

```bash
ros2 action send_goal /led_pattern led_controller/action/LedPattern \
"{groups: [], colors: [255, 255, 255], brightnesses: [0.2], blink_hz: [0.0], duration: 3.0}" \
--feedback
```

取消当前持续运行的命令并关灯：

```bash
ros2 service call /cancel_led_pattern std_srvs/srv/Trigger {}
```

## 9. 常见问题

如果颜色不对，比如红色变成绿色或蓝色，改：

```text
config/led_controller.yaml
```

里的：

```yaml
pixel_order: "BRG"
```

如果不接硬件只想测试 action，把：

```yaml
backend: "hardware"
```

改成：

```yaml
backend: "simulate"
```

如果 ROS launch 提示不能写入 `~/.ros/log`，可以临时把日志目录放到 `/tmp`：

```bash
mkdir -p /tmp/ros_logs
ROS_LOG_DIR=/tmp/ros_logs ros2 launch led_controller led_controller.launch.py
```

如果使用 conda 后构建出现 CMake 或 Python 奇怪报错，重新开一个终端，执行：

```bash
conda deactivate
unset LD_LIBRARY_PATH
source /opt/ros/jazzy/setup.bash
```
