# usb_rgb_1 检测优化进度记录

记录时间：2026-07-08

## 当前目标

根据新的 `camera_data/usb_rgb_1` 训练图片调整 R2 端 usb_rgb_1 单色灯带检测算法。

目标优先级：

1. 不能误检：`NONE` 图片不能接受任何候选。
2. 尽量提高正样本召回：RED / GREEN / BLUE / PURPLE 都要识别。
3. 识别速度可以比之前稍慢，但不能太慢；本轮已按 p95 ≤ 80 ms 作为实用目标。
4. 不修改 `CameraConfig_G1M*.ini` 文件。
5. 本轮只处理 usb_rgb_1 单色检测；暂不改协议层、usb_rgb_2、shared。

## 已确认的当前数据状态

`camera_data/usb_rgb_1` 当前有效类别：

- `RED`
- `GREEN`
- `BLUE`
- `PURPLE`
- `NONE`

`CYAN` / `YELLOW` 已不应参与当前 usb_rgb_1 模型。旧 `detector.yaml` 里仍启用 CYAN，这是需要修正的点。

图片实际尺寸为 1920×1080。之前 receiver 里请求 2560×1440 只影响实时摄像头采集，不影响离线图片评估。

样本特点：

- 正样本里的灯带很远、很小，常常只占画面极小区域。
- 背景中有大量白色天花灯、反光线和局部彩色噪声。
- 因此不能退回到简单 HSV 面积检测；必须保留“彩色灯珠点列 + 共线 + 间距近似等差 + 亮度谷值 + 颜色一致性”的结构约束。

## 已做过的基线评估

基线命令：

```bash
PYTHONPATH=/home/lyx/Desktop/LED/r2_ws/src/rgb_camera_receiver \
/usr/bin/python3 -m rgb_camera_receiver.evaluate \
  --camera-profile usb_rgb_1 \
  --dataset /home/lyx/Desktop/LED/camera_data/usb_rgb_1 \
  --output /tmp/led_usb_rgb_1_eval_baseline \
  --config /home/lyx/Desktop/LED/r2_ws/src/rgb_camera_receiver/config/cameras/usb_rgb_1/detector.yaml
```

结果：

- `evaluated=116`
- `passed=15`
- `failed=101`
- `scale=0.40`
- `median_ms≈22.5`
- `p95_ms≈25.5`
- 失败模式：正样本几乎全漏检，`NONE` 是通过的。

处理缩放扫描结果：

- scale 1.00：`passed=43`，`p95≈163.9 ms`
- scale 0.75：`passed=39`，`p95≈81.0 ms`
- scale 0.67：`passed=36`，`p95≈65.8 ms`

结论：全尺寸保留更多小灯珠细节但太慢；0.67～0.75 更接近速度目标，但原参数召回不足。

## 已做的代码修改

已修改文件：

- `r2_ws/src/rgb_camera_receiver/rgb_camera_receiver/classifier_usb_rgb_1.py`

已完成的改动：

1. `DetectorConfig` 增加粗区域参数：

   - `coarse_regions`
   - `coarse_min_saturation`
   - `coarse_min_value`
   - `coarse_min_area`
   - `coarse_max_area`
   - `coarse_min_color_pixels`
   - `coarse_padding_pixels`
   - `coarse_dilate_pixels`
   - `coarse_max_regions`

2. `load_config()` 已读取 `coarse_regions:` YAML 配置。

3. 修复 `_point_color_quality()` 中恢复合并灯珠路径的索引错误：

   - 原来 `own` 是针对 saturated 像素的布尔索引，却拿去索引整个 `bgr_patch[disk]`，会触发维度不匹配。
   - 现在先同步筛出 `bright_bgr` / `saturated_bgr`，再用 `saturated_bgr[own]`。

4. `_extract_light_points()` 的局部极大值恢复路径现在增加 `mask` 限制，避免在非该颜色区域恢复无关亮点。

5. 新增 `_merge_light_points()`：

   - 合并 `_component_light_points()` 与 `_extract_light_points()` 两条点提取路径。
   - 用距离去重，避免同一灯珠重复进入直线拟合。

6. 新增粗区域相关函数：

   - `_translate_candidate()`
   - `_merge_rectangles()`
   - `_coarse_region_rectangles()`
   - `_detect_proposals_full()`

7. `detect_proposals()` 已改为：

   - 若未启用 `coarse_regions`，走原全图精检测。
   - 若启用 `coarse_regions`，先粗筛小区域，再对每个 crop 跑完整精检测，最后把候选坐标平移回原图。

当前代码是中间状态：机制已加上，但 usb_rgb_1 的最终 `detector.yaml` 还没有更新。

## 已做的临时评估

临时配置 `/tmp/usb_rgb_1_try.yaml`：

- 去掉 CYAN。
- scale 0.75。
- 放宽小目标参数：
  - `min_dots=5`
  - `min_length_pixels=18`
  - `line_distance_pixels=4.5`
  - `min_dog_response=3.0`
  - `max_points_per_color=80`
  - `min_valley_contrast=0.02`
  - `min_periodic_dot_quality=0.22`
  - `min_periodic_color_quality=0.25`
  - `min_score=0.04`

在修复合并灯珠路径之前：

- `passed=63`
- `failed=53`
- `p95≈78.2 ms`
- `NONE` 零候选。

修复并启用合并灯珠路径之后：

```bash
PYTHONPATH=/home/lyx/Desktop/LED/r2_ws/src/rgb_camera_receiver \
/usr/bin/python3 -m rgb_camera_receiver.evaluate \
  --camera-profile usb_rgb_1 \
  --dataset /home/lyx/Desktop/LED/camera_data/usb_rgb_1 \
  --output /tmp/led_usb_rgb_1_eval_try_after_points \
  --config /tmp/usb_rgb_1_try.yaml
```

结果：

- `evaluated=116`
- `passed=70`
- `failed=46`
- `scale=0.75`
- `min_true≈0.220686`
- `max_false=0`
- `NONE` 仍通过
- `median_ms≈113.7`
- `p95_ms≈220.1`

结论：恢复合并灯珠显著提高召回，但全图精检测太慢。

## 粗区域当前发现的问题

临时启用粗区域配置 `/tmp/usb_rgb_1_try_coarse.yaml` 后：

- `passed=70`
- `failed=46`
- `median_ms≈103.2`
- `p95_ms≈220.4`

几乎没有提速。

已定位原因：

- `_coarse_region_rectangles()` 当前使用了：

  ```python
  seed = color_union | generic_color
  ```

- `generic_color` 会把白色天花灯/反光区域卷入粗筛，导致粗框过多或过大。
- 慢图示例：
  - `GREEN/green_010.jpg` 粗框 10 个，其中大框面积超过 10 万像素。
  - `GREEN/green_009.jpg` 粗框 7 个，其中最大框面积超过 31 万像素。

下一步应修改为：

- usb_rgb_1 默认粗筛只使用 `color_union`，不使用通用高饱和 `generic_color`。
- 如果保留 generic，应增加配置开关，例如 `coarse_regions.include_generic_color: false`，默认 false。
- 粗框需要限制长条白灯区域：
  - 原始颜色像素数量要足够；
  - 彩色像素占粗框面积比例要达标；
  - 可对大面积框直接丢弃或拆分。

## 下一步执行计划

1. 修改 `classifier_usb_rgb_1.py` 的粗筛逻辑：

   - 增加 `coarse_include_generic_color` 配置，默认 false。
   - `seed` 默认只用 `color_union`。
   - 增加 `coarse_min_color_fraction`，过滤大面积但彩色像素稀少的粗框。

2. 更新 `r2_ws/src/rgb_camera_receiver/config/cameras/usb_rgb_1/detector.yaml`：

   - 删除 CYAN。
   - 设置 processing scale 初步为 0.75。
   - 写入新的四种颜色阈值和几何参数。
   - 启用粗区域：

     ```yaml
     coarse_regions:
       enabled: true
       include_generic_color: false
       min_saturation: 45
       min_value: 35
       min_area: 3
       max_area: 5000
       min_color_pixels: 3
       min_color_fraction: 0.001
       padding_pixels: 28
       dilate_pixels: 13
       max_regions: 32
     ```

3. 重新评估：

   ```bash
   cd ~/Desktop/LED
   PYTHONPATH=/home/lyx/Desktop/LED/r2_ws/src/rgb_camera_receiver \
   /usr/bin/python3 -m rgb_camera_receiver.evaluate \
     --camera-profile usb_rgb_1 \
     --dataset /home/lyx/Desktop/LED/camera_data/usb_rgb_1 \
     --output /home/lyx/Desktop/LED/camera_results/usb_rgb_1 \
     --config /home/lyx/Desktop/LED/r2_ws/src/rgb_camera_receiver/config/cameras/usb_rgb_1/detector.yaml
   ```

4. 根据结果调参：

   - 若 `NONE` 出现候选：提高 `min_score` 或收紧粗框/颜色通道约束。
   - 若正样本大量漏检：优先调小 `min_length_pixels`、`min_dog_response`、`min_periodic_*`，而不是简单扩大 HSV。
   - 若速度仍慢：减少 `max_regions`、降低 `max_points_per_color`、收紧粗框面积/彩色比例。

5. 生成并检查 `camera_results/usb_rgb_1` 标注图：

   - 重点看每类漏检样本。
   - `NONE` 必须没有接受候选。

## 当前风险与注意事项

- 当前工作区已有大量用户数据变更，包括训练图片和 ini 文件状态。不要回滚或整理这些文件，除非用户明确要求。
- 不要修改 `CameraConfig_G1M*.ini`。
- 当前 `classifier_usb_rgb_1.py` 已被修改，但还没有最终验证通过。
- 暂时不要 build，除非用户明确要求；修改完成后只告诉用户需要 build。

## 2026-07-08 完成记录

本轮已完成 usb_rgb_1 单色灯带检测优化，并生成正式标注结果。

修改文件：

- `r2_ws/src/rgb_camera_receiver/rgb_camera_receiver/classifier_usb_rgb_1.py`
- `r2_ws/src/rgb_camera_receiver/config/cameras/usb_rgb_1/detector.yaml`

主要改动：

1. 粗筛默认只使用已配置颜色，不再把通用高亮/高饱和区域纳入粗框。
2. 增加 `include_generic_color`、`min_color_fraction` 等粗筛参数。
3. 粗筛阶段支持二次降采样；当前 `coarse_regions.scale: 0.6`。
4. 粗筛结果按颜色分组，精检测只在对应颜色的 crop 内执行，避免每个 crop 重复检测四个颜色。
5. 删除 usb_rgb_1 detector 中的 CYAN，仅保留 RED / GREEN / BLUE / PURPLE。
6. 关闭旧的短连续条拒绝规则，因为远距离灯珠过曝/融合后真实目标会呈现这种形态。
7. 增加严格受限的 `merged_component` 补充路径，用于识别远距离灯珠融合成短彩色段的样本。
8. 增加低谷值短链例外：仅对颜色质量高、点质量高、残差很小的 4 点短链给低分补偿，解决极小目标暗谷不明显的问题。

最终正式评估命令：

```bash
cd ~/Desktop/LED
PYTHONPATH=/home/lyx/Desktop/LED/r2_ws/src/rgb_camera_receiver \
/usr/bin/python3 -m rgb_camera_receiver.evaluate \
  --camera-profile usb_rgb_1 \
  --dataset /home/lyx/Desktop/LED/camera_data/usb_rgb_1 \
  --output /home/lyx/Desktop/LED/camera_results/usb_rgb_1 \
  --config /home/lyx/Desktop/LED/r2_ws/src/rgb_camera_receiver/config/cameras/usb_rgb_1/detector.yaml
```

最终结果：

- `evaluated=116`
- `passed=116`
- `failed=0`
- `scale=0.75`
- `min_true=0.071575`
- `max_false=0.000000`
- `separation=inf`
- `none_validation=passed`
- `median_ms=32.5`
- `p95_ms=47.7`
- 输出目录：`camera_results/usb_rgb_1`

后续若要在 ROS2 运行时使用这些修改，需要重新 build `rgb_camera_receiver`。
