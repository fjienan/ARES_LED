# 分相机训练数据

每台相机必须维护独立数据集，禁止跨相机复制颜色样本或 NONE 负样本：

```text
camera_data/
├── usb_rgb/
│   ├── UNCLASSIFIED/
│   ├── RED/ GREEN/ BLUE/ YELLOW/ PURPLE/
│   └── NONE/
└── odin1/
    ├── UNCLASSIFIED/
    ├── RED/ GREEN/ BLUE/ YELLOW/ PURPLE/
    └── NONE/
```

每张彩色类别图片中应只出现对应颜色的灯带；`NONE` 中不得存在任何有效灯带。
采集的新图片先进入对应相机的 `UNCLASSIFIED/`，使用
`capture_<四位编号>.jpg` 命名。人工分类后再移入类别目录，并改为
`<类别小写>_<四位编号>.jpg`。

采集命令见项目根目录 README。`positive_review/` 是实时检测自动保存的复核目录，
不会作为训练数据或提交到 Git；人工确认后再移动到相应类别。
