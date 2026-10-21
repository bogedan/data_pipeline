# 01_add_gripper_width.py

从数据集归档的 MVS 视频离线识别夹爪宽度，对未识别帧插值，并更新 `observation.state` 和统计信息。

## Pipeline 用法

```bash
python 01_add_gripper_width.py --config config.yaml
```

读取 `preprocessing.add_gripper_width`。如果 merge 已启用，输入自动取 merge 输出；否则使用 `source.root`。

## 独立用法

```bash
python 01_add_gripper_width.py \
  --input /path/to/input_dataset \
  --output /path/to/output_dataset \
  --mode add
```

可用参数：

- `--mode add`：新增每个夹爪的 `<gripper_id>.width_m` 字段。
- `--mode overwrite`：覆盖已有的 width 字段。
- `--overwrite`：原子替换已有输出目录。

`--config` 不能和其他输入输出参数混用。

## 配置来源

脚本自动从输入数据集 `assets` 读取：

- 唯一包含 `robot`、`dataset` 和 `gripper_detect_online` 的采集 YAML。
- `robot.arms` 中的夹爪与 MVS 绑定。
- 相机图像尺寸、旋转方式和 marker 参数。
- 每个夹爪的 ROI 与距离标定。

## 插值行为

单帧未检测到 marker 时先记录为缺失值，再使用同一 episode 内有效结果做线性插值。首尾缺失段使用最近的有效值。
如果一个夹爪在整个 episode 中都没有有效检测结果，脚本会报错，不会发布部分输出。

## 输出

输出是输入数据集的完整副本，并更新：

- parquet 中的 `observation.state`。
- `meta/info.json` 中 state 的 shape 和 names。
- episode 和全局 state 统计。
- `meta/gripper_width_report.json`，记录每个夹爪的检测和插值帧数。
