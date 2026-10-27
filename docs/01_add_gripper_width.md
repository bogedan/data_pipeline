# 01_add_gripper_width.py

从数据集归档的 MVS 视频离线识别夹爪宽度，对未识别帧插值，并更新 `observation.state` 和统计信息。

## Pipeline 用法

```bash
python 01_add_gripper_width.py --config config.yaml
```

读取 `preprocessing.add_gripper_width`。如果 merge 已启用，输入自动取 merge 输出；否则使用 `source.root`。输出自动保存到输入数据集同级的 `<输入数据集>_width`。

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

## 数据要求与处理规则

脚本支持右单臂以及 `right + left` 双臂数据。机械臂按 `right`、`left` 排列，每个宽度紧跟对应的七维 TCP：

```text
右单臂：right TCP(7) + right width
双臂：  right TCP(7) + right width + left TCP(7) + left width
```

未绑定到机械臂的其他 state 字段保留在机械臂字段之后。宽度单位为米，表示两个 marker 中心之间的距离。`gripper_detect_online.<gripper_id>.enabled` 只控制采集期间的在线检测，离线处理仍会使用同一配置中的 `roi_xywh`。

`mode: add` 用于原始 state 中没有 width 字段的数据；只要输入中已有任意 `.width_m` 字段就会拒绝执行。采集时关闭在线检测而留下 `-1.0` 占位值时，应使用 `mode: overwrite`，该模式保持 state 的维度和字段顺序不变。

输入视频帧数必须与对应 Parquet 行数一致。输出会先写入临时目录，全部 episode 成功后再发布；处理中途失败不会留下不完整的目标数据集。原始数据集不会被修改。

由于输入是压缩后的 MP4，离线识别结果可能与采集现场显示的实时宽度略有差异。每个 episode 的成功识别数和插值数可在 `meta/gripper_width_report.json` 中查看。
