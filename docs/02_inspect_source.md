# 02_inspect_source.py

在轨迹转换前检查实际源数据集、任务配置、手臂字段、夹爪标定、视频和外部程序。

## 用法

```bash
python 02_inspect_source.py --config config.yaml
```

## 检查内容

- `meta/info.json`、`episodes.jsonl` 和首个 episode parquet。
- `observation.state` 的字段名与 shape。
- `robot.arms` 对 pose source、夹爪和 MVS 相机的绑定。
- 每条手臂的视频文件是否存在。
- 夹爪宽度到弧度的标定 section 与数值范围。
- `trajectory.pose_transform.enabled` 是否明确设置。
- `ffmpeg` 和 `ffprobe` 是否可执行。
- 源 FPS 是否符合当前 30 Hz OpenPI UMI 假设。

## 输出

检查报告写入：

```text
<output.root>_work/02_inspection.json
```

warning 会显示但不阻止运行；存在 error 时脚本以非零状态退出。
