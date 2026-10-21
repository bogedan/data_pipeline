# 03_convert_trajectories.py

把源 `observation.state` 转为 OpenPI UMI pose10，并生成与 state 对齐的 action。该脚本是派生输出的重建起点。

## 用法

```bash
python 03_convert_trajectories.py --config config.yaml
```

## 转换结果

每条手臂输出 10 维：

```text
x, y, z, rot6d_0..5, gripper_angle_rad
```

- 输入四元数按 `xyzw` 读取并转换为旋转矩阵前两列的 column-rot6d。
- 可选应用 `trajectory.pose_transform`。
- 夹爪宽度依据数据集内标定线性映射为弧度。
- `action[t] = state[t + action_offset_frames]`，尾部按 `repeat_last` 补齐。
- `task.prompt` 为字符串时替换所有任务描述；为 `null` 时保留源任务。

## 输出与安全

脚本创建 `output.root`，写入 parquet、任务 metadata 和 `.openpi_data_pipeline.json` marker。视频由 `08` 生成。

如果输出已存在：

- `output.overwrite: false` 时停止。
- `output.overwrite: true` 时，只有目录包含 pipeline marker 才允许重建。

工作目录输出：

- `03_partial_episode_stats.jsonl`
- `03_trajectory_report.json`
- `03_step_limit_reference.json`，仅在 `step_limit.enabled: true` 时生成

百分位参考在平滑前计算，因此后续 `05` 始终使用同一组原始阈值。
