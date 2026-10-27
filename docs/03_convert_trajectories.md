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

脚本创建与原始输入同级的 `<输入数据集>_openpi`，写入 parquet、任务 metadata 和 `.openpi_data_pipeline.json` marker，并将当前有效源数据集的 `assets/` 目录复制到输出数据集。视频由 `08` 生成。

如果启用了 merge 或夹爪宽度预处理，“当前有效源数据集”分别是对应阶段的输出，因此最终保存的是实际参与轨迹转换的任务配置、Rerun blueprint、标定文件及其他归档 assets。源数据集缺少 `assets/` 时，脚本会在修改已有输出前停止。

`*.rrd` 不会复制，因为其中的轨迹点云对应处理前数据；可在最终数据生成后重新创建，避免把旧轨迹误认为处理结果。

如果输出已存在：

- `output.overwrite: false` 时停止。
- `output.overwrite: true` 时，只有目录包含 pipeline marker 才允许重建。

工作目录输出：

- `03_partial_episode_stats.jsonl`
- `03_trajectory_report.json`
- `03_step_limit_reference.json`，仅在 `step_limit.enabled: true` 时生成

百分位参考在平滑前计算，因此后续 `05` 始终使用同一组原始阈值。
