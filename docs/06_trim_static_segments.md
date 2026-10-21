# 06_trim_static_segments.py

检测并裁剪每个 episode 首尾的长静止段，同时保持动作、索引和后续视频选帧一致。

## 用法

```bash
python 06_trim_static_segments.py --config config.yaml
```

`trim.enabled: false` 时成功跳过。启用 smoothing 或 step_limit 时，要求对应前置阶段完成。

## 判定规则

脚本综合所有手臂的相邻帧变化：

- TCP 位置变化达到 `position_threshold_m`。
- TCP 旋转变化达到 `rotation_threshold_deg`。
- 夹爪变化达到 `gripper_threshold_rad`。

任一项连续满足 `min_active_frames` 才视为有效运动。检测边界外额外保留 `pre_roll_frames` 和
`post_roll_frames`，并保证结果不少于 `min_episode_frames`。

## 输出

- 对 parquet 应用 `[start_frame, end_frame)` 切片。
- 重置 episode 内 `timestamp`、`frame_index` 和全局 `index`。
- 根据裁剪后的 state 重建 action。
- 更新 `meta/episodes.jsonl` 中的 episode 长度。
- 写入 `06_trim_manifest.jsonl`、`06_partial_episode_stats.jsonl` 和 `06_trim_report.json`。

`08_convert_videos.py` 使用 manifest 对视频应用相同的帧区间。

该阶段不能重复应用。中断后应从 `03` 重建输出。
