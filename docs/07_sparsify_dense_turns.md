# 07_sparsify_dense_turns.py

删除轨迹前缀和拐弯区域中的过密采样点，同时保护夹爪变化、轨迹形状和已配置的步长上限。

## 用法

```bash
python 07_sparsify_dense_turns.py --config config.yaml
```

`turn_sparsification.enabled: false` 时成功跳过。所有已启用的 `04`、`05`、`06` 前置阶段必须完成。

## 两步处理

1. 对前 `prefix_frames` 做空间采样，只在任一手臂相对上个保留点达到 `prefix_min_spacing_m` 时保留。
2. 在后续轨迹中识别拐弯区域，仅删除同时满足位置、旋转、夹爪和连续删除数量约束的过密点。

双臂数据中，所有手臂都必须允许删除同一帧。拐弯区域的最大曲率点强制保留。

启用 step_limit 时，脚本使用相同的位置和角度容差检查删帧后的新相邻步长。导致超限的区间会恢复中间帧。
全部 episode 在任何 parquet 写入前完成该检查。

## 输出

- 选取保留行并重建 `timestamp`、`frame_index`、全局 `index` 和 action。
- 更新 `meta/episodes.jsonl`。
- `07_turn_sparsification_manifest.jsonl`
- `07_partial_episode_stats.jsonl`
- `07_turn_sparsification_report.json`
- `07_turn_sparsification_comparison.rrd`

manifest 中的 `kept_trimmed_frame_indices` 供 `08` 对视频选择完全相同的帧。

该阶段不能重复应用。中断后应从 `03` 重建输出。
