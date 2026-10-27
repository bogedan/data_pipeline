# 04_smooth_trajectories.py

平滑转换后的 TCP 平移和 SO(3) 旋转，不改变帧数、夹爪值和配置指定的首尾帧。

## 用法

```bash
python 04_smooth_trajectories.py --config config.yaml
```

`smoothing.enabled: false` 时成功跳过。

## 行为

- 位置使用居中 binomial filter。
- 旋转矩阵加权后通过 SVD 投影回 SO(3)。
- 双臂分别平滑。
- 从平滑后的 state 重新生成 action。
- 写入前先计算全部 episode 并检查旋转矩阵质量。

相关字段见 [CONFIG.md](CONFIG.md) 的 `smoothing`。

## 输出

- 原地更新 `<最终输出目录>/data` 中的 state 和 action。
- `04_smoothing_manifest.jsonl`
- `04_partial_episode_stats.jsonl`
- `04_smoothing_report.json`
- `04_smoothing_comparison.rrd`

该阶段通过 marker 防止重复应用。需要修改参数时从 `03` 重建输出。
