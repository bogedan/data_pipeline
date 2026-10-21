# 05_limit_tcp_steps.py

将相邻帧 TCP 位置和旋转步长限制在 `03` 计算的原始轨迹百分位阈值附近。

## 用法

```bash
python 05_limit_tcp_steps.py --config config.yaml
```

`step_limit.enabled: false` 时成功跳过。启用 smoothing 时，本阶段要求 `04` 已完成。

## 阈值和容差

- `percentile` 选择 `03_step_limit_reference.json` 中的原始阈值。
- `storage_margin` 将内部投影目标略微收紧，为 float32 写入留出空间。
- `position_tolerance_mm` 是位置收敛和校验余量，单位为毫米。
- `rotation_tolerance_deg` 是旋转收敛和校验余量，单位为度。
- `preserve_endpoint_frames` 指定首尾固定帧数。
- `max_sweeps` 限制投影迭代轮数。

位置和旋转独立处理，夹爪值保持不变。处理后根据新 state 重建 action。

## 输出

- 原地更新 `output.root/data` 中的 state 和 action。
- `05_step_limit_manifest.jsonl`
- `05_partial_episode_stats.jsonl`
- `05_step_limit_report.json`
- `05_step_limit_comparison.rrd`

报告记录原始阈值、内部目标、配置容差、每个 episode 的迭代次数和最终最大步长。

该阶段不能重复应用。中断后从 `03` 重建，再依次执行 `04` 和 `05`。
