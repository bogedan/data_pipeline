# 09_finalize_metadata.py

根据最终 parquet 和视频生成完整的 LeRobot v2.1 metadata 与统计信息，并将 pipeline marker 标记为完成。

## 用法

```bash
python 09_finalize_metadata.py --config config.yaml
```

## 前置条件

- `03` 已生成输出数据集。
- 所有配置为 enabled 的轨迹处理阶段已经完成。
- `08` 已生成每个 episode、每条手臂的视频。

脚本自动选择最后一个启用阶段的 `*_partial_episode_stats.jsonl` 作为 parquet 统计来源。

## 处理内容

- 为每个视频抽取最多 `metadata.video_stats_frames_per_episode` 帧估计图像统计。
- 写入 `meta/episodes_stats.jsonl`。
- 聚合生成 `meta/stats.json`。
- 重建 `meta/info.json` 中的 state、action、标量和视频 feature 定义。
- 更新 episode、frame、video、chunk 和 split 计数。
- 保留源数据的 `robot_type`。

## 输出

最终 metadata 写入 `<最终输出目录>/meta`。成功后 `.openpi_data_pipeline.json` 包含：

- `complete: true`
- 最终 episode 数
- 最终 frame 数
- 各已完成阶段的状态和报告路径
