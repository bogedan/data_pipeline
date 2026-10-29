# 00_merge_datasets.py

合并多个兼容的 LeRobot v2.1 数据集，连续重排 episode 和全局 frame index，并直接复制视频，不重新编码。

## Pipeline 用法

```bash
python 00_merge_datasets.py --config config.yaml
```

读取 [CONFIG.md](CONFIG.md) 中的 `preprocessing.merge`。`enabled: false` 时成功跳过。

## 独立用法

```bash
python 00_merge_datasets.py \
  --input /path/to/dataset_1 \
  --input /path/to/dataset_2 \
  --output /path/to/merged
```

目标已存在且需要替换时增加 `--overwrite`。`--config` 不能和 `--input`、`--output`、`--overwrite` 混用。

## 输入要求

- 至少两个 LeRobot v2.1 数据集。
- 任务 metadata、特征 schema、视频键和归档 `assets` 必须一致；数据衍生的 `*.rrd` 不参与兼容性检查。
- 每个 episode 的 parquet 行数、`frame_index` 和 metadata 必须匹配。
- 自动发现模式只接受 `<dataset_name>_<date>_HHMMSS`。

## 输出

- 重排后的 `data` 和 `videos`。
- 合并后的 `meta/info.json`、`episodes.jsonl`、`episodes_stats.jsonl` 和 `stats.json`。
- 连续的 `episode_index` 和全局 `index`。

输入数据集中的旧 `*.rrd` 不会复制到合并结果，因为其中的轨迹只对应各自的输入数据。可在合并或完整处理后重新生成。

输出先写入同级临时目录，全部成功后再原子发布。`overwrite: false` 不会复用已有目录，而是直接报错。
