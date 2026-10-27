# run_data_pipeline.sh

按固定顺序执行完整数据预处理和 OpenPI 转换流水线。`validate_openpi.py` 不在该执行链中。

## 用法

使用默认配置：

```bash
./run_data_pipeline.sh
```

指定配置：

```bash
./run_data_pipeline.sh /path/to/config.yaml
```

指定 Python 解释器：

```bash
DATA_PIPELINE_PYTHON=/path/to/python ./run_data_pipeline.sh /path/to/config.yaml
```

不设置 `DATA_PIPELINE_PYTHON` 时使用当前环境中的 `python`。

## 执行顺序

```text
00_merge_datasets.py
01_add_gripper_width.py
02_inspect_source.py
03_convert_trajectories.py
04_smooth_trajectories.py
05_limit_tcp_steps.py
06_trim_static_segments.py
07_sparsify_dense_turns.py
08_convert_videos.py
09_finalize_metadata.py
10_visualize_tcp_distribution.py
```

脚本使用 `set -Eeuo pipefail`，任一阶段失败会立即停止。具有 `enabled` 字段的阶段关闭时会自行成功跳过。

未启用合并时只需配置 `source.root`。夹爪宽度输出自动使用 `<source.root>_width`，最终输出自动使用 `<source.root>_openpi`。

## 复跑

- `00` 和 `01` 的目标已存在时，由各自的 `overwrite` 决定报错或原子替换。
- `03` 的 `output.overwrite: true` 只允许重建带 pipeline marker 的输出目录。
- `04` 到 `07` 是原地轨迹变换，marker 为 `running` 或 `complete` 时不能重复执行。
- 原地变换中断后，从 `03` 重建输出，再按顺序继续。

各配置字段见 [CONFIG.md](CONFIG.md)。
