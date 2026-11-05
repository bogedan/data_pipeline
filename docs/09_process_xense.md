# 09_process_xense.py

使用源数据集保存的 Xense Rectify 视频和每个 episode 的 `xense_runtime_set_id`，在最终帧筛选后生成训练特征。

## 用法

```bash
python 09_process_xense.py --config config.yaml
```

## 输出

- `Difference`：每个传感器一个 RGB H.264 视频，默认 CRF 18。
- `Depth`：固定 `[0,1]` 映射到 uint8 灰度并复制成 RGB H.264，默认 CRF 0。
- `Force`：可独立选择主 Episode Parquet 的 `float32[35,20,3]`、H.264 RGB 视频，或两者同时输出。
- `ForceResultant`：主 Episode Parquet 的 `float32[6]` 字段。

字段名使用任务 YAML 中 `robot.xense` 的逻辑角色，不使用硬件序列号。求解器按顺序读取全部原始 Rectify 帧，只有 trim 和 turn sparsification 最终保留的帧会写入输出。

## 资产和报告

稳定的深度编码资产始终写入；色图 JSON 和预览 PNG 分别由
`visualization_assets.save_colormaps`、`visualization_assets.save_preview_png` 控制：

```text
assets/xense/depth_encoding.json
assets/xense/depth_colormaps.json
assets/xense/depth_colormaps_preview.png
assets/xense/force_encoding.json  # 仅 Force 视频启用时
```

每次处理的数据范围、文件大小、runtime set 和源帧映射写入：

```text
work/09_xense_report.json
work/09_xense_episode_manifest.jsonl
```

Force 视频对最终数据集全部选中帧按传感器、按 XYZ 通道统计完整 min/max，再统一映射为 uint8 RGB。范围和反解公式写入 `force_encoding.json`。Stage 09 会原地修改最终数据集，不能重复应用；失败或参数变化后应从 Stage 03 重建输出。
