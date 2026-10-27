# 08_convert_videos.py

根据手臂与 MVS 相机绑定生成 OpenPI 使用的视频特征，并同步应用轨迹裁剪和稀疏化的帧选择。

## 用法

```bash
python 08_convert_videos.py --config config.yaml
```

## 输入

- 源数据集中的 MVS 视频。
- `<最终输出目录>/meta/episodes.jsonl`。
- 启用 trim 时需要 `06_trim_manifest.jsonl`。
- 启用 turn_sparsification 时需要 `07_turn_sparsification_manifest.jsonl`。

单臂输出键为 `observation.images.fisheye_img`；双臂输出键分别带 `_right` 和 `_left`。

## 视频处理

- `center_square`：居中裁成正方形后缩放。
- `stretch`：直接缩放到目标尺寸。
- 使用配置的 codec、CRF、preset 和 pixel format 编码 MP4。
- 删除音轨。
- 稀疏化后用 ffmpeg `select` 精确选帧，并按源 FPS 重建时间戳。
- 使用 `video.workers` 并行处理不同 episode 和手臂。

每个视频先写入 `.tmp.mp4`，ffmpeg 成功后再替换目标文件。失败信息包含 ffmpeg stderr 的末尾内容。

## 输出

视频写入 `<最终输出目录>` 的 `video_path` 模板位置。该阶段不修改 parquet 和 metadata；最终视频特征由 `09` 写入。
