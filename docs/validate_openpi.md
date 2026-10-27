# validate_openpi.py

独立检查最终数据集的 metadata、state/action 时序、TCP 步长、rot6d、视频以及可选的 LeRobot/OpenPI loader。
该脚本不在默认 pipeline 中。

## 用法

```bash
python validate_openpi.py --config config.yaml
```

应先完成 `09_finalize_metadata.py`。

## 静态检查

- pipeline marker 是否为 `complete`。
- episode 行数和 metadata length 是否一致。
- state/action shape 和有限值。
- `action[t]` 与未来 state 的对齐关系。
- rot6d 的正交性和行列式。
- 启用 step_limit 时，最终最大步长是否在原始阈值加配置容差内。
- 每个视频的分辨率和帧数是否与 parquet 一致。

## Loader 检查

`validation.require_lerobot_loader: true` 时，脚本把 `<openpi_root>/src` 加入 Python path，并实际构造
`LeRobotDatasetMetadata` 和 `LeRobotDataset`，读取数据集开头、中间、末尾样本及 action chunk。

如果当前环境没有对应的 LeRobot/OpenPI 依赖，可以设置：

```yaml
validation:
  require_lerobot_loader: false
```

这只关闭 loader 检查，其他静态检查仍会执行。

## 输出

```text
<最终输出目录>/work/validation_report.json
```

报告包含数据规模、最大误差、loader 状态和错误列表。任何检查失败时脚本以非零状态退出。
