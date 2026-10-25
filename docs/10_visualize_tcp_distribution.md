# 10_visualize_tcp_distribution.py

将处理前和处理后的所有 episode、所有手臂 TCP 位置分别写入 Rerun 3D recording，用于对比空间分布。

## 用法

```bash
python 10_visualize_tcp_distribution.py --config config.yaml
```

可选参数：

```bash
python 10_visualize_tcp_distribution.py \
  --config config.yaml \
  --origin-output /path/to/origin.rrd \
  --output /path/to/processed.rrd \
  --point-radius-m 0.0012 \
  --with-paths \
  --path-radius-m 0.00035
```

| 参数 | 说明 |
| --- | --- |
| `--origin-output` | 覆盖处理前 RRD 的默认输出路径。 |
| `--output` | 覆盖处理后 RRD 的默认输出路径。 |
| `--point-radius-m` | 覆盖配置中的点半径。 |
| `--with-paths` / `--no-with-paths` | 覆盖 `visualization.with_paths`，开启或关闭轨迹连线。 |
| `--path-radius-m` | 覆盖配置中的轨迹线半径。 |

## 输出

默认根据源数据集 `assets/` 中的任务配置文件名，写入最终数据集：

```text
<output.root>/assets/<任务配置名>_origin.rrd
<output.root>/assets/<任务配置名>.rrd
```

`_origin.rrd` 读取管线输入数据中的原始 TCP 位置，另一份读取完成平滑、限步、裁剪和稀疏化后的最终轨迹。不同 episode 使用不同颜色，起点为白色，终点为黑色。脚本还会输出 episode 数、TCP 点数和 XYZ 边界。
该阶段只读取源数据和最终数据的 parquet，不修改 parquet、视频或 metadata。
