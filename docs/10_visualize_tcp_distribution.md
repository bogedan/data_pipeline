# 10_visualize_tcp_distribution.py

将最终数据集中所有 episode、所有手臂的 TCP 位置写入一个 Rerun 3D recording，用于检查空间分布。

## 用法

```bash
python 10_visualize_tcp_distribution.py --config config.yaml
```

可选参数：

```bash
python 10_visualize_tcp_distribution.py \
  --config config.yaml \
  --output /path/to/tcp_distribution.rrd \
  --point-radius-m 0.0012 \
  --with-paths \
  --path-radius-m 0.00035
```

| 参数 | 说明 |
| --- | --- |
| `--output` | 覆盖默认 RRD 输出路径。 |
| `--point-radius-m` | 覆盖配置中的点半径。 |
| `--with-paths` | 除点云外，同时绘制每个 episode 的轨迹线。 |
| `--path-radius-m` | 覆盖配置中的轨迹线半径。 |

## 输出

默认写入：

```text
<output.root>_work/10_all_episode_tcp_pointcloud.rrd
```

不同 episode 使用不同颜色，起点为白色，终点为黑色。脚本还会输出 episode 数、TCP 点数和 XYZ 边界。
该阶段只读，不修改数据集。
