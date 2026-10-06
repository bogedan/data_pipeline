# OpenPI UMI 数据处理流水线

这套脚本把现有 LeRobot v2.1 数据转换为当前 `/share/project/lxy/openpi` 的 UMI 输入格式：

```text
observation.state:               float32[10]
action:                          float32[10]
observation.images.fisheye_img:  240x240 RGB H.264 video
task:                            LeRobot task metadata
```

10 维轨迹语义为：

```text
[x, y, z, rotation-matrix-column-0, rotation-matrix-column-1, gripper_angle_rad]
```

磁盘上的 action 是 absolute TCP target。默认采用：

```text
observation.state[t] = P_t
action[t]            = P_{t+1}
```

episode 最后一帧没有未来观测，因此重复最后一个 pose，以保持 parquet 与视频帧数一致。

## 脚本顺序

```text
00_inspect_source.py          检查源 metadata、parquet、视频和外部程序
01_convert_trajectories.py    quaternion -> column rot6d，width_m -> angle_rad，构造 action
02_trim_static_segments.py    裁掉 episode 首尾长静止段，重建索引/action，并记录视频裁剪范围
03_smooth_trajectories.py     平滑 TCP 平移和 SO(3) 旋转；不改帧数、不平滑夹爪
04_sparsify_dense_turns.py    只减少拐弯处空间距离过密的点；直线和夹爪变化帧不删
05_convert_videos.py          选择主相机，中心裁剪并编码为 240x240 H.264
06_finalize_metadata.py       生成 info、episodes_stats 和全局 stats
07_validate_openpi.py         验证时序、rot6d、视频，并通过 LeRobot loader 读取 action chunk
08_visualize_tcp_distribution.py  将全部 episode 的 TCP 点云写入一个 Rerun RRD
```

## 使用

先检查并修改 `config.yaml`，尤其是：

- `source.camera_key`：训练使用哪一路相机；
- `trajectory.action_offset_frames`：state 与 action 的时间关系；
- `trajectory.pose_transform.enabled`：是否额外应用一次固定的 tracker→TCP 变换，必须明确填 true/false；
- `trajectory.gripper.calibration_file`：同时包含 `distance` 和 `width_to_rad` 的夹爪标定文件；
- `task.prompt`：训练指令；
- `output.root`：输出目录。

脚本不会根据文件名、字段名或 `mock` 等标记猜测位姿变换是否已经使用。当前数据已经在采集
阶段应用了 tracker→TCP 变换，所以配置明确使用 `pose_transform.enabled: false`，直接使用
`observation.state`。不能在转换阶段再次应用 `my_transform`，否则会形成重复变换。

四元数会先正规化并转换成旋转矩阵，再取旋转矩阵前两列形成 column-rot6d。夹爪转换每次
运行时读取 `gripper_01.yaml`，把 `distance.observed_min_m/observed_max_m` 线性映射到
`width_to_rad.min_rad/max_rad`，并进行裁剪，不在流水线配置中复制标定数值。

安装轻量处理依赖；OpenPI 自己的环境若已包含这些包，可以直接使用它：

```bash
source /share/project/liyuanyuan/anaconda3/bin/activate data_pipeline
python -m pip install -r requirements.txt
```

逐步运行：

```bash
source /share/project/liyuanyuan/anaconda3/bin/activate data_pipeline
set -e
python 00_inspect_source.py --config config.yaml
python 01_convert_trajectories.py --config config.yaml
python 02_trim_static_segments.py --config config.yaml
python 03_smooth_trajectories.py --config config.yaml
python 04_sparsify_dense_turns.py --config config.yaml
python 05_convert_videos.py --config config.yaml
python 06_finalize_metadata.py --config config.yaml
python 07_validate_openpi.py --config config.yaml
python 08_visualize_tcp_distribution.py --config config.yaml
```

`set -e` 会让任一步骤失败时立即停止，避免在缺少上一步产物时继续运行后续脚本。修复配置或
脚本后，可以从失败的编号继续；不需要总是从 `00` 重跑。

`08` 只读取最终数据集，不修改 parquet 或视频。默认每个 episode 使用不同颜色绘制 TCP 点云，
白点表示起点、黑点表示终点；添加 `--with-paths` 可以同时绘制轨迹连线。输出位置为
`work_dir/05_all_episode_tcp_pointcloud.rrd`。

远程服务器没有 X server 时，使用 OpenPI 的 Rerun Web Viewer：

```bash
cd /share/project/dcw/projects/openpi

uv run rerun --serve-web \
  --bind=0.0.0.0 \
  --web-viewer-port=9090 \
  --port=9876 \
  /share/project/lxy/data_pipeline/work/test_20261003_155718_openpi_smooth/05_all_episode_tcp_pointcloud.rrd
```

然后转发服务器的 9090 和 9876 端口，在本地浏览器打开终端输出的完整 9090 URL。

### 首尾静止段裁剪

`trim` 配置综合 TCP 平移、旋转和夹爪变化检测 episode 离开初始/最终静止 plateau 的时刻。
默认阈值为 5 mm、3 度和 0.1 rad，并在动作边界前后各保留 5 帧。裁剪阶段会：

- 对 parquet 使用 `[start_frame, end_frame)` 切片；
- 重置 episode 内 `timestamp` 和 `frame_index`，重建全局 `index`；
- 从裁剪后的 `observation.state` 重新生成 shifted action，保证新尾帧不会指向已删除的状态；
- 更新 episode 长度和统计；
- 写入 `work_dir/01_5_trim_manifest.jsonl`，供 05 对视频应用完全相同的帧切片。

如果裁剪阶段中断，重新运行 01（需要 `output.overwrite: true`）后再执行裁剪，避免在已部分
裁剪的数据上重复裁剪。首次使用建议将 `validation.episode_limit` 设为 `1` 检查报告和视频同步。

### TCP 轨迹平滑

`smoothing` 在不删除、不增加帧的前提下平滑 TCP。位置使用居中的 5 帧 binomial filter；姿态先
恢复成旋转矩阵，对邻域矩阵加权后通过 SVD 投影回 SO(3)，再编码成 column-rot6d，因此不会
产生非正交旋转。夹爪值完全不变，避免模糊抓取和释放时刻；episode 首尾默认各 2 帧也保持原值。

平滑后会从新的 `observation.state` 重新构造 shifted action，并写出：

- `work_dir/01_6_smoothing_manifest.jsonl`：每个 episode 的位置/旋转改变量和 jerk；
- `work_dir/01_6_smoothing_report.json`：全数据集汇总；
- `work_dir/01_6_smoothing_comparison.rrd`：灰色原始轨迹、蓝色平滑轨迹和两者姿态坐标轴。

该阶段不改变帧数，因此视频无需选帧。确认 RRD 和报告可接受后，再单独决定是否增加内部静止
平台压缩，避免把“去抖”和“删除重复时间点”混在一次数据变换中。

### 前 50 帧空间取样与拐弯处过密点取舍

`turn_sparsification` 在平滑之后运行。它先对每个 episode 的前 `prefix_frames`（默认 50）帧
进行空间取样：首帧保留，TCP 距离上一个保留帧达到 `prefix_min_spacing_m`（默认 1 mm）后才
保留下一帧，并同时检查到第 50 帧边界的距离。第 50 帧之后不参与这一步取样。

随后使用前后若干帧的运动方向识别拐弯，但只有当前后
窗口各自移动达到 `min_turn_context_distance_m` 时才相信方向，避免低速噪声制造虚假大角度。
平稳直线区域从不进入删除候选。

拐弯区域内还必须同时满足以下约束才能删除一个点：相邻点原本过密；删除后 TCP 跳距不超过
`max_kept_step_m`；位置插值误差、SO(3) 旋转插值误差不超阈值；夹爪基本不变；连续删除帧数
不超上限。每个拐弯区域的最大曲率点强制保留。

阶段会生成：

- `work_dir/01_7_turn_sparsification_manifest.jsonl`：保留/删除索引和每个 episode 的约束结果；
- `work_dir/01_7_turn_sparsification_report.json`：全数据集删点比例；
- `work_dir/01_7_turn_sparsification_comparison.rrd`：蓝色平滑输入、绿色保留点、黄色删除点。

05 根据 manifest 从源视频选择完全相同的帧并重新编码，把输出重新设为固定 30 Hz；parquet 的 timestamp、
frame_index、全局 index 和 shifted action 同步重建。

`requirements.txt` 锁定了与 OpenPI 相同 commit 的 LeRobot。默认
`validation.require_lerobot_loader: true`，所以 `04` 除了检查 parquet、动作时序、rot6d、视频
帧数和尺寸，还会实际通过 LeRobot loader 读取当前帧、图像和 32 步 action chunk。

建议第一次把 `validation.episode_limit` 设为 `1`，并把 `output.root` 指向单独的 smoke 目录。全部通过后再改回 `null` 处理完整数据。

## 安全与复跑

`01` 是新一轮转换的起点。目标目录已经存在时默认停止。只有同时满足以下条件才会清理目标目录：

1. `output.overwrite: true`；
2. 目标目录包含本流水线生成的 `.openpi_data_pipeline.json` 标记。

因此不会覆盖源数据，也不会清理一个来源不明的目录。`02`、`03`、`04` 可以单独复跑。

## 接入 OpenPI

验收通过后，在 `openpi/src/openpi/training/config.py` 中新增配置：

```python
_make_umi_train_config(
    name="pi05_umi_move_bottle",
    repo_id="/share/project/lxy/test_20261003_155718_openpi",
    asset_id="umi_move_bottle",
)
```

然后计算该数据集自己的 normalization stats，再做两步 smoke training。不要复用其他任务的 stats。
