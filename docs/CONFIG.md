# config.yaml 配置说明

`config.yaml` 同时控制数据预处理、轨迹转换、视频转换、metadata 生成和可选验证。完整流水线可执行：

```bash
./run_data_pipeline.sh /path/to/config.yaml
```

各 Python 脚本的 `--config` 默认值都是当前目录下的 `config.yaml`。路径建议使用绝对路径；
`source.task_config` 是例外，相对路径会相对于配置文件所在目录解析。

## 数据目录选择

流水线按以下优先级确定转换阶段的源数据集：

1. `preprocessing.add_gripper_width.enabled: true` 时使用它的 `output`。
2. 否则，`preprocessing.merge.enabled: true` 时使用它的 `output`。
3. 两项都关闭时使用 `source.root`。

因此，启用 merge 和 add_gripper_width 时，`source.root` 可以设为 `null`。

## preprocessing.merge

```yaml
preprocessing:
  merge:
    enabled: true
    discover:
      directory: /path/to/datasets
      dataset_name: task_name
      date: "20261019"
    output: /path/to/task_name_20261019_merged
    overwrite: false
```

| 字段 | 说明 |
| --- | --- |
| `enabled` | `false` 时 `00_merge_datasets.py` 直接跳过。 |
| `discover.directory` | 搜索原始采集数据集的父目录。 |
| `discover.dataset_name` | 数据集名称前缀。 |
| `discover.date` | 8 位采集日期。 |
| `output` | 合并数据集目录。 |
| `overwrite` | `true` 时原子替换已有输出；`false` 时已有输出会报错。 |

自动发现只接受名称严格匹配 `<dataset_name>_<date>_HHMMSS` 的目录，不会把 `_merged`、
`_width` 等派生目录再次合并。

也可以改用手动输入。`discover` 和 `inputs` 必须二选一：

```yaml
preprocessing:
  merge:
    enabled: true
    inputs:
      - /path/to/collection_1
      - /path/to/collection_2
    output: /path/to/merged
    overwrite: false
```

## preprocessing.add_gripper_width

```yaml
preprocessing:
  add_gripper_width:
    enabled: true
    mode: overwrite
    output: /path/to/merged_width
    overwrite: false
```

| 字段 | 说明 |
| --- | --- |
| `enabled` | `false` 时 `01_add_gripper_width.py` 直接跳过。 |
| `mode` | 只支持 `add` 或 `overwrite`。 |
| `output` | 写入夹爪宽度后的完整数据集目录。 |
| `overwrite` | `true` 时原子替换已有输出；`false` 时已有输出会报错。 |

`add` 在 `observation.state` 中新增每个夹爪的 `width_m` 字段；源数据已有该字段时会报错。
`overwrite` 要求每个夹爪已经有且只有一个 `width_m` 字段，并替换其值。采集配置、相机配置、
ROI 和夹爪标定均从输入数据集的 `assets` 读取。

## source

```yaml
source:
  root: null
  # task_config: assets/task.yaml
```

| 字段 | 说明 |
| --- | --- |
| `root` | 两个预处理阶段都关闭时使用的 LeRobot v2.1 数据集。 |
| `task_config` | 可选。`assets` 中存在多个任务 YAML 候选时，用它指定一个文件。 |

任务 YAML 必须包含 `robot` 和 `dataset`。其中 `robot.arms` 定义每条手臂对应的 pose source、
夹爪和 MVS 相机。支持 `right`，以及可选的 `left`。

## output

```yaml
output:
  root: /path/to/output_openpi
  chunks_size: 1000
  overwrite: true
```

| 字段 | 说明 |
| --- | --- |
| `root` | 最终 OpenPI 格式数据集目录。 |
| `chunks_size` | 每个 data/video chunk 最多包含的 episode 数。 |
| `overwrite` | `03` 是否允许重建已有输出。只有包含 `.openpi_data_pipeline.json` 的目录才会被删除。 |

工作报告写入最终数据集内的 `<output.root>/work` 目录。

## task

```yaml
task:
  prompt: null
```

`prompt: null` 保留源数据任务描述。设置字符串时，所有 episode 使用该字符串，并把
`task_index` 统一设为 `0`。

## trajectory

```yaml
trajectory:
  pose_transform:
    enabled: false
    rotation_order: intrinsic_xyz
    rotation_deg: {x: 30.0, y: 0.0, z: -90.0}
    translation_frame: rotated_local
    translation_m: [-0.00680, 0.0, 0.20087]
  action_offset_frames: 1
  tail_policy: repeat_last
  gripper:
    mapping: linear
    distance_section: distance
    radians_section: width_to_rad
    clip: true
```

| 字段 | 说明 |
| --- | --- |
| `pose_transform.enabled` | 必须明确设置。采集时已完成 tracker 到 TCP 变换时应为 `false`。 |
| `rotation_order` | 当前只支持 `intrinsic_xyz`。 |
| `rotation_deg` | 启用变换时的局部 XYZ 旋转角，单位为度。 |
| `translation_frame` | `rotated_local` 或 `tracker`。 |
| `translation_m` | 启用变换时的平移，单位为米。 |
| `action_offset_frames` | `action[t]` 指向未来第几帧的 state，通常为 `1`。 |
| `tail_policy` | 当前只支持 `repeat_last`。 |
| `gripper.mapping` | 当前只支持 `linear`。 |
| `distance_section` | 夹爪标定 YAML 中输入宽度区间的 section 名。 |
| `radians_section` | 夹爪标定 YAML 中输出弧度区间的 section 名。 |
| `clip` | 是否把映射结果裁剪到标定输出范围。 |

输出中每条手臂占 10 维：`xyz + column-rot6d + gripper_angle_rad`。双臂按 right、left 顺序拼接为 20 维。
夹爪宽度为 `-1.0` 的帧在映射后仍保持 `-1.0`。

## smoothing

```yaml
smoothing:
  enabled: true
  window_size: 5
  passes: 1
  preserve_endpoint_frames: 2
  visualization_episode_index: 0
  visualization_pose_stride: 10
  visualization_axis_length_m: 0.005
```

| 字段 | 说明 |
| --- | --- |
| `enabled` | 是否执行轨迹平滑；`false` 时 `04` 跳过。 |
| `window_size` | 平滑窗口大小，必须为正奇数。 |
| `passes` | 重复平滑次数。 |
| `preserve_endpoint_frames` | 首尾各保持不变的帧数。 |
| `visualization_episode_index` | 生成平滑前后 RRD 对比的 episode。 |
| `visualization_pose_stride` | RRD 中姿态坐标轴的帧间隔。 |
| `visualization_axis_length_m` | RRD 姿态坐标轴长度，单位为米。 |

该阶段不改变帧数和夹爪值，并从平滑后的 state 重新生成 action。

## step_limit

```yaml
step_limit:
  enabled: true
  percentile: 95.0
  position_tolerance_mm: 0.01
  rotation_tolerance_deg: 0.01
  preserve_endpoint_frames: 1
  max_sweeps: 3000
  storage_margin: 0.999
  visualization_episode_index: 0
```

| 字段 | 说明 |
| --- | --- |
| `enabled` | 是否执行相邻 TCP 步长限制；`false` 时 `05` 跳过。 |
| `percentile` | 用转换后、平滑前的原始轨迹计算阈值，必须在 0 到 100 之间。 |
| `position_tolerance_mm` | 位置收敛和校验余量，单位为毫米，必须非负。 |
| `rotation_tolerance_deg` | 旋转收敛和校验余量，单位为度，必须非负。 |
| `preserve_endpoint_frames` | 首尾各固定不动的帧数，至少为 1。 |
| `max_sweeps` | 位置和旋转投影算法的最大迭代轮数。 |
| `storage_margin` | 内部目标相对原始百分位阈值的缩放系数，范围 `(0, 1]`。 |
| `visualization_episode_index` | 生成限制前后 RRD 对比的 episode。 |

`03` 将阈值写入 `03_step_limit_reference.json`，`05` 使用固定阈值处理所有 episode。

## trim

```yaml
trim:
  enabled: true
  position_threshold_m: 0.030
  rotation_threshold_deg: 3.0
  gripper_threshold_rad: 0.1
  pre_roll_frames: 5
  post_roll_frames: 5
  min_active_frames: 5
  min_episode_frames: 32
```

| 字段 | 说明 |
| --- | --- |
| `enabled` | 是否裁剪首尾静止段；`false` 时 `06` 跳过。 |
| `position_threshold_m` | 判定位置运动的阈值，单位为米。 |
| `rotation_threshold_deg` | 判定旋转运动的阈值，单位为度。 |
| `gripper_threshold_rad` | 判定夹爪运动的阈值，单位为弧度。 |
| `pre_roll_frames` | 检测到运动前额外保留的帧数。 |
| `post_roll_frames` | 最后运动后额外保留的帧数。 |
| `min_active_frames` | 连续达到阈值后才认为是有效运动。 |
| `min_episode_frames` | 裁剪后允许的最少帧数。 |

## turn_sparsification

```yaml
turn_sparsification:
  enabled: true
  prefix_frames: 60
  prefix_min_spacing_m: 0.001
  turn_angle_threshold_deg: 10.0
  turn_context_frames: 5
  min_turn_context_distance_m: 0.002
  dense_step_threshold_m: 0.0015
  max_kept_step_m: 0.003
  max_position_error_m: 0.0005
  max_rotation_error_deg: 0.3
  gripper_change_threshold_rad: 0.02
  max_consecutive_removed: 1
  visualization_episode_index: 0
```

| 字段 | 说明 |
| --- | --- |
| `enabled` | 是否执行前缀空间采样和拐弯处稀疏化；`false` 时 `07` 跳过。 |
| `prefix_frames` | 参与前缀空间采样的帧数。 |
| `prefix_min_spacing_m` | 前缀保留点的最小 TCP 间距，单位为米。 |
| `turn_angle_threshold_deg` | 进入拐弯候选区域的方向变化阈值。 |
| `turn_context_frames` | 计算拐弯方向时前后使用的上下文帧数。 |
| `min_turn_context_distance_m` | 上下文至少移动该距离才采用方向判断。 |
| `dense_step_threshold_m` | 小于该距离的相邻点才属于过密候选。 |
| `max_kept_step_m` | 删除后两个保留点之间允许的最大位置步长。 |
| `max_position_error_m` | 删除点相对插值轨迹允许的最大位置误差。 |
| `max_rotation_error_deg` | 删除点相对插值姿态允许的最大旋转误差。 |
| `gripper_change_threshold_rad` | 删除候选附近允许的最大夹爪变化。 |
| `max_consecutive_removed` | 允许连续删除的最大帧数。 |
| `visualization_episode_index` | 生成稀疏化前后 RRD 对比的 episode。 |

双臂数据要求所有手臂都允许删除该帧。启用 `step_limit` 时，`07` 会恢复导致最终步长超限的区间。

## video

```yaml
video:
  width: 240
  height: 240
  crop_mode: center_square
  codec: libx264
  crf: 18
  preset: medium
  pixel_format: yuv420p
  workers: 4
  ffmpeg: ffmpeg
  ffprobe: ffprobe
```

| 字段 | 说明 |
| --- | --- |
| `width`, `height` | 输出视频分辨率。 |
| `crop_mode` | `center_square` 先居中裁成正方形，`stretch` 直接缩放。 |
| `codec` | ffmpeg 视频编码器。 |
| `crf` | 编码质量参数，libx264 中数值越低质量越高、文件越大。 |
| `preset` | 编码速度和压缩率预设。 |
| `pixel_format` | 输出像素格式。 |
| `workers` | 并行转换视频的线程数。 |
| `ffmpeg`, `ffprobe` | 可执行文件名或路径。 |

## metadata

```yaml
metadata:
  video_stats_frames_per_episode: 100
```

`video_stats_frames_per_episode` 控制 `09` 每个视频用于估计图像统计量的采样帧数，不改变视频内容。

## visualization

```yaml
visualization:
  with_paths: true
  point_radius_m: 0.0012
  path_radius_m: 0.00035
```

`with_paths` 控制 `10` 是否绘制每个 episode 的轨迹连线。其余两个值分别控制 TCP 点和轨迹线的 Rerun 显示半径，单位为米。命令行参数可以覆盖这些设置。

## validation

```yaml
validation:
  action_horizon: 32
  episode_limit: null
  require_lerobot_loader: true
  openpi_root: /path/to/openpi
```

| 字段 | 说明 |
| --- | --- |
| `action_horizon` | loader 验证时读取的 action chunk 长度，也是 trim 默认最小 episode 长度。 |
| `episode_limit` | `null` 处理全部 episode；整数用于从数据集开头做 smoke run。 |
| `require_lerobot_loader` | 独立验证脚本是否实际加载 LeRobot/OpenPI 数据。 |
| `openpi_root` | 包含 `src` 的 OpenPI 仓库路径。 |

`validate_openpi.py` 不在 `run_data_pipeline.sh` 默认执行链中。

## 阶段关闭和复跑

`smoothing`、`step_limit`、`trim`、`turn_sparsification` 设置 `enabled: false` 后，对应脚本成功退出且不修改数据。
后续 metadata 会选择最后一个实际启用阶段生成的统计。

`04` 到 `07` 会在 marker 中记录 `running` 和 `complete`。这些原地修改轨迹的阶段不能重复应用。
参数变化或中途写入失败时，应保持 `output.overwrite: true`，从 `03_convert_trajectories.py` 重建输出，
然后按编号继续执行。
