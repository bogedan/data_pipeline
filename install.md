# Data Pipeline 安装

Data pipeline 可以独立于数据采集环境运行。建议使用单独的 Conda 环境，因为 `requirements.txt` 固定了与 OpenPI 对齐的 LeRobot、PyAV 和 Rerun 版本，直接安装到采集环境可能改变采集依赖。

## 1. 初始化子模块

从 `omniumix_lerobot` 主仓库使用时，先确认子模块已经初始化：

```bash
cd /home/xcr/pbg/omniumix_lerobot
git submodule update --init data_pipeline
cd data_pipeline
```

如果已经能看到 `00_merge_datasets.py` 到 `10_visualize_tcp_distribution.py`，可以跳过这一步。

## 2. 安装 FFmpeg

视频检查和转换需要命令行程序 `ffmpeg`、`ffprobe`：

```bash
ffmpeg -version
ffprobe -version
```

Ubuntu 系统缺少时可以安装：

```bash
sudo apt update
sudo apt install ffmpeg
```

没有系统安装权限时，可以先完成下一节的环境创建，再在该 Conda 环境中安装：

```bash
conda install -c conda-forge ffmpeg
```

如果程序不在 `PATH` 中，可以在 pipeline 配置的 `video.ffmpeg` 和 `video.ffprobe` 中填写绝对路径。

## 3. 创建 Python 环境

推荐使用 Python 3.11 创建独立环境：

```bash
conda create -n omniumix_data_pipeline python=3.11 pip -y
conda activate omniumix_data_pipeline
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 包含：

- Parquet、YAML、OpenCV 和数值处理依赖；
- 视频及 LeRobot 数据集读取依赖；
- Rerun 轨迹可视化依赖；
- 与 OpenPI loader 对齐的 LeRobot commit。

安装 LeRobot commit 时需要本机存在 `git` 并能访问 GitHub。PyTorch 等依赖体积较大，首次安装需要预留下载时间和磁盘空间。

## 4. 验证安装

确认 Python 包与外部程序可用：

```bash
python -c "import av, cv2, numpy, pyarrow, rerun, torch, yaml; print('Python dependencies OK')"
python -c "import lerobot; print(lerobot.__file__)"
ffmpeg -version
ffprobe -version
```

检查 pipeline 脚本能正常加载：

```bash
python 00_merge_datasets.py --help
python 01_add_gripper_width.py --help
python validate_openpi.py --help
```

这些命令只检查环境和命令行参数，不会处理或修改数据集。

## 5. 配置与运行

复制或修改 `config.yaml`，然后运行：

```bash
./run_data_pipeline.sh config.yaml
```

从 `omniumix_lerobot` 主仓库也可以使用包装脚本：

```bash
./scripts/run_data_pipeline.sh configs/data_pipeline/config.yaml
```

配置字段见 [docs/CONFIG.md](docs/CONFIG.md)，各阶段说明见 `docs/` 下与脚本同名的文档。

## 6. OpenPI loader 验证

`validate_openpi.py` 是独立命令，不在默认 pipeline 中。只运行数据转换时不需要 OpenPI 仓库。

需要验证 LeRobot/OpenPI loader 时，在配置中填写：

```yaml
validation:
  require_lerobot_loader: true
  openpi_root: /path/to/openpi
```

然后运行：

```bash
python validate_openpi.py --config config.yaml
```

`openpi_root` 必须指向包含 `src/` 的 OpenPI 仓库。当前环境中的 LeRobot 版本应保持为 `requirements.txt` 固定的 commit，以匹配 OpenPI 的数据加载行为。

不需要 loader 验证时设置：

```yaml
validation:
  require_lerobot_loader: false
```

## 7. 可选：开发测试

运行单元测试需要额外安装 `pytest`：

```bash
python -m pip install pytest
python -m pytest -q tests
```

测试使用临时目录，不会修改真实数据集。
