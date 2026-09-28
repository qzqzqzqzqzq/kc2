# 三维重建演示基线

GitHub 交接入口见根目录 [HANDOFF.md](../HANDOFF.md)。下面保留原机器运行记录；新安装请使用 `requirements.txt`，不要直接安装带本机路径的 `requirements.lock.txt`。仓库包含最终场景缓存与原始选帧，预训练权重需另行下载。

成品见 [交付素材/使用说明与实验记录.md](交付素材/使用说明与实验记录.md)。本目录已完成官方 VGGT + gsplat 的小场景演示；不修改现有开题 PPT。

## 复现本次结果

以下命令在项目根目录的 PowerShell 中执行。已配置的独立环境为 `D:\vggt311`，独立 CUDA 编译组件为 `D:\vggt_cuda128`，不依赖基础 conda 环境。

```powershell
D:\vggt311\Scripts\python.exe baseline_demo/scripts/prepare.py --scene kitchen --count 12
D:\vggt311\Scripts\python.exe baseline_demo/scripts/infer.py --scene kitchen --count 12
D:\vggt311\Scripts\python.exe baseline_demo/scripts/validate_scene.py --run kitchen_12
D:\vggt311\Scripts\python.exe baseline_demo/scripts/train.py --run kitchen_12 --steps 12000
D:\vggt311\Scripts\python.exe baseline_demo/scripts/make_video.py --run kitchen_12
D:\vggt311\Scripts\python.exe baseline_demo/scripts/package_results.py
```

`prepare.py` 选帧；`infer.py` 调用官方模型并导出几何和 COLMAP；`train.py` 读取实际 COLMAP 文件进行 Gaussian 优化；`make_video.py` 从真实参数渲染连续视角。

## 重新安装时的关键条件

1. Python 3.11；PyTorch 2.7.1+cu128 与 torchvision 0.22.1+cu128；其余依赖见 `requirements.lock.txt`。
2. 官方 VGGT 提交和 gsplat 1.5.3 提交见 `experiment_manifest.json`；应用 `vggt_pycolmap_compat.patch`，使用 pycolmap 3.11.1。GLM 对应提交也在 manifest 中。
3. 独立 CUDA 12.8 Toolkit 由 NVIDIA 官方 nvcc、cudart、CCCL redistributable 组合，保留在 D 盘；编译需要 Visual Studio 2022 C++ 工具。
4. 已编译 gsplat 可直接运行。若重编译：`baseline_demo/scripts/with_cuda.ps1 -m pip install --no-build-isolation --no-deps ./baseline_demo/vendor/gsplat`。换机器时先修改该脚本中的环境路径。
5. 官方权重保存在 `checkpoints/vggt_model.pt`。下载脚本支持分块续传；默认代理为本机 7897 端口，可用 `--proxy ''` 禁用。不要把未完成的下载当作有效权重。

## 保存的实验内容

- `data/kitchen/`：原始 RGB 选帧、索引和拼图；`data/desk/`：TUM 候选选帧。
- `outputs/kitchen_12/`：相机、深度、置信度、初始点云、COLMAP 文件、Gaussian 参数及标准零阶 3DGS PLY。
- `outputs/kitchen_4/`：4 视图前端试跑结果。
- `logs/`：下载、依赖安装、接口失败记录、成功推理、优化及视频生成日志。
- `outputs/kitchen_12/training_3500.json` 和 `gaussians_3500.pt`：第一轮优化记录；最终使用 12,000 步版本。
- `交付素材/`：MP4、GIF、PNG、中文使用说明。

当前的开放性探索、公开数据集定量评测和消融实验尚未执行。先完成本次核心基线，后续再围绕课题开展研究。
