"""Write Chinese provenance, reproducibility notes and a standard degree-0 GS PLY."""
from pathlib import Path
import json,subprocess,hashlib,platform
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/kitchen_12';d=ROOT/'交付素材'
inf=json.loads((out/'inference.json').read_text());train=json.loads((out/'training.json').read_text())
state=torch.load(out/'gaussians.pt',weights_only=True,map_location='cpu')
fields=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
values=np.concatenate([state['means'].numpy(),np.zeros((len(state['means']),3)),(state['colors'].sigmoid().numpy()-.5)/.28209479177387814,state['opacities'].numpy()[:,None],state['scales'].numpy(),state['quats'].numpy()],1)
arr=np.empty(len(values),dtype=[(key,'<f4') for key in fields])
for i,key in enumerate(fields):arr[key]=values[:,i]
with (out/'trained_gaussians.ply').open('wb') as f:
    header='ply\nformat binary_little_endian 1.0\n'+f'element vertex {len(arr)}\n'+''.join(f'property float {key}\n' for key in fields)+'end_header\n'
    f.write(header.encode());f.write(arr.tobytes())
commits={k:subprocess.check_output(['git','-C',str(ROOT/'vendor'/k),'rev-parse','HEAD'],text=True).strip() for k in ['vggt','gsplat']}
manifest=dict(vggt_commit=commits['vggt'],gsplat_commit=commits['gsplat'],glm_commit='33b4a621a697a305bc3a7610d290677b96beb181',checkpoint_repo_commit='860abec7937da0a4c03c41d3c269c366e82abdf9',checkpoint_sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0',checkpoint_sha256_verified_against_official_header=True,python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,source='https://github.com/facebookresearch/vggt/tree/'+commits['vggt']+'/examples/kitchen',inference=inf,training={k:v for k,v in train.items() if k!='history'})
(ROOT/'experiment_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
text=f'''# 多张照片 → 三维场景：开题展示说明

本次已在本机实际完成 **官方预训练 VGGT → 相机、深度与点云 → COLMAP → gsplat 单场景优化 → 三维漫游视频**。视频中的三维画面来自本次重建结果，不是官方成品视频，也不是生成式图生视频。

## 如何使用素材

- `第一步_交互点云.html`：双击用 Edge/Chrome 打开，可以旋转、缩放、平移第一步的真实点云；文件自带数据，可离线使用。
- `三维重建演示.mp4`：18 秒，1280×720、30 fps、H.264。0–3 秒为输入照片，3–6 秒为彩色点云和相机，6–18 秒为真实 Gaussian 渲染。推荐在开题 PPT 中展示这一版。
- `纯三维场景漫游.mp4`：12 秒，1280×720、30 fps，去掉流程侧栏，适合放进已有幻灯片版式。
- `三维重建演示.gif`：640×360、10 fps 的预览。正式投屏优先使用 MP4。
- `01_输入照片.png`、`02_点云与相机.png`、`03_渲染视角_*.png`：可自由排版的静态素材。输入拼图展示 6 张代表照片，实际输入为 12 张。
- `PPT展示帧_*.png`：带完整流程版式的 720p 静态画面。

现有开题 PPT 没有被修改。视频没有音轨，可配合现场讲解。

## 答辩时可以这样说

> 这里展示的是目前完成的小场景基线。我们使用官方预训练 VGGT，从 12 张 RGB 图像中预测相机与深度，生成初始点云，再用这些几何信息初始化并优化 3D Gaussian 场景。右侧视频是重建场景在连续视角下的实际渲染。这个实验初步验证了技术路线可以在本机运行，后续将进一步在公开数据集上开展定量评测与改进研究。

“不需要从头训练基础模型”是准确表述；整条流程并非完全无需优化，Gaussian 场景在本机做了 {train['steps']:,} 步优化。

## 实际数据和参数

| 项目 | 本次设置或结果 |
|---|---|
| 最终展示场景 | VGGT 官方仓库 `examples/kitchen`：桌面黄色积木工程车 |
| 输入 | 12 张 RGB；无真实深度、真实相机轨迹输入 |
| 几何模型 | 官方 `facebook/VGGT-1B`；权重未训练、未微调 |
| 模型预处理 / 原生训练和渲染尺寸 | 518×350 |
| 展示视频尺寸 | 1280×720，原生渲染画面经等比例放大排版；不代表原生 720p 纹理精度 |
| 点云与 Gaussian 数量 | 80,000 |
| 深度置信度过滤 | 阈值 {inf['confidence_threshold']}；剔除无效点及图像边缘 3 像素；固定随机种子 42 |
| Gaussian 参数 | 位置、各向异性尺度、旋转、不透明度、视角无关颜色均可优化 |
| 优化配置 | {train['steps']:,} 步；0.8×L1 + 0.2×局部 SSIM 损失；固定相机；无自适应增密 |
| 前端用时 | {inf['seconds']:.2f} 秒，含本地权重加载、推理与结果导出 |
| 场景优化用时 | {train['seconds']:.2f} 秒，计时从预热之后开始，含末尾训练视图渲染；不含安装、下载、编译及初始化 |
| PyTorch 前端分配峰值 | {inf['peak_allocated_gib']:.3f} GiB |
| PyTorch 优化阶段分配峰值 | {train['peak_allocated_gib']:.3f} GiB |
| 训练视图平均 PSNR | {train['training_view_psnr']:.2f} dB，仅作拟合诊断，不是独立测试集成绩 |
| 硬件 | NVIDIA RTX 4060 Laptop GPU，8 GB 标称显存；约 64 GB 系统内存 |
| 软件 | Windows；Python {platform.python_version()}；PyTorch {torch.__version__}；CUDA Toolkit 12.8；gsplat 1.5.3；pycolmap 3.11.1 |

显存数值是 `torch.cuda.max_memory_allocated()` 的统计，**不等于整卡显存占用**，不含驱动及所有外部库内存。Windows WDDM 下可能涉及共享内存，前端接近显存容量，应保留该口径说明。环境首次准备耗时远大于单场景计算，不能用上表用时代表从零安装的总耗时。

## 来源与可复现性

- [VGGT 官方代码与示例照片](https://github.com/facebookresearch/vggt/tree/{commits['vggt']}/examples/kitchen)
- [VGGT 官方预训练权重](https://huggingface.co/facebook/VGGT-1B)：下载完成后 SHA-256 与官方响应头一致。
- [gsplat 官方代码](https://github.com/nerfstudio-project/gsplat/tree/{commits['gsplat']})
- [TUM RGB-D 官方数据](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)：已下载 `freiburg1_desk` 并整理 RGB 选帧。本次展示改用画面更清楚、主体更直观的官方工程车示例，TUM 没有被冒充为视频数据来源。

代码提交、权重 SHA-256、环境版本、运行参数位于上一级 `experiment_manifest.json`、`requirements.lock.txt`。实际输入文件和官方原图之间的映射位于 `data/kitchen/selection.json`。

VGGT 官方 COLMAP 导出函数只做了 pycolmap 3.11 注册接口兼容修正，补丁保存在 `vggt_pycolmap_compat.patch`。Windows 中文路径由 Python 切换工作目录后向 COLMAP 传入相对英文路径解决。

## 已验证与尚未完成

- 完整视频：540 帧、18 秒、720p；另有 360 帧的纯渲染视频。已检查输入、点云、实际渲染来自同一组数据。
- COLMAP 导出：12 个注册相机，80,000 个点；抽查 1,000 个观测，导出重投影差异小于 0.001 像素。该数值是接口一致性检查，不是独立几何精度。
- 已检查多个插值视角。主体可辨识；背景仍有模糊、拉伸、少量漂浮结构和视角间细节变化，不能宣称已完全解决复杂场景重建问题。
- 本次为显存受控的简化 3DGS 基线：固定数量 Gaussian、零阶颜色、固定预测相机，没有执行论文完整的增密策略、球谐高阶外观或相机 BA。
- 未进行独立测试集 PSNR/SSIM/LPIPS、真实深度/位姿精度、SfM/MVS/NeRF/标准 3DGS 对比及消融；这些仍属于后续研究任务。
- 当前示例验证核心接口与展示效果，尚不能替代 TUM、DTU 等公开数据集的正式实验，也不构成创新方法结论。
'''
(d/'使用说明与实验记录.md').write_text(text,encoding='utf-8')
readme='''# 三维重建演示基线

成品见 [交付素材/使用说明与实验记录.md](交付素材/使用说明与实验记录.md)。本目录已完成官方 VGGT + gsplat 的小场景演示；不修改现有开题 PPT。

## 复现本次结果

以下命令在项目根目录的 PowerShell 中执行。已配置的独立环境为 `D:\\vggt311`，独立 CUDA 编译组件为 `D:\\vggt_cuda128`，不依赖基础 conda 环境。

```powershell
D:\\vggt311\\Scripts\\python.exe baseline_demo/scripts/prepare.py --scene kitchen --count 12
D:\\vggt311\\Scripts\\python.exe baseline_demo/scripts/infer.py --scene kitchen --count 12
D:\\vggt311\\Scripts\\python.exe baseline_demo/scripts/validate_scene.py --run kitchen_12
D:\\vggt311\\Scripts\\python.exe baseline_demo/scripts/train.py --run kitchen_12 --steps 12000
D:\\vggt311\\Scripts\\python.exe baseline_demo/scripts/make_video.py --run kitchen_12
D:\\vggt311\\Scripts\\python.exe baseline_demo/scripts/package_results.py
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
'''
(ROOT/'README.md').write_text(readme,encoding='utf-8')
print('Packaged',len(arr),'Gaussians and Chinese documentation')
