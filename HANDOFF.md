# AI 工作交接

## 研究目标与进度

研究范围以根目录 AGENTS.md 和课题说明.md 为准。已完成可运行的 VGGT + gsplat 小场景简化基线。尚未完成公开数据集正式定量评测、几何精度评估、SfM/MVS、NeRF、标准 3DGS 对比，也未完成开放性改进及消融。

当前工程车来自 VGGT 官方 examples/kitchen，自采网球由用户确认物体静止、仅移动手机。不要将工程车结果称为 TUM 结果。仓库保留最终运行 kitchen_12 和 tennis_context_9；tennis_9 只保留旧相机路径供展示脚本使用，不是最终模型。

## 文件与复现环境

- scripts/infer.py：VGGT 相机与深度预测、反投影、置信度过滤、随机点采样、COLMAP 导出。
- scripts/train.py：读取导出的 COLMAP 与 geometry.npz，进行固定数量各向异性 Gaussian 优化；固定 VGGT 相机、零阶 RGB，L1 + 局部 SSIM，无自适应增密。
- scripts/validate_scene.py：实际导出几何与相机投影的一致性检查。
- scripts/evaluate_lpips.py：两个最终场景的训练视角配对评估。
- outputs/*/geometry.npz、gaussians.pt、colmap/：可直接检查最终状态或重新训练。
- data/kitchen/originals、data/tennis/originals：已上传的实际选帧输入；selection.json 保存来源。历史绝对路径仅作来源记录，不能作为新机器有效路径。

以下路径均相对于仓库根目录。原运行使用 Windows、Python 3.11.6、RTX 4060 Laptop GPU、PyTorch 2.7.1+cu128、CUDA 12.8、gsplat 1.5.3、pycolmap 3.11.1。VGGT 前端峰值 allocated 显存约 8 GiB，显存边界较紧。

requirements.lock.txt 和评估目录的 requirements.txt 是原环境快照，含本机 file:// 引用。新安装使用 requirements.txt；CUDA/PyTorch/gsplat 分步安装。scripts/with_cuda.ps1 仍记录原机器 D 盘及 VS2022 路径，换机器需要修改。绘图/视频脚本使用 C:/Windows/Fonts/msyh.ttc，Linux 需更换字体路径。新环境复现尚未验证，不保证不同 GPU 完全逐位一致。

.gitignore 对 baseline_demo 内的数据、输出和日志采用选择保留规则。新增实验的代码与文档可以放在根目录的研究子目录中；若要提交新的 data/outputs/logs 场景，请先在 .gitignore 中开放对应路径，只保留有必要的实验产物。

## 初始化第三方代码与安装

```powershell
git submodule update --init --recursive
python baseline_demo/scripts/setup_sources.py
py -3.11 -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r baseline_demo/requirements.txt
```

安装 CUDA 12.8 Toolkit 和 VS2022 C++ 工具，确保当前 shell 可找到 nvcc 与 cl，再执行：

```powershell
python -m pip install --no-build-isolation --no-deps ./baseline_demo/vendor/gsplat
```

Linux 安装对应 CUDA 与 C++ 编译器，安装 gsplat 命令相同。上游固定提交见 experiment_manifest.json 与 Git submodule。初始化脚本可重复执行，不覆盖其他本地修改。

## 权重下载

VGGT 官方模型固定版本：

https://huggingface.co/facebook/VGGT-1B/resolve/860abec7937da0a4c03c41d3c269c366e82abdf9/model.pt

保存到 baseline_demo/checkpoints/vggt_model.pt。SHA-256：

`d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`

```powershell
python baseline_demo/scripts/download.py https://huggingface.co/facebook/VGGT-1B/resolve/860abec7937da0a4c03c41d3c269c366e82abdf9/model.pt baseline_demo/checkpoints/vggt_model.pt --sha256 d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0
```

下载脚本默认代理 http://127.0.0.1:7897。没有该代理时使用自己的下载工具，或传入空字符串 --proxy 参数。不要上传多 GiB 的权重或未完成下载。LPIPS 的 AlexNet 权重由 torchvision/LPIPS 获取。

## 工程车完整运行

```powershell
python baseline_demo/scripts/prepare.py --scene kitchen --count 12
python baseline_demo/scripts/infer.py --scene kitchen --count 12
python baseline_demo/scripts/validate_scene.py --run kitchen_12
python baseline_demo/scripts/train.py --run kitchen_12 --steps 12000
python baseline_demo/scripts/make_video.py --run kitchen_12
```

重新运行会更新现有结果；如需保留发布版本，先复制 outputs/kitchen_12 或在新工作目录复现。原始结果是基于已提交的参数与版本获得的，不声称本次上传重新跑过训练。

## 网球最终配置

```powershell
python baseline_demo/scripts/infer.py --scene tennis --count 9 --run tennis_context_9 --max-points 200000 --confidence-threshold 1
python baseline_demo/scripts/validate_scene.py --run tennis_context_9
python baseline_demo/scripts/train.py --run tennis_context_9 --steps 12000
python baseline_demo/scripts/make_video.py --run tennis_context_9 --deliver 网球_交付素材 --label 用户实拍静态网球 --camera-order 5,7,2,8,6
python baseline_demo/scripts/evaluate_lpips.py
```

已保存的最终网球推理使用缓存几何重新过滤。inference.json 中 seconds 约 3.07 秒为缓存处理时间，original_frontend_seconds 约 28.56 秒才是原 VGGT 前端时间。没有权重时仍可基于仓库 geometry.npz 重新训练或评估，不必先运行 infer.py。

package_local_scene.py 依赖原机器 source 绝对路径及展示验证文件；跨机器直接运行前需调整。manifest 中 photos_uploaded=false 是原运行时的历史事实；本次交接已将 data/tennis/originals 上传，不能再理解成照片未进入 GitHub。

## 评估口径与已知问题

PSNR/LPIPS 都是训练视角拟合；训练前没有划分测试集，VGGT 前端也看过所有输入视图。当前几何投影检查不等于带真值的重建精度评估。网球阈值 1 和工程车阈值设置不同，不能据此构造方法优劣结论。

已知伪影包括背景漂浮、拉伸、模糊和网球插值视角重影。网球演示只走部分相机路径，不能作为完整环绕质量证明；全路径诊断参数保存在 outputs/tennis_context_9/diagnostics 中，原诊断视频未上传。训练中间图、最终渲染和失败与修正.md 可用于分析。

显存记录来自 torch.cuda.max_memory_allocated，不等于整卡或进程总显存；训练时间、推理时间、端到端时间应分别报告。

## 下一步优先级

1. 在目标环境运行一个现有场景，确认相机、COLMAP、Gaussian 优化与渲染链路。
2. 在 DTU、TUM RGB-D、Replica 或 ScanNet 中建立固定场景、视图数、分辨率和训练/测试划分；明确测试相机来源及尺度/坐标对齐。先明确评估协议，再避免测试视图进入训练和前端几何初始化。
3. 补 PSNR、SSIM、LPIPS、几何精度、耗时、显存，建立标准 3DGS 与适当 SfM/MVS、NeRF 对照。
4. 在可运行基线上研究深度置信度 Gaussian 初始化，控制视图、分辨率、训练步数、Gaussian 数量及随机种子，做阈值/采样策略消融。
5. 中文记录参数、日志、失败案例与结果，不把已有场景之间的分数差异当作消融。
