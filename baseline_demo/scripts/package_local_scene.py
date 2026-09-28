"""Package the local tennis reconstruction with explicit limitations and provenance."""
from pathlib import Path
import argparse,json,subprocess,platform,hashlib
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',default='tennis_context_9');p.add_argument('--deliver',default='网球_交付素材');a=p.parse_args()
out=ROOT/'outputs'/a.run;d=ROOT/a.deliver;d.mkdir(parents=True,exist_ok=True)
g=np.load(out/'geometry.npz');inf=json.loads((out/'inference.json').read_text());train=json.loads((out/'training.json').read_text());path=json.loads((ROOT/'outputs/tennis_9/path.json').read_text())
video=json.loads((d/'video_validation.json').read_text())
path['full_diagnostic_order_zero_based']=path['camera_order_zero_based']
path['camera_order_zero_based']=video['camera_order_zero_based']
path['camera_order_source_files']=[f'{i+1}.png' for i in path['camera_order_zero_based']]
state=torch.load(out/'gaussians.pt',map_location='cpu',weights_only=True)
fields=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
values=np.concatenate([state['means'].numpy(),np.zeros((len(state['means']),3)),(state['colors'].sigmoid().numpy()-.5)/.28209479177387814,state['opacities'].numpy()[:,None],state['scales'].numpy(),state['quats'].numpy()],1)
arr=np.empty(len(values),dtype=[(f,'<f4') for f in fields])
for i,f in enumerate(fields):arr[f]=values[:,i]
with (out/'trained_gaussians.ply').open('wb') as f:
    f.write(('ply\nformat binary_little_endian 1.0\n'+f'element vertex {len(arr)}\n'+''.join(f'property float {k}\n' for k in fields)+'end_header\n').encode());f.write(arr.tobytes())
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
sheet=Image.new('RGB',(1200,960),'#0d1520');draw=ImageDraw.Draw(sheet)
for i in range(9):
    im=Image.open(out/f'render_{i:03d}.png');im.thumbnail((390,295));x=i%3*400;y=i//3*320;sheet.paste(im,(x,y));draw.text((x+8,y+296),f'照片 {i+1} 对应的实际渲染',font=font,fill='#dce8f0')
sheet.save(d/'04_九个拍摄视角渲染.png')
selection=json.loads((ROOT/'data/tennis/selection.json').read_text(encoding='utf-8'))
for item in selection:
    assert hashlib.sha256(Path(item['source']).read_bytes()).hexdigest()==item['source_sha256']
manifest=dict(scene='用户实拍静态网球',run=a.run,source_files=selection,static_scene_confirmed_by_user=True,photos_uploaded=False,inference=inf,training={k:v for k,v in train.items() if k!='history'},camera_path=path,python=platform.python_version(),torch=torch.__version__,vggt_commit=subprocess.check_output(['git','-C',str(ROOT/'vendor/vggt'),'rev-parse','HEAD'],text=True).strip(),gsplat_commit=subprocess.check_output(['git','-C',str(ROOT/'vendor/gsplat'),'rev-parse','HEAD'],text=True).strip())
(out/'experiment_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
text=f'''# 你的 9 张网球照片：重建结果

本次输入来自项目根目录 `网球/1.png` 到 `网球/9.png`。用户确认球的位置和朝向不变，仅移动手机。原始照片保持不变，全部重建在本机运行。

## 成品对应哪一步

- **第一步：`第一步_交互点云.html`**。用 Edge 或 Chrome 打开，左键旋转、滚轮缩放、右键平移。文件内含全部真实初始点云，可离线使用。默认隐藏置信度低于 1.01 的点，勾选“显示低置信度背景点”可显示全部 20 万个点；仅为查看过滤，不修改训练结果。PNG 展示完整初始点云的一个视角。
- **第二步：`纯三维场景漫游.mp4`**。12 秒的实际 Gaussian 场景渲染。
- **完整过程：`三维重建演示.mp4`**。18 秒，前 3 秒显示原图代表帧，中间 3 秒显示点云，后 12 秒展示真实三维渲染。
- `三维重建演示.gif` 为预览；`04_九个拍摄视角渲染.png` 为 9 个输入相机对应的渲染结果，便于核对。
- 原始三维结果保存在 `../outputs/{a.run}/initial_points.ply`、`trained_gaussians.ply`、`gaussians.pt`，同时保留相机、深度、置信度与 COLMAP 文件。

## 本次实际流程与参数

1. 按 1–9 编号读取全部 9 张照片；归一化 EXIF 朝向，转换 RGB；VGGT 官方预处理到 518×392。
2. 使用同一份官方 VGGT-1B 权重预测深度和相机，无模型训练或微调。
3. 初始阈值 5 删掉了全部点：这组数据的深度置信度最大只有约 2.67。保留这个失败记录，没有把空点云误作有效重建。
4. 阈值 1.01 的 8 万点版本能重建球体，但丢掉了大量桌面背景，渲染出现拉伸。记录保存在 `outputs/tennis_9`，训练视图平均 PSNR 为 18.06 dB。
5. 最终演示版本使用阈值 1.0：保留有效深度，不再按置信度排除低分区域，随机选 {inf['points']:,} 个初始点。它们初始化同等数量 Gaussian，使用全部 RGB 照片和固定 VGGT 相机进行 {train['steps']:,} 步优化。

这是针对这组照片的演示初始化设置调整；低置信度点不能因被保留下来就视为高精度几何，也不是已经验证的通用改进方法。

| 项目 | 实际记录 |
|---|---|
| 输入视图 | 9 |
| 原生训练 / 渲染分辨率 | 518×392 |
| 初始点 / Gaussian | {inf['points']:,} |
| 置信度阈值 | {inf['confidence_threshold']}，保留低置信度有效点用于演示 |
| VGGT 原始前端阶段 | {inf['original_frontend_seconds']:.2f} 秒 |
| 最终版本重新过滤与导出 | {inf['seconds']:.2f} 秒，复用同一份预测，不重跑神经网络 |
| 最终场景优化阶段 | {train['seconds']:.2f} 秒，预热之后计时，不含初始化 |
| 前端 PyTorch 分配峰值 | {inf['peak_allocated_gib']:.3f} GiB |
| 优化 PyTorch 分配峰值 | {train['peak_allocated_gib']:.3f} GiB |
| 训练视图平均 PSNR | {train['training_view_psnr']:.2f} dB，仅反映输入视图拟合 |
| 硬件 | RTX 4060 Laptop，8 GB 标称显存 |
| 视频 | 1280×720、30 fps，原生图像等比例放大排版，无音轨 |

显存是 `torch.cuda.max_memory_allocated()` 的统计，不等同于整卡专用显存。Windows WDDM 可能使用共享内存，当前前端已接近或超过标称专用显存容量。

## 相机路径和效果边界

完整路径依据恢复相机的空间方位排列，但实测跨越部分大间隔时出现明显重影，完整测试保存在 `outputs/{a.run}/diagnostics/全路径测试_存在重影.mp4`。正式视频选取相对稳定的一段：**{' → '.join(path['camera_order_source_files'])}**。训练仍使用全部 9 张照片；只缩小展示相机路径，不删除训练输入。相邻相机间采用旋转插值和位置插值，沿路径往返，不宣称完整 360° 漫游。

照片中桌面有明显反光，球体局部纹理相似，相邻部分视角间距较大。球体、白色接缝和文字可辨识；背景和插值视角仍可能有模糊、拉伸或细节漂移。训练视图清楚不能证明未见视角准确，也不能证明球体尺寸或几何误差达标。

没有使用真实深度或位姿，没有独立测试集、精度对比或消融。本文中两个初始化版本同时改变过滤设置和点数，仅作为失败排查与演示选择，不作为单变量消融结论。这个结果属于个人实拍数据的可行性测试；公开数据集实验仍需另做。

## 复现

在项目根目录 PowerShell 中执行，使用已安装的独立环境：

```powershell
D:\\vggt311\\Scripts\\python.exe baseline_demo/scripts/prepare_local.py --source 网球 --scene tennis
D:\\vggt311\\Scripts\\python.exe baseline_demo/scripts/infer.py --scene tennis --count 9 --run tennis_context_9 --confidence-threshold 1.0 --max-points 200000
D:\\vggt311\\Scripts\\python.exe baseline_demo/scripts/train.py --run tennis_context_9 --steps 12000
D:\\vggt311\\Scripts\\python.exe baseline_demo/scripts/make_point_viewer.py --run tennis_context_9 --deliver 网球_交付素材 --label 用户实拍网球 --confidence-cut 1.01
D:\\vggt311\\Scripts\\python.exe baseline_demo/scripts/make_video.py --run tennis_context_9 --deliver 网球_交付素材 --label 用户实拍网球 --camera-order {','.join(map(str,path['camera_order_zero_based']))}
```

首次运行与当前缓存复用的计时口径不同。随机种子固定为 42，CUDA 并行计算仍可能造成数值差异。软件版本继承 `requirements.lock.txt`，来源哈希、实际参数和提交版本记录在 `outputs/{a.run}/experiment_manifest.json`，日志保存在 `logs/*tennis*`。
'''
(d/'网球重建说明.md').write_text(text,encoding='utf-8')
print('Packaged',a.run,train['training_view_psnr'])
