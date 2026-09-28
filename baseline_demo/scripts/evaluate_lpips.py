"""Evaluate paired saved RGB renders with official calibrated LPIPS-Alex v0.1.

All evaluated images are training views. No held-out or video-frame claims.
"""
from pathlib import Path
import csv,hashlib,importlib.metadata,json,platform,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch,torchvision,lpips

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'评估结果/LPIPS';OUT.mkdir(parents=True,exist_ok=True)
torch.manual_seed(42);np.random.seed(42)
torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
device='cuda' if torch.cuda.is_available() else 'cpu'
metric=lpips.LPIPS(net='alex',version='0.1',lpips=True,pretrained=True,pnet_rand=False,spatial=False,eval_mode=True).to(device).eval()
def tensor(rgb):
    return torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1).unsqueeze(0).to(device,dtype=torch.float32)/127.5-1.
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()
def score(x,y):return float(metric(x,y,normalize=False).item())
font_path='C:/Windows/Fonts/msyh.ttc'
font=lambda size:ImageFont.truetype(font_path,size)
all_rows=[];summaries=[];sanity=[]
for title,run in [('工程车','kitchen_12'),('网球','tennis_context_9')]:
    folder=ROOT/'outputs'/run
    info=json.loads((folder/'inference.json').read_text())
    train=json.loads((folder/'training.json').read_text())
    selection=json.loads((ROOT/'data'/info['scene']/'selection.json').read_text(encoding='utf-8'))
    source_map={item['file']:item['source'] for item in selection}
    with np.load(folder/'geometry.npz') as g:refs=g['images']
    assert len(refs)==info['views']
    assert len(list(folder.glob('render_*.png')))==len(refs)
    if device=='cuda':torch.cuda.synchronize()
    started=time.perf_counter();rows=[]
    with torch.inference_mode():
        x=tensor(refs[0]);self_distance=score(x,x);black_distance=score(x,torch.full_like(x,-1.))
        assert abs(self_distance)<1e-6 and black_distance>self_distance+1e-3
        sanity.append(dict(scene=title,identical_images=self_distance,reference_vs_black=black_distance))
        for i,ref in enumerate(refs):
            render_file=folder/f'render_{i:03d}.png';reference_file=folder/'colmap/images'/f'{i:03d}.png'
            with Image.open(reference_file) as im:reference=np.array(im.convert('RGB'))
            with Image.open(render_file) as im:pred=np.array(im.convert('RGB'))
            assert np.array_equal(reference,ref),'Exported reference differs from actual optimization input'
            assert pred.shape==ref.shape
            distance=score(tensor(pred),tensor(ref));assert np.isfinite(distance) and distance>=-1e-6
            mse=np.mean(((pred.astype(np.float64)-ref.astype(np.float64))/255.)**2)
            row=dict(scene=title,run=run,view_index_zero_based=i,input_name=info['input_files'][i],source_image=source_map[info['input_files'][i]],render_path=str(render_file.relative_to(ROOT)),reference_path=str(reference_file.relative_to(ROOT)),width=ref.shape[1],height=ref.shape[0],lpips_alex_v01=distance,psnr_png_db=float(-10*np.log10(mse)) if mse>0 else None,split='training',render_sha256=digest(render_file),reference_sha256=digest(reference_file))
            rows.append(row);all_rows.append(row);print(title,i,f'LPIPS={distance:.6f}',flush=True)
    if device=='cuda':torch.cuda.synchronize()
    values=np.array([r['lpips_alex_v01'] for r in rows]);order=np.argsort(values)
    summary=dict(scene=title,run=run,views=len(rows),resolution=[refs.shape[2],refs.shape[1]],lpips_mean=float(values.mean()),lpips_median=float(np.median(values)),lpips_std_population=float(values.std()),lpips_min=float(values.min()),lpips_max=float(values.max()),lowest_view_zero_based=int(order[0]),highest_view_zero_based=int(order[-1]),mean_psnr_png_db=float(np.mean([r['psnr_png_db'] for r in rows])),gaussians=train['gaussians'],optimization_steps=train['steps'],split='training',evaluation_seconds=time.perf_counter()-started)
    summaries.append(summary)
    # Visual audit: lowest, median-ranked and highest LPIPS views, same full frame.
    chosen=[int(order[0]),int(order[len(order)//2]),int(order[-1])]
    sheet=Image.new('RGB',(1120,1240),'#0d1520');draw=ImageDraw.Draw(sheet)
    draw.text((25,18),f'{title}：训练视角 LPIPS-Alex v0.1',font=font(28),fill='#eef4fa')
    draw.text((25,61),'左：原始输入经 VGGT 预处理后的图像    右：同一相机的实际渲染',font=font(19),fill='#b5c7d8')
    for j,(idx,label) in enumerate(zip(chosen,['最低分','中间排名','最高分'])):
        y=105+j*372;r=rows[idx]
        draw.text((25,y),f'{label} · 输入 {r["input_name"]} · LPIPS = {r["lpips_alex_v01"]:.4f}',font=font(22),fill='#77e5c1')
        for k,array in enumerate([refs[idx],np.array(Image.open(folder/f'render_{idx:03d}.png').convert('RGB'))]):
            im=Image.fromarray(array);im.thumbnail((528,320),Image.Resampling.LANCZOS);sheet.paste(im,(25+k*555+(528-im.width)//2,y+38))
    draw.text((25,1210),'仅评价训练图像的拟合；不能据此判断漫游新视角或三维几何精度。',font=font(17),fill='#b5c7d8')
    sheet.save(OUT/f'{title}_LPIPS原图渲染对照.png')

backbone=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
linear=Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
manifest=dict(metric='LPIPS',backbone='AlexNet',calibration_version='0.1',learned_linear_weights=True,random_backbone=False,spatial=False,device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,lpips_package=importlib.metadata.version('lpips'),precision='float32, no autocast, TF32 disabled',normalization='RGB uint8 / 127.5 - 1; NCHW; normalize=False',preprocessing='No extra resize, alignment, masks or crop. Full saved native frame. Compared with the exact preprocessing output used for optimization.',split='training views only',aggregation='Arithmetic mean of per-view scalar LPIPS. Population std; no combined cross-scene average.',backbone_sha256=digest(backbone),linear_weights_sha256=digest(linear),backbone_weights_path=str(backbone),linear_weights_path=str(linear),script_sha256=digest(__file__),sanity_checks=sanity,results=summaries)
(OUT/'results.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'per_view.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
(OUT/'per_view.json').write_text(json.dumps(all_rows,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 工程车与网球：LPIPS 渲染拟合评估','', '**评估口径：训练视角。所有参考照片均参与过 Gaussian 场景优化，没有独立测试集。**','',
'LPIPS 使用预训练视觉网络的特征来衡量两张图像的感知差异，数值越低表示越接近；它不是相似度百分比，也没有适用于所有任务的固定合格线。本次使用官方 LPIPS-Alex v0.1，官方 Python 包版本 0.1.4。','',
'| 场景 | 配对数量 | 原生分辨率 | LPIPS 平均 ↓ | 中位数 ↓ | 最小–最大 | Gaussian 数量 |',
'|---|---:|---|---:|---:|---|---:|']
for s in summaries:lines.append(f'| {s["scene"]} | {s["views"]} | {s["resolution"][0]}×{s["resolution"][1]} | **{s["lpips_mean"]:.4f}** | {s["lpips_median"]:.4f} | {s["lpips_min"]:.4f}–{s["lpips_max"]:.4f} | {s["gaussians"]:,} |')
lines += ['', '## 怎样理解', '',
'以上数值评价“在原拍摄相机位置，渲染结果能多好地拟合输入照片”。网球与工程车属于不同场景，分辨率、视图数、Gaussian 数量和初始化设置不同，因此不能把分数差异解释为某个算法改进带来的提升，也不能单凭这两个分数判定哪个三维模型更准确。', '',
'漫游视频的中间视角没有同视角真实照片，不能直接计算有参考意义的 LPIPS；尤其不能把不同角度的照片或 GIF 帧随意配对。训练视角低 LPIPS 仍可能与新视角重影并存，网球就是需要注意这种现象的例子。','',
'正式评价新视角质量，需要在优化前划分训练图像和测试图像，重新训练只使用训练图像的场景，再在测试相机下渲染并与测试照片比较。若要评价整条 RGB 重建系统的泛化，VGGT 几何前端也应遵守相应划分，明确测试相机如何获得，避免把测试视图泄漏进几何初始化。', '',
'## 计算方式与核查', '',
'- 工程车使用 `outputs/kitchen_12/render_000.png` 至 `render_011.png`，网球使用最终版 `outputs/tennis_context_9/render_000.png` 至 `render_008.png`。',
'- 参考图是对应 `colmap/images` 中的 RGB 图；逐像素检查其与 `geometry.npz` 中实际优化输入完全一致。没有使用带文字、边框的视频或 GIF。',
'- 不做额外缩放、裁剪、对齐或前景掩码；评价整个画面，包含背景。与原始高分辨率照片相比，参考图已经过原流程的 VGGT 预处理。',
'- 输入按 RGB、NCHW、float32 转为 [-1,1]，使用预训练 AlexNet 和官方学习得到的线性校准权重，关闭梯度、autocast 和 TF32。每个视角单独评分后取等权平均。',
'- 原图与自身的 LPIPS 应接近 0；原图与全黑图应明显大于 0。这两项 sanity check 已通过，具体值见 `results.json`。',
'- `per_view.csv` 和 `per_view.json` 保存每对图像的分数、来源和 SHA-256；`results.json` 保存模型权重哈希、依赖版本、参数和汇总。CSV 附带从同一对 PNG 计算的 PSNR，仅作辅助诊断；8 位量化使其可能与训练时浮点 PSNR 略有不同。',
'- `工程车_LPIPS原图渲染对照.png` 与 `网球_LPIPS原图渲染对照.png` 展示各场景最低分、中间排名、最高分的配对图。','',
'## 复现','', '在项目根目录运行：','', '```powershell', 'D:\\vggt311\\Scripts\\python.exe baseline_demo/scripts/evaluate_lpips.py','```','',
'使用时在表格标题或注释中保留“训练视角、LPIPS-Alex v0.1、越低越好”，不要标为独立测试集或新视角评测结果。','',
'官方实现：https://github.com/richzhang/PerceptualSimilarity','论文：The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018。']
(OUT/'LPIPS评估报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)
