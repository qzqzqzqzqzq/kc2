"""Small-scene 3DGS optimization using the official gsplat rasterizer.

Fixed-count anisotropic Gaussians (no adaptive densification), degree-0 color.
This is a resource-constrained baseline, not the paper's full training recipe.
"""
from pathlib import Path
import argparse,json,time,os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from PIL import Image
from gsplat import rasterization
import pycolmap
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',default='desk_12');p.add_argument('--steps',type=int,default=3500)
a=p.parse_args();out=ROOT/'outputs'/a.run
torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(6)
g=np.load(out/'geometry.npz');images=torch.tensor(g['images']/255,dtype=torch.float32,device='cuda')
# Consume the actual exported COLMAP cameras and point cloud.
previous_cwd=os.getcwd()
try:
    os.chdir(out/'colmap');r=pycolmap.Reconstruction('sparse/0')
finally:
    os.chdir(previous_cwd)
ids=sorted(r.points3D);pts=np.array([r.points3D[i].xyz for i in ids]);cols=np.array([r.points3D[i].color for i in ids])/255
ex=[];ks=[]
for i in sorted(r.images):
    im=r.images[i];mat=np.eye(4);mat[:3]=im.cam_from_world.matrix();ex.append(mat);ks.append(r.cameras[im.camera_id].calibration_matrix())
view=torch.tensor(np.array(ex),dtype=torch.float32,device='cuda');K=torch.tensor(np.array(ks),dtype=torch.float32,device='cuda')
dist=cKDTree(pts).query(pts,k=4)[0][:,1:];sc=np.sqrt((dist**2).mean(1)).clip(1e-5)
extent=float(np.linalg.norm(np.percentile(pts,90,axis=0)-np.percentile(pts,10,axis=0)))
def param(x):return torch.nn.Parameter(torch.tensor(x,dtype=torch.float32,device='cuda'))
params=torch.nn.ParameterDict(dict(means=param(pts),scales=param(np.log(sc[:,None]*np.ones((1,3)))),quats=param(np.tile([1.,0.,0.,0.],(len(pts),1))),opacities=param(np.full(len(pts),-2.2)),colors=param(np.log(cols.clip(.001,.999)/(1-cols.clip(.001,.999))))))
rates=dict(means=extent*.00012,scales=.004,quats=.001,opacities=.03,colors=.01)
opts={k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in params.items()}
n,h,w,_=images.shape
def render(idx):
    return rasterization(means=params['means'],quats=params['quats'],scales=params['scales'].exp(),opacities=params['opacities'].sigmoid(),colors=params['colors'].sigmoid(),viewmats=view[idx:idx+1],Ks=K[idx:idx+1],width=w,height=h,packed=False,near_plane=.01,far_plane=100.)[0]
def ssim(x,y):
    x=x.permute(0,3,1,2);y=y.permute(0,3,1,2)
    avg=lambda z:F.avg_pool2d(z,7,1,3)
    mx,my=avg(x),avg(y);vx=avg(x*x)-mx*mx;vy=avg(y*y)-my*my;cov=avg(x*y)-mx*my
    return (((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean()
# Forces compilation and tests both CUDA forward/backward before the full run.
test=render(0);test.mean().backward()
for opt in opts.values():opt.zero_grad(set_to_none=True)
print('CUDA_FORWARD_BACKWARD_OK',len(pts),flush=True)
t=time.perf_counter();torch.cuda.reset_peak_memory_stats();history=[]
for step in range(a.steps):
    idx=int(np.random.randint(n));pred=render(idx);gt=images[idx:idx+1]
    loss=.8*(pred-gt).abs().mean()+.2*(1-ssim(pred,gt))
    loss.backward()
    for k,opt in opts.items():
        if k=='means':opt.param_groups[0]['lr']=rates[k]*(.1**(step/a.steps))
        opt.step();opt.zero_grad(set_to_none=True)
    if step%100==0 or step==a.steps-1:
        rec=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-t);history.append(rec);print(rec,flush=True)
    if step in [0,499,1499,a.steps-1]:
        with torch.no_grad():im=render(n//2)[0].clamp(0,1).cpu().numpy()
        Image.fromarray((im*255).astype('uint8')).save(out/f'train_{step+1:05d}.png')
torch.save({k:v.detach().cpu() for k,v in params.items()},out/'gaussians.pt')
psnrs=[]
with torch.no_grad():
    for i in range(n):
        pred=render(i)[0].clamp(0,1);psnrs.append(float(-10*torch.log10(((pred-images[i])**2).mean())))
        Image.fromarray((pred.cpu().numpy()*255).astype('uint8')).save(out/f'render_{i:03d}.png')
info=dict(steps=a.steps,gaussians=len(pts),seconds=time.perf_counter()-t,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,training_view_psnr=float(np.mean(psnrs)),psnr_per_view=psnrs,held_out_evaluation=False,recipe='fixed-count anisotropic 3DGS; RGB degree 0; L1 + local SSIM; fixed VGGT cameras',learning_rates=rates,history=history)
(out/'training.json').write_text(json.dumps(info,indent=2));print(info,flush=True)
