"""Official VGGT camera/depth inference and official COLMAP conversion."""
from pathlib import Path
import argparse, sys, time, json, gc, os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/vggt'))
import numpy as np
import torch
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
import trimesh

p=argparse.ArgumentParser();p.add_argument('--scene',default='desk');p.add_argument('--count',type=int,default=4);p.add_argument('--max-points',type=int,default=80000);p.add_argument('--confidence-threshold',type=float);p.add_argument('--reuse-geometry',action='store_true');p.add_argument('--run')
a=p.parse_args();torch.manual_seed(42);np.random.seed(42)
base=ROOT/'data'/a.scene
files=sorted((base/'originals').glob('*.png'))
files=[files[i] for i in np.linspace(0,len(files)-1,a.count).astype(int)]
out=ROOT/'outputs'/(a.run or f'{a.scene}_{a.count}');out.mkdir(parents=True,exist_ok=True)
t=time.perf_counter()
previous_info=None
if a.reuse_geometry:
    previous_info=json.loads((out/'inference.json').read_text())
    with np.load(out/'geometry.npz') as cached:
        ex=cached['extrinsics'];ki=cached['intrinsics'];depth=cached['depth'];conf=cached['confidence'];rgb=cached['images']
    assert len(rgb)==a.count
    peak=previous_info['peak_allocated_gib']
else:
    model=VGGT(enable_point=False,enable_track=False)
    state=torch.load(ROOT/'checkpoints/vggt_model.pt',map_location='cpu',weights_only=True,mmap=True)
    state={k:v for k,v in state.items() if not k.startswith(('point_head.','track_head.'))}
    model.load_state_dict(state,strict=True);del state;gc.collect()
    model.eval().cuda()
    images=load_and_preprocess_images([str(f) for f in files]).cuda()
    torch.cuda.reset_peak_memory_stats(); print('INFERENCE',images.shape,flush=True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        pred=model(images)
        ex,ki=pose_encoding_to_extri_intri(pred['pose_enc'],images.shape[-2:])
    ex=ex[0].float().cpu().numpy();ki=ki[0].float().cpu().numpy()
    depth=pred['depth'][0].float().cpu().numpy();conf=pred['depth_conf'][0].float().cpu().numpy()
    rgb=(images.cpu().permute(0,2,3,1).numpy()*255).astype(np.uint8)
    peak=torch.cuda.max_memory_allocated()/1024**3
    del model,pred,images;gc.collect();torch.cuda.empty_cache()
xyz=unproject_depth_map_to_point_map(depth,ex,ki)
n,h,w=conf.shape
yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
xyf=np.stack([np.broadcast_to(xx,(n,h,w)),np.broadcast_to(yy,(n,h,w)),np.broadcast_to(np.arange(n)[:,None,None],(n,h,w))],axis=-1)
threshold=a.confidence_threshold if a.confidence_threshold is not None else max(5,float(np.percentile(conf,35)))
mask=(conf>=threshold) & np.isfinite(xyz).all(-1) & (depth[...,0]>0)
mask[:,:3]=False;mask[:,-3:]=False;mask[:,:,:3]=False;mask[:,:,-3:]=False
idx=np.flatnonzero(mask);idx=np.random.choice(idx,min(len(idx),a.max_points),replace=False)
pts=xyz.reshape(-1,3)[idx];colors=rgb.reshape(-1,3)[idx];coords=xyf.reshape(-1,3)[idx]
np.savez_compressed(out/'geometry.npz',extrinsics=ex,intrinsics=ki,depth=depth,confidence=conf,images=rgb,points=pts,colors=colors)
trimesh.PointCloud(pts,colors).export(out/'initial_points.ply')
scene=out/'colmap';(scene/'images').mkdir(parents=True,exist_ok=True);(scene/'sparse/0').mkdir(parents=True,exist_ok=True)
recon=batch_np_matrix_to_pycolmap_wo_track(pts,coords,colors,ex,ki,np.array([w,h]),shared_camera=False,camera_type='PINHOLE')
for i in range(n):
    name=f'{i:03d}.png';recon.images[i+1].name=name;Image.fromarray(rgb[i]).save(scene/'images'/name)
previous_cwd=os.getcwd()
try:
    os.chdir(scene)
    recon.write('sparse/0')  # PyCOLMAP Windows file API requires an ASCII path.
finally:
    os.chdir(previous_cwd)
info=dict(scene=a.scene,views=n,input_files=[f.name for f in files],resolution=[w,h],points=len(pts),seconds=time.perf_counter()-t,peak_allocated_gib=peak,torch=torch.__version__,gpu=torch.cuda.get_device_name(),seed=42,confidence_threshold=threshold,ground_truth_used=False,reused_geometry=a.reuse_geometry)
if previous_info:info['original_frontend_seconds']=previous_info.get('original_frontend_seconds',previous_info['seconds'])
(out/'inference.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print(info,flush=True)
