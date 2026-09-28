"""Create a presentation-ready movie from actual RGB, point cloud and 3DGS renders."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
import cv2
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation,Slerp
from gsplat import rasterization
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',default='desk_12');p.add_argument('--deliver',default='交付素材');p.add_argument('--label',default='VGGT 官方工程车示例');p.add_argument('--camera-order',default='');a=p.parse_args()
out=ROOT/'outputs'/a.run;deliver=ROOT/a.deliver;deliver.mkdir(parents=True,exist_ok=True)
g=np.load(out/'geometry.npz');meta=json.loads((out/'inference.json').read_text());train=json.loads((out/'training.json').read_text())
state=torch.load(out/'gaussians.pt',map_location='cuda',weights_only=True)
images=g['images'];n,h,w,_=images.shape
cams=np.tile(np.eye(4),(n,1,1));cams[:,:3]=g['extrinsics'];c2w=np.linalg.inv(cams)
order=[int(v) for v in a.camera_order.split(',')] if a.camera_order else list(range(n))
assert len(order)>=2 and all(0<=i<n for i in order)
path_cams=c2w[order];path_ks=g['intrinsics'][order];path_n=len(order)
rot=Slerp(np.arange(path_n),Rotation.from_matrix(path_cams[:,:3,:3]));fps=30;duration=18
fontpath='C:/Windows/Fonts/msyh.ttc'
font=lambda s:ImageFont.truetype(fontpath,s)
fonts={s:font(s) for s in [14,17,20,24,28,36,42]}
BG='#0d1520';FG='#eff4f8';MUTED='#9caec0';ACC='#77e5c1'
def base(stage,subtitle):
    im=Image.new('RGB',(1280,720),BG);d=ImageDraw.Draw(im)
    d.text((38,30),'多视图三维重建',font=fonts[28],fill=FG)
    d.text((38,74),'VGGT  →  3D Gaussian Splatting',font=fonts[14],fill=MUTED)
    d.line((38,118,1242,118),fill='#2b3b4c',width=1)
    steps=['输入 RGB 图像','恢复三维结构','连续视角渲染']
    for j,s in enumerate(steps):
        y=182+j*106;d.text((38,y),f'0{j+1}',font=fonts[20],fill=ACC if stage==j else MUTED)
        d.text((38,y+34),s,font=fonts[24],fill=FG if stage==j else MUTED)
    d.text((38,530),f'{n} 张 RGB · 单场景',font=fonts[17],fill=ACC)
    d.text((38,560),'官方预训练 VGGT',font=fonts[17],fill=MUTED)
    d.text((38,590),'本机运行与场景优化',font=fonts[17],fill=MUTED)
    d.text((38,672),subtitle,font=fonts[14],fill=MUTED)
    return im
def fitted(im,box):
    scale=min((box[2]-box[0])/im.width,(box[3]-box[1])/im.height)
    im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
    return im,(box[0]+(box[2]-box[0]-im.width)//2,box[1]+(box[3]-box[1]-im.height)//2)
def place(im,content):
    pic,pos=fitted(content,(304,136,1242,654));im.paste(pic,pos);return im
sheet=Image.new('RGB',(960,540),BG);d=ImageDraw.Draw(sheet)
for j,idx in enumerate(np.linspace(0,n-1,min(6,n)).astype(int)):
    x=(j%3)*320;y=(j//3)*270
    pic,pos=fitted(Image.fromarray(images[idx]),(x+5,y+5,x+315,y+239));sheet.paste(pic,pos)
    d.text((x+12,y+245),f'VIEW {idx+1:02d} / {n:02d}',font=fonts[14],fill=MUTED)
sheet.save(deliver/'01_输入照片.png')
pts=g['points'];colors=g['colors'];center=np.median(pts,axis=0)
span=np.linalg.norm(np.percentile(pts,90,axis=0)-np.percentile(pts,10,axis=0))
def point_frame(t):
    # An overview camera derived from the first recovered camera, with modest orbit.
    R=c2w[0,:3,:3].copy();angle=.18*np.sin(2*np.pi*t)
    R=R@Rotation.from_euler('y',angle).as_matrix()
    eye=center-R[:,2]*span*1.0
    pc=(pts-eye)@R;z=pc[:,2]
    px=pc[:,0]/np.maximum(z,.001)*660+480;py=pc[:,1]/np.maximum(z,.001)*660+270
    mask=(z>.01)&(px>=1)&(px<959)&(py>=1)&(py<539)
    idx=np.flatnonzero(mask);idx=idx[np.argsort(z[idx])[::-1]]
    canvas=np.zeros((540,960,3),dtype=np.uint8);canvas[:]=[13,21,32]
    xx=px[idx].astype(int);yy=py[idx].astype(int)
    canvas[yy,xx]=colors[idx]
    # 2-pixel points for clear structure at presentation resolution.
    canvas[yy+1,xx]=colors[idx];canvas[yy,xx+1]=colors[idx]
    def project(q):
        q=(q-eye)@R
        if np.any(q[:,2]<=.01):return None
        return np.stack([q[:,0]/q[:,2]*660+480,q[:,1]/q[:,2]*660+270],-1).astype(int)
    for c in c2w:
        size=span*.027
        corners=np.array([[-1,-.75,1.5],[1,-.75,1.5],[1,.75,1.5],[-1,.75,1.5]])*size
        q=np.concatenate([c[None,:3,3],corners@c[:3,:3].T+c[:3,3]],0);q=project(q)
        if q is not None:
            for j in range(4):cv2.line(canvas,tuple(q[0]),tuple(q[j+1]),(119,229,193),1,cv2.LINE_AA);cv2.line(canvas,tuple(q[j+1]),tuple(q[(j+1)%4+1]),(119,229,193),1,cv2.LINE_AA)
    return Image.fromarray(canvas)
point_frame(.15).save(deliver/'02_点云与相机.png')
def gauss_frame(t):
    u=(1-np.cos(2*np.pi*t))/2*(path_n-1)
    idx=min(int(u),path_n-2);f=u-idx
    c=np.eye(4);c[:3,:3]=rot(u).as_matrix();c[:3,3]=(1-f)*path_cams[idx,:3,3]+f*path_cams[idx+1,:3,3]
    k=(1-f)*path_ks[idx]+f*path_ks[idx+1]
    with torch.inference_mode():
        rgb,_,_=rasterization(means=state['means'],quats=state['quats'],scales=state['scales'].exp(),opacities=state['opacities'].sigmoid(),colors=state['colors'].sigmoid(),viewmats=torch.tensor(np.linalg.inv(c)[None],device='cuda',dtype=torch.float32),Ks=torch.tensor(k[None],device='cuda',dtype=torch.float32),width=w,height=h,packed=False,near_plane=.01,far_plane=100.)
    return Image.fromarray((rgb[0].clamp(0,1).cpu().numpy()*255).astype('uint8'))
for j,t in enumerate([0,.18,.32]):gauss_frame(t).save(deliver/f'03_渲染视角_{j+1}.png')
gif=[];t0=time.perf_counter()
writer=imageio.get_writer(str(deliver/'三维重建演示.mp4'),fps=fps,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1,ffmpeg_log_level='error')
clean_writer=imageio.get_writer(str(deliver/'纯三维场景漫游.mp4'),fps=fps,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1,ffmpeg_log_level='error')
for frame in range(duration*fps):
    sec=frame/fps
    if sec<3:
        im=place(base(0,f'{a.label} · 展示代表照片，实际输入 {n} 张'),sheet)
    elif sec<6:
        im=place(base(1,'VGGT 预测相机与深度 · 彩色点云由预测深度反投影得到'),point_frame((sec-3)/3))
    else:
        rendered=gauss_frame((sec-6)/12)
        im=place(base(2,f'实际 3DGS 渲染 · 优化 {train["steps"]} 步 · 原生 {w}×{h}，排版输出 1280×720'),rendered)
        clean=Image.new('RGB',(1280,720),BG);pic,pos=fitted(rendered,(0,0,1280,720));clean.paste(pic,pos)
        clean_writer.append_data(np.array(clean))
    draw=ImageDraw.Draw(im);draw.rectangle((0,714,int(1280*(frame+1)/(duration*fps)),719),fill=ACC)
    writer.append_data(np.array(im))
    if frame%3==0:gif.append(im.resize((640,360),Image.Resampling.LANCZOS))
    if frame in [45,135,315]:im.save(deliver/f'PPT展示帧_{frame:03d}.png')
    if frame%90==0:print('FRAME',frame,flush=True)
writer.close()
clean_writer.close()
gif[0].save(deliver/'三维重建演示.gif',save_all=True,append_images=gif[1:],duration=100,loop=0,optimize=False)
reader=imageio.get_reader(str(deliver/'三维重建演示.mp4'));vm=reader.get_meta_data();count=reader.count_frames();reader.close()
assert count==540 and vm['size']==(1280,720)
info=dict(frames=count,fps=vm['fps'],duration=vm['duration'],size=vm['size'],native_render_size=[w,h],generation_seconds=time.perf_counter()-t0,camera_order_zero_based=order,source_label=a.label)
(deliver/'video_validation.json').write_text(json.dumps(info,indent=2));print(info,flush=True)
