"""Self-contained WebGL viewer for the actual first-stage point cloud."""
from pathlib import Path
import base64,argparse,os,struct
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--run',default='kitchen_12');parser.add_argument('--deliver',default='交付素材');parser.add_argument('--label',default='VGGT 官方工程车示例');parser.add_argument('--confidence-cut',type=float);a=parser.parse_args()
g=np.load(ROOT/'outputs'/a.run/'geometry.npz')
p=g['points'].astype('float32');total_points=len(p);keep=np.ones(len(p),dtype=bool)
if a.confidence_cut is not None:
    scores=np.zeros(len(p),dtype=np.float32)
    # COLMAP binary observations map each initial point to its source depth pixel.
    # Vectorized parsing avoids expensive per-observation Python bindings.
    with (ROOT/'outputs'/a.run/'colmap/sparse/0/images.bin').open('rb') as f:
        num_images=struct.unpack('<Q',f.read(8))[0]
        for _ in range(num_images):
            header=f.read(64);image_id=struct.unpack('<I',header[:4])[0]
            while f.read(1)!=b'\0':pass
            count=struct.unpack('<Q',f.read(8))[0]
            obs=np.frombuffer(f.read(count*24),dtype=[('xy','<f8',(2,)),('pid','<i8')])
            xy=np.rint(obs['xy']).astype(int);ids=obs['pid']-1
            assert ((ids>=0)&(ids<len(p))).all()
            scores[ids]=g['confidence'][image_id-1,xy[:,1],xy[:,0]]
    keep=scores>=a.confidence_cut
    assert keep.any()
center=np.median(p[keep],axis=0);scale=np.linalg.norm(np.percentile(p[keep],95,axis=0)-np.percentile(p[keep],5,axis=0))
p=(p-center)/scale
# Put the first input camera's right/up axes into the initial presentation view.
p=p@g['extrinsics'][0,:,:3].T;p[:,1:]*=-1
display_points=int(keep.sum());p=np.concatenate([p[keep],p]);render_colors=np.concatenate([g['colors'][keep],g['colors']])
payload=base64.b64encode(p.astype('<f4').tobytes()).decode()
colors=base64.b64encode(render_colors.astype('uint8').tobytes()).decode()
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第一步结果 · 交互式点云</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0d1520;color:#eef4fa;font:16px 'Microsoft YaHei',sans-serif}canvas{display:block;width:100vw;height:100vh;touch-action:none}aside{position:absolute;top:24px;left:24px;padding:22px;background:#101c2be8;border:1px solid #334356;border-radius:16px;max-width:360px}h1{font-size:24px;margin:0 0 12px}p{color:#aebdce;line-height:1.7;margin:8px 0}.badge{color:#77e5c1}label{display:flex;align-items:center;gap:12px;margin-top:18px}button{margin-top:16px;background:#77e5c1;color:#0c2220;border:0;border-radius:7px;padding:10px 16px;cursor:pointer}footer{position:absolute;bottom:20px;left:24px;color:#9aabba;font-size:13px}#error{color:#ffb5a7}</style>
<canvas id="view" aria-label="可旋转和缩放的三维点云"></canvas><aside><h1>第一步：三维点云</h1><p class="badge">12 张照片 → VGGT → 80,000 个三维点</p><p>左键拖动：旋转视角<br>滚轮：缩放<br>右键拖动：平移</p><label>显示点大小 <input id="size" type="range" min="1" max="7" step="0.25" value="2.25"></label><p>调大点的显示尺寸，可以观察点之间的空隙如何变化。此操作只改变显示效果。</p><button id="reset">恢复初始视角</button><p id="error"></p></aside><footer>VGGT 官方工程车示例 · 本机实际重建 · 此页展示初始点云，不是 Gaussian 渲染</footer>
<script>
const canvas=document.getElementById('view'),gl=canvas.getContext('webgl',{antialias:true,preserveDrawingBuffer:true});
if(!gl){document.getElementById('error').textContent='浏览器未启用 WebGL，请使用 Chrome 或 Edge 打开。';throw Error('WebGL unavailable')}
const decode=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const pts=new Float32Array(decode('__POINTS__').buffer),col=decode('__COLORS__');
function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,`attribute vec3 pos;attribute vec3 color;uniform vec2 angles;uniform vec2 pan;uniform float distance;uniform float aspect;uniform float pointSize;varying vec3 rgb;void main(){float cy=cos(angles.x),sy=sin(angles.x),cx=cos(angles.y),sx=sin(angles.y);vec3 q=vec3(cy*pos.x+sy*pos.z,pos.y,-sy*pos.x+cy*pos.z);q=vec3(q.x,cx*q.y-sx*q.z,sx*q.y+cx*q.z);float z=distance-q.z;q.xy+=pan;float near=.02,far=40.;gl_Position=vec4(q.x*1.7/aspect,q.y*1.7,(far+near)/(far-near)*z-2.*far*near/(far-near),z);gl_PointSize=pointSize;rgb=color;}`));
gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`precision mediump float;varying vec3 rgb;void main(){if(distance(gl_PointCoord,vec2(.5))>.5)discard;gl_FragColor=vec4(rgb,1.);}`));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
function attr(name,data,type,normalized){let b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);let a=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,type,normalized,0,0)}
attr('pos',pts,gl.FLOAT,false);attr('color',col,gl.UNSIGNED_BYTE,true);gl.enable(gl.DEPTH_TEST);gl.clearColor(13/255,21/255,32/255,1);
const u=Object.fromEntries(['angles','pan','distance','aspect','pointSize'].map(k=>[k,gl.getUniformLocation(program,k)]));let yaw=0,pitch=0,dist=1.7,pan=[.25,0],drag=null;
function draw(){let dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(innerWidth*dpr);canvas.height=Math.round(innerHeight*dpr);gl.viewport(0,0,canvas.width,canvas.height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.uniform2f(u.angles,yaw,pitch);gl.uniform2fv(u.pan,pan);gl.uniform1f(u.distance,dist);gl.uniform1f(u.aspect,canvas.width/canvas.height);gl.uniform1f(u.pointSize,+document.getElementById('size').value*dpr);gl.drawArrays(gl.POINTS,0,pts.length/3)}
canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY,e.button];canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!drag)return;let dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(drag[2]===2){pan[0]+=dx/innerHeight*dist;pan[1]-=dy/innerHeight*dist}else{yaw+=dx*.006;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*.006))}drag[0]=e.clientX;drag[1]=e.clientY;draw()});canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('contextmenu',e=>e.preventDefault());canvas.addEventListener('wheel',e=>{e.preventDefault();dist=Math.max(.25,Math.min(8,dist*Math.exp(e.deltaY*.001)));draw()},{passive:false});document.getElementById('size').oninput=draw;document.getElementById('reset').onclick=()=>{yaw=0;pitch=0;dist=1.7;pan=[.25,0];document.getElementById('size').value=2.25;draw()};addEventListener('resize',draw);draw();document.body.dataset.points=pts.length/3;
</script></html>'''
dest=ROOT/a.deliver/'第一步_交互点云.html';dest.parent.mkdir(parents=True,exist_ok=True)
html=html.replace('12 张照片',f'{len(g["images"])} 张照片').replace('80,000',f'{total_points:,}').replace('VGGT 官方工程车示例',a.label)
html=html.replace('gl.drawArrays(gl.POINTS,0,pts.length/3)',f'gl.drawArrays(gl.POINTS,document.getElementById("showLow")?.checked?{display_points}:0,document.getElementById("showLow")?.checked?{total_points}:{display_points})')
if a.confidence_cut is not None:
    html=html.replace('<button id="reset">',f'<label><input type="checkbox" id="showLow">显示低置信度背景点</label><p>默认显示 {display_points:,} 个点（置信度 ≥ {a.confidence_cut}）；勾选显示全部 {total_points:,} 个点。只改变查看范围，不修改训练结果。</p><button id="reset">')
    html=html.replace("document.getElementById('size').oninput=draw;","document.getElementById('size').oninput=draw;document.getElementById('showLow').onchange=draw;")
dest.write_text(html.replace('__POINTS__',payload).replace('__COLORS__',colors),encoding='utf-8')
print(dest)
