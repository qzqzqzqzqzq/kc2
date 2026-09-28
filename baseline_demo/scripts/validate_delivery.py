"""Decode every delivered video frame and inspect all still/GIF files."""
import json,subprocess,hashlib,zipfile,argparse
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio
import imageio_ffmpeg
parser=argparse.ArgumentParser();parser.add_argument('--deliver',default='交付素材');parser.add_argument('--zip',default='开题展示素材.zip');a=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1];d=ROOT/a.deliver;report={}
for name,expected in [('三维重建演示.mp4',540),('纯三维场景漫游.mp4',360)]:
    path=d/name
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],capture_output=True,text=True)
    assert result.returncode==0 and not result.stderr,(name,result.stderr)
    with imageio.get_reader(str(path)) as r:
        meta=r.get_meta_data();count=r.count_frames()
    assert count==expected and meta['size']==(1280,720) and meta['fps']==30
    report[name]=dict(frames=count,fps=meta['fps'],duration=meta['duration'],size=meta['size'],all_frames_decoded=True,bytes=path.stat().st_size)
for path in d.glob('*.png'):
    with Image.open(path) as im:im.verify()
    report[path.name]={'valid_png':True}
with Image.open(d/'三维重建演示.gif') as im:
    duration=0
    for i in range(im.n_frames):im.seek(i);im.load();duration+=im.info['duration']
    assert duration==18000
    report['三维重建演示.gif']=dict(frames=im.n_frames,duration_ms=duration,size=im.size)
h=hashlib.sha256()
with (ROOT/'checkpoints/vggt_model.pt').open('rb') as f:
    while b:=f.read(8*1024*1024):h.update(b)
assert h.hexdigest()=='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
report['checkpoint_sha256_verified']=h.hexdigest()
(d/'delivery_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(ROOT/a.zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for path in sorted(d.iterdir()):z.write(path,arcname=d.name+'/'+path.name)
print(json.dumps(report,ensure_ascii=False,indent=2))
