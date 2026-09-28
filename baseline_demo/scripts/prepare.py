"""Select sharp, overlapping RGB frames. No depth/ground-truth input is used."""
from pathlib import Path
import argparse, json, tarfile, shutil, io
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--scene', default='desk')
p.add_argument('--count', type=int, default=12)
a = p.parse_args()
if a.scene == 'desk':
    with tarfile.open(ROOT/'data/desk.tgz') as t:
        members = []; payload = {}
        for m in t:
            if '/rgb/' in m.name and m.name.endswith('.png'):
                members.append(m);payload[m.name]=t.extractfile(m).read()
            # Official archive stores the complete RGB directory before depth.
            if '/depth/' in m.name and members: break
        members.sort(key=lambda m: m.name)
        # First six seconds: continuous tabletop coverage, avoiding sequence-wide jumps.
        candidates = members[15:195]
        selected = []
        for group in np.array_split(np.arange(len(candidates)), a.count):
            scores = []
            for idx in group:
                im = np.array(Image.open(io.BytesIO(payload[candidates[idx].name])).convert('L').resize((320,240)), dtype=float)
                lap = -4*im[1:-1,1:-1]+im[:-2,1:-1]+im[2:,1:-1]+im[1:-1,:-2]+im[1:-1,2:]
                scores.append(lap.var())
            selected.append((candidates[group[int(np.argmax(scores))]],float(max(scores))))
        out = ROOT/'data'/a.scene/'originals'; out.mkdir(parents=True,exist_ok=True)
        records=[]
        for i,(m,s) in enumerate(selected):
            dest=out/f'{i:03d}.png'
            dest.write_bytes(payload[m.name])
            records.append(dict(file=dest.name,source=m.name,sharpness=s))
else:
    files=sorted((ROOT/'vendor/vggt/examples'/a.scene/'images').glob('*'))
    out=ROOT/'data'/a.scene/'originals'; out.mkdir(parents=True,exist_ok=True)
    records=[]
    for i,idx in enumerate(np.linspace(0,len(files)-1,min(a.count,len(files))).astype(int)):
        dest=out/f'{i:03d}.png'; Image.open(files[idx]).convert('RGB').save(dest)
        records.append(dict(file=dest.name,source=str(files[idx].relative_to(ROOT))))
(out.parent/'selection.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
sheet=Image.new('RGB',(960,260*((len(records)+3)//4)), '#101820'); d=ImageDraw.Draw(sheet)
for i,r in enumerate(records):
    im=Image.open(out/r['file']); im.thumbnail((234,225))
    x=(i%4)*240; y=(i//4)*260
    sheet.paste(im,(x,y));d.text((x+5,y+230),r['file'],fill='white')
sheet.save(out.parent/'contact.png')
print(json.dumps(records,indent=2))
