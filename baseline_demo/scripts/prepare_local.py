"""Import local photos without modifying the originals; save source hashes."""
from pathlib import Path
import argparse,json,hashlib,re
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--scene',required=True);a=p.parse_args()
source=Path(a.source).resolve();out=ROOT/'data'/a.scene/'originals';out.mkdir(parents=True,exist_ok=True)
files=[p for p in source.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png','.webp']]
files.sort(key=lambda p:[int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)',p.name)])
assert len(files)>=2,'At least two photos are required'
sheet=Image.new('RGB',(1200,330*((len(files)+2)//3)),'#0d1520');draw=ImageDraw.Draw(sheet);records=[]
for i,path in enumerate(files):
    with Image.open(path) as raw:im=ImageOps.exif_transpose(raw).convert('RGB')
    name=f'{i:03d}.png';im.save(out/name)
    records.append(dict(file=name,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=im.size))
    im.thumbnail((390,290));x=i%3*400;y=i//3*330;sheet.paste(im,(x+(400-im.width)//2,y));draw.text((x+8,y+300),path.name+' '+str(records[-1]['size']),fill='white')
sheet.save(out.parent/'contact.png');(out.parent/'selection.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Imported {len(files)} photos into {out}')
