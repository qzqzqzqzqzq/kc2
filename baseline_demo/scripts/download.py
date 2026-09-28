"""Resumable range downloader; verifies each response and optional SHA256."""
import argparse, concurrent.futures, hashlib, json, time, urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('output');p.add_argument('--proxy',default='http://127.0.0.1:7897');p.add_argument('--workers',type=int,default=12);p.add_argument('--sha256');a=p.parse_args()
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
parts=out.with_name(out.name+'.parts');parts.mkdir(exist_ok=True)
def opener(): return urllib.request.build_opener(urllib.request.ProxyHandler({'http':a.proxy,'https':a.proxy} if a.proxy else {}))
with opener().open(urllib.request.Request(a.url,method='HEAD'),timeout=60) as r:
    size=int(r.headers['Content-Length']);url=r.url
print('SIZE',size,flush=True);chunk=16*1024*1024;start=time.time()
def fetch(i):
    lo=i*chunk;hi=min(size,lo+chunk)-1;f=parts/f'{i:05d}'
    if f.exists() and f.stat().st_size==hi-lo+1:return
    for attempt in range(12):
        try:
            req=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi}'})
            with opener().open(req,timeout=90) as r:
                assert r.status==206,(r.status,lo)
                assert r.headers['Content-Range'].startswith(f'bytes {lo}-{hi}/'),r.headers
                with f.open('wb') as w:
                    while data:=r.read(1024*1024):w.write(data)
            assert f.stat().st_size==hi-lo+1
            return
        except Exception as e:
            print('RETRY',i,attempt,repr(e),flush=True);time.sleep(2*(attempt+1))
    raise RuntimeError(f'Failed part {i}')
total=(size+chunk-1)//chunk
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
    for j,_ in enumerate(ex.map(fetch,range(total))):
        if j%8==0:print('PARTS',j+1,total,'SECONDS',round(time.time()-start),flush=True)
h=hashlib.sha256()
with out.open('wb') as w:
    for i in range(total):
        with (parts/f'{i:05d}').open('rb') as r:
            while data:=r.read(4*1024*1024):w.write(data);h.update(data)
digest=h.hexdigest();assert not a.sha256 or digest==a.sha256,(digest,a.sha256)
out.with_name(out.name+'.download.json').write_text(json.dumps(dict(url=a.url,size=size,sha256=digest,seconds=time.time()-start),indent=2))
print('DONE',digest,flush=True)
