"""Export clean point-cloud and Gaussian fly-through GIFs from verified movies."""
from pathlib import Path
import json,subprocess,zipfile
from PIL import Image
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'点云与漫游GIF';OUT.mkdir(exist_ok=True)
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
records=[]
for scene,folder in [('工程车','交付素材'),('网球','网球_交付素材')]:
    for kind in ['点云','纯三维场景漫游']:
        if kind=='点云':
            src=ROOT/folder/'三维重建演示.mp4'
            # The 3–6s segment contains only point cloud + recovered cameras in
            # this viewport. Remove the surrounding presentation UI entirely.
            prefix='trim=start=3:end=6,setpts=PTS-STARTPTS,crop=920:518:312:136,'
            expected=3000
        else:
            src=ROOT/folder/'纯三维场景漫游.mp4';prefix='';expected=12000
        dest=OUT/f'{scene}_{kind}.gif'
        filters=f'[0:v]{prefix}fps=12,scale=768:432:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a'
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(src),'-filter_complex',filters,'-loop','0',str(dest)],check=True)
        with Image.open(dest) as im:
            duration=0
            assert im.size==(768,432)
            for i in range(im.n_frames):im.seek(i);im.load();duration+=im.info.get('duration',0)
            assert abs(duration-expected)<=100,(dest,duration)
            assert im.n_frames>=expected/1000*12-1
            records.append(dict(file=dest.name,source=str(src.relative_to(ROOT)),size=list(im.size),frames=im.n_frames,duration_ms=duration,loop=im.info.get('loop'),bytes=dest.stat().st_size))
        print(records[-1],flush=True)
(OUT/'文件说明.md').write_text('''# 点云和纯三维漫游 GIF

四个文件均为 768×432、约 12 fps、无限循环，可插入 PowerPoint。

- 工程车_点云.gif、网球_点云.gif：3 秒循环；来自第一步实际初始点云与相机的动态展示，已去掉流程侧栏和字幕。
- 工程车_纯三维场景漫游.gif、网球_纯三维场景漫游.gif：12 秒循环；来自第二步真实 Gaussian 渲染，已去掉流程侧栏和字幕。

网球点云 GIF 展示完整初始点云，包含低置信度背景点；与交互页默认过滤后的视图不同。两种点云均只有观察视角移动，物体本身不运动。GIF 为有限色彩预览格式，可能比 MP4 有更多色带或颗粒。网球部分插值视角的重影和背景模糊来自原重建结果，未作生成式修饰。
''',encoding='utf-8')
(OUT/'validation.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(ROOT/'点云与漫游GIF_四个文件.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for path in sorted(OUT.iterdir()):z.write(path,arcname=path.name)
