"""Verify exported camera/point conventions against actual COLMAP observations."""
import os,json,argparse
from pathlib import Path
import numpy as np
import pycolmap
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',default='kitchen_12');a=p.parse_args();out=ROOT/'outputs'/a.run
g=np.load(out/'geometry.npz');old=os.getcwd()
try:
    os.chdir(out/'colmap');r=pycolmap.Reconstruction('sparse/0')
finally:os.chdir(old)
assert r.num_reg_images()==len(g['images'])
errors=[]
for pid in list(r.points3D)[::80]:
    point=r.points3D[pid]
    for obs in point.track.elements:
        im=r.images[obs.image_id];cam=r.cameras[im.camera_id]
        q=im.cam_from_world.matrix()@np.r_[point.xyz,1.]
        assert q[2]>0
        pix=cam.calibration_matrix()@q;pix=pix[:2]/pix[2]
        errors.append(np.linalg.norm(pix-im.points2D[obs.point2D_idx].xy))
assert np.max(errors)<.02, max(errors)
for key in ['extrinsics','intrinsics','depth','points']:
    assert np.isfinite(g[key]).all(),key
R=g['extrinsics'][:,:,:3];orth=np.max(np.abs(R@R.transpose(0,2,1)-np.eye(3)))
assert orth<1e-4
info=dict(registered_images=r.num_reg_images(),points=r.num_points3D(),sampled_observations=len(errors),max_export_reprojection_error_px=float(max(errors)),rotation_orthogonality_error=float(orth),note='Export coordinate consistency only; not independent reconstruction accuracy or held-out evaluation.')
(out/'geometry_validation.json').write_text(json.dumps(info,indent=2));print(info)
