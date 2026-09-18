"""Compare actual GPU queries before/after CPU Rec.2020 export."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("spectral",type=Path)
p.add_argument("rgb",type=Path)
p.add_argument("--out",type=Path,required=True)
a = p.parse_args()
names = {p.name for p in a.spectral.glob("lut_*.f32")}
assert names and names == {p.name for p in a.rgb.glob("lut_*.f32")}, "incomplete or mismatched comparison sets"
rows = []
for path in sorted(a.spectral.glob("lut_*.f32")):
    other = a.rgb/path.name
    info = json.loads(path.with_suffix(".json").read_text())
    rgb_info = json.loads(other.with_suffix(".json").read_text())
    for key in ["width","height","view_yaw_pitch_fov_exposure","asset","sun_elevation_deg","sun_azimuth_deg","altitude_km"]:
        assert info[key]==rgb_info[key],(path,key)
    w,h=info["width"],info["height"]
    x=np.fromfile(path,dtype="<f4").reshape(4,h,w,4)[0,...,:3]
    y=np.fromfile(other,dtype="<f4").reshape(4,h,w,4)[0,...,:3]
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    yaw,pitch,fov=np.deg2rad(info["view_yaw_pitch_fov_exposure"][:3])
    forward=np.array([np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)])
    right=np.array([np.cos(yaw),0,-np.sin(yaw)])
    up=np.cross(forward,right)
    px,py=np.meshgrid((np.arange(w)+.5)/w*2-1,1-(np.arange(h)+.5)/h*2)
    ray=forward+np.tan(fov/2)*(px[...,None]*w/h*right+py[...,None]*up)
    ray/=np.linalg.norm(ray,axis=2)[...,None]
    solar,az=np.deg2rad([info["sun_elevation_deg"],info["sun_azimuth_deg"]])
    sun=np.array([np.sin(az)*np.cos(solar),np.sin(solar),np.cos(az)*np.cos(solar)])
    height=info["altitude_km"]
    horizon=-np.sqrt(height*(2*6360+height))/(6360+height)
    sky=(ray[...,1]>horizon)&(ray@sun<np.cos(np.deg2rad(1)))
    delta=y-x
    regions={}
    for name,mask in [("all",np.ones((h,w),bool)),("sky_excluding_1deg_sun",sky)]:
        magnitude=np.linalg.norm(x[mask],axis=1)
        denominator=float(np.linalg.norm(x[mask]))
        relative=np.linalg.norm(delta[mask],axis=1)/np.maximum(magnitude,max(float(magnitude.max())*1e-6,1e-30))
        regions[name]=dict(pixels=int(mask.sum()),
            relative_rgb_l2=float(np.linalg.norm(delta[mask])/denominator) if denominator>0 else None,
            pixel_relative_p95=float(np.quantile(relative,.95)),
            pixel_relative_p99=float(np.quantile(relative,.99)),
            pixel_relative_max=float(relative.max()))
    rows.append(dict(file=path.name,regions=regions))
a.out.write_text(json.dumps(dict(note="GPU linear Rec.2020 queries; relative pixel errors use an RGB norm floor at 1e-6 of regional peak; includes RGB solar-table interpolation changes",results=rows),indent=2))
for row in rows:
    s=row["regions"]["sky_excluding_1deg_sun"]
    print(row["file"],{k:round(v*100,5) for k,v in s.items() if k!="pixels"})
