"""Independent physical queries; no compression fitting samples are used."""
import json
from pathlib import Path
import numpy as np

out=Path("out/lut_joint16_fresh_queries")
if out.exists():raise ValueError("output exists")
out.mkdir()
queries=[]
regions={}
def add(h,s,e,az):
    queries.append(dict(altitude_km=float(h),sun_elevation_deg=float(s),
        view_elevation_deg=float(np.clip(e,-90,90)),relative_azimuth_deg=float(az%360)))
def horizon(h):
    h=np.float32(h);r=np.float32(6360)
    return np.rad2deg(np.arcsin(-np.sqrt(h*(2*r+h))/(r+h)))

start=len(queries)
e=np.deg2rad(np.float32(85))
sun=np.array([np.cos(e),0,np.sin(e)],np.float32)
tangent=np.array([-np.sin(e),0,np.cos(e)],np.float32)
side=np.array([0,1,0],np.float32)
for angle in np.geomspace(.27,3,64,dtype=np.float32):
    gamma=np.deg2rad(angle)
    for az in np.linspace(0,360,64,endpoint=False,dtype=np.float32):
        phi=np.deg2rad(az)
        direction=np.cos(gamma)*sun+np.sin(gamma)*(np.cos(phi)*tangent+np.sin(phi)*side)
        add(.2,85,np.rad2deg(np.arcsin(direction[2])),np.rad2deg(np.arctan2(direction[1],direction[0])))
regions['noon_halo_0.27_to_3_deg']=[start,len(queries)]

start=len(queries)
for h in np.linspace(26,36,9,dtype=np.float32):
    hor=horizon(h)
    for ds in [-.8,-.3,0,.3,.8]:
        for dv in [-1,-.1,.02,.1,1,3,10]:
            for az in np.linspace(0,180,33,dtype=np.float32):add(h,hor+ds,hor+dv,az)
regions['moving_shadow_26_to_36_km']=[start,len(queries)]

start=len(queries)
for h in [90,100,108,112,116,119,119.9,120,120.1,150,400,2000]:
    hor=horizon(h)
    for s in [-12,-6,0,45,85]:
        for dv in [-1,-.1,.02,.1,1,3,10,30,60,90]:
            for az in [0,30,60,90,120,180]:add(h,s,hor+dv,az)
regions['upper_atmosphere_and_space']=[start,len(queries)]
(out/'queries.json').write_text(json.dumps(queries,indent=2))
(out/'regions.json').write_text(json.dumps(regions,indent=2))
print(len(queries),regions)
