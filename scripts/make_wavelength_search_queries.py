"""Disjoint atmospheric poses for fitting, validation and dense visual checks."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args()
if a.out.exists():raise ValueError('output exists')
F=np.float32
rng=np.random.default_rng(2026091808)
queries=[];labels=[];pairs=[];images=[]
def horizon(h):return F(np.rad2deg(-np.arcsin(np.sqrt(F(h)*(F(12720)+F(h)))/(F(6360)+F(h)))))
def direction(e,az):
    e,az=np.deg2rad(np.array([e,az],dtype=np.float32))
    return np.array([np.cos(e)*np.cos(az),np.cos(e)*np.sin(az),np.sin(e)],dtype=np.float32)
def classify(h,s,view,az):
    nu=np.clip(np.dot(direction(view,az),direction(s,0)),-1,1)
    if np.rad2deg(np.arccos(nu))<F(3.01) and s>0:return 'aureole'
    if h>=90:return 'upper_space'
    if h>=20 and abs(s-horizon(h))<4:return 'high_shadow'
    if s<-2:return 'twilight'
    if s<=3:return 'sunset'
    if view<=horizon(h):return 'ground_day'
    return 'daylight'
def append(h,s,view,az,split,pose,region=None):
    queries.append(dict(altitude_km=float(F(h)),sun_elevation_deg=float(F(s)),view_elevation_deg=float(F(np.clip(view,-90,90))),relative_azimuth_deg=float(F(az))))
    labels.append(dict(split=split,pose=pose,region=region or classify(h,s,view,az)))
    return len(queries)-1
def solar_offset(s,theta,phi):
    sun=direction(s,0);side=np.array([0,1,0],dtype=np.float32);up=np.cross(sun,side)
    t,p=np.deg2rad(np.array([theta,phi],dtype=np.float32))
    v=sun*np.cos(t)+(side*np.cos(p)+up*np.sin(p))*np.sin(t)
    return float(np.rad2deg(np.arcsin(np.clip(v[2],-1,1)))),float(np.rad2deg(np.arctan2(v[1],v[0])))

specs=[('train',[.003,.08,.3,1.3,4.8,10.5,13,23,28,34,47,72,103,115,119.6,121,550,3100],[-16,-9,-5,-1,1,15,43,72,88],[-.7,.2,2]),
       ('validation',[.015,.16,.8,2.6,7.5,11.5,17.5,26,32,39,58,86,110,118.2,119.95,150,800,12000],[-13,-7,-3.5,-.2,3.5,27,61,83],[-.4,.6,3])]
for split,heights,suns,relative in specs:
    for h in heights:
        for si,s in enumerate(suns+[float(horizon(h)+x) for x in relative]):
            pose=f'{split}_{h}_{si}'
            offsets=[-.5,.02,.15,.6,2.5,8,25,75] if split=='train' else [-.2,.05,.3,1.2,4,12,35,82]
            for az in ([0,90,180] if split=='train' else [17,107,167]):
                last=None
                for delta in offsets:
                    i=append(h,s,horizon(h)+delta,az,split,pose)
                    if last is not None:pairs.append([last,i])
                    last=i
            for k in range(8):
                u=F((k+float(rng.random()))/8);v=F(rng.random())
                append(h,s,np.rad2deg(np.arcsin(F(2)*u-F(1))),F(360)*v,split,pose)
            if h<90 and s>10:
                angles=[.3,.7,2,8] if split=='train' else [.4,1.2,3,15]
                for theta in angles:
                    for phi in ([0,90,180,270] if split=='train' else [35,125,215,305]):
                        e,az=solar_offset(s,theta,phi);append(h,s,e,az,split,pose,'aureole' if theta<=3 else None)

# None of these time/altitude pairs appears in fitting or validation.
scenes=[
    ('noon_aureole',.2,85,85,0,12),
    ('noon_horizon',.2,85,4,0,100),
    ('sunset',.2,0,4,0,100),
    ('blue_sunward',.2,-6,5,0,90),
    ('blue_antisolar',.2,-6,5,180,90),
    ('aircraft_twilight',12,-7,float(horizon(12)+4),0,90),
    ('moving_shadow',30,float(horizon(30)-.3),float(horizon(30)+3),0,30),
    ('upper_limb',108,0,-.5,90,12),
    ('orbital_twilight',400,-6,float(horizon(400)+3),0,38),
]
w,hp=192,96
for name,alt,sun,pitch,yaw,hfov in scenes:
    start=len(queries);forward=direction(pitch,yaw);right=direction(0,yaw+90);up=np.cross(forward,right)
    tangent=F(np.tan(np.deg2rad(F(hfov)*F(.5))))
    for y in range(hp):
        for x in range(w):
            u=F((F(x)+F(.5))/F(w)*F(2)-F(1))*tangent
            v=F(F(1)-(F(y)+F(.5))/F(hp)*F(2))*tangent*F(hp/w)
            d=forward+u*right+v*up;d/=np.linalg.norm(d)
            e=np.rad2deg(np.arcsin(np.clip(d[2],-1,1)));az=np.rad2deg(np.arctan2(d[1],d[0]))
            append(alt,sun,e,az,'visual',name,name)
    images.append(dict(name=name,start=start,end=len(queries),width=w,height=hp,altitude_km=alt,sun_elevation_deg=sun,pitch=pitch,yaw=yaw,horizontal_fov=hfov))
metadata=dict(kind='wavelength_search_queries_v1',seed=2026091808,queries=queries,labels=labels,adjacent_pairs=pairs,images=images,
    note='Atmospheric poses disjoint across fitting, validation and visual sets. Visual images are excluded from candidate fitting and numerical shortlist selection. No PT data.')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(metadata,separators=(',',':')))
print({s:sum(l['split']==s for l in labels) for s in ['train','validation','visual']})
