"""Unfitted camera/time/altitude probes, including representation boundaries."""
import json,math
from pathlib import Path
scenes=[]
def add(name,h,s,pitch,yaw=0,fov=110,group=''):
    scenes.append(dict(name=name,altitude_km=h,sun_elevation_deg=s,pitch=pitch,yaw=yaw,horizontal_fov=fov,width=96,height=48,group=group))
for i in range(31):
    s=30+2*i
    add(f'noon_horizon_{i}',.2,s,1,group='noon_horizon')
    add(f'noon_sun_{i}',.2,s,s,fov=16,group='noon_sun')
for i in range(45):
    s=-9+i*.25
    for yaw in [0,180]:add(f'blue_{yaw}_{i}',.2,s,3,yaw,group=f'blue_{yaw}')
heights=[0,.001,.01,.199,.201,.99,1,1.01,1.99,2,2.01,10.99,11,11.01,11.99,12,12.01,34.9,34.99,35,35.01,35.1,60,108,119,119.9,119.99,120,120.01,120.1,121,160,400,2000,36000]
for i,h in enumerate(heights):
    horizon=-math.degrees(math.acos(6360/(6360+h)))
    outer=-math.degrees(math.acos(min(1,6480/(6360+h))))
    for solar in [47,-6]:
        add(f'alt_{solar}_{i}',h,solar,(horizon+outer)/2 if h>=120 else horizon+1,fov=max(.3,min(35,3*(outer-horizon))) if h>=120 else 35,group=f'alt_{solar}')
out=Path('out/hybrid_mapping_v3/sweeps.json');out.write_text(json.dumps(dict(images=scenes),indent=2))
print(len(scenes),'new views')
