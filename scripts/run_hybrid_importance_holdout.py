"""Unfitted high-altitude shadow cameras; all variants use the same 4D grid."""
import json,math,subprocess
from pathlib import Path
root=Path('out/hybrid_perf_v1');scenes=[]
for h in [48,72,119,121,180,280,700,1400]:
    hor=-math.degrees(math.acos(6360/(6360+h)))
    for ds in [-.55,.13,.65]:
        for yaw in [60,135]:
            scenes.append(dict(name=f'h{h}_s{ds}_a{yaw}',altitude_km=h,sun_elevation_deg=hor+ds,
                pitch=hor+1,yaw=yaw,horizontal_fov=16,width=128,height=64))
queries=root/'shadow_holdout.json';queries.write_text(json.dumps(dict(images=scenes),indent=2))
for name,config in [('control','both_v2'),('selected','selected_nocache'),('dense','q64_high_sun0'),('dense48','q48_high_sun0'),('dense_old','q64_p3')]:
    out=root/('holdout_'+name);out.mkdir(exist_ok=True)
    if (out/'runs.json').exists():continue
    with (out/'process.log').open('w') as log:
        subprocess.run(['target/release/examples/evaluate.exe','--out',str(out),'--queries',str(queries),'--config',str(root/'configs'/(config+'.json'))],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,'complete',flush=True)
