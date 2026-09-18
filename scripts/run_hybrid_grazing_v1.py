"""Dense horizontal strips and Sun close-ups, including aerosol stress tests."""
import json,math,subprocess
from pathlib import Path
ROOT=Path('out/hybrid_budget_v1');scenes=[]
for h in [.002,.02,.2,1.0]:
    hor=-math.degrees(math.acos(6360/(6360+h)))
    for sun in [-8,-6,-1,-.1,.1,1,47,85]:
        for yaw in [0,90,180]:
            scenes.append(dict(name=f'g_h{h}_s{sun}_a{yaw}',altitude_km=h,sun_elevation_deg=sun,pitch=hor+.05,yaw=yaw,horizontal_fov=100,width=512,height=24))
    for sun in [-.5,0,.5,5]:
        scenes.append(dict(name=f'g_close_h{h}_s{sun}',altitude_km=h,sun_elevation_deg=sun,pitch=hor if sun<1 else sun,yaw=0,horizontal_fov=6,width=128,height=128))
q=ROOT/'grazing_queries.json';q.write_text(json.dumps(dict(images=scenes),indent=2))
base=json.loads((ROOT/'baseline.json').read_text())|dict(batch_heights=8)
lean=json.loads((ROOT/'lean_layers64/config.json').read_text())
configs={
    'baseline':base,
    'lean':lean,
    'uniform':lean|dict(azimuth_gauss=False,angular_phi=32),
    'dense':base|dict(angular_mu=64,angular_phi=64,ray_steps=192,layer_split=True,iterations=12),
}
for scale in [1,4]:
    for name,c in configs.items():
        p=ROOT/f'grazing_{scale}_{name}';p.mkdir(exist_ok=True)
        if (p/'runs.json').exists():continue
        cfg=p/'config.json';cfg.write_text(json.dumps(c,indent=2))
        with (p/'process.log').open('w') as log:
            subprocess.run([str(ROOT/'evaluate_trials.exe'),'--out',str(p),'--queries',str(q),'--config',str(cfg),'--steps','256' if name=='dense' else '128','--aerosol-scale',str(scale)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print(scale,name,'complete',flush=True)
