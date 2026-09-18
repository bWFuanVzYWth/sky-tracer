"""One-axis reductions, with a frozen executable for reproducible controls."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path('out/hybrid_budget_v1')
BASE=json.loads((ROOT/'baseline.json').read_text())
VARIANTS={
    'baseline':{},'h40':{'heights':40},'h32':{'heights':32},
    's144':{'suns':144},'s128':{'suns':128},'p20':{'phases':20},'p16':{'phases':16},
    'c10':{'cones':10},'c8':{'cones':8},
    'mu32':{'angular_mu':32},'mu24':{'angular_mu':24},
    'phi24':{'angular_phi':24},'phi16':{'angular_phi':16},
    'steps64':{'ray_steps':64},'steps48':{'ray_steps':48},
    'orders7':{'iterations':7},'orders6':{'iterations':6},'orders5':{'iterations':5},
    'opt192_768':{'optical':[192,768]},'opt128_512':{'optical':[128,512]},
    'opt_steps512':{'optical_steps':512},'opt_steps256':{'optical_steps':256},
    'sky224':{'sky_size':224},'sky192':{'sky_size':192},
    'runtime96':{'runtime_steps':96},'runtime64':{'runtime_steps':64},
}
def run(name,changes,exe=None,queries='out/four_wave_sun_v1/queries.json'):
    out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'runs.json').exists():return
    config=BASE|changes;steps=config.pop('runtime_steps',128)
    if exe is not None and str(exe).replace('\\','/')=='target/release/examples/evaluate.exe':
        for key in ['cache_single','layer_split','azimuth_gauss','high_angular_phi','shadow_split','sky_half','sky_chord','space_limb_weight']:
            value=config.pop(key,None)
            if value not in [None,False,0]:
                raise ValueError(f'{key} is an archived experiment; use its saved executable')
    path=out/'config.json';path.write_text(json.dumps(config,indent=2))
    with (out/'process.log').open('w') as log:
        subprocess.run([str(exe or ROOT/'evaluate_baseline.exe'),'--out',str(out),'--queries',queries,'--config',str(path),'--steps',str(steps)],stdout=log,stderr=subprocess.STDOUT,check=True)
    solve=json.loads((out/'solve.json').read_text())['solve']
    print(name,round(solve['wall_seconds'],3),'s',round(solve['resident_bytes']/1048576,3),'MiB',flush=True)
if __name__=='__main__':
    for name in sys.argv[1:] or VARIANTS:run(name,VARIANTS[name])
