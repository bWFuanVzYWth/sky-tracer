"""Independent speed and deterministic importance-quadrature experiments."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path('out/hybrid_perf_v1');BASE=json.loads(Path('crates/sky-hybrid-atmosphere/configs/balanced.json').read_text())
BASE.update(cache_single=True,specialize_constants=True,high_sun_weight=1.0)
VARIANTS={
    'plain':dict(cache_single=False,specialize_constants=False),
    'specialize':dict(cache_single=False,specialize_constants=True),
    'cache':dict(cache_single=True,specialize_constants=False),
    'both':dict(cache_single=True,specialize_constants=True),
    'horizon1':dict(horizon_power=1),
    'horizon2':dict(horizon_power=2),
    'horizon25':dict(horizon_power=2.5),
    'horizon35':dict(horizon_power=3.5),
    'horizon4':dict(horizon_power=4),
    'sun_narrow':dict(sun_importance_width=.001),
    'sun_wide':dict(sun_importance_width=.1),
    'path_uniform':dict(path_height_scale=0),
    'path_narrow':dict(path_height_scale=.1),
    'path_wide':dict(path_height_scale=1),
    'q64_p3':dict(angular_mu=64,angular_phi=64),
    'q64_p4':dict(angular_mu=64,angular_phi=64,horizon_power=4),
    'q64_p2':dict(angular_mu=64,angular_phi=64,horizon_power=2),
    'high_sun0':dict(high_sun_weight=0),
    'high_sun25':dict(high_sun_weight=.25),
    'q64_high_sun0':dict(angular_mu=64,angular_phi=64,high_sun_weight=0),
    'q48_high_sun0':dict(angular_mu=48,angular_phi=48,high_sun_weight=0),
    'plain_v2':dict(cache_single=False,specialize_constants=False),
    'specialize_v2':dict(cache_single=False,specialize_constants=True),
    'cache_v2':dict(cache_single=True,specialize_constants=False),
    'both_v2':dict(cache_single=True,specialize_constants=True),
    'selected_v2':dict(high_sun_weight=0),
    'selected_nocache':dict(high_sun_weight=0,cache_single=False),
}
def main():
    (ROOT/'configs').mkdir(parents=True,exist_ok=True)
    for name in sys.argv[1:] or VARIANTS:
        out=ROOT/name;out.mkdir(exist_ok=True)
        if (out/'runs.json').exists():continue
        c=BASE|VARIANTS[name];config=ROOT/'configs'/(name+'.json');config.write_text(json.dumps(c,indent=2))
        with (out/'process.log').open('w') as log:
            subprocess.run(['target/release/examples/evaluate.exe','--out',str(out),'--queries','out/four_wave_sun_v1/queries.json','--config',str(config)],stdout=log,stderr=subprocess.STDOUT,check=True)
        solve=json.loads((out/'solve.json').read_text())['solve']
        print(name,round(solve['wall_seconds'],3),'s',round(solve['peak_payload_bytes']/1048576,1),'MiB peak',flush=True)
if __name__=='__main__':main()
