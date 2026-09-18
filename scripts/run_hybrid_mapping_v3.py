"""Sequential GPU ablations; logs and full configs accompany each result."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path('out/hybrid_mapping_v3')
BASE=json.loads(Path('crates/sky-hybrid-atmosphere/configs/balanced.json').read_text())
# Equal-budget mapping comparison was defined before selecting the final C=12
# preset; keep its eight-cone control fixed when reproducing the experiment.
BASE['cones']=8
BASE.update(cache_single=False,specialize_constants=False,high_sun_weight=1.0,
            sun_importance_width=.01,path_height_scale=.25)
VARIANTS={
    'fitted_initial':dict(),
    'selected_final':dict(cones=12),
    'legacy':dict(mapping_flags=0),
    'height_only':dict(mapping_flags=1),
    'solar_only':dict(mapping_flags=2),
    'phase_only':dict(mapping_flags=4),
    'cone_only':dict(mapping_flags=8),
    'optical_only':dict(mapping_flags=16),
    'source_fitted':dict(mapping_flags=15),
    'h72':dict(heights=72),
    's256':dict(suns=256),
    'p32':dict(phases=32),
    'c16':dict(cones=16),
    'c12':dict(cones=12),
    'c16_opt512':dict(cones=16,optical=[256,512]),
    'quad48':dict(angular_mu=48,angular_phi=48),
    'quad64_32':dict(angular_mu=64),
    'quad40_48':dict(angular_phi=48),
    'steps192':dict(ray_steps=192),
    'orders12':dict(iterations=12),
    'runtime256':dict(runtime_steps=256),
    'optical512':dict(optical=[256,512]),
    'optical_dense':dict(optical=[512,2048],optical_steps=4096),
    'sky512':dict(sky_size=512),
    'dense_grid':dict(heights=72,suns=224,phases=32,cones=16),
    'dense':dict(heights=72,suns=224,phases=32,cones=16,angular_mu=48,angular_phi=48,ray_steps=192,iterations=12,runtime_steps=256),
    'dense_check':dict(heights=72,suns=224,phases=32,cones=16,angular_mu=56,angular_phi=48,ray_steps=256,iterations=16,runtime_steps=512),
}
for name,changes in list(VARIANTS.items()):
    if name in ['h72','s256','p32','quad48','quad64_32','quad40_48','steps192','orders12','runtime256','optical_dense','sky512']:
        VARIANTS['final_'+name]=changes|dict(cones=12)
def main():
    configs=ROOT/'configs';configs.mkdir(parents=True,exist_ok=True)
    for name in sys.argv[1:] or VARIANTS:
        out=ROOT/name
        if (out/'runs.json').exists():continue
        c=BASE|VARIANTS[name];steps=c.pop('runtime_steps',128)
        config=configs/(name+'.json');config.write_text(json.dumps(c,indent=2))
        out.mkdir(exist_ok=True)
        with (out/'process.log').open('w') as log:
            subprocess.run(['target/release/examples/evaluate.exe','--out',str(out),'--queries','out/four_wave_sun_v1/queries.json','--config',str(config),'--steps',str(steps)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print(name,json.loads((out/'solve.json').read_text())['solve']['wall_seconds'],flush=True)
if __name__=='__main__':main()
