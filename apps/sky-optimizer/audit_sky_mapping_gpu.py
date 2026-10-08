"""Strictly serialized GPU audits. Run only in an explicitly assigned GPU slot."""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['static','motion','cost'],required=True)
    p.add_argument('--candidate',choices=['256','224'],default='256',help='Chosen candidate for independent motion holdout')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--candidate-dir',type=Path,default=ROOT/'experiments/validation/sky_mapping_candidates_v7',
                   help='Frozen directory containing sky256.json and sky224.json, or cold-fit outputs')
    p.add_argument('--baseline-mapping',type=Path,default=ROOT/'experiments/validation/sky_mapping_candidates_v7/baseline.json',
                   help='Frozen original mapping; independent of subsequent production defaults')
    p.add_argument('--config',type=Path,default=ROOT/'experiments/validation/sky_mapping_candidates_v7/config.json',
                   help='Frozen solver budget; this audit overrides only SkyView size and ground culling')
    p.add_argument('--wavelengths',type=Path,default=ROOT/'experiments/validation/sky_mapping_candidates_v7/wavelengths.json')
    p.add_argument('--dry-run',action='store_true',help='Print commands; no files/devices are created')
    a=p.parse_args()
    if a.out.exists():raise ValueError('Output exists; choose a new directory')
    out=a.out
    current=json.loads(a.config.read_text())
    def fitted_path(size):
        promoted=a.candidate_dir/f'sky{size}.json'
        return promoted if promoted.exists() else a.candidate_dir/f'sky{size}_mapping.json'
    mappings={'baseline':a.baseline_mapping,'256':fitted_path(256),'224':fitted_path(224)}
    configs={name:dict(current,sky_size=256 if name=='baseline' else int(name),skip_unused_ground=True)
             for name in mappings}
    configs['control_ground']=dict(current,sky_size=256,skip_unused_ground=False)
    jobs=[]
    exe=ROOT/'target/release/sky-audit.exe'
    def profile(name,variant,trajectory=None,size=(1920,1080),frames=120,quality=0,rounds=3,warmup=8,disk=True):
        mapping='baseline' if variant=='control_ground' else variant
        command=[str(exe),'profile','--out',str(out/name),'--config',str(out/f'{variant}_config.json'),
                 '--mapping',str(out/f'{mapping}_mapping.json'),'--steps','64',
                 '--wavelengths',str(out/'wavelengths.json'),'--width',str(size[0]),'--height',str(size[1]),
                 '--frames',str(frames),'--quality-frames',str(quality),'--rounds',str(rounds),
                 '--warmup-frames',str(warmup)]
        if trajectory:command+=['--trajectories',str(trajectory)]
        if not disk:command+=['--no-sun-disk']
        jobs.append(dict(name=name,command=command))
    if a.phase=='static':
        aero=json.loads((ROOT/'experiments/validation/grazing.json').read_text())
        aero['images']=[i for i in aero['images'] if i['altitude_km'] in [.002,.2]
                        and ((i['sun_elevation_deg']==-8 and i['yaw'] in [0,180]) or i['name'].startswith('g_close') and i['sun_elevation_deg']==0)]
        for variant in mappings:
            for name,queries,scale in [('43',ROOT/'experiments/validation/sky_cases.json',1),
                                      ('extra22',ROOT/'experiments/validation/mapping_extra_cases.json',1),
                                      ('aerosol4',out/'aerosol4_queries.json',4),
                                      ('hd14',ROOT/'experiments/validation/moving_sun_1080_queries.json',1)]:
                command=[str(exe),'evaluate','--out',str(out/f'{variant}_{name}'),
                         '--config',str(out/f'{variant}_config.json'),'--mapping',str(out/f'{variant}_mapping.json'),
                         '--steps','64','--wavelengths',str(out/'wavelengths.json'),'--queries',str(queries),
                         '--aerosol-scale',str(scale)]
                if name=='43':command+=['--export-source','--export-optical']
                jobs.append(dict(name=f'{variant}_{name}',command=command))
            profile(f'{variant}_hd_display',variant,ROOT/'experiments/validation/moving_sun_1080_anchors.json',frames=3,quality=1,rounds=1,warmup=0)
            profile(f'{variant}_sequence',variant,ROOT/'experiments/validation/moving_sun_sensitive.json',
                    size=(320,180),frames=65,quality=65,rounds=1,warmup=0,disk=False)
    elif a.phase=='motion':
        # These independent physical poses are withheld from fitting and from
        # shortlist selection. Do not iterate the fit against their result.
        for variant in ['baseline',a.candidate]:
            profile(f'{variant}_holdout',variant,ROOT/'experiments/validation/moving_sun_holdout.json',
                    size=(320,180),frames=65,quality=65,rounds=1,warmup=0,disk=False)
    else:
        for variant in ['control_ground','baseline','256','224']:
            profile(f'{variant}_1080_cost',variant)
    if a.dry_run:
        print(json.dumps({'phase':a.phase,'jobs':jobs,'gpu_initialized':False},indent=2));return
    out.mkdir(parents=True)
    for variant,config in configs.items():(out/f'{variant}_config.json').write_text(json.dumps(config,indent=2)+'\n')
    for variant,path in mappings.items():shutil.copyfile(path,out/f'{variant}_mapping.json')
    shutil.copyfile(a.wavelengths,out/'wavelengths.json')
    if a.phase=='static':(out/'aerosol4_queries.json').write_text(json.dumps(aero,indent=2)+'\n')
    (out/'plan.json').write_text(json.dumps({'phase':a.phase,'jobs':jobs,'strict_serial':True},indent=2)+'\n')
    records=[]
    for job in jobs:
        start=time.perf_counter()
        with (out/f'{job["name"]}.log').open('w') as log:
            try:
                result=subprocess.run(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                code=result.returncode
            except subprocess.TimeoutExpired:
                code=-1
        records.append(dict(job,exit_code=code,elapsed_seconds=time.perf_counter()-start))
        (out/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
        print(f'{job["name"]}: exit={code}, {records[-1]["elapsed_seconds"]:.2f}s',flush=True)
        if code:raise SystemExit('GPU audit failed/timed out; later jobs were not launched')
    if a.phase=='static':
        baseline=(out/'baseline_43/source.rgba16f').read_bytes()
        for variant in ['256','224']:
            if (out/f'{variant}_43/source.rgba16f').read_bytes()!=baseline:
                raise ValueError(f'{variant} unexpectedly changed the precomputed normalized source')
            if (out/f'{variant}_43/optical.rgba16f').read_bytes()!=(out/'baseline_43/optical.rgba16f').read_bytes():
                raise ValueError(f'{variant} unexpectedly changed optical depth')
        (out/'source_exact.json').write_text(json.dumps({'all_byte_identical':True,'variants':['baseline','256','224']})+'\n')

if __name__=='__main__':main()
