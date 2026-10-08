"""Prepare dense teacher curves and refit the existing SkyView softsign CDF."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PI = np.pi

def geometry(h, chart, bottom=6360., top=120.):
    horizon = -np.arccos(bottom / (bottom + h))
    space = h >= top
    outer = -np.arccos((bottom + top)/(bottom + h)) if space else horizon
    bounds = [(horizon, outer if space else 0.), (0., PI/2), (-PI/2, horizon)][chart]
    return bounds, horizon, outer, space

def parameters(calibration, h, sun, chart):
    bounds, hor, outer, space = geometry(h, chart)
    group = 'ground' if chart == 2 else 'space' if space else 'upper' if chart == 1 else 'low'
    weights = np.array(calibration['sky'][group]['weights'], dtype=float)
    widths = np.array(calibration['sky'][group]['widths'], dtype=float)
    if chart == 2 and not space:
        x = np.clip((h-1)/3, 0, 1)
        t = 1 - x*x*(3-2*x)
        weights = weights*(1-t) + np.array(calibration['sky']['ground_near']['weights'])*t
        widths = widths*(1-t) + np.array(calibration['sky']['ground_near']['widths'])*t
        if h <= 1:
            group = 'ground_near'
    return bounds, np.array([hor, sun, outer]), weights, widths, group

def cdf(e, bounds, centers, weights, widths):
    lo, hi = bounds
    x = np.clip(e, lo, hi)
    y = weights[0]*(x-lo)/max(hi-lo, 1e-10)
    for j, (center, width) in enumerate(zip(centers, widths)):
        if center < lo or center > hi:
            y += weights[j+1]*(x-lo)/max(hi-lo, 1e-10)*(abs(hi-center)+width)/(abs(x-center)+width)
        else:
            soft = lambda d: d/(abs(d)+width)
            a, b = soft(lo-center), soft(hi-center)
            y += weights[j+1]*(soft(x-center)-a)/max(b-a, 1e-10)
    return np.clip(y, 0, 1)

def inverse(u, bounds, centers, weights, widths):
    lo, hi = np.full_like(u, bounds[0]), np.full_like(u, bounds[1])
    for _ in range(36):
        mid = (lo+hi)/2
        less = cdf(mid, bounds, centers, weights, widths) < u
        lo, hi = np.where(less, mid, lo), np.where(less, hi, mid)
    return np.where(u<=0, bounds[0], np.where(u>=1, bounds[1], (lo+hi)/2))

def prepare(args):
    if args.out.exists(): raise ValueError('Output exists; preserve frozen inputs')
    args.out.mkdir(parents=True)
    calibration = json.loads(args.mapping.read_text())
    # Physical altitude/Sun poses are disjoint. Validation includes both sides
    # of the atmospheric top and additional thin-limb solar states.
    poses = {
        'train': [(.002,-6),(.2,0),(.2,75),(12,-7),(30,-6),(108,0),(400,-6),(400,-20)],
        'validation': [(.05,-8),(.6,-.5),(.6,47),(16,-8.7),(28,-5.1),(112,-1),(121,-.8),(360,-9),(540,-18)],
    }
    curves, queries, vacuum = [], [], []
    for split, states in poses.items():
        azimuths = [0,45,90,180] if split=='train' else [0,35,125,180]
        for h, sun_deg in states:
            for azimuth in azimuths:
                for chart in range(3):
                    if h>=120 and chart==1: continue
                    bounds, centers, weights, widths, group = parameters(calibration,h,np.deg2rad(sun_deg),chart)
                    grid = inverse(np.linspace(0,1,args.points),bounds,centers,weights,widths)
                    grid = np.union1d(grid,np.linspace(*bounds,257))
                    # Horizon endpoint samples belong to their own boundary
                    # branch, matching build_sky's ±1e-7 cosine displacement.
                    if chart==0: grid[0] += 1e-7/max(np.cos(bounds[0]),1e-6)
                    if chart==2: grid[-1] -= 1e-7/max(np.cos(bounds[1]),1e-6)
                    degrees = np.unique(np.rad2deg(grid).astype(np.float32))
                    start = len(queries)
                    queries.extend(dict(altitude_km=h,sun_elevation_deg=sun_deg,
                        view_elevation_deg=float(e),relative_azimuth_deg=azimuth) for e in degrees)
                    if h>=120 and chart==0:
                        vacuum.append(len(queries)-1)
                    curves.append(dict(split=split,altitude_km=h,sun_elevation_deg=sun_deg,
                        azimuth_deg=azimuth,chart=chart,group=group,start=start,end=len(queries)))
    plan = dict(kind='dense_sky_coordinate_curves_v1',curves=curves,queries=queries,
                vacuum_query_indices=vacuum,
                geometry=dict(bottom_km=6360,top_height_km=120),points_per_curve=args.points,
                mapping=calibration,split_policy='Disjoint physical altitude/Sun poses; all azimuths from a pose share its split',
                note='Only teacher total radiance is needed. No repeated single-scattering path integrations.')
    (args.out/'queries.json').write_text(json.dumps(plan,separators=(',',':'))+'\n')
    (args.out/'baseline_mapping.json').write_text(json.dumps(calibration,indent=2)+'\n')
    print(json.dumps(dict(curves=len(curves),queries=len(queries),out=str(args.out),gpu_initialized=False)))

def predict(curve, calibration, size, stride=1):
    h, chart = curve['altitude_km'], curve['chart']
    bounds, centers, weights, widths, _ = parameters(calibration,h,np.deg2rad(curve['sun_elevation_deg']),chart)
    sky = size*3//4
    count = sky if h>=120 and chart==0 else size-sky if chart==2 else sky//2
    node_e = inverse(np.linspace(0,1,count),bounds,centers,weights,widths)
    e, rgb = curve['e'], curve['rgb']
    node_rgb = np.array([np.interp(node_e,e,rgb[:,k]) for k in range(3)]).T
    log_rgb = np.log(np.maximum(node_rgb,1e-30))
    x = cdf(e[::stride],bounds,centers,weights,widths)*(count-1)
    i = np.minimum(x.astype(int),count-2)
    t = x-i
    result = np.exp(log_rgb[i]*(1-t[:,None])+log_rgb[i+1]*t[:,None])-1e-30
    if h>=120 and chart==0:
        # Mirror the production last-cell chord extrapolation. Plain log
        # interpolation into its zero endpoint would bias the fitted CDF.
        radius = 6360+h
        c = (h-120)*(radius+6480)
        d = np.maximum((radius*np.sin(e[::stride]))**2-c,0)
        d0 = max((radius*np.sin(node_e[-2]))**2-c,1e-20)
        d1 = max((radius*np.sin(node_e[-3]))**2-c,d0+1e-10)
        change = np.exp(log_rgb[-3]-log_rgb[-2])*np.sqrt(d0/d1)-1
        tail = node_rgb[-2]*np.sqrt(np.clip(d/d0,0,1))[:,None]*(1+change*(d-d0)[:,None]/(d1-d0))
        result = np.where((i==count-2)[:,None],tail,result)
        result[e[::stride]>=bounds[1]]=0
    return np.maximum(result,0)

def errors(curves, calibration, size, stride):
    values=[]
    for curve in curves:
        truth=curve['rgb'][::stride]
        norm=np.linalg.norm(truth,axis=1)
        # Match the established radiance audits' absolute floor. Per-curve
        # concentration also prevents a very bright aureole from dominating.
        # Numerically tiny night/vacuum values must not drive node allocation.
        floor=max(float(np.quantile(np.linalg.norm(curve['rgb'],axis=1),.99))*1e-5,1e-7)
        valid=norm>1e-8
        delta=np.linalg.norm(predict(curve,calibration,size,stride)-truth,axis=1)/np.maximum(norm,floor)
        values.append(delta[valid])
    result=np.concatenate(values) if values else np.zeros(1)
    return result if result.size else np.zeros(1)

def metrics(values):
    return dict(zip(['p50','p95','p99','max'],(np.quantile(values,[.5,.95,.99,1])*100).tolist()),
                rms=float(np.sqrt(np.mean(values**2))*100))

def fit(args):
    if args.out.exists(): raise ValueError('Output exists; preserve frozen inputs')
    args.out.mkdir(parents=True)
    plan=json.loads((args.dataset/'queries.json').read_text())
    meta=json.loads((args.dataset/'dataset.json').read_text())
    rgb=np.fromfile(args.dataset/'rgb.f32',dtype='<f4').reshape(-1,3).astype(float)
    if rgb.shape[0]!=len(plan['queries']) or not np.isfinite(rgb).all(): raise ValueError('Incomplete/nonfinite dataset')
    curves=[]
    for row in plan['curves']:
        start,end=row['start'],row['end']
        e=np.deg2rad([q['view_elevation_deg'] for q in plan['queries'][start:end]])
        values=np.maximum(rgb[start:end].copy(),0)
        if row['altitude_km']>=120 and row['chart']==0:
            # The mathematical outer tangent is vacuum. A f32 angle cannot in
            # general represent it exactly; raw lookup may return a tiny chord.
            # Enforce that declared endpoint, rather than fitting a discontinuity.
            e[-1]=geometry(row['altitude_km'],0)[0][1]
            values[-1]=0
        curves.append(dict(row,e=e,rgb=values))
    baseline=plan['mapping']
    rng=np.random.default_rng(args.seed)
    reports=[]
    start_time=time.perf_counter()
    for size in args.sizes:
        candidate=copy.deepcopy(baseline)
        group_reports={}
        for group in ['low','upper','space','ground','ground_near']:
            if args.sky_only and group in ['ground','ground_near']:
                # Preserve the deterministic sky search sequence from the
                # original all-chart audit, without fitting ground parameters.
                for _ in range(args.iterations):
                    rng.normal(0,1,4);rng.normal(0,1,3)
                continue
            train=[c for c in curves if c['split']=='train' and c['group']==group]
            old=candidate['sky'][group]
            def objective(mapping):
                v=errors(train,mapping,size,8)
                return float(np.sqrt(np.mean(v*v))+.35*np.quantile(v,.95)+.20*np.quantile(v,.99))
            best_score=objective(candidate)
            initial_weights=np.array(old['weights'])
            initial_widths=np.array(old['widths'])
            for iteration in range(args.iterations):
                proposal=copy.deepcopy(candidate)
                scale=.35*(1-iteration/max(args.iterations,1))+.04
                center=candidate['sky'][group] if iteration%4 else old
                weights=np.array(center['weights'])*np.exp(rng.normal(0,scale,4))
                weights=weights/weights.sum()*.98+.005
                widths=np.array(center['widths'])*np.exp(rng.normal(0,scale,3))
                widths=np.clip(widths,initial_widths*.4,initial_widths*2.5)
                proposal['sky'][group]=dict(old,weights=weights.tolist(),widths=widths.tolist())
                score=objective(proposal)
                if score<best_score: candidate,best_score=proposal,score
            group_reports[group]=dict(train_before=metrics(errors(train,baseline,size,1)),
                train_after=metrics(errors(train,candidate,size,1)))
            heldout=[c for c in curves if c['split']=='validation' and c['group']==group]
            group_reports[group].update(validation_before=metrics(errors(heldout,baseline,size,1)),
                validation_after=metrics(errors(heldout,candidate,size,1)))
        report={'size':size,'groups':group_reports,'teacher_full_spectrum':meta['full_spectrum'],
                'seed':args.seed,'iterations_per_group':args.iterations,'sky_only':args.sky_only,
                'dataset_sha256':hashlib.sha256((args.dataset/'dataset.json').read_bytes()).hexdigest(),
                'queries_sha256':hashlib.sha256((args.dataset/'queries.json').read_bytes()).hexdigest(),
                'rgb_sha256':hashlib.sha256((args.dataset/'rgb.f32').read_bytes()).hexdigest(),
                'teacher_source_checksums':meta['source_checksums'],
                'runtime_formula':'unchanged four-weight, three-width softsign CDF; identical node count and shader operations',
                'interpretation':'Isolated vertical interpolation of teacher curves at fixed azimuth. Does not certify production integral/source or horizontal interpolation quality.',
                'endpoint_policy':'Sky/ground horizon samples lie on their own branch; space outer tangent is exact vacuum, not the nearby representable f32 teacher angle',
                'relative_error_policy':'Reference RGB norm >1e-8; denominator >=max(1e-7, each curve P99 RGB norm *1e-5). Exact vacuum is not fitted as a radiance discontinuity.',
                'elapsed_seconds':time.perf_counter()-start_time}
        for split in ['train','validation']:
            selected=[c for c in curves if c['split']==split]
            report[split]={'baseline_same_size':metrics(errors(selected,baseline,size,1)),
                           'fitted':metrics(errors(selected,candidate,size,1)),
                           'baseline256':metrics(errors(selected,baseline,256,1))}
            sky=[c for c in selected if c['chart']!=2]
            report[split]['sky_only']={'baseline_same_size':metrics(errors(sky,baseline,size,1)),
                                      'fitted':metrics(errors(sky,candidate,size,1)),
                                      'baseline256':metrics(errors(sky,baseline,256,1))}
        name=f'sky{size}'
        for group, values in group_reports.items():
            candidate['sky'][group]['train']=values['train_after']
            candidate['sky'][group]['heldout']=values['validation_after']
        candidate['sky_fit_provenance']={k:report[k] for k in ['size','seed','iterations_per_group',
            'sky_only','dataset_sha256','queries_sha256','rgb_sha256','teacher_source_checksums',
            'teacher_full_spectrum','runtime_formula','interpretation','endpoint_policy','relative_error_policy']}
        (args.out/f'{name}_mapping.json').write_text(json.dumps(candidate,indent=2)+'\n')
        reports.append(report)
        (args.out/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(json.dumps(dict(size=size,validation=report['validation'],elapsed_seconds=report['elapsed_seconds'])),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--mapping',type=Path,default=ROOT/'experiments/validation/sky_mapping_candidates_v7/baseline.json')
    p.add_argument('--points',type=int,default=4097)
    p.add_argument('--out',type=Path,required=True)
    p.set_defaults(run=prepare)
    p=sub.add_parser('fit')
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--sizes',type=int,nargs='+',default=[256,224,192])
    p.add_argument('--iterations',type=int,default=80)
    p.add_argument('--seed',type=int,default=20261008)
    p.add_argument('--sky-only',action='store_true',help='Fit low/upper/space only; preserve ground and ground_near')
    p.add_argument('--out',type=Path,required=True)
    p.set_defaults(run=fit)
    args=parser.parse_args()
    if args.command=='prepare' and not 257<=args.points<=16385:
        parser.error('--points must be in [257,16385]')
    if args.command=='fit' and (not 1<=args.iterations<=10000 or any(n<64 or n>1024 or n%8 for n in args.sizes)):
        parser.error('Use 1..10000 iterations and SkyView sizes divisible by eight in [64,1024]')
    args.run(args)

if __name__=='__main__':main()
