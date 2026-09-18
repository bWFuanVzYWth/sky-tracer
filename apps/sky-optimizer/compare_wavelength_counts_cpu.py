"""CPU/f32 exhaustive 3/4-node search and fixed 540nm-centered 8-node audit.

Best means the global minimum of a stated training objective on the 41-node
teacher grid, with positive scalar weights and exact reference solar RGB.
Validation and dense image pixels never enter that exhaustive training search.
"""
import argparse
import itertools
import json
import time
from pathlib import Path
from search_wavelengths_cpu import F, REGIONS, Objective, feature_basis, metrics, read_dataset
import numpy as np


def exhaustive_geometry(C, count):
    indices = np.array(list(itertools.combinations(range(C.shape[1]), count)), dtype=np.int32)
    all_count = len(indices)
    matrices = C[:, indices].transpose(1, 0, 2)
    u, s, vh = np.linalg.svd(matrices, full_matrices=True)
    rank_ok = s[:, -1] > F(1e-7)
    indices, matrices, u, s, vh = [v[rank_ok] for v in (indices, matrices, u, s, vh)]
    # U^T times [1,1,1] is the sum of each column of U.
    x0 = np.einsum('nki,nk->ni', vh[:, :3, :], u.sum(axis=1, dtype=F) / s)
    if count == 3:
        feasible = (x0.min(axis=1) >= F(.02)) & (x0.max(axis=1) <= F(20))
        null, lo, hi = None, None, None
    else:
        # One remaining degree of freedom: intersect all positive-weight bounds.
        null = vh[:, 3, :]
        active = np.abs(null) > F(1e-10)
        safe = np.where(active, null, F(1))
        first = (F(.02)-x0) / safe
        second = (F(20)-x0) / safe
        lo = np.max(np.where(active, np.minimum(first, second), -np.inf), axis=1)
        hi = np.min(np.where(active, np.maximum(first, second), np.inf), axis=1)
        feasible = (lo <= hi) & np.all(active | ((x0 >= F(.02)) & (x0 <= F(20))), axis=1)
        null, lo, hi = null[feasible], lo[feasible], hi[feasible]
    return dict(count=count, enumerated=all_count, rank_valid=int(np.count_nonzero(rank_ok)),
                indices=indices[feasible], x0=x0[feasible], null=null, lo=lo, hi=hi)


def solve_all(B, geometry):
    indices, x0 = geometry['indices'], geometry['x0']
    sub = B[:, indices].transpose(1, 0, 2)
    target = B.sum(axis=1, dtype=F)
    residual0 = np.einsum('nki,ni->nk', sub, x0) - target
    if geometry['count'] == 3:
        weights, residual = x0, residual0
    else:
        direction = np.einsum('nki,ni->nk', sub, geometry['null'])
        denom = np.sum(direction*direction, axis=1, dtype=F)
        optimum = -np.sum(direction*residual0, axis=1, dtype=F) / np.maximum(denom,F(1e-30))
        z = np.clip(optimum, geometry['lo'], geometry['hi'])
        weights = x0 + geometry['null']*z[:, None]
        residual = residual0 + direction*z[:, None]
    loss = np.sum(residual*residual, axis=1, dtype=F)
    assert weights.dtype == loss.dtype == np.float32
    assert np.isfinite(weights).all() and np.isfinite(loss).all()
    assert weights.min() >= F(.01999) and weights.max() <= F(20.00001)
    return loss, weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset',type=Path)
    p.add_argument('--base',type=Path,help='Optional eight-wave search result for comparison')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise ValueError('use a new output directory')
    a.out.mkdir(parents=True)
    start=time.perf_counter()
    meta, plan, data=read_dataset(a.dataset)
    previous=json.loads((a.base/'search.json').read_text()) if a.base else None
    if previous is not None:
        assert meta['source_band_checksums']==previous['source_band_checksums']
    numerical_n=next(i for i,l in enumerate(plan['labels']) if l['split']=='visual')
    data={k:v[:numerical_n] for k,v in data.items()}
    split=np.array([l['split'] for l in plan['labels'][:numerical_n]])
    region=np.array([l['region'] for l in plan['labels'][:numerical_n]])
    pairs=np.array(plan['adjacent_pairs'],dtype=np.int32)
    W=np.array(meta['rgb_from_integrated'],dtype=F)
    total_rgb=data['total']@W
    norm=np.linalg.norm(total_rgb,axis=1)
    train=(split=='train')&(norm>F(1e-8))
    nm=np.array([b['center_nm'] for b in meta['bands']],dtype=F)
    widths=np.array([b['upper_nm']-b['lower_nm'] for b in meta['bands']],dtype=F)
    solar=np.array([b['solar_irradiance_w_m2'] for b in meta['bands']],dtype=F)
    C=(solar[:, None]*W/(solar@W)[None, :]).T
    names=['balanced','twilight','aureole','gradient','general']
    bases={name:feature_basis(data,W,norm,region,train,pairs,name) for name in names}

    def candidate(indices,x,**extras):
        indices=list(map(int,indices));x=np.asarray(x,dtype=F)
        return dict(indices=indices,wavelengths_nm=nm[indices].tolist(),
                    quadrature_weights_nm=(x*widths[indices]).tolist(),
                    rgb_from_integrated=(x[:, None]*W[indices]).tolist(),
                    constraint_max_residual=float(np.max(np.abs(C[:,indices]@x-F(1)))),**extras)

    fixed_nm=[400,440,480,520,560,600,640,680]
    fixed=tuple(int(np.flatnonzero(nm==F(n))[0]) for n in fixed_nm)
    # Both literal 40nm quadrature and fitted weights on the same equally spaced nodes.
    equal=candidate(fixed,F(40)/widths[list(fixed)],id='U8_equal',role='literal equal 40nm weights',
                    exact_solar_white=False)
    obj=Objective('balanced',bases['balanced'],C)
    fit=obj.bounded_baseline(fixed)
    if fit is None:
        raise RuntimeError('fixed equally spaced nodes have no feasible positive fit')
    fitted=candidate(fixed,fit[1],id='U8_fit',role='fixed equal spacing, optimized weights',
                     train_loss=fit[0],exact_solar_white=True)
    selected=[equal,fitted]
    pool=[];summaries=[];balanced_best={}
    for count in [4,3]:
        g=exhaustive_geometry(C,count)
        summary={k:g[k] for k in ['count','enumerated','rank_valid']}
        summary['positive_feasible']=len(g['indices'])
        summary['objectives']=[]
        # Keep exhaustive raw scores/weights as reproducible evidence of the grid optimum.
        saved=dict(indices=g['indices'])
        for name,B in bases.items():
            loss,weights=solve_all(B,g)
            constraint_error=np.max(np.abs(np.einsum('cni,ni->nc',C[:,g['indices']],weights)-F(1)))
            assert constraint_error < F(5e-5),constraint_error
            order=np.argsort(loss,kind='stable')[:64]
            best=int(order[0])
            summary['objectives'].append(dict(name=name,minimum=float(loss[best]),
                wavelengths_nm=nm[g['indices'][best]].tolist(),constraint_max_residual=float(constraint_error)))
            saved[name+'_loss']=loss;saved[name+'_weights']=weights
            for index in order:
                item=candidate(g['indices'][index],weights[index],count=count,objective=name,
                    train_loss=float(loss[index]),exact_solar_white=True)
                pool.append(item)
                if name=='balanced' and index==best:
                    balanced_best[count]=item
            print(f'{count} wavelengths / {name}: exhaustive {g["enumerated"]}, feasible {len(loss)}, best {nm[g["indices"][best]].tolist()}, loss {loss[best]:.8g}',flush=True)
        np.savez_compressed(a.out/f'exhaustive_{count}.npz',**saved)
        summaries.append(summary)

    baselines=[]
    if previous and previous['selected']:
        baseline8=dict(min(previous['selected'], key=lambda c:c['metrics']['worst_region_p99_percent']))
        baseline8['id']='R8_reference'
        baselines.append(baseline8)
    for item in baselines+selected+pool:
        item['metrics']=metrics(item['indices'],np.array(item['rgb_from_integrated'],dtype=F),data,W,total_rgb,norm,region,split,pairs)
    for count in [4,3]:
        best=dict(balanced_best[count]);best['id']=f'N{count}_best';best['role']='exhaustive balanced training optimum'
        selected.append(best)
        # Retain an alternative; prior user request explicitly cares about regional/visual tradeoffs.
        eligible=[c for c in pool if c['count']==count and c['indices']!=best['indices']]
        tail=dict(min(eligible,key=lambda c:c['metrics']['worst_region_p99_percent']))
        tail['id']=f'N{count}_tail';tail['role']='validation worst-region alternative from training shortlist'
        selected.append(tail)
    for item in baselines+selected:
        indices=item['indices']
        item['kind']='frozen_lut_spectral_candidate_v2'
        item['model']=meta['model'];item['source_band_checksums']=meta['source_band_checksums']
        item['wavelength_count']=len(indices)
        item['rec2020_from_per_nm']=(np.array(item['rgb_from_integrated'],dtype=F)*widths[indices,None]).tolist()
        item['sun_irradiance_per_nm']=(solar[indices]/widths[indices]).tolist()
        (a.out/(item['id']+'.json')).write_text(json.dumps(item,indent=2))
    result=dict(kind='current_lut_wavelength_count_comparison_v1',dataset=str(a.dataset),model=meta['model'],
        source_band_checksums=meta['source_band_checksums'],baselines=baselines,selected=selected,pool=pool,
        search=summaries,elapsed_seconds=time.perf_counter()-start,
        numeric_precision='float32 spectra, features, batched SVD, constrained least-squares and scoring',
        constraints='3 or 4 distinct nodes from 380..780nm in 10nm steps; scalar weights 0.2..200nm; exact solar RGB; no weight-sum constraint. U8_equal deliberately retains literal 40nm weights and does not enforce solar white.',
        optimality='For each objective all node combinations enumerated. 3-node weights fixed by solar RGB; 4-node remaining scalar optimized on complete feasible interval including bounds. Best is global on this grid for balanced training loss, not globally minimal validation percentiles or arbitrary continuous wavelengths.',
        error_definition='norm((S1+B)_N-(S1+B)_41)/max(norm(Ltotal_41),1e-7); exclude norms <=1e-8; percent',
        selection='N3_best/N4_best use train only. Tail alternatives selected on numerical validation from the training top64 per objective. Visual pixels excluded.',
        counts={s:int(np.count_nonzero(split==s)) for s in ['train','validation']},
        region_validation_counts={r:int(np.count_nonzero((split=='validation') & (region==r) & (norm>F(1e-8)))) for r in REGIONS})
    (a.out/'search.json').write_text(json.dumps(result,indent=2))
    for c in baselines+selected:
        print(c['id'],c['wavelengths_nm'],c['metrics']['direct_error_over_total_percent'],flush=True)
    print('seconds',result['elapsed_seconds'])


if __name__=='__main__':
    main()
