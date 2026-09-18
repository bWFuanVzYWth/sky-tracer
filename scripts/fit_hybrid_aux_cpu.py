"""Fit cheap auxiliary coordinates; RGB/optical fields and interpolation are f32.

SciPy optimizer bookkeeping is CPU-only. Sky curves use the frozen RGB teacher;
optical allocation uses curvature of physically integrated optical depth. The
result must also pass image/physical-path and fresh GPU validation.
"""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution

F = np.float32
ROOT = Path('out/hybrid_mapping_v3')

def forward(e, lo, hi, centers, weights, widths):
    def soft(d, w): return d/(abs(d)+w)
    result=weights[0]*(e-lo)/np.maximum(hi-lo,F(1e-10))
    for j in range(3):
        a=soft(lo-centers[...,j,None],widths[j])
        b=soft(hi-centers[...,j,None],widths[j])
        # Runtime evaluates this same rational CDF with a cancellation-free
        # identity when the center is outside the chart (far-space extrapolation).
        result += weights[j+1]*(soft(e-centers[...,j,None],widths[j])-a)/np.maximum(b-a,F(1e-10))
    return np.clip(result,0,1)

def parameters(p):
    weights=np.exp(np.r_[F(0),p[:3]].astype(F));weights/=weights.sum()
    return weights,np.exp(np.asarray(p[3:],dtype=F))

def fit_sky(groups=None):
    root=ROOT/'sky_curves';meta=json.loads((root/'audit.json').read_text())['scenes']
    result={}
    for group in groups or ['low','upper','ground','space','ground_near']:
        all_curves=[];centers=[];bounds=[];train=[];counts=[]
        chart={'low':0,'upper':1,'ground':2,'space':0,'ground_near':2}[group]
        for si,s in enumerate(meta):
            if group=='ground_near' and s['h']>1:continue
            if (s['h']>=120) != (group=='space') and group!='ground':continue
            lo,hi=s['bounds'][chart]
            if hi-lo<1e-5:continue
            values=np.fromfile(root/(s['scene']+'.curves.f32'),dtype=F).reshape(s['shape'])[chart]
            for v in values:
                all_curves.append(v);bounds.append([lo,hi]);centers.append([s['horizon'],s['solar'],s['outer']])
                train.append(si%3!=1);counts.append(192 if group=='space' else 64 if group.startswith('ground') else 96)
        data=np.array(all_curves,dtype=F);centers=np.array(centers,dtype=F);bounds=np.array(bounds,dtype=F)
        train=np.array(train);counts=np.array(counts)
        def measure(p,mask,stride):
            weights,widths=parameters(p);d=data[mask];b=bounds[mask];cs=centers[mask];n=counts[mask][0]
            coords=forward(d[:,:,0],b[:,0,None],b[:,1,None],cs,weights,widths)
            predictions=[];truth=d[:,::stride,1:]
            for v,u in zip(d,coords):
                un=np.linspace(0,1,n,dtype=F)
                nodes=np.array([np.interp(un,u,v[:,k]) for k in range(1,4)],dtype=F)
                predictions.append(np.array([np.interp(u[::stride],un,k) for k in nodes],dtype=F).T)
            # Relative RGB, computed in a per-point shifted log domain.
            shift=truth.max(-1,keepdims=True);t=np.exp(truth-shift)
            pred=np.exp(np.minimum(np.array(predictions)-shift,F(30)))
            e=np.linalg.norm(pred-t,axis=-1)/np.maximum(np.linalg.norm(t,axis=-1),F(1e-20))
            valid=truth.max(-1)>F(-18)
            e=e[valid]
            return e
        def objective(p):
            e=measure(p,train,16)
            return float(np.mean(np.minimum(e,F(1))**2)**.5+.3*np.percentile(e,95)+.05*np.percentile(e,99))
        initial=np.r_[np.log(np.array([.4,.35,.1])/.15),np.log(np.deg2rad([1,.7,.5]))]
        parameter_bounds=[(-2.5,2.5)]*3+[(np.log(np.deg2rad(.1)),np.log(np.deg2rad(12)))]*3
        if group=='ground_near':
            # Preserve a grazing-distance density prior for the ground chart.
            parameter_bounds[3]=(np.log(np.deg2rad(.03)),np.log(np.deg2rad(1)))
            parameter_bounds[5]=(np.log(np.deg2rad(.03)),np.log(np.deg2rad(1)))
        opt=differential_evolution(objective,parameter_bounds,
                                   seed=19,popsize=5,maxiter=24,polish=False,x0=initial,workers=1)
        w,wi=parameters(opt.x)
        def stats(e):return dict(p95=float(np.percentile(e,95)*100),p99=float(np.percentile(e,99)*100),rms=float(np.mean(e*e)**.5*100))
        result[group]=dict(weights=w.tolist(),widths=wi.tolist(),train=stats(measure(opt.x,train,2)),heldout=stats(measure(opt.x,~train,2)))
        print(group,result[group],flush=True)
    return result

def fit_optical():
    a=np.fromfile(ROOT/'optical_curves/optical_curves.f32',dtype=F).reshape(6,129,129,6)
    tau=a[...,2:];d2=(tau[:,2:]-2*tau[:,1:-1]+tau[:,:-2])*F(128**2)
    # Relative T error is approximately |delta tau|. Exclude effectively opaque
    # lanes, balance segments equally, and retain a floor for thin high layers.
    active=np.exp(-tau[:,1:-1])>F(1e-4)
    cost=np.array([np.mean(np.minimum(abs(d[ok]),F(1e4))**2)**.2 for d,ok in zip(d2,active)])
    count=np.maximum(8,np.round(255*cost/cost.sum())).astype(int)
    while count.sum()!=255:
        i=np.argmax(cost/np.maximum(count,1)) if count.sum()<255 else np.argmax(count/np.maximum(cost,1e-8))
        count[i]+=1 if count.sum()<255 else -1
    return dict(anchors_km=[0,1,2,11,12,35,120],indices=np.r_[0,np.cumsum(count)].tolist(),curvature_cost=cost.tolist(),angle_power=3)

def main():
    if '--ground-near' in sys.argv:
        p=Path('crates/sky-hybrid-atmosphere/configs/mapping_fit.json');fit=json.loads(p.read_text())
        fit['sky'].update(fit_sky(['ground_near']))
        (ROOT/'aux_fit.json').write_text(json.dumps(fit,indent=2));p.write_text(json.dumps(fit,indent=2));return
    optical=fit_optical();print('optical',optical,flush=True)
    sky=fit_sky()
    source=json.loads(Path('out/hybrid_allocation_v2/source_fit176/fit.json').read_text())['results']
    fit=dict(source_height=source['height']['candidate']['nodes_km'],
             solar_weights=source['solar']['candidate']['weights'],solar_widths=source['solar']['candidate']['widths_radians'],
             phase_weight=.61,cone_warp=.164285714,optical=optical,sky=sky)
    (ROOT/'aux_fit.json').write_text(json.dumps(fit,indent=2))
    Path('crates/sky-hybrid-atmosphere/configs/mapping_fit.json').write_text(json.dumps(fit,indent=2))
if __name__=='__main__':main()
