"""Summarize measured 176-state GPU comparison and CPU auxiliary-table probes."""
import json
from pathlib import Path
import numpy as np
from fit_hybrid_mapping_cpu import F, Dataset, interp, invert, solar_old, solar_angle_fast, old_heights

ROOT=Path("out/hybrid_allocation_v2")

def stat(a,b,relative_floor=1e-7):
    norm=np.linalg.norm(b,axis=-1)
    e=100*np.linalg.norm(a-b,axis=-1)/np.maximum(norm,F(relative_floor))
    valid=norm>F(relative_floor*.1)
    return dict(p95=float(np.percentile(e[valid],95)),p99=float(np.percentile(e[valid],99)),max=float(e[valid].max()))

def image(path):
    return np.fromfile(path,dtype=F).reshape(-1,4)[:,:3]

def main():
    output={}
    cases=json.loads(Path("out/four_wave_sun_v1/queries.json").read_text())["images"]
    rows=[]
    for s in cases:
        name=s["name"]
        old=image(ROOT/"s177"/f"{name}_source.f32")
        new=image(ROOT/"s176"/f"{name}_source.f32")
        dense=image(Path("out/hybrid_v1/dense_final")/f"{name}_source.f32")
        sky=image(ROOT/"s176"/f"{name}_sky.f32")
        rows.append(dict(scene=name,change_176_vs_177=stat(new,old),old_vs_dense=stat(old,dense),new_vs_dense=stat(new,dense),sky_vs_direct=stat(sky,new)))
    output["solar_gpu"]=rows
    output["solves"]={str(n):json.loads((ROOT/f"s{n}"/"solve.json").read_text())["solve"] for n in [176,177]}
    root=ROOT/"aux_final";meta=json.loads((root/"audit.json").read_text())
    q=np.fromfile(root/"optical_queries.f32",dtype=F).reshape(-1,10)
    truth=np.exp(-q[:,6:]);mask=truth>F(1e-4)
    def tstat(a):
        e=100*abs(a-truth)/np.maximum(truth,F(1e-4))
        i,k=np.unravel_index(np.argmax(np.where(mask,e,0)),e.shape)
        return dict(p95=float(np.percentile(e[mask],95)),p99=float(np.percentile(e[mask],99)),max=float(e[mask].max()),
            max_absolute=float(abs(a-truth).max()),worst=dict(height_km=float(q[i,0]),elevation_deg=float(np.rad2deg(np.arcsin(q[i,1]))),lane=int(k)))
    output["optical"]={"queries":len(q),"reference_4096_vs_8192":tstat(np.exp(-q[:,2:6])),"variants":[]}
    for v in meta["variants"]:
        values=np.fromfile(root/(v["id"]+".f32"),dtype=F).reshape(-1,4)
        output["optical"]["variants"].append(v|tstat(np.exp(-values)))
    output["phase"]=meta["phase"]
    output["sky_cpu"]=json.loads((ROOT/"sky_cpu/audit.json").read_text())

    # Ground irradiance has its own smoothness and need not use the source's
    # optimal density. This remains a frozen finite-Sun teacher, not a fresh solve.
    resource=Path("out/four_wave_source_v1")
    r=json.loads((resource/"resource.json").read_text())
    aux=np.fromfile(resource/"aux.f32",dtype=F).reshape(-1,4)
    ground=aux[r["aux_offsets"][5]:r["aux_offsets"][5]+r["ground_count"]][None]
    old_u=np.linspace(0,1,r["ground_count"],dtype=F)[None]
    query_u=np.linspace(0,1,16385,dtype=F)[None]
    mu=invert(lambda x:solar_old(x,F(0)),query_u,F(-1),F(1))
    truth=interp(old_u,ground,solar_old(mu,F(0)))[0]
    fit=json.loads((ROOT/"source_fit176/fit.json").read_text())["results"]
    params=fit["solar"]["candidate"]["parameters"]
    output["ground"]=[]
    for n in [96,176,256,512]:
        for label,fun in [("current",lambda x:solar_old(x,F(0))),("source_fitted",lambda x:solar_angle_fast(x,F(0),params))]:
            u=np.linspace(0,1,n,dtype=F)[None]
            nodes=invert(fun,u,F(-1),F(1))
            values=interp(old_u,ground,solar_old(nodes,F(0)))
            for logarithmic in [False,True]:
                payload=np.log(np.maximum(values,F(1e-30))) if logarithmic else values
                prediction=interp(u,payload,fun(mu))[0]
                if logarithmic:prediction=np.exp(prediction)
                output["ground"].append(dict(nodes=n,mapping=label,log=logarithmic,
                    **stat(prediction,truth,max(float(np.linalg.norm(truth,axis=-1).max())*1e-6,1e-20))))

    curves_root=Path("out/hybrid_mapping_curves_shifted_v1")
    metadata=json.loads((curves_root/"curves.json").read_text())
    output["log_mean_axes"]={}
    for axis in ["height","solar"]:
        spec=next(x for x in metadata["datasets"] if x["axis"]==axis)
        # Mean brightness does not depend on outgoing direction: one per pose.
        poses=[];ids=[]
        for i,c in enumerate(spec["curves"]):
            key=c["pose"][1] if axis=="height" else c["pose"][0]
            if key not in poses:poses.append(key);ids.append(i)
        d=Dataset(curves_root,spec,ids)
        d.y[:,:,:4]=F(1);d.query_y=d.y;d.truth=d.decode(d.y)
        d.norm=np.linalg.norm(d.truth,axis=-1)
        d.floor=np.maximum(d.norm.max(axis=1,keepdims=True)*F(1e-5),F(1e-25))
        d.active=(d.norm>d.floor)&(d.shift[:,0,0,None]>F(-35));d.den=np.maximum(d.norm,d.floor)
        if axis=="height":
            old=d.predicted(nodes=old_heights());new=d.predicted(nodes=fit[axis]["candidate"]["nodes_km"])
        else:
            old=d.predicted(lambda x:solar_old(x,d.pose[:,0,None]),176)
            new=d.predicted(lambda x:solar_angle_fast(x,d.pose[:,0,None],params),176)
        output["log_mean_axes"][axis]=dict(baseline=d.metrics(old),candidate=d.metrics(new))

    (ROOT/"summary.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print('176 vs 177 worst',sorted(rows,key=lambda x:x['change_176_vs_177']['p95'],reverse=True)[:3])
    print('ground',output['ground'])
    print('mean',[(k,v['baseline']['p95_percent'],v['candidate']['p95_percent']) for k,v in output['log_mean_axes'].items()])
    # Reference size; low-height count 24 is unchanged by this turn.
    n=176
    output_sizes=dict(ms_source=24*n*24*8*8,log_mean=48*n*16,high_moments=25*n*7*16,
                      ground=n*16,optical=256*1024*8,sky_view=256*256*16,phase=4096*16)
    print('payloads',output_sizes)
if __name__=="__main__":main()
