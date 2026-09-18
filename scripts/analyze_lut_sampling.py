"""Summarize CPU-only axis probes and node allocation of the existing v4 LUT."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

root=Path("out/lut_sampling_audit")
nodes=json.loads((root/"nodes.json").read_text())
def read(name):
    return np.genfromtxt(root/name,delimiter=",",names=True,dtype=None,encoding="utf8")
d=read("axis_probes.csv")
interior=read("interior_probes.csv")
columns=["height_only","view_only","solar_only","phase_only","direct256"]
def relative_l2(data,mask,column):
    ref=data["direct"][mask]
    den=np.linalg.norm(ref)
    return float(np.linalg.norm(data[column][mask]-ref)/den) if den>0 else None
groups=[
    ("Noon horizon",d,(d["case"]=="ground_sweep")&(d["sun_deg"]>=60)&(d["above_horizon_deg"]<=3)),
    ("Twilight horizon/sky",d,(d["case"]=="ground_sweep")&(d["sun_deg"]<0)),
    ("Sun aureole",d,d["case"]=="aureole"),
    ("Space / 400 km",d,d["case"]=="orbit"),
    ("Interior / 0.2 km",interior,interior["observer_km"]==.2),
    ("Interior / 60 km",interior,interior["observer_km"]==60),
]
summary={"probe_count":len(d)+len(interior),"groups":[]}
for name,data,mask in groups:
    summary["groups"].append(dict(name=name,points=int(mask.sum()),
        relative_l2={c:relative_l2(data,mask,c) for c in ["first"]+columns}))

height=np.array(nodes["height_km"])
solar=np.array(nodes["solar"][1]["elevation_deg"])
phase=np.array(nodes["phase_angle_deg"])
height_gaps=[]
for h in [.2,1,2,10,30,60,100]:
    i=np.clip(np.searchsorted(height,h)-1,0,len(height)-2)
    height_gaps.append(dict(query_km=h,lo_km=float(height[i]),hi_km=float(height[i+1]),gap_km=float(height[i+1]-height[i])))
summary["height_cells"]=height_gaps
summary["solar_interval_counts"]=[dict(low=a,high=b,count=int(((solar>=a)&(solar<b)).sum())) for a,b in [(-12,-4),(-4,0),(0,5),(5,20),(20,45),(45,61.4),(61.4,80),(80,85),(85,90.1)]]
summary["phase_cells"]=[]
for angle in [.25,1,5,20,45,60,75,80,85,90,100,120,150]:
    i=np.clip(np.searchsorted(phase,angle)-1,0,len(phase)-2)
    summary["phase_cells"].append(dict(angle=angle,lo=float(phase[i]),hi=float(phase[i+1]),gap=float(phase[i+1]-phase[i])))
summary["point_phase_midpoint_max_error"]={}
for p in nodes["phase"]:
    values=np.array(p["node_phase"])
    exact=np.array(p["mid_phase"])
    summary["point_phase_midpoint_max_error"][str(p["nm"])]=float(np.max(abs((values[:-1]+values[1:])*.5/exact-1)))
summary["sky_clamped_phase_fraction_by_height"]=[]
for s in nodes["solar"]:
    h=s["altitude_km"]
    horizon=-np.sqrt(h*(2*6360+h))/(6360+h)
    beta=np.rad2deg(np.arccos(horizon))
    alpha=90-np.array(s["elevation_deg"])
    low=np.maximum(alpha-beta,0)
    high=np.minimum(alpha+beta,180)
    outside=(phase[None,:]<low[:,None]-1e-4)|(phase[None,:]>high[:,None]+1e-4)
    summary["sky_clamped_phase_fraction_by_height"].append(dict(height_km=h,fraction=float(outside.mean())))

angular=read("angular_source.csv")
summary["frozen_source_angular"]=[]
for i in range(0,len(angular),6):
    block=angular[i:i+6]
    ref=block[-1]["source"]
    summary["frozen_source_angular"].append(dict(height_km=float(block[0]["height_km"]),sun_deg=float(block[0]["sun_deg"]),azimuth_deg=float(block[0]["azimuth_deg"]),
        relative_to_128x256={f'{int(r["n_mu"])}x{int(r["n_phi"])}':float(r["source"]/ref-1) for r in block[:-1]}))
summary["view_axis_dim_region_examples"]=[]
valid=interior["direct"]>1e-6
v=interior[valid]
errors=abs(v["view_only"]/v["direct"]-1)
for r in v[np.argsort(errors)[-6:]]:
    summary["view_axis_dim_region_examples"].append({**{k:float(r[k]) for k in ["observer_km","sun_deg","azimuth_deg","above_horizon_deg","direct","full","q_view"]},
        "view_only_relative_error":float(r["view_only"]/r["direct"]-1),"direct_over_baked_full":float(r["direct"]/r["full"])})
tau=read("optical_depth.csv")
mask=tau["tau8192"]<10
t=tau[mask]
te=abs(np.exp(-(t["tau_lut"]-t["tau8192"]))-1)
summary["optical_depth"]={"points":len(tau),"points_tau_below_10":int(mask.sum()),"max_abs_tau_error":float(np.max(abs(tau["tau_lut"]-tau["tau8192"]))),
    "max_abs_tau_step_error":float(np.max(abs(tau["tau2048"]-tau["tau8192"]))),"max_relative_transmittance_error_tau_below_10":float(te.max()),"worst_examples":[]}
for r in t[np.argsort(te)[-6:]]:
    summary["optical_depth"]["worst_examples"].append({**{k:float(r[k]) for k in ["height_km","above_horizon_deg","ground","tau8192"]},
        "relative_transmittance_error":{c:float(np.exp(-(r[c]-r["tau8192"]))-1) for c in ["tau_lut","tau_height_only","tau_view_only"]}})
(root/"summary.json").write_text(json.dumps(summary,indent=2))

fig,axes=plt.subplots(2,2,figsize=(13.5,9))
ax=axes[0,0]
ax.plot((solar[:-1]+solar[1:])*.5,np.diff(solar),"o-",ms=3)
ax.set(xlim=(-15,90),yscale="log",ylim=(.02,12),xlabel="Solar elevation (degrees)",ylabel="Adjacent solar-node spacing (degrees)",title="97 solar nodes at observer height 200 m")
ax.axvline(solar[62],color="#bf683d",ls="--",label="Zenith-cap join")
ax.legend(fontsize=8)
ax=axes[0,1]
ax.plot((phase[:-1]+phase[1:])*.5,np.diff(phase),"-",color="#99702d")
ax.set(xlim=(0,180),ylim=(.04,3),yscale="log",xlabel="Scattering angle (degrees)",ylabel="Adjacent phase-node spacing (degrees)",title="256 scattering-angle nodes")
ax=axes[1,0]
ax.plot((height[:-1]+height[1:])*.5,np.diff(height),"-",color="#397068")
ax.set(yscale="log",xlabel="Altitude (km)",ylabel="Adjacent height-node spacing (km)",title="64 height nodes")
ax=axes[1,1]
values=np.array([[row["relative_l2"][c]*100 for c in columns] for row in summary["groups"]])
im=ax.imshow(values,norm=LogNorm(vmin=.005,vmax=40),cmap="YlOrRd",aspect="auto")
ax.set_xticks(range(5),["Height","Cone view","Solar","Phase","256 steps"],rotation=20)
ax.set_yticks(range(len(groups)),[g[0] for g in groups],fontsize=8)
for j in range(values.shape[0]):
    for i in range(5):
        ax.text(i,j,f"{values[j,i]:.3g}",ha="center",va="center",fontsize=8,color="white" if values[j,i]>4 else "black")
ax.set_title("Isolated first-order relative L2 (%)\nNon-additive probes, 550 nm")
fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
for ax in axes.flat:
    if ax!=axes[1,1]:ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(root/"allocation_and_axis_error.png",dpi=160)

profiles=read("dense_profiles.csv")
fig,axes=plt.subplots(3,2,figsize=(13,9))
for row,sun in enumerate([60,80,85]):
    a=profiles[(profiles["sun_deg"]==sun)&(profiles["azimuth_deg"]==0)]
    x=a["above_horizon_deg"]
    y=a["full"]
    axes[row,0].plot(x,y/y[0],color="#267080")
    axes[row,1].plot((x[1:]+x[:-1])*.5,np.diff(np.log(y))/np.diff(x),color="#267080",lw=1)
    knots=np.where(np.diff(np.floor(a["q_phase"]))!=0)[0]
    for i in knots:
        axes[row,1].axvline(x[i],color="#b5743f",alpha=.25,lw=.8)
    axes[row,0].set(ylabel="Radiance / horizon radiance",title=f"Solar elevation {sun} degrees",xlim=(0,10))
    axes[row,1].set(ylabel="d log(radiance) / degree",title="Gradient; orange lines = phase-cell boundaries",xlim=(0,10))
    for ax in axes[row]:ax.grid(alpha=.15)
for ax in axes[-1]:ax.set_xlabel("Degrees above geometric horizon")
fig.suptitle("Current v4 full-transport LUT, 550 nm, observer 200 m, solar-facing view\nCPU queries of existing data; curve changes align with interpolation-cell boundaries")
fig.tight_layout()
fig.savefig(root/"noon_gradients.png",dpi=160)
print(json.dumps(summary,indent=2))
