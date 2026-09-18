"""Summarize joint compression, including the independent physical query sweep."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names=["lut_16mb_candidate","lut_joint16_global_log1p","lut_joint16_global_log",
       "lut_joint16_local_log1p","lut_joint16_local_log","lut_joint16_tt_weighted_log1p",
       "lut_joint16_tt_weighted_log","lut_joint16_hybrid","lut_joint16_hybrid_hsv",
       "lut_joint16_hybrid_floor8"]
summary={"cpu_only":True,"selected":"lut_joint16_hybrid_floor8", "candidates":{},"fresh":{}}
for name in names:
    p=Path("out")/name
    summary["candidates"][name]=json.loads((p/"validation.json").read_text())

regions=json.loads(Path("out/lut_joint16_fresh_queries/regions.json").read_text())
reference=np.fromfile("out/lut_joint16_fresh_teacher/samples.f32",dtype="<f4").reshape(-1,3)
norm=np.linalg.norm(reference,axis=1)
for name in ["lut_16mb_candidate","lut_joint16_hybrid","lut_joint16_hybrid_floor8"]:
    decoded=np.fromfile(Path("out")/name/"fresh_query_rgb.f32",dtype="<f4").reshape(-1,3)
    absolute=np.linalg.norm(decoded-reference,axis=1)
    relative=absolute/np.maximum(norm,np.float32(1e-30))
    result={}
    for region,(begin,end) in regions.items():
        ids=np.arange(begin,end)
        thresholds={}
        for floor in [1e-8,1e-6,1e-4]:
            valid=ids[norm[ids]>floor]
            thresholds[str(floor)]={"count":len(valid),
                "p50_p95_p99_max":np.percentile(relative[valid],[50,95,99,100]).tolist(),
                "absolute_max":float(absolute[valid].max())}
        result[region]=thresholds
    summary["fresh"][name]=result

root=Path("out/lut_joint16_analysis");root.mkdir(exist_ok=True)
(root/"summary.json").write_text(json.dumps(summary,indent=2))
baseline=summary["candidates"]["lut_16mb_candidate"]
joint=summary["candidates"][summary["selected"]]
fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
cases=["noon","sunset","blue_hour","stratosphere_shadow","atmosphere_edge","orbit"]
labels=["Noon (whole scene)","Sunset","Blue hour","30 km shadow","120 km edge","400 km orbit"]
x=np.arange(len(cases))
for offset,report,label,color in [(-.18,baseline,"15.994 MB coarse grid","#bc8b5d"),(.18,joint,"15.701 MB joint + boundary","#287d83")]:
    v=[report["cases"][c]["vs_spectral"]["relative_p99"]*100 for c in cases]
    bars=axes[0].barh(x+offset,v,.34,label=label,color=color)
    axes[0].bar_label(bars,fmt="%.2f",padding=3,fontsize=8)
axes[0].set_yticks(x,labels);axes[0].invert_yaxis();axes[0].set_xlim(0,24)
axes[0].set_xlabel("P99 relative RGB error (%)")
axes[0].set_title("Original validation scenes")
axes[0].legend(loc="lower right",fontsize=8)

for color,name,label in [("#bc8b5d","lut_16mb_candidate","Coarse grid"),("#287d83",summary["selected"],"Joint + boundary")]:
    v=[summary["fresh"][name][r]["1e-08"]["p50_p95_p99_max"][2]*100 for r in regions]
    axes[1].plot(range(3),v,"o-",color=color,label=label)
    for i,value in enumerate(v):axes[1].annotate(f"{value:.2f}%",(i,value),xytext=(7,5 if name=="lut_16mb_candidate" else -14),textcoords="offset points",fontsize=9)
axes[1].set_xticks(range(3),["Dense solar halo","Moving Earth shadow","Upper atmosphere\nand space"])
axes[1].set_yscale("log");axes[1].set_ylim(.2,200);axes[1].set_xlim(-.1,2.35)
axes[1].set_ylabel("P99 relative RGB error (%)");axes[1].grid(alpha=.2)
axes[1].set_title("18,091 additional physical queries; includes dark tails")
axes[1].legend(loc="upper left",fontsize=8)
fig.suptitle("Joint 4D compression helps, but does not remove every local failure",fontsize=14)
fig.supxlabel("Frozen spectral LUT teacher, RGB norm > 1e-8. No direct solar disc, finite-segment fog, or GPU performance test.",fontsize=9)
fig.savefig(root/"comparison.png",dpi=160)
plt.close(fig)
