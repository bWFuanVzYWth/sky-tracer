"""CPU diagnostic profiles, before/after zenith-cap mapping; no display transforms."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("out")
old = np.genfromtxt(root / "noon_horizon_before.csv", delimiter=",", names=True)
new = np.genfromtxt(root / "noon_horizon_cap97_full.csv", delimiter=",", names=True)
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
rows = []
for ax, solar in zip(axes.flat, [60, 70, 80, 85, 88, 89, 89.5, 90]):
    before = old[(old["sun_deg"] == solar) & (old["azimuth_deg"] == 0)]
    after = new[(new["sun_deg"] == solar) & (new["azimuth_deg"] == 0)]
    for data, name, color in [(before, "v3 (65 solar nodes)", "#c1544e"),
                              (after, "v4 (97 solar nodes)", "#21718c")]:
        error = (data["first"] / data["direct"] - 1) * 100
        ax.plot(data["above_horizon_deg"], error, label=name, color=color)
    ax.axhline(0, color="0.5", lw=.6)
    ax.set_title(f"Sun elevation {solar:g} degrees")
    ax.set_xlim(0, 3)
    ax.grid(alpha=.2)
    m = after["above_horizon_deg"] <= 1
    a = before["above_horizon_deg"] <= 1
    rows.append(dict(sun_deg=solar,
        old_max_relative=float(np.max(abs(before["first"][a]/before["direct"][a]-1))),
        new_max_relative=float(np.max(abs(after["first"][m]/after["direct"][m]-1))),
        full_change_relative_l2=float(np.linalg.norm(after["full"][m]-before["full"][a])/np.linalg.norm(before["full"][a]))))
axes[0,0].legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("Degrees above geometric horizon")
for ax in axes[:,0]:
    ax.set_ylabel("First-scattering error (%)")
fig.suptitle("Solar-facing horizon, 550 nm, observer 200 m\nReference: independent 4096-step ray integration, 64 solar-disk samples")
fig.tight_layout()
fig.savefig(root / "noon_horizon_profiles.png", dpi=160)
(root / "noon_horizon_metrics.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))

# Independent PT full-transport profile. Bilinear resampling matches the demo;
# finite PT texels near the geometric horizon can straddle ground and sky.
import OpenEXR
pt = OpenEXR.File(str(root / "pt_noon_085/bands/sky_550nm.exr")).channels()["RGB"].pixels[...,0]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
pt_metrics = []
for ax, azimuth in zip(axes, [0, 30, 90, 180]):
    a = old[(old["sun_deg"]==85)&(old["azimuth_deg"]==azimuth)]
    b = new[(new["sun_deg"]==85)&(new["azimuth_deg"]==azimuth)]
    horizon = -np.sqrt(.2*(2*6360+.2))/(6360+.2)
    theta_h = np.arccos(horizon)
    theta = np.pi/2-np.deg2rad(b["elevation_deg"])
    u = np.sqrt((1-np.cos(np.deg2rad(azimuth)))/2)
    v = .75*(1-np.sqrt(np.maximum(1-theta/theta_h,0)))
    x,y=u*255,v*255
    xi,yi=int(np.floor(x)),np.floor(y).astype(int)
    xf,yf=x-xi,y-yi
    ref = (1-yf)*((1-xf)*pt[yi,xi]+xf*pt[yi,min(xi+1,255)])+yf*((1-xf)*pt[yi+1,xi]+xf*pt[yi+1,min(xi+1,255)])
    ax.plot(b["above_horizon_deg"],ref,label="PT, 8192 spp",color="#20242b")
    ax.plot(a["above_horizon_deg"],a["full"],label="v3",color="#c1544e")
    ax.plot(b["above_horizon_deg"],b["full"],label="v4",color="#21718c")
    ax.set_title(f"Relative azimuth {azimuth} degrees")
    ax.set_xlabel("Degrees above geometric horizon")
    ax.set_xlim(0,3)
    ax.grid(alpha=.2)
    mask = b["above_horizon_deg"]<=1
    pt_metrics.append(dict(azimuth=azimuth,
        old_full_relative_l2=float(np.linalg.norm(a["full"][mask]-ref[mask])/np.linalg.norm(ref[mask])),
        new_full_relative_l2=float(np.linalg.norm(b["full"][mask]-ref[mask])/np.linalg.norm(ref[mask])),
        old_full_relative_l2_3deg=float(np.linalg.norm((a["full"]-ref)[b["above_horizon_deg"]<=3])/np.linalg.norm(ref[b["above_horizon_deg"]<=3])),
        new_full_relative_l2_3deg=float(np.linalg.norm((b["full"]-ref)[b["above_horizon_deg"]<=3])/np.linalg.norm(ref[b["above_horizon_deg"]<=3]))))
axes[0].set_ylabel("550 nm band radiance (W / m2 / sr)")
axes[0].legend()
fig.suptitle("Full transport at sun elevation 85 degrees, observer 200 m\nPT includes Monte Carlo noise and 256x256 reference resampling")
fig.tight_layout()
fig.savefig(root/"noon_horizon_full_profiles.png",dpi=160)
(root/"noon_horizon_full_metrics.json").write_text(json.dumps(pt_metrics,indent=2))
print(json.dumps(pt_metrics,indent=2))
