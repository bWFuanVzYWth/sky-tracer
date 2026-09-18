"""Plot measured node placement and CPU axis-restoration diagnostics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("out/lut_allocation_audit")
data = json.loads((root / "allocation.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
ax = axes[0, 0]
h = np.array(data["height_nodes_km"])
ax.scatter(h, np.zeros_like(h), marker="|", s=300)
ax.set_xscale("symlog", linthresh=0.001)
ax.set_xlim(-0.00015, 150)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xlabel("Altitude (km), logarithmic beyond 1 m")
ax.set_title("Current 24 altitude nodes: 0 to 120 km")
ax.annotate("27.23 to 35 km: 7.77 km gap", (30, 0), (1, .3),
            arrowprops={"arrowstyle": "->"})
ax.grid(axis="x", alpha=.2)

ax = axes[0, 1]
near = min(data["charts"], key=lambda c: abs(c["altitude_km"]-.2))
for i, chart in enumerate([near, data["charts"][-1]]):
    solar = np.array(chart["solar_deg"])
    ax.scatter(solar, np.full_like(solar, i), marker="|", s=100,
               label=f'Observer {chart["altitude_km"]:g} km')
ax.set_xlim(-91, 91)
ax.set_ylim(-.6, 1.6)
ax.set_yticks([0, 1], ["0.2 km", "120 km"])
ax.set_xlabel("Solar elevation (degrees)")
ax.set_title("97 Sun nodes; position depends on altitude")
ax.grid(axis="x", alpha=.2)

ax = axes[1, 0]
for i, e in enumerate([0, 47, 85]):
    p = next(p for p in near["phase"] if not p["ground"] and p["sun_deg"] == e)
    theta = np.unique(p["theta_deg"])
    ax.scatter(theta, np.full_like(theta, i), marker="|", s=150)
ax.set_xlim(-.05, 5)
ax.set_ylim(-.6, 2.6)
ax.set_yticks([0, 1, 2], ["Sun 0 deg", "Sun 47 deg", "Sun 85 deg"])
ax.set_xlabel("Angle from Sun (degrees)")
ax.set_title("Forward phase nodes: same 65 slots, different density")
ax.grid(axis="x", alpha=.2)

ax = axes[1, 1]
cases = ["noon_sun_0.27_to_2_deg", "stratosphere_shadow", "atmosphere_edge", "orbit"]
masks = [0, 1, 2, 4]
v = np.array([[100*data["reports"][m]["cases"][c]["vs_spectral"]["p99"] for m in masks] for c in cases])
ax.imshow(np.log10(np.maximum(v, .01)), cmap="YlOrRd", vmin=-1.2, vmax=1.4, aspect="auto")
for i in range(len(cases)):
    for j in range(len(masks)):
        ax.text(j, i, f"{v[i,j]:.2f}%", ha="center", va="center", color="black",
                bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 2})
ax.set_xticks(range(4), ["Base grid", "Height 80", "Sun 193", "Phase 257"])
ax.set_yticks(range(4), ["Noon halo", "30 km shadow", "120 km edge", "400 km orbit"])
ax.set_title("P99 after restoring ONE axis (unquantized)")
fig.suptitle("16 MB allocation diagnosis: current layout and where its error comes from", fontsize=15)
fig.supxlabel("CPU only. Restorations exceed 16 MB and are diagnostic. Errors relative to the frozen spectral LUT.", fontsize=10)
fig.savefig(root / "allocation.png", dpi=160)
plt.close(fig)
