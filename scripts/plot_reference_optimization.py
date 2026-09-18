"""CPU-only plots of sparse node-oracle experiments and exact work counts."""
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
folder = root / "out/lut_v6_design"
with (folder / "interpolation.json").open() as f:
    interp = json.load(f)
with (folder / "allocation.json").open() as f:
    allocation = json.load(f)
with (folder / "work.json").open() as f:
    work = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), layout="constrained")
rows = [r for r in interp["rows"] if r["h"] == 120]
axes[0].plot([r["sun"] for r in rows], [r["direct"] for r in rows], "o-", label="Direct CPU integral")
axes[0].plot([r["sun"] for r in rows], [r["old_oracle"] for r in rows], "s--", label="v5 lookup, recomputed nodes")
axes[0].plot([r["sun"] for r in rows], [r["variants"][7]["oracle"] for r in rows], ".-", label="Candidate, recomputed nodes")
axes[0].set(title="120 km: solar-corner interpolation", xlabel="Solar elevation (degrees)", ylabel="550 nm first-order radiance")
axes[0].ticklabel_format(useOffset=False)
axes[0].legend(fontsize=8)

selected = [a for a in allocation if a["name"] in ("base", "height_x2", "view_x2", "solar_x2", "phase_x2", "cross_cubic")]
x = np.arange(len(selected))
for case, color in enumerate(["#235789", "#dd7042", "#469b74"]):
    values = [100 * abs(a["rows"][case]["relative_error"]) for a in selected]
    axes[1].bar(x + (case - 1) * .26, values, .25, color=color, label=f"Sun {selected[0]['rows'][case]['sun']:.2f} deg")
axes[1].set_xticks(x, ["Base", "Height x2", "View x2", "Sun x2", "Phase x2", "Cross warp"], rotation=28, ha="right")
axes[1].set(title="30 km: allocation study", ylabel="Absolute relative error (%)")
axes[1].legend(fontsize=8)

counts = [work[k] / 1e6 for k in ["dense_states", "phase_unique_states", "cpu_unique_states"]]
bars = axes[2].bar(["Dense", "Phase reuse", "+ View reuse*"], counts, color=["#7d8794", "#235789", "#469b74"])
axes[2].bar_label(bars, fmt="%.2f M", padding=3)
axes[2].set(title="Exact transport work per band / order", ylabel="Evaluated states (millions)", ylim=(0, 144))
axes[2].text(.5, .9, "* View equality rechecked on GPU\nTwo-order ABBA: 23.6% less time (v5 grid)", ha="center", transform=axes[2].transAxes, fontsize=8)
for ax in axes:
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
fig.suptitle("CPU design study — first order at 550 nm, not acceptance of a new full bake", fontsize=12)
fig.savefig(folder / "optimization.png", dpi=160)
plt.close(fig)
