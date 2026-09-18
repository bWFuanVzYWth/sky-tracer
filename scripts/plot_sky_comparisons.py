"""Build scientific comparison plates from the demo's own display previews."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
source = root / "out/lut_validation_v3"
destination = root / "out"
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)

def plate(filename, cases):
    row_height = 330
    canvas = Image.new("RGB", (1296, 56 + row_height * len(cases)), "#151b23")
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(["LUT v3 / 41 bands", "PT / 41 bands", "Current UE / 8 wavelengths"]):
        draw.text((12 + i*432, 16), name, font=font, fill="white")
    for row, (name, label) in enumerate(cases):
        meta = json.loads((source / f"lut_{name}.json").read_text())
        lut = Image.open(source / f"lut_{name}.png")
        ue = Image.open(source / f"ue_{name}.png")
        w, h = meta["width"], meta["height"]
        y = 56 + row * row_height
        draw.text((12, y), f"{label} | EV {meta['view_yaw_pitch_fov_exposure'][3]:g} | PT {meta['pt_spp']} spp", font=small, fill="#c4cedd")
        panels = [lut.crop((0,0,w,h)), lut.crop((w,0,2*w,h)), ue.crop((0,0,w,h))]
        for i, panel in enumerate(panels):
            canvas.paste(panel.resize((416, 312), Image.Resampling.LANCZOS), (8+i*432, y+18))
    canvas.save(destination / filename)

plate("lut_v3_ground_comparison.png", [
    ("near_sun_20", "Sun 20 deg / near sun"),
    ("ring_45", "Sun 45 deg / wide sky, ring check"),
    ("sunset_0", "Sun 0 deg / sunset"),
    ("earth_shadow_m06", "Sun -6 deg / anti-solar shadow (PT remains noisy)"),
])
plate("lut_v3_orbit_comparison.png", [
    ("orbit_limb_20", "400 km / Sun 20 deg (UE height clamped inside atmosphere)"),
    ("orbit_shadow_m06", "400 km / dark limb (PT insufficient; UE outside supported range)"),
])

data = np.genfromtxt(destination / "lut_v3_orbit_first_order.csv", delimiter=",", names=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for azimuth in [0, 90, 180]:
    v = data[(data["sun_deg"] == 20) & (data["azimuth_deg"] == azimuth)]
    axes[0].semilogy(v["tangent_height_km"], v["direct"], label=f"Direct, az {azimuth}")
    axes[0].semilogy(v["tangent_height_km"], v["first"], "--", label=f"LUT, az {azimuth}")
    axes[1].plot(v["tangent_height_km"], 100*(v["first"]/v["direct"]-1), label=f"Azimuth {azimuth}")
axes[0].set(ylabel="550 nm band radiance [W m-2 sr-1]", xlabel="Ray tangent height [km]", title="400 km observer / Sun 20 deg / first order")
axes[1].set(ylabel="Relative difference [%]", xlabel="Ray tangent height [km]", title="LUT versus 4096-step direct integration")
for ax in axes:
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
fig.savefig(destination / "lut_v3_orbit_error.png", dpi=160)
plt.close(fig)
