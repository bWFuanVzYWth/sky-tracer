"""Prepare and measure CPU reference queries against existing PT exports.

prepare -> sky-atmosphere-lut synthesize -> analyze. Never creates a GPU device.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("out/lut_v5_cpu_quality")
Y = np.array([.2627, .6780, .0593])


def rays(info):
    w, h = info["width"], info["height"]
    yaw, pitch, fov, _ = np.deg2rad(info["view_yaw_pitch_fov_exposure"])
    f = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
    r = np.array([np.cos(yaw), 0, -np.sin(yaw)])
    up = np.cross(f, r)
    x, y = np.meshgrid((np.arange(w)+.5)/w*2-1, 1-(np.arange(h)+.5)/h*2)
    v = f + np.tan(fov/2)*(x[..., None]*(w/h)*r + y[..., None]*up)
    return v/np.linalg.norm(v, axis=2)[..., None]


def horizon(h):
    return -np.sqrt(h*(2*6360+h))/(6360+h)


def masks(info, ray):
    # Conservative footprint of all four contributing PT skyview texels.
    assert info["reference_kind"] == "spectral_sky_view_lut_v0"
    w, h = info["reference_dimensions"]
    e, a = np.deg2rad([info["sun_elevation_deg"], info["sun_azimuth_deg"]])
    th = np.arccos(horizon(info["altitude_km"]))
    theta = np.arccos(np.clip(ray[..., 1], -1, 1))
    xy = ray[..., [0, 2]]
    length = np.linalg.norm(xy, axis=2)
    cosphi = np.divide(xy @ np.array([np.sin(a), np.cos(a)]), length,
                       out=np.ones_like(length), where=length > 1e-5)
    u = np.sqrt(np.clip((1-cosphi)/2, 0, 1))
    v = np.where(theta < th, .75*(1-np.sqrt(np.maximum(1-theta/th, 0))),
                 .75+.25*np.sqrt(np.maximum((theta-th)/(np.pi-th), 0)))
    ul = np.clip((np.floor(u*(w-1))-.5)/(w-1), 0, 1)
    vl = np.clip((np.floor(v*(h-1))-.5)/(h-1), 0, 1)
    vh = np.clip((np.floor(v*(h-1))+1.5)/(h-1), 0, 1)
    def angle(t):
        return np.where(t < .75, th*(1-(1-t/.75)**2),
                        th+(np.pi-th)*((t-.75)/.25)**2)
    lo, hi = angle(vl), angle(vh)
    sine, cosine = np.cos(e)*(1-2*ul*ul), np.sin(e)
    candidate = np.arctan2(sine, cosine)
    dot = lambda t: sine*np.sin(t)+cosine*np.cos(t)
    closest = np.maximum(dot(lo), dot(hi))
    closest = np.maximum(closest, np.where((candidate >= lo) & (candidate <= hi), dot(candidate), -1))
    sun = np.array([np.sin(a)*np.cos(e), np.sin(e), np.cos(a)*np.cos(e)])
    separation = np.arccos(np.clip(ray @ sun, -1, 1))
    margin = 2*np.tan(np.deg2rad(info["view_yaw_pitch_fov_exposure"][2])/2)/info["height"]*np.sqrt(2)
    valid = (closest < np.cos(.00465047)) & (separation > .00465047+margin)
    above = th-theta
    sky = above > 0
    return {"all": valid, "sky": valid & sky,
            "near_sun": valid & sky & (separation < np.deg2rad(3)),
            "aureole": valid & sky & (separation >= np.deg2rad(3)) & (separation < np.deg2rad(10)),
            "horizon": valid & (abs(above) < np.deg2rad(3))}


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    queries, groups = [], []
    paths = sorted(Path("out/noon_validation_v4_rgb").glob("lut_*.json"))
    paths += sorted(Path("out/lut_validation_v3").glob("lut_orbit*.json"))
    for path in paths:
        info = json.loads(path.read_text())
        if "width" not in info:
            continue
        assert info["transport_version"] == "wgpu-layered-surface-v2"
        ray = rays(info).reshape(-1, 3)
        start = len(queries)
        for r in ray:
            queries.append(dict(altitude_km=info["altitude_km"],
                sun_elevation_deg=info["sun_elevation_deg"],
                view_elevation_deg=float(np.rad2deg(np.arcsin(np.clip(r[1], -1, 1)))),
                relative_azimuth_deg=float(np.rad2deg(np.arctan2(r[0], r[2]))-info["sun_azimuth_deg"])))
        groups.append(dict(kind="image", path=str(path), start=start, count=len(ray)))
    # Smooth profiles at physical coordinates, including high altitude/space.
    for h in [.001, .2, 2., 10., 30., 60., 120., 400., 1000.]:
        for sun in [-12., -6., 0., 20., 60., 85., 89., 90.]:
            for az in [0., 180.]:
                start = len(queries)
                for above in np.linspace(.002, 10.002, 1001):
                    queries.append(dict(altitude_km=h, sun_elevation_deg=sun,
                        view_elevation_deg=float(np.rad2deg(np.arcsin(horizon(h)))+above),
                        relative_azimuth_deg=az))
                groups.append(dict(kind="profile", altitude=h, sun=sun, azimuth=az,
                                   start=start, count=1001))
    # Noon cap and solar axis smoothness; fixed view, moving Sun.
    for h in [.2, 30., 120.]:
        for above in [.1, 5., 30.]:
            for az in [0., 180.]:
                start = len(queries)
                for sun in np.linspace(-18, 90, 5401):
                    queries.append(dict(altitude_km=h, sun_elevation_deg=float(sun),
                        view_elevation_deg=float(np.rad2deg(np.arcsin(horizon(h)))+above),
                        relative_azimuth_deg=az))
                groups.append(dict(kind="solar", altitude=h, above=above, azimuth=az,
                                   start=start, count=5401))
    (ROOT / "queries.json").write_text(json.dumps(queries), encoding="utf-8")
    (ROOT / "groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    print(f"Prepared {len(queries)} queries in {len(groups)} groups")


def stats(s, r, mask):
    if not mask.any():
        return None
    a, b = s[mask], r[mask]
    norm = np.linalg.norm(b)
    return dict(pixels=int(mask.sum()), rgb_l2_percent=float(100*np.linalg.norm(a-b)/norm) if norm else None,
                mean_y_bias_percent=float(100*((a@Y).sum()/(b@Y).sum()-1)) if (b@Y).sum() else None,
                absolute_rgb_rmse=float(np.sqrt(np.mean((a-b)**2))))


def coarse(s, mask, n=16):
    h, w = mask.shape
    m = mask.reshape(h//n, n, w//n, n)
    count = m.sum(axis=(1, 3))
    a = (s*mask[..., None]).reshape(h//n, n, w//n, n, 3).sum(axis=(1, 3))
    return a/np.maximum(count[..., None], 1), count > n*n*.75


def preview(rgb, exposure):
    # Common display transform for every solver and PT in a scene. Metrics stay linear.
    to_srgb = np.array([[1.660491, -.587641, -.07285], [-.12455, 1.13290, -.00835], [-.01815, -.10058, 1.11873]])
    rgb = np.maximum(rgb @ to_srgb.T * 2**exposure, 0)
    # Scale by peak channel so the scientific preview retains chromaticity.
    # This is not the demo's Oklab display transform.
    rgb /= 1+np.max(rgb, axis=-1, keepdims=True)
    return np.clip(np.where(rgb <= .0031308, rgb*12.92, 1.055*rgb**(1/2.4)-.055), 0, 1)


def analyze(dataset):
    groups = json.loads((ROOT / "groups.json").read_text())
    rgb = np.fromfile(dataset / "samples.f32", dtype="<f4").reshape(-1, 3)
    assert np.isfinite(rgb).all() and len(rgb) == sum(g["count"] for g in groups)
    images, profiles, solar = [], [], []
    output = ROOT / "comparisons"
    output.mkdir(exist_ok=True)
    for g in groups:
        values = rgb[g["start"]:g["start"]+g["count"]]
        if g["kind"] == "image":
            path = Path(g["path"])
            info = json.loads(path.read_text())
            h, w = info["height"], info["width"]
            old = np.fromfile(path.with_suffix(".f32"), dtype="<f4").reshape(4, h, w, 4)[..., :3]
            new, pt = values.reshape(h, w, 3), old[1]
            region_masks = masks(info, rays(info))
            ue_path = path.with_name(path.name.replace("lut_", "ue_"))
            ue = np.fromfile(ue_path.with_suffix(".f32"), dtype="<f4").reshape(4, h, w, 4)[0, ..., :3]
            results = {}
            for region, mask in region_masks.items():
                results[region] = {k: stats(v, pt, mask) for k, v in [("previous", old[0]), ("v5_spectral", new), ("ue8", ue)]}
                ref, valid = coarse(pt, mask)
                results[region]["block16"] = {k: stats(coarse(v, mask)[0], ref, valid)
                    for k, v in [("previous", old[0]), ("v5_spectral", new), ("ue8", ue)]}
            images.append(dict(scene=path.stem, baseline=info["lut"], regions=results))
            fig, axes = plt.subplots(1, 4, figsize=(16, 3.7), layout="constrained")
            for ax, v, label in zip(axes, [old[0], new, pt, ue], ["Previous LUT", "v5 CPU spectral", "Existing PT", "UE 8 wavelength"]):
                ax.imshow(preview(v, info["view_yaw_pitch_fov_exposure"][3])); ax.set_title(label); ax.axis("off")
            fig.suptitle(path.stem+" | common chromaticity-preserving preview")
            fig.savefig(output / (path.stem+".png"), dpi=140); plt.close(fig)
        else:
            luminance = values @ Y
            spacing = .01 if g["kind"] == "profile" else .02
            valid = luminance > max(luminance.max()*1e-4, 1e-15)
            slope = 100*np.diff(np.log(np.maximum(luminance, 1e-30)))/spacing
            mask = valid[2:] & valid[1:-1] & valid[:-2]
            jump = abs(np.diff(slope))
            item = {**g, "max_y": float(luminance.max()), "min_rgb": float(values.min()),
                    "max_slope_jump": float(jump[mask].max()) if mask.any() else None,
                    "p99_slope_jump": float(np.percentile(jump[mask], 99)) if mask.any() else None}
            (profiles if g["kind"] == "profile" else solar).append(item)
    result = dict(dataset=str(dataset), samples=len(rgb), finite=True,
                  note="CPU spectral lookup vs existing GPU PT texture snapshots; no new GPU work. Direct Sun disk footprints excluded. Raw L2 contains PT noise; block16 averages also suppress real small-scale error and are not an error bound.",
                  images=images, profiles=profiles, solar_scans=solar)
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for r in images:
        print(r["scene"], json.dumps(r["regions"]["sky"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "analyze"])
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else analyze(args.dataset)
