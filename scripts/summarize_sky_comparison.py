"""Measure raw demo exports in linear Rec.2020; no PNG or tone-map metrics."""
import argparse
import json
from pathlib import Path
import numpy as np


def reference_disk_footprint(info, ray, solar_elevation, solar_azimuth, horizon):
    """Reject a query if any bilinear source texel can integrate the solar disk.

    Source pixels are finite solid-angle footprints; an output-pixel margin
    alone cannot remove the bright disk interpolated from a lower-res PT LUT.
    Bounds below use the actual PT pixel domain and its inverse display map.
    """
    width, height = info["reference_dimensions"]
    theta = np.arccos(np.clip(ray[..., 1], -1, 1))
    if info["reference_kind"] == "spectral_sky_view_lut_v0":
        theta_h = np.arccos(horizon)
        horizontal = ray[..., [0, 2]]
        norm = np.linalg.norm(horizontal, axis=2)
        sun_horizontal = np.array([np.sin(solar_azimuth), np.cos(solar_azimuth)])
        cos_phi = np.divide(horizontal @ sun_horizontal, norm,
                            out=np.ones_like(norm), where=norm > 1e-5)
        u = np.sqrt(np.clip((1-cos_phi)*.5, 0, 1))
        v = np.where(theta < theta_h,
                     .75*(1-np.sqrt(np.maximum(1-theta/theta_h, 0))),
                     .75+.25*np.sqrt(np.maximum((theta-theta_h)/(np.pi-theta_h), 0)))
        u_low = np.clip((np.floor(u*(width-1))-.5)/(width-1), 0, 1)
        closest_cos_phi = 1-2*u_low*u_low
        v_low = np.clip((np.floor(v*(height-1))-.5)/(height-1), 0, 1)
        v_high = np.clip((np.floor(v*(height-1))+1.5)/(height-1), 0, 1)
        def angle(v):
            return np.where(v < .75, theta_h*(1-(1-v/.75)**2),
                            theta_h+(np.pi-theta_h)*((v-.75)/.25)**2)
        theta_low, theta_high = angle(v_low), angle(v_high)
    else:
        u = (np.arctan2(ray[..., 0], ray[..., 2])/(2*np.pi)+.5) % 1
        v = theta/np.pi
        center = (np.floor(u*width-.5)+1)/width*2*np.pi-np.pi
        delta = (center-solar_azimuth+np.pi) % (2*np.pi)-np.pi
        closest_cos_phi = np.cos(np.maximum(np.abs(delta)-2*np.pi/width, 0))
        theta_low = np.clip(np.floor(v*height-.5)/height, 0, 1)*np.pi
        theta_high = np.clip((np.floor(v*height-.5)+2)/height, 0, 1)*np.pi
    sine = np.cos(solar_elevation)*closest_cos_phi
    cosine = np.sin(solar_elevation)
    candidate = np.arctan2(sine, cosine)
    angular_dot = lambda t: sine*np.sin(t)+cosine*np.cos(t)
    closest = np.maximum(angular_dot(theta_low), angular_dot(theta_high))
    closest = np.maximum(closest, np.where((candidate >= theta_low) & (candidate <= theta_high),
                                          angular_dot(candidate), -1))
    return closest >= np.cos(.00465047)

p=argparse.ArgumentParser()
p.add_argument("directory",type=Path)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args()
results=[]
for path in sorted(a.directory.glob("*.f32")):
    info=json.loads(path.with_suffix(".json").read_text())
    w,h=info["width"],info["height"]
    images=np.fromfile(path,dtype="<f4").reshape(4,h,w,4)[...,:3]
    assert np.all(np.isfinite(images)), f"nonfinite image: {path}"
    solver,pt=images[:2]
    assert np.allclose(images[2], abs(solver-pt), rtol=2e-5, atol=1e-7), f"absolute panel mismatch: {path}"
    assert np.allclose(images[3], solver-pt, rtol=2e-5, atol=1e-7), f"signed panel mismatch: {path}"
    yaw,pitch,fov,_=np.deg2rad(info["view_yaw_pitch_fov_exposure"])
    forward=np.array([np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)])
    right=np.array([np.cos(yaw),0,-np.sin(yaw)])
    up=np.cross(forward,right)
    x,y=np.meshgrid((np.arange(w)+.5)/w*2-1,1-(np.arange(h)+.5)/h*2)
    ray=forward+np.tan(fov/2)*(x[...,None]*(w/h)*right+y[...,None]*up)
    ray/=np.linalg.norm(ray,axis=2)[...,None]
    elevation,azimuth=np.deg2rad([info["sun_elevation_deg"],info["sun_azimuth_deg"]])
    sun=np.array([np.sin(azimuth)*np.cos(elevation),np.sin(elevation),np.cos(azimuth)*np.cos(elevation)])
    angle=np.arccos(np.clip(ray@sun,-1,1))
    altitude=info["altitude_km"]
    horizon=-np.sqrt(altitude*(2*6360+altitude))/(6360+altitude)
    relative_elevation=np.arcsin(np.clip(ray[...,1],-1,1))-np.arcsin(horizon)
    valid=angle>0.00465047+2*np.tan(fov/2)/h*np.sqrt(2)
    reference_contamination=reference_disk_footprint(info,ray,elevation,azimuth,horizon)
    valid &= ~reference_contamination
    sky=relative_elevation>0
    masks={"all":valid,"sky":valid&sky,
           "near_sun":valid&sky&(angle<np.deg2rad(3)),
           "aureole":valid&sky&(angle>=np.deg2rad(3))&(angle<np.deg2rad(10)),
           "horizon":valid&(abs(relative_elevation)<np.deg2rad(3))}
    weights=np.array([.2627,.6780,.0593])
    regions={}
    for name,mask in masks.items():
        if not mask.any():continue
        s,r=solver[mask],pt[mask]
        sy,ry=s@weights,r@weights
        norm_rgb, norm_y = float(np.linalg.norm(r)), float(np.linalg.norm(ry))
        # A zero-sample PT image is not evidence for zero physical radiance.
        # Leave relative errors undefined instead of dividing by an invented floor.
        regions[name]=dict(pixels=int(mask.sum()),
            pt_nonzero_pixels=int(np.count_nonzero(np.any(r != 0,axis=1))),
            absolute_rgb_rmse=float(np.sqrt(np.mean((s-r)**2))),
            relative_rgb_l2=float(np.linalg.norm(s-r)/norm_rgb) if norm_rgb>0 else None,
            relative_luminance_l2=float(np.linalg.norm(sy-ry)/norm_y) if norm_y>0 else None,
            mean_luminance_solver=float(sy.mean()),mean_luminance_pt=float(ry.mean()),
            mean_luminance_bias=float(sy.mean()/ry.mean()-1) if ry.mean()>0 else None)
    results.append(dict(file=path.name,metadata=info,regions=regions,
                        reference_solar_footprint_pixels=int(reference_contamination.sum())))
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(dict(note="linear Rec.2020; PT texture interpolation and Monte Carlo noise remain; direct disk, output-pixel margin, and all bilinear PT texel footprints intersecting the solar disk excluded",results=results),indent=2))
print(f"measured {len(results)} comparisons")
