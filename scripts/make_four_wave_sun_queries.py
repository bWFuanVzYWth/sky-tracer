"""Bounded solar-model validation: existing views, hard cases, continuous sweeps."""
import json
import math
from pathlib import Path

out = Path('out/four_wave_sun_v1')
out.mkdir(exist_ok=True)
images = json.loads(Path('out/wavelength_search_dataset_v1/queries.json').read_text())['images']
def scene(name, h, e, pitch, yaw, fov, width=192, height=96):
    return dict(name=name, width=width, height=height, altitude_km=h,
                sun_elevation_deg=e, pitch=pitch, yaw=yaw, horizontal_fov=fov)
for e in [85, 47, 5, 0, -.25, -.45, -.65, -.8]:
    images.append(scene(f'sun_close_{e:g}', .2, e, e, 0, 4, 256, 128))
for h in [.002, .2, 12, 30, 108, 400]:
    horizon = -math.degrees(math.acos(6360/(6360+h)))
    for offset in [-.3, 0, .3]:
        images.append(scene(f'shadow_{h:g}_{offset:g}', h, horizon+offset, horizon+1, 90, 16))
for e in [-8, -6, -4, -2, -1, -.5, 0, .5]:
    images.append(scene(f'twilight_{e:g}', .2, e, 5, 0, 120))
(out/'queries.json').write_text(json.dumps(dict(images=images), indent=2))
sweeps = []
horizon = -math.degrees(math.acos(6360/6372))
for i in range(21):
    sweeps.append(scene(f'sweep_shadow_{i:02}', 12, horizon+.2+.01*i, horizon+1, 90, 16, 96, 48))
for i in range(33):
    e = -.8+.05*i
    sweeps.append(scene(f'sweep_sunset_{i:02}', .2, e, e, 0, 4))
(out/'sweep_queries.json').write_text(json.dumps(dict(images=sweeps), indent=2))
names = ['noon_aureole', 'blue_sunward', 'aircraft_twilight', 'sun_close_0',
         'sun_close_-0.65', 'shadow_12_0.3', 'shadow_30_0.3', 'shadow_108_0.3']
(out/'convergence_queries.json').write_text(json.dumps(dict(images=[s for s in images if s['name'] in names]), indent=2))
print(f'{len(images)} static views, {len(sweeps)} sweep frames, {len(names)} convergence views')
