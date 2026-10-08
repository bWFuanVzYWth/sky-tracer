"""CPU-only static/temporal residual diagnostics for audit evaluate sequences."""
import argparse
import json
import struct
import zlib
from pathlib import Path
import numpy as np


def pixels(directory, image, mode):
    path = directory / image.get('linear', f"{image['name']}_{mode}.f32")
    data = np.fromfile(path, dtype='<f4')
    expected = image['width'] * image['height'] * 4
    if data.size != expected or not np.isfinite(data).all():
        raise ValueError(f"Invalid complete image {path}: {data.size}/{expected} finite floats")
    return data.reshape(image['height'], image['width'], 4)[..., :3].astype(np.float64)


def statistics(values, mask=None):
    if mask is not None:
        values = values[mask]
    if values.size == 0:
        return {"pixels": 0}
    q = np.percentile(values, [50, 95, 99, 100])
    return dict(zip(['p50', 'p95', 'p99', 'max'], q.tolist()), pixels=int(values.size))


def png(path, rgb):
    height, width = rgb.shape[:2]
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    rows = b''.join(b'\0' + row.tobytes() for row in rgb)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--queries', type=Path, help='Evaluate queries; profile capture inputs are read automatically')
    parser.add_argument('--mode', choices=['sky', 'source'], default='sky')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-sky-mapping-change', action='store_true',
                        help='Allow intentional low/upper/space CDF changes, retaining source/ground mappings')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output exists; choose a new directory')
    if (args.reference / 'profile.json').exists():
        if args.mode != 'sky':
            raise ValueError('Profile captures contain sky output only')
        captures = json.loads((args.reference / 'profile.json').read_text())['captures']
        other = json.loads((args.candidate / 'profile.json').read_text())['captures']
        a = json.loads((args.reference / 'inputs.json').read_text())
        b = json.loads((args.candidate / 'inputs.json').read_text())
        for key in ['wavelengths', 'mapping_calibration', 'shader_checksum', 'size', 'include_visible_sun_disk', 'trajectories']:
            va, vb = a.get(key), b.get(key)
            if key == 'mapping_calibration' and args.allow_sky_mapping_change:
                import copy
                va, vb = copy.deepcopy(va), copy.deepcopy(vb)
                for v in [va, vb]:
                    for group in ['low', 'upper', 'space']:
                        v['sky'].pop(group, None)
                    v.pop('sky_fit_provenance', None)
            if va != vb:
                raise ValueError(f'Incompatible frozen profile input: {key}')
        signature = lambda c: (c['trajectory'], c['frame'], c['size'], c['view'], c['exposure'])
        if list(map(signature, captures)) != list(map(signature, other)):
            raise ValueError('Reference and candidate profile capture trajectories differ')
        images = [dict(name=f"{c['trajectory']}_{c['frame']:03}", frame=c['frame'],
                       width=c['size'][0], height=c['size'][1],
                       sun_elevation_deg=c['view']['sun_elevation_deg'],
                       linear=c['linear'], display=c['display']) for c in captures]
    else:
        if args.queries is None:
            raise ValueError('--queries is required for evaluate output directories')
        images = json.loads(args.queries.read_text())['images']
    groups = {}
    for image in images:
        groups.setdefault(image['name'].rsplit('_', 1)[0], []).append(image)
    records = []
    for trajectory, group in groups.items():
        # A fixed floor over the whole trajectory avoids artificial denominator
        # changes. Vacuum is counted in all-sky statistics and reported separately.
        scale = max(float(np.percentile(np.linalg.norm(pixels(args.reference, i, args.mode), axis=-1), 99)) for i in group)
        floor = max(scale * 1e-5, 1e-30)
        previous = None
        previous_difference = None
        previous_difference_norm = None
        previous_frame = None
        for image in group:
            b = pixels(args.reference, image, args.mode)
            c = pixels(args.candidate, image, args.mode)
            norm = np.linalg.norm(b, axis=-1)
            lit = norm > max(scale * 1e-8, 1e-30)
            residual = c - b
            relative = np.linalg.norm(residual, axis=-1) / np.maximum(norm, floor)
            row_b = b.mean(axis=1)
            row_e = residual.mean(axis=1)
            row_jump = np.linalg.norm(np.diff(row_e, axis=0), axis=-1) / np.maximum(
                np.maximum(np.linalg.norm(row_b[1:], axis=-1), np.linalg.norm(row_b[:-1], axis=-1)), floor)
            row = {'name': image['name'], 'trajectory': trajectory,
                   'sun_elevation_deg': image['sun_elevation_deg'],
                   'static_all_sky': statistics(relative), 'static_illuminated': statistics(relative, lit),
                   'adjacent_row_residual_jump': statistics(row_jump), 'trajectory_norm_floor': floor,
                   'frame': int(image['name'].rsplit('_', 1)[1])}
            if 'display' in image:
                a = np.fromfile(args.reference / image['display'], dtype='u1')
                d = np.fromfile(args.candidate / image['display'], dtype='u1')
                if a.size != image['width'] * image['height'] * 4 or a.shape != d.shape:
                    raise ValueError('Incomplete display capture')
                difference = np.abs(a.reshape(-1, 4)[:, :3].astype(np.int16) - d.reshape(-1, 4)[:, :3].astype(np.int16))
                row['display_max_channel_difference_lsb'] = statistics(difference.max(axis=1))
            if previous is not None and row['frame'] == previous_frame + 1:
                prev_e, prev_norm = previous
                difference = residual - prev_e
                denominator = np.maximum(np.maximum(norm, prev_norm), floor)
                first = np.linalg.norm(difference, axis=-1) / denominator
                row['first_temporal_residual'] = statistics(first)
                if previous_difference is not None:
                    row['second_temporal_residual'] = statistics(
                        np.linalg.norm(difference - previous_difference, axis=-1)
                        / np.maximum(denominator, previous_difference_norm))
                previous_difference = difference
                previous_difference_norm = denominator
            else:
                previous_difference = None
                previous_difference_norm = None
            previous = residual, norm
            previous_frame = row['frame']
            records.append(row)
    args.out.mkdir(parents=True)
    summary = {}
    for metric in ['static_illuminated', 'static_all_sky', 'first_temporal_residual',
                   'second_temporal_residual', 'adjacent_row_residual_jump', 'display_max_channel_difference_lsb']:
        available = [r for r in records if metric in r and 'p99' in r[metric]]
        if not available:
            summary[metric] = {'worst_frames': []}
            continue
        worst = sorted(available, key=lambda r: r[metric]['p99'], reverse=True)[:5]
        summary[metric] = {'worst_frame_p95': max(r[metric]['p95'] for r in available),
                           'worst_frame_p99': max(r[metric]['p99'] for r in available),
                           'worst_frames': [r['name'] for r in worst]}
    lookup = {i['name']: i for i in images}
    for name in summary['static_illuminated']['worst_frames']:
        image = lookup[name]
        b, c = pixels(args.reference, image, args.mode), pixels(args.candidate, image, args.mode)
        floor = next(r['trajectory_norm_floor'] for r in records if r['name'] == name)
        relative = np.linalg.norm(c - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), floor)
        # Black→red→yellow; red reaches 1%, yellow reaches 5% relative RGB norm.
        rgb = np.stack([np.clip(relative / .01, 0, 1), np.clip((relative - .01) / .04, 0, 1), np.zeros_like(relative)], axis=-1)
        png(args.out / f'{name}_relative.png', (rgb * 255 + .5).astype(np.uint8))
    report = {'reference': str(args.reference), 'candidate': str(args.candidate), 'mode': args.mode,
              'queries': str(args.queries), 'frames': len(records), 'all_images_finite': True,
              'relative_metric_units': 'fraction, multiply by 100 for percent; display differences in 8-bit LSB',
              'summary': summary, 'records': records, 'gpu_initialized': False,
              'interpretation': 'Residual differences remove physical baseline Sun motion. No automated visual acceptance threshold. Full-resolution worst-frame inspection is still required.',
              'heatmap_scale': {'black': 0, 'red': .01, 'yellow': .05}}
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
