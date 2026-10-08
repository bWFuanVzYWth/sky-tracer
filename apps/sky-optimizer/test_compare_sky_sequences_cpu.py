"""Small CPU fixture: physical motion cancels, a one-frame pulse is detected."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

directory = Path(__file__).resolve().parent
tool = directory / 'compare_sky_sequences_cpu.py'
fixture = directory.parents[1] / 'target/test-tmp/realtime-sequence-v7'
fixture.mkdir(parents=True, exist_ok=True)
reference, candidate = fixture / 'reference', fixture / 'candidate'
reference.mkdir(exist_ok=True)
candidate.mkdir(exist_ok=True)
images = []
for frame in range(5):
    name = f'fixture_{frame:03}'
    images.append({'name': name, 'width': 8, 'height': 4, 'sun_elevation_deg': frame})
    baseline = np.ones((4, 8, 4), dtype='<f4')
    baseline[..., :3] = 1 + frame * .125
    baseline.tofile(reference / f'{name}_sky.f32')
    value = baseline.copy()
    if frame == 2:
        value[..., :3] += .125
    value.tofile(candidate / f'{name}_sky.f32')
queries = fixture / 'queries.json'
queries.write_text(json.dumps({'images': images}))
for stem, path in [('identical', reference), ('pulse', candidate)]:
    # New process refuses overwriting a report; retained fixtures are harmless.
    import tempfile
    report = Path(tempfile.mkdtemp(dir=fixture, prefix=stem)) / 'report'
    subprocess.run([sys.executable, str(tool), str(reference),
                    str(path), '--queries', str(queries), '--out', str(report)], check=True,
                   stdout=subprocess.DEVNULL)
    result = json.loads((report / 'summary.json').read_text())
    summary = result['summary']
    if stem == 'identical':
        assert all(row.get('worst_frame_p99', 0) == 0 for row in summary.values())
    else:
        assert summary['static_all_sky']['worst_frame_p99'] > .09
        assert summary['first_temporal_residual']['worst_frame_p99'] > .09
        assert summary['second_temporal_residual']['worst_frame_p99'] > .15
        assert summary['adjacent_row_residual_jump']['worst_frame_p99'] == 0

# Profile schema and actual SDR payloads, without creating a device.
profile_reference = fixture / 'profile_reference'
profile_candidate = fixture / 'profile_candidate'
profile_reference.mkdir(exist_ok=True)
profile_candidate.mkdir(exist_ok=True)
captures = []
for frame in range(5):
    name = f'fixture_{frame:03}'
    captures.append({'trajectory': 'fixture', 'frame': frame, 'linear': f'{name}.rgba32f',
                     'display': f'{name}.srgba8', 'size': [8, 4], 'exposure': .5,
                     'view': {'sun_elevation_deg': frame, 'sun_azimuth_deg': 0}})
    for old, new in [(reference, profile_reference), (candidate, profile_candidate)]:
        (new / f'{name}.rgba32f').write_bytes((old / f'{name}_sky.f32').read_bytes())
        sdr = np.full((4, 8, 4), 64 + frame * 4, dtype='u1')
        sdr[..., 3] = 255
        if old == candidate and frame == 2:
            sdr[..., :3] += 20
        sdr.tofile(new / f'{name}.srgba8')
for path in [profile_reference, profile_candidate]:
    (path / 'profile.json').write_text(json.dumps({'captures': captures}))
    (path / 'inputs.json').write_text(json.dumps({'wavelengths': 'synthetic',
        'mapping_calibration': 'synthetic', 'shader_checksum': 'synthetic', 'size': [8, 4],
        'include_visible_sun_disk': False, 'trajectories': ['synthetic']}))
profile_report = Path(tempfile.mkdtemp(dir=fixture, prefix='profile')) / 'report'
subprocess.run([sys.executable, str(tool), str(profile_reference), str(profile_candidate),
                '--out', str(profile_report)], check=True, stdout=subprocess.DEVNULL)
result = json.loads((profile_report / 'summary.json').read_text())
assert result['summary']['display_max_channel_difference_lsb']['worst_frame_p99'] == 20
assert result['summary']['first_temporal_residual']['worst_frame_p99'] > .09

bad = json.loads((profile_candidate / 'profile.json').read_text())
bad['captures'][0]['view']['sun_elevation_deg'] += .01
(profile_candidate / 'profile.json').write_text(json.dumps(bad))
bad_report = Path(tempfile.mkdtemp(dir=fixture, prefix='bad-profile')) / 'report'
run = subprocess.run([sys.executable, str(tool), str(profile_reference), str(profile_candidate),
                      '--out', str(bad_report)], capture_output=True)
assert run.returncode != 0 and b'trajectories differ' in run.stderr
assert not bad_report.exists()

# An intentional SkyView CDF replacement must not silently allow a source or
# ground mapping change. Use the same captures so only metadata varies.
(profile_candidate / 'profile.json').write_text((profile_reference / 'profile.json').read_text())
import copy
mapping = {'sky': {'low': {'weights': [1]}, 'upper': {'weights': [1]},
                   'space': {'weights': [1]}, 'ground': {'weights': [1]}},
           'source': {'heights': [0, 1]}}
inputs = json.loads((profile_reference / 'inputs.json').read_text())
inputs['mapping_calibration'] = mapping
(profile_reference / 'inputs.json').write_text(json.dumps(inputs))
for modified in ['sky', 'ground', 'source']:
    current = copy.deepcopy(inputs)
    current['mapping_calibration']['sky']['low']['weights'] = [2]
    current['mapping_calibration']['sky_fit_provenance'] = {'fixture': True}
    if modified == 'ground':
        current['mapping_calibration']['sky']['ground']['weights'] = [2]
    if modified == 'source':
        current['mapping_calibration']['source']['heights'] = [0, 2]
    (profile_candidate / 'inputs.json').write_text(json.dumps(current))
    report = Path(tempfile.mkdtemp(dir=fixture, prefix=f'mapping-{modified}')) / 'report'
    result = subprocess.run([sys.executable, str(tool), str(profile_reference), str(profile_candidate),
                             '--allow-sky-mapping-change', '--out', str(report)], capture_output=True)
    if modified == 'sky':
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and b'mapping_calibration' in result.stderr
        assert not report.exists()
print('CPU fixtures passed: physical motion cancels, pulse/SDR detected, profile schema accepted, mismatched input/source/ground rejected.')
