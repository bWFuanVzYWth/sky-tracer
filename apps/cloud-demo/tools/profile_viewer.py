"""Archive finite, serial cloud GUI runs and compare measured compute cadence.

Running this tool starts a GPU window; use only in an assigned GPU slot.
`compare`, `--dry-run`, and its unit tests never initialize a GPU.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PREFIX = 'CLOUD_VIEW_METRICS '
METRICS = ['cloud_gpu_duty_percent', 'nvml_device_activity_percent_mean', 'completed_sample_paths_per_second',
           'completed_paths_per_requested_second',
           'presents', 'work_dispatches', 'completed_spp', 'whole_sample_paths_per_requested_second',
           'presents_per_second', 'dispatches_per_group', 'gpu_group_p95_ms',
           'gpu_group_max_ms', 'gpu_chunk_p95_ms', 'gpu_chunk_max_ms',
           'submit_to_map_wall_p95_ms', 'non_gpu_completion_wait_ms',
           'encode_cpu_ms', 'observed_budget_idle_ms']


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def derived(metrics):
    window = metrics.get('compute_window_ms')
    result = {key: metrics.get(key) for key in METRICS}
    result['cloud_gpu_duty_percent'] = None
    result['completed_sample_paths_per_second'] = None
    result['presents_per_second'] = None
    result['dispatches_per_group'] = None
    if positive(window):
        gpu = metrics.get('cloud_gpu_ms')
        if metrics.get('timestamps_supported') and isinstance(gpu, (int, float)) and math.isfinite(gpu) and gpu >= 0:
            result['cloud_gpu_duty_percent'] = 100 * gpu / window
        width, height = metrics.get('film_width'), metrics.get('film_height')
        spp, pending = metrics.get('completed_spp'), metrics.get('current_batch_completed_paths')
        finished = metrics.get('current_batch_finished')
        completed = metrics.get('complete_sample_paths')
        if isinstance(completed, int) and completed >= 0:
            result['completed_sample_paths_per_second'] = completed / (window / 1000)
        elif all(isinstance(v, int) and v >= 0 for v in [width, height, spp, pending]) and isinstance(finished, bool):
            paths = spp * width * height + (0 if finished else pending)
            result['completed_sample_paths_per_second'] = paths / (window / 1000)
        presents = metrics.get('presents')
        if isinstance(presents, int) and presents >= 0:
            # Presentation uses the full viewer window, including startup.
            elapsed = metrics.get('viewer_elapsed_ms')
            result['presents_per_second'] = presents / (elapsed / 1000) if positive(elapsed) else None
    groups, dispatches = metrics.get('work_groups'), metrics.get('work_dispatches')
    if positive(groups) and isinstance(dispatches, int) and dispatches >= 0:
        result['dispatches_per_group'] = dispatches / groups
    return result


def parse(log):
    records = [json.loads(line.split(PREFIX, 1)[1]) for line in log.splitlines() if PREFIX in line]
    if len(records) != 1 or records[0].get('schema_version') != 1:
        raise ValueError('Expected exactly one schema_version=1 CLOUD_VIEW_METRICS line')
    return records[0]


def legacy_metrics(log):
    matches = re.findall(r'Cloud diagnostic duration reached:\s*(\d+) completed displays\s*/\s*'
                         r'(\d+) completed bounded work chunks\s*/\s*(\d+) complete spp', log)
    if len(matches) != 1:
        raise ValueError('Missing legacy automatic-exit counters; no GPU time can be inferred')
    frames, chunks, spp = map(int, matches[0])
    # Some old builds print partial progress; c35c108 normally does not.
    partial = re.findall(r'(?:completed|converged) (?:pixels|paths)\s+(\d+)\s*/\s*(\d+)', log, re.I)
    return dict(schema_version=0, presents=frames, work_dispatches=chunks, work_groups=chunks,
                completed_spp=spp, timestamps_supported=False, cloud_gpu_ms=None,
                compute_window_ms=None, viewer_elapsed_ms=None,
                partial_progress=tuple(map(int, partial[-1])) if partial else None,
                scope='Legacy automatic-exit stdout counters. Partial progress and GPU timings unavailable unless printed.')


def nvml_summary(path):
    # Driver device activity, sampled over the whole owned process including
    # startup. These percentages are not SM occupancy or per-process counters.
    values=[]; clocks=[]; temperatures=[]; states={}
    for line in path.read_text(errors='replace').splitlines():
        columns=line.split(',')
        if len(columns)>=4:
            try:
                value=float(columns[1].strip())
                if math.isfinite(value) and 0<=value<=100: values.append(value)
            except ValueError:
                pass
        if len(columns)>=7:
            # Each optional driver counter can be unavailable independently.
            # Retain valid values without turning N/A or nonfinite values into zero.
            for column, target in [(columns[4], clocks), (columns[5], temperatures)]:
                try:
                    value=float(column.strip())
                    if math.isfinite(value) and (target is not clocks or value>=0):
                        target.append(value)
                except ValueError:
                    pass
            state=columns[6].strip()
            if re.fullmatch(r'P\d+', state):
                states[state]=states.get(state,0)+1
    return dict(samples=len(values), gpu_activity_percent_mean=statistics.mean(values) if values else None,
                gpu_activity_percent_median=statistics.median(values) if values else None,
                graphics_clock_mhz_median=statistics.median(clocks) if clocks else None,
                graphics_clock_mhz_range=[min(clocks),max(clocks)] if clocks else None,
                temperature_c_max=max(temperatures) if temperatures else None,performance_states=states,
                scope='nvidia-smi device-wide 200ms activity samples, whole owned process including startup; '
                      'not SM occupancy, not exclusive per-process utilization; raw CSV contains timestamps/power')


def summarize(rows):
    return {key: statistics.median(values) if (values := [row['derived'][key] for row in rows
             if isinstance(row['derived'].get(key), (int, float)) and math.isfinite(row['derived'][key])]) else None
            for key in METRICS}


def run(args):
    if args.out.exists():
        raise ValueError('Choose a new output directory; profiling inputs/logs are frozen')
    binary = args.binary.resolve()
    # Freeze the accepted c35c108 physical viewer defaults rather than allowing
    # a later binary's defaults to change the model between comparisons.
    scene = json.loads((args.scene or ROOT/'experiments/clouds/disney-eighth.json').read_text())
    if args.scene is None:
        scene['transport']['ground'] = None
        direction = scene['transport']['sun_direction']
        norm = math.sqrt(sum(x*x for x in direction))
        scene['transport']['sun_direction'] = [x/norm for x in direction]
        scene['render']['seed'] = 0xC10D_2026_1008
    scene['render'].update(width=args.width, height=args.height, spp=args.spp, sample_batch_size=0)
    frozen_scene = args.out.resolve()/'scene.json'
    command = [str(binary), 'view', '--width', str(args.width), '--height', str(args.height),
               '--spp', str(args.spp), '--exit-after-seconds', str(args.seconds), '--scene', str(frozen_scene)]
    if not args.legacy:
        command += ['--gpu-budget-percent', str(args.budget), '--work-group-size', str(args.dispatches),
                    '--sample-batch-size', str(args.parallel_samples)]
    plan = dict(command=command, round_count=args.rounds, strict_serial=True, maximum_process_seconds=args.seconds + 20,
                gpu_initialized=False if args.dry_run else None)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if not binary.is_file():
        raise ValueError('Build the requested cloud-demo binary before profiling')
    args.out.mkdir(parents=True)
    frozen_scene.write_text(json.dumps(scene, indent=2)+'\n')
    vdb = ROOT/'assets/DisneyCloudDataset/wdas_cloud/wdas_cloud_eighth.vdb'
    inputs = dict(width=args.width, height=args.height, spp=args.spp, seconds=args.seconds,
                  scene=scene, vdb_sha256=sha(vdb))
    provenance = dict(binary=str(binary), binary_sha256=sha(binary), legacy=args.legacy,
                      budget_percent=None if args.legacy else args.budget,
                      group_dispatches_maximum=None if args.legacy else args.dispatches,
                      parallel_samples_requested=None if args.legacy else args.parallel_samples,
                      label=args.label, input_signature=inputs)
    (args.out / 'plan.json').write_text(json.dumps(dict(plan, **provenance), indent=2) + '\n')
    rows = []
    for index in range(args.rounds):
        start = time.perf_counter()
        monitor=None; monitor_file=None; monitor_path=args.out/f'round_{index:02}_nvml.csv'
        if args.nvml:
            monitor_binary=shutil.which('nvidia-smi')
            if monitor_binary:
                monitor_file=monitor_path.open('w',encoding='utf-8')
                monitor=subprocess.Popen([monitor_binary,'--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.gr,temperature.gpu,pstate',
                    '--format=csv,noheader,nounits','-lms','200'],stdout=monitor_file,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        timed_out = False
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace',
                                       creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            log, _ = process.communicate(timeout=args.seconds + 20)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()  # Only the process created above, never other GPU jobs.
            log, _ = process.communicate(timeout=5)
        finally:
            if monitor:
                if monitor.poll() is None: monitor.terminate()
                monitor.wait(timeout=5)
                monitor_file.close()
        (args.out / f'round_{index:02}.log').write_text(log, encoding='utf-8')
        row = dict(round=index, exit_code=process.returncode, timed_out=timed_out,
                   process_wall_seconds=time.perf_counter()-start, metrics=None, derived={}, valid=False)
        row['nvml']=nvml_summary(monitor_path) if monitor_path.exists() else None
        fatal = timed_out or process.returncode != 0 or any(marker in log.lower() for marker in
                 ['cloud viewer stopped:', 'cloud gpu film invalid', 'device lost', 'thread \'main\' panicked'])
        try:
            row['metrics'] = legacy_metrics(log) if args.legacy else parse(log)
            row['derived'] = derived(row['metrics'])
            row['derived']['nvml_device_activity_percent_mean'] = row['nvml']['gpu_activity_percent_mean'] if row['nvml'] else None
            paths=row['metrics'].get('complete_sample_paths')
            row['derived']['completed_paths_per_requested_second'] = paths/args.seconds if isinstance(paths,int) and paths>=0 else None
            elapsed = row['metrics'].get('viewer_elapsed_ms')
            row['valid'] = not fatal and (row['metrics']['work_dispatches'] > 0 if args.legacy else
                positive(row['metrics'].get('compute_window_ms')) and positive(elapsed) and elapsed >= args.seconds*1000-100)
            row['derived']['whole_sample_paths_per_requested_second'] = row['metrics']['completed_spp']*args.width*args.height/args.seconds
            if not row['valid'] and not fatal:
                row['error'] = 'Viewer ended early or had no measured work; do not rank this run'
            if any(isinstance(row['metrics'].get(key),(int,float)) and row['metrics'][key]>25
                   for key in ['gpu_group_max_ms','gpu_chunk_max_ms']):
                row['valid']=False
                row['error']='GPU group/kernel exceeded25ms; later rounds must not run'
        except (ValueError, KeyError, TypeError) as error:
            row['error'] = str(error)
        rows.append(row)
        summary = dict(provenance=provenance, rows=rows, medians=summarize(rows),
                       all_runs_valid=all(row['valid'] for row in rows),
                       scope='Compute GPU time / first-work-to-snapshot window is duty, not SM utilization. '
                             'Host completion includes mapping/polling. Process wall includes startup. '
                             'Raw program metrics retain their recent-quantile and lifetime-max scopes. '
                             'Completed paths/s uses the compute window; completed paths/requested-second includes startup. '
                             'Legacy GPU time/duty and completed partial paths/s remain unavailable. '
                             'Whole-sample paths per requested second is a lower bound when the last batch is partial.')
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(dict(round=index, valid=row['valid'], derived=row['derived'])), flush=True)
        if not row['valid']:
            raise SystemExit('Invalid/missing metrics or failed GPU run; later rounds were not started')


def compare(args):
    if args.out.exists():
        raise ValueError('Choose a new comparison JSON path')
    a, b = [json.loads(path.read_text()) for path in [args.reference, args.candidate]]
    if not a['all_runs_valid'] or not b['all_runs_valid']:
        raise ValueError('Cannot rank failed or incomplete profiling runs')
    if a['provenance']['input_signature'] != b['provenance']['input_signature']:
        raise ValueError('Physical scene/film/target/duration inputs differ')
    def medians(data):
        rows=[]
        seconds=data['provenance']['input_signature']['seconds']
        for row in data['rows']:
            values=dict(row['derived'])
            paths=row['metrics'].get('complete_sample_paths')
            values['completed_paths_per_requested_second']=paths/seconds if isinstance(paths,int) and paths>=0 else None
            rows.append(dict(derived=values))
        return summarize(rows)
    am,bm=medians(a),medians(b)
    changes = {key: dict(reference=am[key], candidate=bm[key],
                        relative_change_percent=(bm[key]/am[key]-1)*100
                        if positive(am[key]) and bm[key] is not None else None)
               for key in METRICS}
    report = dict(reference=str(args.reference), candidate=str(args.candidate), metrics=changes,
                  gpu_initialized=False, interpretation='Compare completed paths/s and latency with the same scene. '
                  'Higher GPU duty alone does not establish more useful work; no Task Manager 3D percentage is used.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(changes, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run')
    r.add_argument('--binary', type=Path, default=ROOT/'target/release/cloud-demo.exe')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--scene', type=Path)
    r.add_argument('--legacy', action='store_true', help='Frozen old binary: omit new flags, archive only available stdout counters')
    r.add_argument('--nvml', action='store_true', help='Optional read-only nvidia-smi200ms driver activity monitor, identical for old/new')
    r.add_argument('--label', default='current')
    r.add_argument('--width', type=int, default=128)
    r.add_argument('--height', type=int, default=72)
    r.add_argument('--spp', type=int, default=1024)
    r.add_argument('--seconds', type=float, default=10)
    r.add_argument('--rounds', type=int, default=3)
    r.add_argument('--budget', type=int, default=80)
    r.add_argument('--dispatches', type=int, default=4)
    r.add_argument('--parallel-samples', type=int, default=4)
    r.add_argument('--dry-run', action='store_true')
    r.set_defaults(fn=run)
    c = sub.add_parser('compare')
    c.add_argument('reference', type=Path)
    c.add_argument('candidate', type=Path)
    c.add_argument('--out', type=Path, required=True)
    c.set_defaults(fn=compare)
    a = p.parse_args()
    if a.action == 'run' and (not math.isfinite(a.seconds) or not 1<=a.seconds<=30 or not 1<=a.rounds<=6
            or not 1<=a.width<=4096 or not 1<=a.height<=4096 or not 1<=a.spp<=16_777_216
            or not 10<=a.budget<=100 or not 1<=a.dispatches<=8 or not 1<=a.parallel_samples<=4):
        p.error('Use 1..30s, 1..6 rounds, positive film/spp, budget10..100%, group1..8, parallel samples1..4')
    a.fn(a)


if __name__ == '__main__':
    main()
