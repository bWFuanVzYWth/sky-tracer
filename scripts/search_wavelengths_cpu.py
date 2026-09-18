"""Search positive eight-node quadratures against the current frozen LUT (CPU/f32).

Fit on train, select diverse alternatives on validation, never use visual pixels.
The teacher and all transport remain full spectral. A candidate changes only the
single-scattering + ground-boundary conversion; the indirect RGB is held fixed.
"""
import argparse
import json
import itertools
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

F = np.float32
REGIONS = ['daylight', 'aureole', 'sunset', 'twilight', 'high_shadow', 'upper_space', 'ground_day']


def read_dataset(path):
    meta = json.loads((path / 'dataset.json').read_text())
    plan = json.loads((path / 'queries.json').read_text())
    shape = tuple(meta['array_shape'])
    arrays = {}
    for name in ['total', 'single', 'boundary']:
        file = path / (name + '.f32')
        assert file.stat().st_size == int(np.prod(shape)) * 4
        arrays[name] = np.array(np.memmap(file, dtype='<f4', mode='r', shape=shape).T, dtype=F)
        assert np.isfinite(arrays[name]).all()
    arrays['direct'] = arrays['single'] + arrays['boundary']
    return meta, plan, arrays


def stats(values):
    if len(values) == 0:
        return None
    return {k: float(v) for k, v in zip(['p50', 'p95', 'p99', 'max'],
        np.quantile(values, np.array([.5, .95, .99, 1], dtype=F)) * F(100))}


def metrics(indices, matrix, data, W, total_rgb, norm, labels, split, pairs):
    mask = (split == 'validation') & (norm > F(1e-8))
    delta = data['direct'][:, indices] @ matrix - data['direct'] @ W
    rel = np.linalg.norm(delta, axis=1) / np.maximum(norm, F(1e-7))
    full_delta = data['total'][:, indices] @ matrix - total_rgb
    out = {'direct_error_over_total_percent': stats(rel[mask]),
           'bright_direct_error_percent': stats(rel[mask & (norm > F(1e-4))]),
           'regions': {r: stats(rel[mask & (labels == r)]) for r in REGIONS},
           'full_spectrum_error_percent': stats((np.linalg.norm(full_delta, axis=1) / np.maximum(norm, F(1e-7)))[mask])}
    valid_pairs = pairs[(split[pairs[:, 0]] == 'validation') &
                        (norm[pairs[:, 0]] > F(1e-8)) & (norm[pairs[:, 1]] > F(1e-8))]
    p, q = valid_pairs.T
    gradients = np.linalg.norm(delta[p] - delta[q], axis=1) / np.maximum((norm[p] + norm[q]) * F(.5), F(1e-7))
    out['adjacent_error_change_percent'] = stats(gradients)
    out['worst_region_p99_percent'] = max(v['p99'] for v in out['regions'].values() if v)
    return out


class Objective:
    def __init__(self, name, B, constraints):
        self.name, self.B, self.C = name, B, constraints
        self.target = B.sum(axis=1, dtype=F)
        self.cache = {}

    def fit(self, indices):
        indices = tuple(sorted(indices))
        if indices in self.cache:
            return self.cache[indices]
        C = self.C[:, indices]
        constraint_count = len(C)
        u, s, vh = np.linalg.svd(C, full_matrices=True)
        result = None
        if s[-1] > F(1e-7):
            x0 = vh[:constraint_count].T @ ((u.T @ np.ones(constraint_count, dtype=F)) / s)
            null = vh[constraint_count:].T
            sub = self.B[:, indices]
            z = np.linalg.lstsq(sub @ null, self.target - sub @ x0, rcond=F(1e-6))[0]
            weights = x0 + null @ z
            if np.min(weights) >= F(.02) and np.max(weights) <= F(20) and np.max(np.abs(C @ weights - F(1))) < F(5e-5):
                residual = sub @ weights - self.target
                result = (float(np.dot(residual, residual)), weights)
        self.cache[indices] = result
        return result

    def bounded_baseline(self, indices):
        """Exact face enumeration for the old-node control, entirely in f32.

        The interior optimum can need negative weights on the old nodes. Keep
        those nodes fixed and find the constrained optimum rather than dropping
        the control or comparing against an unfairly unconstrained fit.
        """
        indices = tuple(indices)
        C, sub = self.C[:, indices], self.B[:, indices]
        constraint_count = len(C)
        best = self.fit(indices)
        for active_count in range(1, 9-constraint_count):
            for active_tuple in itertools.combinations(range(8), active_count):
                active = list(active_tuple)
                free = [i for i in range(8) if i not in active]
                for bounds in itertools.product([F(.02), F(20)], repeat=active_count):
                    weights = np.zeros(8, dtype=F)
                    weights[active] = bounds
                    rhs = np.ones(constraint_count, dtype=F) - C[:, active] @ weights[active]
                    u, s, vh = np.linalg.svd(C[:, free], full_matrices=True)
                    if s[-1] < F(1e-7):
                        continue
                    x0 = vh[:constraint_count].T @ ((u.T @ rhs) / s)
                    null = vh[constraint_count:].T
                    target = self.target - sub[:, active] @ weights[active] - sub[:, free] @ x0
                    z = np.linalg.lstsq(sub[:, free] @ null, target, rcond=F(1e-6))[0]
                    weights[free] = x0 + null @ z
                    if np.min(weights) < F(.01999) or np.max(weights) > F(20.00001) or np.max(np.abs(C @ weights - F(1))) > F(5e-5):
                        continue
                    residual = sub @ weights - self.target
                    score = float(np.dot(residual, residual))
                    if best is None or score < best[0]:
                        best = (score, weights)
        return best


def feature_basis(data, W, norm, region, train, pairs, name):
    weights = np.zeros(len(norm), dtype=F)
    multiplier = dict.fromkeys(REGIONS, 1)
    if name == 'twilight':
        multiplier.update(twilight=6, high_shadow=6, sunset=2)
    elif name == 'aureole':
        multiplier.update(aureole=8, daylight=2)
    for r in REGIONS:
        sel = train & (region == r)
        weights[sel] = F(multiplier[r] / max(1, np.count_nonzero(sel)))
    weights /= weights.sum(dtype=F)
    denominator = np.maximum(norm, F(1e-7))
    factor = np.sqrt(weights[train]) / denominator[train]
    # Fit components as well as their sum, avoiding S1/boundary cancellation.
    parts = [('direct', .55), ('single', .35), ('boundary', .10)]
    if name == 'general':
        parts = [('direct', .4), ('single', .2), ('boundary', .1), ('total', .3)]
    blocks = []
    for key, fraction in parts:
        A = data[key][train, None, :] * W.T[None, :, :]
        A *= (factor * F(np.sqrt(F(fraction))))[:, None, None]
        blocks.append(A.reshape(-1, len(W)))
    if name == 'gradient':
        selected = pairs[train[pairs[:, 0]] & train[pairs[:, 1]]]
        p, q = selected.T
        scale = np.sqrt((weights[p] + weights[q]) * F(.5)) / np.maximum((norm[p] + norm[q]) * F(.5), F(1e-7))
        A = (data['direct'][p] - data['direct'][q])[:, None, :] * W.T[None, :, :]
        blocks.append((A * (F(3) * scale)[:, None, None]).reshape(-1, len(W)))
    A = np.concatenate(blocks)
    assert A.dtype == np.float32
    _, s, vh = np.linalg.svd(A, full_matrices=False)
    return s[:, None] * vh


def search(obj, old, seed, starts):
    rng = np.random.default_rng(seed)
    n = obj.B.shape[1]
    minima = []
    for start in range(starts):
        if start == 0 and obj.fit(old) is not None:
            nodes = tuple(old)
        else:
            for attempt in range(10000):
                # Stratification includes blue, green and red on every restart.
                nodes = tuple(sorted([int(rng.integers(lo, hi)) for lo, hi in
                    [(0, 7), (7, 12), (12, 17), (17, 21), (21, 25), (25, 29), (29, 34), (34, n)]]))
                if obj.fit(nodes) is not None:
                    break
            else:
                raise RuntimeError('No positive feasible seed')
        for sweep in range(12):
            best_nodes, best = nodes, obj.fit(nodes)
            for lane in range(8):
                for new in range(n):
                    if new in nodes:
                        continue
                    trial = tuple(sorted(nodes[:lane] + nodes[lane + 1:] + (new,)))
                    fit = obj.fit(trial)
                    if fit is not None and fit[0] < best[0] * (1 - 1e-6):
                        best_nodes, best = trial, fit
            if nodes == best_nodes:
                break
            nodes = best_nodes
        minima.append(nodes)
        print(f'{obj.name}: start {start+1}/{starts}, loss={obj.fit(nodes)[0]:.7g}, unique={len(obj.cache)}', flush=True)
    feasible = [(v[0], k, v[1]) for k, v in obj.cache.items() if v is not None]
    return sorted(feasible, key=lambda x: x[0])[:64], minima


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--starts', type=int, default=16)
    p.add_argument('--free-span', action='store_true', help='Constrain solar RGB, without requiring weights to sum to 410nm')
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('use a new output directory')
    a.out.mkdir(parents=True)
    started = time.perf_counter()
    meta, plan, data = read_dataset(a.dataset)
    W = np.array(meta['rgb_from_integrated'], dtype=F)
    # No visual data enters the optimizer or candidate validation/ranking.
    numerical_n = next(i for i, l in enumerate(plan['labels']) if l['split'] == 'visual')
    data = {k: v[:numerical_n] for k, v in data.items()}
    split = np.array([l['split'] for l in plan['labels'][:numerical_n]])
    region = np.array([l['region'] for l in plan['labels'][:numerical_n]])
    pairs = np.array(plan['adjacent_pairs'], dtype=np.int32)
    total_rgb = data['total'] @ W
    norm = np.linalg.norm(total_rgb, axis=1)
    train = (split == 'train') & (norm > F(1e-8))
    bands = meta['bands']
    widths = np.array([b['upper_nm'] - b['lower_nm'] for b in bands], dtype=F)
    nm = np.array([b['center_nm'] for b in bands], dtype=F)
    solar = np.array([b['solar_irradiance_w_m2'] for b in bands], dtype=F)
    C = np.vstack(((solar[:, None] * W / (solar @ W)[None, :]).T, widths / widths.sum(dtype=F)))
    if a.free_span:
        C = C[:3]
    old = tuple(meta['old_demo_indices'])
    baselines = []
    pool = []
    search_summary = []
    all_combinations = set()
    for oi, name in enumerate(['balanced', 'twilight', 'aureole', 'gradient', 'general']):
        obj = Objective(name, feature_basis(data, W, norm, region, train, pairs, name), C)
        if name == 'balanced':
            old_refit = obj.bounded_baseline(old)
            if old_refit is not None:
                baselines.append(dict(id='old_refit', indices=list(old), quadrature_weights_nm=(old_refit[1] * widths[list(old)]).tolist(),
                    rgb_from_integrated=(old_refit[1][:, None] * W[list(old)]).tolist()))
        best, minima = search(obj, old, 2026091808 + oi * 731, a.starts)
        for loss, indices, x in best:
            pool.append(dict(objective=name, train_loss=loss, indices=list(indices),
                wavelengths_nm=nm[list(indices)].tolist(), quadrature_weights_nm=(x * widths[list(indices)]).tolist(),
                rgb_from_integrated=(x[:, None] * W[list(indices)]).tolist(),
                constraint_max_residual=float(np.max(np.abs(C[:, indices] @ x - F(1))))))
        search_summary.append(dict(objective=name, evaluated=len(obj.cache), feasible=sum(v is not None for v in obj.cache.values()),
            local_minima=len(set(minima))))
        all_combinations.update(obj.cache)
    baselines.insert(0, dict(id='old_demo', indices=list(old), rgb_from_integrated=meta['old_demo_rgb_from_integrated']))
    for item in baselines + pool:
        item['wavelengths_nm'] = nm[item['indices']].tolist()
        item['metrics'] = metrics(item['indices'], np.array(item['rgb_from_integrated'], dtype=F), data, W, total_rgb, norm, region, split, pairs)
    # Validation chooses representative tradeoffs; full visual images stay sealed.
    roles = [
        ('balanced', lambda m: m['direct_error_over_total_percent']['p95']),
        ('worst_tail', lambda m: m['worst_region_p99_percent']),
        ('twilight', lambda m: max(m['regions'][r]['p99'] for r in ['twilight', 'high_shadow'])),
        ('aureole', lambda m: m['regions']['aureole']['p99']),
        ('gradient', lambda m: m['adjacent_error_change_percent']['p99']),
        ('general', lambda m: m['full_spectrum_error_percent']['p99']),
    ]
    selected = []
    for role, key in roles:
        eligible = [v for v in pool if all(len(set(v['indices']) ^ set(s['indices'])) >= 4 for s in selected)]
        # Exclude obviously failed alternatives; preserve tradeoffs, not a winner.
        eligible = [v for v in eligible if v['metrics']['direct_error_over_total_percent']['p95'] < baselines[0]['metrics']['direct_error_over_total_percent']['p95']]
        if not eligible:
            continue
        choice = dict(min(eligible, key=lambda v: key(v['metrics'])))
        choice['id'] = f'C{len(selected)+1}_{role}'
        choice['role'] = role
        selected.append(choice)
    for item in baselines + selected:
        indices = item['indices']
        item['kind'] = 'frozen_lut_eight_wave_candidate_v1'
        item['model'] = meta['model']
        item['source_band_checksums'] = meta['source_band_checksums']
        item['rec2020_from_per_nm'] = (np.array(item['rgb_from_integrated'], dtype=F) * widths[indices, None]).tolist()
        item['sun_irradiance_per_nm'] = (solar[indices] / widths[indices]).tolist()
        (a.out / (item['id'] + '.json')).write_text(json.dumps(item, indent=2))
    result = dict(kind='current_lut_wavelength_search_v1', dataset=str(a.dataset), model=meta['model'],
        source_band_checksums=meta['source_band_checksums'], seed=2026091808, starts=a.starts,
        numeric_precision='float32 transport, features, SVD and fitting',
        constraints='8 distinct 10nm teacher nodes; positive scalar quadrature weights; exact full-reference solar RGB; ' + ('weight sum unconstrained' if a.free_span else '410nm constant integral') + '; no new white adaptation',
        error_definition='norm((S1+B)_8-(S1+B)_41) / max(norm(Ltotal_41),1e-7); excludes norms <=1e-8; percent',
        selection='train fit; validation numerical selection with >=2 node changes between alternatives; visual images never used here',
        counts={s: int(np.count_nonzero(split == s)) for s in ['train','validation']},
        region_validation_counts={r: int(np.count_nonzero((split == 'validation') & (region == r) & (norm > F(1e-8)))) for r in REGIONS},
        search=search_summary, unique_combinations=len(all_combinations), baselines=baselines, selected=selected, pool=pool, elapsed_seconds=time.perf_counter()-started)
    (a.out / 'search.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(elapsed=result['elapsed_seconds'], candidates=[dict(id=c['id'],nm=c['wavelengths_nm'],metrics=c['metrics']) for c in selected]), indent=2))


if __name__ == '__main__':
    main()
