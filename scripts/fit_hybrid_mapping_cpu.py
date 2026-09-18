"""Fit cheap source-LUT mappings on CPU, without changing the runtime solver.

Input: `cargo run --release -p sky-atmosphere-lut --example mapping_curves_cpu`.
Curvature is an allocation prior; final ranking measures actual interpolation.
The teacher is SH16/f16 derived from the frozen reference, not ground truth.
Field data, physical mapping evaluation and interpolation objectives use float32.
Optimizer bookkeeping, statistics and NumPy node resampling may use float64;
nodes/parameters are cast before evaluation. No GPU work or runtime inversion.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution

F = np.float32
PI = F(np.pi)


def interp(x, y, q):
    """Independent sorted curves: x [C,N], y [C,N,K], q [C,Q]."""
    ids = np.stack([np.searchsorted(xx, qq, side="right") - 1 for xx, qq in zip(x, q)])
    ids = np.clip(ids, 0, x.shape[1] - 2)
    rows = np.arange(len(x))[:, None]
    a, b = x[rows, ids], x[rows, ids + 1]
    t = np.clip((q - a) / np.maximum(b - a, F(1e-20)), F(0), F(1))
    return y[rows, ids] * (F(1) - t[..., None]) + y[rows, ids + 1] * t[..., None]


def invert(mapping, u, lo, hi):
    lo, hi = np.broadcast_arrays(np.asarray(lo, dtype=F) + np.zeros_like(u),
                                  np.asarray(hi, dtype=F) + np.zeros_like(u))
    lo, hi = lo.copy(), hi.copy()
    for _ in range(25):
        m = (lo + hi) * F(.5)
        above = mapping(m) >= u
        hi = np.where(above, m, hi)
        lo = np.where(above, lo, m)
    return (lo + hi) * F(.5)


def horizon(h):
    return -np.sqrt(h * (F(12720) + h)) / (F(6360) + h)


def solar_old(mu, h):
    e = np.arcsin(np.clip(mu, -1, 1))
    eh = np.arcsin(horizon(h))
    def ac(c, w):
        w = F(w * np.pi / 180)
        a = np.arctan((-PI * F(.5) - c) / w)
        return (np.arctan((e - c) / w) - a) / (np.arctan((PI * F(.5) - c) / w) - a)
    d, cap = PI * F(.5) - e, F(5 * np.pi / 180)
    return np.clip(F(.25) * (e / PI + F(.5)) + F(.20) * ac(eh, 2)
        + F(.20) * ac(eh - F(6 * np.pi / 180), 6) + F(.15) * ac(-eh, .5)
        + F(.20) * (F(1) - np.sqrt(d / (d + cap)) / np.sqrt(PI / (PI + cap))), 0, 1)


def solar_parameters(p):
    p = np.asarray(p, dtype=F)
    weights = np.exp(np.r_[F(0), p[:3]].astype(F))
    weights /= weights.sum(dtype=F)
    return weights, np.exp(p[3:6]).astype(F)


def solar_fast(mu, h, p):
    mu, h = np.broadcast_arrays(mu, h)
    weights, widths = solar_parameters(p)
    hor = horizon(h)
    result = weights[0] * (mu + F(1)) * F(.5)
    for w, width, c in zip(weights[1:], widths, [hor, hor - F(.10452846), -hor]):
        def soft(x):
            return x / (np.abs(x) + width)
        a, b = soft(-F(1) - c), soft(F(1) - c)
        result += w * (soft(mu - c) - a) / (b - a)
    return np.clip(result, 0, 1)


def height_cdf(h):
    return (F(.35) * np.log1p(np.sqrt(h / F(.01))) / np.log1p(np.sqrt(F(12000)))
            + F(.65) * h / F(120))


def old_heights():
    u = np.linspace(0, 1, 48, dtype=F)[None]
    nodes = invert(lambda x: height_cdf(x * x), u, F(0), np.sqrt(F(120)))[0] ** 2
    nodes[0], nodes[-1] = 0, 120
    nodes[1:-1][np.argmin(abs(nodes[1:-1] - F(35)))] = 35
    return nodes


def phase_old(q):
    return F(.5) * np.arccos(np.clip(F(1) - F(2) * q, -1, 1)) / PI + F(.5) * np.cbrt(q)


def phase_weighted(q, a):
    return F(a) * np.arccos(np.clip(F(1) - F(2) * q, -1, 1)) / PI + F(1-a) * np.cbrt(q)


def phase_fast(q, p):
    a, b = np.exp(F(p[0])), F(p[1])
    left, right = np.sqrt(q), np.sqrt(F(1) - q)
    z = left / (left + right)
    return b * z + (F(1) - b) * (F(1) + a) * z / (z + a)


def solar_angle_parameters(p):
    p = np.asarray(p, dtype=F)
    weights = np.exp(np.r_[F(0), p[:4]].astype(F))
    weights /= weights.sum(dtype=F)
    return weights, np.exp(p[4:7]).astype(F)


def solar_angle_fast(mu, h, p):
    mu, h = np.broadcast_arrays(mu, h)
    weights, widths = solar_angle_parameters(p)
    e = np.arcsin(np.clip(mu, -1, 1))
    hor = np.arcsin(horizon(h))  # Precomputed per stored height, not per query.
    result = weights[0] * (e / PI + F(.5))
    for w, width, c in zip(weights[1:4], widths, [hor, hor - F(.10471976), -hor]):
        def soft(x):
            return x / (np.abs(x) + width)
        a, b = soft(-PI*F(.5) - c), soft(PI*F(.5) - c)
        result += w * (soft(e - c) - a) / (b - a)
    d, cap = PI*F(.5)-e, F(.08726646)
    result += weights[4] * (F(1)-np.sqrt(d/(d+cap))/np.sqrt(PI/(PI+cap)))
    return np.clip(result, 0, 1)


def cone_fast(c, k):
    return c + F(k) * c * (F(1) - c) * (F(2) * c - F(1))


class Dataset:
    def __init__(self, root, metadata, ids=None, stride=1):
        self.name = metadata["axis"]
        c, n, k = metadata["shape"]
        if ids is None:
            ids = np.arange(c)
        self.x = np.fromfile(root / f"{self.name}.x.f32", dtype="<f4").reshape(c, n)[ids]
        self.y = np.fromfile(root / f"{self.name}.values.f32", dtype="<f4").reshape(c, n, k)[ids]
        self.pose = np.array([metadata["curves"][i]["pose"] for i in ids], dtype=F)
        self.query_x = self.x[:, ::stride]
        self.query_y = self.y[:, ::stride]
        self.shift = self.y[:, :, 4:].max(axis=(1, 2), keepdims=True)
        self.truth = self.decode(self.query_y)
        self.norm = np.linalg.norm(self.truth, axis=-1)
        self.floor = np.maximum(self.norm.max(axis=1, keepdims=True) * F(1e-5), F(1e-25))
        self.active = (self.norm > self.floor) & (self.shift[:, 0, 0, None] > F(-35))
        self.den = np.maximum(self.norm, self.floor)

    def decode(self, y):
        return y[:, :, :4] * np.exp(y[:, :, 4:] - self.shift)

    def predicted(self, mapping=None, count=None, nodes=None):
        if nodes is not None:
            nodes = np.broadcast_to(np.asarray(nodes, dtype=F)[None], (len(self.x), len(nodes)))
            values = interp(self.x, self.y, nodes)
            return self.decode(interp(nodes, values, self.query_x))
        u = np.broadcast_to(np.linspace(0, 1, count, dtype=F), (len(self.x), count))
        left, right = self.x[:, :1], self.x[:, -1:]
        nodes = invert(mapping, u, left, right)
        nodes[:, 0], nodes[:, -1] = left[:, 0], right[:, 0]
        values = interp(self.x, self.y, nodes)
        return self.decode(interp(u, values, mapping(self.query_x)))

    def error(self, predicted):
        return np.linalg.norm(predicted - self.truth, axis=-1) / self.den

    def objective(self, predicted):
        e = self.error(predicted)[self.active]
        # Relative-error objective avoids letting bright daytime drown twilight.
        # Cap only optimizer outliers; published statistics are never capped.
        e = np.minimum(e, F(2))
        return float(np.mean(e * e, dtype=F))

    def metrics(self, predicted):
        error = self.error(predicted)
        e = error[self.active]
        per_curve = []
        for i, row in enumerate(error):
            active = self.active[i]
            if active.any():
                per_curve.append(dict(pose=self.pose[i].tolist(), p95_percent=float(100*np.percentile(row[active], 95))))
        return dict(points=int(len(e)), rms_percent=float(100*np.sqrt(np.mean(e*e, dtype=F))),
                    p95_percent=float(100*np.percentile(e, 95)), p99_percent=float(100*np.percentile(e, 99)),
                    max_percent=float(100*e.max()), worst_curves=sorted(per_curve, key=lambda x:x["p95_percent"], reverse=True)[:8])


def height_curvature(d):
    """Physical-height error monitor for linear shape + linear log-mean.

    Leading reconstructed-source error is exp(g)*(S''+S*g'')*dx^2/8.
    sqrt(normalized curvature) is an equal-local-error allocation prior.
    Gaussian smoothing is in sample index, not a claim of physical smoothness.
    """
    x = d.x[0]
    all_k = []
    for i, y in enumerate(d.y):
        if d.shift[i, 0, 0] < -35:
            continue
        smooth = gaussian_filter1d(y, 2, axis=0, mode="nearest")
        first = np.gradient(smooth, x, axis=0)
        second = np.gradient(first, x, axis=0)
        value = np.exp(smooth[:, 4:] - d.shift[i]) * (second[:, :4] + smooth[:, :4] * second[:, 4:])
        norm = np.linalg.norm(y[:, :4] * np.exp(y[:, 4:] - d.shift[i]), axis=-1)
        floor = max(norm.max()*F(1e-5), F(1e-25))
        den = np.maximum(norm, floor)
        k = np.linalg.norm(value, axis=-1) / den
        k[norm <= floor] = 0
        all_k.append(k)
    k = np.percentile(np.array(all_k, dtype=F), 85, axis=0).astype(F)
    # Protect against derivative noise at the tiny first/last height cells.
    k[:4], k[-4:] = k[4], k[-5]
    rho = gaussian_filter1d(np.sqrt(np.maximum(k, F(0))), 3).astype(F)
    return x, rho


def height_candidate(x, rho, strength):
    result = []
    for lo, hi, count in [(0, 35, 24), (35, 120, 25)]:
        mask = (x >= lo) & (x <= hi)
        xx = np.r_[F(lo), x[mask & (x > lo) & (x < hi)], F(hi)].astype(F)
        rr = np.interp(xx, x, rho).astype(F)
        cdf = np.r_[F(0), np.cumsum((rr[1:]+rr[:-1])*F(.5)*np.diff(xx), dtype=F)]
        cdf /= max(cdf[-1], F(1e-20))
        base = (height_cdf(xx) - height_cdf(F(lo))) / (height_cdf(F(hi)) - height_cdf(F(lo)))
        cdf = F(strength)*cdf + F(1-strength)*base
        nodes = np.interp(np.linspace(0, 1, count, dtype=F), cdf, xx).astype(F)
        # Low-atmosphere physical-profile anchors, retained without extra nodes.
        if lo == 0:
            used = {0, count-1}
            for anchor in [1, 2, 11, 12]:
                ids = [i for i in range(1, count-1) if i not in used]
                j = min(ids, key=lambda i:abs(nodes[i]-anchor))
                nodes[j] = anchor
                used.add(j)
            nodes.sort()
        result.extend(nodes if not result else nodes[1:])
    return np.array(result, dtype=F)


def self_test():
    q = np.linspace(0, 1, 1001, dtype=F)[None]
    for fun in [phase_old, lambda q:phase_fast(q, [-3,.2]), lambda q:cone_fast(q, -.8)]:
        u = fun(q)
        assert u.dtype == F and np.all(np.diff(u)>0) and abs(u[0,0])<1e-6 and abs(u[0,-1]-1)<1e-6
        restored = invert(fun, u, F(0), F(1))
        assert np.max(abs(restored-q))<2e-6
    h = np.array([0, .2, 12, 35, 108, 120], dtype=F)[:,None]
    x = np.broadcast_to(np.linspace(-1,1,2001,dtype=F), (len(h),2001))
    for p in [[0,0,0,-3,-2,-4], [2,-2,1,-4,-3,-1]]:
        u = solar_fast(x,h,p)
        assert np.all(np.diff(u,axis=1)>0) and np.max(abs(u[:,0]))<1e-6 and np.max(abs(u[:,-1]-1))<1e-6
    u = solar_angle_fast(x,h,[0,0,0,0,-3,-2,-4])
    assert np.all(np.diff(u,axis=1)>0) and np.max(abs(u[:,0]))<1e-6 and np.max(abs(u[:,-1]-1))<1e-6
    assert np.array_equal(u, solar_angle_fast(x[:1],h,[0,0,0,0,-3,-2,-4]))
    nodes = old_heights()
    assert len(nodes)==48 and np.all(np.diff(nodes)>0) and sum(nodes<=35)==24
    x = np.array([[0,.3,1]],dtype=F)
    y = (2*x+3)[...,None]
    q = np.linspace(0,1,101,dtype=F)[None]
    assert np.max(abs(interp(x,y,q)[...,0]-(2*q+3)))<1e-6


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--solar-nodes", type=int, default=176)
    p.add_argument("--check-fit", type=Path, help="Evaluate saved candidates on new curves; no fitting")
    a = p.parse_args()
    if not 8 <= a.solar_nodes <= 257:
        p.error("--solar-nodes must be in 8..257, matching the hybrid config")
    self_test()
    if a.out.exists():
        raise ValueError("Choose a new output directory")
    a.out.mkdir(parents=True)
    metadata = json.loads((a.source / "curves.json").read_text())
    if a.check_fit:
        check_fit(metadata,a)
        return
    results = {}
    start = time.perf_counter()
    for m in metadata["datasets"]:
        axis = m["axis"]
        train = [i for i,c in enumerate(m["curves"]) if c["split"]=="train"]
        val = [i for i,c in enumerate(m["curves"]) if c["split"]=="validation"]
        # Selection uses a fixed subset of training contexts only; all training
        # and held-out contexts are evaluated once for the selected candidate.
        chosen = np.array(train)[np.linspace(0,len(train)-1,min(40,len(train)),dtype=int)]
        d = Dataset(a.source,m,chosen,stride=4)
        count = {"height":48,"solar":a.solar_nodes,"phase":24,"cone":8}[axis]
        if axis == "height":
            prior_x, prior_rho = height_curvature(d)
            choices = [(d.objective(d.predicted(nodes=height_candidate(prior_x,prior_rho,s))),s)
                       for s in np.linspace(0,1,21)]
            _, strength = min(choices)
            nodes = height_candidate(prior_x,prior_rho,strength)
            candidate = dict(kind="explicit_height_nodes", nodes_km=nodes.tolist(), curvature_blend=float(strength),
                             low_count=24, high_count=25, runtime="existing binary search; unchanged sample count")
            def baseline(data): return data.predicted(nodes=old_heights())
            def fitted(data): return data.predicted(nodes=nodes)
        elif axis == "solar":
            def loss(v): return d.objective(d.predicted(lambda mu:solar_angle_fast(mu,d.pose[:,0,None],v),count))
            opt = differential_evolution(loss, [(-3,3)]*4+[(-5.5,-.5)]*3,
                seed=20260919, popsize=6, maxiter=40, polish=False, workers=1)
            params = np.asarray(opt.x,dtype=F)
            weights,widths = solar_angle_parameters(params)
            candidate = dict(kind="three_softsign_angle_cdfs", parameters=params.tolist(), weights=weights.tolist(), widths_radians=widths.tolist(),
                centers="horizon_elevation, horizon_elevation-6deg, -horizon_elevation",
                runtime="per height precompute centers/amplitudes/offset; query asin shared across layers + 3 softsign terms + sqrt cap; no atan")
            def baseline(data): return data.predicted(lambda mu:solar_old(mu,data.pose[:,0,None]),count)
            def fitted(data): return data.predicted(lambda mu:solar_angle_fast(mu,data.pose[:,0,None],params),count)
        elif axis == "phase":
            # Keep the original operator count; cheaper cosine-only alternatives
            # lost too much accuracy at one or both polar ends in this pilot.
            choices = [(d.objective(d.predicted(lambda q:phase_weighted(q,w),count)),float(w))
                       for w in np.linspace(.05,1,96)]
            _,weight = min(choices)
            candidate = dict(kind="weighted_current_phase", angle_weight=weight,
                formula="q=(1-nu)/2; u=a*acos(nu)/pi+(1-a)*cbrt(q)",
                runtime="same acos + cube root as current; only two constants change")
            def baseline(data): return data.predicted(phase_old,count)
            def fitted(data): return data.predicted(lambda q:phase_weighted(q,weight),count)
        else:
            choices = [(d.objective(d.predicted(lambda c:cone_fast(c,k),count)),float(k))
                       for k in np.r_[np.linspace(-1.8,.95,57),0.0]]
            _, k = min(choices)
            candidate = dict(kind="cubic_cosine", k=k, formula="c=(1-cos(psi))/2; u=c+k*c*(1-c)*(2*c-1)",
                             runtime="polynomial after existing geometric cosine; no acos")
            def baseline(data): return data.predicted(lambda c:np.arccos(np.clip(1-2*c,-1,1))/PI,count)
            def fitted(data): return data.predicted(lambda c:cone_fast(c,k),count)
        row = dict(count=count,candidate=candidate)
        for split,ids in [("train",train),("validation",val)]:
            data = Dataset(a.source,m,ids)
            row[split] = dict(curves=len(ids),baseline=data.metrics(baseline(data)),candidate=data.metrics(fitted(data)))
        results[axis] = row
        v = row["validation"]
        print(f'{axis}: validation P95 {v["baseline"]["p95_percent"]:.4f}% -> {v["candidate"]["p95_percent"]:.4f}%; RMS {v["baseline"]["rms_percent"]:.4f}% -> {v["candidate"]["rms_percent"]:.4f}% ({time.perf_counter()-start:.1f}s)',flush=True)
        (a.out / "fit.json").write_text(json.dumps(dict(kind="cpu_source_mapping_fit_pilot_v1",source=str(a.source),
            results=results,seconds=time.perf_counter()-start, teacher_limitations=metadata["limitations"],
            metric="four-wave normalized-source reconstructed with log incident mean; relative vector error; per-curve 1e-5 peak mask; axis-isolated",
            runtime_installed=False),indent=2),encoding="utf-8")
    plot(results,a.out)


def plot(results,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    q = np.linspace(0,1,2049,dtype=F)[None]
    for axis,ax in zip(["height","solar","phase","cone"],axes.flat):
        row = results[axis]
        c = row["candidate"]
        if axis=="height":
            for nodes,label in [(old_heights(),"Current"),(np.array(c["nodes_km"]),"Candidate")]:
                ax.plot(nodes,np.arange(len(nodes)),label=label)
            ax.set(xlabel="Altitude (km)",ylabel="Node index")
        elif axis=="solar":
            mu=q*2-1;h=np.array([[.2]],dtype=F)
            ax.plot(np.rad2deg(np.arcsin(mu[0])),solar_old(mu,h)[0]*(row["count"]-1),label="Current")
            ax.plot(np.rad2deg(np.arcsin(mu[0])),solar_angle_fast(mu,h,c["parameters"])[0]*(row["count"]-1),label="Candidate")
            ax.set(xlabel="Solar elevation at 200 m (degrees)",ylabel="Node index")
        elif axis=="phase":
            x=np.rad2deg(np.arccos(1-2*q[0]))
            ax.plot(x,phase_old(q)[0]*23,label="Current")
            ax.plot(x,phase_weighted(q,c["angle_weight"])[0]*23,label="Candidate")
            ax.set(xlabel="Angle to Sun (degrees)",ylabel="Node index")
        else:
            x=np.rad2deg(np.arccos(1-2*q[0]))
            ax.plot(x,x/180*7,label="Current")
            ax.plot(x,cone_fast(q,c["k"])[0]*7,label="Candidate")
            ax.set(xlabel="Cone angle around Sun (degrees)",ylabel="Node index")
        v=row["validation"]
        ax.set_title(f'{axis}: held-out source P95 {v["baseline"]["p95_percent"]:.3f}% -> {v["candidate"]["p95_percent"]:.3f}%')
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle("CPU mapping pilot: fixed node budgets; reference-derived SH source teacher\nAxis-isolated interpolation, not final sky error")
    fig.savefig(out/"mapping_allocation.png",dpi=150)
    plt.close(fig)


def check_fit(metadata,a):
    saved = json.loads(a.check_fit.read_text())
    results = {}
    for m in metadata["datasets"]:
        axis = m["axis"]
        data = Dataset(a.source,m)
        row = saved["results"][axis]
        c,n = row["candidate"],row["count"]
        if axis=="height":
            old = data.predicted(nodes=old_heights())
            new = data.predicted(nodes=c["nodes_km"])
        elif axis=="solar":
            old = data.predicted(lambda mu:solar_old(mu,data.pose[:,0,None]),n)
            new = data.predicted(lambda mu:solar_angle_fast(mu,data.pose[:,0,None],c["parameters"]),n)
        elif axis=="phase":
            old = data.predicted(phase_old,n)
            new = data.predicted(lambda q:phase_weighted(q,c["angle_weight"]),n)
        else:
            old = data.predicted(lambda q:np.arccos(np.clip(1-2*q,-1,1))/PI,n)
            new = data.predicted(lambda q:cone_fast(q,c["k"]),n)
        results[axis] = dict(baseline=data.metrics(old),candidate=data.metrics(new))
        r = results[axis]
        print(f'{axis}: new contexts P95 {r["baseline"]["p95_percent"]:.4f}% -> {r["candidate"]["p95_percent"]:.4f}%; RMS {r["baseline"]["rms_percent"]:.4f}% -> {r["candidate"]["rms_percent"]:.4f}%',flush=True)
    (a.out/"check.json").write_text(json.dumps(dict(fit=str(a.check_fit), source=str(a.source),
        sample_multiplier=metadata.get("sample_multiplier",1), shifted_validation=metadata.get("shifted_validation",False),
        results=results, teacher_limitations=metadata["limitations"]),indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
