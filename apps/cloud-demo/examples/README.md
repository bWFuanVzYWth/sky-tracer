# Explicit cloud GPU checks

These examples create a GPU device only when run. Use separate output directories
and run them serially with baking, fitting, or other rendering work stopped.

`environment_smoke` is independent of the window, UI, and realtime sky renderer.
The short preview check compares 4×2×4 Disney-cloud mean/variance bits against
`out/cloud_bounded_qa_v1/budget128_4x2x4/reference`, then checks a 128×72 zero-density
scene. Its first 4096-path chunk covers every image row; all 9216 pixels receive a
sample after three chunks. It checks partial-film and target/environment mutation
guards, reset and target changes, exact constant RGB, and zero variance.

```powershell
cargo build -p cloud-demo --example environment_smoke --release --offline
target\release\examples\environment_smoke.exe out\cloud_preview_qa_v1 --preview-only
```

Without `--preview-only`, it also reruns the offline constant-sky mode and the
directional 1×1 environment/solar-input switching checks. Every compute chunk is
bounded to 4096 path slots and a fixed transition count per slot. Timestamp checks stop new
submissions if a chunk exceeds 100 ms. Offline chunks use 128 transitions;
preview chunks use `PREVIEW_WORK_TRANSITIONS` (currently 128). No legacy full-path
kernel is used.

The explicit workload probe runs a real 128×72 cloud for at most 64 chunks. It
stops new submissions above 16 ms, records per-chunk GPU time and visible pixel
coverage, and exports no partially sampled reference:

```powershell
target\release\examples\environment_smoke.exe out\cloud_preview_work_probe_new --preview-work-probe
```

The grouped check compares four parallel samples per pixel with the frozen
4×2×4 reference using groups of 1, 4 and 8 bounded kernels. A zero-density
128×72 scene then verifies partial-count rejection, reset, and a four-sample
batch followed by a three-sample batch. Each group uses ordered uniform
snapshots and one final progress completion. Individual and whole-group GPU
timestamps are recorded; kernels retain the 128-transition bound.

```powershell
target\release\examples\environment_smoke.exe out\cloud_grouped_preview_new --grouped-preview
target\release\examples\environment_smoke.exe out\cloud_grouped_work_probe_new --grouped-work-probe
```

The grouped workload probe tries real-cloud group sizes 1, 4 and 8 in sequence.
It stops above 100 ms and skips a larger size if the preceding group predicts
that limit. Its short latency observations guide interactive submission size;
they do not establish throughput over complete paths or a stable P99.

`bounded_smoke` retains the offline scheduling check (dimensions at most 8×4 and
32 samples), with EXR/variance export and work-ticket/reset checks:

```powershell
cargo run -p cloud-demo --example bounded_smoke --release --offline -- 4 2 4 out\cloud_bounded_smoke_new
```

CPU compilation and shader validation create no GPU device:

```powershell
cargo check -p cloud-demo --examples --offline
cargo test -p cloud-pt --lib --offline
```
