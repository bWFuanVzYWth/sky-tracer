// Independently seeded pixel/sample paths, with optional parallel tracing and
// ordered reduction. Any diagnostic makes the complete film unusable.
struct Params {
    image: vec4<u32>, // width, height, sample index, roulette start
    control: vec4<u32>, // seed low/high, event watchdog, tile count
    storage: vec4<u32>, // brick/majorant hash masks, runtime-zero bits barrier, ground/global flags
    camera_origin: vec4<f32>, // xyz, tan(horizontal FOV / 2)
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    camera_forward: vec4<f32>,
    index_min: vec4<f32>, // w: safe coprime pixel permutation stride for preview
    index_max: vec4<f32>,
    transform: vec4<f32>, // translation xyz, positive uniform scale
    optics: vec4<f32>, // extinction scale, density majorant, HG g, ground y
    albedo: vec4<f32>,
    sun_direction: vec4<f32>, // direction toward the sun
    sun_irradiance: vec4<f32>,
    sky_radiance: vec4<f32>,
    ground_albedo: vec4<f32>,
    batch: vec4<u32>, // sample count, tile pixel offset/count, transitions per chunk
    work: vec4<u32>, // host epoch low/high, submission serial low/high
}
struct Tile { origin_size: vec4<u32>, density_pad: vec4<u32> }
struct Diagnostics {
    flags: atomic<u32>, first_pixel: atomic<u32>,
    first_sample: atomic<u32>, completed: atomic<u32>,
    ticket: vec4<u32>,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> brick_hash: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> density_values: array<f32>;
@group(0) @binding(3) var<storage, read> tiles: array<Tile>;
@group(0) @binding(4) var<storage, read_write> film_mean: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> film_m2: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> diagnostics: Diagnostics;
@group(0) @binding(7) var<storage, read> cell_majorant_hash: array<vec4<u32>>;
@group(0) @binding(9) var environment: texture_2d<f32>;
@group(0) @binding(10) var sunlight: texture_2d<f32>;

// Private state belongs to one invocation and is only used for diagnostics.
var<private> current_sample_index: u32;
var<private> current_pixel_index: u32;

const PI: f32 = 3.141592653589793;
const MAX_FLOAT: f32 = 3.402823466e38;
const ERROR_MAJORANT: u32 = 1u;
const ERROR_NONFINITE: u32 = 2u;
const ERROR_STAGNATION: u32 = 4u;
const ERROR_WATCHDOG: u32 = 8u;
const ERROR_PRECISION: u32 = 16u;
const ERROR_LOOKUP: u32 = 32u;
const ERROR_ENVIRONMENT: u32 = 64u;
// Static volume/camera transforms retain the host's 2^18 budget. Paired-float
// slab arithmetic permits continuation/ground origins up to 2^40 voxel scales.
const MAX_RAY_INDEX_MAGNITUDE: f32 = 1099511627776.0;

fn finite(v: f32) -> bool { return v == v && abs(v) <= MAX_FLOAT; }
fn finite3(v: vec3<f32>) -> bool { return all(v == v) && all(abs(v) <= vec3<f32>(MAX_FLOAT)); }
fn fail(flag: u32, pixel: u32) {
    let previous = atomicOr(&diagnostics.flags, flag);
    if previous == 0u {
        atomicStore(&diagnostics.first_pixel, pixel);
        atomicStore(&diagnostics.first_sample, current_sample_index);
    }
}
fn event(events: ptr<function, u32>, pixel: u32) -> bool {
    if *events >= p.control.z { fail(ERROR_WATCHDOG, pixel); return false; }
    *events += 1u;
    return true;
}
fn mix_bits(input: u32) -> u32 {
    var x = input;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}
fn random_u32(rng: ptr<function, u32>) -> u32 {
    // PCG RXS-M-XS 32/32. The state transition is a full-period LCG.
    *rng = *rng * 747796405u + 2891336453u;
    let word = ((*rng >> ((*rng >> 28u) + 4u)) ^ *rng) * 277803737u;
    return (word >> 22u) ^ word;
}
fn open01(rng: ptr<function, u32>) -> f32 {
    // 23 bits plus a half-bin remain strictly within (0, 1) in f32.
    return (f32(random_u32(rng) >> 9u) + 0.5) * (1.0 / 8388608.0);
}
fn find_brick(brick: vec3<i32>) -> u32 {
    let key = bitcast<vec3<u32>>(brick);
    var slot = (key.x * 73856093u ^ key.y * 19349663u ^ key.z * 83492791u) & p.storage.x;
    for (var probe = 0u; probe <= p.storage.x; probe += 1u) {
        let entry = brick_hash[slot];
        if entry.w == 0xffffffffu { return 0xffffffffu; }
        if all(entry.xyz == key) {
            return entry.w;
        }
        slot = (slot + 1u) & p.storage.x;
    }
    fail(ERROR_LOOKUP, current_pixel_index);
    return 0xffffffffu;
}
fn uniform_density(index: vec3<i32>) -> f32 {
    for (var i = 0u; i < p.control.w; i += 1u) {
        let tile = tiles[i];
        let origin = bitcast<vec3<i32>>(tile.origin_size.xyz);
        let size = i32(tile.origin_size.w);
        if all(index >= origin) && all(index - origin < vec3<i32>(size)) {
            return bitcast<f32>(tile.density_pad.x);
        }
    }
    return 0.0;
}
fn lattice_density(index: vec3<i32>) -> f32 {
    // Signed integer division truncates toward zero, so derive Euclidean floor
    // from the exactly representable power-of-two division instead.
    let brick = vec3<i32>(floor(vec3<f32>(index) * 0.125));
    let local = vec3<u32>(index - brick * 8);
    let offset = find_brick(brick);
    if offset != 0xffffffffu {
        return density_values[offset + local.x * 64u + local.y * 8u + local.z];
    }
    return uniform_density(index);
}
fn density_index(q: vec3<f32>, pixel: u32) -> f32 {
    if !finite3(q) { fail(ERROR_NONFINITE, pixel); return -1.0; }
    if any(q < p.index_min.xyz) || any(q > p.index_max.xyz) { return 0.0; }
    let base = vec3<i32>(floor(q));
    let f = q - floor(q);
    let brick = vec3<i32>(floor(vec3<f32>(base) * 0.125));
    let local = vec3<u32>(base - brick * 8);
    var value: f32;
    if all(local < vec3<u32>(7u)) {
        // All eight corners share one brick: one hash lookup, eight values.
        // Aligned larger uniform tiles are also constant over a whole brick.
        let offset = find_brick(brick);
        if offset == 0xffffffffu {
            let constant = uniform_density(base);
            let z = mix(constant, constant, f.z);
            let y = mix(z, z, f.y);
            value = mix(y, y, f.x);
        } else {
            let i = offset + local.x * 64u + local.y * 8u + local.z;
            let z00 = mix(density_values[i], density_values[i + 1u], f.z);
            let z01 = mix(density_values[i + 8u], density_values[i + 9u], f.z);
            let z10 = mix(density_values[i + 64u], density_values[i + 65u], f.z);
            let z11 = mix(density_values[i + 72u], density_values[i + 73u], f.z);
            value = mix(mix(z00, z01, f.y), mix(z10, z11, f.y), f.x);
        }
    } else {
        let z00 = mix(lattice_density(base), lattice_density(base + vec3<i32>(0, 0, 1)), f.z);
        let z01 = mix(lattice_density(base + vec3<i32>(0, 1, 0)), lattice_density(base + vec3<i32>(0, 1, 1)), f.z);
        let z10 = mix(lattice_density(base + vec3<i32>(1, 0, 0)), lattice_density(base + vec3<i32>(1, 0, 1)), f.z);
        let z11 = mix(lattice_density(base + vec3<i32>(1, 1, 0)), lattice_density(base + vec3<i32>(1, 1, 1)), f.z);
        value = mix(mix(z00, z01, f.y), mix(z10, z11, f.y), f.x);
    }
    if !finite(value) { fail(ERROR_NONFINITE, pixel); return -1.0; }
    if value < 0.0 || value > p.optics.y { fail(ERROR_MAJORANT, pixel); return -1.0; }
    return value;
}
struct Ray { origin: vec3<f32>, direction: vec3<f32> }
// A double-float distance preserves small positive free flights when the high
// component's ULP is larger than the sampled step. Samples are never redrawn or
// rounded up to a minimum step. Density positions still use normal f32 arithmetic.
struct Distance { hi: f32, lo: f32 }
struct Interval { begin: Distance, end: Distance, hit: bool }
fn rounded_bits(value: f32) -> f32 {
    // storage.z is always zero, but remains a runtime uniform. Observing the
    // intermediate float bits prevents algebraic reassociation from erasing
    // TwoSum's rounding errors on shader compilers without a WGSL precise mode.
    return bitcast<f32>(bitcast<u32>(value) ^ p.storage.z);
}
fn advance_distance(old: Distance, step: f32) -> Distance {
    let sum = rounded_bits(old.hi + step);
    let recovered = rounded_bits(sum - old.hi);
    let removed = rounded_bits(sum - recovered);
    let lost_hi = rounded_bits(old.hi - removed);
    let lost_step = rounded_bits(step - recovered);
    let error = rounded_bits(lost_hi + lost_step);
    let tail = rounded_bits(old.lo + error);
    let hi = rounded_bits(sum + tail);
    let correction = rounded_bits(hi - sum);
    return Distance(hi, rounded_bits(tail - correction));
}
fn distance_progressed(old: Distance, next: Distance) -> bool {
    return next.hi > old.hi || (next.hi == old.hi && next.lo > old.lo);
}
fn distance_less(a: Distance, b: Distance) -> bool {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
fn distance_equal(a: Distance, b: Distance) -> bool { return a.hi == b.hi && a.lo == b.lo; }
fn two_sum(a: f32, b: f32) -> Distance {
    let hi = rounded_bits(a + b);
    let recovered = rounded_bits(hi - a);
    let lost_a = rounded_bits(a - rounded_bits(hi - recovered));
    let lost_b = rounded_bits(b - recovered);
    return Distance(hi, rounded_bits(lost_a + lost_b));
}
fn two_product(a: f32, b: f32) -> Distance {
    let hi = rounded_bits(a * b);
    return Distance(hi, rounded_bits(fma(a, b, -hi)));
}
fn distance_min(a: Distance, b: Distance) -> Distance {
    if distance_less(a, b) { return a; }
    return b;
}
fn distance_max(a: Distance, b: Distance) -> Distance {
    if distance_less(a, b) { return b; }
    return a;
}
fn finite_distance(value: Distance) -> bool { return finite(value.hi) && finite(value.lo); }
fn distance_add(a: Distance, b: Distance) -> Distance {
    let leading = two_sum(a.hi, b.hi);
    let trailing = two_sum(a.lo, b.lo);
    let tail = rounded_bits(leading.lo + trailing.hi);
    let merged = two_sum(leading.hi, tail);
    return two_sum(merged.hi, rounded_bits(merged.lo + trailing.lo));
}
fn distance_subtract(a: Distance, b: Distance) -> Distance {
    return distance_add(a, Distance(-b.hi, -b.lo));
}
fn distance_divide(numerator: Distance, denominator: f32) -> Distance {
    let quotient = rounded_bits(numerator.hi / denominator);
    // Nearly parallel axes can have no representable plane within the segment.
    if !finite(quotient) {
        // Preserve the sign for a plane behind the ray on a nearly parallel
        // axis. Saturating both signs to +MAX_FLOAT can create a false hit.
        return Distance(select(-MAX_FLOAT, MAX_FLOAT, quotient >= 0.0), 0.0);
    }
    let remainder = rounded_bits(fma(-quotient, denominator, numerator.hi));
    let tail = rounded_bits(rounded_bits(remainder + numerator.lo) / denominator);
    return two_sum(quotient, tail);
}
fn world_plane(index: f32, axis: u32) -> Distance {
    return distance_add(two_product(index, p.transform.w), Distance(p.transform[axis], 0.0));
}
fn cloud_interval(ray: Ray) -> Interval {
    var lo = Distance(0.0, 0.0);
    var hi = Distance(MAX_FLOAT, 0.0);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        // Expand only the proposal interval. Both exact affine plane products
        // are retained as pairs; the original density bounds remain unchanged.
        let minimum = world_plane(p.index_min[axis] - 1.0, axis);
        let maximum = world_plane(p.index_max[axis] + 1.0, axis);
        let origin = Distance(ray.origin[axis], 0.0);
        if ray.direction[axis] == 0.0 {
            if distance_less(origin, minimum) || distance_less(maximum, origin) {
                return Interval(lo, hi, false);
            }
        } else {
            let a = distance_divide(distance_subtract(minimum, origin), ray.direction[axis]);
            let b = distance_divide(distance_subtract(maximum, origin), ray.direction[axis]);
            lo = distance_max(lo, distance_min(a, b));
            hi = distance_min(hi, distance_max(a, b));
            if !distance_less(lo, hi) { return Interval(lo, hi, false); }
        }
    }
    return Interval(lo, hi, true);
}
fn index_at_distance(ray: Ray, distance: Distance) -> vec3<f32> {
    var result: vec3<f32>;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let displacement = distance_add(two_product(distance.hi, ray.direction[axis]),
            two_product(distance.lo, ray.direction[axis]));
        let world = distance_add(Distance(ray.origin[axis], 0.0), displacement);
        let relative = distance_subtract(world, Distance(p.transform[axis], 0.0));
        let index = distance_divide(relative, p.transform.w);
        result[axis] = rounded_bits(index.hi + index.lo);
    }
    return result;
}
fn ground_distance(ray: Ray) -> f32 {
    if (p.storage.w & 1u) != 0u && ray.direction.y != 0.0 {
        let t = (p.optics.w - ray.origin.y) / ray.direction.y;
        if t > 0.0 || (t == 0.0 && ray.direction.y < 0.0) { return t; }
    }
    return MAX_FLOAT;
}
struct Collision { position: vec3<f32>, kind: u32 } // 0 miss, 1 real, 2 error
struct TrackingResult { position: vec3<f32>, kind: u32, transmittance: f32 }
fn local_majorant(cell: vec3<i32>) -> f32 {
    let key = bitcast<vec3<u32>>(cell);
    var slot = (key.x * 73856093u ^ key.y * 19349663u ^ key.z * 83492791u) & p.storage.y;
    var majorant = 0.0;
    var found = false;
    for (var probe = 0u; probe <= p.storage.y; probe += 1u) {
        let entry = cell_majorant_hash[slot];
        if entry.w == 0xffffffffu { found = true; break; }
        if all(entry.xyz == key) { majorant = bitcast<f32>(entry.w); found = true; break; }
        slot = (slot + 1u) & p.storage.y;
    }
    if !found { fail(ERROR_LOOKUP, current_pixel_index); return -1.0; }
    // A missing leaf majorant must still cover nonzero large uniform tiles.
    let cell_min = vec3<f32>(cell) * 8.0 - vec3<f32>(1.0);
    let cell_max = cell_min + vec3<f32>(10.0);
    for (var i = 0u; i < p.control.w; i += 1u) {
        let tile = tiles[i];
        let origin = vec3<f32>(bitcast<vec3<i32>>(tile.origin_size.xyz));
        let end = origin + vec3<f32>(f32(tile.origin_size.w) - 1.0);
        if all(cell_min <= end) && all(cell_max >= origin) {
            majorant = max(majorant, bitcast<f32>(tile.density_pad.x));
        }
    }
    return majorant * (1.0 + 1.0 / 1048576.0);
}
struct Dda {
    cell: vec3<i32>, step: vec3<i32>,
    origin: vec3<f32>, direction: vec3<f32>, next: array<Distance, 3>,
    t: Distance,
}
fn start_dda(origin: vec3<f32>, direction: vec3<f32>) -> Dda {
    var dda: Dda;
    dda.origin = origin;
    dda.direction = direction;
    dda.cell = vec3<i32>(floor(origin * 0.125));
    dda.t = Distance(0.0, 0.0);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if direction[axis] == 0.0 {
            dda.step[axis] = 0;
            dda.next[axis] = Distance(MAX_FLOAT, 0.0);
        } else {
            dda.step[axis] = select(-1, 1, direction[axis] > 0.0);
            // An exact plane belongs to the cell in the propagation direction.
            if direction[axis] < 0.0 && origin[axis] == f32(dda.cell[axis] * 8) {
                dda.cell[axis] -= 1;
            }
            let boundary_cell = dda.cell[axis] + select(0, 1, direction[axis] > 0.0);
            let boundary = f32(boundary_cell * 8);
            dda.next[axis] = distance_divide(two_sum(boundary, -origin[axis]), direction[axis]);
        }
    }
    return dda;
}
fn dda_end(dda: Dda, span: Distance) -> Distance {
    var end = span;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if distance_less(dda.next[axis], end) { end = dda.next[axis]; }
    }
    return end;
}
fn advance_dda(dda: ptr<function, Dda>, end: Distance, pixel: u32) -> bool {
    var advanced = false;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if distance_equal((*dda).next[axis], end) && (*dda).step[axis] != 0 {
            (*dda).cell[axis] += (*dda).step[axis];
            let boundary_cell = (*dda).cell[axis] + select(0, 1, (*dda).step[axis] > 0);
            // Recompute from the integer plane rather than repeatedly adding a
            // rounded cell width. This preserves thin slivers at long distances.
            let numerator = two_sum(f32(boundary_cell * 8), -(*dda).origin[axis]);
            let next = distance_divide(numerator, (*dda).direction[axis]);
            if !distance_progressed((*dda).next[axis], next) {
                fail(ERROR_STAGNATION, pixel); return false;
            }
            (*dda).next[axis] = next;
            advanced = true;
        }
    }
    (*dda).t = end;
    if !advanced { fail(ERROR_STAGNATION, pixel); }
    return advanced;
}
// A tracking state survives dispatch boundaries, including the current
// Poisson segment and compensated distance. No random draw is discarded.
struct TrackingState {
    dda: Dda,
    entry: vec3<f32>, majorant: f32,
    direction: vec3<f32>, weight: f32,
    position: vec3<f32>, kind: u32,
    span: Distance, segment_end: Distance, t: Distance,
    shadow_candidates: u32, mode: u32, phase: u32,
}
struct PathState {
    origin: vec3<f32>, rng: u32,
    direction: vec3<f32>, stage: u32,
    beta: vec3<f32>, events: u32,
    radiance: vec3<f32>, bounces: u32,
    vertex: vec3<f32>, vertex_kind: u32,
    nee: vec3<f32>, ground_t: f32,
    tracking: TrackingState,
}
@group(0) @binding(8) var<storage, read_write> path_states: array<PathState>;
const TRACK_DONE: u32 = 0u;
const TRACK_CELL: u32 = 1u;
const TRACK_POISSON: u32 = 2u;
const TRACK_ADVANCE: u32 = 3u;
const PATH_INIT: u32 = 0u;
const PATH_START: u32 = 1u;
const PATH_DELTA: u32 = 2u;
const PATH_SHADOW: u32 = 3u;
const PATH_SCATTER: u32 = 4u;
const PATH_DONE: u32 = 5u;

fn begin_tracking(ray: Ray, end_distance: f32, mode: u32, pixel: u32) -> TrackingState {
    var state: TrackingState;
    state.weight = 1.0;
    state.mode = mode;
    if mode == 1u && ground_distance(ray) < MAX_FLOAT { state.weight = 0.0; return state; }
    if p.optics.x == 0.0 || p.optics.y == 0.0 { return state; }
    let scaled_world_origin = ray.origin / p.transform.w;
    if !finite3(scaled_world_origin) || any(abs(scaled_world_origin) > vec3<f32>(MAX_RAY_INDEX_MAGNITUDE)) {
        fail(ERROR_PRECISION, pixel); state.kind = 2u; return state;
    }
    let interval = cloud_interval(ray);
    let clipped_end = distance_min(interval.end, Distance(end_distance, 0.0));
    if !interval.hit || !distance_less(interval.begin, clipped_end) { return state; }
    if !finite_distance(interval.begin) || !finite_distance(clipped_end) {
        fail(ERROR_NONFINITE, pixel); state.kind = 2u; return state;
    }
    state.entry = index_at_distance(ray, interval.begin);
    state.direction = ray.direction / p.transform.w;
    if !finite3(state.entry) || !finite3(state.direction) {
        fail(ERROR_NONFINITE, pixel); state.kind = 2u; return state;
    }
    state.span = distance_subtract(clipped_end, interval.begin);
    if (p.storage.w & 2u) != 0u {
        state.majorant = p.optics.y;
        state.segment_end = state.span;
        state.phase = TRACK_POISSON;
    } else {
        state.dda = start_dda(state.entry, state.direction);
        state.phase = TRACK_CELL;
    }
    return state;
}
fn tracking_error(state: ptr<function, TrackingState>) {
    (*state).kind = 2u; (*state).phase = TRACK_DONE;
}
// Exactly one bounded transition: at most one candidate or one DDA boundary.
fn tracking_step(state: ptr<function, TrackingState>, rng: ptr<function, u32>,
    events: ptr<function, u32>, pixel: u32) {
    if (*state).phase == TRACK_CELL {
        if !event(events, pixel) { tracking_error(state); return; }
        (*state).segment_end = dda_end((*state).dda, (*state).span);
        (*state).t = (*state).dda.t;
        if distance_less((*state).segment_end, (*state).t) {
            fail(ERROR_STAGNATION, pixel); tracking_error(state); return;
        }
        (*state).majorant = local_majorant((*state).dda.cell);
        (*state).phase = TRACK_POISSON;
        if (*state).majorant == 0.0 || distance_equal((*state).t, (*state).segment_end) {
            (*state).phase = TRACK_ADVANCE;
        }
        return;
    }
    if (*state).phase == TRACK_ADVANCE {
        if (*state).weight == 0.0 || distance_equal((*state).segment_end, (*state).span) {
            (*state).phase = TRACK_DONE; return;
        }
        // Local arguments avoid Naga29's SPIR-V caching bug for a nested
        // function-pointer member passed to another pointer-taking function.
        var dda = (*state).dda;
        let advanced = advance_dda(&dda, (*state).segment_end, pixel);
        (*state).dda = dda;
        if !advanced { tracking_error(state); return; }
        (*state).phase = TRACK_CELL; return;
    }
    if (*state).phase != TRACK_POISSON { return; }
    let rate = p.optics.x * (*state).majorant;
    if !finite(rate) || rate <= 0.0 {
        fail(ERROR_NONFINITE, pixel); tracking_error(state); return;
    }
    if !event(events, pixel) { tracking_error(state); return; }
    let step = -log(open01(rng)) / rate;
    let remaining = distance_subtract((*state).segment_end, (*state).t);
    if step > remaining.hi || (step == remaining.hi && remaining.lo <= 0.0) {
        (*state).phase = TRACK_ADVANCE; return;
    }
    let next = advance_distance((*state).t, step);
    if !finite_distance(next) { fail(ERROR_NONFINITE, pixel); tracking_error(state); return; }
    if !distance_less(next, (*state).segment_end) { (*state).phase = TRACK_ADVANCE; return; }
    if !distance_progressed((*state).t, next) { fail(ERROR_STAGNATION, pixel); tracking_error(state); return; }
    (*state).t = next;
    let point = ((*state).entry + next.hi * (*state).direction) + next.lo * (*state).direction;
    let rho = density_index(point, pixel);
    if rho < 0.0 { tracking_error(state); return; }
    if rho > (*state).majorant { fail(ERROR_MAJORANT, pixel); tracking_error(state); return; }
    if (*state).mode == 0u {
        if open01(rng) < rho / (*state).majorant {
            (*state).position = point * p.transform.w + p.transform.xyz;
            (*state).kind = 1u; (*state).phase = TRACK_DONE;
        }
    } else {
        (*state).weight *= 1.0 - rho / (*state).majorant;
        if !finite((*state).weight) { fail(ERROR_NONFINITE, pixel); tracking_error(state); return; }
        if (*state).weight == 0.0 { (*state).phase = TRACK_DONE; return; }
        (*state).shadow_candidates += 1u;
        if (p.storage.w & 4u) != 0u && ((*state).shadow_candidates & 15u) == 0u && (*state).weight < 1e-4 {
            if open01(rng) < 0.5 { (*state).weight = 0.0; (*state).phase = TRACK_DONE; return; }
            (*state).weight *= 2.0;
        }
    }
}
fn hg_phase(cosine: f32) -> f32 {
    let g = p.optics.z;
    var denominator = (1.0 - g) * (1.0 - g) + 2.0 * g * (1.0 - cosine);
    if g < 0.0 { denominator = (1.0 + g) * (1.0 + g) - 2.0 * g * (1.0 + cosine); }
    return ((1.0 - g) * (1.0 + g)) / (4.0 * PI * denominator * sqrt(denominator));
}
fn tangent(normal: vec3<f32>) -> vec3<f32> {
    var helper = vec3<f32>(0.0, 0.0, 1.0);
    if abs(normal.z) >= 0.999 { helper = vec3<f32>(1.0, 0.0, 0.0); }
    return normalize(cross(helper, normal));
}
fn sample_hg(direction: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let g = p.optics.z;
    let u = open01(rng);
    let a = 2.0 * u - 1.0;
    let numerator = a + 0.5 * g * (a * a + 3.0) + g * g * a
        + 0.5 * g * g * g * (a * a - 1.0);
    let denominator = 1.0 + g * a;
    var cosine = numerator / (denominator * denominator);
    if abs(g) >= 0.01 {
        var d = (1.0 - g) + 2.0 * g * u;
        if g < 0.0 { d = (1.0 + g) - 2.0 * g * (1.0 - u); }
        let q = ((1.0 - g) * (1.0 + g)) / d;
        cosine = (1.0 + g * g - q * q) / (2.0 * g);
    }
    cosine = clamp(cosine, -1.0, 1.0);
    let sine = sqrt(max(0.0, 1.0 - cosine * cosine));
    let azimuth = 2.0 * PI * open01(rng);
    let t = tangent(direction);
    return normalize(direction * cosine + (t * cos(azimuth) + cross(direction, t) * sin(azimuth)) * sine);
}
fn sample_ground(rng: ptr<function, u32>) -> vec3<f32> {
    let squared_radius = open01(rng);
    let radius = sqrt(squared_radius);
    let azimuth = 2.0 * PI * open01(rng);
    return vec3<f32>(radius * cos(azimuth), sqrt(1.0 - squared_radius), radius * sin(azimuth));
}
// One path transition never contains a variable-length transport loop.
fn resume_path_tracking(state: ptr<function, PathState>, pixel: u32) {
    var tracking = (*state).tracking;
    var rng = (*state).rng;
    var events = (*state).events;
    tracking_step(&tracking, &rng, &events, pixel);
    (*state).tracking = tracking;
    (*state).rng = rng;
    (*state).events = events;
}
fn valid_boundary_rgb(value: vec3<f32>) -> bool {
    return finite3(value) && all(value >= vec3<f32>(0.0));
}
fn boundary_radiance(direction: vec3<f32>, pixel: u32) -> vec3<f32> {
    if (p.storage.w & 8u) == 0u { return p.sky_radiance.xyz; }
    let size = vec2<i32>(textureDimensions(environment));
    var phi = atan2(direction.x, direction.z);
    if phi < 0.0 { phi += 2.0 * PI; }
    let uv = vec2<f32>(phi / (2.0 * PI), acos(clamp(direction.y, -1.0, 1.0)) / PI);
    let xy = uv * vec2<f32>(size) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(xy));
    let f = fract(xy);
    let x0 = ((base.x % size.x) + size.x) % size.x;
    let x1 = (x0 + 1) % size.x;
    let y0 = clamp(base.y, 0, size.y - 1);
    let y1 = clamp(base.y + 1, 0, size.y - 1);
    let a = textureLoad(environment, vec2<i32>(x0, y0), 0).xyz;
    let b = textureLoad(environment, vec2<i32>(x1, y0), 0).xyz;
    let c = textureLoad(environment, vec2<i32>(x0, y1), 0).xyz;
    let d = textureLoad(environment, vec2<i32>(x1, y1), 0).xyz;
    if !valid_boundary_rgb(a) || !valid_boundary_rgb(b) || !valid_boundary_rgb(c) || !valid_boundary_rgb(d) {
        fail(ERROR_ENVIRONMENT, pixel); return vec3<f32>(0.0);
    }
    // Manual interpolation works with unfilterable RGBA32Float textures.
    // Difference-form interpolation preserves a constant boundary exactly.
    let row0 = a + (b - a) * f.x;
    let row1 = c + (d - c) * f.x;
    return row0 + (row1 - row0) * f.y;
}
fn direct_sun_irradiance(pixel: u32) -> vec3<f32> {
    if (p.storage.w & 8u) == 0u { return p.sun_irradiance.xyz; }
    let value = textureLoad(sunlight, vec2<i32>(0), 0).xyz;
    if !valid_boundary_rgb(value) {
        fail(ERROR_ENVIRONMENT, pixel); return vec3<f32>(0.0);
    }
    return value;
}
fn path_step(state: ptr<function, PathState>, pixel: u32) {
    if (*state).stage == PATH_INIT {
        var rng = mix_bits(p.control.x ^ mix_bits(p.control.y) ^ mix_bits(pixel) ^ mix_bits(current_sample_index + 0x9e3779b9u));
        let id = vec2<u32>(pixel % p.image.x, pixel / p.image.x);
        let xy = (vec2<f32>(id) + vec2<f32>(open01(&rng), open01(&rng))) / vec2<f32>(p.image.xy);
        let aspect = f32(p.image.y) / f32(p.image.x);
        (*state).origin = p.camera_origin.xyz;
        (*state).direction = normalize(p.camera_forward.xyz
            + p.camera_right.xyz * ((2.0 * xy.x - 1.0) * p.camera_origin.w)
            + p.camera_up.xyz * ((1.0 - 2.0 * xy.y) * p.camera_origin.w * aspect));
        (*state).rng = rng;
        (*state).beta = vec3<f32>(1.0);
        (*state).stage = PATH_START; return;
    }
    if (*state).stage == PATH_START {
        if !finite3((*state).origin) || !finite3((*state).direction) || !finite3((*state).beta) || !finite3((*state).radiance) {
            fail(ERROR_NONFINITE, pixel); return;
        }
        let ray = Ray((*state).origin, (*state).direction);
        (*state).ground_t = ground_distance(ray);
        (*state).tracking = begin_tracking(ray, (*state).ground_t, 0u, pixel);
        (*state).stage = PATH_DELTA; return;
    }
    if (*state).stage == PATH_DELTA {
        if (*state).tracking.phase != TRACK_DONE {
            resume_path_tracking(state, pixel);
            return;
        }
        if (*state).tracking.kind == 2u { return; }
        var direct: f32;
        if (*state).tracking.kind == 1u {
            (*state).beta *= p.albedo.xyz;
            (*state).vertex = (*state).tracking.position;
            (*state).vertex_kind = 1u;
            direct = hg_phase(clamp(dot((*state).direction, p.sun_direction.xyz), -1.0, 1.0));
        } else if (*state).ground_t < MAX_FLOAT {
            var events = (*state).events;
            let valid = event(&events, pixel);
            (*state).events = events;
            if !valid { return; }
            if (*state).direction.y >= 0.0 { (*state).stage = PATH_DONE; return; }
            (*state).vertex = (*state).origin + (*state).ground_t * (*state).direction;
            (*state).vertex.y = p.optics.w;
            (*state).beta *= p.ground_albedo.xyz;
            (*state).vertex_kind = 0u;
            direct = max(0.0, p.sun_direction.y) / PI;
        } else {
            (*state).radiance += (*state).beta * boundary_radiance((*state).direction, pixel);
            if !finite3((*state).radiance) { fail(ERROR_NONFINITE, pixel); return; }
            (*state).stage = PATH_DONE; return;
        }
        if all((*state).beta == vec3<f32>(0.0)) { (*state).stage = PATH_DONE; return; }
        // Preserve the original RNG schedule even if a tiny positive cosine
        // makes cosine/PI round to zero in f32.
        let sun = direct_sun_irradiance(pixel);
        if any(sun > vec3<f32>(0.0))
            && ((*state).vertex_kind == 1u || p.sun_direction.y > 0.0) {
            (*state).nee = (*state).beta * sun * direct;
            (*state).tracking = begin_tracking(Ray((*state).vertex, p.sun_direction.xyz), MAX_FLOAT, 1u, pixel);
            (*state).stage = PATH_SHADOW;
        } else { (*state).stage = PATH_SCATTER; }
        return;
    }
    if (*state).stage == PATH_SHADOW {
        if (*state).tracking.phase != TRACK_DONE {
            resume_path_tracking(state, pixel);
            return;
        }
        if (*state).tracking.kind == 2u { return; }
        (*state).radiance += (*state).nee * (*state).tracking.weight;
        (*state).stage = PATH_SCATTER; return;
    }
    if (*state).stage == PATH_SCATTER {
        var rng = (*state).rng;
        if (*state).vertex_kind == 1u { (*state).direction = sample_hg((*state).direction, &rng); }
        else { (*state).direction = sample_ground(&rng); }
        (*state).origin = (*state).vertex;
        (*state).bounces += 1u;
        if (*state).bounces >= p.image.w {
            let survival = min(1.0, max((*state).beta.x, max((*state).beta.y, (*state).beta.z)));
            if survival <= 0.0 || open01(&rng) >= survival { (*state).rng = rng; (*state).stage = PATH_DONE; return; }
            (*state).beta /= survival;
        }
        (*state).rng = rng;
        (*state).stage = PATH_START;
    }
}
@compute @workgroup_size(64, 1, 1)
fn trace_work(@builtin(global_invocation_id) id: vec3<u32>) {
    let slots = p.batch.x * p.batch.z;
    if id.x >= slots || atomicLoad(&diagnostics.flags) != 0u { return; }
    let preview = (p.storage.w & 16u) != 0u;
    let path_slot = p.batch.y + id.x;
    var pixel = p.batch.y + id.x % p.batch.z;
    if preview { pixel = (path_slot * u32(p.index_min.w)) % (p.image.x * p.image.y); }
    current_pixel_index = pixel;
    current_sample_index = p.image.z + id.x / p.batch.z;
    let path_index = select(id.x, path_slot, preview);
    var state = path_states[path_index];
    let already_done = state.stage == PATH_DONE;
    // Fixed finite work. A long legal path remains pending, never truncated.
    for (var transition = 0u; transition < p.batch.w; transition += 1u) {
        if state.stage == PATH_DONE || atomicLoad(&diagnostics.flags) != 0u { break; }
        path_step(&state, pixel);
    }
    path_states[path_index] = state;
    if state.stage == PATH_DONE && (!preview || !already_done) { atomicAdd(&diagnostics.completed, 1u); }
}
@compute @workgroup_size(64, 1, 1)
fn reduce_work(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.batch.z || atomicLoad(&diagnostics.flags) != 0u { return; }
    let preview = (p.storage.w & 16u) != 0u;
    let path_slot = p.batch.y + id.x;
    var pixel = path_slot;
    if preview { pixel = (path_slot * u32(p.index_min.w)) % (p.image.x * p.image.y); }
    // Display completed pixels promptly, without waiting for another pixel's
    // long path. The persisted count deduplicates samples across dispatches.
    // A later finished sample waits for every preceding sample of this pixel.
    let count = u32(film_mean[pixel].w);
    if count < p.image.z || count > p.image.z + p.batch.x { fail(ERROR_NONFINITE, pixel); return; }
    for (var z = count - p.image.z; z < p.batch.x; z += 1u) {
        current_sample_index = p.image.z + z;
        let path_index = select(z * p.batch.z + id.x, path_slot, preview);
        let sample = path_states[path_index];
        if sample.stage != PATH_DONE { break; }
        let previous_mean = film_mean[pixel].xyz;
        let delta = sample.radiance - previous_mean;
        let mean = previous_mean + delta / f32(current_sample_index + 1u);
        let m2 = film_m2[pixel].xyz + delta * (sample.radiance - mean);
        if !finite3(mean) || !finite3(m2) { fail(ERROR_NONFINITE, pixel); return; }
        film_mean[pixel] = vec4<f32>(mean, f32(current_sample_index + 1u));
        film_m2[pixel] = vec4<f32>(m2, 0.0);
    }
}
