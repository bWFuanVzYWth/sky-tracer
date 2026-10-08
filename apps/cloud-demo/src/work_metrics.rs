//! CPU scheduling and bounded timing history. GPU execution and host waiting
//! are measured separately; callback latency is never multiplied into rest.
use cloud_pt::Result;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

const HISTORY: usize = 4096;
const GROUP_TARGET_MS: f64 = 12.0;

#[derive(Default)]
pub struct WorkScheduler {
    recent_chunks: VecDeque<f64>,
    actual: u32,
    last_span_ms: f64,
}
impl WorkScheduler {
    pub fn reset(&mut self) {
        self.recent_chunks.clear();
        self.actual = 1;
        self.last_span_ms = 0.0;
    }
    pub fn completed(&mut self, gpu: Option<&GpuTimes>) {
        let Some(gpu) = gpu else {
            self.reset();
            return;
        };
        self.last_span_ms = gpu.group_ms;
        for &ms in &gpu.chunks_ms {
            if self.recent_chunks.len() == 8 {
                self.recent_chunks.pop_front();
            }
            self.recent_chunks.push_back(ms);
        }
    }
    /// Start with one chunk; grow by at most one after a measured group.
    /// A slow group or recent slow chunk lowers the cap immediately.
    pub fn next(&mut self, maximum: u32) -> u32 {
        let maximum = maximum.clamp(1, 8);
        let cap = self
            .recent_chunks
            .iter()
            .copied()
            .reduce(f64::max)
            .map_or(1, |ms| {
                ((GROUP_TARGET_MS / ms.max(0.001)).floor() as u32).clamp(1, maximum)
            });
        let mut desired = cap.min(self.actual.max(1).saturating_add(1));
        if self.last_span_ms > GROUP_TARGET_MS {
            desired = desired.min(self.actual.saturating_sub(1).max(1));
        }
        self.actual = desired;
        desired
    }
    pub fn actual(&self) -> u32 {
        self.actual.max(1)
    }
}

#[derive(Debug)]
pub struct GpuTimes {
    pub group_ms: f64,
    pub chunks_ms: Vec<f64>,
}
pub fn decode_gpu_times(bytes: &[u8], period_ns: f32, chunks: u32) -> Result<GpuTimes> {
    if !(1..=8).contains(&chunks)
        || bytes.len() != (chunks as usize + 1) * 16
        || !period_ns.is_finite()
        || period_ns <= 0.0
    {
        return Err("invalid asynchronous cloud timing readback".into());
    }
    let pair = |at: usize| -> Result<(u64, u64)> {
        let start = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        let end = u64::from_le_bytes(bytes[at + 8..at + 16].try_into().unwrap());
        if end < start {
            return Err("cloud GPU timestamp ordering is invalid".into());
        }
        Ok((start, end))
    };
    let (start, end) = pair(0)?;
    let ms = |ticks: u64| ticks as f64 * f64::from(period_ns) * 1e-6;
    let mut previous = start;
    let mut chunks_ms = Vec::with_capacity(chunks as usize);
    for chunk in 0..chunks as usize {
        let (a, b) = pair(16 + chunk * 16)?;
        if a < previous
            || b > end
            || (chunk == 0 && a != start)
            || (chunk + 1 == chunks as usize && b != end)
        {
            return Err("cloud chunk timestamps fall outside their group".into());
        }
        previous = b;
        chunks_ms.push(ms(b - a));
    }
    Ok(GpuTimes {
        group_ms: ms(end - start),
        chunks_ms,
    })
}

/// The wall interval already includes time spent outside the timed GPU group.
/// Only any remaining budget gap is scheduled, so slow map/CPU waiting cannot
/// throttle another work group. Unsupported timestamps add no artificial rest.
pub fn budget_rest(gpu_ms: Option<f64>, wall: Duration, budget_percent: u32) -> Duration {
    let Some(gpu_ms) = gpu_ms.filter(|n| n.is_finite() && *n >= 0.0) else {
        return Duration::ZERO;
    };
    let budget = f64::from(budget_percent.clamp(10, 100)) / 100.0;
    let gap_ms = gpu_ms * (1.0 / budget - 1.0);
    let already_waited_ms = (wall.as_secs_f64() * 1000.0 - gpu_ms).max(0.0);
    Duration::from_secs_f64(((gap_ms - already_waited_ms).max(0.0) * 0.001).min(5.0))
}
fn push(history: &mut VecDeque<f64>, value: f64) {
    if history.len() == HISTORY {
        history.pop_front();
    }
    history.push_back(value);
}
fn quantile(history: &VecDeque<f64>, fraction: f64) -> Option<f64> {
    if history.is_empty() {
        return None;
    }
    let mut sorted: Vec<_> = history.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * fraction;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    Some(sorted[low] + (sorted[high] - sorted[low]) * position.fract())
}

#[derive(Default)]
pub struct Performance {
    pub started: Option<Instant>,
    pub groups: u64,
    pub chunks: u64,
    pub timestamps_supported: bool,
    pub gpu_group_ms: f64,
    pub gpu_chunk_ms: f64,
    pub gpu_inter_chunk_gap_ms: f64,
    pub gpu_group_max_ms: f64,
    pub gpu_chunk_max_ms: f64,
    pub wall_ms: f64,
    pub wall_max_ms: f64,
    pub wait_ms: f64,
    pub encode_ms: f64,
    pub planned_throttle_ms: f64,
    pub observed_throttle_ms: f64,
    actual_groups: [u64; 8],
    gpu_groups: VecDeque<f64>,
    gpu_chunks: VecDeque<f64>,
    walls: VecDeque<f64>,
    throttle: Option<(Instant, Duration)>,
}
impl Performance {
    pub fn new(timestamps_supported: bool) -> Self {
        Self {
            timestamps_supported,
            ..Self::default()
        }
    }
    pub fn submitted(&mut self, now: Instant, encode_ms: f64) {
        self.started.get_or_insert(now);
        self.encode_ms += encode_ms;
    }
    pub fn completed(
        &mut self,
        now: Instant,
        wall: Duration,
        gpu: Option<&GpuTimes>,
        chunks: u32,
        budget: u32,
    ) -> Duration {
        self.groups += 1;
        self.actual_groups[chunks as usize - 1] += 1;
        self.chunks += u64::from(chunks);
        let wall_ms = wall.as_secs_f64() * 1000.0;
        self.wall_ms += wall_ms;
        self.wall_max_ms = self.wall_max_ms.max(wall_ms);
        push(&mut self.walls, wall_ms);
        if let Some(gpu) = gpu {
            self.timestamps_supported = true;
            let sum: f64 = gpu.chunks_ms.iter().sum();
            self.gpu_group_ms += gpu.group_ms;
            self.gpu_chunk_ms += sum;
            self.gpu_inter_chunk_gap_ms += (gpu.group_ms - sum).max(0.0);
            self.gpu_group_max_ms = self.gpu_group_max_ms.max(gpu.group_ms);
            self.wait_ms += (wall_ms - gpu.group_ms).max(0.0);
            push(&mut self.gpu_groups, gpu.group_ms);
            for &ms in &gpu.chunks_ms {
                self.gpu_chunk_max_ms = self.gpu_chunk_max_ms.max(ms);
                push(&mut self.gpu_chunks, ms);
            }
        }
        let rest = budget_rest(gpu.map(|g| g.group_ms), wall, budget);
        self.planned_throttle_ms += rest.as_secs_f64() * 1000.0;
        self.settle_throttle(now);
        self.throttle = (!rest.is_zero()).then_some((now, rest));
        rest
    }
    /// Count the requested wall delay only up to its deadline. Presentation or
    /// pausing after that deadline is not attributed to this budget throttle.
    pub fn settle_throttle(&mut self, now: Instant) {
        if let Some((begin, duration)) = self.throttle.take() {
            self.observed_throttle_ms += now
                .saturating_duration_since(begin)
                .min(duration)
                .as_secs_f64()
                * 1000.0;
        }
    }
    pub fn window_ms(&self, now: Instant) -> f64 {
        self.started.map_or(0.0, |s| {
            now.saturating_duration_since(s).as_secs_f64() * 1000.0
        })
    }
    pub fn duty_percent(&self, now: Instant) -> Option<f64> {
        (self.timestamps_supported && self.window_ms(now) > 0.0)
            .then(|| 100.0 * self.gpu_chunk_ms / self.window_ms(now))
    }
    pub fn json(&self, now: Instant) -> serde_json::Value {
        let timed = |n: f64| self.timestamps_supported.then_some(n);
        serde_json::json!({
            "compute_window_ms":self.window_ms(now),"work_groups":self.groups,"work_dispatches":self.chunks,
            "work_dispatches_kind":"trace dispatches; each also has one reduction dispatch",
            "actual_group_dispatches_histogram":{"1":self.actual_groups[0],"2":self.actual_groups[1],"3":self.actual_groups[2],"4":self.actual_groups[3],"5":self.actual_groups[4],"6":self.actual_groups[5],"7":self.actual_groups[6],"8":self.actual_groups[7]},
            "actual_group_dispatches_mean":(self.groups>0).then(||self.chunks as f64/self.groups as f64),
            "group_policy":"start every new cohort/scene at 1; recent maximum of 8 chunks bounds group near 12ms; grow by at most 1; unsupported timestamps stay at 1",
            "timestamps_supported":self.timestamps_supported,"cloud_gpu_ms":timed(self.gpu_chunk_ms),
            "gpu_group_total_ms":timed(self.gpu_group_ms),"gpu_inter_chunk_gap_ms":timed(self.gpu_inter_chunk_gap_ms),
            "gpu_group_mean_ms":(self.timestamps_supported && self.groups>0).then(||self.gpu_group_ms/self.groups as f64),
            "gpu_chunk_mean_ms":(self.timestamps_supported && self.chunks>0).then(||self.gpu_chunk_ms/self.chunks as f64),
            "cloud_gpu_duty_percent":self.duty_percent(now),
            "gpu_group_p50_ms":quantile(&self.gpu_groups,0.5),"gpu_group_p95_ms":quantile(&self.gpu_groups,0.95),"gpu_group_max_ms":timed(self.gpu_group_max_ms),
            "gpu_chunk_p50_ms":quantile(&self.gpu_chunks,0.5),"gpu_chunk_p95_ms":quantile(&self.gpu_chunks,0.95),"gpu_chunk_max_ms":timed(self.gpu_chunk_max_ms),
            "submit_to_map_wall_sum_ms":self.wall_ms,"submit_to_map_wall_p50_ms":quantile(&self.walls,0.5),"submit_to_map_wall_p95_ms":quantile(&self.walls,0.95),"submit_to_map_wall_max_ms":self.wall_max_ms,
            "submit_to_map_wall_mean_ms":(self.groups>0).then(||self.wall_ms/self.groups as f64),
            "non_gpu_completion_wait_ms":timed(self.wait_ms),"encode_cpu_ms":self.encode_ms,
            "planned_active_idle_ms":self.planned_throttle_ms,"observed_budget_idle_ms":self.observed_throttle_ms,
            "quantile_scope":"most recent 4096 completed groups/chunks; total and max cover the run",
            "quantile_method":"linear interpolation between empirical order statistics",
            "quantile_group_samples":self.walls.len(),"quantile_chunk_samples":self.gpu_chunks.len(),
            "duty_scope":"completed cloud trace+reduce chunks only; excludes presentation and atmosphere update; compute-window wall includes CPU waiting and idle",
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scheduler_starts_small_grows_gradually_and_caps_with_slow_recent_work() {
        let mut s = WorkScheduler::default();
        assert_eq!(s.next(4), 1);
        s.completed(Some(&GpuTimes {
            group_ms: 12.0,
            chunks_ms: vec![12.0],
        }));
        assert_eq!(s.next(4), 1);
        for _ in 0..8 {
            s.completed(Some(&GpuTimes {
                group_ms: 0.1,
                chunks_ms: vec![0.1],
            }));
        }
        assert_eq!(s.next(4), 2);
        s.completed(Some(&GpuTimes {
            group_ms: 0.4,
            chunks_ms: vec![0.2, 0.2],
        }));
        assert_eq!(s.next(4), 3);
        s.completed(Some(&GpuTimes {
            group_ms: 45.0,
            chunks_ms: vec![15.0; 3],
        }));
        assert_eq!(s.next(4), 1);
        s.completed(None);
        assert_eq!(s.next(8), 1);
        s.reset();
        assert_eq!(s.next(8), 1);
        for _ in 0..8 {
            s.completed(Some(&GpuTimes {
                group_ms: 0.1,
                chunks_ms: vec![0.1],
            }));
        }
        assert_eq!(s.next(1), 1);
    }
    #[test]
    fn delayed_readback_never_creates_more_throttle() {
        assert_eq!(
            budget_rest(Some(8.0), Duration::from_millis(8), 80),
            Duration::from_millis(2)
        );
        assert_eq!(
            budget_rest(Some(8.0), Duration::from_millis(9), 80),
            Duration::from_millis(1)
        );
        assert_eq!(
            budget_rest(Some(8.0), Duration::from_millis(50), 80),
            Duration::ZERO
        );
        assert_eq!(
            budget_rest(Some(8.0), Duration::from_millis(8), 100),
            Duration::ZERO
        );
        assert_eq!(
            budget_rest(None, Duration::from_millis(500), 80),
            Duration::ZERO
        );
    }
    #[test]
    fn gpu_clock_unpack_checks_order_and_scope() {
        let words = [10u64, 90, 10, 40, 50, 90];
        let bytes: Vec<_> = words.into_iter().flat_map(u64::to_le_bytes).collect();
        let g = decode_gpu_times(&bytes, 1000.0, 2).unwrap();
        assert!((g.group_ms - 0.08).abs() < 1e-12);
        assert_eq!(g.chunks_ms, vec![0.03, 0.04]);
        assert!(decode_gpu_times(&bytes, 1000.0, 1).is_err());
        assert!(decode_gpu_times(&bytes, f32::NAN, 2).is_err());
        let bad: Vec<_> = [10u64, 90, 10, 40, 30, 90]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        assert!(decode_gpu_times(&bad, 1.0, 2).is_err());
        let bad_endpoint: Vec<_> = [10u64, 100, 10, 40, 50, 90]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        assert!(decode_gpu_times(&bad_endpoint, 1.0, 2).is_err());
    }
    #[test]
    fn timing_quantiles_report_fractional_order_statistics() {
        let data = VecDeque::from([4.0, 1.0, 3.0, 2.0]);
        assert_eq!(quantile(&data, 0.5), Some(2.5));
        assert!((quantile(&data, 0.95).unwrap() - 3.85).abs() < 1e-12);
        assert_eq!(quantile(&VecDeque::new(), 0.5), None);
    }
    #[test]
    fn metrics_keep_totals_and_bound_history_without_double_counting_idle() {
        let begin = Instant::now();
        let mut p = Performance::default();
        p.submitted(begin, 1.0);
        let g = GpuTimes {
            group_ms: 8.0,
            chunks_ms: vec![3.0, 4.0],
        };
        let now = begin + Duration::from_millis(8);
        assert_eq!(
            p.completed(now, Duration::from_millis(8), Some(&g), 2, 80),
            Duration::from_millis(2)
        );
        p.settle_throttle(now + Duration::from_millis(1));
        p.settle_throttle(now + Duration::from_secs(1));
        assert_eq!(p.observed_throttle_ms, 1.0);
        assert_eq!(p.gpu_chunk_ms, 7.0);
        assert_eq!(p.gpu_inter_chunk_gap_ms, 1.0);
        let json = p.json(now);
        assert_eq!(json["gpu_group_mean_ms"], 8.0);
        assert_eq!(json["gpu_chunk_mean_ms"], 3.5);
        assert_eq!(json["submit_to_map_wall_mean_ms"], 8.0);
        assert!((p.duty_percent(begin + Duration::from_millis(100)).unwrap() - 7.0).abs() < 1e-12);
        for _ in 0..HISTORY + 1 {
            p.completed(now, Duration::from_millis(8), Some(&g), 2, 100);
        }
        assert_eq!(p.walls.len(), HISTORY);
        assert_eq!(p.gpu_chunks.len(), HISTORY);
        assert_eq!(p.groups, HISTORY as u64 + 2);
        let untimed = Performance::default().json(now);
        assert!(untimed["cloud_gpu_ms"].is_null());
    }
}
