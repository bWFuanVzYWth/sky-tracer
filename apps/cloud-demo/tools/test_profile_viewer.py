"""Profiler accounting fixtures; no subprocess window or GPU is created."""
import json
import unittest
from pathlib import Path
import tempfile
import profile_viewer as profile


class Accounting(unittest.TestCase):
    def fixture(self):
        return dict(schema_version=1, compute_window_ms=10_000, viewer_elapsed_ms=10_600,
                    cloud_gpu_ms=6400, timestamps_supported=True, film_width=128, film_height=72,
                    completed_spp=4, current_batch_completed_paths=9000, current_batch_finished=False,
                    work_groups=100, work_dispatches=400, presents=260)

    def test_parse_actual_prefix_and_completed_paths(self):
        m = self.fixture()
        got = profile.derived(profile.parse('startup\n'+profile.PREFIX+json.dumps(m)+'\n'))
        self.assertEqual(got['cloud_gpu_duty_percent'], 64)
        self.assertEqual(got['completed_sample_paths_per_second'], (4*128*72+9000)/10)
        self.assertEqual(got['dispatches_per_group'], 4)
        self.assertAlmostEqual(got['presents_per_second'], 260/10.6)

    def test_completed_batch_is_not_counted_twice(self):
        m = self.fixture()
        m['current_batch_finished'] = True
        m['current_batch_completed_paths'] = 4*128*72
        self.assertEqual(profile.derived(m)['completed_sample_paths_per_second'], 4*128*72/10)
        m['complete_sample_paths'] = 123
        self.assertEqual(profile.derived(m)['completed_sample_paths_per_second'], 12.3)

    def test_unavailable_timestamps_and_progress_are_not_zero(self):
        m = self.fixture()
        m['timestamps_supported'] = False
        del m['current_batch_finished']
        r = profile.derived(m)
        self.assertIsNone(r['cloud_gpu_duty_percent'])
        self.assertIsNone(r['completed_sample_paths_per_second'])
        for log in ['no metrics', profile.PREFIX+'{}', profile.PREFIX+json.dumps(m)+'\n'+profile.PREFIX+json.dumps(m)]:
            with self.assertRaises(ValueError):
                profile.parse(log)

    def test_legacy_stdout_does_not_invent_gpu_or_partial_counts(self):
        m = profile.legacy_metrics('Cloud diagnostic duration reached: 295 completed displays / '
                                  '442 completed bounded work chunks / 0 complete spp; no reference exported.')
        self.assertEqual(m['work_dispatches'], 442)
        self.assertEqual(m['presents'], 295)
        self.assertIsNone(m['partial_progress'])
        self.assertIsNone(profile.derived(m)['cloud_gpu_duty_percent'])
        self.assertIsNone(profile.derived(m)['completed_sample_paths_per_second'])
        with self.assertRaises(ValueError):
            profile.legacy_metrics('error only')

    def test_nvml_samples_are_activity_only_and_ignore_na(self):
        folder=Path(__file__).resolve().parents[3]/'target/test-tmp/cloud-profiler'
        folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as directory:
            path=Path(directory)/'monitor.csv'
            path.write_text('2026/10/08 12:00:00, 20, 8, 180\n2026/10/08 12:00:00.200, 80, 9, 240\nN/A, N/A, N/A, N/A\n')
            got=profile.nvml_summary(path)
            self.assertEqual(got['samples'],2)
            self.assertEqual(got['gpu_activity_percent_mean'],50)
            self.assertIn('not SM occupancy',got['scope'])

    def test_nvml_optional_counters_reject_nonfinite_and_keep_valid_fields(self):
        folder=Path(__file__).resolve().parents[3]/'target/test-tmp/cloud-profiler'
        folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as directory:
            path=Path(directory)/'monitor.csv'
            path.write_text('t, 20, 8, 180, 1125, 37, P0\n'
                            't, 80, 9, 240, N/A, 38, P0\n'
                            't, NaN, 9, 240, 1140, N/A, P8\n'
                            't, Inf, 9, 240, NaN, Inf, N/A\n'
                            't, -1, 9, 240, -1, NaN, N/A\n')
            got=profile.nvml_summary(path)
            self.assertEqual(got['samples'],2)
            self.assertEqual(got['graphics_clock_mhz_range'],[1125,1140])
            self.assertEqual(got['graphics_clock_mhz_median'],1132.5)
            self.assertEqual(got['temperature_c_max'],38)
            self.assertEqual(got['performance_states'],{'P0':2,'P8':1})
            json.dumps(got,allow_nan=False)
            path.write_text('t, N/A, N/A, N/A, NaN, Inf, N/A\n')
            unavailable=profile.nvml_summary(path)
            self.assertEqual(unavailable['samples'],0)
            for key in ['gpu_activity_percent_mean','graphics_clock_mhz_median',
                        'graphics_clock_mhz_range','temperature_c_max']:
                self.assertIsNone(unavailable[key])
            self.assertEqual(unavailable['performance_states'],{})
            json.dumps(unavailable,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
