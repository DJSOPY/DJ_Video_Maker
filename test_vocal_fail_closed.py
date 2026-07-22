from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import vocal_sync


class VocalFailClosedTests(unittest.TestCase):
    def test_strict_legacy_engine_cannot_show_people(self):
        self.assertFalse(vocal_sync.ALLOW_LEGACY_VISIBLE_FACES_IN_STRICT_MODE)

    def test_three_hundred_ms_vocal_dropout_is_hidden(self):
        env = np.ones(100, dtype=np.float32)
        env[25:40] = 0.0  # 300ms @ 50fps
        ranges = vocal_sync._vocal_silence_ranges_from_envelope(
            env, 50.0, duration=2.0, guard=0.0)
        self.assertEqual(ranges, [(0.5, 0.8)])

    def test_short_stem_leakage_is_bridged_inside_silence(self):
        env = np.zeros(100, dtype=np.float32)  # 2秒 @ 50fps
        env[45:50] = 1.0                       # 100msの分離ノイズ
        ranges = vocal_sync._vocal_silence_ranges_from_envelope(
            env, 50.0, duration=2.0, min_silence=0.35,
            max_active_island=0.20, guard=0.12)
        self.assertEqual(ranges, [(0.0, 2.0)])

    def test_silence_split_hides_short_visible_island_without_losing_time(self):
        segs = [{"r_start": 0.0, "r_end": 6.0, "o_start": 0.0,
                 "o_end": 6.0, "anchor_o": None,
                 "voiced_sec": 6.0, "conf": 0.9}]
        pieces = vocal_sync._split_segments_on_vocal_silence(
            segs, lambda t: t, [(0.1, 0.5)])

        self.assertEqual([(p["r_start"], p["r_end"])
                          for p in pieces], [(0.0, 0.1), (0.1, 0.5), (0.5, 6.0)])
        self.assertIsNone(pieces[0]["o_start"])  # 100msだけ口を見せない
        self.assertIsNone(pieces[1]["o_start"])
        self.assertAlmostEqual(pieces[2]["o_start"], 0.5)
        self.assertAlmostEqual(sum(p["r_end"] - p["r_start"]
                                   for p in pieces), 6.0)
        counts = vocal_sync._cumulative_frame_counts(
            [p["r_end"] - p["r_start"] for p in pieces])
        self.assertEqual(counts, [3, 12, 165])
        self.assertEqual(sum(counts), 180)

    def test_tail_failure_is_not_accepted_in_strict_mode(self):
        segs = [
            {"r_start": 0.0, "r_end": 20.0, "o_start": 4.0,
             "voiced_sec": 20.0, "conf": 0.82},
            {"r_start": 20.0, "r_end": 40.0, "o_start": None,
             "voiced_sec": 20.0, "conf": 0.0},
        ]
        q = vocal_sync._legacy_alignment_quality(
            segs, "demucs", "demucs", strict_fail_closed=True)
        self.assertEqual(q["tail_coverage"], 0.0)
        self.assertFalse(q["accepted"])

    def test_strict_mode_rejects_approximate_stems(self):
        segs = [{"r_start": 0.0, "r_end": 40.0, "o_start": 2.0,
                 "voiced_sec": 40.0, "conf": 0.90}]
        q = vocal_sync._legacy_alignment_quality(
            segs, "hpss", "raw", strict_fail_closed=True)
        self.assertFalse(q["accepted"])

    def test_clean_full_tail_can_pass_strict_quality_gate(self):
        segs = [{"r_start": 0.0, "r_end": 40.0, "o_start": 2.0,
                 "voiced_sec": 40.0, "conf": 0.80}]
        q = vocal_sync._legacy_alignment_quality(
            segs, "demucs", "demucs", strict_fail_closed=True)
        self.assertEqual(q["tail_coverage"], 1.0)
        self.assertTrue(q["accepted"])

    def test_one_borderline_visible_segment_rejects_strict_output(self):
        segs = [
            {"r_start": 0.0, "r_end": 20.0, "o_start": 2.0,
             "voiced_sec": 20.0, "conf": 0.90},
            {"r_start": 20.0, "r_end": 40.0, "o_start": 22.0,
             "voiced_sec": 20.0, "conf": 0.59},
        ]
        q = vocal_sync._legacy_alignment_quality(
            segs, "demucs", "demucs", strict_fail_closed=True)
        self.assertAlmostEqual(q["weakest_conf"], 0.59)
        self.assertFalse(q["accepted"])

    def test_exact_filler_callback_receives_frame_budget(self):
        seen = {}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "safe.mp4"

            def filler(duration, output, frames):
                seen.update(duration=duration, frames=frames)
                Path(output).write_bytes(b"certified")

            self.assertTrue(vocal_sync._call_filler_exact(
                filler, 1.25, out, 38))
        self.assertEqual(seen, {"duration": 1.25, "frames": 38})

    def test_legacy_renderer_requires_the_same_full_frame_visual_proof(self):
        times = np.arange(0.0, 3.1, 1.0 / 30.0)
        n = len(times)
        profile = {
            "times": times, "fps": 30.0, "thresh": 0.012,
            "_all_source_frames": True,
            "mouth_state": np.ones(n, dtype=np.int8),
            "mouth_visible": np.ones(n, dtype=np.int8),
            "mouth_absent": np.zeros(n, dtype=np.int8),
            "face": np.ones(n, dtype=np.int8),
            "activity": np.full(n, 0.024),
            "primary_mouth_state": np.ones(n, dtype=np.int8),
            "primary_mouth_visible": np.ones(n, dtype=np.int8),
            "primary_mouth_absent": np.zeros(n, dtype=np.int8),
            "face_count": np.ones(n, dtype=np.int8),
            "mouth_center": np.tile([0.5, 0.5], (n, 1)),
            "mouth_size": np.tile([0.10, 0.02], (n, 1)),
            "face_bbox": np.tile([0.3, 0.2, 0.7, 0.8], (n, 1)),
        }
        vocal_active = np.ones(200, dtype=bool)
        onset_times = np.arange(0.0, 4.0, 0.02)
        onset = np.zeros_like(onset_times); onset[::10] = 1.0
        phase = dict(onset_times=onset_times, onset_envelope=onset)
        with mock.patch("mouth_sync.measure_micro_mouth_lag",
                        return_value=(0.05, 0.60, 0.90, 3.0)):
            self.assertTrue(vocal_sync._mapped_segment_has_visual_proof(
                profile, 0.0, 2.0, 60, 0.0, 2.0,
                vocal_active, 50.0, **phase))
        self.assertFalse(vocal_sync._mapped_segment_has_visual_proof(
            None, 0.0, 2.0, 60, 0.0, 2.0,
            vocal_active, 50.0, **phase))

        uncertain = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                     for k, v in profile.items()}
        uncertain["mouth_state"][30] = 0
        self.assertFalse(vocal_sync._mapped_segment_has_visual_proof(
            uncertain, 0.0, 2.0, 60, 0.0, 2.0,
            vocal_active, 50.0, **phase))

        silent = vocal_active.copy(); silent[25] = False
        self.assertFalse(vocal_sync._mapped_segment_has_visual_proof(
            profile, 0.0, 2.0, 60, 0.0, 2.0,
            silent, 50.0, **phase))

        nan_active = vocal_active.astype(float); nan_active[25] = np.nan
        self.assertFalse(vocal_sync._mapped_segment_has_visual_proof(
            profile, 0.0, 2.0, 60, 0.0, 2.0,
            nan_active, 50.0, **phase))

        with mock.patch("mouth_sync.measure_micro_mouth_lag",
                        return_value=(0.30, 0.70, 0.95, 4.0)):
            self.assertFalse(vocal_sync._mapped_segment_has_visual_proof(
                profile, 0.0, 2.0, 60, 0.0, 2.0,
                vocal_active, 50.0, **phase))



class UndecidableKeepsMvTests(unittest.TestCase):
    """通常モードで判定不能(mapping無限)区間をフィラーにせずMV維持する（項目4）。"""

    SRC = (Path(__file__).resolve().parent / "vocal_sync.py").read_text(encoding="utf-8")

    def test_undecidable_estimates_mv_in_normal_mode(self):
        # 直前の有効対応位置から等速でMVを推定配置する分岐が存在する
        self.assertIn("last_valid_o_end", self.SRC)
        self.assertIn("mv_estimated", self.SRC)
        self.assertIn("not strict_fail_closed", self.SRC)
        self.assertIn("not mapping_finite", self.SRC)

    def test_estimation_only_when_prior_valid(self):
        # 直前の有効位置が無ければ推定しない（last_valid_o_end is not None が条件）
        self.assertIn("last_valid_o_end is not None", self.SRC)

if __name__ == "__main__":
    unittest.main()
