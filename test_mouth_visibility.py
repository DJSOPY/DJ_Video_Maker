"""口元が明示的に不在のBロール選択とキャッシュの回帰テスト。"""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mouth_sync


def _profile(seconds=30.0, fps=10.0):
    times = np.arange(0.0, seconds, 1.0 / fps, dtype=float)
    n = len(times)
    return {
        "times": times, "fps": fps, "thresh": 0.012,
        "activity": np.zeros(n), "mar": np.zeros(n),
        "face": np.ones(n, dtype=np.int8),
        "mouth_state": np.full(n, mouth_sync.MOUTH_CLEAR, dtype=np.int8),
        "mouth_visible": np.ones(n, dtype=np.int8),
        "mouth_absent": np.zeros(n, dtype=np.int8),
        "primary_mouth_state": np.full(
            n, mouth_sync.MOUTH_CLEAR, dtype=np.int8),
        "primary_mouth_visible": np.ones(n, dtype=np.int8),
        "primary_mouth_absent": np.zeros(n, dtype=np.int8),
        "face_count": np.ones(n, dtype=np.int8),
        "mouth_center": np.tile([0.5, 0.5], (n, 1)),
        "mouth_size": np.tile([0.10, 0.02], (n, 1)),
        "face_bbox": np.tile([0.3, 0.2, 0.7, 0.8], (n, 1)),
    }


class LandmarkStateTests(unittest.TestCase):
    @staticmethod
    def _points(mouth_width=0.10, center_x=0.5):
        points = [(500.0, 500.0)] * 468
        points[mouth_sync.UP_INNER] = (center_x * 1000.0, 490.0)
        points[mouth_sync.LO_INNER] = (center_x * 1000.0, 510.0)
        points[mouth_sync.L_CORNER] = (
            (center_x - mouth_width / 2.0) * 1000.0, 500.0)
        points[mouth_sync.R_CORNER] = (
            (center_x + mouth_width / 2.0) * 1000.0, 500.0)
        return points

    def test_visible_mouth_is_clear_and_geometry_is_normalized(self):
        result = mouth_sync._landmark_visibility(self._points(), 1000, 1000)
        self.assertEqual(result["mouth_state"], mouth_sync.MOUTH_CLEAR)
        self.assertTrue(result["mouth_visible"])
        self.assertFalse(result["mouth_absent"])
        self.assertAlmostEqual(result["mouth_size"][0], 0.1)

    def test_tiny_mouth_is_uncertain_not_absent(self):
        result = mouth_sync._landmark_visibility(
            self._points(mouth_width=0.01), 1000, 1000)
        self.assertEqual(result["mouth_state"], mouth_sync.MOUTH_UNCERTAIN)
        self.assertFalse(result["mouth_absent"])

    def test_wholly_offscreen_mouth_is_absent_observation(self):
        result = mouth_sync._landmark_visibility(
            self._points(mouth_width=0.10, center_x=1.2), 1000, 1000)
        self.assertEqual(result["mouth_state"], mouth_sync.MOUTH_ABSENT)
        self.assertTrue(result["mouth_absent"])

    def test_no_detection_without_successful_aux_check_is_uncertain(self):
        state = mouth_sync._aggregate_mouth_observation([], None)
        self.assertEqual(state, mouth_sync.MOUTH_UNCERTAIN)

    def test_no_detection_stays_uncertain_even_after_aux_checks_are_clear(self):
        state = mouth_sync._aggregate_mouth_observation([], False)
        self.assertEqual(state, mouth_sync.MOUTH_UNCERTAIN)

    def test_one_visible_mouth_among_multiple_faces_makes_frame_unsafe(self):
        absent = {"mouth_state": mouth_sync.MOUTH_ABSENT}
        clear = {"mouth_state": mouth_sync.MOUTH_CLEAR}
        state = mouth_sync._aggregate_mouth_observation([absent, clear], False)
        self.assertEqual(state, mouth_sync.MOUTH_CLEAR)


class ExtractorSafetyTests(unittest.TestCase):
    def test_tasks_backend_is_forced_to_cpu(self):
        """macOS Metal初期化失敗はSIGABRTになるためdelegate未指定へ戻さない。"""
        captured = {}

        class FakeBaseOptions:
            class Delegate:
                CPU = object()

            def __init__(self, **kwargs):
                captured.update(kwargs)

        class FakeLandmarker:
            @staticmethod
            def create_from_options(options):
                return object()

        fake_mp = types.ModuleType("mediapipe")
        fake_tasks = types.ModuleType("mediapipe.tasks")
        fake_python = types.ModuleType("mediapipe.tasks.python")
        fake_vision = types.ModuleType("mediapipe.tasks.python.vision")
        fake_python.BaseOptions = FakeBaseOptions
        fake_vision.RunningMode = types.SimpleNamespace(VIDEO="video")
        fake_vision.FaceLandmarkerOptions = lambda **kwargs: kwargs
        fake_vision.FaceLandmarker = FakeLandmarker
        fake_python.vision = fake_vision
        fake_tasks.python = fake_python
        fake_mp.tasks = fake_tasks

        modules = {
            "mediapipe": fake_mp,
            "mediapipe.tasks": fake_tasks,
            "mediapipe.tasks.python": fake_python,
            "mediapipe.tasks.python.vision": fake_vision,
        }
        with mock.patch.dict(sys.modules, modules):
            with mock.patch.object(mouth_sync.sys, "platform", "linux"):
                with mock.patch.object(mouth_sync.os.path, "exists", return_value=True):
                    mouth_sync._make_extractor()
        self.assertIs(captured.get("delegate"), FakeBaseOptions.Delegate.CPU)

    def test_tasks_backend_is_not_started_on_macos(self):
        fake_mp = types.ModuleType("mediapipe")
        with mock.patch.dict(sys.modules, {"mediapipe": fake_mp}):
            with mock.patch.object(mouth_sync.sys, "platform", "darwin"):
                with self.assertRaisesRegex(RuntimeError, "Tasks"):
                    mouth_sync._make_extractor()


class StrictNoMouthPickerTests(unittest.TestCase):
    def test_face_with_still_mouth_is_rejected(self):
        profile = _profile(seconds=12.0)
        self.assertIsNotNone(mouth_sync.pick_quiet_mv_time(
            profile, want_dur=2.0, mv_dur=12.0))
        self.assertIsNone(mouth_sync.pick_no_mouth_mv_time(
            profile, want_dur=2.0, mv_dur=12.0))

    def test_face_detection_failure_is_rejected(self):
        profile = _profile(seconds=12.0)
        profile["face"][:] = 0
        profile["mouth_state"][:] = mouth_sync.MOUTH_UNCERTAIN
        profile["mouth_visible"][:] = 0
        # mouth_absentは0のまま。「取れない」は不在証明ではない。
        self.assertIsNone(mouth_sync.pick_no_mouth_mv_time(
            profile, want_dur=2.0, mv_dur=12.0))

    def test_only_continuous_explicit_absent_run_is_selected(self):
        profile = _profile()
        hidden = (profile["times"] >= 8.0) & (profile["times"] < 16.0)
        profile["mouth_state"][hidden] = mouth_sync.MOUTH_ABSENT
        profile["mouth_visible"][hidden] = 0
        profile["mouth_absent"][hidden] = 1
        selected = mouth_sync.pick_no_mouth_mv_time(
            profile, want_dur=4.0, mv_dur=30.0)
        self.assertIsNotNone(selected)
        self.assertGreaterEqual(selected, 8.2)
        self.assertLessEqual(selected + 4.0, 15.8)

    def test_avoid_window_uses_other_absent_run(self):
        profile = _profile(seconds=32.0)
        first = (profile["times"] >= 4.0) & (profile["times"] < 10.0)
        second = (profile["times"] >= 20.0) & (profile["times"] < 26.0)
        profile["mouth_absent"][first | second] = 1
        selected = mouth_sync.pick_no_mouth_mv_time(
            profile, 2.0, 32.0, avoid=[6.0], avoid_win=8.0)
        self.assertIsNotNone(selected)
        self.assertGreaterEqual(selected, 20.0)

    def test_single_absent_dropout_is_rejected_by_hysteresis(self):
        profile = _profile(seconds=10.0)
        profile["mouth_absent"][50] = 1
        self.assertIsNone(mouth_sync.pick_no_mouth_mv_time(
            profile, 0.1, 10.0, hysteresis_sec=0.25))

    def test_old_profile_without_absent_evidence_fails_closed(self):
        profile = _profile()
        del profile["mouth_absent"]
        self.assertIsNone(mouth_sync.pick_no_mouth_mv_time(
            profile, 2.0, 30.0))


class VerifiedLipSyncVisualTests(unittest.TestCase):
    def _active_profile(self, seconds=3.0, fps=30.0):
        profile = _profile(seconds=seconds, fps=fps)
        profile["_all_source_frames"] = True
        profile["activity"][:] = profile["thresh"] * 2.0
        return profile

    def test_every_mapped_frame_must_be_clear_active_and_voiced(self):
        profile = self._active_profile()
        source = np.arange(0.25, 2.25, 1.0 / 30.0)
        active = np.ones(len(source), dtype=bool)
        self.assertTrue(
            mouth_sync.mapped_frames_have_verified_lipsync_visual(
                profile, source, active))

        bad_i = int(np.argmin(np.abs(profile["times"] - 1.0)))
        mutations = {
            "mouth_state": mouth_sync.MOUTH_UNCERTAIN,
            "face": 0,
            "mouth_visible": 0,
            "mouth_absent": 1,
            "activity": 0.0,
        }
        for key, value in mutations.items():
            broken = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                      for k, v in profile.items()}
            broken[key][bad_i] = value
            self.assertFalse(
                mouth_sync.mapped_frames_have_verified_lipsync_visual(
                    broken, source, active), key)

        inactive = active.copy(); inactive[len(inactive) // 2] = False
        self.assertFalse(
            mouth_sync.mapped_frames_have_verified_lipsync_visual(
                profile, source, inactive))

    def test_missing_sparse_old_and_out_of_range_profiles_fail_closed(self):
        profile = self._active_profile()
        source = np.arange(0.25, 2.25, 1.0 / 30.0)
        active = np.ones(len(source), dtype=bool)

        self.assertFalse(
            mouth_sync.mapped_frames_have_verified_lipsync_visual(
                None, source, active))
        old = dict(profile); old.pop("_all_source_frames")
        self.assertFalse(
            mouth_sync.mapped_frames_have_verified_lipsync_visual(
                old, source, active))
        sparse = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                  for k, v in profile.items()}
        keep = ~((sparse["times"] > 0.9) & (sparse["times"] < 1.2))
        for key in ("times", "mar", "activity", "face", "mouth_state",
                    "mouth_visible", "mouth_absent", "primary_mouth_state",
                    "primary_mouth_visible", "primary_mouth_absent",
                    "face_count", "mouth_center", "mouth_size", "face_bbox"):
            sparse[key] = sparse[key][keep]
        self.assertFalse(
            mouth_sync.mapped_frames_have_verified_lipsync_visual(
                sparse, source, active))
        self.assertFalse(
            mouth_sync.mapped_frames_have_verified_lipsync_visual(
                profile, np.array([99.0]), np.array([True])))

    def test_unselected_source_frame_between_30fps_outputs_is_still_checked(self):
        # 60fps sourceを30fps出力にすると、出力時刻の間にも
        # source frameがある。ffmpegの丸め方依存にせず全て見る。
        profile = self._active_profile(seconds=3.0, fps=60.0)
        source = np.arange(0.0, 2.0, 1.0 / 30.0)
        active = np.ones(len(source), dtype=bool)
        between = int(np.argmin(np.abs(profile["times"] - (1.0 / 60.0))))
        profile["mouth_state"][between] = mouth_sync.MOUTH_UNCERTAIN
        profile["mouth_visible"][between] = 0
        self.assertFalse(
            mouth_sync.mapped_frames_have_verified_lipsync_visual(
                profile, source, active))

    def test_one_missing_source_frame_invalidates_all_frame_claim(self):
        profile = self._active_profile(seconds=3.0, fps=30.0)
        source = np.arange(0.0, 2.0, 1.0 / 30.0)
        active = np.ones(len(source), dtype=bool)
        drop = int(np.argmin(np.abs(profile["times"] - 1.0)))
        for key in ("times", "mar", "activity", "face", "mouth_state",
                    "mouth_visible", "mouth_absent", "primary_mouth_state",
                    "primary_mouth_visible", "primary_mouth_absent",
                    "face_count", "mouth_center", "mouth_size", "face_bbox"):
            profile[key] = np.delete(profile[key], drop)
        self.assertFalse(
            mouth_sync.mapped_frames_have_verified_lipsync_visual(
                profile, source, active))

    def test_fast_mapping_cannot_jump_over_a_missing_source_frame(self):
        profile = self._active_profile(seconds=3.0, fps=30.0)
        drop = int(np.argmin(np.abs(profile["times"] - 1.0)))
        for key in ("times", "mar", "activity", "face", "mouth_state",
                    "mouth_visible", "mouth_absent", "primary_mouth_state",
                    "primary_mouth_visible", "primary_mouth_absent",
                    "face_count", "mouth_center", "mouth_size", "face_bbox"):
            profile[key] = np.delete(profile[key], drop)
        self.assertFalse(
            mouth_sync.mapped_frames_have_verified_lipsync_visual(
                profile, np.array([0.95, 1.05]), np.array([True, True])))

    def test_ffmpeg_seek_edge_guard_checks_the_next_source_frame(self):
        profile = self._active_profile(seconds=1.0, fps=30.0)
        profile["mouth_state"][1] = mouth_sync.MOUTH_UNCERTAIN
        profile["primary_mouth_state"][1] = mouth_sync.MOUTH_UNCERTAIN
        profile["mouth_visible"][1] = 0
        profile["primary_mouth_visible"][1] = 0
        self.assertFalse(
            mouth_sync.mapped_frames_have_verified_lipsync_visual(
                profile, np.array([0.001]), np.array([True])))

    def test_multiple_faces_or_primary_identity_jump_is_never_combined(self):
        source = np.arange(0.25, 2.25, 1.0 / 30.0)
        active = np.ones(len(source), dtype=bool)
        for mutation in ("multi", "primary", "jump"):
            profile = self._active_profile()
            i = int(np.argmin(np.abs(profile["times"] - 1.0)))
            if mutation == "multi":
                profile["face_count"][i] = 2
            elif mutation == "primary":
                profile["primary_mouth_state"][i] = mouth_sync.MOUTH_UNCERTAIN
                profile["primary_mouth_visible"][i] = 0
            else:
                profile["mouth_center"][i:] += np.array([0.3, 0.0])
            self.assertFalse(
                mouth_sync.mapped_frames_have_verified_lipsync_visual(
                    profile, source, active), mutation)

    def test_shot_change_does_not_create_fake_mouth_activity_or_flux(self):
        n = 60; fps = 30.0
        mar = np.r_[np.full(30, 0.10), np.full(30, 0.40)]
        face = np.ones(n, dtype=np.int8)
        center = np.tile([0.3, 0.5], (n, 1)); center[30:, 0] = 0.7
        size = np.tile([0.10, 0.02], (n, 1))
        bbox = np.tile([0.1, 0.2, 0.5, 0.8], (n, 1))
        bbox[30:] = [0.5, 0.2, 0.9, 0.8]
        activity = mouth_sync._activity_from_mar(
            mar, face, fps, mouth_center=center,
            mouth_size=size, face_bbox=bbox)
        self.assertAlmostEqual(float(np.max(activity)), 0.0)
        profile = {"times": np.arange(n) / fps, "mar": mar, "face": face,
                   "mouth_center": center, "mouth_size": size,
                   "face_bbox": bbox}
        _times, flux = mouth_sync.mouth_open_flux(profile)
        self.assertAlmostEqual(float(np.max(flux)), 0.0)


class MouthCacheTests(unittest.TestCase):
    def test_old_cache_loads_as_uncertain(self):
        with tempfile.TemporaryDirectory(
                dir="/private/tmp" if os.path.isdir("/private/tmp") else None) as tmp:
            old_dir = mouth_sync._MOUTH_CACHE_DIR
            mouth_sync._MOUTH_CACHE_DIR = tmp
            try:
                np.savez_compressed(
                    os.path.join(tmp, "old.npz"), times=np.arange(3.0),
                    mar=np.zeros(3), activity=np.zeros(3), face=np.ones(3),
                    fps=1.0, thresh=0.012, face_rate=1.0,
                    backend="old")
                loaded = mouth_sync._mouth_cache_load("old")
            finally:
                mouth_sync._MOUTH_CACHE_DIR = old_dir
        self.assertIsNotNone(loaded)
        self.assertTrue(np.all(loaded["mouth_state"] == mouth_sync.MOUTH_UNCERTAIN))
        self.assertEqual(int(np.sum(loaded["mouth_absent"])), 0)

    def test_new_visibility_arrays_round_trip(self):
        profile = _profile(seconds=1.0)
        n = len(profile["times"])
        profile.update({
            "face_rate": 1.0, "backend": "synthetic",
            "face_visibility": np.ones(n),
            "face_bbox": np.zeros((n, 4)),
            "mouth_center": np.zeros((n, 2)),
            "mouth_size": np.zeros((n, 2)),
            "mouth_bbox": np.zeros((n, 4)),
            "face_count": np.ones(n, dtype=np.int8),
            "aux_presence": np.full(n, -1, dtype=np.int8),
        })
        profile["mouth_absent"][3:5] = 1
        with tempfile.TemporaryDirectory(
                dir="/private/tmp" if os.path.isdir("/private/tmp") else None) as tmp:
            old_dir = mouth_sync._MOUTH_CACHE_DIR
            mouth_sync._MOUTH_CACHE_DIR = tmp
            try:
                mouth_sync._mouth_cache_save("new", profile)
                loaded = mouth_sync._mouth_cache_load("new")
            finally:
                mouth_sync._MOUTH_CACHE_DIR = old_dir
        np.testing.assert_array_equal(loaded["mouth_absent"],
                                      profile["mouth_absent"])

    def test_use_cache_false_skips_cache_key_load_and_save(self):
        times = np.array([0.0, 0.1])
        vis = {
            "mouth_state": np.zeros(2, np.int8),
            "mouth_visible": np.zeros(2, np.int8),
            "mouth_absent": np.zeros(2, np.int8),
            "face_visibility": np.zeros(2), "face_bbox": np.zeros((2, 4)),
            "mouth_center": np.zeros((2, 2)), "mouth_size": np.zeros((2, 2)),
            "mouth_bbox": np.zeros((2, 4)), "face_count": np.zeros(2, np.int8),
            "aux_presence": np.full(2, -1, np.int8),
        }
        analyzed = (times, np.zeros(2), np.zeros(2), np.zeros(2),
                    0.2, "mock", vis)
        with mock.patch.object(mouth_sync, "_mouth_cache_key",
                               side_effect=AssertionError("cache key called")):
            with mock.patch.object(mouth_sync, "analyze_mouth",
                                   return_value=analyzed):
                result = mouth_sync.build_mouth_profile(
                    "not-needed.mov", use_cache=False)
        self.assertIsNotNone(result)
        self.assertEqual(int(np.sum(result["mouth_absent"])), 0)


if __name__ == "__main__":
    unittest.main()
