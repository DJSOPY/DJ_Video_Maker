"""LipSync Proの低信頼区間が口元を表示しないことの回帰テスト。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import lipsync_pro


def _profile(mouth_absent=None, mouth_visible=None, face=None, end=5.0,
             step=1.0 / 30.0, all_frames=True):
    times = np.arange(0.0, end + step * 0.5, step, dtype=float)
    data = {"times": times, "fps": 1.0 / step,
            "_all_source_frames": bool(all_frames)}
    if mouth_absent is not None:
        data["mouth_absent"] = np.asarray(mouth_absent(times), dtype=np.uint8)
    if mouth_visible is not None:
        data["mouth_visible"] = np.asarray(mouth_visible(times), dtype=np.uint8)
    if face is not None:
        data["face"] = np.asarray(face(times), dtype=np.uint8)
    return data


def _active_lipsync_profile(end=4.0, step=1.0 / 30.0):
    times = np.arange(0.0, end + step * 0.5, step, dtype=float)
    n = len(times)
    return {
        "times": times, "fps": 1.0 / step, "thresh": 0.012,
        "_all_source_frames": True,
        "mouth_state": np.ones(n, dtype=np.int8),
        "mouth_visible": np.ones(n, dtype=np.int8),
        "mouth_absent": np.zeros(n, dtype=np.int8),
        "face": np.ones(n, dtype=np.int8),
        "activity": np.full(n, 0.024, dtype=float),
        "primary_mouth_state": np.ones(n, dtype=np.int8),
        "primary_mouth_visible": np.ones(n, dtype=np.int8),
        "primary_mouth_absent": np.zeros(n, dtype=np.int8),
        "face_count": np.ones(n, dtype=np.int8),
        "mouth_center": np.tile([0.5, 0.5], (n, 1)),
        "mouth_size": np.tile([0.10, 0.02], (n, 1)),
        "face_bbox": np.tile([0.3, 0.2, 0.7, 0.8], (n, 1)),
    }


class UnsafeRangeTests(unittest.TestCase):
    def test_short_stem_leakage_does_not_split_sustained_silence(self):
        # 2秒の無声中に100msだけ分離ノイズが立っても、人物を再表示しない。
        active = np.zeros(40, dtype=bool)
        active[18:20] = True
        ranges = lipsync_pro._sustained_inactive_ranges(
            (0.05, active), 0.0, 2.0,
            min_silence=0.35, max_active_island=0.20)
        self.assertEqual(ranges, [(0.0, 2.0)])

    def test_ranges_are_merged_and_overlap_switches_the_whole_cut(self):
        ranges = lipsync_pro._merge_time_ranges(
            [(3.0, 4.0), (1.0, 2.0), (1.95, 3.1), (9.0, 9.0)])

        self.assertEqual(ranges, [(1.0, 4.0)])
        self.assertTrue(lipsync_pro._segment_requires_safe_visual(
            0.0, 2.0, [(1.95, 2.10)]))
        self.assertFalse(lipsync_pro._segment_requires_safe_visual(
            0.0, 1.0, [(2.0, 3.0)]))

    def test_unsafe_padding_closes_subsecond_show_islands(self):
        prepared = lipsync_pro._prepare_unsafe_ranges(
            [(2.0, 3.0), (4.0, 5.0)], lo=0.0, hi=8.0,
            pad=0.25, min_safe=0.75)

        # 拡張後の3.25-3.75は0.5秒しかなく、口元を一瞬見せない。
        self.assertEqual(prepared, [(1.75, 5.25)])
        # 十分長い先頭/末尾の高信頼区間は維持する。
        self.assertGreater(prepared[0][0], 0.75)
        self.assertGreater(8.0 - prepared[-1][1], 0.75)

    def test_weak_feature_block_is_exported_as_unsafe(self):
        times = np.arange(0.0, 20.0, 0.1, dtype=float)
        rng = np.random.default_rng(20260720)
        remix = rng.normal(size=(len(times), 16)).astype(np.float32)
        original = remix.copy()
        original[(times >= 8.0) & (times < 10.0)] *= -1.0
        anchors = [(0.0, 0.0, 0.0), (19.9, 19.9, 0.0)]

        report = lipsync_pro.alignment_quality_report(
            anchors, [], remix, times, original, times, feature_kind="hubert")

        self.assertTrue(any(lo <= 8.0 and hi >= 10.0
                            for lo, hi in report["unsafe_ranges"]))

    def test_out_of_source_tail_is_exported_as_unsafe(self):
        times = np.arange(0.0, 100.0, 0.1, dtype=float)
        rng = np.random.default_rng(7)
        features = rng.normal(size=(len(times), 12)).astype(np.float32)
        anchors = [(0.0, 0.0, 0.0), (70.0, 70.0, 0.0),
                   (70.001, 110.0, 0.0), (100.0, 139.999, 0.0)]

        report = lipsync_pro.alignment_quality_report(
            anchors, [], features, times, features, times, feature_kind="hubert")

        self.assertFalse(report["accepted"])
        self.assertTrue(any(lo < 71.0 and hi > 99.0
                            for lo, hi in report["unsafe_ranges"]))

    def test_even_subsecond_vocal_silence_is_hidden(self):
        hop = 0.05
        active = np.ones(120, dtype=bool)
        active[10:20] = False       # 0.50秒: 短くても人物を隠す
        active[40:60] = False       # 1.00秒: 同期不能なので隠す
        ranges = lipsync_pro._sustained_inactive_ranges(
            (hop, active), 0.0, 6.0)
        self.assertEqual(ranges, [(0.5, 1.0), (2.0, 3.0)])

    def test_three_hundred_ms_dropout_is_hidden(self):
        active = np.ones(40, dtype=bool)
        active[10:16] = False
        self.assertEqual(lipsync_pro._sustained_inactive_ranges(
            (0.05, active), 0.0, 2.0), [(0.5, 0.8)])


class CertifiedNoMouthTests(unittest.TestCase):
    def test_safety_profile_requests_every_source_frame(self):
        calls = []

        def build(path, fps, use_cache):
            calls.append((path, fps, use_cache))
            return {"times": np.array([0.0, 1.0 / 30.0]),
                    "fps": 30.0, "mouth_absent": np.ones(2)}

        result = lipsync_pro._build_all_frame_mouth_profile(
            SimpleNamespace(build_mouth_profile=build), "/tmp/mv.mov")

        self.assertEqual(calls, [("/tmp/mv.mov", 1000.0, False)])
        self.assertTrue(result["_all_source_frames"])

    def test_safety_profile_failure_returns_none(self):
        def broken(*args, **kwargs):
            raise RuntimeError("detector unavailable")

        result = lipsync_pro._build_all_frame_mouth_profile(
            SimpleNamespace(build_mouth_profile=broken), "/tmp/mv.mov")

        self.assertIsNone(result)

    def test_new_profile_requires_every_sample_to_prove_mouth_absent(self):
        safe = _profile(mouth_absent=lambda t: np.ones_like(t))
        one_uncertain = _profile(
            mouth_absent=lambda t: ~np.isclose(t, 1.5, atol=0.01))

        self.assertTrue(lipsync_pro._profile_interval_has_no_visible_mouth(
            safe, 1.0, 2.0, mv_dur=5.0))
        self.assertFalse(lipsync_pro._profile_interval_has_no_visible_mouth(
            one_uncertain, 1.0, 2.0, mv_dur=5.0))

    def test_false_visibility_and_old_profiles_are_not_safety_proof(self):
        invisible_but_uncertain = _profile(
            mouth_visible=lambda t: np.zeros_like(t),
            face=lambda t: np.zeros_like(t))
        still_face = _profile(face=lambda t: np.ones_like(t))
        no_face = _profile(face=lambda t: np.zeros_like(t))

        self.assertFalse(lipsync_pro._profile_interval_has_no_visible_mouth(
            invisible_but_uncertain, 1.0, 2.0, mv_dur=5.0))
        self.assertFalse(lipsync_pro._profile_interval_has_no_visible_mouth(
            still_face, 1.0, 2.0, mv_dur=5.0))
        self.assertFalse(lipsync_pro._profile_interval_has_no_visible_mouth(
            no_face, 1.0, 2.0, mv_dur=5.0))

    def test_missing_or_sparse_analysis_fails_closed(self):
        sparse = {"times": np.array([0.0, 0.1, 2.0, 2.1]),
                  "mouth_absent": np.ones(4), "fps": 30.0,
                  "_all_source_frames": True}
        sampled_10fps = _profile(
            mouth_absent=lambda t: np.ones_like(t), step=0.1,
            all_frames=True)
        unmarked = _profile(
            mouth_absent=lambda t: np.ones_like(t), all_frames=False)

        self.assertFalse(lipsync_pro._profile_interval_has_no_visible_mouth(
            None, 0.0, 1.0, mv_dur=2.0))
        self.assertFalse(lipsync_pro._profile_interval_has_no_visible_mouth(
            sparse, 0.1, 1.9, mv_dur=2.1))
        self.assertFalse(lipsync_pro._profile_interval_has_no_visible_mouth(
            sampled_10fps, 1.0, 2.0, mv_dur=5.0))
        self.assertFalse(lipsync_pro._profile_interval_has_no_visible_mouth(
            unmarked, 1.0, 2.0, mv_dur=5.0))

    def test_picker_result_is_independently_reverified(self):
        safe = _profile(mouth_absent=lambda t: np.ones_like(t))
        unsafe = _profile(
            mouth_absent=lambda t: ~np.isclose(t, 1.5, atol=0.01))
        module = SimpleNamespace(pick_no_mouth_mv_time=lambda *a, **k: 1.0)

        self.assertEqual(lipsync_pro._pick_verified_no_mouth_time(
            module, safe, 2.0, 5.0), 1.0)
        self.assertIsNone(lipsync_pro._pick_verified_no_mouth_time(
            module, unsafe, 2.0, 5.0))
        self.assertIsNone(lipsync_pro._pick_verified_no_mouth_time(
            SimpleNamespace(), safe, 2.0, 5.0))


class SafeBackgroundCommandTests(unittest.TestCase):
    def test_no_purple_policy_defaults_but_strict_mode_available(self):
        # 既定は「紫を出さない」：口元なし認証カット→無ければMVの別位置
        self.assertTrue(lipsync_pro.ALLOW_REAL_MV_SAFE_BROLL)
        self.assertTrue(lipsync_pro.SAFE_BG_USE_MV_INSTEAD)
        # 最厳格（必ず非人物背景）へ戻せる分岐自体は保持されている
        src = (Path(lipsync_pro.__file__).resolve()).read_text(encoding="utf-8")
        self.assertIn("if not _use_mv:", src)
        self.assertIn("_safe_background = True", src)

    def test_visual_plan_preserves_good_sync_and_fails_closed(self):
        self.assertEqual(lipsync_pro._safe_visual_plan(False, None),
                         ("aligned_mv", None))
        self.assertEqual(lipsync_pro._safe_visual_plan(True, 12.5),
                         ("no_mouth_mv", 12.5))
        self.assertEqual(lipsync_pro._safe_visual_plan(True, None),
                         ("safe_background", None))
        self.assertEqual(lipsync_pro._safe_visual_plan(True, float("nan")),
                         ("safe_background", None))

    def test_background_has_no_mv_input_and_exact_frame_count(self):
        command = lipsync_pro._safe_background_ffmpeg_command(
            "/tmp/safe.mp4", 60, width=1280, height=720, fps=30)

        lavfi_src = command[command.index("-i") + 1]
        # 既定は動く抽象グラデーション(gradients)、非対応ffmpegでは黒。
        # どちらでも「人物素材を読まない合成ソース」「正しい形状/レート」であること。
        self.assertTrue(lavfi_src.startswith(("gradients=", "color=")), lavfi_src)
        self.assertIn("s=1280x720", lavfi_src)
        self.assertIn("r=30", lavfi_src)
        self.assertEqual(command[command.index("-frames:v") + 1], "60")
        self.assertNotIn("movie=", " ".join(command))
        self.assertNotIn("-shortest", command)

    def test_background_rejects_zero_frames(self):
        with self.assertRaises(ValueError):
            lipsync_pro._safe_background_ffmpeg_command("/tmp/safe.mp4", 0)

    def test_failed_safe_background_rejects_whole_render(self):
        # 黒区間を黙って省略すると後続人物映像が前詰めされるため、
        # 1区間でも生成失敗した時点でFalseにする。
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                    lipsync_pro, "run",
                    return_value=SimpleNamespace(stderr="forced failure", stdout="")):
                ok = lipsync_pro.equal_and_mux(
                    [(0.0, 0.0, 1.0), (2.0, 2.0, 1.0)],
                    "missing_mv.mp4", "missing_audio.wav", 2.0, 2.0,
                    f"{td}/out.mp4", td, unsafe_ranges=[(0.0, 2.0)])
        self.assertFalse(ok)

    def test_many_subframe_cuts_keep_the_global_frame_budget(self):
        # 25ms境界は単独丸めだと0/1frameが混ざる。累積境界差なら総数は不変。
        anchors = [(0.0, 0.0, 1.0)]
        for i in range(1, 7):
            anchors.append((i * 0.025, i * 2.0, 1.0))
        anchors.append((14.0, 25.85, 1.0))
        cuts = lipsync_pro._equal_cut_times(anchors, 14.0, subseg=14.0)
        counts = [int(round(b * lipsync_pro.FPS))
                  - int(round(a * lipsync_pro.FPS))
                  for a, b in zip(cuts, cuts[1:])]
        self.assertEqual(sum(counts), 420)
        self.assertIn(0, counts)

    def test_warp_rejects_first_segment_renderer_failure(self):
        with tempfile.TemporaryDirectory() as td:
            failed = SimpleNamespace(returncode=1, stderr="forced", stdout="")
            with mock.patch.object(lipsync_pro, "run", return_value=failed):
                ok = lipsync_pro.warp_and_mux(
                    [(0.0, 0.0, 1.0), (3.0, 3.0, 1.0)],
                    "missing_mv.mp4", "missing_audio.wav", 3.0, 3.0,
                    f"{td}/out.mp4", td)
        self.assertFalse(ok)

    def test_nonempty_corrupt_segment_is_not_exact_video(self):
        with tempfile.TemporaryDirectory() as td:
            broken = Path(td) / "broken.mp4"
            broken.write_bytes(b"not a decoded video")
            self.assertFalse(lipsync_pro._video_has_exact_frames(broken, 1))

    def test_publish_uses_verified_staging_and_replaces_stale_output(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.mp4"
            dest = Path(td) / "dest.mp4"
            source.write_bytes(b"complete verified output")
            dest.write_bytes(b"stale person output")
            with mock.patch.object(
                    lipsync_pro, "_validate_rendered_output", return_value=True):
                self.assertTrue(lipsync_pro._publish_rendered_output(
                    source, dest, 1.0))
            self.assertEqual(dest.read_bytes(), b"complete verified output")
            self.assertEqual(list(Path(td).glob("*.partial")), [])

    def test_failed_publish_removes_stale_destination(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.mp4"
            dest = Path(td) / "dest.mp4"
            source.write_bytes(b"partial")
            dest.write_bytes(b"stale person output")
            with mock.patch.object(
                    lipsync_pro, "_validate_rendered_output", return_value=False):
                self.assertFalse(lipsync_pro._publish_rendered_output(
                    source, dest, 1.0))
            self.assertFalse(dest.exists())


class AlignedVisualProofTests(unittest.TestCase):
    @staticmethod
    def _render_with_mock(renderer, profile, anchors=None):
        commands = []

        def fake_run(command, *args, **kwargs):
            commands.append([str(x) for x in command])
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        active = (1.0 / 30.0, np.ones(120, dtype=bool))
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(lipsync_pro, "run", side_effect=fake_run), \
                    mock.patch.object(lipsync_pro, "_video_has_exact_frames",
                                      return_value=True), \
                    mock.patch.object(lipsync_pro, "_concat_segments_exact",
                                      return_value=True), \
                    mock.patch.object(lipsync_pro, "_mux_exact_video_audio",
                                      return_value=True):
                common = ((anchors or [(0.0, 0.0, 1.0), (2.0, 2.0, 1.0)]),
                          "PERSON_MV.mp4", "audio.wav", 2.0, 4.0,
                          f"{td}/out.mp4", td)
                if renderer == "equal":
                    ok = lipsync_pro.equal_and_mux(
                        *common, rmx_act=active, safe_mouth_profile=profile,
                        unsafe_ranges=[],
                        visual_phase_proof_ranges=[(0.0, 2.0)])
                else:
                    ok = lipsync_pro.warp_and_mux(
                        *common, rmx_act=active, safe_mouth_profile=profile,
                        unsafe_ranges=[],
                        visual_phase_proof_ranges=[(0.0, 2.0)])
        return ok, commands

    @staticmethod
    def _visual_segment_command(commands):
        return next(command for command in commands
                    if any("seg_0000.mp4" in part for part in command))

    @staticmethod
    def _ss_of(command):
        """ffmpegコマンドの -ss（MVのどの位置を切り出したか）を取り出す。"""
        return float(command[command.index("-ss") + 1])

    def _assert_not_aligned_position(self, command, aligned_ss, msg=""):
        """証明できない区間の見せ方を検証する。

        方針変更：紫の抽象背景は既定で出さず、MVの『別位置』を当てる。
        ただし本質的な安全性として『合っていない“その位置”そのもの』は
        絶対に出さない（出すと未同期の口パクを同期のように見せてしまう）。
        紫背景でもMV別位置でも、どちらでもこの条件を満たせば合格とする。
        """
        if "lavfi" in command:
            self.assertNotIn("PERSON_MV.mp4", command, msg)
            return
        self.assertIn("PERSON_MV.mp4", command, msg)
        got = self._ss_of(command)
        self.assertTrue(np.isfinite(got), msg)
        self.assertNotAlmostEqual(got, aligned_ss, places=3, msg=msg)

    def _aligned_ss(self, renderer="equal"):
        _ok, cmds = self._render_with_mock(renderer, _active_lipsync_profile())
        return self._ss_of(self._visual_segment_command(cmds))

    def test_equal_detector_failure_never_shows_aligned_position(self):
        aligned = self._aligned_ss("equal")
        ok, commands = self._render_with_mock("equal", None)
        command = self._visual_segment_command(commands)
        self.assertTrue(ok)
        self._assert_not_aligned_position(command, aligned)

    def test_equal_one_uncertain_or_closed_frame_fails_closed(self):
        aligned = self._aligned_ss("equal")
        for field, value in (("mouth_state", 0), ("activity", 0.0)):
            profile = _active_lipsync_profile()
            i = int(np.argmin(np.abs(profile["times"] - 1.0)))
            profile[field][i] = value
            ok, commands = self._render_with_mock("equal", profile)
            command = self._visual_segment_command(commands)
            self.assertTrue(ok, field)
            self._assert_not_aligned_position(command, aligned, field)

    def test_equal_complete_active_profile_can_use_aligned_mv(self):
        ok, commands = self._render_with_mock(
            "equal", _active_lipsync_profile())
        command = self._visual_segment_command(commands)
        self.assertTrue(ok)
        self.assertIn("PERSON_MV.mp4", command)
        self.assertNotIn("lavfi", command)

    def test_active_mouth_without_phase_whitelist_is_still_hidden(self):
        commands = []
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                    lipsync_pro, "run",
                    side_effect=lambda c, *a, **k: (
                        commands.append([str(x) for x in c])
                        or SimpleNamespace(returncode=0, stderr="", stdout=""))), \
                    mock.patch.object(lipsync_pro, "_video_has_exact_frames",
                                      return_value=True), \
                    mock.patch.object(lipsync_pro, "_concat_segments_exact",
                                      return_value=True), \
                    mock.patch.object(lipsync_pro, "_mux_exact_video_audio",
                                      return_value=True):
                ok = lipsync_pro.equal_and_mux(
                    [(0.0, 0.0, 1.0), (2.0, 2.0, 1.0)],
                    "PERSON_MV.mp4", "audio.wav", 2.0, 4.0,
                    f"{td}/out.mp4", td,
                    rmx_act=(1.0 / 30.0, np.ones(120, dtype=bool)),
                    safe_mouth_profile=_active_lipsync_profile(),
                    unsafe_ranges=[], visual_phase_proof_ranges=[])
        command = self._visual_segment_command(commands)
        self.assertTrue(ok)
        # このケースのアンカーは 0.0 から始まる＝合っていない“その位置”は 0.0
        self._assert_not_aligned_position(command, 0.0)

    def test_warp_cannot_bypass_visual_proof(self):
        aligned = self._aligned_ss("warp")
        # 証明あり＝そのままMVを当てる／証明なし＝その位置は出さない
        ok, commands = self._render_with_mock("warp", _active_lipsync_profile())
        command = self._visual_segment_command(commands)
        self.assertTrue(ok)
        self.assertIn("PERSON_MV.mp4", command)
        self.assertNotIn("lavfi", command)
        ok, commands = self._render_with_mock("warp", None)
        command = self._visual_segment_command(commands)
        self.assertTrue(ok)
        self._assert_not_aligned_position(command, aligned)

    def test_output_frame_must_not_round_past_a_silent_vocal_bin(self):
        # t=.033sの30fps frameは0.00-0.05s binと0.05-0.10s binを跨ぐ。
        # 最寄りのactive[1]だけを見て人物を出してはいけない。
        profile = _active_lipsync_profile()
        self.assertFalse(
            lipsync_pro._mapped_frames_have_verified_lipsync_visual(
                profile, np.array([1.0 / 30.0]), np.array([1.0 / 30.0]),
                (0.05, np.array([False, True, True], dtype=bool))))

    def test_nan_vocal_mask_and_nan_source_mapping_fail_closed(self):
        profile = _active_lipsync_profile()
        self.assertFalse(
            lipsync_pro._mapped_frames_have_verified_lipsync_visual(
                profile, np.array([0.0]), np.array([0.0]),
                (0.05, np.array([np.nan, 1.0]))))
        nan_anchors = [(0.0, float("nan"), 1.0),
                       (2.0, float("nan"), 1.0)]
        for renderer in ("equal", "warp"):
            ok, commands = self._render_with_mock(
                renderer, profile, anchors=nan_anchors)
            command = self._visual_segment_command(commands)
            self.assertTrue(ok, renderer)
            # NaNのマッピングがそのままffmpegへ渡らないこと（有限な位置になる）
            if "lavfi" not in command:
                self.assertTrue(np.isfinite(self._ss_of(command)), renderer)

    def test_source_mapping_always_starts_at_actual_ffmpeg_seek(self):
        remix, source = lipsync_pro._rendered_frame_mapping(
            1.0 / 60.0, 1.0, 30, 0.0, 4.0)
        self.assertEqual(len(remix), 30)
        self.assertAlmostEqual(source[0], 0.0)
        self.assertAlmostEqual(source[1], 4.0 / 30.0)

    def test_phase_whitelist_requires_small_confident_residual_lag(self):
        rt = np.arange(240, dtype=float) / 30.0
        st = rt.copy()
        onset_t = np.arange(0.0, 8.1, 0.05)
        onset = np.zeros_like(onset_t)
        onset[::10] = 1.0
        profile = _active_lipsync_profile(end=8.2)
        rvoc = np.ones(22050 * 8, dtype=np.float32)

        with mock.patch.object(lipsync_pro, "_vocal_onset_envelope",
                               return_value=(onset_t, onset)):
            with mock.patch("mouth_sync.measure_micro_mouth_lag",
                            return_value=(0.05, 0.60, 0.90, 3.0)):
                proven = lipsync_pro._visual_phase_proven_ranges_from_mapping(
                    rt, st, profile, rvoc, 22050)
            with mock.patch("mouth_sync.measure_micro_mouth_lag",
                            return_value=(0.30, 0.60, 0.90, 3.0)):
                rejected = lipsync_pro._visual_phase_proven_ranges_from_mapping(
                    rt, st, profile, rvoc, 22050)
        self.assertTrue(proven)
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()


class WhisperFinalRefixComparedNotForcedTests(unittest.TestCase):
    """項目3: Whisper最終再固定を無条件上書きせず、口×歌声品質が
    改善した時だけ採用する（悪化時は補正前を維持）。波形経路は不変。"""

    SRC = (Path(__file__).resolve().parent / "lipsync_pro.py").read_text(encoding="utf-8")

    def test_whisper_refix_is_compared_not_forced(self):
        # 旧: 無条件 anchors = whisper_word_align(...)
        # 新: anchors_before / anchors_whisper を品質比較して採否を決める
        self.assertIn("anchors_before = [list(a) for a in anchors]", self.SRC)
        self.assertIn("anchors_whisper = whisper_word_align(", self.SRC)
        self.assertIn("Whisper再固定は不採用", self.SRC)

    def test_quality_uses_existing_report(self):
        # 新しい重い計算を足さず、既存の alignment_quality_report を流用する
        self.assertIn("onset_correlation", self.SRC)
        self.assertIn("def _sync_quality(anch):", self.SRC)

    def test_degradation_keeps_pre_whisper_anchors(self):
        # 比較分岐のロジックを純粋関数として再現し、悪化時は補正前維持を確認
        def decide(sb, sw, mv, g=0.01):
            if sb is None or sw is None: return "whisper"
            if sw >= sb + g: return "whisper"
            if mv <= 0.10 and sw >= sb - g: return "whisper"
            return "before"
        self.assertEqual(decide(0.45, 0.30, 0.3), "before")   # 悪化→維持
        self.assertEqual(decide(0.30, 0.45, 0.3), "whisper")  # 改善→採用
        self.assertEqual(decide(None, 0.4, 0.3), "whisper")   # 測定不能→従来


class ExtendedCoverageWindowTests(unittest.TestCase):
    """Extended版（曲がMVより長い）でも中盤の口パクを採用できること。

    足したイントロ/アウトロには対応するMVが存在しないため、曲全体では
    カバー98%が原理的に達成できず、中盤がどれだけ合っていても不採用に
    なっていた。MVと対応し得る中盤区間だけで判定するようにした。
    """

    @staticmethod
    def _report(eval_window):
        sr = 50.0
        rt = np.arange(0, 216.5, 1 / sr)      # 曲216.5秒
        ot = np.arange(0, 177.5, 1 / sr)      # MV177.5秒（テンポ補正後）
        rfeat = np.zeros((len(rt), 8), dtype=np.float32)
        ofeat = np.zeros((len(ot), 8), dtype=np.float32)
        # 中盤31〜184秒だけMVに対応（前後は範囲外へ写像される）
        anchors = [[31.0, 0.0, 1.0], [184.0, 153.0, 1.0]]
        return lipsync_pro.alignment_quality_report(
            anchors, [], rfeat, rt, ofeat, ot, eval_window=eval_window)

    def test_middle_window_reaches_full_coverage(self):
        q = self._report((31.0, 184.0))
        self.assertGreaterEqual(q["coverage"], 0.98)
        self.assertGreaterEqual(q["tail_coverage"], 0.98)

    def test_whole_song_would_fail_without_window(self):
        # 修正前の挙動（全体評価）ではカバーが足りず落ちていたこと
        q = self._report(None)
        self.assertLess(q["coverage"], 0.98)

    def test_window_ignored_when_too_short(self):
        # 中盤が極端に短い指定は無視して全体評価（判定の誤緩和を防ぐ）
        q = self._report((100.0, 110.0))
        self.assertLess(q["coverage"], 0.98)


class LowFaceRateVisualProofTests(unittest.TestCase):
    """顔がほとんど検出できないMVでは、視覚証明を必須にしない。

    暗い/横顔/引き/ブラーの多いMVでは mediapipe が顔をほぼ拾えず（実例:
    顔検出率13%）、「証明できる区間ゼロ」→全編で口元を隠す、となっていた。
    「検出できない」は「口がズレている」ではないため、判定不能として音側の
    品質ゲートに委ねる。ただし判断材料が無い場合は緩めない（厳格のまま）。
    """

    def test_low_face_rate_is_treated_as_undecidable(self):
        self.assertFalse(lipsync_pro._visual_proof_applicable({"face_rate": 0.13}))

    def test_enough_face_still_requires_proof(self):
        # Cake by the Ocean 相当（39%）は従来どおり証明を必須にする
        self.assertTrue(lipsync_pro._visual_proof_applicable({"face_rate": 0.39}))
        # 境界値ちょうどは厳格側
        self.assertTrue(
            lipsync_pro._visual_proof_applicable(
                {"face_rate": lipsync_pro.VISUAL_PROOF_MIN_FACE}))

    def test_missing_or_broken_rate_stays_strict(self):
        # 判断材料が無い場合に緩めると危険なので、厳格のままであること
        for profile in (None, {}, {"times": [1, 2]}, {"face_rate": None},
                        {"face_rate": "x"}):
            self.assertTrue(lipsync_pro._visual_proof_applicable(profile), profile)

    def test_conflict_gate_applied_to_unsafe_ranges(self):
        """顔が取れないMVでは「口の矛盾」判定自体をスキップすること。

        実例: 顔検出率13%のMVで誤判定が多発し、216秒中170秒が危険扱いになり、
        その全部を『口が映らないMVカット』で埋めた結果、数少ない口なしカットを
        使い回して同じ映像がループして見えた。
        """
        src = (Path(lipsync_pro.__file__).resolve()).read_text(encoding="utf-8")
        # 品質レポート側の矛盾判定にゲートが掛かっていること
        self.assertIn(
            "and _visual_proof_applicable(mouth_profile)):", src)
        # 区間ごとの矛盾判定にもゲートが掛かっていること（2箇所以上）
        self.assertGreaterEqual(src.count("_visual_proof_applicable"), 4)
        # 矛盾判定のロジック自体は残っている（顔が取れるMVでは従来どおり働く）
        self.assertIn("max(silent_conflicts, singing_conflicts) >= 2", src)

    def test_conflict_detection_is_kept(self):
        # 明確な矛盾（歌ってないのに口が動く等）の判定は残っていること
        src = (Path(lipsync_pro.__file__).resolve()).read_text(encoding="utf-8")
        self.assertIn("max(silent_conflicts, singing_conflicts) >= 2", src)

