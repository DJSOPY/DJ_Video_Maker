#!/usr/bin/env python3
"""Remixのfail-closed口元非表示経路の回帰テスト。"""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


CORE_PATH = Path(__file__).with_name("dj_maker_core.py")


def _load_core_prefix():
    # dj_maker_core.pyは対話CLIがtop-levelにあるため、メイン以前だけを読む。
    source = CORE_PATH.read_text(encoding="utf-8")
    prefix = source.split("# ─── メイン ───", 1)[0]
    ns = {"__file__": str(CORE_PATH)}
    exec(compile(prefix, str(CORE_PATH), "exec"), ns)
    return ns


CORE = _load_core_prefix()


def _video_probe(path):
    raw = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    return json.loads(raw)


class ProfileCertificationTests(unittest.TestCase):
    def test_only_explicit_absent_with_complete_samples_passes(self):
        certify = CORE["_profile_certifies_no_mouth"]
        self.assertTrue(certify({"mouth_absent": np.ones(30)}, 30))

    def test_detection_unknown_and_legacy_profiles_fail(self):
        certify = CORE["_profile_certifies_no_mouth"]
        self.assertFalse(certify({"face": np.zeros(30)}, 30))
        self.assertFalse(certify({"mouth_visible": np.zeros(30)}, 30))
        self.assertFalse(certify({"mouth_absent": np.r_[np.ones(29), np.nan]}, 30))
        self.assertFalse(certify({"mouth_absent": np.ones(29)}, 30))
        self.assertFalse(certify({"mouth_absent": np.r_[np.ones(29), 0]}, 30))


class UnsafePlanTests(unittest.TestCase):
    def test_absolute_mode_disables_real_source_broll(self):
        self.assertFalse(CORE["_ALLOW_REAL_SOURCE_SAFE_BROLL"])

    def test_waveform_first_is_default_and_strict_mode_is_available(self):
        # 既定は波形同期ファースト（元祖版の順序）。Trueにすると発音証明
        # ファーストの最厳格モードへ切り替わる（分岐自体は保持されている）。
        self.assertFalse(CORE["REQUIRE_VOCAL_PROOF_FOR_VISIBLE_FACES"])

    def test_hybrid_context_clip_is_not_advertised_as_final_output(self):
        # 回帰: Web UIはログの「✅ 完成: *.mp4」をダウンロード一覧に載せる。
        # ハイブリッド局所Proの作業用クリップ(_pro_context)がその書式で
        # 印字されると中間ファイルが一覧に混入する。
        pro_src = CORE_PATH.with_name("lipsync_pro.py").read_text(encoding="utf-8")
        self.assertIn('if "_pro_context" in Path(out_path).name:', pro_src)
        self.assertIn("Pro区間クリップ生成", pro_src)

    def test_identity_plan_masks_only_sustained_vocal_silence(self):
        # 回帰: min_silence=0.02 は息継ぎ(0.2〜0.5秒)を全部隠して波形プランを
        # 細断し、同一音源のEditまでリップシンク/フィラー行きにしていた。
        # 同一音源(same_source)は持続無声(≥1.6秒)のみ、推定配置は≥0.8秒のみ検出する。
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("min_silence=0.02", src)
        self.assertIn("_mask_min_silence = 1.6 if same_source else 0.8", src)
        # Demucs不可の環境でも、同一音源なら波形プランを潰さない(音の同一性が証明)
        self.assertIn("clean vocal確認は省略（波形厳密一致＝同一音源のため", src)


    def test_clean_vocal_silence_masks_even_a_mapped_waveform_interval(self):
        got = CORE["_expand_unsafe_plan_ranges"](
            [(0.0, 4.0, 10.0)], 4.0, pad=0.25, min_show=0.75,
            extra_unsafe_ranges=[(1.0, 2.0)])
        self.assertEqual(got, [
            (0.0, 0.75, 10.0),
            (0.75, 2.25, None),
            (2.25, 4.0, 12.25),
        ])

    def test_padding_and_mv_offset_are_preserved(self):
        plan = [(0.0, 2.0, 10.0), (2.0, 4.0, None), (4.0, 8.0, 14.0)]
        got = CORE["_expand_unsafe_plan_ranges"](
            plan, 8.0, pad=0.25, min_show=0.75)
        self.assertEqual(got, [
            (0.0, 1.75, 10.0),
            (1.75, 4.25, None),
            (4.25, 8.0, 14.25),
        ])

    def test_tiny_show_is_removed(self):
        plan = [(0.0, 1.0, None), (1.0, 1.5, 4.0), (1.5, 3.0, None)]
        got = CORE["_expand_unsafe_plan_ranges"](
            plan, 3.0, pad=0.0, min_show=0.75)
        self.assertEqual(got, [(0.0, 3.0, None)])

    def test_cumulative_frame_budget_avoids_rounding_drift(self):
        count = CORE["_hybrid_segment_frame_count"]
        # 1.75秒を単独丸めすると52.5→実装/ffmpeg次第で53になり得るが、
        # 全体境界の差ではこの区間の担当枚数が一意になる。
        self.assertEqual(count(0.0, 1.75), 52)
        self.assertEqual(count(1.75, 3.50), 53)
        self.assertEqual(count(0.0, 3.50), 105)

    def test_subframe_boundaries_assign_only_the_global_budget(self):
        count = CORE["_hybrid_segment_frame_count"]
        ranges = [(0.0, 0.01), (0.01, 0.025), (0.025, 0.05)]
        assigned = [count(a, b) for a, b in ranges]
        self.assertEqual(assigned, [0, 1, 1])
        self.assertEqual(sum(assigned), count(0.0, 0.05))


class WaveformFirstOrderTests(unittest.TestCase):
    """元祖版の絶対順序『①波形合わせ → ②微調整 → ③無理ならリップシンク』を固定する。

    この順序・構成要素がリファクタで崩れたらここで落ちる。
    ソース構造のテストなので文言マーカーに依存する（意図的）。
    """

    SRC = CORE_PATH.read_text(encoding="utf-8")

    def _idx(self, marker):
        self.assertIn(marker, self.SRC, f"マーカー消失: {marker}")
        return self.SRC.index(marker)

    def test_stage1_waveform_alignment_comes_first(self):
        # 証明ファースト分岐は存在してよいが、既定Falseで波形節が先頭で実行される
        i_flag = self._idx("if REQUIRE_VOCAL_PROOF_FOR_VISIBLE_FACES:")
        i_wf = self._idx("波形ファースト：edit / Remix / 原曲すべて、まず波形でMVに合わせる")
        i_plan = self._idx("waveform_track_plan(\n        music_audio, video_audio")
        self.assertLess(i_flag, i_wf)
        self.assertLess(i_wf, i_plan)

    def test_stage2_fine_adjustments_are_inside_waveform_path(self):
        # 微調整: テンポ補正MV・波形オフセット精密化・末尾ズレの内部伸縮
        i_wf = self._idx("波形ファースト：edit / Remix / 原曲すべて、まず波形でMVに合わせる")
        i_tempo_adj = self._idx("make_tempo_adjusted_mv(video_path, best_rate, tmp_dir)")
        i_stretch = self._idx("内部で×{r:.3f} 伸縮して補正")
        self.assertLess(i_wf, i_tempo_adj)
        self.assertLess(i_wf, i_stretch)
        self.assertIn("refine_offset_waveform(video_audio, music_audio", self.SRC)

    def test_stage3_lipsync_is_fallback_only(self):
        # リップシンクへ行くのは『波形が揃わない』時だけ:
        #   テンポ不揃い(lock不足) / クロマ・音内容アラインでも合わない時
        i_plan = self._idx("waveform_track_plan(\n        music_audio, video_audio")
        i_tempo_fb = self._idx("どのテンポでも波形が一直線に揃わない（別アレンジ）→ リップシンクに切替")
        i_chroma = self._idx("→ クロマ（メロディ）で合わせ直します")
        i_chroma_fb = self._idx("クロマでも合わない（{cconf:.2f}）→ リップシンクに切替")
        self.assertLess(i_chroma, i_chroma_fb)
        # クロマ経路（一致率表示つき）が波形プランの後段にあること
        self.assertLess(i_plan, i_chroma)
        self.assertIn("クロマで一致（スコア{cconf:.2f} / 一致率{cmatch*100:.0f}% / 一意性{cuniq:.2f}）",
                      self.SRC)
        # 波形ファースト節より前に無条件のリップシンク実行が無いこと
        #（証明ファースト分岐の内側は except: フラグFalseで到達しない）
        head = self.SRC[:self._idx("波形ファースト：edit / Remix / 原曲すべて、まず波形でMVに合わせる")]
        flag_branch_start = head.index("if REQUIRE_VOCAL_PROOF_FOR_VISIBLE_FACES:")
        before_flag = head[:flag_branch_start]
        self.assertNotIn("_try_vocal_lipsync(\n                music_path", before_flag)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "ffmpeg/ffprobeが必要")
class SafeBackgroundIntegrationTests(unittest.TestCase):
    def _make_audio(self, path, duration=2.03):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
             "sine=frequency=440:sample_rate=44100", "-t", str(duration),
             str(path)], check=True)

    def test_abstract_background_has_exact_frames_and_no_audio(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "abstract.mp4"
            self.assertTrue(CORE["make_abstract_no_mouth_segment"](
                2.0, out, frame_count=60))
            info = _video_probe(out)
            self.assertEqual(info["streams"][0]["r_frame_rate"], "30/1")
            self.assertEqual(info["streams"][0]["nb_read_frames"], "60")
            audio = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=index", "-of", "csv=p=0", str(out)],
                check=True, capture_output=True, text=True).stdout.strip()
            self.assertEqual(audio, "")
            # 回帰: 真っ黒(YAVG=16)の“壊れて見える”背景に戻らないこと。
            # gradients対応ffmpegなら十分な明るさが出る(非対応環境は黒退避を許容)。
            stats = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(out), "-frames:v", "1",
                 "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-",
                 "-f", "null", "-"], capture_output=True, text=True).stdout
            m = re.search(r"YAVG=([0-9.]+)", stats)
            probe = subprocess.run(
                ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                 "gradients=s=64x36:r=30", "-frames:v", "1", "-f", "null", "-"],
                capture_output=True, text=True)
            if m and probe.returncode == 0:
                self.assertGreater(float(m.group(1)), 25.0,
                                   "安全背景が真っ黒に退行")

    def test_missing_or_uncertified_source_falls_back_to_background(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "fallback.mp4"
            self.assertTrue(CORE["make_safe_no_mouth_filler_segment"](
                Path(td) / "missing.mp4", 1.1, out, Path(td), frame_count=33))
            info = _video_probe(out)
            self.assertEqual(info["streams"][0]["nb_read_frames"], "33")

    def test_full_safe_fallback_overwrites_stale_output_and_has_av(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            audio = td / "music.wav"
            out = td / "result.mp4"
            self._make_audio(audio)
            out.write_bytes(b"stale previous person video")

            self.assertTrue(CORE["make_plain_mv_sync"](
                td / "missing_mv.mp4", audio, out, td,
                safe_no_mouth=True))
            info = _video_probe(out)
            self.assertEqual(info["streams"][0]["nb_read_frames"], "61")
            streams = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "stream=codec_type", "-of", "csv=p=0", str(out)],
                check=True, capture_output=True, text=True).stdout.split()
            self.assertEqual(set(streams), {"video", "audio"})

    def test_failed_safe_fallback_removes_stale_output(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            audio = td / "music.wav"
            out = td / "result.mp4"
            self._make_audio(audio, duration=0.5)
            out.write_bytes(b"stale previous person video")
            old = CORE["make_safe_no_mouth_filler_segment"]
            CORE["make_safe_no_mouth_filler_segment"] = lambda *a, **k: False
            try:
                with self.assertRaises(RuntimeError):
                    CORE["make_plain_mv_sync"](
                        td / "missing_mv.mp4", audio, out, td,
                        safe_no_mouth=True)
            finally:
                CORE["make_safe_no_mouth_filler_segment"] = old
            self.assertFalse(out.exists())

    def test_aligned_segment_filter_outputs_exact_cumulative_frames(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "src.mp4"
            out = td / "aligned.mp4"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                 "color=c=blue:s=320x180:r=24000/1001", "-t", "2.0",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
                check=True)
            frames = CORE["_hybrid_segment_frame_count"](0.0, 1.75)
            vf = CORE["_hybrid_exact_frame_filter"](CORE["VF_NORM"], frames)
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(src),
                 "-t", "1.75", "-vf", vf, "-frames:v", str(frames),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
                check=True)
            info = _video_probe(out)
            self.assertEqual(info["streams"][0]["r_frame_rate"], "30/1")
            self.assertEqual(info["streams"][0]["nb_read_frames"], "52")


if __name__ == "__main__":
    unittest.main()
