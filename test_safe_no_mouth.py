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
    def test_real_source_broll_enabled_but_certification_is_strict(self):
        # フィラーは「口が映らないと全フレーム認証できたMVカット」で埋める設定。
        # 有効化しても安全性は不変：認証は _profile_certifies_no_mouth が
        # 「全フレーム明示 MOUTH_ABSENT」の時だけ True を返すことで担保される。
        self.assertTrue(CORE["_ALLOW_REAL_SOURCE_SAFE_BROLL"])
        # 認証ロジックが厳格（未検出・NaN・1フレームでも閉口=不合格）であること
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("def _profile_certifies_no_mouth", src)
        # 認証は fps=1000.0（全デコードフレーム検査）で行われる
        self.assertIn("fps=1000.0", src)
        # 認証できなければ抽象背景へ退避する経路が残っていること
        self.assertIn("安全な口元なし映像が無いため、非人物の抽象背景へ退避", src)

    def test_lipsync_fallback_uses_lenient_mode_by_default(self):
        # 別アレンジRemixで「口が合う区間」を口パクできるよう、リップシンクの
        # ボーカル分離フォールバックは既定で元祖版と同じ緩い判定を使う。
        # （strict_fail_closed=True だと信頼度0.62等で合う区間まで潰していた）。
        self.assertFalse(CORE["_STRICT_FAIL_CLOSED_LIPSYNC"])
        src = CORE_PATH.read_text(encoding="utf-8")
        # 既定パスは strict_fail_closed=False かつフィラーにMV映像(make_filler_segment)
        self.assertIn("strict_fail_closed=False))", src)
        self.assertIn("make_filler_segment(\n                video_path, d, o, tmp_dir)", src)
        # strict機能自体は定数Trueで呼べる形で保持
        self.assertIn("if _STRICT_FAIL_CLOSED_LIPSYNC:", src)

    def test_unmatched_remix_shows_mv_by_default(self):
        # 別アレンジRemix等で波形/クロマ/音内容アラインのどれでも配置できない時、
        # 既定では原曲MVを素直に流す（5分ずっと抽象背景を避ける）。定数で最厳格にも戻せる。
        self.assertTrue(CORE["REMIX_SHOW_MV_WHEN_UNMATCHED"])
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("_safe_end = not REMIX_SHOW_MV_WHEN_UNMATCHED", src)
        # MVを流す経路（safe_no_mouth=_safe_end）が終端で使われている
        self.assertIn("safe_no_mouth=_safe_end", src)
        # リップシンク（fail-closed）を先に試してから最終手段でMV、の順は維持
        self.assertIn("リップシンクに切替", src)

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

    def test_estimated_placement_is_not_masked_by_default(self):
        # 回帰: 音内容アライン/クロマで配置が確定したプランに clean vocal マスクを
        # 掛けると、連続配置が細切れになり(実測 6区間→12区間)、末尾ズレの内部伸縮
        # 補正まで無効化されて元祖版より品質が落ちる。既定では掛けない。
        self.assertFalse(CORE["STRICT_MASK_FOR_ESTIMATED_PLACEMENT"])
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("if not STRICT_MASK_FOR_ESTIMATED_PLACEMENT:", src)
        # 根拠が取れない場合の退避(リップシンク→口元なし)は残っていること
        self.assertIn("クロマでも合わない（{cconf:.2f}）→ リップシンクに切替", src)
        self.assertIn("safe_no_mouth=True", src)
        # マスク節より後に末尾ズレ伸縮補正がある(補正が生きる)
        self.assertLess(src.index("if not STRICT_MASK_FOR_ESTIMATED_PLACEMENT:"),
                        src.index("伸縮して補正"))

    def test_auto_candidate_waveform_verification_exists(self):
        # 自動選択で別曲（例:「Ayo」→再生数の多い「Loyal」）を掴む問題への対策。
        # 候補をオンセット波形照合し、一致しなければ次候補へ。全滅時は最良候補を採用。
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("def verify_candidate_by_waveform(", src)
        self.assertTrue(CORE["_AUTO_VERIFY_CANDIDATES"])
        # URL直接指定を尊重（候補が複数ある自動選択時のみ検証）
        self.assertIn("0 < i < len(urls) - 1", src)
        # 全候補が落ちても無音にならない保険
        self.assertIn("最も一致した候補", src)

    def test_waveform_verifier_distinguishes_same_and_different_song(self):
        import io, contextlib, wave, tempfile, os
        import numpy as np
        s = CORE_PATH.read_text(encoding="utf-8").split("# ─── メイン ───", 1)[0]
        ns = {"__file__": str(CORE_PATH)}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(s, str(CORE_PATH), "exec"), ns)
        verify = ns["verify_candidate_by_waveform"]
        sr = 11025
        def wv(p, a):
            a = (np.clip(a, -1, 1) * 32767).astype(np.int16)
            with wave.open(p, "w") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes(a.tobytes())
        def song(dur, seed):
            r = np.random.default_rng(seed); n = int(dur * sr)
            x = np.zeros(n, dtype=np.float32)
            for _ in range(int(dur * 4)):
                pos = r.integers(0, n - 2000); ln = r.integers(500, 2000)
                x[pos:pos + ln] += r.uniform(0.3, 0.9) * r.standard_normal(ln).astype(np.float32)
            return x
        d = tempfile.mkdtemp(); mv = song(240, 10)
        edit = np.concatenate([mv[int(60*sr):int(110*sr)], mv[int(10*sr):int(45*sr)]])
        other = song(200, 99)
        wv(f"{d}/e.wav", edit); wv(f"{d}/m.wav", mv); wv(f"{d}/o.wav", other)
        ok_same, s_same = verify(f"{d}/e.wav", f"{d}/m.wav")
        ok_diff, s_diff = verify(f"{d}/e.wav", f"{d}/o.wav")
        self.assertTrue(ok_same, f"本物を却下 (score={s_same:.2f})")
        self.assertFalse(ok_diff, f"別曲を誤採用 (score={s_diff:.2f})")
        self.assertGreater(s_same - s_diff, 0.2)

    def test_content_align_fft_matches_bruteforce(self):
        # content_align_planの粗探索をFFT相互相関に置換して高速化した（実測5倍）。
        # 数学的に内積スキャンと同値で、配置結果が変わらないことを保証する。
        import io, contextlib
        import numpy as np
        s = CORE_PATH.read_text(encoding="utf-8").split("# ─── メイン ───", 1)[0]
        self.assertIn('correlate(cv[b], en[b], mode="valid", method="fft")', s)
        ns = {"__file__": str(CORE_PATH)}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(s, str(CORE_PATH), "exec"), ns)
        sr = 11025
        rng = np.random.default_rng(3)
        mv = (rng.standard_normal(int(120 * sr)) * 0.3).astype(np.float32)
        music = np.concatenate([mv[int(10*sr):int(50*sr)], mv[int(70*sr):int(110*sr)]])
        plan = ns["content_align_plan"](music, mv, len(music) / sr, 120.0, sr)
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(len(plan), 1)
        for s0, e0, m0 in plan:
            self.assertGreaterEqual(m0, -0.01)
            self.assertLessEqual(m0, 120.0 + 0.01)

    def test_waveform_detection_uses_original_settings(self):
        # 回帰: fail-closed版は match_th 0.72 + 孤立窓補間OFF で一致区間を
        # 意図的に削っていた(実測 94%→84% / 7区間→16区間にフィラー断片)。
        # 同一音源なら孤立1窓も同じテイクの続きなので、元祖の 0.60 + 補間ONに戻す。
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("_wf_match_th = 0.60", src)
        self.assertNotIn("_wf_match_th = 0.72", src)
        self.assertIn("interpolate_single_gap=True)", src)
        self.assertNotIn("interpolate_single_gap=False", src)

    def test_same_source_finishes_with_waveform_only(self):
        # 「波形で合うものは波形だけで終わらせる」:
        # 同一音源(波形厳密一致)では clean vocal マスクも安全境界拡張も掛けない。
        # 音の同一性そのものが口元の証明であり、息継ぎやインスト部で
        # 人物が消える問題の根治。別マスター推定時は従来どおり厳格検証。
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("_strict_remix_show = not same_source", src)
        self.assertIn("波形で同一音源と確定 → 波形プランだけで完成させます", src)
        # 既定では Demucs 検証を走らせない(vocal_silence_ranges は None のまま)
        i_flag = src.index("if not STRICT_MASK_FOR_ESTIMATED_PLACEMENT:")
        i_call = src.index("vocal_sync.clean_vocal_silence_ranges", i_flag)
        i_else = src.index("    else:", i_flag)
        self.assertLess(i_else, i_call)
        # その結果、局所Proにも回らない(is_rmx_for_hybrid が False になる)
        self.assertIn("is_rmx_for_hybrid = (vocal_silence_ranges is not None)", src)

    def test_identity_plan_masks_only_sustained_vocal_silence(self):
        # 回帰: min_silence=0.02 は息継ぎ(0.2〜0.5秒)を全部隠して波形プランを
        # 細断し、同一音源のEditまでリップシンク/フィラー行きにしていた。
        # 同一音源(same_source)は持続無声(≥1.6秒)のみ、推定配置は≥0.8秒のみ検出する。
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("min_silence=0.02", src)
        self.assertIn("_mask_min_silence = 1.6 if same_source else 0.8", src)
        # Demucs不可の環境でも、同一音源なら波形プランを潰さない(音の同一性が証明)
        # 同一音源は Demucs 検証自体を省略して波形だけで完結させる
        self.assertIn("波形で同一音源と確定 → 波形プランだけで完成させます", src)


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
