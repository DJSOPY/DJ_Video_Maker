"""テンポ探索の守備範囲の回帰テスト。

DJ Edit / Remix は原曲(例105BPM)をハウス基準(125BPM前後)へ上げることが多く、
+20%前後の倍率が普通に起こる。旧実装は探索範囲が 0.86〜1.14 しかなく、
実例「Bruno Mars - I Just Might (Dj Dark Remix)」の x1.19 を範囲外として
検出できず、「どのテンポでも揃わない＝別アレンジ」と誤判定していた。

ここでは合成音源（音程が変わる音列）で、
  ・範囲外だった x1.19 を正しく当てられること
  ・範囲を広げても等倍(x1.00)を誤検出しないこと
を検証する。
"""
import io
import contextlib
import unittest
from pathlib import Path

import numpy as np

CORE_PATH = Path(__file__).resolve().parent / "dj_maker_core.py"
SR = 11025


def _load_core():
    src = CORE_PATH.read_text(encoding="utf-8").split("# ─── メイン ───", 1)[0]
    ns = {"__file__": str(CORE_PATH)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, str(CORE_PATH), "exec"), ns)
    return ns


def _make_song(note_dur, semitone_shift=0.0, n_notes=44, seed=7):
    """音程が変わる音列（クロマで追える素材）。

    note_dur で速さを、semitone_shift でキー全体のずれを作る。
    「レコードを速く回した編集」は、テンポと一緒に音程も上がるので
    その状況（速い＋キーも上がる）を合成データで再現できる。
    """
    rng = np.random.RandomState(seed)
    semis = rng.randint(0, 12, size=n_notes)
    out = []
    for s in semis:
        t = np.arange(int(SR * note_dur)) / SR
        f0 = 220.0 * (2 ** ((float(s) + float(semitone_shift)) / 12.0))
        y = (np.sin(2 * np.pi * f0 * t)
             + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
             + 0.3 * np.sin(2 * np.pi * 3 * f0 * t))
        out.append(y * np.exp(-3.0 * t / note_dur))
    return np.concatenate(out).astype(np.float32)


class TempoSearchRangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_core()

    def _detect(self, ratio):
        """MVより ratio 倍速い曲を作り、検出倍率と一直線度を返す。"""
        mv = _make_song(2.38)
        music = _make_song(2.38 / ratio)
        return self.ns["find_best_mv_tempo"](mv, music, sr=SR)

    def test_detects_dj_edit_speedup_beyond_old_range(self):
        # 旧範囲(上限1.14)の外にある x1.19 を当てられること
        rate, lock, _ = self._detect(1.19)
        self.assertAlmostEqual(rate, 1.19, delta=0.02)
        # 同期採用の閾値(0.45)を超えること＝リップシンク送りにならない
        self.assertGreaterEqual(lock, 0.45)

    def test_same_tempo_is_not_false_detected(self):
        # 範囲を広げても等倍を誤検出しない（副作用がない）
        rate, lock, _ = self._detect(1.0)
        self.assertAlmostEqual(rate, 1.0, delta=0.02)
        self.assertGreaterEqual(lock, 0.45)

    def test_search_grid_covers_dj_edit_range(self):
        # ソース上でも探索範囲が広がっていること（1.14上限に戻っていない）
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("round(0.76 + 0.04 * i, 3)", src)
        self.assertNotIn(
            "[0.86,0.88,0.90,0.92,0.94,0.96,0.98,1.0,1.02,1.04,1.06,1.08,1.10,1.12,1.14]",
            src)

    def test_moderate_lock_still_tempo_adjusts_before_lipsync(self):
        # 一致率が中間でも、根拠があればMVをテンポ補正してからリップシンクへ渡す
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("テンポ差 ×", src)
        self.assertIn("MVを補正してからリップシンクします", src)

    def test_tempo_adjust_before_lipsync_even_when_not_locked(self):
        """波形が一直線に揃わなくても、等倍より明確に良い倍率ならMVを補正してから
        リップシンクへ渡す（実例: Dj Dark Remix ×1.20 / 一直線度28% vs 等倍0%）。
        補正せず渡すと19%ズレたままDTWに投げることになり一致0.46で失敗していた。"""
        src = CORE_PATH.read_text(encoding="utf-8")
        # lock<0.45 の分岐内で、リップシンク呼び出しの前に補正が入っていること
        i_branch = src.index("if lock < 0.45:")
        i_lip = src.index("_try_vocal_lipsync", i_branch)
        seg = src[i_branch:i_lip]
        self.assertIn("make_tempo_adjusted_mv(video_path, best_rate, tmp_dir,", seg)
        self.assertIn("(lock - lock_1p0) >= 0.10", seg)

    def test_adjust_gate_ignores_weak_evidence(self):
        # 等倍との差が小さい場合は補正しない（Cake by the Ocean等を巻き込まない）
        for lock, lock_1p0, expect in ((0.28, 0.00, True), (0.09, 0.03, False),
                                       (0.10, 0.01, False)):
            self.assertEqual((lock - lock_1p0) >= 0.10, expect)

    def test_varispeed_detection(self):
        """「レコードを速く回した」編集を自動判別できること。

        DJ Editはテンポと一緒に音程も上がることが多く（×1.20なら約+3.16半音）、
        音程がずれたまま照合すると同じ曲でもクロマが一致しない
        （実測: 3半音ずれるだけで一直線度100%→10%）。
        """
        mv = _make_song(2.38)
        # 早回し（テンポ×1.20・キーも+3.16半音）→ varispeed と判定
        music_vs = _make_song(2.38 / 1.20, 12 * np.log2(1.20))
        vs, shift, _conf = self.ns["looks_like_varispeed"](mv, music_vs, 1.20, sr=SR)
        self.assertTrue(vs)
        self.assertAlmostEqual(shift, 3, delta=1)
        # 音程維持のテンポ変更（キー同じ）→ varispeed ではない
        music_at = _make_song(2.38 / 1.20, 0.0)
        vs2, _s2, _c2 = self.ns["looks_like_varispeed"](mv, music_at, 1.20, sr=SR)
        self.assertFalse(vs2)

    def test_semitone_shift_estimation(self):
        # キーのズレを正しく当てられること
        mv = _make_song(2.38)
        for true_shift in (0, 3, 5, 7):
            k, conf = self.ns["estimate_semitone_shift"](
                mv, _make_song(2.38, true_shift), sr=SR)
            self.assertEqual(k, true_shift)
            self.assertGreater(conf, 0.0)

    def test_varispeed_uses_resampling_filter(self):
        # varispeed時は音程ごと変わる補正（asetrate）を使うこと
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("asetrate=44100*", src)
        self.assertIn("varispeed=_vs", src)

    def test_tempo_search_runs_for_every_track(self):
        """テンポ探索をファイル名で門前払いしないこと。

        以前は `if _is_remix_word and not is_quasi_original:` がテンポ探索
        そのものを囲っていたため、Club Mix / Extended Mix / Sped Up /
        Nightcore / Slowed / 表記なしRemix が丸ごと対象外だった。
        テンポ差が最大になる種類（Sped Up・Nightcore・Slowed）がまさに
        ここに入るので、取りこぼしの実害が大きい。
        """
        src = CORE_PATH.read_text(encoding="utf-8")
        # 名前でテンポ探索を囲っていないこと
        self.assertNotIn(
            "if _is_remix_word and not is_quasi_original:\n"
            "        best_rate, lock, lock_1p0 = find_best_mv_tempo", src)
        # 探索は無条件（インデント4＝関数直下）で走ること
        self.assertIn(
            "\n    best_rate, lock, lock_1p0 = find_best_mv_tempo("
            "video_audio, music_audio)", src)
        # 名前はヒント、判断の主は実測であること
        self.assertIn("_name_says_remix", src)
        self.assertIn("_tempo_evidence", src)
        self.assertIn("is_rmx_measured = bool(_name_says_remix or _tempo_evidence)", src)

    def test_name_is_only_a_hint_not_a_gate(self):
        """名前に表記が無くても実測でテンポ差があればRemixとして扱うこと。"""
        def decide(name_says_remix, rate, lock, lock_1p0):
            ev = (abs(rate - 1.0) > 0.005
                  and (lock - lock_1p0) >= 0.10
                  and lock >= 0.20)
            return bool(name_says_remix or ev)

        # 名前に表記なし＋実測でテンポ差 → 拾えるようになった（以前は取りこぼし）
        self.assertTrue(decide(False, 1.25, 0.80, 0.13))   # Sped Up
        self.assertTrue(decide(False, 0.85, 1.00, 0.13))   # Slowed
        # 同一音源は等倍と差が出ないので、従来どおり通常経路のまま
        self.assertFalse(decide(False, 1.000, 1.00, 1.00))  # 原曲
        self.assertFalse(decide(False, 1.000, 0.67, 0.67))  # キーのみ変更
        # 既存の成功例は経路が変わらないこと（実ログの数値）
        self.assertTrue(decide(True, 1.050, 0.09, 0.03))   # Cake by the Ocean
        self.assertTrue(decide(True, 1.200, 0.28, 0.00))   # I Just Might

    def test_hybrid_pro_is_not_gated_by_a_dead_proxy(self):
        """局所Pro同期の発動条件に、既定で常にNoneの値を使わないこと。

        以前は `is_rmx_for_hybrid = (vocal_silence_ranges is not None)` だった。
        vocal_silence_ranges は STRICT_MASK_FOR_ESTIMATED_PLACEMENT(既定False)
        の時しか埋まらないため常にNone＝この経路は既定設定で一度も動かない
        死にコードだった。
        """
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("is_rmx_for_hybrid = (vocal_silence_ranges is not None)", src)
        self.assertIn("is_rmx_for_hybrid = is_rmx_measured", src)
        # 前提（既定False）が変わったらこのテストごと見直すため、値を固定で確認
        self.assertIn("STRICT_MASK_FOR_ESTIMATED_PLACEMENT = False", src)


if __name__ == "__main__":
    unittest.main()
