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


def _make_song(note_dur, n_notes=44, seed=7):
    """音程が変わる音列（クロマで追える素材）。note_dur で速さを変える。"""
    rng = np.random.RandomState(seed)
    semis = rng.randint(0, 12, size=n_notes)
    out = []
    for s in semis:
        t = np.arange(int(SR * note_dur)) / SR
        f0 = 220.0 * (2 ** (s / 12.0))
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
        self.assertIn("make_tempo_adjusted_mv(video_path, best_rate, tmp_dir)", seg)
        self.assertIn("(lock - lock_1p0) >= 0.10", seg)

    def test_adjust_gate_ignores_weak_evidence(self):
        # 等倍との差が小さい場合は補正しない（Cake by the Ocean等を巻き込まない）
        for lock, lock_1p0, expect in ((0.28, 0.00, True), (0.09, 0.03, False),
                                       (0.10, 0.01, False)):
            self.assertEqual((lock - lock_1p0) >= 0.10, expect)


if __name__ == "__main__":
    unittest.main()
