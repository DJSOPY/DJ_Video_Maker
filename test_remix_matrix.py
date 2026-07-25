"""Remixパターン網羅テスト（これが本命の再発防止策）。

1曲ずつ壊れて1つずつ直す、を繰り返さないための検査。
DJが実際に持ち込むRemixのタイプを人工的に作り、
「どのタイプがどの経路に進むべきか」をまとめて検証する。

対応表（期待する動き）:
  同一音源のEdit / テンポ変更 / キー変更 / 早回し / テンポ+キー
      → 波形で配置できる（リップシンクに送らなくてよい）
  構成を並べ替えたRemix
      → 波形では配置できない＝リップシンク送りが正常
  別曲（無関係）
      → 同期しないのが正解。誤って同期＝MV取り違えの温床

何かを直すたびにこれを流せば、別のタイプを壊していないか一度に分かる。
"""
import io
import contextlib
import unittest
from pathlib import Path

import numpy as np

CORE_PATH = Path(__file__).resolve().parent / "dj_maker_core.py"
SR = 11025
NOTE_DUR = 2.38          # MV側の1音の長さ
LOCK_TH = 0.45           # これ以上なら「波形で配置できる」と判断される閾値


def _load_core():
    src = CORE_PATH.read_text(encoding="utf-8").split("# ─── メイン ───", 1)[0]
    ns = {"__file__": str(CORE_PATH)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, str(CORE_PATH), "exec"), ns)
    return ns


def _notes(n=48, seed=7):
    return np.random.RandomState(seed).randint(0, 12, size=n)


def _render(notes, note_dur, semitone_shift=0.0):
    """音列を鳴らす。note_dur で速さ、semitone_shift でキーを変える。"""
    out = []
    for k in notes:
        t = np.arange(int(SR * note_dur)) / SR
        f0 = 220.0 * (2 ** ((float(k) + float(semitone_shift)) / 12.0))
        y = (np.sin(2 * np.pi * f0 * t)
             + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
             + 0.3 * np.sin(2 * np.pi * 3 * f0 * t))
        out.append(y * np.exp(-3.0 * t / note_dur))
    return np.concatenate(out).astype(np.float32)


def _rearranged(notes, note_dur, semitone_shift=0.0, seed=3):
    """構成を並べ替えたRemix（サビ始まり等）を作る。"""
    blocks = [notes[i:i + 8] for i in range(0, len(notes), 8)]
    rng = np.random.RandomState(seed)
    rng.shuffle(blocks)
    return _render(np.concatenate(blocks), note_dur, semitone_shift)


class RemixPatternMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_core()
        cls.notes = _notes()
        cls.mv = _render(cls.notes, NOTE_DUR)

    def _detect(self, music):
        return self.ns["find_best_mv_tempo"](self.mv, music, sr=SR)

    # ---- 波形で配置できるべきタイプ ----

    def test_same_source_edit(self):
        rate, lock, _ = self._detect(_render(self.notes, NOTE_DUR))
        self.assertGreaterEqual(lock, LOCK_TH)
        self.assertAlmostEqual(rate, 1.0, delta=0.03)

    def test_tempo_changes(self):
        # DJ Editでよくある速度変更（遅くする/速くする、大小)
        for r in (0.80, 0.92, 1.08, 1.20, 1.35):
            rate, lock, _ = self._detect(_render(self.notes, NOTE_DUR / r))
            self.assertGreaterEqual(lock, LOCK_TH, f"×{r}")
            self.assertAlmostEqual(rate, r, delta=0.03, msg=f"×{r}")

    def test_key_changes_only(self):
        # キーだけ上げ下げしたRemix（テンポは同じ）
        for k in (-3.0, -1.0, 1.0, 2.0, 4.0):
            rate, lock, _ = self._detect(_render(self.notes, NOTE_DUR, k))
            self.assertGreaterEqual(lock, LOCK_TH, f"{k:+.0f}半音")
            self.assertAlmostEqual(rate, 1.0, delta=0.03, msg=f"{k:+.0f}半音")

    def test_varispeed_edits(self):
        # レコードを速く/遅く回した編集（テンポとキーが連動）
        for r in (0.90, 1.20):
            music = _render(self.notes, NOTE_DUR / r, 12 * np.log2(r))
            rate, lock, _ = self._detect(music)
            self.assertGreaterEqual(lock, LOCK_TH, f"×{r}")
            self.assertAlmostEqual(rate, r, delta=0.03, msg=f"×{r}")
            vs, _shift, _conf = self.ns["looks_like_varispeed"](
                self.mv, music, rate, sr=SR)
            self.assertTrue(vs, f"×{r} は早回しと判定されるべき")

    def test_tempo_and_independent_key_change(self):
        # テンポもキーも変えた（連動していない）Remix
        music = _render(self.notes, NOTE_DUR / 1.10, 4.0)
        rate, lock, _ = self._detect(music)
        self.assertGreaterEqual(lock, LOCK_TH)
        self.assertAlmostEqual(rate, 1.10, delta=0.03)

    # ---- 波形では配置できない（リップシンク送りが正常）なタイプ ----

    def test_rearranged_goes_to_lipsync(self):
        for note_dur, label in ((NOTE_DUR, "同テンポ"),
                                (NOTE_DUR / 1.15, "テンポ+15%")):
            _rate, lock, _ = self._detect(_rearranged(self.notes, note_dur))
            self.assertLess(lock, LOCK_TH, f"並べ替え({label})はリップシンク送り")

    # ---- 誤同期してはいけないタイプ ----

    def test_unrelated_song_is_not_matched(self):
        _rate, lock, _ = self._detect(_render(_notes(seed=99), NOTE_DUR))
        self.assertLess(lock, LOCK_TH, "別曲を波形で配置してはいけない")

    def test_unrelated_song_does_not_trigger_tempo_adjust(self):
        """別曲でリップシンク前のテンポ補正が誤発動しないこと。

        「等倍より明確に良い倍率」だけを条件にすると、無関係な曲でも
        等倍0%・別倍率14%のような差が出て、誤った倍率でMVを伸縮して
        しまう。最低限の一直線度も条件にして防ぐ。
        """
        min_lock = self.ns["TEMPO_ADJUST_MIN_LOCK"]
        rate, lock, lock_1p0 = self._detect(_render(_notes(seed=99), NOTE_DUR))
        fired = (abs(rate - 1.0) > 0.005
                 and (lock - lock_1p0) >= 0.10
                 and lock >= min_lock)
        self.assertFalse(fired, "別曲でテンポ補正が発動してはいけない")

    def test_real_tempo_difference_still_triggers_adjust(self):
        # 実機の I Just Might 相当（一直線度28% / 等倍0%）では発動すること
        min_lock = self.ns["TEMPO_ADJUST_MIN_LOCK"]
        lock, lock_1p0, rate = 0.28, 0.00, 1.20
        fired = (abs(rate - 1.0) > 0.005
                 and (lock - lock_1p0) >= 0.10
                 and lock >= min_lock)
        self.assertTrue(fired, "本物のテンポ違いでは補正が必要")


if __name__ == "__main__":
    unittest.main()
