"""MV自動選択の回帰テスト。

Ayo（Chris Brown ft. Tyga）を探して、同じアーティストで再生数が桁違いに
多い別曲 Loyal を掴んでしまう問題（実機で発生）を防ぐ。曲名がタイトルに
含まれる候補だけを再生数勝負に参加させるガードを検証する。
"""
import io
import contextlib
import unittest
from pathlib import Path

CORE_PATH = Path(__file__).resolve().parent / "dj_maker_core.py"


def _load_core_namespace():
    """dj_maker_core.py の対話CLI（# ─── メイン ───以降）より前だけを実行する。"""
    src = CORE_PATH.read_text(encoding="utf-8").split("# ─── メイン ───", 1)[0]
    ns = {"__file__": str(CORE_PATH)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, str(CORE_PATH), "exec"), ns)
    return ns


class MvSongNameGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_core_namespace()

    def _compact(self, s):
        return self.ns["_compact"](s)

    def _has_song_name(self, core_song, title):
        cs = self._compact(core_song)
        return bool(cs and len(cs) >= 3 and cs in self._compact(title))

    def test_ayo_does_not_pick_loyal(self):
        # 「Ayo」を探して、再生数15億のLoyalではなくAyoの公式MVを選ぶ
        core_song = "Ayo"
        cands = [
            {"title": "Chris Brown - Loyal (Official Video) ft. Lil Wayne, Tyga",
             "views": 1_575_689_148},
            {"title": "Chris Brown - Ayo (Official Video) ft. Tyga",
             "views": 45_000_000},
            {"title": "Chris Brown, Tyga - Ayo (Audio)", "views": 12_000_000},
        ]
        named = [c for c in cands if self._has_song_name(core_song, c["title"])]
        named.sort(key=lambda r: r["views"], reverse=True)
        self.assertTrue(named, "曲名を含む候補が1つ以上あるべき")
        self.assertIn("Ayo", named[0]["title"])
        # Loyalは曲名'ayo'を含まないので候補から外れている
        self.assertFalse(any("Loyal" in c["title"] for c in named))

    def test_multiword_song_name_kept(self):
        # 複数単語の曲名（原曲もRemixも）は正しく候補に残る
        core_song = "CAKE BY THE OCEAN"
        for title in ["DNCE - Cake By The Ocean",
                      "DNCE - Cake By The Ocean (JUMP SMOKERS RMX)"]:
            self.assertTrue(self._has_song_name(core_song, title), title)

    def test_short_song_name_is_not_checked(self):
        # 2文字以下の曲名はチェックをスキップ（従来動作＝誤爆しない）
        for core_song in ["Go", "17"]:
            cs = self._compact(core_song)
            self.assertLess(len(cs), 3)

    def test_guard_code_present_in_source(self):
        # ソースに曲名必須ガードが入っていること
        src = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("_has_song_name", src)
        self.assertIn("named = [r for r in plausible if _has_song_name(r)]", src)


if __name__ == "__main__":
    unittest.main()
