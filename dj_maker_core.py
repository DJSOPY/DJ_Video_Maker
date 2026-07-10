import subprocess, sys, os, tempfile, shutil, json, re
from pathlib import Path
import numpy as np
from scipy.signal import correlate, stft
from scipy.ndimage import median_filter

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))  # 同じフォルダの lipsync_pro.py / vocal_sync.py を確実に見つける
CONFIG_DIR = Path.home() / ".dj_video_maker"
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "config.json"

def load_config():
    if CONFIG_FILE.exists():
        try: return json.loads(CONFIG_FILE.read_text())
        except: return {}
    return {}

def save_config(config):
    CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2))

# ─── yt-dlp の YouTube ボット判定回避（ログイン済みブラウザのCookieを使う）───
# 全yt-dlp呼び出しに *YTDLP_COOKIE_ARGS を差し込む。mainで設定値を反映する。
YTDLP_COOKIE_ARGS = []

def detect_browsers():
    """macOSでインストール済みの主要ブラウザ名を返す（yt-dlp --cookies-from-browser 用）"""
    apps = [
        ("chrome",  "/Applications/Google Chrome.app"),
        ("brave",   "/Applications/Brave Browser.app"),
        ("edge",    "/Applications/Microsoft Edge.app"),
        ("firefox", "/Applications/Firefox.app"),
        ("safari",  "/Applications/Safari.app"),
    ]
    found = [name for name, p in apps if Path(p).exists()]
    if "safari" not in found:
        found.append("safari")  # Safariは標準搭載なので常に候補に
    return found

def cookie_args_from_config(config):
    b = config.get("cookies_browser")
    if b and b != "none":
        return ["--cookies-from-browser", b]
    return []

def configure_cookies(config):
    """初回のみ: YouTubeにログイン済みのブラウザを聞いて保存する。"""
    if "cookies_browser" in config:
        return cookie_args_from_config(config)
    print("\n🔐 YouTubeのボット判定を回避するため、ログイン済みブラウザのCookieを使います。")
    print("   （そのブラウザでYouTubeにログイン済みである必要があります）")
    found = detect_browsers()
    opts = found + ["none"]
    label = {"chrome":"Google Chrome","brave":"Brave","edge":"Microsoft Edge",
             "firefox":"Firefox","safari":"Safari","none":"使わない（Cookie無しで試す）"}
    for i, b in enumerate(opts, 1):
        rec = " ← おすすめ" if b == "chrome" else ""
        print(f"   {i}. {label.get(b,b)}{rec}")
    sel = input("  どのブラウザでYouTubeにログインしてますか？ 番号でEnter [Enter→1]: ").strip()
    idx = (int(sel) - 1) if (sel.isdigit() and 1 <= int(sel) <= len(opts)) else 0
    config["cookies_browser"] = opts[idx]
    save_config(config)
    print(f"  ✅ {label.get(opts[idx], opts[idx])} を使います（次回から自動・変更は ~/.dj_video_maker/config.json を削除）")
    return cookie_args_from_config(config)


def run(cmd, capture=True):
    r = subprocess.run(cmd, capture_output=capture, text=capture, errors="replace")
    if r.returncode != 0:
        err = r.stderr if capture else ''
        print(f"\n❌ 失敗: {' '.join(map(str,cmd))}\n{err}")
        sys.exit(1)
    return r

def get_duration(path):
    """コンテナ→ストリームの順で長さ取得。N/A対応。"""
    r = subprocess.run(["ffprobe","-v","quiet","-show_entries","format=duration",
             "-of","default=noprint_wrappers=1:nokey=1",str(path)],
             capture_output=True, text=True, errors="replace")
    val = r.stdout.strip()
    if val and val != "N/A":
        try: return float(val)
        except ValueError: pass
    # ストリーム側から取得を試みる
    r = subprocess.run(["ffprobe","-v","quiet","-show_entries","stream=duration",
             "-of","default=noprint_wrappers=1:nokey=1",str(path)],
             capture_output=True, text=True, errors="replace")
    for line in r.stdout.strip().splitlines():
        if line and line != "N/A":
            try: return float(line)
            except ValueError: continue
    # 最終手段: デコードして測る
    d = _decode_duration(path)
    if d > 0:
        return d
    print(f"\n❌ ファイルの長さを取得できません: {path}")
    sys.exit(1)

def _decode_duration(path):
    """実デコードで長さ測定。-progress pipe:1 はffmpegの全バージョンで機械可読。"""
    r = subprocess.run(
        ["ffmpeg","-nostats","-i",str(path),"-f","null","-progress","pipe:1","-"],
        capture_output=True, text=True, errors="replace")
    last_us = 0
    for line in r.stdout.splitlines():
        if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
            try:
                last_us = max(last_us, int(line.split("=")[1]))
            except ValueError:
                pass
    return last_us / 1_000_000

def get_audio_duration_accurate(path):
    """実際にデコードして正確な長さを測る（VBR MP3のヘッダ詐称対策）"""
    d = _decode_duration(path)
    if d > 0.5:
        return d
    # デコード測定が失敗したらコンテナ情報にフォールバック
    return get_duration(path)

def has_audio(path):
    r = subprocess.run(
        ["ffprobe","-v","quiet","-show_streams","-select_streams","a",
         "-of","default=noprint_wrappers=1:nokey=1",str(path)],
        capture_output=True, text=True, errors="replace")
    return bool(r.stdout.strip())

def get_metadata(music_path):
    r = subprocess.run(
        ["ffprobe","-v","quiet","-show_entries",
         "format_tags=title,artist,album_artist",
         "-of","default=noprint_wrappers=1",str(music_path)],
        capture_output=True, text=True, errors="replace")
    tags = {}
    for line in r.stdout.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            tags[k.lower().replace("tag:","").strip()] = v.strip()
    # 空白だけの無意味なタグ値は空にする（後段でファイル名にフォールバック）
    for k in list(tags.keys()):
        if not tags[k].strip():
            tags[k] = ""
    return tags

MASHUP_SYMBOLS = ["❌","✘","✖","×","╳"]

def is_mashup(music_path):
    import re
    tags = get_metadata(music_path)
    title  = tags.get("title", "")
    artist = tags.get("artist", tags.get("album_artist", ""))
    name   = music_path.stem.lower()
    for kw in ["mashup","mash-up","mash up","medley","メドレー","マッシュアップ"]:
        if kw in name or kw in title.lower():
            return True
    # 明確なマッシュアップ区切り記号（❌✘✖×╳）がファイル名/タイトルにある時だけ
    for sym in MASHUP_SYMBOLS:
        if sym in music_path.stem or sym in title:
            return True
    # 独立した "vs" / "VS"（前後がスペースや区切り）= マッシュアップ。
    # Elvis や vsop のような単語内のvsは拾わない。
    for text in [name, title.lower()]:
        if re.search(r"(?:^|[\s\-_．.])vs(?:[\s\-_．.]|$)", text):
            return True
    # "-to-"（ハイフン挟み）= マッシュアップ。例: SongA-to-SongB
    for text in [name, title.lower()]:
        if "-to-" in text:
            return True
    return False

def parse_mashup_songs(music_path):
    """ファイル名/タイトルから曲名リストを抽出（x や ❌ で分割）"""
    tags = get_metadata(music_path)
    text = tags.get("title", "") or music_path.stem

    # (Mashup)などの表記を除去
    text = re.sub(r'[\(\[\{].*?(mashup|mash-up|mash up|medley).*?[\)\]\}]', '', text, flags=re.I)
    text = re.sub(r'(mashup|mash-up|mash up|medley)', '', text, flags=re.I)

    # アンダースコア→スペース（先に変換して _x_ も分割可能に）
    text = text.replace("_", " ")
    # 区切り文字で分割: ❌ ✘ × x X vs VS
    for sym in MASHUP_SYMBOLS:
        text = text.replace(sym, "|")
    # " x " 単語境界のxのみ（曲名のxを壊さないように前後スペース必須）
    text = re.sub(r'\s+[xX]\s+', '|', text)
    text = re.sub(r'\s+vs\.?\s+', '|', text, flags=re.I)
    text = text.replace("-to-", "|")

    songs = [s.strip(" -_") for s in text.split("|")]
    songs = [s for s in songs if len(s) >= 2]
    return songs

REMIX_PATTERNS = [
    # カッコ内のremix系表記: (xxx Remix) [Club Mix] <HMC Bootleg> など（< >も対象）
    r'[\(\[\{<][^\)\]\}>]*(remix|bootleg|edit|extended|club mix|vip|flip|rework|refix|mix|version|ver\.|ver)[^\)\]\}>]*[\)\]\}>]',
    # カッコ無しの末尾表記: - xxx Remix など
    r'[-–—]\s*[^-–—]*(remix|bootleg|edit|extended mix|club mix|vip|flip|rework|refix)\s*$',
    # その他のノイズ表記（< >も対象）
    r'[\(\[\{<][^\)\]\}>]*(sped up|slowed|nightcore|bass boosted|8d audio|tiktok|lyrics|audio|hq|hd|2019|2020|2021|2022|2023|2024|2025|hmc)[^\)\]\}>]*[\)\]\}>]',
]

def clean_song_query(text):
    """Remix系の表記を取り除いて原曲名にする"""
    cleaned = text
    for pat in REMIX_PATTERNS:
        cleaned = re.sub(pat, '', cleaned, flags=re.I)
    # 残った < > [ ] 内も中身ごと除去（年号やレーベル名などのノイズ）
    cleaned = re.sub(r'[<\[\(\{][^>\]\)\}]*[>\]\)\}]', ' ', cleaned)
    # 単独の remix/bootleg などの単語も除去
    cleaned = re.sub(r'\b(remix|bootleg|rework|refix|flip|edit|extended|vip)\b', '', cleaned, flags=re.I)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(' -–—_')
    return cleaned if len(cleaned) >= 2 else text

def extract_core_title(artist, title, full_name):
    """
    検索の核となる「アーティスト名 + 原曲名」を抽出する。
    マッシュアップ/ブートレグの長い表記（A VS. B & C - 曲名 <Label Bootleg>）から
    実際の曲名部分だけを取り出す。
    戻り値: (core_artist, core_song)
    """
    raw = title if (title and title.strip()) else full_name
    # まず Remix/Bootleg 表記と括弧ノイズを全部除去（★等の記号もstrip_filename_noiseで除去）
    s = clean_song_query(strip_filename_noise(raw))
    # artistにも装飾記号が入りうるので除去
    if artist:
        artist = re.sub(r'[★☆◎●○■□▲▼◆◇♪♫➤➡⇒※\u2600-\u27BF\U0001F000-\U0001FAFF【】「」『』〈〉《》〔〕]', ' ', artist)
        artist = re.sub(r'\s+', ' ', artist).strip()
    # "ARTIST VS. ARTIST - SONG" や "ARTIST - SONG" の形なら、最後の " - " 以降が曲名
    core_song = s
    core_artist = artist
    if " - " in s or " – " in s:
        parts = re.split(r'\s[-–]\s', s)
        raw_parts = re.split(r'\s[-–]\s', strip_filename_noise(raw))
        # ★後ろ側に“裸の”Remix語がある（括弧の外）なら、それはリミキサー名で前側が曲名。
        #   例: 「前前前世 - Natino Remix」→ 前=曲名 / 「DNCE - CAKE <JUMP SMOKERS RMX>」→ 括弧内なので従来通り
        tail_raw = raw_parts[-1].strip() if len(raw_parts) == len(parts) else parts[-1].strip()
        tail_no_br = re.sub(r'[<\[\(\{][^>\]\)\}]*[>\]\)\}]', ' ', tail_raw)
        tail_was_remix = bool(re.search(
            r'\b(remix|rmx|bootleg|boot|edit|rework|refix|flip|vip|mashup)\b', tail_no_br, re.I))
        if tail_was_remix and len(parts) >= 2:
            core_song = parts[0].strip()          # 前側＝曲名
            # 後ろ側はリミキサー名なのでアーティストには使わない
        else:
            # 一番最後のパートを曲名候補に（複数アーティスト名の後ろが曲名のことが多い）
            core_song = parts[-1].strip()
            # 前半にアーティスト情報があれば、VS等を除いた先頭1組だけ使う
            head = parts[0].strip()
            head = re.split(r'\s+(?:vs\.?|x|×|&|feat\.?|ft\.?)\s+', head, flags=re.I)[0].strip()
            if not core_artist and head and len(head) < 30:
                core_artist = head
    # アーティスト名にVS/&/カンマ/feat等が入ってたら先頭1組だけ
    # （例: "Lady Gaga, Ellison Hard" → "Lady Gaga"。リミキサー名を検索から除く）
    if core_artist:
        core_artist = re.split(r'\s*(?:,|/|;|\bvs\.?\b|\bx\b|×|&|\bfeat\.?\b|\bft\.?\b|\bwith\b)\s*',
                               core_artist, flags=re.I)[0].strip()
    return core_artist, core_song.strip()

def strip_filename_noise(text):
    """
    ファイル名特有のノイズを除去して、検索しやすい曲名にする。
    - 先頭のトラック番号: "1-01 ", "01. ", "03_", "12 - " など
    - 拡張子・余計な記号
    - [ ] ( ) 内のタグ的表記（feat.除く）
    """
    t = text
    # 装飾記号を除去（★☆ などファイル整理用の印、絵文字、各種括弧記号）
    t = re.sub(r'[★☆◎●○■□▲▼◆◇♪♫➤➡⇒※\u2600-\u27BF\U0001F000-\U0001FAFF]', ' ', t)
    t = re.sub(r'[【】「」『』〈〉《》〔〕]', ' ', t)
    # 先頭トラック番号（ディスク-トラック / 番号. / 番号- / 番号_ / 番号空白）
    t = re.sub(r'^\s*\d{1,2}[-_.\s]\d{1,2}[\s_.\-]+', '', t)   # 1-01
    t = re.sub(r'^\s*\d{1,3}[\s_.\-]+', '', t)                  # 01. / 03_ / 12 -
    # 末尾/中間の括弧タグ（Audio, HD, MV, Lyric等）— feat系は残す
    def _strip_bracket(m):
        inside = m.group(1).lower()
        if any(k in inside for k in ["feat", "ft.", " featuring"]):
            return m.group(0)
        return ' '
    t = re.sub(r'[\(\[\{]([^\)\]\}]*)[\)\]\}]', _strip_bracket, t)
    # 区切り記号やアンダースコアを空白に
    t = t.replace("_", " ")
    t = re.sub(r'\s+', ' ', t).strip(' -–—_.')
    return t if len(t) >= 2 else text

def title_match_score(query, result_title):
    """検索クエリと結果タイトルの単語一致率（0〜1）"""
    stop = {"official","music","video","mv","the","a","an","of","feat","ft",
            "audio","lyric","lyrics","hd","hq","full","version","ver"}
    q_words = set(w.lower() for w in re.findall(r"[\w']+", query) if len(w) > 1) - stop
    t_words = set(w.lower() for w in re.findall(r"[\w']+", result_title) if len(w) > 1)
    if not q_words: return 0.0
    return len(q_words & t_words) / len(q_words)

def _compact(s):
    """英数字だけに圧縮（"UCHIDA 1" と "UCHIDA1" を同一視するため）"""
    return re.sub(r'[^a-z0-9]', '', (s or "").lower())

# 「別バージョン」を示す語（=自分の曲でなければ原曲MVを優先するため減点する）
OTHER_VERSION_MARKERS = (
    "remix","rmx","bootleg","rework","refix","flip","vip","mashup","mash-up",
    "nightcore","slowed","sped up","spedup","8d","instrumental","karaoke",
    "cover","visualizer","reverb",
    # 音を改変した再アップ（波形が一致しない＝避ける）
    "重低音","重低音強化","bass boost","bassboost","bass boosted","低音強化",
)

# ライブ/アコースティック等の"実演"動画：別歌唱・観客カットで口パク同期に不向き。
# 公式アップでも本物MVより必ず下げる（"Official Live"対策）。語境界で誤爆回避（alive/deliver等）。
PERFORMANCE_RE = re.compile(r"\b(live|acoustic|unplugged|in concert|concert|tour)\b", re.I)

# MV選定：タイトル妥当性のしきい値＆「本物MV」とみなす最低再生数
MV_TIER_MIN = 0.6          # この一致以上の候補は「タイトル妥当」→その中で再生数最多を本物MVとする
MV_MIN_TRUST_VIEWS = 50000 # 早期確定の再生数しきい値。これ未満は再アップ/偽公式疑い→他クエリも探す

def smart_search_mv(full_name, n=5, artist="", title=""):
    """
    複数の検索パターンを順に試し、タイトル一致率が最も高い結果を採用する。
    Remix/Bootlegの長い表記からは核心の曲名を抽出して検索する。
    戻り値: (results, used_query, used_remix_mv)
    """
    clean_title = strip_filename_noise(title) if (title and title.strip()) else ""
    clean_full  = strip_filename_noise(full_name)
    cleaned_orig = clean_song_query(clean_full)
    is_remix = (cleaned_orig != clean_full)

    # 核心の「アーティスト + 原曲名」を抽出（VS/Bootleg/年号などを除去）
    core_artist, core_song = extract_core_title(artist, title, full_name)

    # 試す検索クエリ（核心曲名を最優先）
    queries = []
    if core_song:
        if core_artist:
            queries.append(f"{core_artist} {core_song} official music video")
            queries.append(f"{core_artist} {core_song}")
        queries.append(f"{core_song} official music video")
        queries.append(f"{core_song}")
    # 補助: タグの曲名
    if clean_title and clean_title != core_song:
        queries.append(f"{clean_title}")
    # 最後の手段: 原曲名（Remix除去版）
    if cleaned_orig and cleaned_orig != core_song:
        queries.append(f"{cleaned_orig}")

    seen = set(); uniq = []
    for q in queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower()); uniq.append(q)

    # 一致判定の基準 = 核心曲名（これと一致しない結果は信用しない）
    match_ref = f"{core_artist} {core_song}".strip() if core_song else clean_full

    # 候補スコア: 核心一致 + 公式加点 − 別バージョン(他人のRemix/cover/live等)減点。
    #   ・ユーザーの“その曲の正確なRemix”MVが見つかればそれを最優先（+1.0）
    #   ・それ以外は「原曲MV（別バージョン語を含まない）」を優先する
    _cs = _compact(core_song) if core_song else ""
    def cand_score(t, uploader=""):
        base = title_match_score(match_ref, t)
        # スペース/桁つき表記の揺れ対策（"UCHIDA 1" ⇔ "UCHIDA1"）
        if _cs and len(_cs) >= 3 and _cs in _compact(t):
            base = max(base, 0.7)
        # ユーザーが指定した、その曲の正確なRemixのMVなら最優先
        if is_remix and title_match_score(clean_full, t) >= 0.6:
            return base + 1.0
        tl = t.lower()
        score = base
        is_official = any(k in tl for k in ("official", "music video", "official video", "公式"))
        if is_official:
            score += 0.4   # 公式アップを強く優先（重低音再アップ/海外ファン動画より上）
        # 実演動画（ライブ/アコースティック等）は別歌唱・観客カットで口パク不可。
        #   公式でも本物MVより必ず下げる（"Official Live" を主候補にしない）。
        if PERFORMANCE_RE.search(t):
            score -= 0.6
        # 別バージョン/リリック/Visualizer減点。ただし公式アップなら本物なので軽くする
        #   （例: "Our Style (Official Visualizer 歌詞付き)" は公式＝沈めない）
        if any(mk in tl for mk in OTHER_VERSION_MARKERS):
            score -= 0.15 if is_official else 0.5
        # 海外ファンの歌詞/翻訳re-up（テキスト被せ・本物でない）を減点
        if any(fk in tl for fk in ("가사", "번역", "한국어", "translation", "traducc",
                                   "traduç", "vietsub", "subtitulado", "letra", "แปล")):
            score -= 0.5
        # 音源のみアップロード（静止画ジャケ）を強く減点:
        #   ・"◯◯ - Topic" チャンネル＝YouTube自動生成のArt Track（映像なし）
        #   ・タイトルの (Official Audio) / (Audio) / Art Track 等
        up = (uploader or "").strip().lower()
        if up.endswith("- topic") or up.endswith("- topic'"):
            score -= 0.8
        if re.search(r"\b(official audio|audio only|art track|full album)\b", tl) \
           or re.search(r"[\(\[]\s*audio\s*[\)\]]", tl):
            score -= 0.5
        return score

    # --- 候補をクエリ横断でプールし、「タイトル妥当な中で再生数最多」を本物MVとして選ぶ ---
    #   official/music video をタイトルに詰めただけの低再生“偽公式”を、再生回数で見抜く。
    def _is_exact_remix(t):
        return is_remix and title_match_score(clean_full, t) >= 0.6

    def _pick(items):
        cands = list(items)
        if not cands:
            return [], -1.0
        # 1) ユーザー指定の正確なRemix MVが居れば最優先（再生数より上）
        exact = [r for r in cands if _is_exact_remix(r["title"])]
        if exact:
            exact.sort(key=lambda r: (cand_score(r["title"], r.get("uploader","")), r.get("views", 0)), reverse=True)
            return exact, cand_score(exact[0]["title"], exact[0].get("uploader",""))
        # 2) タイトルが十分妥当な候補の中で「再生回数が最多」＝本物の公式MV
        #    （詐称タイトルの低再生アップを、ここで再生数によって弾く）
        plausible = [r for r in cands if cand_score(r["title"], r.get("uploader","")) >= MV_TIER_MIN]
        if plausible:
            plausible.sort(key=lambda r: (r.get("views", 0), cand_score(r["title"], r.get("uploader",""))), reverse=True)
            return plausible, cand_score(plausible[0]["title"], plausible[0].get("uploader",""))
        # 3) 妥当な一致が無い → スコア最良で妥協
        cands.sort(key=lambda r: (cand_score(r["title"], r.get("uploader","")), r.get("views", 0)), reverse=True)
        return cands, cand_score(cands[0]["title"], cands[0].get("uploader",""))

    pool = {}
    best = None; best_query = uniq[0] if uniq else full_name
    for q in uniq[:6]:
        results = search_youtube_mv(q, n)
        if not results:
            continue
        for r in results:
            rid = r.get("id")
            if rid and rid not in pool:
                r["_q"] = q; pool[rid] = r
        picked, top_score = _pick(pool.values())
        if picked:
            best = picked; best_query = picked[0].get("_q", q)
            top_title = picked[0]["title"]
            # 早期確定：実演でなく、かつ「正確Remix or 十分な再生がある本物MV」が見つかったら打ち切り。
            #   低再生（偽公式の疑い）の時は止まらず、他クエリで本物を探し続ける。
            if top_score >= 0.6 and not PERFORMANCE_RE.search(top_title) and \
               (_is_exact_remix(top_title) or picked[0].get("views", 0) >= MV_MIN_TRUST_VIEWS):
                break

    # 一致率が低すぎる(誤検索)場合は、核心曲名そのものを最終クエリにして再検索
    top_now = cand_score(best[0]["title"], best[0].get("uploader","")) if best else -1.0
    if best is None or top_now < 0.34:
        fallback_q = core_song if core_song else clean_full
        results = search_youtube_mv(fallback_q, n)
        if results:
            for r in results:
                rid = r.get("id")
                if rid and rid not in pool:
                    r["_q"] = fallback_q; pool[rid] = r
            best, _ = _pick(pool.values())
            if best:
                best_query = best[0].get("_q", fallback_q)

    if best is None:
        return [], best_query, False
    used_remix = is_remix and (title_match_score(clean_full, best[0]["title"]) >= 0.6)
    return best, best_query, used_remix

def search_youtube_mv(query, n=3):
    # リリック動画を除外すると件数が減るので、多め(n+3)に取得してから絞る
    fetch_n = n + 3
    r = subprocess.run([
        "yt-dlp", *YTDLP_COOKIE_ARGS, f"ytsearch{fetch_n}:{query}",
        "--no-playlist",
        "--print", "%(title)s\t%(id)s\t%(duration)s\t%(view_count)s\t%(uploader)s",
        "--no-download"
    ], capture_output=True, text=True, errors="replace")
    results = []
    lyric_results = []
    for line in r.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            t = parts[0]
            tl = t.lower()
            try:
                vc = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            except Exception:
                vc = 0
            item = {"title": t, "id": parts[1],
                    "duration": parts[2] if len(parts) > 2 else "?",
                    "views": vc,
                    "uploader": parts[4] if len(parts) > 4 else ""}
            # リリック/歌詞付きは原則後回し（口パク不可）。ただし公式アップ
            # （"Official Visualizer 歌詞付き" 等）は本物なので主候補に残す。
            is_off = any(k in tl for k in ("official", "公式"))
            if (not is_off) and any(kw in tl for kw in ("lyric", "lyrics", "歌詞", "lyric video", "with lyrics")):
                lyric_results.append(item)
                continue
            results.append(item)
    # 非リリック候補が足りなければ、リリック/Visualizerも候補に足す（公式がそれだけの曲対策）
    if len(results) < n:
        results += lyric_results
    return results[:n]

def choose_video(music_path):
    """YouTube候補から動画を選んでURLを返す。Remixは候補選択を省いて自動採用。"""
    tags = get_metadata(music_path)
    title  = tags.get("title", "")
    artist = tags.get("artist", tags.get("album_artist", ""))

    # ---- URLモード（DJVM_MANUAL_URL=1 で起動時）: 検索せずURLを直接貼ってもらう ----
    if os.environ.get("DJVM_MANUAL_URL", "0") == "1":
        _disp = f"{artist} - {title}".strip(" -") or music_path.stem
        print(f"\n  🎵 曲: {_disp}")
        while True:
            u = input("  📋 この曲に使うYouTubeのURLを貼り付けてください:\n  > ").strip()
            if u.startswith("http") and ("youtube.com" in u or "youtu.be" in u):
                return [u]
            print("  ⚠️ YouTubeのURLではないようです（https://www.youtube.com/watch?v=... の形で貼ってください）")
    if title and title.strip():
        # タイトルタグがある → アーティスト + タイトル
        full_name = f"{artist} {title}".strip()
        print(f"\n  🎵 曲情報: {artist} - {title}")
    elif artist and artist.strip():
        # タイトルが無い → アーティスト + ファイル名（ファイル名が曲名のことが多い）
        full_name = f"{artist} {music_path.stem}".strip()
        print(f"\n  🎵 曲情報: {artist} - {strip_filename_noise(music_path.stem)}")
        # 以降の曲名抽出でもファイル名を使えるよう title にファイル名を入れる
        title = music_path.stem
    else:
        full_name = music_path.stem
        print(f"\n  🎵 ファイル名から検索: {strip_filename_noise(full_name)}")

    # Remix/Bootlegかどうか（候補を聞かず自動採用するかの判定にのみ使う）
    _raw_for_remix = (title or full_name)
    _cleaned_for_remix = clean_song_query(_raw_for_remix)
    is_remix = (_cleaned_for_remix != _raw_for_remix)

    print(f"  🔍 YouTube検索中...")
    results, used_query, used_remix = smart_search_mv(full_name, 5, artist=artist, title=title)
    print(f"     検索ワード: {used_query}")
    if not results:
        print("  ❌ 検索結果なし")
        return [input("  URLを手動入力:\n  > ").strip()]

    def _url(r): return f"https://www.youtube.com/watch?v={r['id']}"
    all_urls = [_url(r) for r in results]   # best-first（静止画なら順に次を試す）

    # Edit/Remixも自動で上位候補を採用（Enter不要）。候補は表示だけする。
    if is_remix:
        _v = results[0].get("views", 0)
        _vtxt = f"（再生{_v:,}回）" if _v else ""
        print(f"  ✅ 自動候補: {results[0]['title']} {_vtxt}")
        if _v and _v < 50000:
            print("  ⚠️ 再生回数が少なめです（本物の公式MVでない可能性）。")
        return all_urls

    # 通常曲も自動で上位候補を採用（Enter不要）。
    print(f"  ✅ 自動選択: {results[0]['title']} ({results[0]['duration']}秒)")
    return all_urls

def extract_mono_wav(src, dst, sr=11025, duration=None):
    cmd = ["ffmpeg","-y","-i",str(src)]
    if duration: cmd += ["-t",str(duration)]
    cmd += ["-ar",str(sr),"-ac","1","-f","wav",str(dst)]
    run(cmd)

def wav_to_array_path(path, sr=11025, duration=None):
    cmd = ["ffmpeg","-y","-i",str(path)]
    if duration: cmd += ["-t",str(duration)]
    cmd += ["-f","s16le","-ac","1","-ar",str(sr),"-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

# ============================================================
# オフセット検出（3手法）
# ============================================================

HOP = 512

# 全セグメント共通の映像規格（これを揃えないと結合時にズレる）
VF_NORM = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1"
ENC_ARGS = ["-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p","-video_track_timescale","15360"]

def compute_features(audio, sr=11025, hop=HOP):
    """オンセットエンベロープ + クロマグラム（12次元）"""
    nperseg = 2048
    f, t, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=nperseg-hop)
    mag = np.abs(Z)
    fps = sr / hop

    log_mag = np.log1p(mag * 100)
    flux = np.diff(log_mag, axis=1, prepend=log_mag[:, :1])
    onset = np.maximum(flux, 0).sum(axis=0)
    if onset.std() > 0:
        onset = (onset - onset.mean()) / onset.std()

    chroma = np.zeros((12, mag.shape[1]))
    for i, freq in enumerate(f):
        if freq < 60 or freq > 4000: continue
        midi = 69 + 12 * np.log2(freq / 440.0)
        chroma[int(round(midi)) % 12] += mag[i]
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1
    chroma = chroma / norms

    return onset, chroma, fps

def xcorr_norm(a, b):
    c = correlate(a, b, mode='full', method='fft')
    denom = np.sqrt((a**2).sum() * (b**2).sum()) + 1e-9
    return c / denom

def chroma_xcorr(cm, cv):
    """12次元クロマの2D相関（時間軸方向）"""
    total = None
    for b in range(12):
        c = correlate(cm[b], cv[b], mode='full', method='fft')
        total = c if total is None else total + c
    denom = np.sqrt((cm**2).sum() * (cv**2).sum()) + 1e-9
    return total / denom

def verify_alignment(cm, cv, lag_frames):
    """指定ラグで重ねた時の平均コサイン類似度（0〜1）= 検証スコア"""
    if lag_frames >= 0:
        m_seg = cm[:, lag_frames:]
        v_seg = cv
    else:
        m_seg = cm
        v_seg = cv[:, -lag_frames:]
    n = min(m_seg.shape[1], v_seg.shape[1])
    if n < 10: return 0.0
    sims = (m_seg[:, :n] * v_seg[:, :n]).sum(axis=0)
    return float(sims.mean())

def refine_offset_waveform(video_audio, music_audio, coarse_sec, sr=11025, window=1.0):
    """粗いオフセットの±window秒を波形相関で微調整（同一音源時にサンプル精度）"""
    try:
        n_v = min(len(video_audio), sr * 30)
        v = video_audio[:n_v]
        start = int(max(0, (coarse_sec - window) * sr))
        end   = int(min(len(music_audio), (coarse_sec + window) * sr + n_v))
        m = music_audio[start:end]
        if len(m) < n_v: return coarse_sec, 0.0
        c = correlate(m, v, mode='valid', method='fft')
        denom = np.sqrt((m**2).sum() * (v**2).sum()) + 1e-9
        peak = int(np.argmax(c))
        conf = float(c[peak] / denom)
        refined = (start + peak) / sr
        return refined, conf
    except Exception:
        return coarse_sec, 0.0

def find_offset_advanced(video_audio, music_audio, sr=11025):
    """
    改良版検出:
    1. オンセット相関 + クロマ2D相関で複数のラグ候補を生成
    2. 各候補をクロマ重ね合わせで検証し、最良を選ぶ
    3. 波形相関で微調整（同一音源ならサンプル精度に）
    戻り値: (オフセット秒, 検証スコア0-1, 手法名)
    """
    ov, cv, fps = compute_features(video_audio, sr)
    om, cm, _   = compute_features(music_audio, sr)
    nb = len(ov)

    candidates = set()
    c_onset = xcorr_norm(om, ov)
    for idx in np.argsort(c_onset)[-5:]:
        candidates.add(int(idx - (nb - 1)))
    c_chroma = chroma_xcorr(cm, cv)
    for idx in np.argsort(c_chroma)[-5:]:
        candidates.add(int(idx - (nb - 1)))

    best_lag, best_score = 0, -1.0
    for lag in candidates:
        score = verify_alignment(cm, cv, lag)
        if score > best_score:
            best_score, best_lag = score, lag

    coarse_sec = best_lag / fps

    # 波形微調整
    refined_sec, wf_conf = refine_offset_waveform(video_audio, music_audio, coarse_sec, sr)
    if wf_conf > 0.3 and abs(refined_sec - coarse_sec) < 1.5:
        return refined_sec, best_score, "波形精密"
    return coarse_sec, best_score, "クロマ検証"

SEG_WINDOW_SEC = 4.0
SEG_SCORE_TH   = 0.50

def align_segments(video_audio, music_audio, sr=11025, window_sec=SEG_WINDOW_SEC):
    """
    Edit/Remix対応の区間アライメント。
    曲を4秒窓に分割し、各窓がMVのどこに対応するか検索。
    連続する窓をグループ化してセグメントにする。
    戻り値: [(music_start, music_end, mv_start or None), ...]
      mv_start=None はMVに対応箇所なし（フィラー区間）
    """
    ov, cv, fps = compute_features(video_audio, sr)
    om, cm, _   = compute_features(music_audio, sr)
    win = max(8, int(window_sec * fps))
    n_win = max(1, int(np.ceil(cm.shape[1] / win)))

    def verify_at(seg, pos):
        F = seg.shape[1]
        if pos < 0 or pos + F > cv.shape[1]: return -1.0
        return float((cv[:, pos:pos+F] * seg).sum(axis=0).mean())

    def peak_uniqueness(total, pos, exclude):
        # その窓のMV対応が「一意か」を測る。反復曲（YMCA等）は別位置も同じくらい
        # 高く相関する→2位/1位が1に近い→一意性が低い。本物の一致は1位だけ尖る。
        if total.size <= 1:
            return 1.0
        best = float(total[pos])
        if best <= 0:
            return 0.0
        lo = max(0, pos - exclude); hi = min(total.size, pos + exclude + 1)
        masked = total.copy()
        masked[lo:hi] = -np.inf
        m = np.max(masked)
        second = float(m) if np.isfinite(m) else 0.0
        if second <= 0:
            return 1.0
        return max(0.0, 1.0 - second / best)

    # 各窓のMV対応位置を検索（連続性優先 + 平坦度ゲート）
    UNIFORM = 1.0 / np.sqrt(12)  # 完全に平坦なクロマの最大ビン値
    maps = []
    uniq_list = []
    prev_best = None
    for w in range(n_win):
        s = w * win
        e = min(s + win, cm.shape[1])
        seg = cm[:, s:e]
        F = seg.shape[1]
        if F < 8 or cv.shape[1] < F:
            maps.append((None, 0.0)); prev_best = None; continue

        # 平坦度ゲート: 音程的内容がない（ノイズ/無音）窓はフィラーへ
        peakiness = float(seg.max(axis=0).mean())
        if peakiness < UNIFORM * 1.35:
            maps.append((None, 0.0)); prev_best = None; continue

        total = None
        for b in range(12):
            c = correlate(cv[b], seg[b], mode='valid', method='fft')
            total = c if total is None else total + c
        pos = int(np.argmax(total))
        score = verify_at(seg, pos)
        uniq_list.append(peak_uniqueness(total, pos, win))

        # 連続性優先: 前の窓の続き位置でもほぼ同等に合うならジャンプしない
        if prev_best is not None:
            cont_pos = prev_best + win
            cont_score = verify_at(seg, cont_pos)
            if cont_score >= max(SEG_SCORE_TH, score * 0.88):
                pos, score = cont_pos, cont_score

        maps.append((pos, score))
        prev_best = pos if score >= SEG_SCORE_TH else None

    # グループ化: MV位置が連続している窓をまとめる
    tol = int(1.0 * fps)  # 1秒の許容ズレ
    segments = []
    cur_start_w = 0
    cur_mv_pos  = None  # グループ先頭のMVフレーム位置
    prev_pos    = None

    def close_group(end_w):
        nonlocal cur_start_w, cur_mv_pos
        m_start = cur_start_w * win / fps
        m_end   = end_w * win / fps
        mv_s    = (cur_mv_pos / fps) if cur_mv_pos is not None else None
        if m_end > m_start + 0.3:
            segments.append([m_start, m_end, mv_s])
        cur_start_w = end_w

    for w, (pos, score) in enumerate(maps):
        valid = (pos is not None and score >= SEG_SCORE_TH)
        if w == 0:
            cur_mv_pos = pos if valid else None
            prev_pos = pos if valid else None
            continue
        if valid and prev_pos is not None and abs(pos - (prev_pos + win)) <= tol:
            prev_pos = pos  # 連続 → グループ継続
        elif valid and cur_mv_pos is None and prev_pos is None:
            # フィラーから有効区間へ
            close_group(w)
            cur_mv_pos = pos
            prev_pos = pos
        elif valid:
            # ジャンプ（構成の切り替わり）→ グループを閉じて新規開始
            close_group(w)
            cur_mv_pos = pos
            prev_pos = pos
        else:
            # 無効窓
            if cur_mv_pos is not None:
                close_group(w)
                cur_mv_pos = None
            prev_pos = None

    close_group(n_win)

    # 末尾を音楽の実長に合わせる
    music_total = len(music_audio) / sr
    if segments:
        segments[-1][1] = music_total

    # 短すぎるMV区間（<3秒）はフィラー化（チラつき防止）
    for seg in segments:
        if seg[2] is not None and (seg[1] - seg[0]) < 3.0:
            seg[2] = None

    # 隣接フィラーを統合
    merged = []
    for seg in segments:
        if merged and merged[-1][2] is None and seg[2] is None:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    # 有効窓の平均一致スコア（同一音源=高い / 別アレンジRemix=低い）を信頼度として返す
    valid_scores = [sc for (p, sc) in maps if p is not None and sc >= SEG_SCORE_TH]
    avg_score = float(np.mean(valid_scores)) if valid_scores else 0.0
    # 一致の一意性（反復曲=低い）。スコアが高くても一意性が低ければ偽の一致。
    uniqueness = float(np.median(uniq_list)) if uniq_list else 0.0
    return [(s, e, mv) for s, e, mv in merged], avg_score, uniqueness

def content_align_plan(music_audio, video_audio, music_dur, vid_dur, sr=11025,
                       win_sec=4.0, hop_sec=1.0, slope_tol=0.25, min_run=4):
    """音内容アライン（6/24 fix_mv_align 方式）。
    クロマで edit↔MV のアンカー列を作り、slope≈1（1:1で前進）の連続ランだけを本体採用、
    反復で飛んだ窓は捨てる。各ランは中央値オフセットで頑健に配置。先頭の不安定域はイントロ。
    align_segments の窓ごと当て込みと違い『連続して前進している区間』しか使わないので、
    反復曲でもドリフト/末尾巻き戻りが出ない。戻り: seg_plan [(s,e,mv)] または None。"""
    try:
        om, cm, fps = compute_features(music_audio, sr)
        ov, cv, _   = compute_features(video_audio, sr)
    except Exception:
        return None
    W = int(win_sec * fps); step = max(1, int(hop_sec * fps))
    Te, To = cm.shape[1], cv.shape[1]
    if Te < W + step or To < W:
        return None
    anchors = []
    for s in range(0, max(1, Te - W), step):
        e = cm[:, s:s+W]; en = e / (np.linalg.norm(e) + 1e-9)
        best, bestj = -1.0, 0
        for j in range(0, max(1, To - W), 2):                 # 粗探索
            o = cv[:, j:j+W]; sim = float(np.sum(en * o)) / (np.linalg.norm(o) + 1e-9)
            if sim > best: best, bestj = sim, j
        for j in range(max(0, bestj-2), min(To - W, bestj+3)):  # 微調整
            o = cv[:, j:j+W]; sim = float(np.sum(en * o)) / (np.linalg.norm(o) + 1e-9)
            if sim > best: best, bestj = sim, j
        anchors.append(((s + W//2) / fps, (bestj + W//2) / fps, best))
    if len(anchors) < 2:
        return None
    # slope≈1 の連続ランを抽出
    runs = []; cur = [anchors[0]]
    for prev, a in zip(anchors, anchors[1:]):
        d_e = a[0] - prev[0]; d_m = a[1] - prev[1]
        ok = d_e > 0 and abs(d_m / d_e - 1.0) <= slope_tol
        if ok: cur.append(a)
        else: runs.append(cur); cur = [a]
    runs.append(cur)
    runs = [r for r in runs if len(r) >= min_run]
    if not runs:
        return None
    segs = []
    for r in runs:
        offs = np.array([mv - ed for ed, mv, _ in r])
        offset = float(np.median(offs))                       # 区間内は中央値オフセットで頑健
        e0 = r[0][0]; e1 = r[-1][0] + hop_sec
        segs.append([e0, e1, max(0.0, e0 + offset)])
    segs.sort(key=lambda s: s[0])
    intro_end = segs[0][0]
    for i in range(len(segs) - 1):
        segs[i][1] = segs[i+1][0]                             # 次の開始まで伸ばす（継ぎ目なめらか）
    segs[-1][1] = music_dur
    plan = []
    if intro_end > 0.3:
        lead = max(0.0, segs[0][2] - intro_end)              # イントロはMVを本体へ繋ぐように流す
        plan.append((0.0, intro_end, lead))
    for e0, e1, m0 in segs:
        plan.append((e0, e1, m0))
    return plan

def robust_linear_offset(music_audio, video_audio, sr=11025, win_sec=4.0, hop_sec=2.0):
    """同テンポ前提で、本編をMVに乗せる単一オフセット d（MV時刻 = 曲時刻 + d）を多数決で求める。
    反復で一部の窓が隣の繰り返しに飛んでも、多数の窓が真の d に集まるので頑健。
    戻り: (d秒, インライア率 0..1)。率が高いほど『同テンポ・並べ替え無し』で信頼できる。"""
    try:
        om, cm, fps = compute_features(music_audio, sr)
        ov, cv, _   = compute_features(video_audio, sr)
    except Exception:
        return 0.0, 0.0
    win = int(win_sec * fps); hop = max(1, int(hop_sec * fps))
    UNIFORM = 1.0 / np.sqrt(12)
    offs = []
    for s in range(0, max(1, cm.shape[1] - win), hop):
        seg = cm[:, s:s+win]
        if seg.shape[1] < win:
            break
        if float(seg.max(axis=0).mean()) < UNIFORM * 1.35:   # 無音/ノイズ窓は無視
            continue
        total = None
        for b in range(12):
            c = correlate(cv[b], seg[b], mode='valid', method='fft')
            total = c if total is None else total + c
        if total is None or total.size == 0:
            continue
        pos = int(np.argmax(total))
        offs.append(pos / fps - s / fps)
    if len(offs) < 4:
        return 0.0, 0.0
    offs = np.array(offs, dtype=np.float64)
    med = np.median(offs)
    inl = offs[np.abs(offs - med) < 1.5]      # 中央値付近に集まる窓＝正しいオフセット
    frac = inl.size / offs.size
    d = float(np.mean(inl)) if inl.size else float(med)
    return d, frac

def _refine_plan_offset_by_onset(seg_plan, music_audio, video_audio, sr=11025, max_shift=4.0):
    """クロマ配置後の“最後修正”。オンセット(拍)で全区間を同じだけ微調整して数秒ズレを詰める。
    拍は別マスター(リミックス)でも共有されるので有効。ズレ≈0なら変化なし。
    戻り: (補正後プラン, delta秒)。"""
    mv_segs = [(s, e, mv) for s, e, mv in seg_plan if mv is not None]
    if not mv_segs:
        return seg_plan, 0.0
    try:
        om, _, fps = compute_features(music_audio, sr)
        ov, _, _   = compute_features(video_audio, sr)
    except Exception:
        return seg_plan, 0.0
    vid_dur = ov.size / fps
    # 最長の有効区間で代表的にオフセットを測る（拍が一番効く）
    s, e, mv = max(mv_segs, key=lambda x: x[1] - x[0])
    a0 = int(s * fps); a1 = int(e * fps)
    seg_on = om[a0:a1]
    if seg_on.size < int(2 * fps):
        return seg_plan, 0.0
    center = int(mv * fps)
    half = int(max_shift * fps)
    lo = max(0, center - half)
    hi = min(ov.size - seg_on.size, center + half)
    if hi <= lo:
        return seg_plan, 0.0
    best = -1e18; blag = center
    for pos in range(lo, hi + 1):
        sc = float((ov[pos:pos + seg_on.size] * seg_on).sum())
        if sc > best:
            best = sc; blag = pos
    delta = (blag - center) / fps
    if abs(delta) < 0.15:
        return seg_plan, 0.0   # ほぼズレ無し → 触らない（既に合ってる場合は無害）
    new_plan = []
    for s2, e2, mv2 in seg_plan:
        if mv2 is None:
            new_plan.append((s2, e2, None))
        else:
            seglen = e2 - s2
            nm = min(max(0.0, mv2 + delta), max(0.0, vid_dur - seglen))
            new_plan.append((s2, e2, nm))
    return new_plan, delta

def measure_pos_chroma(music_audio, video_audio, m_t, mv_guess, sr=11025, half=4.0, search=4.0):
    """音楽の m_t 付近(±half秒)がMVのどこに当たるかをchromaで測る（mv_guess±search内で探索）。
    Remixでも効く（chroma=メロディ基準）。戻り: (true_mv_pos, confidence 0..1)。
    """
    try:
        mw0 = max(0, int((m_t - half) * sr)); mw1 = int((m_t + half) * sr)
        mwav = music_audio[mw0:mw1]
        v_start = max(0.0, mv_guess - search)
        v0 = int(v_start * sr); v1 = int((mv_guess + 2 * half + search) * sr)
        vwav = video_audio[v0:v1]
        if len(mwav) < sr or len(vwav) < sr:
            return mv_guess, 0.0
        _, cmm, fps = compute_features(mwav, sr)
        _, cvv, _ = compute_features(vwav, sr)
        F = cmm.shape[1]
        if cvv.shape[1] < F:
            return mv_guess, 0.0
        best, bestsc = 0, -1.0
        for p in range(cvv.shape[1] - F + 1):
            sc = float((cvv[:, p:p+F] * cmm).sum(axis=0).mean())
            if sc > bestsc:
                bestsc, best = sc, p
        # best=vwav内フレーム → (m_t-half) に対応するMV秒。m_t に直すには +half。
        true_pos = (v0 / sr) + best / fps + half
        return float(true_pos), float(bestsc)
    except Exception:
        return mv_guess, 0.0


def find_best_mv_tempo(video_audio, music_audio, sr=11025, window_sec=SEG_WINDOW_SEC):
    """MVを色々な倍率に伸縮し、曲と波形が最も一貫して合う倍率を探す（BPM検出に頼らない）。
    各倍率で「曲の窓ごとの“生”ベスト一致位置」が一直線（順番通り）に並ぶ度合い(locked率)を測り、最良を返す。
    間違ったテンポだと位置が一直線から外れていくので、locked率が最大の倍率＝正しいテンポ。
    戻り: (best_rate, best_lock 0..1)
    """
    try:
        ov, cv0, fps = compute_features(video_audio, sr)
        om, cm, _    = compute_features(music_audio, sr)
        win = max(8, int(window_sec * fps))
        n_win = max(1, int(cm.shape[1] // win))
        UNIFORM = 1.0 / np.sqrt(12)
        # 音程内容のある曲の窓だけ集める
        mwins = []
        for w in range(n_win):
            s = w * win; e = s + win
            if e > cm.shape[1]:
                break
            seg = cm[:, s:e]
            if float(seg.max(axis=0).mean()) >= UNIFORM * 1.35:
                mwins.append((w, seg))
        if len(mwins) < 6:
            return 1.0, 0.0

        Torig = cv0.shape[1]; x0 = np.arange(Torig)

        def stretched_cv(r):
            if abs(r - 1.0) < 1e-6:
                return cv0
            Tnew = max(8, int(round(Torig / r)))
            xi = np.linspace(0, Torig - 1, Tnew)
            cv = np.vstack([np.interp(xi, x0, cv0[b]) for b in range(12)])
            nrm = np.linalg.norm(cv, axis=0, keepdims=True); nrm[nrm == 0] = 1.0
            return cv / nrm

        def lock_of(cv, tol_sec=1.0):
            res = []
            for (w, seg) in mwins:
                F = seg.shape[1]
                if cv.shape[1] < F:
                    continue
                total = None
                for b in range(12):
                    c = correlate(cv[b], seg[b], mode='valid', method='fft')
                    total = c if total is None else total + c
                pos = int(np.argmax(total))
                sc = float((cv[:, pos:pos+F] * seg).sum(axis=0).mean())
                if sc >= SEG_SCORE_TH:
                    res.append(pos - w * win)     # 一直線（順番通り）なら一定値
            if len(res) < 6:
                return 0.0
            med = float(np.median(res))
            tol = max(tol_sec * fps, 1)
            return float(np.mean([abs(x - med) <= tol for x in res]))

        def search(rate_list, tol_sec):
            scored = [(r, lock_of(stretched_cv(r), tol_sec)) for r in rate_list]
            bl = max(lk for _, lk in scored)
            plateau = [r for (r, lk) in scored if lk >= bl - 0.03]
            return (float(np.median(plateau)) if plateau else 1.0), bl

        # テンポ探索（±2%刻み・許容±1s）。最良lock付近のプラトー中央を採用。
        coarse_rates = [0.86,0.88,0.90,0.92,0.94,0.96,0.98,1.0,1.02,1.04,1.06,1.08,1.10,1.12,1.14]
        coarse_scored = [(r, lock_of(stretched_cv(r), 1.0)) for r in coarse_rates]
        lock_1p0 = next((lk for (r, lk) in coarse_scored if abs(r - 1.0) < 1e-6), 0.0)
        best_lock = max(lk for _, lk in coarse_scored)
        plateau = [r for (r, lk) in coarse_scored if lk >= best_lock - 0.03]
        best_rate = float(np.median(plateau)) if plateau else 1.0
        return best_rate, best_lock, lock_1p0
    except Exception as e:
        print(f"  (テンポ探索失敗: {e})")
        return 1.0, 0.0, 0.0


def make_tempo_adjusted_mv(video_path, rate, tmp_dir):
    """MVを rate 倍のテンポに補正した動画を作る（rate>1=速く=短く / <1=遅く=長く）。
    曲BPM÷MV BPM をrateに渡すと、MVが曲のテンポに揃う。失敗時 None。
    """
    try:
        if rate <= 0:
            return None
        out = Path(tmp_dir) / "mv_tempo_adj.mp4"
        # 映像: setpts=PTS/rate（rate倍速）/ 音声: atempo=rate（0.5〜2.0対応・rateは0.67〜1.5で安全）
        cmd = ["ffmpeg", "-y", "-i", str(video_path),
               "-filter:v", f"setpts=PTS/{rate:.6f}",
               "-filter:a", f"atempo={rate:.6f}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-c:a", "aac", "-b:a", "192k", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out
        print(f"  (テンポ補正エンコード失敗)")
    except Exception as e:
        print(f"  (テンポ補正失敗: {e})")
    return None


def refine_segment_start(video_audio, music_audio, music_start, mv_start, sr=11025):
    """セグメントのMV開始位置を波形相関で微調整（±0.7秒）"""
    try:
        chunk = 8 * sr  # 8秒で照合
        m_s = int(music_start * sr)
        m = music_audio[m_s : m_s + chunk]
        if len(m) < sr: return mv_start
        v_s = int(max(0, (mv_start - 0.7) * sr))
        v_e = int((mv_start + 0.7) * sr) + len(m)
        v = video_audio[v_s : v_e]
        if len(v) < len(m): return mv_start
        c = correlate(v, m, mode='valid', method='fft')
        denom = np.sqrt((v**2).sum() * (m**2).sum()) + 1e-9
        peak = int(np.argmax(c))
        if c[peak] / denom > 0.25:
            return (v_s + peak) / sr
    except Exception:
        pass
    return mv_start

# ============================================================
# マッシュアップ: VJモード（曲ごとに映像切り替え）
# ============================================================

def classify_mashup_segments(mashup_path, song_audios, sr=11025, window_sec=5.0):
    """
    マッシュアップを時間窓ごとに「どの曲が流れているか」分類。
    song_audios: [(name, np.array), ...]
    戻り値: [(start_sec, end_sec, song_index), ...]
    """
    m = wav_to_array_path(mashup_path, sr)
    mashup_dur = len(m) / sr

    # 各曲のクロマ特徴を事前計算
    hop = 2048
    song_chromas = []
    for name, audio in song_audios:
        ch, fps = chroma_for_loop(audio, sr, hop=hop)
        song_chromas.append(ch)

    cm, fps = chroma_for_loop(m, sr, hop=hop)
    win_frames = max(4, int(window_sec * fps))
    n_windows = max(1, cm.shape[1] // win_frames)

    assignments = []
    for w in range(n_windows):
        s = w * win_frames
        e = min(s + win_frames, cm.shape[1])
        seg = cm[:, s:e]
        seg_flat = seg.mean(axis=1)  # この窓の平均クロマ

        scores = []
        for idx, sc in enumerate(song_chromas):
            song_means = []
            step = max(1, win_frames // 2)
            for ss in range(0, max(1, sc.shape[1] - win_frames), step):
                cand = sc[:, ss:ss+win_frames].mean(axis=1)
                sim = float(np.dot(seg_flat, cand) /
                            (np.linalg.norm(seg_flat) * np.linalg.norm(cand) + 1e-9))
                song_means.append(sim)
            scores.append(max(song_means) if song_means else 0)
        best_song = int(np.argmax(scores))
        best_score = scores[best_song]
        second = sorted(scores, reverse=True)[1] if len(scores) >= 2 else 0.0
        # 信頼度が低い窓は -1（フィラー＝フェス映像に逃がす）
        if best_score < 0.55 or (best_score - second) < 0.04:
            assignments.append(-1)
        else:
            assignments.append(best_song)

    # メディアンフィルタでチラつき除去
    if len(assignments) >= 3:
        assignments = list(median_filter(np.array(assignments), size=3))

    # 連続区間にまとめる
    segments = []
    seg_start = 0
    cur = assignments[0]
    for w in range(1, len(assignments)):
        if assignments[w] != cur:
            segments.append((seg_start * window_sec,
                             w * window_sec, int(cur)))
            seg_start = w
            cur = assignments[w]
    segments.append((seg_start * window_sec, mashup_dur, int(cur)))
    return segments

def download_festival_filler(tmp_dir, tag):
    """マッシュアップ曖昧区間用に「music festival edm」を検索しランダム1本DL"""
    import random
    print(f"     🎉 フィラー映像を検索中（music festival edm）...")
    results = search_youtube_mv("music festival edm crowd", n=8)
    if not results:
        return None
    pick = random.choice(results)
    url = f"https://www.youtube.com/watch?v={pick['id']}"
    fdir = tmp_dir / f"filler_{tag}"; fdir.mkdir(exist_ok=True)
    out = fdir / "filler.mp4"
    subprocess.run(["yt-dlp", *YTDLP_COOKIE_ARGS,
        "-f","bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "--no-playlist","--merge-output-format","mp4","-o",str(out), url], capture_output=True)
    if not out.exists():
        cands = list(fdir.glob("filler.*")); out = cands[0] if cands else None
    if out:
        print(f"     ✅ {pick['title']}")
    return out

def make_vj_mashup(mashup_path, song_names, loop_path, output_path, tmp_dir):
    """マッシュアップをVJ風に: 各曲のMVを検出セグメントで切り替え"""
    print(f"\n  🎛  VJモード: {len(song_names)}曲を検出")
    for i, s in enumerate(song_names, 1):
        print(f"     [{i}] {s}")

    # 各曲のMVをダウンロード
    videos = []
    for i, song in enumerate(song_names):
        print(f"\n  🔍 [{i+1}/{len(song_names)}] {song} のMVを検索中...")
        results, used_q, used_remix = smart_search_mv(song, 1)
        if used_remix:
            print(f"     ✨ Remix版MVを使用")
        if not results:
            print(f"     ⚠️  見つからず → 別の曲のMVで代用")
            videos.append(None)
            continue
        print(f"     ✅ {results[0]['title']}")
        url = f"https://www.youtube.com/watch?v={results[0]['id']}"
        vdir = tmp_dir / f"song_{i}"
        vdir.mkdir(exist_ok=True)
        out = vdir / "video.mp4"
        subprocess.run([
            "yt-dlp", *YTDLP_COOKIE_ARGS,
            "-f","bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
            "--no-playlist","--merge-output-format","mp4","-o",str(out), url
        ], capture_output=True)
        if not out.exists():
            cands = list(vdir.glob("video.*"))
            out = cands[0] if cands else None
        videos.append(out)

    # 各曲の音声を抽出（分類用）
    print(f"\n  🔍 マッシュアップを解析中（どの曲がいつ流れているか）...")
    song_audios = []
    for i, (song, vid) in enumerate(zip(song_names, videos)):
        if vid and has_audio(vid):
            audio = wav_to_array_path(vid, duration=240)
        else:
            audio = np.zeros(11025)  # ダミー
        song_audios.append((song, audio))

    segments = classify_mashup_segments(mashup_path, song_audios)

    print(f"\n  🎬 検出されたセグメント構成:")
    for s, e, idx in segments:
        if idx == -1:
            print(f"     {s:6.1f}秒 〜 {e:6.1f}秒 : 🎬 別MVで代用（曲が曖昧な区間）")
        else:
            print(f"     {s:6.1f}秒 〜 {e:6.1f}秒 : {song_names[idx]}")

    # 映像セグメントを作成
    mashup_dur = get_audio_duration_accurate(mashup_path)
    video_positions = [0.0] * len(videos)  # 各MVの再生位置を記憶
    seg_files = []

    # フェス廃止: 曖昧区間/MV取得失敗区間は「取得済みのいずれかのMV」で埋める
    valid_vids = [v for v in videos if v and get_duration(v) > 0.5]
    fill_turn = 0

    for si, (s, e, idx) in enumerate(segments):
        seg_dur = e - s
        if seg_dur < 0.5: continue
        out_seg = tmp_dir / f"seg_{si}.mp4"
        vid = videos[idx] if idx >= 0 else None

        # 曲が曖昧(idx==-1) or MV取得失敗 or MVが壊れている → 別MVをループで代用
        if idx == -1 or vid is None or get_duration(vid) <= 0.5:
            if valid_vids:
                fv = valid_vids[fill_turn % len(valid_vids)]
                fill_turn += 1
                make_filler_segment(fv, seg_dur, out_seg, tmp_dir)
            else:
                # MVが1本も取れていない → 黒画面で破綻回避（フェスは使わない）
                run(["ffmpeg","-y","-f","lavfi",
                     "-i",f"color=c=black:s=1280x720:d={seg_dur:.3f}:r=30",
                     *ENC_ARGS,"-an",str(out_seg)])
        else:
            vid_dur = get_duration(vid)
            start_pos = video_positions[idx]
            if start_pos + seg_dur > vid_dur:
                start_pos = 0.0  # 足りなければ最初から
            video_positions[idx] = start_pos + seg_dur
            run(["ffmpeg","-y","-ss",f"{start_pos:.3f}","-i",str(vid),
                 "-t",f"{seg_dur:.3f}",
                 "-vf",VF_NORM,*ENC_ARGS,"-an",str(out_seg)])
        seg_files.append(out_seg)

    # 結合
    print(f"\n  🔗 映像を結合中...")
    list_file = tmp_dir / "concat.txt"
    with open(list_file, "w") as f:
        for seg in seg_files:
            f.write(f"file '{seg}'\n")
    combined = tmp_dir / "combined.mp4"
    run(["ffmpeg","-y","-f","concat","-safe","0",
         "-i",str(list_file),"-c","copy",str(combined)])

    # 映像が音声より短い場合はMVで延長（フェスは使わない・長さ完全一致の保証）
    _pad_src = valid_vids[0] if valid_vids else Path("__NOFILL__")
    combined = ensure_video_length(combined, mashup_dur, _pad_src, tmp_dir)

    # 音楽と合成
    print(f"  🎵 音楽と合成中...")
    run(["ffmpeg","-y",
         "-i",str(combined),"-i",str(mashup_path),
         "-map","0:v:0","-map","1:a:0",
         "-c:v","copy","-c:a","aac","-b:a","320k",
         "-t",f"{mashup_dur:.3f}","-movflags","+faststart",
         str(output_path)])

    mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ✅ 完成: {output_path.name} ({mb:.1f} MB)")

# ============================================================
# 通常処理
# ============================================================

def _ensure_dtw_libs():
    """librosa と fastdtw が使えるか確認（無ければインストール試行）"""
    try:
        import librosa, fastdtw
        return True
    except Exception:
        try:
            import sys as _sys
            subprocess.run([_sys.executable, "-m", "pip", "install",
                            "librosa", "fastdtw", "--break-system-packages", "-q"],
                           capture_output=True, timeout=300)
            import librosa, fastdtw
            return True
        except Exception:
            return False

def download_video(url, out_dir, fatal=True):
    """fatal=False のときは失敗しても終了せず None を返す（候補を順に試すループ用）。"""
    print(f"\n🎬 映像をダウンロード中...")
    out = out_dir / "video.mp4"
    # YouTubeの403(Forbidden)対策:
    #   形式(AV1系)やクライアント経路によって拒否されることがあるため、
    #   ①通常 → ②H.264強制 → ③別クライアント(ios/android)+H.264 の順で自動リトライする。
    attempts = [
        ("標準", True,
         ["-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]"]),
        ("H.264形式に変更して再試行", True,
         ["-f", "bestvideo[height<=720][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/18"]),
        ("別経路(iOSクライアント)で再試行", True,
         ["--extractor-args", "youtube:player_client=ios",
          "-f", "bestvideo[height<=720][vcodec^=avc1]+bestaudio/best[height<=720]/18"]),
        ("別経路(Androidクライアント)で再試行", True,
         ["--extractor-args", "youtube:player_client=android",
          "-f", "best[height<=720]/18"]),
        # ブラウザcookieが未ログイン/不整合だと、cookie有りの方が弾かれることがある。
        # 最後の砦として cookie無し（匿名アクセス）でも試す。
        ("Cookie無しで再試行", False,
         ["-f", "bestvideo[height<=720][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/18"]),
        ("Cookie無し+別経路(Android)で再試行", False,
         ["--extractor-args", "youtube:player_client=android",
          "-f", "best[height<=720]/18"]),
    ]
    for i, (label, use_cookies, fmt_args) in enumerate(attempts):
        if i > 0:
            print(f"  🔁 ダウンロード拒否(403等) → {label}...")
        cookie_args = YTDLP_COOKIE_ARGS if use_cookies else []
        r = subprocess.run([
            "yt-dlp", *cookie_args, *fmt_args,
            "--no-playlist", "--merge-output-format", "mp4", "-o", str(out), url
        ], capture_output=(i > 0), text=True, errors="replace")
        if out.exists() and out.stat().st_size > 0:
            print(f"  ✅ 完了")
            return out
        candidates = list(out_dir.glob("video.*"))
        if candidates:
            print(f"  ✅ 完了")
            return candidates[0]
        # 403以外の致命的エラー（動画削除/非公開など）はリトライしても無駄なので打ち切る
        err = (r.stderr or "") if i > 0 else ""
        if i > 0 and any(k in err for k in ("Private video", "Video unavailable", "removed")):
            print("  ❌ この動画は非公開/削除されています")
            break
    print("❌ 映像ファイルが見つかりません")
    print("   （READMEの『403エラー』の章を参照：ChromeでYouTubeにログイン→再実行 が一番効きます）")
    if fatal:
        sys.exit(1)
    return None


def is_static_video(video_path, samples=6):
    """ダウンロードした動画が実質"静止画"（ジャケ画像/音声のみアップロード）かを判定。
    均等な時刻のフレームを抜き、フレーム間の平均差分が極小なら静止画とみなす。"""
    try:
        dur = get_duration(video_path)
        if dur <= 1.0:
            return False
        W, H = 64, 36
        frames = []
        for i in range(samples):
            t = dur * (i + 1) / (samples + 1)
            r = subprocess.run(
                ["ffmpeg", "-v", "quiet", "-ss", f"{t:.2f}", "-i", str(video_path),
                 "-frames:v", "1", "-vf", f"scale={W}:{H},format=gray",
                 "-f", "rawvideo", "-"],
                capture_output=True)
            buf = r.stdout or b""
            if len(buf) >= W * H:
                frames.append(np.frombuffer(buf[:W * H], dtype=np.uint8).astype(np.float32))
        if len(frames) < 2:
            return False
        diffs = [float(np.mean(np.abs(frames[i] - frames[i - 1]))) for i in range(1, len(frames))]
        return float(np.mean(diffs)) < 2.0   # 0-255スケールで平均差<2 ≒ 実質静止
    except Exception:
        return False

import random

_FESTIVAL_CACHE = {}

def get_festival_filler_cached(tmp_dir):
    """
    フェス映像（music festival edm）を取得してキャッシュ。
    1回のツール実行中は使い回す（毎回DLしない）。複数本貯めてランダム性も確保。
    戻り値: フェス動画ファイルのリスト or None
    """
    key = "festival"
    cached = _FESTIVAL_CACHE.get(key, [])
    cached = [p for p in cached if Path(p).exists()]
    if len(cached) >= 2:
        return [Path(p) for p in cached]
    # 2本貯まるまで取得
    results = search_youtube_mv("music festival edm crowd", n=10)
    import random as _rnd
    _rnd.shuffle(results)
    fdir = tmp_dir / "festival_cache"; fdir.mkdir(parents=True, exist_ok=True)
    for r in results:
        if len(cached) >= 2: break
        out = fdir / f"fest_{r['id']}.mp4"
        if out.exists() and out.stat().st_size > 0:
            if str(out) not in cached: cached.append(str(out))
            continue
        subprocess.run(["yt-dlp", *YTDLP_COOKIE_ARGS,
            "-f","bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
            "--no-playlist","--merge-output-format","mp4","-o",str(out),
            f"https://www.youtube.com/watch?v={r['id']}"], capture_output=True)
        if out.exists() and out.stat().st_size > 0:
            cached.append(str(out))
    _FESTIVAL_CACHE[key] = cached
    return [Path(p) for p in cached] if cached else None

def get_loop_videos(loop_path):
    """
    フィラー映像のソースを返す。
    フェス映像は廃止。実在パス（原曲MV/関連MV/フォルダ/ファイル）のみ使う。
    解決できない指定（旧"FESTIVAL"やダミー）は None を返し、呼び出し側で黒画面にする。
    """
    if str(loop_path) == "FESTIVAL":
        # フェス映像は廃止。取得しない（→ make_filler_segment 側で黒画面フォールバック）
        return None
    p = Path(loop_path)
    if not p.exists():
        return None
    if p.is_dir():
        vids = sorted([f for f in p.iterdir()
                       if f.suffix.lower() in [".mp4",".mov",".m4v",".webm",".avi"]])
        return vids if vids else None
    return [p]

def pick_loop(loop_path):
    """ランダムに1本選ぶ"""
    vids = get_loop_videos(loop_path)
    return random.choice(vids)

def make_filler_segment(loop_path, duration, out_path, tmp_dir=None):
    """フィラー映像をduration秒分作る。フォルダ指定なら複数動画をランダムにつなぐ"""
    vids = get_loop_videos(loop_path)
    # 壊れた素材(長さ0や読めないもの)を除外
    if vids:
        vids = [v for v in vids if get_duration(v) > 0.5]
    if not vids:
        # フェス映像が取れない（ネット無し等）→ 黒背景で埋める（最低限破綻させない）
        run(["ffmpeg","-y","-f","lavfi","-i",f"color=c=black:s=1280x720:d={duration:.3f}:r=30",
             *ENC_ARGS,"-an",str(out_path)])
        return
    if len(vids) == 1:
        # 単一ファイル: 従来のループ
        loop_dur = get_duration(vids[0])
        loops = int(duration / loop_dur) + 2
        run(["ffmpeg","-y","-stream_loop",str(loops),"-i",str(vids[0]),
             "-t",f"{duration:.3f}",
             "-vf",VF_NORM,*ENC_ARGS,"-an",str(out_path)])
        return

    # 複数ファイル: ランダムに切り替えながら埋める
    if tmp_dir is None:
        tmp_dir = out_path.parent
    pieces = []
    remaining = duration
    pi = 0
    last = None
    while remaining > 0.3:
        v = random.choice(vids)
        if len(vids) > 1 and v == last:
            v = random.choice([x for x in vids if x != last])
        last = v
        v_dur = get_duration(v)
        # 1カット5〜12秒くらいでランダムに
        cut = min(remaining, random.uniform(5, 12), v_dur)
        start = random.uniform(0, max(0.0, v_dur - cut))
        piece = tmp_dir / f"filler_{out_path.stem}_{pi}.mp4"
        run(["ffmpeg","-y","-ss",f"{start:.3f}","-i",str(v),
             "-t",f"{cut:.3f}",
             "-vf",VF_NORM,*ENC_ARGS,"-an",str(piece)])
        # 切り出しに成功（実ファイルがある）ものだけ採用
        if piece.exists() and piece.stat().st_size > 0:
            pieces.append(piece)
        remaining -= cut
        pi += 1

    # 有効なpieceが1つも無ければ黒背景で埋める（破綻回避）
    if not pieces:
        run(["ffmpeg","-y","-f","lavfi","-i",f"color=c=black:s=1280x720:d={duration:.3f}:r=30",
             *ENC_ARGS,"-an",str(out_path)])
        return

    if len(pieces) == 1:
        shutil.copy(pieces[0], out_path)
    else:
        list_file = tmp_dir / f"filler_concat_{out_path.stem}.txt"
        with open(list_file, "w") as f:
            for p_ in pieces:
                f.write(f"file '{p_}'\n")
        run(["ffmpeg","-y","-f","concat","-safe","0",
             "-i",str(list_file),"-c","copy",str(out_path)])
        # 結合に失敗（出力が無い/空）したら先頭pieceで代用
        if not out_path.exists() or out_path.stat().st_size == 0:
            shutil.copy(pieces[0], out_path)
    for p_ in pieces:
        p_.unlink(missing_ok=True)

def ensure_video_length(video_path, required_dur, loop_path, tmp_dir):
    """映像がrequired_durより短ければフィラーを足して延長した新ファイルを返す"""
    vid_dur = get_duration(video_path)
    if vid_dur >= required_dur - 0.05:
        return video_path
    # 不足分+1秒のフィラーを作って後ろに連結
    shortage = required_dur - vid_dur + 1.0
    pad = tmp_dir / f"pad_{video_path.stem}.mp4"
    make_filler_segment(loop_path, shortage, pad, tmp_dir)
    extended = tmp_dir / f"ext_{video_path.stem}.mp4"
    list_file = tmp_dir / f"extlist_{video_path.stem}.txt"
    with open(list_file, "w") as f:
        f.write(f"file '{video_path}'\n")
        f.write(f"file '{pad}'\n")
    run(["ffmpeg","-y","-f","concat","-safe","0",
         "-i",str(list_file),"-c","copy",str(extended)])
    return extended

def make_plain_mv_sync(mv_path, music_path, output_path, tmp_dir):
    """
    リップシンクや同期が使えない時の最終フォールバック。
    原曲MVを頭からそのまま流し、音楽の長さに合わせる（短ければループ、長ければカット）。
    2デッキ・カット編集の代わり：関連動画やフェスを使わず、原曲MVだけで作る。
    """
    music_dur = get_audio_duration_accurate(music_path)
    mv_dur = get_duration(mv_path) if mv_path else 0
    if not mv_path or mv_dur <= 0.5:
        # MVが無い/壊れている → 黒背景で音楽の長さぶん作る（最低限破綻させない）
        tmp_black = tmp_dir / "plain_black.mp4"
        run(["ffmpeg","-y","-f","lavfi","-i",f"color=c=black:s=1280x720:r=30:d={music_dur+2.0:.1f}",
             *ENC_ARGS,"-an",str(tmp_black)])
        run(["ffmpeg","-y","-i",str(tmp_black),"-i",str(music_path),
             "-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","320k",
             "-t",f"{music_dur:.3f}","-movflags","+faststart",str(output_path)])
        tmp_black.unlink(missing_ok=True)
        return
    tmp_video = tmp_dir / "plain_mv_tmp.mp4"
    # MVが音楽より短ければループ回数を計算（+1で余裕）
    loops = max(0, int(music_dur / mv_dur) + 1)
    run(["ffmpeg","-y","-stream_loop",str(loops),"-i",str(mv_path),
         "-t",f"{music_dur + 2.0:.3f}","-vf",VF_NORM,*ENC_ARGS,"-an",str(tmp_video)])
    # 音声を載せて正確な長さにカット
    run(["ffmpeg","-y",
         "-i",str(tmp_video),"-i",str(music_path),
         "-map","0:v:0","-map","1:a:0",
         "-c:v","copy","-c:a","aac","-b:a","320k",
         "-t",f"{music_dur:.3f}","-movflags","+faststart",
         str(output_path)])
    tmp_video.unlink(missing_ok=True)


def make_loop_only(loop_path, music_path, output_path):
    music_dur = get_audio_duration_accurate(music_path)
    tmp_video = output_path.parent / "loop_only_tmp.mp4"
    # 音声より2秒長く作る → 最終-tで正確に音声の長さにカット
    make_filler_segment(loop_path, music_dur + 2.0, tmp_video)
    run(["ffmpeg","-y",
         "-i",str(tmp_video),"-i",str(music_path),
         "-map","0:v:0","-map","1:a:0",
         "-c:v","copy","-c:a","aac","-b:a","320k",
         "-t",f"{music_dur:.3f}","-movflags","+faststart",
         str(output_path)])
    tmp_video.unlink(missing_ok=True)

def make_loop_segment(loop_path, duration, out_path):
    make_filler_segment(loop_path, duration, out_path)

def concat_segments(segments, out_path, tmp_dir):
    list_file = tmp_dir / "concat_list.txt"
    with open(list_file, "w") as f:
        for seg in segments:
            f.write(f"file '{seg}'\n")
    run(["ffmpeg","-y","-f","concat","-safe","0",
         "-i",str(list_file),"-c","copy",str(out_path)])

CONFIDENCE_THRESHOLD = 0.35

def fit_to_size(video_path, music_path, target_mb, tmp_dir):
    """
    完成MP4を target_mb 以内に収める。音声は元のまま（音質維持）、
    映像ビットレートだけ落として調整する。既に収まっていれば何もしない。
    """
    try:
        cur = video_path.stat().st_size / (1024*1024)
    except Exception:
        return video_path
    if cur <= target_mb:
        return video_path  # 既に収まってる

    dur = get_duration(video_path)
    if dur <= 0:
        return video_path

    # 音声のビットレートを取得（元のまま使うので、その分を引く）
    abr_kbps = 320  # 取得失敗時の保守的な見積り
    try:
        r = subprocess.run(["ffprobe","-v","quiet","-select_streams","a:0",
            "-show_entries","stream=bit_rate","-of","default=noprint_wrappers=1:nokey=1",
            str(video_path)], capture_output=True, text=True, errors="replace")
        v = r.stdout.strip()
        if v.isdigit() and int(v) > 0:
            abr_kbps = int(v) / 1000
    except Exception:
        pass

    # 使える総ビット量から映像ビットレートを逆算
    #   目標サイズ(bits) = target_mb * 8 * 1024 * 1024
    #   映像bps = 総bps - 音声bps、コンテナ余裕として95%に抑える
    target_bits = target_mb * 8 * 1024 * 1024 * 0.95
    total_kbps = target_bits / dur / 1000
    video_kbps = total_kbps - abr_kbps
    if video_kbps < 150:
        video_kbps = 150  # 最低画質の下限（これ以下は見るに耐えない）

    print(f"  📦 {cur:.0f}MB → {target_mb}MB以内に圧縮中（映像 {video_kbps:.0f}kbps・音声は維持）...")
    out = tmp_dir / ("fit_" + video_path.name)
    # 2パスエンコードで目標ビットレートを正確に当てる
    passlog = tmp_dir / "ff2pass"
    # 1パス目（解析）
    subprocess.run(["ffmpeg","-y","-i",str(video_path),
        "-c:v","libx264","-b:v",f"{video_kbps:.0f}k","-pass","1",
        "-passlogfile",str(passlog),"-an","-f","mp4","-preset","medium",
        "/dev/null"], capture_output=True)
    # 2パス目（音声はコピー＝元のまま）
    r = subprocess.run(["ffmpeg","-y","-i",str(video_path),
        "-c:v","libx264","-b:v",f"{video_kbps:.0f}k","-pass","2",
        "-passlogfile",str(passlog),"-c:a","copy","-pix_fmt","yuv420p",
        "-movflags","+faststart","-preset","medium",str(out)], capture_output=True)
    # passログ掃除
    for p in tmp_dir.glob("ff2pass*"):
        p.unlink(missing_ok=True)
    if out.exists() and out.stat().st_size > 0:
        newmb = out.stat().st_size/(1024*1024)
        print(f"     ✅ {newmb:.0f}MB に収まりました")
        return out
    return video_path

def dtw_align_video(remix_audio_path, source_video_path, out_path, tmp_dir):
    """
    DTW（動的時間伸縮法）で、テンポの違う原曲映像をRemix音源に同期させる。
    速度は変えず、ビート区間ごとに原曲映像の対応箇所を切り出して並べる（カット編集）。
    成功なら out_path を返す。DTWが使えない/対応が悪い場合は None。
    """
    try:
        import librosa
    except Exception:
        return None

    import numpy as _np
    try:
        sr = 22050
        y_remix, _ = librosa.load(str(remix_audio_path), sr=sr)
        y_src,   _ = librosa.load(str(source_video_path), sr=sr)
        if len(y_remix) < sr*5 or len(y_src) < sr*5:
            return None

        # ビート検出
        _, beats_r = librosa.beat.beat_track(y=y_remix, sr=sr)
        _, beats_s = librosa.beat.beat_track(y=y_src,   sr=sr)
        if len(beats_r) < 8 or len(beats_s) < 8:
            return None

        # ビート同期クロマ（chroma_stft=高速。精度はcqtとほぼ同等）
        chroma_r = librosa.feature.chroma_stft(y=y_remix, sr=sr)
        chroma_s = librosa.feature.chroma_stft(y=y_src,   sr=sr)
        cr = librosa.util.sync(chroma_r, beats_r, aggregate=_np.median)
        cs = librosa.util.sync(chroma_s, beats_s, aggregate=_np.median)

        # DTW（librosaのC実装。Sakoe-Chiba band で高速化&暴走防止）
        # コスト行列はクロマ間のコサイン距離
        D, wp = librosa.sequence.dtw(X=cr, Y=cs, metric='cosine',
                                     global_constraints=True, band_rad=0.25)
        # wp は (remix_beat_idx, src_beat_idx) の配列（逆順）
        wp = wp[::-1]
        norm = float(D[-1, -1]) / max(len(wp), 1)
        if norm > 0.6:   # 対応が悪すぎ（別曲）
            return None

        beat_times_r = librosa.frames_to_time(beats_r, sr=sr)
        beat_times_s = librosa.frames_to_time(beats_s, sr=sr)
        r2s = {}
        for ri, si in wp:
            r2s.setdefault(int(ri), []).append(int(si))

        src_dur = get_duration(source_video_path)
        n_r = len(beats_r)

        pieces = []; pidx = 0; bi = 0
        STEP = 8  # 8ビート(≒2小節)ごとにカット（カット数を抑えて高速化）
        while bi < n_r - 1:
            r_start = beat_times_r[bi]
            r_end_i = min(bi + STEP, n_r - 1)
            r_end = beat_times_r[r_end_i]
            seg_dur = r_end - r_start
            if seg_dur < 0.2:
                bi = r_end_i; continue
            if bi in r2s:
                si = int(_np.median(r2s[bi]))
            else:
                si = min(int(bi * len(beats_s) / n_r), len(beats_s)-1)
            src_start = beat_times_s[min(si, len(beats_s)-1)]
            if src_start + seg_dur > src_dur:
                src_start = max(0, src_dur - seg_dur)
            piece = tmp_dir / f"dtw_piece_{pidx:04d}.mp4"
            # -ss を入力前に置く=高速シーク。エンコードはfast/crf20で軽く
            run(["ffmpeg","-y","-ss",f"{src_start:.3f}","-i",str(source_video_path),
                 "-t",f"{seg_dur:.3f}","-an",
                 "-vf","scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1",
                 "-c:v","libx264","-preset","ultrafast","-crf","20","-pix_fmt","yuv420p",
                 "-video_track_timescale","15360",str(piece)])
            if piece.exists() and piece.stat().st_size > 0:
                pieces.append(piece); pidx += 1
            bi = r_end_i

        if not pieces:
            return None
        listf = tmp_dir / "dtw_concat.txt"
        with open(listf, "w") as f:
            for p in pieces:
                f.write(f"file '{p}'\n")
        concat = tmp_dir / "dtw_video.mp4"
        run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(listf),
             "-c","copy",str(concat)])
        for p in pieces:
            p.unlink(missing_ok=True)
        if concat.exists() and concat.stat().st_size > 0:
            return concat
        return None
    except Exception as e:
        print(f"     ⚠️  DTW同期失敗: {e}")
        return None

def remix_tempo_close_to_mv(music_path, video_path, music_dur, vid_dur, tol=0.03):
    """
    RemixとMVのBPMを軽く検出して、テンポがほぼ同じか判定する（Demucs不要・軽量）。
    近ければ True（=普通のMV同期で合う＝リップシンク不要）。
    BPMはオクターブ違い（2倍/半分）を吸収して比較する。
    """
    try:
        import librosa
        sr = 11025  # wav_to_array_path のデフォルトサンプルレート
        ar = wav_to_array_path(music_path, sr=sr, duration=min(music_dur, 90))
        ao = wav_to_array_path(video_path, sr=sr, duration=min(vid_dur, 90))
        ar = np.asarray(ar, dtype=np.float32)
        ao = np.asarray(ao, dtype=np.float32)
        if ar.size < sr or ao.size < sr:
            return False, 0.0, 0.0
        br, _ = librosa.beat.beat_track(y=ar, sr=sr)
        bo, _ = librosa.beat.beat_track(y=ao, sr=sr)
        br = float(br) if np.ndim(br) == 0 else float(br[0])
        bo = float(bo) if np.ndim(bo) == 0 else float(bo[0])
        if br <= 0 or bo <= 0:
            return False, br, bo
        # オクターブ違いを吸収（比を0.67〜1.5に収める）
        ratio = br / bo
        while ratio > 1.5: ratio /= 2
        while ratio < 0.67: ratio *= 2
        is_close = abs(ratio - 1.0) <= tol
        return is_close, br, bo
    except Exception:
        return False, 0.0, 0.0


def _try_vocal_lipsync(music_path, video_path, output_path, tmp_dir, music_dur):
    """まず Pro版エンジン（HuBERT中間層＋subseq DTW＋Viterbi全体経路）で同期を試す。
    Pro版が未導入/失敗なら、従来のボーカル分離リップシンク（doc12）へフォールバック。
    成功で True、最終的に失敗で False（呼び出し側で別法へ）。"""
    # --- Pro版エンジンを優先（口パクだけ高精度に差し替え。他機能は本体のまま）---
    try:
        import lipsync_pro
        print("  🧠 高精度リップシンク(Pro: HuBERT層9＋subseq DTW＋Viterbi経路)で同期...")
        if lipsync_pro.process(str(music_path), str(video_path), str(output_path),
                               use_hubert=True, placement="equal", sync_offset_ms="auto"):
            return True
        print("  ⚠️ Pro同期が完走せず → 従来方式へフォールバック")
    except Exception as e:
        print(f"  ⚠️ Pro同期で例外 → 従来方式へフォールバック: {e}")
    # --- フォールバック: 従来(doc12)のボーカル分離リップシンク ---
    try:
        import vocal_sync
        return bool(vocal_sync.make_vocal_lipsync_remix(
            music_path, video_path, output_path, tmp_dir, music_dur,
            filler_cb=lambda d, o: make_filler_segment(video_path, d, o, tmp_dir)))
    except Exception as e:
        print(f"  ⚠️ ボーカル分離リップシンクで例外: {e}")
        return False


def _remix_with_lipsync(music_path, video_path, chosen_url, loop_path, output_path, tmp_dir, music_dur):
    """
    Remix経路: まず Demucs ボーカル分離でリップシンク同期を試し、
    成功すればそれを採用。素材不足/分離不可/一致不足なら原曲MVをそのまま流す。
    """
    if _try_vocal_lipsync(music_path, video_path, output_path, tmp_dir, music_dur):
        return
    print(f"  → リップシンク不可 → 原曲MVをそのまま流します")
    make_plain_mv_sync(video_path, music_path, output_path, tmp_dir)


STRAIGHT_MODE = False   # True: MVを時系列そのまま等速で被せる（リップシンク・多様化オフ）


def _wf_best_pos(m, video_audio, sr):
    """音楽チャンク m が MV波形のどこに当たるかを、生波形の正規化相互相関(NCC)で
    全域から一意に探す（単調制約なし）。同一録音なら繰り返しサビでも"そのテイク"を指す。
    戻り: (mv秒, conf 0..1)。"""
    L = len(m)
    if L < int(sr * 0.4) or len(video_audio) < L + 2:
        return 0.0, 0.0
    v = video_audio
    c = correlate(v, m, mode='valid', method='fft')        # len = len(v)-L+1
    v2 = v.astype(np.float64) ** 2
    csum = np.concatenate(([0.0], np.cumsum(v2)))
    e_v = csum[L:] - csum[:len(csum) - L]                   # 各窓のエネルギー
    e_m = float((m.astype(np.float64) ** 2).sum())
    n = min(len(c), len(e_v))
    # 分母の床上げ：窓がクエリより静かな所(クレジット/フェード等)で比が水増しされて
    # 誤って勝つのを防ぐ。窓エネルギーをクエリ側で下限クリップ → ncc は必ず ≤1。
    denom = np.sqrt(np.maximum(e_v[:n], e_m) * max(e_m, 1e-12)) + 1e-9
    ncc = c[:n] / denom
    p = int(np.argmax(ncc))
    return p / sr, float(ncc[p])


def waveform_track_plan(music_audio, video_audio, sr, music_dur, vid_dur,
                        win=4.0, hop=2.0, match_th=0.60, jump_tol=0.5, strong=0.82):
    """全編を hop秒ごとに区切り、各窓の【絶対ベスト】MV位置を生波形NCCで実測（単調制約なし）。
    ・conf>=match_th の窓 → 波形が指すMV位置に配置（連続はまとめ、飛びでカット）
    ・conf<match_th の窓 → MVに無い区間（追加イントロ/つなぎ/SE）とみなしフィラー(None)
      （孤立した1窓だけの落ち込みは等速予測で補間してフィラー断片化を防ぐ）
    同一音源判定は「合う所がどれだけ強いか」で行う。戻り:
      (seg_plan[(s,e,mv|None)], conf列, score{p80, strong_frac, match_ratio})"""
    pts = []
    t = 0.0
    while t < music_dur - 0.05:
        e = min(music_dur, t + win)
        m = music_audio[int(t * sr):int(e * sr)]
        pos, conf = _wf_best_pos(m, video_audio, sr)
        pts.append([t, min(t + hop, music_dur), pos, conf])
        t += hop
    if not pts:
        return [], [], {"p80": 0.0, "strong_frac": 0.0, "match_ratio": 0.0}
    confs = np.array([p[3] for p in pts], dtype=float)
    N = len(pts)
    p80 = float(np.percentile(confs, 80))
    strong_frac = float(np.mean(confs >= strong))
    match_ratio = float(np.mean(confs >= match_th))
    score = {"p80": p80, "strong_frac": strong_frac, "match_ratio": match_ratio}

    # 孤立した1窓だけの落ち込み(transient)は等速予測で補間し有効化＝フィラー断片化を防ぐ
    for i in range(1, N - 1):
        if pts[i][3] < match_th and pts[i - 1][3] >= match_th and pts[i + 1][3] >= match_th:
            pts[i][2] = pts[i - 1][2] + (pts[i][0] - pts[i - 1][0])
            pts[i][3] = match_th

    # セグメント構築：弱窓の連続=フィラー(None)、マッチ窓=波形位置（飛びでカット）
    def _clamp(x): return max(0.0, min(x, max(0.1, vid_dur - 0.1)))
    seg_plan = []
    i = 0
    while i < N:
        if pts[i][3] < match_th:
            s0 = pts[i][0]; e0 = pts[i][1]
            while i < N and pts[i][3] < match_th:
                e0 = pts[i][1]; i += 1
            seg_plan.append((s0, e0, None))            # MVに無い区間 → フィラー
        else:
            cs, ce, cmv = pts[i][0], pts[i][1], pts[i][2]
            last_t, last_mv = pts[i][0], pts[i][2]
            i += 1
            while i < N and pts[i][3] >= match_th:
                t0, t1, mv, _c = pts[i]
                expected = last_mv + (t0 - last_t)
                if abs(mv - expected) <= jump_tol:     # 連続 → 延長
                    ce = t1; last_t, last_mv = t0, mv; i += 1
                else:
                    break                              # 波形が飛ぶ → カット
            seg_plan.append((cs, ce, _clamp(cmv)))
    return seg_plan, confs, score


PENDING_WARNINGS = []   # 曲ごとの重要警告（静止画MV等）を溜めて最後に必ず見せる

def process_with_youtube(urls, music_path, loop_path, output_path, tmp_dir):
    # urls は候補URLのリスト（後方互換で単一strも可）。静止画/短すぎ候補は飛ばす。
    if isinstance(urls, str):
        urls = [urls]
    # 曲の長さ（候補が短すぎないかの判定に使う）
    try:
        _music_dur_pre = get_audio_duration_accurate(music_path)
    except Exception:
        _music_dur_pre = 0
    video_path = None
    chosen_url = urls[0] if urls else ""
    _static_last = None            # 全滅時の最後の砦（静止画でも一応使う用）
    for i, u in enumerate(urls):
        cand_dir = tmp_dir / f"cand{i}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        vp = download_video(u, cand_dir, fatal=False)
        if vp is None or not Path(vp).exists():
            continue
        # 静止画（ジャケ画像/音声のみのアップロード）は最後の候補でも採用しない
        if is_static_video(vp):
            print("  ⚠️ この候補は静止画（ジャケ画像/音声のみのアップロード）でした → 次を試します")
            _static_last = (vp, u)
            continue
        if i < len(urls) - 1:
            try:
                _vd = get_duration(vp)
            except Exception:
                _vd = 0
            if _music_dur_pre and _vd and _vd < _music_dur_pre * 0.6:
                print(f"  ⚠️ この候補はMVが短すぎ（{_vd:.0f}秒 < 曲{_music_dur_pre:.0f}秒の6割）= 短縮版/別物の可能性 → 次の候補を試します")
                continue
        video_path = vp
        chosen_url = u
        break
    # 候補が全部 静止画/取得失敗 だった → 「music video」で救済再検索して動く映像を探す
    if video_path is None and _static_last is not None:
        try:
            tags = get_metadata(music_path)
            _t = tags.get("title", "") or strip_filename_noise(music_path.stem)
            _a = tags.get("artist", "")
            rq = clean_song_query(f"{_a} {_t}".strip()) + " music video"
            print(f"  🔎 静止画しか無かったため再検索: {rq}")
            tried = set()
            for u in urls:
                if "watch?v=" in u:
                    tried.add(u.split("watch?v=")[-1].split("&")[0])
            rescue = [r for r in search_youtube_mv(rq, n=5) if r["id"] not in tried]
            for j, r0 in enumerate(rescue):
                cand_dir = tmp_dir / f"rescue{j}"
                cand_dir.mkdir(parents=True, exist_ok=True)
                vp = download_video(f"https://www.youtube.com/watch?v={r0['id']}", cand_dir, fatal=False)
                if vp and Path(vp).exists() and not is_static_video(vp):
                    print(f"  ✅ 動く映像を発見: {r0['title']}")
                    video_path = vp
                    chosen_url = f"https://www.youtube.com/watch?v={r0['id']}"
                    break
        except Exception as _e:
            pass
        if video_path is None:
            print("")
            print("  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            print("  ┃ ⚠️  この曲は【静止画（ジャケ写）の映像】しか見つかりません ┃")
            print("  ┃     公式MVが存在しない可能性が高いです。                 ┃")
            print("  ┃     このまま作ると、映像はジャケ写のままのMP4になります。 ┃")
            print("  ┃     別の映像を使いたい場合は URL版 で好きな動画を指定を。 ┃")
            print("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
            print("")
            PENDING_WARNINGS.append(
                f"⚠️ {music_path.stem}: 静止画（ジャケ写）ベースで作成（公式MVが見つからない曲）")
            video_path, chosen_url = _static_last
    if video_path is None:
        print("  ❌ 有効な映像が取得できませんでした")
        return

    # 自動で選ばれたMVが曲より極端に短い＝短縮版/別物の可能性。
    # 候補が1個しか無いと「短すぎスキップ」が飛び先を持てないので、ここで手動URL上書きの機会を与える。
    try:
        _chosen_dur = get_duration(video_path)
    except Exception:
        _chosen_dur = 0
    if _music_dur_pre and _chosen_dur and _chosen_dur < _music_dur_pre * 0.6:
        print(f"  ⚠️ 見つかったMVが短いです（{_chosen_dur:.0f}秒 / 曲{_music_dur_pre:.0f}秒）= 短縮版/別物かもしれません。")
        print("     ※短いMVだと後半の映像が足りず繰り返しになりがちですが、このまま自動で進めます。")
        _alt = ""   # 自動モード: Enter入力を求めず、上書きせずそのまま進む
        if _alt.startswith("http"):
            _adir = tmp_dir / "cand_manual"
            _adir.mkdir(parents=True, exist_ok=True)
            _avp = download_video(_alt, _adir)
            if _avp and Path(_avp).exists():
                video_path = _avp
                chosen_url = _alt
                try:
                    _nd = get_duration(video_path)
                except Exception:
                    _nd = 0
                print(f"  ✅ 手動指定のMVを使用します（{_nd:.0f}秒）")
            else:
                print("  ⚠️ 手動URLのダウンロードに失敗 → 元のMVのまま進めます")

    vid_dur    = get_duration(video_path)
    music_dur  = get_audio_duration_accurate(music_path)
    if music_dur < 1.0:
        print(f"\n❌ 音楽ファイルの長さが取得できません（{music_dur:.2f}秒）: {music_path.name}")
        return

    # 「準・原曲エディット」判定: Intro/Extended/Short等のDJエディットで、
    # Remix/Bootlegではないもの。中身がほぼ原曲なのでカット編集に飛ばさず同期優先にする。
    _tags = get_metadata(music_path)
    _name_for_edit = f"{_tags.get('title','')} {music_path.stem}".lower()
    _is_remix_word = bool(re.search(r'(remix|bootleg|rmx|rework|refix|flip|mashup|mash-up|bounce|vip|\bdub\b)', _name_for_edit))
    _is_edit_word  = bool(re.search(r'(intro|extended|\bshort\b|\bclean\b|\bradio edit\b|\bradio\b|\bedit\b|\bedits\b)', _name_for_edit))
    is_quasi_original = _is_edit_word and not _is_remix_word
    if is_quasi_original:
        print(f"  ℹ️  Edit/原曲系（Intro/Extended/Edit等・同一音源）→ 波形ファーストで合わせます")
    elif _is_remix_word:
        print(f"  ℹ️  Remix → 波形でテンポを探して合わせます（無理ならリップシンク）")

    print(f"\n  音楽: {music_dur:.1f}秒 / YouTube映像: {vid_dur:.1f}秒")

    # 時系列そのまま等速モード：手動フラグ STRAIGHT_MODE=True の時だけ（editは口パク同期に戻す）
    if STRAIGHT_MODE:
        print("  🎞 時系列そのまま等速で配置（口パク・多様化オフ）→ MVを頭から流します")
        make_plain_mv_sync(video_path, music_path, output_path, tmp_dir)
        try:
            mb = output_path.stat().st_size / 1024 / 1024
            print(f"  ✅ 完成: {output_path.name} ({mb:.1f} MB)")
        except Exception:
            pass
        return

    if not has_audio(video_path):
        print(f"  ⚠️  映像に音声なし → 原曲MVをそのまま流して全編作成")
        make_plain_mv_sync(video_path, music_path, output_path, tmp_dir)
        mb = output_path.stat().st_size / 1024 / 1024
        print(f"  ✅ 完成: {output_path.name} ({mb:.1f} MB)")
        return

    # 波形ファースト：edit / Remix / 原曲すべて、まず波形でMVに合わせる。
    print(f"\n🔍 区間アライメント解析中（まず波形でMV対応箇所を検索）...")
    video_audio = wav_to_array_path(video_path, duration=min(vid_dur, 360))
    music_audio = wav_to_array_path(music_path, duration=music_dur)

    # Remixはテンポが数%違うことがある（例: 75↔70）。BPM検出は当てにならないので、
    # 波形を色々な倍率で試して「一番一直線に揃う倍率」を探し、合えばMVを補正して等速で乗せる。
    remix_aligned = False  # リミックスで拍が一直線に揃った（＝MV全体を当てて流せる）
    if _is_remix_word and not is_quasi_original:
        best_rate, lock, lock_1p0 = find_best_mv_tempo(video_audio, music_audio)
        print(f"  🎚 テンポ探索: 最良 ×{best_rate:.3f}（一直線度 {lock*100:.0f}% / 等倍は {lock_1p0*100:.0f}%）")
        if lock < 0.45:
            print(f"  ⚠️ どのテンポでも波形が一直線に揃わない（別アレンジ）→ リップシンクに切替")
            if _try_vocal_lipsync(music_path, video_path, output_path, tmp_dir, music_dur):
                return
            print(f"  → リップシンクも弱い → 原曲MVをそのまま流します")
            make_plain_mv_sync(video_path, music_path, output_path, tmp_dir)
            try:
                mb = output_path.stat().st_size / 1024 / 1024
                print(f"  ✅ 完成: {output_path.name} ({mb:.1f} MB)")
            except Exception:
                pass
            return
        # 「MV全体を1オフセットで当てて流せる」のは拍が十分に一直線な時だけ。
        # lock 0.45〜0.80 は揃いが中途半端＝当て込み不可（後段でリップシンクへ回す）。
        remix_aligned = (lock >= 0.80)
        if abs(best_rate - 1.0) > 0.003 and (lock - lock_1p0) >= 0.10:
            print(f"  🎚 MVを ×{best_rate:.3f} にテンポ補正して波形を合わせます...")
            adj = make_tempo_adjusted_mv(video_path, best_rate, tmp_dir)
            if adj is not None:
                video_path = adj
                vid_dur = get_duration(video_path) or (vid_dur / best_rate)
                video_audio = wav_to_array_path(video_path, duration=min(vid_dur, 360))
        else:
            print(f"  ℹ️ 等倍と大差なし → テンポ補正せず等倍で配置")

    # ── 全編・生波形・等間隔追従（単調制約なし）──
    #   各窓の絶対ベストMV位置を生波形NCCで実測し、波形が指す所へそのまま置く。
    #   クロマ補正もインスタンス選択もドリフト補正も使わない（同一音源なら波形が一意に指す）。
    print(f"  🌊 全編を生波形で追従中（単調制約なし）...")
    seg_plan, _confs, score = waveform_track_plan(
        music_audio, video_audio, 11025, music_dur, vid_dur, win=4.0, hop=2.0)
    p80 = score["p80"]; strong_frac = score["strong_frac"]; match_ratio = score["match_ratio"]
    print(f"  📊 波形一致: 強一致率{strong_frac*100:.0f}% / p80={p80:.2f} / 一致区間{match_ratio*100:.0f}%")
    print(f"     （短いEdit/Hook/Introは“合う所が強ければ”同一音源。MV外の区間はフィラーに回します）")

    if not seg_plan:
        print(f"  ⚠️  MVと対応する区間が見つからず → 原曲MVをそのまま流します")
        make_plain_mv_sync(video_path, music_path, output_path, tmp_dir)
        mb = output_path.stat().st_size / 1024 / 1024
        print(f"  ✅ 完成: {output_path.name} ({mb:.1f} MB)")
        return

    # 同一音源判定：全体の平均ではなく「合う所がどれだけ強いか」で見る。
    #   ・p80が高い（大半が高精度で一致）   …通常のEdit/原曲
    #   ・または強一致率が一定以上（一部しかMVに無くても、その一部が確実に一致）
    #     →短いHook/Intro/Short Editでも同一音源と正しく判定できる
    same_source = (p80 >= 0.78) or (strong_frac >= 0.30)
    _content_align_used = False   # 音内容アライン採用時のみ末尾ズレ補正を効かせる
    if not same_source:
        # 生波形が合わない＝別マスターの可能性。だが同じ曲ならクロマ(メロディ)は合う。
        print(f"  ↪︎ 生波形では合わない（強一致率{strong_frac*100:.0f}%）→ クロマ（メロディ）で合わせ直します...")
        cseg_plan, cconf, cuniq = align_segments(video_audio, music_audio)
        cmatch = (sum(e - s for s, e, mv in cseg_plan if mv is not None) / music_dur) if music_dur > 0 else 0.0
        cn = sum(1 for s, e, mv in cseg_plan if mv is not None)
        chroma_ok = (cn > 0 and cconf >= 0.50 and cmatch >= 0.40)
        is_rmx = bool(_is_remix_word and not is_quasi_original)

        if chroma_ok and cuniq >= 0.12:
            # メロディが一意に合う（別マスターでも素直に対応）→ そのまま採用（令和ver等）
            print(f"  ✅ クロマで一致（スコア{cconf:.2f} / 一致率{cmatch*100:.0f}% / 一意性{cuniq:.2f}）→ メロディ基準で配置")
            seg_plan = cseg_plan
        elif chroma_ok and (_cap := content_align_plan(music_audio, video_audio, music_dur, vid_dur, 11025)):
            # 音内容アライン（6/24 fix_mv_align 方式）。slope≈1 の連続ランだけ使うので
            # 反復のドリフト・末尾巻き戻りが出ない。リミックスにも“原曲＋イントロ”なEditにも有効。
            print(f"  ↪︎ 音内容アラインで配置（slope≈1ラン {len(_cap)}区間 / 連続前進）")
            seg_plan = _cap
            _content_align_used = True
        else:
            # クロマも安定ランも無い。確立済みの設計方針に従う:
            #   ・テンポ差Remix（構成も別）→ リップシンク → ダメなら原曲MV
            #   ・原曲/Intro/Extended系 → 無理に口パクせず “原曲MVをそのまま流す”
            if is_rmx:
                print(f"  ⚠️ クロマでも合わない（{cconf:.2f}）→ リップシンクに切替")
                if _try_vocal_lipsync(music_path, video_path, output_path, tmp_dir, music_dur):
                    return
                print(f"  → リップシンクも弱い → 原曲MVをそのまま流します")
            else:
                print(f"  ⚠️ クロマでも合わない（{cconf:.2f} / 一意性{cuniq:.2f}）→ 原曲MVをそのまま流します（原曲MV主体）")
            make_plain_mv_sync(video_path, music_path, output_path, tmp_dir)
            try:
                mb = output_path.stat().st_size / 1024 / 1024
                print(f"  ✅ 完成: {output_path.name} ({mb:.1f} MB)")
            except Exception:
                pass
            return

    # 波形が指すプランをそのまま採用（並べ替え・hook-firstは自然に出る）
    print(f"\n🎬 映像構成プラン（波形追従・{len(seg_plan)}区間）:")
    for s, e, mv in seg_plan:
        if mv is None:
            print(f"  曲 {s:6.1f}〜{e:6.1f}秒 : フィラー映像（MVに無い区間）")
        else:
            print(f"  曲 {s:6.1f}〜{e:6.1f}秒 : MVの {mv:.1f}秒〜")
    seg_rates = [1.0] * len(seg_plan)
    if _content_align_used:
        # 区間末尾ズレ→内部で微伸縮: 各区間を、次区間のMV開始（最終区間はMV終端＝フィナーレ）に
        # 滑らかに着地するよう内部だけ伸縮。継ぎ目の小ジャンプを消し、最後はMVの終わりで終わる。
        for i, (s, e, mv) in enumerate(seg_plan):
            if mv is None:
                continue
            dur = e - s
            if dur < 1.0:
                continue
            is_last = (i == len(seg_plan) - 1)
            if is_last:
                continue                         # 最終区間は等速のまま（無理にフィナーレへ着地させない）
            nxt = seg_plan[i+1][2] if seg_plan[i+1][2] is not None else vid_dur
            span = nxt - mv                      # この区間でカバーすべきMVの長さ
            if span <= 0.1:
                continue
            r = span / dur
            if 0.85 <= r <= 1.18 and abs(r - 1.0) > 0.015:   # 中間区間の継ぎ目だけ小補正
                seg_rates[i] = r
                print(f"  🩹 区間{i} 末尾ズレ {span - dur:+.1f}秒 → 内部で×{r:.3f} 伸縮して補正")


    # セグメント生成
    seg_files = []
    for si, (s, e, mv) in enumerate(seg_plan):
        dur = e - s
        if dur < 0.2: continue
        out_seg = tmp_dir / f"seg_{si:03d}.mp4"
        if mv is None:
            make_filler_segment(video_path, dur, out_seg, tmp_dir)
        else:
            # MVの該当箇所を切り出し。rで内部伸縮する場合は「消費するMV秒 = r*dur」で判定する。
            r = seg_rates[si]
            avail = max(0.0, vid_dur - mv)
            src_want = max(0.1, r * dur)          # この区間が消費すべきMV秒
            if avail < 0.5:
                make_filler_segment(video_path, dur, out_seg, tmp_dir)
            elif src_want <= avail + 0.05:
                # rで伸縮すればMV内に収まる（r=1なら等速）
                if abs(r - 1.0) > 0.002:
                    run(["ffmpeg","-y","-ss",f"{mv:.3f}","-t",f"{src_want:.3f}","-i",str(video_path),
                         "-vf",f"setpts=PTS/{r:.6f},{VF_NORM}",*ENC_ARGS,"-an",str(out_seg)])
                else:
                    run(["ffmpeg","-y","-ss",f"{mv:.3f}","-i",str(video_path),
                         "-t",f"{dur:.3f}","-vf",VF_NORM,*ENC_ARGS,"-an",str(out_seg)])
            else:
                # rで伸縮してもMVが足りない → 使える分を出し、残りを埋める
                out_v_dur = avail / r             # 出力での長さ
                part_v = tmp_dir / f"seg_{si:03d}_v.mp4"
                if abs(r - 1.0) > 0.002:
                    run(["ffmpeg","-y","-ss",f"{mv:.3f}","-t",f"{avail:.3f}","-i",str(video_path),
                         "-vf",f"setpts=PTS/{r:.6f},{VF_NORM}",*ENC_ARGS,"-an",str(part_v)])
                else:
                    run(["ffmpeg","-y","-ss",f"{mv:.3f}","-i",str(video_path),
                         "-t",f"{avail:.3f}","-vf",VF_NORM,*ENC_ARGS,"-an",str(part_v)])
                rest = max(0.1, dur - out_v_dur)
                part_f = tmp_dir / f"seg_{si:03d}_f.mp4"
                if _content_align_used:
                    # 音内容アライン: MVが尽きた残りは、フェス/静止ではなくMV内の別の映像を流す
                    # （MVの頭の方から rest 秒。MV尺が足りなければ0秒から）
                    fb_start = max(0.0, min(mv * 0.5, vid_dur - rest - 0.1))
                    run(["ffmpeg","-y","-ss",f"{fb_start:.3f}","-i",str(video_path),
                         "-t",f"{rest:.3f}","-vf",VF_NORM,*ENC_ARGS,"-an",str(part_f)])
                else:
                    make_filler_segment(video_path, rest, part_f, tmp_dir)
                lst = tmp_dir / f"seg_{si:03d}_l.txt"
                with open(lst, "w") as f:
                    f.write(f"file '{part_v}'\nfile '{part_f}'\n")
                run(["ffmpeg","-y","-f","concat","-safe","0",
                     "-i",str(lst),"-c","copy",str(out_seg)])
        seg_files.append(out_seg)

    print(f"\n🔗 結合・合成中...")
    combined = tmp_dir / "combined.mp4"
    if len(seg_files) == 1:
        shutil.copy(seg_files[0], combined)
    else:
        concat_segments(seg_files, combined, tmp_dir)

    combined = ensure_video_length(combined, music_dur, video_path, tmp_dir)

    run(["ffmpeg","-y",
         "-i",str(combined),"-i",str(music_path),
         "-map","0:v:0","-map","1:a:0",
         "-c:v","copy","-c:a","aac","-b:a","320k",
         "-t",f"{music_dur:.3f}","-movflags","+faststart",
         str(output_path)])

    mb = output_path.stat().st_size / 1024 / 1024
    out_dur = get_duration(output_path)
    print(f"  ✅ 完成: {output_path.name} ({mb:.1f} MB / {out_dur:.2f}秒 = 音声{music_dur:.2f}秒)")

# ============================================================
# Remix カット編集モード
#   前提: 同期が最優先。通常同期(process_with_youtube)が破綻した時だけここに来る。
#   方針: 歌詞ズレを避けリリックビデオは不使用。フェス映像も使わない。
#         「その曲の関連動画」(ライブ/別バージョンMV)を集めて
#         原曲MVと混ぜ、ビートに合わせてカット編集する。
#   解析: HPSS分離 + ダウンビート補正 + 小節グリッドスナップ
# ============================================================

def download_yt_video(url, out_path):
    """yt-dlpで動画1本DL。失敗理由も表示する。成功時パス、失敗時None。"""
    if not url or "watch?v=" not in url:
        print(f"        yt-dlp: 無効なURL: {url!r}")
        return None
    cmd = [
        "yt-dlp", *YTDLP_COOKIE_ARGS,
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "--no-playlist", "--merge-output-format", "mp4",
        "-o", str(out_path), url
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    cands = list(out_path.parent.glob(out_path.stem + ".*"))
    if cands:
        return cands[0]
    for line in r.stderr.splitlines():
        if "ERROR" in line or "error" in line.lower():
            print(f"        yt-dlp: {line.strip()}")
            break
    return None

def collect_related_videos(artist, core_song, mv_url, tmp_dir):
    """
    その曲の関連動画を集める: ライブ映像・別バージョンMV。
    各カテゴリ1本ずつ、DL成功したものだけリストで返す。
    mv_url は原曲MVのURL（重複回避用）。
    戻り値: [Path, ...]（0本もありうる）
    """
    rdir = tmp_dir / "related_dl"; rdir.mkdir(exist_ok=True)
    mv_id = ""
    if mv_url and "watch?v=" in mv_url:
        mv_id = mv_url.split("watch?v=")[-1].split("&")[0]

    # カテゴリごとの検索クエリ（原曲名＋アーティストで探す）
    base = f"{artist} {core_song}".strip() if artist else core_song
    categories = [
        ("ライブ",   [f"{base} live", f"{base} live performance"]),
        ("別MV",     [f"{base} official video", f"{base}"]),
    ]

    collected = []
    used_ids = {mv_id} if mv_id else set()
    for label, queries in categories:
        got = False
        for q in queries:
            if got: break
            for r in search_youtube_mv(q, n=5):
                rid = r.get("id")
                if not rid or rid in used_ids:
                    continue
                # リリックビデオは歌詞がズレるので除外（タイトルにlyric/歌詞を含むもの）
                tl = r.get("title", "").lower()
                if any(kw in tl for kw in ("lyric", "lyrics", "歌詞", "lyric video")):
                    continue
                # 極端に長い動画(30分超)は配信アーカイブ等の可能性が高いので除外
                try:
                    dur = float(r.get("duration") or 0)
                except (ValueError, TypeError):
                    dur = 0
                if dur > 1800:
                    continue
                lurl = f"https://www.youtube.com/watch?v={rid}"
                out = rdir / f"rel_{label}_{rid}.mp4"
                print(f"     [{label}] {r['title']}")
                p = download_yt_video(lurl, out)
                if p:
                    collected.append(p)
                    used_ids.add(rid)
                    got = True
                    break
        if not got:
            print(f"     [{label}] 取得できず（スキップ）")
    return collected

def analyze_remix_blocks(music_path, music_dur):
    """
    HPSS分離 + ダウンビート補正 + 小節グリッドで楽曲を解析。
    戻り値: (bar_times[小節境界の秒リスト], bar_perc[各小節のドラム音圧0-1], norm_bpm)
    HPSSに失敗した場合は通常RMS + 等間隔グリッドにフォールバック。
    """
    try:
        import librosa
        sr = 22050
        y, _ = librosa.load(str(music_path), sr=sr, mono=True)

        y_h, y_p = librosa.effects.hpss(y)
        # オンセット強度（ドラムの打点）を計算してビート検出の基礎にする
        onset_env = librosa.onset.onset_strength(y=y_p, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y_p, sr=sr, onset_envelope=onset_env)
        bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])

        # 実際の打点（オンセット）の時刻リスト。後で小節アタマをここにスナップする
        onset_frames = librosa.onset.onset_detect(y=y_p, sr=sr, onset_envelope=onset_env)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        norm_bpm = bpm if bpm > 0 else 120.0
        while norm_bpm > 160: norm_bpm /= 2
        while norm_bpm < 80:  norm_bpm *= 2

        beat_times = librosa.frames_to_time(beats, sr=sr)

        if len(beat_times) >= 8:
            # ダウンビート補正: 最初の4拍でドラムが最強の拍を1拍目に
            rms_per_beat = []
            for i in range(len(beat_times) - 1):
                s = int(beat_times[i] * sr); e = int(beat_times[i+1] * sr)
                seg = y_p[s:e]
                rms_per_beat.append(float(np.sqrt((seg**2).mean())) if len(seg) > 0 else 0.0)
            first4 = rms_per_beat[:4] if len(rms_per_beat) >= 4 else [0]
            downbeat_offset = int(np.argmax(first4))
            bar_times = list(beat_times[downbeat_offset::4])
        else:
            beat_len = 60.0 / norm_bpm
            bar = 4 * beat_len
            start = float(beat_times[0]) if len(beat_times) else 0.0
            n_bars = max(1, int((music_dur - start) / bar))
            bar_times = [start + i * bar for i in range(n_bars)]

        if not bar_times:
            bar_times = [0.0]

        # --- 小節アタマを実際の打点(オンセット)にスナップ ---
        # 検出ビートは実際のキックから数十msズレることがあるので、
        # 各小節アタマを近傍(±60ms)の最寄り打点に吸着させてカットのキレを出す
        if len(onset_times) > 0:
            snapped = []
            for bt in bar_times:
                cand = onset_times[np.argmin(np.abs(onset_times - bt))]
                if abs(cand - bt) <= 0.060:   # 60ms以内なら打点に吸着
                    snapped.append(float(cand))
                else:
                    snapped.append(float(bt))  # 近くに打点がなければ元のまま
            # 単調増加を保証（重複・逆転を除去）
            bar_times = []
            for t in snapped:
                if not bar_times or t > bar_times[-1] + 0.05:
                    bar_times.append(t)

        if bar_times[-1] < music_dur:
            bar_times.append(music_dur)

        bar_perc = []
        for i in range(len(bar_times) - 1):
            s = int(bar_times[i] * sr); e = int(bar_times[i+1] * sr)
            seg = y_p[s:e]
            bar_perc.append(float(np.sqrt((seg**2).mean())) if len(seg) > 0 else 0.0)
        peak = max(bar_perc) if bar_perc and max(bar_perc) > 0 else 1.0
        bar_perc = [v / peak for v in bar_perc]
        return bar_times, bar_perc, norm_bpm

    except Exception as ex:
        print(f"     ⚠️  HPSS解析失敗（{ex}）→ 通常RMSにフォールバック")
        mono = wav_to_array_path(music_path, sr=11025)
        fb2, beat_len, bpm2, _ = get_song_grid(music_path, mono, 11025)
        nb = bpm2
        while nb > 160: nb /= 2
        while nb < 80:  nb *= 2
        beat_len = 60.0 / nb
        bar = 4 * beat_len
        n_bars = max(1, int((len(mono)/11025 - fb2) / bar))
        bar_times = [fb2 + i * bar for i in range(n_bars)]
        if bar_times[-1] < music_dur:
            bar_times.append(music_dur)
        bar_perc = []
        for i in range(len(bar_times) - 1):
            s = int(bar_times[i] * 11025); e = int(bar_times[i+1] * 11025)
            seg = mono[s:e]
            bar_perc.append(float(np.sqrt((seg**2).mean())) if len(seg) > 0 else 0.0)
        peak = max(bar_perc) if bar_perc and max(bar_perc) > 0 else 1.0
        bar_perc = [v / peak for v in bar_perc]
        return bar_times, bar_perc, nb

def find_vocal_matches(remix_y, orig_y, sr, chunk_sec=4.0):
    """
    ボーカルベースの部分一致検索（構成組み替え対応・安全弁つき）。
    戻り値: [{"r_start","r_end","o_start"(or None),"speed","conf"}, ...]
    """
    import librosa
    from scipy.signal import correlate
    remix_h, _ = librosa.effects.hpss(remix_y)
    orig_h, _  = librosa.effects.hpss(orig_y)

    bpm_r, _ = librosa.beat.beat_track(y=remix_y, sr=sr)
    bpm_o, _ = librosa.beat.beat_track(y=orig_y, sr=sr)
    bpm_r = float(bpm_r) if np.ndim(bpm_r)==0 else float(bpm_r[0])
    bpm_o = float(bpm_o) if np.ndim(bpm_o)==0 else float(bpm_o[0])
    speed_ratio = bpm_r/bpm_o if bpm_o > 0 else 1.0

    try:
        orig_h_stretched = librosa.effects.time_stretch(orig_h, rate=speed_ratio)
    except Exception:
        orig_h_stretched = orig_h

    chunk_samples = int(chunk_sec*sr)
    matches = []

    # 安全弁1: 全体の最大RMSで足切りライン
    rms_all = [np.sqrt(np.mean(remix_h[i:i+chunk_samples]**2))
               for i in range(0, len(remix_h), chunk_samples)
               if len(remix_h[i:i+chunk_samples]) > 0]
    max_rms = max(rms_all) if rms_all else 1.0

    for i in range(0, len(remix_h), chunk_samples):
        chunk = remix_h[i:i+chunk_samples]
        if len(chunk) < sr: break
        rs = i/sr; re_ = (i+len(chunk))/sr
        chunk_rms = np.sqrt(np.mean(chunk**2))

        # 安全弁1: ボーカルが小さい区間はVJモード
        if chunk_rms < max_rms * 0.15:
            matches.append({"r_start":rs,"r_end":re_,"o_start":None,"speed":speed_ratio,"conf":0.0})
            continue

        try:
            corr = correlate(orig_h_stretched, chunk, mode='valid')
            pk = int(np.argmax(corr))
            # 安全弁2: ピークの鋭さ（2番目に高いピークと比べて明確か）
            radius = int(sr*1.5)
            masked = corr.copy()
            masked[max(0,pk-radius):min(len(masked),pk+radius)] = 0
            peak_val = float(corr[pk])
            second = float(np.max(masked)) if len(masked)>0 else 0.0
            orig_slice = orig_h_stretched[pk:pk+len(chunk)]
            norm = (np.linalg.norm(orig_slice)*np.linalg.norm(chunk)) + 1e-9
            conf = peak_val / norm
        except Exception:
            conf = 0.0; pk = 0; peak_val = 0.0; second = 1.0

        if conf > 0.35 and (peak_val > second*1.25):
            orig_start = (pk/sr) * speed_ratio
            matches.append({"r_start":rs,"r_end":re_,"o_start":orig_start,"speed":speed_ratio,"conf":float(conf)})
        else:
            matches.append({"r_start":rs,"r_end":re_,"o_start":None,"speed":speed_ratio,"conf":float(conf)})

    # 安全弁4: 連続するチャンクの結合（長回し優先）
    merged = []
    for m in matches:
        if not merged:
            merged.append(m); continue
        prev = merged[-1]
        if prev["o_start"] is not None and m["o_start"] is not None:
            expected = prev["o_start"] + (prev["r_end"]-prev["r_start"])
            if abs(m["o_start"]-expected) < 0.5:
                prev["r_end"] = m["r_end"]; continue
        if prev["o_start"] is None and m["o_start"] is None:
            prev["r_end"] = m["r_end"]; continue
        merged.append(m)
    return merged

def get_vj_effect_filter(bar_times, start_sec, end_sec):
    """セグメント区間内の小節頭でフラッシュ（明るさ）を発生させるFFmpegフィルタ文字列。
    ※zoompanは時刻ベースの式(between/t)を扱えずffmpegでエラーになるため不使用。
      明るさ(eq brightness)のみ時刻ベースで動くのでフラッシュだけ採用。"""
    beats_in_seg = [t - start_sec for t in bar_times if start_sec <= t < end_sec]
    if not beats_in_seg:
        return ""
    conds = "+".join([f"between(t,{t:.3f},{t+0.10:.3f})" for t in beats_in_seg])
    eq = f"eq=brightness='if({conds},0.18,0.0)'"
    return eq

def make_hybrid_vj_remix(music_path, mv_path, mv_url, loop_path, output_path, tmp_dir):
    """
    ハイブリッドVJ方式（Gemini提案）:
      チャンク単位でボーカル一致を判定し、
      ・一致(信頼度>0.35) → リップシンクモード（原曲MVをBPM比でストレッチ）
      ・不一致/ドロップ   → VJフラッシュモード（速いカット＋ズーム/フラッシュ）
    """
    import librosa, random as _r
    print(f"\n  🎛  ハイブリッドVJモード（実験）")
    music_dur = get_audio_duration_accurate(music_path)
    sr = 22050

    # 原曲MVの音声を抽出（リップシンク照合用）
    orig_wav = tmp_dir / "orig_for_match.wav"
    run(["ffmpeg","-y","-i",str(mv_path),"-ar",str(sr),"-ac","1",str(orig_wav)])
    try:
        remix_y, _ = librosa.load(str(music_path), sr=sr, mono=True)
        orig_y,  _ = librosa.load(str(orig_wav),  sr=sr, mono=True)
    except Exception as ex:
        print(f"  ⚠️  音声読み込み失敗（{ex}）→ 通常カット編集にフォールバック")
        return make_plain_mv_sync(mv_path, music_path, output_path, tmp_dir)

    # 小節グリッド（エフェクトのキックタイミング用）
    bar_times, bar_perc, norm_bpm = analyze_remix_blocks(music_path, music_dur)
    beat_len = 60.0/norm_bpm if norm_bpm > 0 else 0.5

    print(f"  🔬 ボーカル一致を解析中（チャンク照合）...")
    matches = find_vocal_matches(remix_y, orig_y, sr, chunk_sec=4.0)
    n_lip = sum(1 for m in matches if m["o_start"] is not None)
    print(f"     セグメント {len(matches)}個中 リップシンク採用 {n_lip}個 / VJ {len(matches)-n_lip}個")
    if matches:
        confs = [m["conf"] for m in matches]
        print(f"     信頼度: 平均 {sum(confs)/len(confs):.3f} / 最大 {max(confs):.3f}")

    # VJモード(リップシンク不採用)の長いセグメントは2拍ごとに刻んでカットアップにする。
    # リップシンク採用セグメントは口パクを保つため刻まず長回しのまま。
    cut_unit = beat_len * 2  # 2拍ごと
    expanded = []
    for m in matches:
        seg_len = m["r_end"] - m["r_start"]
        if m["o_start"] is None and seg_len > cut_unit * 1.5:
            # VJモード長尺 → cut_unitごとに分割
            t = m["r_start"]
            while t < m["r_end"] - 0.05:
                te = min(t + cut_unit, m["r_end"])
                expanded.append({"r_start": t, "r_end": te, "o_start": None,
                                 "speed": m["speed"], "conf": m["conf"]})
                t = te
        else:
            expanded.append(m)
    matches = expanded
    print(f"     カットアップ展開後: {len(matches)}カット（2拍刻み）")

    mv_dur = get_duration(mv_path) if mv_path else 0
    seg_files = []
    for si, m in enumerate(matches):
        rs = m["r_start"]; re_ = m["r_end"]; orig_start = m["o_start"]; ratio = m["speed"]
        seg_dur = re_ - rs
        if seg_dur < 0.2: continue
        out_seg = tmp_dir / f"hyb_{si:03d}.mp4"

        if orig_start is not None and mv_dur > 0.5:
            # リップシンクモード: 原曲MVの該当箇所をBPM比でストレッチ
            src_len = seg_dur * ratio  # ストレッチ前の原曲尺
            ss = min(max(0.0, orig_start), max(0.0, mv_dur - src_len - 0.1))
            pts = 1.0 / ratio if ratio > 0 else 1.0
            print(f"     [{rs:5.1f}〜] 🎤 リップシンク (MV {orig_start:5.1f}s, x{ratio:.2f})")
            vf = f"setpts={pts:.4f}*PTS,{VF_NORM}"
            run(["ffmpeg","-y","-ss",f"{ss:.3f}","-i",str(mv_path),
                 "-t",f"{src_len:.3f}","-vf",vf,*ENC_ARGS,"-an",str(out_seg)])
        else:
            # VJモード: MVのランダム位置を切り出し（エフェクトなし・シンプルなカット）
            if mv_dur > seg_dur + 1:
                ss = _r.uniform(0, mv_dur - seg_dur - 0.5)
            else:
                ss = 0.0
            run(["ffmpeg","-y","-ss",f"{ss:.3f}","-i",str(mv_path),
                 "-t",f"{seg_dur:.3f}","-vf",VF_NORM,*ENC_ARGS,"-an",str(out_seg)])

        if out_seg.exists() and out_seg.stat().st_size > 0:
            seg_files.append(out_seg)

    if not seg_files:
        print(f"  ⚠️  セグメント生成失敗 → 通常カット編集にフォールバック")
        return make_plain_mv_sync(mv_path, music_path, output_path, tmp_dir)

    print(f"\n  🔗 結合・合成中...")
    combined = tmp_dir / "hyb_combined.mp4"
    if len(seg_files) == 1:
        shutil.copy(seg_files[0], combined)
    else:
        concat_segments(seg_files, combined, tmp_dir)
    _pad_src = mv_path if (mv_path and get_duration(mv_path) > 0.5) else Path("__NOFILL__")
    combined = ensure_video_length(combined, music_dur, _pad_src, tmp_dir)
    run(["ffmpeg","-y","-i",str(combined),"-i",str(music_path),
         "-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","320k",
         "-t",f"{music_dur:.3f}","-movflags","+faststart",str(output_path)])
    mb = output_path.stat().st_size/1024/1024
    print(f"  ✅ 完成: {output_path.name} ({mb:.1f} MB)")

def analyze_video_bpm(video_path):
    """動画の音声からBPMを検出する。音声無し/失敗時はEDM標準の128.0を返す。"""
    try:
        import librosa
        cmd = ["ffmpeg","-y","-i",str(video_path),"-t","40","-ac","1","-ar","22050","-f","wav","-"]
        r = subprocess.run(cmd, capture_output=True)
        if not r.stdout:
            return 128.0
        import io, soundfile as sf
        try:
            y, srr = sf.read(io.BytesIO(r.stdout))
            y = y.astype(np.float32)
            if srr != 22050:
                y = librosa.resample(y, orig_sr=srr, target_sr=22050)
        except Exception:
            # soundfileで読めない時はnumpyで生PCM化（wavヘッダ44byte想定）
            raw = np.frombuffer(r.stdout[44:], dtype=np.int16).astype(np.float32)/32768.0
            y = raw
        if y.size == 0 or np.max(np.abs(y)) < 0.01:
            return 128.0
        tempo, _ = librosa.beat.beat_track(y=y, sr=22050)
        bpm = float(tempo) if np.ndim(tempo)==0 else float(tempo[0])
        if bpm < 60 or bpm > 200:
            return 128.0
        return bpm
    except Exception:
        return 128.0

def make_remix_cut(music_path, mv_path, mv_url, loop_path, output_path, tmp_dir):
    """
    Remix用 2デッキ・スイッチング方式（細切れ禁止・音ズレ防止）。
      Deck A = 原曲MV（そのまま長回し）… 静かな区間
      Deck B = フェス映像(BPM同期) + 関連動画(ライブ/別MV) をランダム … 激しい区間
    セグメントは4〜8小節の長いブロック単位。1〜2秒の細切れはしない。
    """
    import random as _r
    print(f"\n  🎛  2デッキVJモード（Deck A=原曲MV / Deck B=関連動画・ライブ）")
    music_dur = get_audio_duration_accurate(music_path)
    tags = get_metadata(music_path)
    title  = tags.get("title", "")
    artist = tags.get("artist", tags.get("album_artist", ""))
    full_name = f"{artist} {title}".strip() if (artist or title) else music_path.stem
    core_artist, core_song = extract_core_title(artist, title, full_name)

    # --- Deck B素材を集める: 関連動画(ライブ/別MV)のみ。フェス映像は使わない ---
    related = []
    ca = (core_artist or "").strip()
    if len(ca) >= 2:
        print(f"  🔍 関連動画を収集中（{core_artist} {core_song}）...")
        related = collect_related_videos(core_artist, core_song, mv_url, tmp_dir) or []
        if related:
            print(f"     ✅ 関連動画 {len(related)}本 取得")
    else:
        print(f"  ⚠️  アーティスト名不明 → 関連動画はスキップ")

    # Deck B候補リスト = 関連動画のみ（フェス映像は廃止）
    deckB = []  # (path, is_festival)  ※is_festivalは常にFalse
    for rv in related:
        deckB.append((rv, False))
    if not deckB:
        print(f"     ℹ️  関連動画なし → ドロップ区間も原曲MVの別シーンで構成")

    # --- HPSS + ダウンビート補正 + 小節グリッド解析 ---
    print(f"\n  📊 HPSS解析中（ドラム成分で構成を判定）...")
    bar_times, bar_perc, norm_bpm = analyze_remix_blocks(music_path, music_dur)
    n_bars = len(bar_perc)
    if n_bars < 2:
        print(f"  ⚠️  小節解析失敗 → 原曲MVで全編作成")
        make_plain_mv_sync(mv_path, music_path, output_path, tmp_dir); return

    median_perc = float(np.median(bar_perc)) if bar_perc else 0.5
    # 激しい/静か の判定閾値（中央値以上を「激しい=Deck B」とする）
    DROP_TH = median_perc

    song_bpm = norm_bpm
    fest_speed = {}  # フェス廃止につき未使用

    # --- 4〜8小節のマクロブロックに分割 ---
    MIN_BARS = 4
    blocks = []  # (t_start, t_end, is_drop)
    bi = 0
    while bi < n_bars:
        be = min(bi + MIN_BARS, n_bars)
        block_perc = float(np.mean(bar_perc[bi:be]))
        t_s = bar_times[bi]
        t_e = bar_times[be] if be < len(bar_times) else music_dur
        is_drop = block_perc >= DROP_TH
        blocks.append([t_s, t_e, is_drop])
        bi = be

    # 同じ種別が続くブロックは結合（最大8小節相当まで長回し）
    merged = []
    for blk in blocks:
        if merged and merged[-1][2] == blk[2] and (merged[-1][1]-merged[-1][0]) < (8*4*60.0/song_bpm):
            merged[-1][1] = blk[1]
        else:
            merged.append(blk)

    print(f"  🎬 構成（{len(merged)}ブロック・最小{MIN_BARS}小節、BPM {song_bpm:.1f}）:")
    for t_s, t_e, is_drop in merged:
        lbl = "⚡Deck B(関連動画/MV)" if is_drop else "🎬 Deck A(原曲MV)"
        print(f"     {t_s:6.1f}〜{t_e:6.1f}秒 : {lbl}")

    # --- セグメント切り出し ---
    mv_dur = get_duration(mv_path) if mv_path else 0
    seg_files = []
    mv_pos = 0.0
    deckB_pos = {}   # 素材ごとの再生位置
    deckB_turn = 0

    for si, (t_s, t_e, is_drop) in enumerate(merged):
        seg_dur = t_e - t_s
        if seg_dur < 0.3: continue
        out_seg = tmp_dir / f"deck_{si:03d}.mp4"

        use_deckB = is_drop and deckB
        if use_deckB:
            # Deck B: 関連動画（ライブ/別MV）を順にローテーション。等速で長回し
            src_path, _ = deckB[deckB_turn % len(deckB)]
            deckB_turn += 1
            src_dur = get_duration(src_path)
            key = str(src_path)
            pos = deckB_pos.get(key, 0.0)
            if src_dur > 0.5:
                if pos + seg_dur > src_dur: pos = 0.0
                ss = min(pos, max(0.0, src_dur - seg_dur - 0.1))
                run(["ffmpeg","-y","-ss",f"{ss:.3f}","-i",str(src_path),
                     "-t",f"{seg_dur:.3f}","-vf",VF_NORM,*ENC_ARGS,"-an",str(out_seg)])
                deckB_pos[key] = pos + seg_dur
            else:
                # 関連動画が壊れていた → 原曲MVの別シーンで埋める（フェスは使わない）
                if mv_path and mv_dur > 0.5:
                    if mv_pos + seg_dur > mv_dur: mv_pos = 0.0
                    ss = min(mv_pos, max(0.0, mv_dur - seg_dur - 0.1))
                    run(["ffmpeg","-y","-ss",f"{ss:.3f}","-i",str(mv_path),
                         "-t",f"{seg_dur:.3f}","-vf",VF_NORM,*ENC_ARGS,"-an",str(out_seg)])
                    mv_pos += seg_dur
                else:
                    run(["ffmpeg","-y","-f","lavfi",
                     "-i",f"color=c=black:s=1280x720:d={seg_dur:.3f}:r=30",
                     *ENC_ARGS,"-an",str(out_seg)])
        else:
            # Deck A: 原曲MVを順に長回し（等速・エフェクトなし）
            if mv_path and mv_dur > 0.5:
                if mv_pos + seg_dur > mv_dur: mv_pos = 0.0
                ss = min(mv_pos, max(0.0, mv_dur - seg_dur - 0.1))
                run(["ffmpeg","-y","-ss",f"{ss:.3f}","-i",str(mv_path),
                     "-t",f"{seg_dur:.3f}","-vf",VF_NORM,*ENC_ARGS,"-an",str(out_seg)])
                mv_pos += seg_dur
            else:
                run(["ffmpeg","-y","-f","lavfi",
                     "-i",f"color=c=black:s=1280x720:d={seg_dur:.3f}:r=30",
                     *ENC_ARGS,"-an",str(out_seg)])

        if out_seg.exists() and out_seg.stat().st_size > 0:
            seg_files.append(out_seg)

    if not seg_files:
        print(f"  ⚠️  セグメント生成失敗 → 原曲MVで代用")
        make_plain_mv_sync(mv_path, music_path, output_path, tmp_dir); return

    print(f"\n  🔗 結合・合成中...")
    combined = tmp_dir / "deck_combined.mp4"
    if len(seg_files) == 1:
        shutil.copy(seg_files[0], combined)
    else:
        concat_segments(seg_files, combined, tmp_dir)
    _pad_src = mv_path if (mv_path and get_duration(mv_path) > 0.5) else Path("__NOFILL__")
    combined = ensure_video_length(combined, music_dur, _pad_src, tmp_dir)
    run(["ffmpeg","-y","-i",str(combined),"-i",str(music_path),
         "-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","320k",
         "-t",f"{music_dur:.3f}","-movflags","+faststart",str(output_path)])
    mb = output_path.stat().st_size/1024/1024
    print(f"  ✅ 完成: {output_path.name} ({mb:.1f} MB)")


def _onset_env_fine(audio, sr=11025, hop=128):
    nperseg = 1024
    f, t, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=nperseg-hop)
    mag = np.abs(Z)
    log_mag = np.log1p(mag * 100)
    flux = np.diff(log_mag, axis=1, prepend=log_mag[:,:1])
    onset = np.maximum(flux, 0).sum(axis=0)
    if onset.std() > 0:
        onset = (onset - onset.mean()) / onset.std()
    return onset, sr / hop

def detect_beat_len_precise(audio, sr=11025):
    """長距離自己相関+放物線補間で1拍の長さを高精度検出"""
    onset, fps = _onset_env_fine(audio, sr)
    onset = onset[:int(90*fps)]
    ac = correlate(onset, onset, mode='full', method='fft')[len(onset)-1:]
    min_lag = int(fps * 60 / 180)
    max_lag = int(fps * 60 / 70)
    beat_lag = int(np.argmax(ac[min_lag:max_lag])) + min_lag
    bpm = 60 * fps / beat_lag
    while bpm < 90: bpm *= 2
    while bpm >= 180: bpm /= 2
    beat_lag = fps * 60 / bpm

    # できるだけ長いラグ（最大256拍 or 解析範囲の8割）でピーク補間 → 誤差最小化
    m = min(256, int((len(ac) * 0.8) / beat_lag))
    m = max(16, m)
    target = int(beat_lag * m)
    if target + 2 < len(ac):
        w = max(2, int(beat_lag * 0.4))
        seg = ac[target-w : target+w]
        p = int(np.argmax(seg)) + target - w
        if 0 < p < len(ac)-1:
            y0, y1, y2 = ac[p-1], ac[p], ac[p+1]
            denom = (y0 - 2*y1 + y2)
            d = 0.5*(y0 - y2)/denom if abs(denom) > 1e-9 else 0
            p_precise = p + d
        else:
            p_precise = p
        beat_len = (p_precise / fps) / m
    else:
        beat_len = beat_lag / fps
    return beat_len, 60 / beat_len

def _find_first_beat(audio, sr=11025):
    onset, fps = _onset_env_fine(audio, sr)
    th = onset.max() * 0.3
    idx = int(np.argmax(onset > th))
    return idx / fps

def find_loop_candidates(audio, sr, beat_len, loop_bars=8, n_candidates=5):
    """曲全体からループに向いた区間の候補を複数返す [(秒, 類似度), ...]"""
    ch, fps = chroma_for_loop(audio, sr)
    L_frames = int(loop_bars * 4 * beat_len * fps)
    first = _find_first_beat(audio, sr)
    bar_frames = max(1, int(4 * beat_len * fps))

    limit = min(ch.shape[1] - 2*L_frames, int(ch.shape[1] * 0.8))
    scored = []
    t = int(first * fps)
    while t < limit:
        a = ch[:, t:t+L_frames].flatten()
        b = ch[:, t+L_frames:t+2*L_frames].flatten()
        c = float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
        scored.append((t / fps, c))
        t += bar_frames

    # スコア降順、ただし近接候補は除外して多様な位置を出す
    scored.sort(key=lambda x: -x[1])
    min_gap = loop_bars * 4 * beat_len
    picked = []
    for sec, c in scored:
        if all(abs(sec - p[0]) >= min_gap for p in picked):
            picked.append((sec, c))
        if len(picked) >= n_candidates:
            break
    # おすすめ = 最高スコアとほぼ同等(差0.03以内)の中で最も早い位置
    # （イントロループは曲の前半から取るのが自然）
    if picked:
        max_score = max(c for _, c in picked)
        near_best = [(s, c) for s, c in picked if c >= max_score - 0.03]
        best = min(near_best, key=lambda x: x[0])
    else:
        best = (first, 0.0)
    picked.sort(key=lambda x: x[0])
    return picked, best

def chroma_for_loop(audio, sr=11025, hop=2048):
    nperseg = 4096
    f, t, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=nperseg-hop)
    mag = np.abs(Z)
    ch = np.zeros((12, mag.shape[1]))
    for i, freq in enumerate(f):
        if freq < 60 or freq > 4000: continue
        midi = 69 + 12*np.log2(freq/440.0)
        ch[int(round(midi)) % 12] += mag[i]
    n = np.linalg.norm(ch, axis=0, keepdims=True); n[n==0]=1
    return ch/n, sr/hop

def read_serato_grid(path):
    """Seratoが書き込んだbeatgridタグを読む → (最初の拍の秒, BPM) or None"""
    try:
        import base64, struct
        r = subprocess.run(
            ["ffprobe","-v","quiet","-show_entries","format_tags=beatgrid",
             "-of","default=noprint_wrappers=1:nokey=1",str(path)],
            capture_output=True, text=True, errors="replace")
        b64 = "".join(r.stdout.split())
        if not b64: return None
        data = base64.b64decode(b64)
        idx = data.find(b"Serato BeatGrid\x00")
        if idx < 0: return None
        p = idx + len(b"Serato BeatGrid\x00") + 2  # バージョン2バイトスキップ
        count = struct.unpack(">I", data[p:p+4])[0]; p += 4
        if count < 1 or count > 1000: return None
        first_pos = None
        bpm = None
        for i in range(count):
            pos = struct.unpack(">f", data[p:p+4])[0]; p += 4
            if first_pos is None: first_pos = pos
            if i < count - 1:
                p += 4  # beats_till_next
            else:
                bpm = struct.unpack(">f", data[p:p+4])[0]; p += 4
        if bpm is None or bpm <= 0: return None
        return float(first_pos), float(bpm)
    except Exception:
        return None

def read_bpm_tag(path):
    """TBPM/bpm/tmpoタグを読む → float or None"""
    try:
        r = subprocess.run(
            ["ffprobe","-v","quiet","-show_entries",
             "format_tags=TBPM,bpm,tmpo,BPM",
             "-of","default=noprint_wrappers=1",str(path)],
            capture_output=True, text=True, errors="replace")
        for line in r.stdout.strip().splitlines():
            if "=" in line:
                v = line.split("=",1)[1].strip()
                try:
                    b = float(v)
                    if 40 < b < 300: return b
                except ValueError: continue
    except Exception:
        pass
    return None

def get_song_grid(music_path, mono, sr=11025):
    """
    曲のグリッド情報を取得: (最初の拍の秒, 1拍の長さ秒, BPM, ソース名)
    優先順位: Seratoタグ > BPMタグ+オンセット検出 > 完全自動検出
    """
    serato = read_serato_grid(music_path)
    if serato is not None:
        fb, bpm = serato
        # 90未満は倍に正規化（Seratoはハーフテンポで記録することがある）
        norm_bpm = bpm
        while norm_bpm < 90: norm_bpm *= 2
        while norm_bpm >= 180: norm_bpm /= 2
        return fb, 60.0 / norm_bpm, norm_bpm, "Seratoタグ"

    tag_bpm = read_bpm_tag(music_path)
    if tag_bpm is not None:
        norm_bpm = tag_bpm
        while norm_bpm < 90: norm_bpm *= 2
        while norm_bpm >= 180: norm_bpm /= 2
        fb = _find_first_beat(mono, sr)
        return fb, 60.0 / norm_bpm, norm_bpm, "BPMタグ"

    beat_len, bpm = detect_beat_len_precise(mono, sr)
    fb = _find_first_beat(mono, sr)
    return fb, beat_len, bpm, "自動検出"

# ============================================================
# Serato キューポイント自動設定
# ============================================================

def _build_markers2(cues):
    """cues: [(index, ms, rgb_hex, name)] → Serato Markers2 rawバイト"""
    import base64 as _b64
    import struct as _st
    entries = b"\x01\x01"
    color_payload = b"\x00\xff\xff\xff"
    entries += b"COLOR\x00" + _st.pack(">I", len(color_payload)) + color_payload
    for idx, ms, rgb, name in cues:
        color = bytes.fromhex(rgb)
        payload = (b"\x00" + bytes([idx]) + _st.pack(">I", int(ms)) +
                   b"\x00" + color + b"\x00\x00" + name.encode("utf-8") + b"\x00")
        entries += b"CUE\x00" + _st.pack(">I", len(payload)) + payload
    entries += b"BPMLOCK\x00" + _st.pack(">I", 1) + b"\x00"
    entries += b"\x00"
    # 本物のSerato形式: 内側b64は'='パディングなし、512バイト固定長まで\x00埋め
    inner_b64 = _b64.b64encode(entries).rstrip(b"=")
    inner_padded = inner_b64.ljust(512, b"\x00")
    return (b"application/octet-stream\x00\x00Serato Markers2\x00" +
            b"\x01\x01" + inner_padded)

def _build_beatgrid(anchor_sec, bpm):
    import struct as _st
    return (b"application/octet-stream\x00\x00Serato BeatGrid\x00" +
            b"\x01\x00" + _st.pack(">I", 1) +
            _st.pack(">f", float(anchor_sec)) + _st.pack(">f", float(bpm)) +
            b"\x00")


def _parse_markers2_entries(entries):
    """復号済み Serato Markers2 blob から CUE を取り出す → [(index, ms, rgb_hex, name)]"""
    import struct as _st
    cues = []
    i = 0; n = len(entries)
    if entries[:2] == b"\x01\x01":   # 先頭バージョン
        i = 2
    while i < n:
        j = entries.find(b"\x00", i)
        if j < 0:
            break
        name = entries[i:j]
        i = j + 1
        if not name or i + 4 > n:
            break
        ln = _st.unpack(">I", entries[i:i+4])[0]
        i += 4
        data = entries[i:i+ln]
        i += ln
        if name == b"CUE" and len(data) >= 10:
            idx = data[1]
            ms = _st.unpack(">I", data[2:6])[0]
            rgb = data[7:10].hex()
            nm = data[12:].split(b"\x00")[0].decode("utf-8", "replace") if len(data) > 12 else ""
            cues.append((idx, ms, rgb, nm))
    return cues


def read_serato_cues(src_path):
    """ドロップ元ファイル(mp3/m4a/mp4/mov)のSerato Markers2からCUEを読む。
    見つからなければ [] を返す（タグ未書き込み等）。"""
    import base64 as _b64
    p = str(src_path); ext = Path(p).suffix.lower()
    inner_b64 = None
    try:
        if ext == ".mp3":
            from mutagen.id3 import ID3
            try:
                tags = ID3(p)
            except Exception:
                return []
            for fr in tags.getall("GEOB"):
                if getattr(fr, "desc", "") == "Serato Markers2":
                    d = bytes(fr.data)
                    if d[:2] == b"\x01\x01":
                        d = d[2:]
                    inner_b64 = d
                    break
        else:
            from mutagen.mp4 import MP4
            try:
                mp4 = MP4(p)
            except Exception:
                return []
            key = "----:com.serato.dj:markersv2"
            if key in mp4:
                outer = bytes(mp4[key][0])
                blob = _b64.b64decode(b"".join(outer.split()) + b"=" * (-len(b"".join(outer.split())) % 4))
                marker = b"Serato Markers2\x00"
                k = blob.find(marker)
                if k >= 0:
                    d = blob[k+len(marker):]
                    if d[:2] == b"\x01\x01":
                        d = d[2:]
                    inner_b64 = d
    except Exception:
        return []
    if not inner_b64:
        return []
    try:
        s = bytes(inner_b64).replace(b"\x00", b"").replace(b"\n", b"").replace(b"\r", b"").strip()
        s += b"=" * (-len(s) % 4)
        entries = _b64.b64decode(s)
        return _parse_markers2_entries(entries)
    except Exception:
        return []

def detect_song_start(mono, sr, fb, beat_len):
    """
    イントロ明け（曲本体の開始）を検出 → 秒。
    頭4小節が静かならジャンプ位置、そうでなければグリッド第1拍(fb)。
    外す時はfb側に倒す（安全）。
    """
    bar = 4 * beat_len
    n_bars = int((len(mono)/sr - fb) / bar)
    if n_bars < 12: return fb
    rms = []
    for b in range(min(n_bars, 24)):
        s = int((fb + b*bar) * sr); e = int((fb + (b+1)*bar) * sr)
        seg = mono[s:e]
        rms.append(float(np.sqrt((seg**2).mean())) if len(seg) else 0.0)
    rms = np.array(rms)
    peak = rms.max()
    if peak <= 0: return fb
    head_level = rms[:4].mean() / peak
    if head_level >= 0.55:
        return fb
    for b in range(4, min(len(rms)-2, 17)):
        if b % 4 != 0: continue
        before = rms[max(0,b-4):b].mean()
        after  = rms[b:b+4].mean()
        if before <= 0: continue
        if after / before >= 1.4:
            return fb + b * bar
    return fb

def detect_first_chorus_mixout(mono, sr, fb, beat_len):
    """
    1回目のサビの終わり8小節前=ミックスアウト位置を返す（ジャンル汎用版v2）。

    手法: 高エネルギー区間を列挙し、各候補を3要素でスコアリング
      ① エネルギー（サビは曲中で最も音圧が高い）
      ② 長さ（サビは12〜16小節持続が多い、フックは8小節）
      ③ 開始時刻の事前確率（Top40/K-POP/J-POPともサビは20〜80秒に来る。
         15秒未満の区間はイントロ直後のフック/ヴァースの可能性が高い）
    同点なら早い区間（=1回目）を優先。
    """
    bar = 4 * beat_len
    n_bars = int((len(mono)/sr - fb) / bar)
    if n_bars < 24: return None
    rms = np.array([
        float(np.sqrt((mono[int((fb+b*bar)*sr):int((fb+(b+1)*bar)*sr)]**2).mean())
              if int((fb+(b+1)*bar)*sr) <= len(mono) else 0.0)
        for b in range(n_bars)])
    peak = rms.max()
    if peak <= 0: return None

    def find_sections(th):
        """高エネルギー区間の列挙（1小節の凹みは2小節窓最大値で許容）"""
        secs = []
        b = 0
        while b < n_bars - 1:
            if max(rms[b], rms[b+1]) >= peak * th:
                start = b
                while b < n_bars - 1 and max(rms[b], rms[b+1]) >= peak * th:
                    b += 1
                if b - start >= 4:
                    secs.append((start, b))
            else:
                b += 1
        return secs

    def pick_chorus(secs):
        """1回目のサビ = 12小節以上続く最初の区間（実証済みルール）。
        頭サビ型(Gasolina/Top40)もABサビ型(J-POP)も掴め、8小節フックはスキップされる"""
        for s, e in secs:
            if e - s >= 12:
                return (s, e)
        return max(secs, key=lambda x: x[1]-x[0]) if secs else None

    # 二段階方式: 通常閾値(0.78)で検出 → 選ばれた区間が長すぎる(24小節超)場合は
    # 音圧パンパンの音源(ライブ/ラウドネス系)とみなし高閾値(0.85)で構造を再分離
    sections = find_sections(0.78)
    chorus = pick_chorus(sections)
    if chorus is None:
        return None
    if chorus[1] - chorus[0] > 24:
        sections_hi = find_sections(0.85)
        chorus_hi = pick_chorus(sections_hi)
        if chorus_hi is not None and chorus_hi[1] - chorus_hi[0] <= 24:
            chorus = chorus_hi
    s, e = chorus

    mix_bar = max(0, e - 7)
    return fb + mix_bar * bar

def detect_climax_end(mono, sr, fb, beat_len):
    """曲の後半でエネルギーが大きく落ちる4小節境界＝最後の盛り上がりの終わりを検出 → 秒 or None"""
    bar = 4 * beat_len
    n_bars = int((len(mono)/sr - fb) / bar)
    if n_bars < 24: return None
    rms = []
    for b in range(n_bars):
        s = int((fb + b*bar) * sr)
        e = int((fb + (b+1)*bar) * sr)
        seg = mono[s:e]
        rms.append(float(np.sqrt((seg**2).mean())) if len(seg) else 0.0)
    rms = np.array(rms)
    best_b, best_score = None, -1.0
    # 曲の後半50%〜末尾で「直前8小節が大きく、直後8小節が小さい」境界を探す
    for b in range(max(8, int(n_bars*0.5)), n_bars - 4):
        if b % 4 != 0: continue
        before = rms[max(0,b-8):b].mean()
        after  = rms[b:min(n_bars,b+8)].mean()
        score = (before - after) + before * 0.3
        if score > best_score:
            best_score, best_b = score, b
    if best_b is None:
        best_b = max(8, n_bars - 16)  # フォールバック: 末尾16小節前
    return fb + best_b * bar

def detect_chorus(mono, sr, fb, beat_len):
    """エネルギーが跳ね上がる4小節境界をサビ/盛り上がり開始と判定 → 秒 or None"""
    bar = 4 * beat_len
    n_bars = int((len(mono)/sr - fb) / bar)
    if n_bars < 24: return None
    rms = []
    for b in range(n_bars):
        s = int((fb + b*bar) * sr)
        e = int((fb + (b+1)*bar) * sr)
        seg = mono[s:e]
        rms.append(float(np.sqrt((seg**2).mean())) if len(seg) else 0.0)
    rms = np.array(rms)
    best_b, best_score = None, -1.0
    for b in range(8, int(n_bars*0.7)):
        if b % 4 != 0: continue
        before = rms[b-8:b].mean()
        after  = rms[b:b+8].mean()
        score = (after - before) + after * 0.3
        if score > best_score:
            best_score, best_b = score, b
    if best_b is None: return None
    return fb + best_b * bar

def _ensure_mutagen():
    try:
        import mutagen  # noqa
        return True
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mutagen"],
                           check=True, capture_output=True)
            import mutagen  # noqa
            return True
        except Exception:
            return False

# ============================================================
# ウォーターマーク
# ============================================================
WATERMARK_TEXT = "DJ SOPY video edit"
WATERMARK_ENABLED = False   # ← True にすると右下にウォーターマークを焼く（既定OFF）

def _find_font():
    import os
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/HelveticaNeue.ttc",
              "/System/Library/Fonts/Avenir.ttc",
              "/System/Library/Fonts/AvenirNext.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/Library/Fonts/Arial.ttf",
              "/System/Library/Fonts/SFNS.ttf"]:
        if os.path.exists(p):
            return p
    return None

def _ffmpeg_has_drawtext():
    """このffmpegがdrawtext(文字焼き)に対応しているか"""
    try:
        r = subprocess.run(["ffmpeg","-hide_banner","-filters"],
                           capture_output=True, text=True, errors="replace")
        return "drawtext" in r.stdout
    except Exception:
        return False

def _make_rainbow_text_pngs(text, tmp_dir, n_frames=12):
    """虹色グラデの文字PNGをn_frames枚作る（色相をずらしてアニメ用）。
    戻り値: PNGパスのリスト（失敗なら空）"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        try:
            import sys as _sys
            subprocess.run([_sys.executable, "-m", "pip", "install", "Pillow",
                            "--break-system-packages", "-q"], capture_output=True, timeout=120)
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return []
    import colorsys
    fp = _find_font(); fontsize = 33
    try:
        font = ImageFont.truetype(fp, fontsize) if fp else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    tmp_img = Image.new("RGBA", (10,10), (0,0,0,0)); d = ImageDraw.Draw(tmp_img)
    try:
        bbox = d.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        tw, th = len(text)*fontsize//2, fontsize
    pad = 10
    W, H = tw+pad*2, th+pad*2
    paths = []
    ALPHA = 150  # うっすら控えめ
    for fi in range(n_frames):
        img = Image.new("RGBA", (W, H), (0,0,0,0))
        d = ImageDraw.Draw(img)
        # 影
        d.text((pad+2, pad+2), text, font=font, fill=(0,0,0,90))
        # 文字を1字ずつ虹色（色相を文字位置＋フレームでずらす）
        x = pad
        for ci, ch in enumerate(text):
            hue = ((ci/max(len(text),1)) + (fi/n_frames)) % 1.0
            r,g,b = [int(c*255) for c in colorsys.hsv_to_rgb(hue, 0.85, 1.0)]
            d.text((x, pad), ch, font=font, fill=(r,g,b,ALPHA))
            try:
                cb = d.textbbox((0,0), ch, font=font); x += cb[2]-cb[0]
            except Exception:
                x += fontsize//2
        p = tmp_dir / f"wm_rb_{fi:02d}.png"
        img.save(p); paths.append(p)
    return paths

def add_watermark(video_path, times, tmp_dir, text=None):
    """times=[(s,e),...] の区間だけ右下に虹色アニメのウォーターマークを焼く。
    drawtext対応ffmpegなら色相を時間変化、非対応なら虹色PNGを切替アニメ。"""
    if not times:
        return video_path
    label = text if text else WATERMARK_TEXT
    out = tmp_dir / ("wm_" + Path(video_path).name)
    conds = "+".join(f"between(t,{s:.2f},{e:.2f})" for s, e in times)

    if _ffmpeg_has_drawtext():
        # drawtext: 色相を時間で回す虹色（hue=mod(t*60,360)）うっすら(alpha 0.6)
        font = _find_font()
        fontfile = f"fontfile='{font}':" if font else ""
        # fontcolor_expr で時間ごとに色を変える
        draw = (f"drawtext={fontfile}text='{label}':"
                f"fontcolor_expr=%{{eif\\:trunc(128+127*sin(2*PI*t/3))\\:x\\:2}}"
                f"%{{eif\\:trunc(128+127*sin(2*PI*t/3+2))\\:x\\:2}}"
                f"%{{eif\\:trunc(128+127*sin(2*PI*t/3+4))\\:x\\:2}}80:"
                f"fontsize=h/22:x=w-tw-20:y=h-th-20:"
                f"shadowcolor=black@0.4:shadowx=2:shadowy=2:enable='{conds}'")
        r = subprocess.run(["ffmpeg","-y","-i",str(video_path),"-vf",draw,
            "-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p",
            "-c:a","copy",str(out)], capture_output=True)
        if out.exists() and out.stat().st_size > 0:
            return out

    # PNGオーバーレイ: 虹色PNGを複数枚作り、0.25秒ごとに切り替えてアニメ
    pngs = _make_rainbow_text_pngs(label, tmp_dir, n_frames=12)
    if pngs:
        # 各PNGを入力にして、時間で切り替えるoverlayチェーンを作る
        inputs = []
        for p in pngs:
            inputs += ["-i", str(p)]
        # フレーム切替: 0.25秒ごとに次のPNG。mod(t,3)で12枚を3秒ループ
        # 各PNGの表示区間を作る
        n = len(pngs); seg = 0.25
        filt_parts = []
        prev = "[0:v]"
        for i, p in enumerate(pngs):
            # このPNGを表示する条件: サビ区間内 かつ このフレームの番
            frame_cond = f"lt(mod(t/{seg:.3f}\\,{n})\\,{i+1})*gte(mod(t/{seg:.3f}\\,{n})\\,{i})"
            both = f"({conds})*({frame_cond})"
            lbl = f"[v{i}]"
            filt_parts.append(f"{prev}[{i+1}:v]overlay=W-w-20:H-h-20:enable='{both}'{lbl}")
            prev = lbl
        filt = ";".join(filt_parts)
        cmd = ["ffmpeg","-y","-i",str(video_path)] + inputs + \
              ["-filter_complex", filt, "-map", prev, "-map", "0:a?",
               "-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p",
               "-c:a","copy", str(out)]
        r = subprocess.run(cmd, capture_output=True)
        if out.exists() and out.stat().st_size > 0:
            return out
        # 複雑なアニメが失敗したら、虹色1枚を静止表示（最低限）
        filt2 = f"[0:v][1:v]overlay=W-w-20:H-h-20:enable='{conds}'"
        r = subprocess.run(["ffmpeg","-y","-i",str(video_path),"-i",str(pngs[0]),
            "-filter_complex",filt2,"-c:v","libx264","-preset","fast","-crf","18",
            "-pix_fmt","yuv420p","-c:a","copy",str(out)], capture_output=True)
        if out.exists() and out.stat().st_size > 0:
            return out

    return video_path

def build_title_text(music_path):
    """タグ/ファイル名から『アーティスト - 曲名 (Remix名)』形式のタイトル文字列を組み立てる"""
    import re
    tags = get_metadata(music_path)
    title  = (tags.get("title") or "").strip()
    artist = (tags.get("artist") or tags.get("album_artist") or "").strip()

    # Remix名は「タグの曲名」と「ファイル名」両方から探す（タグに入ってないことが多い）
    rmx_kw = r'(remix|rmx|edit|bootleg|rework|refix|flip|mashup|mash-up|bounce|vip|dub|re-?fix|re-?work|mix)'
    def find_remix(s):
        if not s:
            return ""
        # 1) 括弧/角括弧で囲まれたRemix名: (XXX Remix) [XXX Edit]
        m = re.search(r'[\(\[\{]([^\)\]\}]*' + rmx_kw + r'[^\)\]\}]*)[\)\]\}]', s, re.I)
        if m:
            return m.group(1).strip()
        # 2) 括弧なし「- XXX Remix」「 XXX Remix」末尾パターン
        m = re.search(r'[-–—]\s*([^\-–—\(\)\[\]]*' + rmx_kw + r'[^\-–—\(\)\[\]]*)$', s, re.I)
        if m:
            return m.group(1).strip()
        # 3) どこかに "XXX Remix" があれば、その語+直前1語くらい
        m = re.search(r'([A-Za-z0-9!&\'\.]+\s+' + rmx_kw + r')\b', s, re.I)
        if m:
            return m.group(1).strip()
        return ""

    rmx = find_remix(title) or find_remix(music_path.stem)

    # 曲名本体（Remix表記やノイズを除去）。タグ優先、無ければファイル名
    base_for_song = title if (title and title.strip()) else music_path.stem
    song = clean_song_query(strip_filename_noise(base_for_song)).strip()
    # clean_song_queryで取り切れない括弧なしRemix表記を曲名から削る
    if rmx:
        song = re.sub(r'\s*[-–—]?\s*\(?' + re.escape(rmx) + r'\)?\s*$', '', song, flags=re.I).strip()
        song = re.sub(r'[-–—]\s*[^\-–—]*' + rmx_kw + r'[^\-–—]*$', '', song, flags=re.I).strip()

    # タグにアーティストが無ければ "Artist - Title" 形式から取る
    if not artist and " - " in song:
        parts = re.split(r'\s-\s', song, 1)
        artist = parts[0].strip(); song = parts[1].strip()
    # ファイル名からアーティストを補完（タグ空のとき）
    if not artist and " - " in music_path.stem:
        cand = music_path.stem.split(" - ", 1)[0].strip()
        if cand and not re.search(rmx_kw, cand, re.I):
            artist = cand

    # 装飾記号・絵文字を除去（ファイル名整理用の ★ ☆ ◎ ● 【】 などやemoji）
    def _clean_deco(s):
        if not s: return s
        # 記号類を除去（英数字・かな漢字・基本的な区切り( ) - & ' . , スペースは残す）
        s = re.sub(r'[★☆◎●○■□▲▼◆◇♪♫➤➡⇒\*※\u2000-\u206F\u2600-\u27BF\U0001F000-\U0001FAFF]', '', s)
        s = re.sub(r'[【】「」『』〈〉《》〔〕\[\]\{\}]', '', s)
        # よくある日本語ノイズ語
        s = re.sub(r'(公式|オフィシャル|フル|高音質|歌詞付き?|字幕付き?)', '', s)
        return re.sub(r'\s+', ' ', s).strip(' -–—_.')

    artist = _clean_deco(artist)
    song   = _clean_deco(song)
    rmx    = _clean_deco(rmx)

    # 組み立て: アーティスト - 曲名 (Remix名)
    pieces = []
    if artist: pieces.append(artist)
    if song:   pieces.append(("- " + song) if artist else song)
    text = " ".join(pieces).strip()
    if rmx:
        text = f"{text} ({rmx})"
    text = re.sub(r'\s+', ' ', text).strip()
    return text.upper() if text else _clean_deco(music_path.stem).upper()

def make_title_png(text, out_path, W=1280, H=720):
    """カラフルなタイトルテロップの透過PNGを作る（1単語ごとに色違い・縁取りつき）。
    Remix名の (..) は途中改行しない。成功でTrue。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        try:
            import sys as _sys
            subprocess.run([_sys.executable, "-m", "pip", "install", "Pillow",
                            "--break-system-packages", "-q"], capture_output=True, timeout=120)
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return False
    import re, random
    # ネオン系パレット（明るく視認性の良い色）
    NEON = [(255,90,160),(90,200,255),(255,225,90),(130,255,150),(255,150,80),(190,130,255)]
    fp = _find_font()
    fontsize = 54
    try:
        font = ImageFont.truetype(fp, fontsize) if fp else ImageFont.load_default(size=fontsize)
    except Exception:
        try:
            font = ImageFont.load_default(size=fontsize)
        except Exception:
            font = ImageFont.load_default()

    img = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    # トークン化: (..) のかたまりは1トークン（途中改行しない）
    tokens = re.findall(r'\([^)]*\)|\S+', text)
    space_w = draw.textlength(" ", font=font)
    max_w = W - 120
    # 折り返し
    lines = []; cur = []; cur_w = 0
    for tk in tokens:
        tw = draw.textlength(tk, font=font)
        if cur and cur_w + space_w + tw > max_w:
            lines.append(cur); cur = [tk]; cur_w = tw
        else:
            cur.append(tk); cur_w += (space_w + tw if cur_w else tw)
    if cur: lines.append(cur)

    x0, y0 = 60, 70
    line_h = fontsize + 16
    prev_color = None
    for li, line in enumerate(lines):
        x = x0; y = y0 + li * line_h
        for tk in line:
            choices = [c for c in NEON if c != prev_color] or NEON
            c = random.choice(choices); prev_color = c
            # 縁取り(黒)
            for dx in (-2,-1,0,1,2):
                for dy in (-2,-1,0,1,2):
                    if dx or dy:
                        draw.text((x+dx, y+dy), tk, font=font, fill=(0,0,0,210))
            draw.text((x, y), tk, font=font, fill=c+(255,))
            x += draw.textlength(tk, font=font) + space_w
    img.save(out_path)
    return True

def apply_title_overlay(video_path, music_path, tmp_dir):
    """完成動画にカラフルなタイトルテロップを全編オーバーレイする。成功で新パス、失敗で元パス。"""
    text = build_title_text(music_path)
    if not text:
        return video_path
    png = tmp_dir / "title_overlay.png"
    if not make_title_png(text, png):
        print(f"  ⚠️  テロップ生成に失敗（スキップ）")
        return video_path
    out = tmp_dir / ("titled_" + Path(video_path).name)
    # PNGを全編オーバーレイ（音声はそのままコピー）
    r = subprocess.run(["ffmpeg","-y","-i",str(video_path),"-i",str(png),
        "-filter_complex","[0:v][1:v]overlay=0:0",
        "-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p",
        "-c:a","copy","-movflags","+faststart", str(out)], capture_output=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"  🎨 タイトルテロップを追加: {text}")
        return out
    print(f"  ⚠️  テロップ合成に失敗（スキップ）")
    return video_path

def detect_chorus_heads(mono, sr, fb, beat_len):
    """サビ頭（高エネルギー区間の開始小節index）を全部検出"""
    bar = 4 * beat_len
    n_bars = int((len(mono)/sr - fb) / bar)
    if n_bars < 12: return []
    rms = np.array([float(np.sqrt((mono[int((fb+b*bar)*sr):int((fb+(b+1)*bar)*sr)]**2).mean()) if int((fb+(b+1)*bar)*sr)<=len(mono) else 0) for b in range(n_bars)])
    if rms.max() <= 0: return []
    try:
        from scipy.signal import stft as _stft
        f2, t2, Z2 = _stft(mono, fs=sr, nperseg=2048, noverlap=1024)
        mag2 = np.abs(Z2)**2; hi = (f2 >= 2000); fps2 = len(t2)/(len(mono)/sr)
        bright = np.zeros(n_bars)
        for b in range(n_bars):
            fs_=int((fb+b*bar)*fps2); fe_=int((fb+(b+1)*bar)*fps2)
            if fe_<=mag2.shape[1]:
                tot=mag2[:,fs_:fe_].sum()
                bright[b]=mag2[hi][:,fs_:fe_].sum()/tot if tot>0 else 0
        rn = rms/(rms.max()+1e-9); bn = bright/(bright.max()+1e-9)
        inten = rn*(0.55+0.45*bn)
    except Exception:
        inten = rms.copy()
    inten = inten/(inten.max()+1e-9)
    def find_heads(signal, th, min_len=8):
        heads=[]; b=0
        while b < n_bars-1:
            if max(signal[b],signal[b+1])>=th:
                s0=b
                while b<n_bars-1 and max(signal[b],signal[b+1])>=th: b+=1
                if b-s0>=min_len: heads.append(s0)
            else: b+=1
        return heads
    heads = find_heads(inten, 0.78)
    if not heads:
        rn2=rms/(rms.max()+1e-9); heads=find_heads(rn2,0.78)
    if not heads:
        rn2=rms/(rms.max()+1e-9); heads=find_heads(rn2,0.68)
    return heads

def snap_heads_to_8bar(heads, intro_end_bar):
    """サビ頭をIntro Endからの8小節グリッド最寄りにスナップ（音楽理論）"""
    out=[]
    for h in heads:
        head_bar=h+1
        offset=head_bar-intro_end_bar
        if offset<4:
            out.append(h); continue
        nearest=round(offset/8)*8
        out.append(max(0, intro_end_bar+nearest-1))
    return out

def detect_chorus_heads_snapped(mono, sr, fb, beat_len):
    """サビ頭検出→8小節スナップ。ゼロ時はIntro End+32小節から32周期で保険"""
    heads=detect_chorus_heads(mono,sr,fb,beat_len)
    bar=4*beat_len
    n_bars=int((len(mono)/sr-fb)/bar)
    ie=detect_song_start(mono,sr,fb,beat_len)
    ie=ie if ie is not None else fb
    ie_bar=round((ie-fb)/bar)+1
    if heads:
        return sorted(set(snap_heads_to_8bar(heads, ie_bar)))
    fb_list=[]; b=(ie_bar-1)+32
    while b+8<n_bars and len(fb_list)<3:
        fb_list.append(b); b+=32
    return fb_list

def write_serato_cues(mp4_path, orig_music, intro_offset_sec):
    """完成MP4にビートグリッド(BPM)を書き込み、元ファイルのSerato Cueがあれば引き継ぐ。"""
    if not _ensure_mutagen():
        print("  ⚠️  mutagen未導入のためグリッド書き込みをスキップ")
        return
    try:
        mono = wav_to_array_path(orig_music, sr=11025)
        mono = np.asarray(mono, dtype=np.float32).reshape(-1)   # 念のため1次元化（MP4等の形状対策）
        fb, beat_len, bpm, src = get_song_grid(Path(orig_music), mono, 11025)

        # グリッドのアンカー = 第1拍（Extended時は足したイントロ分を加算）
        grid_anchor = intro_offset_sec + fb
        anchor = grid_anchor - int(grid_anchor / beat_len) * beat_len

        from mutagen.mp4 import MP4, MP4FreeForm

        def _serato_b64(raw):
            """Serato MP4形式: base64テキスト・72文字改行・末尾改行"""
            import base64 as _b64
            b = _b64.b64encode(raw)
            lines = [b[i:i+72] for i in range(0, len(b), 72)]
            return b"\n".join(lines) + b"\n"

        mp4 = MP4(str(mp4_path))
        # ビートグリッド(BPM)
        mp4["----:com.serato.dj:beatgrid"] = [MP4FreeForm(_serato_b64(_build_beatgrid(anchor, bpm)))]

        # 元ファイルのCueを引き継ぐ（完成動画の音声は元Remixと同じなので位置はそのまま合う）
        cue_msg = ""
        try:
            cues = read_serato_cues(orig_music)
            if cues:
                off_ms = int(round(intro_offset_sec * 1000))   # Extended化した分だけ後ろへ
                cues2 = [(idx, max(0, ms + off_ms), rgb, name) for (idx, ms, rgb, name) in cues]
                mp4["----:com.serato.dj:markersv2"] = [MP4FreeForm(_serato_b64(_build_markers2(cues2)))]
                cue_msg = f"・Cue {len(cues2)}個を引き継ぎ"
            else:
                cue_msg = "・元にCueなし(引き継ぎなし)"
        except Exception as _ce:
            cue_msg = f"・Cue引き継ぎ失敗({_ce})"

        mp4.save()
        print(f"     BPM {bpm:.2f}・ビートグリッド書き込み済み{cue_msg}")
    except Exception as e:
        print(f"  ⚠️  グリッド書き込み失敗: {e}")

def synth_kick(sr=44100, dur=0.35):
    """909系キックを合成（ピッチが落ちるサイン波）"""
    n = int(dur * sr)
    t = np.arange(n) / sr
    f0, f1, sweep = 150.0, 48.0, 0.09
    k = np.log(f1 / f0) / sweep
    freq = np.where(t < sweep, f0 * np.exp(k * t), f1)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.exp(-t * 8)
    attack_click = np.exp(-t * 300) * 0.4
    return (np.sin(phase) * env + attack_click).astype(np.float32)

def make_beat_extended_audio(music_path, intro_bars, tmp_dir):
    """
    シンプルな4つ打ちキックのイントロ + 原曲 のExtended版WAVを生成。
    """
    SR_OUT = 44100
    XF = int(0.05 * SR_OUT)

    print(f"\n  🎚  Extended化: シンプルビートイントロ{intro_bars}小節を生成中...")

    mono = wav_to_array_path(music_path, sr=11025)
    if len(mono) < 11025 * 20:
        print("  ⚠️  曲が短すぎるためスキップ")
        return None

    fb, beat_len, bpm, src = get_song_grid(music_path, mono, 11025)
    print(f"     BPM: {bpm:.2f}（{src}）/ 最初の拍: {fb:.3f}秒")

    # 原曲ステレオ読み込み
    raw = subprocess.run(
        ["ffmpeg","-y","-i",str(music_path),
         "-f","s16le","-ac","2","-ar",str(SR_OUT),"-"],
        capture_output=True).stdout
    stereo = np.frombuffer(raw, dtype=np.int16).astype(np.float32).reshape(-1, 2) / 32768.0

    song_rms = float(np.sqrt((stereo[:SR_OUT*30]**2).mean()))
    target_rms = max(0.08, song_rms * 0.9)

    # グリッド位相保証: 曲の最初の拍が「1拍目から数えてちょうどN拍目」に来るよう
    # イントロの長さを N*beat - fb にする → 全体が1本のビートグリッドに乗る
    n_beats = intro_bars * 4
    intro_sec = n_beats * beat_len - fb
    while intro_sec <= beat_len:  # fbが大きい曲への保険
        n_beats += 4
        intro_sec = n_beats * beat_len - fb
    intro_len = int(round(intro_sec * SR_OUT))

    intro = np.zeros(intro_len, dtype=np.float32)
    kick = synth_kick(SR_OUT)
    for b in range(n_beats):
        pos = int(round(b * beat_len * SR_OUT))
        if pos >= intro_len: break
        end = min(pos + len(kick), intro_len)
        intro[pos:end] += kick[:end-pos]
    rms = float(np.sqrt((intro**2).mean()))
    if rms > 0:
        intro *= target_rms / rms
    intro = np.clip(intro, -0.99, 0.99)
    intro_st = np.stack([intro, intro], axis=1)

    # 結合（原曲の頭をクロスフェードイン）
    total_len = intro_len + len(stereo)
    out = np.zeros((total_len, 2), dtype=np.float32)
    out[:intro_len] = intro_st
    t_fade = np.linspace(0, np.pi/2, XF, dtype=np.float32)
    song = stereo.copy()
    song[:XF] *= np.sin(t_fade)[:, None]
    out[intro_len : intro_len + len(song)] += song

    peak = np.abs(out).max()
    if peak > 0.99:
        out *= 0.99 / peak

    out_path = tmp_dir / f"{music_path.stem} (Extended).wav"
    pcm = (out * 32767).astype(np.int16)
    import wave
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR_OUT)
        w.writeframes(pcm.tobytes())

    grid_check = (intro_len/SR_OUT + fb) / beat_len
    print(f"     ✅ キックイントロ {intro_len/SR_OUT:.1f}秒 追加（@ {bpm:.2f}BPM）")
    print(f"     ✅ グリッド検証: 曲の1拍目 = イントロ開始から {grid_check:.3f}拍目（整数なら完璧）")
    return out_path

def make_extended_audio(music_path, intro_bars, tmp_dir, loop_bars=8):
    """
    イントロにループを足したExtended版WAVを生成。
    loop_bars(4or8)小節単位のループを繰り返して intro_bars 分のイントロを作り、
    原曲の頭にクロスフェードで接続する。
    戻り値: 生成したWAVのパス（失敗時はNone）
    """
    LOOP_BARS = loop_bars
    SR_OUT = 44100
    XF = int(0.05 * SR_OUT)  # 50msクロスフェード

    print(f"\n  🎚  Extended化: イントロ{intro_bars}小節を生成中...")

    # 解析用モノラル
    mono = wav_to_array_path(music_path, sr=11025)
    if len(mono) < 11025 * 20:
        print("  ⚠️  曲が短すぎるためExtended化をスキップ")
        return None

    beat_len, bpm = detect_beat_len_precise(mono, 11025)
    print(f"     検出BPM: {bpm:.2f}")

    candidates, best = find_loop_candidates(mono, 11025, beat_len, LOOP_BARS)
    if not candidates:
        print("  ⚠️  ループ候補が見つからずスキップ")
        return None
    loop_dur = LOOP_BARS * 4 * beat_len

    bm, bs = int(best[0] // 60), best[0] % 60
    print(f"\n     ループ候補（{LOOP_BARS}小節 = {loop_dur:.1f}秒）:")
    print(f"     [Enter] おすすめ: {bm}:{bs:04.1f}〜 (類似度{best[1]:.2f})")
    for ci, (sec, c) in enumerate(candidates, 1):
        mark = " ★おすすめ" if abs(sec - best[0]) < 0.01 else ""
        mins, secs = int(sec // 60), sec % 60
        print(f"     [{ci}]     {mins}:{secs:04.1f}〜 (類似度{c:.2f}){mark}")
    print(f"     [秒数]  任意の開始秒数の指定も可（例: 45.5）")

    choice = input("\n     選択 > ").strip()
    if choice == "":
        loop_start_sec = best[0]
    else:
        try:
            v = float(choice)
            if v <= len(candidates) and v == int(v) and v >= 1:
                loop_start_sec = candidates[int(v)-1][0]
            else:
                loop_start_sec = v  # 秒数直接指定
        except ValueError:
            loop_start_sec = best[0]
    lm, ls = int(loop_start_sec // 60), loop_start_sec % 60
    print(f"     ✅ ループ区間: {lm}:{ls:04.1f}〜")

    # 出力用ステレオ44.1kHz
    raw = subprocess.run(
        ["ffmpeg","-y","-i",str(music_path),
         "-f","s16le","-ac","2","-ar",str(SR_OUT),"-"],
        capture_output=True).stdout
    stereo = np.frombuffer(raw, dtype=np.int16).astype(np.float32).reshape(-1, 2) / 32768.0

    L = int(round(LOOP_BARS * 4 * beat_len * SR_OUT))   # 8小節のサンプル数
    s = int(round(loop_start_sec * SR_OUT))
    if s + L + XF >= len(stereo):
        print("  ⚠️  ループ区間が確保できずスキップ")
        return None
    loop_seg = stereo[s : s + L + XF].copy()  # XF分余分に取る（フェード用）

    repeats = max(1, int(round(intro_bars / LOOP_BARS)))
    intro_len = repeats * L
    total_len = intro_len + len(stereo)
    out = np.zeros((total_len, 2), dtype=np.float32)

    # イコールパワー・クロスフェード曲線
    t_fade = np.linspace(0, np.pi/2, XF, dtype=np.float32)
    fade_in  = np.sin(t_fade)[:, None]
    fade_out = np.cos(t_fade)[:, None]

    # ループを正確に k*L 位置に配置（グリッド維持）、境界はクロスフェード
    for k in range(repeats):
        pos = k * L
        out[pos : pos + L] += loop_seg[:L]
        # 次のループ頭に向けて尻尾XFをフェードで重ねる
        tail = loop_seg[L : L + XF] * fade_out
        if pos + L + XF <= total_len:
            out[pos + L : pos + L + XF] += tail
        out[pos : pos + XF] *= 1.0 if k == 0 else 1.0  # 加算で自然にブレンド済み
    # ループ境界のフェードイン側
    for k in range(1, repeats):
        pos = k * L
        out[pos : pos + XF] *= fade_in[:,0].reshape(-1,1) * 0 + 1  # no-op保険
    # ↑加算ブレンド方式: 各ループ頭はフルゲイン、前ループ尻尾がfade_outで重なる

    # グリッド位相補正: 曲の最初の拍位置(fb)ぶんイントロを短縮して
    # 全体が1本のビートグリッドに乗るようにする
    fb_grid, _, _, _ = get_song_grid(music_path, mono, 11025)
    fb_samples = int(round(fb_grid * SR_OUT))
    song_pos = max(0, intro_len - fb_samples)

    # 原曲を配置（頭XFはフェードインで重ねる）
    song = stereo.copy()
    song[:XF] *= fade_in
    end_pos = min(total_len, song_pos + len(song))
    out[song_pos : end_pos] += song[:end_pos - song_pos]

    # クリッピング防止
    peak = np.abs(out).max()
    if peak > 0.99:
        out *= 0.99 / peak

    # WAV書き出し
    out_path = tmp_dir / f"{music_path.stem} (Extended).wav"
    pcm = (out * 32767).astype(np.int16)
    import wave
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR_OUT)
        w.writeframes(pcm.tobytes())

    added = intro_len / SR_OUT
    print(f"     ✅ イントロ {added:.1f}秒 追加（{repeats}ループ × {LOOP_BARS}小節）")
    return out_path

def clean_path(p):
    p = p.strip().strip("'\"")
    for old, new in [(r"\ ", " "), (r"\(", "("), (r"\)", ")"), (r"\[", "["),
                     (r"\]", "]"), (r"\&", "&"), (r"\,", ","), (r"\'", "'"),
                     (r"\!", "!"), (r"\$", "$"), (r"\#", "#"), (r"\@", "@")]:
        p = p.replace(old, new)
    return p

# ─── メイン ───
config = load_config()
print()

# yt-dlpのYouTubeボット判定回避: ログイン済みブラウザのCookieを使う（初回のみ選択）
YTDLP_COOKIE_ARGS = configure_cookies(config)

# フェス映像は廃止。MVが足りない区間は「原曲MV（無ければ別MV／黒）」で埋める。
# loop_path はレガシー互換のための未使用パススルー（フィラー源としては使われない）。
loop_path = Path("FESTIVAL")  # ※実際にはフェス取得されない（get_loop_videosで無効化済）

music_list = []
print("\n🎵 音楽ファイルをドラッグ&ドロップ（MP3・m4a・wav等、複数可、空Enterで終了）:")
while True:
    raw = input(f"  音楽ファイル [{len(music_list)+1}]: ")
    if not raw.strip():
        if music_list: break
        print("  ※ 最低1つ入力してください")
        continue
    p = clean_path(raw)
    path = Path(p).expanduser()
    if not path.exists(): print(f"  ❌ 見つかりません: {path}"); continue
    music_list.append(path)
    print(f"  ✅ 追加: {path.name}")

out_input = input(f"\n💾 出力フォルダ（Enter → デスクトップ）:\n> ").strip().strip("'\"")
out_dir = Path(out_input).expanduser() if out_input else Path.home()/"Desktop"
out_dir.mkdir(parents=True, exist_ok=True)

extend_bars = 0
extend_loop_unit = 8
extend_type = "beat"
ext_yn = input(f"\n🎚 Extended化（イントロを追加）しますか？ (y → する / Enter → しない):\n> ").strip().lower()
if ext_yn in ("y", "yes"):
    type_in = input("  イントロのタイプ [Enter → シンプルビート(ドンドン4つ打ち) / 1 → 曲のループ]:\n  > ").strip()
    extend_type = "loop" if type_in == "1" else "beat"

    if extend_type == "beat":
        bars_in = input(f"  イントロの長さ [Enter → 8小節 / 4 → 4小節]:\n  > ").strip()
        extend_bars = 4 if bars_in == "4" else 8
        print(f"  ✅ 全曲に {extend_bars}小節のキックイントロを追加します")
    else:
        unit_in = input("  ループ単位を選択 [Enter → 8小節 / 4 → 4小節]:\n  > ").strip()
        extend_loop_unit = 4 if unit_in == "4" else 8
        bars_in = input(f"  イントロの小節数 [Enter → 32]（{extend_loop_unit}の倍数に丸めます）:\n  > ").strip()
        extend_bars = int(bars_in) if bars_in.isdigit() else 32
        extend_bars = max(extend_loop_unit, round(extend_bars / extend_loop_unit) * extend_loop_unit)
        print(f"  ✅ 全曲に {extend_bars}小節（{extend_loop_unit}小節ループ × {extend_bars // extend_loop_unit}回）のイントロを追加します")

tmp = Path(tempfile.mkdtemp(prefix="djvm_"))

# タイトルテロップを入れるか
title_yn = input(f"\n🎨 タイトルテロップ（アーティスト名・曲名）を入れますか？ (y → 入れる / Enter → 入れない):\n> ").strip().lower()
add_title = title_yn in ("y", "yes")

try:
    for i, music in enumerate(music_list, 1):
        print(f"\n{'─'*54}")
        print(f"  [{i}/{len(music_list)}] {music.name}")
        print(f"{'─'*54}")
        orig_music = music  # メタデータコピー用に元ファイルを保持
        sub_tmp = tmp / f"track_{i}"
        sub_tmp.mkdir()

        # Extended化
        intro_offset = 0.0
        if extend_bars > 0:
            if extend_type == "beat":
                ext = make_beat_extended_audio(music, extend_bars, sub_tmp)
            else:
                ext = make_extended_audio(music, extend_bars, sub_tmp, extend_loop_unit)
            if ext is not None:
                intro_offset = get_audio_duration_accurate(ext) - get_audio_duration_accurate(music)
                music = ext
                out_path = out_dir / f"{Path(music).stem}.mp4"
            else:
                out_path = out_dir / f"{music.stem}.mp4"
        else:
            out_path = out_dir / f"{music.stem}.mp4"

        if is_mashup(music):
            songs = parse_mashup_songs(music)
            if len(songs) >= 2:
                # VJモード: 各曲のMVに切り替え
                make_vj_mashup(music, songs, loop_path, out_path, sub_tmp)
            else:
                print(f"\n  🎛  マッシュアップ検出（曲名解析不可）→ 通常のMV検索で作成")
                urls = choose_video(music)
                process_with_youtube(urls, music, loop_path, out_path, sub_tmp)
        else:
            urls = choose_video(music)
            process_with_youtube(urls, music, loop_path, out_path, sub_tmp)

        # 元ファイルのメタデータ（曲名・アーティスト・BPMタグ）を最終MP4にコピー
        if out_path.exists() and out_path.stat().st_size > 0:
            meta_tmp = out_path.parent / (out_path.stem + ".meta.mp4")
            r = subprocess.run(
                ["ffmpeg","-y","-i",str(out_path),"-i",str(orig_music),
                 "-map","0","-map_metadata","1",
                 "-c","copy","-movflags","+faststart",str(meta_tmp)],
                capture_output=True)
            if r.returncode == 0 and meta_tmp.exists() and meta_tmp.stat().st_size > 0:
                meta_tmp.replace(out_path)
            else:
                meta_tmp.unlink(missing_ok=True)

            # タイトルテロップ（選択時のみ）。元ファイルからアーティスト/曲名を取る
            if add_title:
                _titled = apply_title_overlay(out_path, orig_music, tmp)
                if _titled != out_path and Path(_titled).exists():
                    Path(_titled).replace(out_path)

            # Seratoキューポイント自動設定
            # ウォーターマーク: 曲を三等分し各ブロック頭で7秒ずつ
            try:
                _tdur = get_duration(out_path)
            except Exception:
                _tdur = 0
            _wm_times = []
            if _tdur > 21:
                _blk = _tdur / 3.0
                for _k in range(3):
                    _st = _k * _blk
                    _wm_times.append((_st, min(_st + 7.0, _tdur)))
            elif _tdur > 0:
                _wm_times.append((0.0, min(7.0, _tdur)))
            if WATERMARK_ENABLED and _wm_times:
                _wm_text = "DJ SOPY video/club edit" if extend_bars > 0 else WATERMARK_TEXT
                _wm = add_watermark(out_path, _wm_times, tmp, text=_wm_text)
                if _wm != out_path and Path(_wm).exists():
                    Path(_wm).replace(out_path)
                    print(f"  🏷  ウォーターマーク追加")

            # 100MB以内に収める（音質は維持・画質で調整）
            _fit = fit_to_size(out_path, orig_music, 100, tmp)
            if _fit != out_path and Path(_fit).exists():
                Path(_fit).replace(out_path)

            write_serato_cues(out_path, orig_music, intro_offset)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'='*54}")
print(f"🎉 全{len(music_list)}曲 完了！ → {out_dir}")
if PENDING_WARNINGS:
    print("")
    print("──── ⚠️ 確認が必要な曲 ────")
    for w in PENDING_WARNINGS:
        print(f"  {w}")
    print("──────────────────────")
print(f"{'='*54}")
if os.environ.get("DJVM_WEB") != "1":
    try:
        input("\nEnterキーで閉じる...")
    except EOFError:
        pass
