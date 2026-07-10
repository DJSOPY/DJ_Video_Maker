# ============================================================
#  vocal_sync.py  —  Demucsボーカル分離ベースの Remix → 原曲MV リップシンク
# ------------------------------------------------------------
#  dj_maker_core.py から `import vocal_sync` で読み込む追加モジュール。
#  core 側には依存しない（=importしてもcoreのmainは走らない）独立設計。
#
#  なぜ作るか:
#    既存の find_vocal_matches は HPSS の harmonic成分を「ボーカル代わり」に
#    使っているが、harmonic にはシンセ/ピアノ/ベース等の音程楽器が全部入る。
#    伴奏を差し替えた Remix では harmonic 同士が一致せず破綻する。
#    → 本物のボーカルstem(Demucs)同士を、音程に依存しない「声の包絡(リズム/
#      フレージング)」で相関させることで、キー変更・伴奏差し替えに強くする。
#
#  公開API:
#    make_vocal_lipsync_remix(music_path, mv_path, output_path, tmp_dir,
#                             music_dur, filler_cb=None) -> bool
# ============================================================

import subprocess, shutil, sys, importlib
from pathlib import Path
import numpy as np

SR = 22050

# core 側の規格と必ず一致させること（結合時のズレ防止）
VF_NORM = ("scale=1280:720:force_original_aspect_ratio=decrease,"
           "pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1")
ENC_ARGS = ["-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-video_track_timescale", "15360"]

# 顔優先（歌唱シーン寄せ）の探索窓（秒）。
#   0    = 顔優先オフ（DTWの対応時刻をそのまま使う）← 現在の推奨
#   2.0  = ±2秒だけ顔の良い所に微調整（DTWのスパン整合を崩すため非推奨）
#   12.0 = ±12秒（広すぎて歌詞がズレるため非推奨）
# ※下のスパン整合(setpts)方式では、開始を顔へ寄せると歌詞対応が崩れるため 0 にする。
FACE_PRIORITY_WINDOW = 0.0


# ------------------------------------------------------------
# 低レベルヘルパ（coreに依存しないよう最小限を自前で持つ）
# ------------------------------------------------------------
def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace")


def _load_mono(path, sr=SR, duration=None):
    """ffmpegで mono float32 [-1,1] にデコード"""
    cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", str(path)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-f", "s16le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, errors="replace")
    try:
        return float(r.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def _tail(text, n=15):
    lines = (text or "").strip().splitlines()
    return lines[-n:] if lines else []


# ------------------------------------------------------------
# Demucs ボーカル分離（診断つき）
# ------------------------------------------------------------
def demucs_runtime_status():
    """
    demucs が「実際に実行可能か」を確認する。
    部分インストールや torch/numpy 不整合を検出する。
    戻り値: (ok: bool, detail: str)
    """
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import torch, demucs; print('OK', torch.__version__)"],
            capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and "OK" in r.stdout:
            return True, r.stdout.strip()
        return False, (r.stderr or r.stdout or "unknown error")
    except Exception as e:
        return False, repr(e)


def ensure_demucs(verbose=True):
    """
    demucs を実行可能な状態にする。未導入なら pip install を試みる。
    torch/numpy 不整合などで壊れている場合は、その原因を表示する。
    戻り値: True=実行可能 / False=不可（呼び出し側はHPSSへ）
    """
    ok, detail = demucs_runtime_status()
    if ok:
        return True

    if verbose:
        print("     📦 Demucs(ボーカル分離)を導入中… 初回のみ数分かかります")
    # 1) まず demucs を入れる（torch も依存で入る）
    pip_cmds = [
        [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", "demucs"],
        [sys.executable, "-m", "pip", "install", "-q", "demucs"],  # 古いpip保険
    ]
    last = None
    for cmd in pip_cmds:
        last = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if last.returncode == 0:
            break

    ok, detail = demucs_runtime_status()
    if ok:
        if verbose:
            print(f"     ✅ Demucs 準備OK（{detail}）")
        return True

    # 2) torch が numpy2 系と衝突しているなら、互換torchの再導入を一度だけ試す
    if "numpy" in detail.lower() or "_ARRAY_API" in detail or "binary incompatib" in detail.lower():
        if verbose:
            print("     ⚠️ torch と numpy のバージョン不整合を検出 → 互換版を再導入します")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--break-system-packages", "--upgrade",
                        "torch", "torchaudio"], capture_output=True, text=True, errors="replace")
        ok, detail = demucs_runtime_status()
        if ok:
            if verbose:
                print(f"     ✅ Demucs 準備OK（{detail}）")
            return True

    if verbose:
        print("     ❌ Demucs を実行可能にできませんでした。原因（末尾）:")
        for ln in _tail(detail, 12):
            print("        ", ln)
        if last is not None and last.returncode != 0:
            print("        pip(失敗)出力末尾:")
            for ln in _tail(last.stderr or last.stdout, 6):
                print("        ", ln)
    return False


def have_demucs():
    """軽量チェック（CLI/モジュールの存在のみ）。実行可否は demucs_runtime_status で。"""
    if shutil.which("demucs"):
        return True
    try:
        importlib.import_module("demucs.separate")
        return True
    except Exception:
        return False


def separate_vocals(audio_path, work_dir, sr=SR, max_sec=None, verbose=True,
                    demucs_ready=None):
    """
    音源からボーカルだけを mono np.array で返す。
    戻り値: (vocal_array, method)   method = 'demucs' / 'hpss' / 'raw'
    demucs_ready=False ならDemucsを使わず最初からHPSS。
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 長尺は分離が重いので、必要なら頭から max_sec 秒だけに切る
    src = Path(audio_path)
    if max_sec:
        clip = work_dir / (src.stem + "_clip.wav")
        _run(["ffmpeg", "-v", "quiet", "-y", "-i", str(src), "-t", str(max_sec),
              "-ac", "2", "-ar", "44100", str(clip)])
        if clip.exists() and clip.stat().st_size > 0:
            src = clip

    use_demucs = have_demucs() if demucs_ready is None else demucs_ready
    if use_demucs:
        # --- 決定的化: 同じ音源は分離結果をキャッシュして再利用（毎回まったく同じ結果に） ---
        # Demucsは実行ごとに分離が微妙にブレる→DTWが前半できわどい判定をひっくり返す原因。
        # 音源の中身をハッシュ化し、一度分離したらキャッシュを使い回すことで結果を固定する。
        import hashlib
        try:
            h = hashlib.md5(Path(src).read_bytes()).hexdigest()[:16]
        except Exception:
            h = None
        cache_dir = Path.home() / ".dj_video_maker" / "voc_cache"
        cache_npy = (cache_dir / f"{h}_htdemucs_sr{sr}.npy") if h else None
        if cache_npy and cache_npy.exists() and cache_npy.stat().st_size > 0:
            try:
                arr = np.load(cache_npy)
                if arr.size and float(np.max(np.abs(arr))) > 1e-4:
                    if verbose:
                        print("     🎤 ボーカル分離（キャッシュ再利用＝毎回同じ結果）")
                    return arr, "demucs"
            except Exception:
                pass
        outdir = work_dir / "demucs"
        outdir.mkdir(exist_ok=True)
        # 一番速いデバイスから試す: MPS(Apple GPU) → CUDA → CPU。失敗したら順に落ちる。
        def _demucs_device():
            try:
                import torch
                mps = getattr(torch.backends, "mps", None)
                if mps is not None and mps.is_available():
                    return "mps"
                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
            return "cpu"
        devs = [_demucs_device()]
        if devs[0] != "cpu":
            devs.append("cpu")          # 速いデバイスがダメなら必ずCPUで再試行
        # --two-stems=vocals → vocals.wav と no_vocals.wav の2本だけ（高速）/ --shifts 0 → 決定性
        r = None
        for dev in devs:
            r = _run([sys.executable, "-m", "demucs", "--two-stems=vocals", "--shifts", "0",
                      "-d", dev, "-n", "htdemucs", "--out", str(outdir), str(src)])
            voc = list(outdir.glob("**/vocals.wav"))
            if voc:
                arr = _load_mono(voc[0], sr)
                if arr.size and float(np.max(np.abs(arr))) > 1e-4:
                    # 次回から同じ結果になるよう、解析配列そのものをキャッシュ保存
                    if cache_npy:
                        try:
                            cache_dir.mkdir(parents=True, exist_ok=True)
                            np.save(cache_npy, arr.astype(np.float32))
                        except Exception:
                            pass
                    if verbose:
                        tag = f"（{dev.upper()}）" if dev != "cpu" else ""
                        print(f"     🎤 Demucsでボーカル分離 成功{tag}")
                    return arr, "demucs"
            if verbose and dev != devs[-1]:
                print(f"     ⚠️ Demucs {dev.upper()} で分離できず → CPUで再試行")
        # 失敗 → 本当のエラーを表示してから HPSS へ
        if verbose:
            print("     ⚠️ Demucs分離に失敗 → HPSSにフォールバック。Demucs出力(末尾):")
            for ln in _tail(((r.stderr if r else "") or "") + "\n" + ((r.stdout if r else "") or ""), 12):
                print("        ", ln)

    # フォールバック: HPSS harmonic（=従来の find_vocal_matches と同等の近似）
    try:
        import librosa
        y = _load_mono(src, sr)
        h, _p = librosa.effects.hpss(y)
        return h, "hpss"
    except Exception:
        return _load_mono(src, sr), "raw"


# ------------------------------------------------------------
# 声の「包絡(エンベロープ)」相関でリップシンク区間を探す
# ------------------------------------------------------------
def detect_face_score_timeline(video_path, sample_fps=1.0, max_sec=420, verbose=False):
    """
    原曲MVを sample_fps でサンプリングし、各時刻の「顔アップ度」スコアを返す。
    スコア = 顔面積比 × 中央度（0〜1）。顔なし=0。歌唱シーンほど高くなる傾向。
    OpenCVが無ければ空リストを返す（顔優先なし＝DTWのまま）。
    戻り値: (times[np.array], scores[np.array])  失敗時は (None, None)
    """
    try:
        import cv2
    except Exception:
        if verbose:
            print("     ℹ️ OpenCV未導入 → 顔優先なし（DTW対応のまま使用）")
        return None, None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = min(total / fps if fps > 0 else 0, max_sec)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        times = []; scores = []
        t = 0.0; step = 1.0 / sample_fps
        while t < dur:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                t += step; continue
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                             minSize=(int(w*0.06), int(h*0.06)))
            score = 0.0
            for (fx, fy, fw, fh) in faces:
                area_ratio = (fw * fh) / (w * h)
                cx = fx + fw/2; cy = fy + fh/2
                center_dist = ((cx - w/2)/w)**2 + ((cy - h/2)/h)**2
                center_score = max(0.0, 1.0 - center_dist*4)
                s = area_ratio * (0.5 + 0.5*center_score)
                score = max(score, s)
            times.append(t); scores.append(score)
            t += step
        cap.release()
        if not times:
            return None, None
        return np.array(times), np.array(scores)
    except Exception as e:
        if verbose:
            print(f"     ⚠️ 顔検出に失敗（{e}）→ 顔優先なし")
        return None, None


def _best_face_time(face_times, face_scores, center, window=15.0, min_score=0.01):
    """center時刻の前後window秒で、顔スコアが最大の時刻を返す。
    その窓に顔がほぼ無ければ center をそのまま返す。"""
    if face_times is None or len(face_times) == 0:
        return center
    lo = center - window; hi = center + window
    mask = (face_times >= lo) & (face_times <= hi)
    if not np.any(mask):
        return center
    idx = np.where(mask)[0]
    best_i = idx[int(np.argmax(face_scores[idx]))]
    if face_scores[best_i] < min_score:
        return center  # 窓内に顔が無い → 元の対応時刻
    return float(face_times[best_i])


def _vocal_envelope(y, sr=SR, fps=50):
    """フレームRMSで声の振幅包絡を作る。戻り値: (env[0..1], fps_actual)"""
    hop = max(1, int(sr / fps))
    n = len(y) // hop
    if n < 1:
        return np.zeros(1, dtype=np.float32), float(fps)
    env = np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    mx = float(env.max())
    if mx > 0:
        env = env / mx
    return env.astype(np.float32), sr / hop


def _tempo_ratio(remix_mix, orig_mix, sr=SR):
    """remix と原曲のテンポ比 (remix/orig) を推定。librosa無し/失敗時は1.0。"""
    try:
        import librosa
        br, _ = librosa.beat.beat_track(y=remix_mix, sr=sr)
        bo, _ = librosa.beat.beat_track(y=orig_mix, sr=sr)
        br = float(br) if np.ndim(br) == 0 else float(br[0])
        bo = float(bo) if np.ndim(bo) == 0 else float(bo[0])
        if br > 0 and bo > 0:
            r = br / bo
            for cand in (r, r / 2.0, r * 2.0):
                if 0.5 <= cand <= 2.0:
                    return cand
    except Exception:
        pass
    return 1.0


def _beat_sync_chroma(y, sr):
    """ボーカル音声から拍同期クロマ特徴とビート時刻を作る。"""
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    if len(beats) < 4:
        # ビートが取れない → フレーム時刻をそのまま使う
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=512)
        return chroma, times
    bsync = librosa.util.sync(chroma, beats, aggregate=np.median)
    bt = librosa.frames_to_time(beats, sr=sr, hop_length=512)
    # syncはbeats区間ごとに集約するので列数がlen(beats)+1になることがある→時刻列を合わせる
    if bsync.shape[1] == len(bt) + 1:
        bt = np.concatenate([[0.0], bt])
    elif bsync.shape[1] != len(bt):
        m = min(bsync.shape[1], len(bt))
        bsync = bsync[:, :m]; bt = bt[:m]
    return bsync, bt


def find_vocal_lipsync_segments(remix_voc, orig_voc, sr=SR, ratio=1.0,
                                chunk_sec=6.0, conf_th=0.45, verbose=False):
    """
    ビート同期クロマ + DTW で「remix時刻 → 原曲MV時刻」の連続対応を作り、
    ボーカルがある区間にその対応箇所（口パクが合うMVシーン）を割り当てる。
    戻り値: [{"r_start","r_end","o_start"(or None),"conf"}, ...]
    ※ratioは使わない（DTWがテンポ差を吸収するため）。互換のため引数は残す。
    """
    import librosa

    # remix側の声量エンベロープ（歌ってる/無音の判定用）
    env_r, fps = _vocal_envelope(remix_voc, sr)
    if len(env_r) < 10:
        return [], (lambda t: t)
    env_max = float(env_r.max()) if env_r.size else 1.0
    voice_th = max(0.06, env_max * 0.12)   # これ未満は「声が小さい=ドロップ」

    # --- 拍同期クロマ + DTW ---
    try:
        cr, bt_r = _beat_sync_chroma(remix_voc, sr)
        co, bt_o = _beat_sync_chroma(orig_voc, sr)
        if cr.shape[1] < 4 or co.shape[1] < 4:
            raise RuntimeError("ビート不足")
        # 制約付きDTW（Sakoe-Chiba band で不自然な遠回りを抑制）
        D, wp = librosa.sequence.dtw(X=cr, Y=co, subseq=False, backtrack=True,
                                     global_constraints=True, band_rad=0.25)
        wp = wp[::-1]  # 時間順 (remix拍idx, orig拍idx)
        # remix時刻 → 原曲時刻 の写像テーブルを作る
        map_rt = []; map_ot = []
        for (ri, oi) in wp:
            if ri < len(bt_r) and oi < len(bt_o):
                map_rt.append(bt_r[ri]); map_ot.append(bt_o[oi])
        if len(map_rt) < 4:
            raise RuntimeError("DTW対応不足")
        map_rt = np.array(map_rt); map_ot = np.array(map_ot)
    except Exception as e:
        if verbose:
            print(f"     ⚠️ DTW失敗（{e}）→ リップシンクなし")
        return [], (lambda t: t)

    def remix_to_orig(t):
        """remix時刻tに対応する原曲MV時刻を線形補間で返す"""
        return float(np.interp(t, map_rt, map_ot))

    # DTW対応の局所信頼度: その拍付近でクロマがどれだけ一致してるか
    # （対応点ごとのクロマ内積を測り、チャンクの平均一致度をconfにする）
    def local_chroma_sim(rt):
        # rtに最も近いremix拍indexのクロマと、対応する原曲拍のクロマの類似度
        ri = int(np.argmin(np.abs(bt_r - rt)))
        ot = remix_to_orig(rt)
        oi = int(np.argmin(np.abs(bt_o - ot)))
        a = cr[:, ri]; b = co[:, oi]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    # 独立マッチ: remixの[r0,r1]の和音列を、原曲MV全体から最も合う開始位置を探して返す。
    #   DTW（前進のみ）が苦手な「リミックスの並べ替え」区間を当て直すのに使う。
    #   戻り値: (原曲開始秒 or None, 平均コサイン類似度)
    crn = cr / (np.linalg.norm(cr, axis=0, keepdims=True) + 1e-9)
    con = co / (np.linalg.norm(co, axis=0, keepdims=True) + 1e-9)
    def independent_anchor(r0, r1):
        # remixの[r0,r1]の和音列を、原曲MV全体から最も合う開始位置を探す。
        # 比較を公平にするため、DTW予測位置のスコアも“同じ窓平均”で同時に返す。
        # 戻り値: (最良の原曲開始秒, 最良スコア, DTW位置のスコア)
        ri0 = int(np.argmin(np.abs(bt_r - r0)))
        ri1 = int(np.argmin(np.abs(bt_r - r1)))
        L = ri1 - ri0
        if L < 3 or con.shape[1] < L + 1:
            return None, 0.0, 0.0
        win = crn[:, ri0:ri1]                      # (12, L) 正規化済み
        def score_at(oi):
            oi = max(0, min(oi, con.shape[1] - L))
            return float(np.mean(np.sum(win * con[:, oi:oi+L], axis=0)))
        # DTW予測位置（同じ窓尺度で）
        oi_dtw = int(np.argmin(np.abs(bt_o - remix_to_orig(r0))))
        dtw_winscore = score_at(oi_dtw)
        # 原曲MV全体から最良位置を探索
        best_oi, best_sim = oi_dtw, dtw_winscore
        for oi in range(0, con.shape[1] - L):
            sim = float(np.mean(np.sum(win * con[:, oi:oi+L], axis=0)))
            if sim > best_sim:
                best_sim, best_oi = sim, oi
        return float(bt_o[best_oi]), best_sim, dtw_winscore

    def earliest_good_anchor(r0, r1, tol=0.15):
        # 歌い出し安定化用: [r0,r1]に対し、最良スコアから tol 以内で合う
        # 「最も早い原曲位置」を返す。曲の歌い出し=MVの歌い出し、という前提に使う。
        # 戻り値: (最も早い良マッチ秒 or None, そのスコア, DTW位置スコア)
        ri0 = int(np.argmin(np.abs(bt_r - r0)))
        ri1 = int(np.argmin(np.abs(bt_r - r1)))
        L = ri1 - ri0
        if L < 3 or con.shape[1] < L + 1:
            return None, 0.0, 0.0
        win = crn[:, ri0:ri1]
        scores = np.empty(con.shape[1] - L, dtype=np.float32)
        for oi in range(con.shape[1] - L):
            scores[oi] = float(np.mean(np.sum(win * con[:, oi:oi+L], axis=0)))
        oi_dtw = max(0, min(int(np.argmin(np.abs(bt_o - remix_to_orig(r0)))), len(scores) - 1))
        dtw_score = float(scores[oi_dtw])
        best = float(scores.max())
        good = np.where(scores >= best - tol)[0]   # 最良から tol 以内
        early_oi = int(good[0]) if len(good) else int(np.argmax(scores))
        return float(bt_o[early_oi]), float(scores[early_oi]), dtw_score

    # --- 6秒チャンクごとに判定 ---
    segs = []
    total = len(env_r) / fps
    t = 0.0
    while t < total - 1.5:
        rs = t
        re_ = min(t + chunk_sec, total)
        # チャンク内の平均声量
        i0 = int(rs * fps); i1 = int(re_ * fps)
        seg = env_r[i0:i1]
        vmean = float(seg.mean()) if seg.size else 0.0

        if vmean < voice_th:
            segs.append({"r_start": rs, "r_end": re_, "o_start": None, "conf": 0.0})
            t = re_; continue

        # チャンク中央のクロマ一致度を信頼度に
        mid = (rs + re_) / 2
        sim = local_chroma_sim(mid)
        o_start = remix_to_orig(rs)

        if sim >= conf_th:
            segs.append({"r_start": rs, "r_end": re_, "o_start": float(o_start),
                         "o_end": float(remix_to_orig(re_)), "conf": sim})
        else:
            segs.append({"r_start": rs, "r_end": re_, "o_start": None,
                         "o_end": None, "conf": sim})
        t = re_

    # --- 連続するリップシンク区間を結合（長回し優先）---
    merged = []
    for s in segs:
        if merged:
            p = merged[-1]
            if p["o_start"] is not None and s["o_start"] is not None:
                # DTWは単調なので、対応が連続していれば結合
                expected = p["o_start"] + (p["o_end"] - p["o_start"])
                if abs(s["o_start"] - p["o_end"]) < 2.0:
                    p["r_end"] = s["r_end"]
                    p["o_end"] = s["o_end"]
                    continue
            if p["o_start"] is None and s["o_start"] is None:
                p["r_end"] = s["r_end"]
                continue
        merged.append(dict(s))

    # --- 後半のみ再アンカー（前半は絶対に触らない＝doc12の前半を維持） ---
    # 前半（曲の前半50%）は素のDTWのまま。後半だけ、別位置が“同じ窓尺度”で
    # 明確に上回る(+MARGIN)時に当て直す（並べ替え・DTW誤マッチで一致が低い後半を救済）。
    REANCHOR_MARGIN = 0.15
    front_guard = total * 0.5
    for s in merged:
        s["anchor_o"] = None
        if s["o_start"] is None:
            continue
        if s["r_start"] < front_guard:        # 前半は対象外（最優先で前半保護）
            continue
        o_indep, indep_conf, dtw_conf = independent_anchor(s["r_start"], s["r_end"])
        dtw_o = remix_to_orig(s["r_start"])
        if (o_indep is not None
                and indep_conf >= dtw_conf + REANCHOR_MARGIN   # 同尺度で明確に上回る
                and abs(o_indep - dtw_o) > 3.0):                # 別の場所のときだけ
            s["anchor_o"] = o_indep
            s["o_start"] = o_indep
            s["conf"] = indep_conf
            s["reanchored"] = True

    if verbose:
        for s in merged:
            if s["o_start"] is None:
                kind = "⚡ ドロップ/VJ"
            elif s.get("reanchored"):
                kind = f"🎯 MV {s['o_start']:.1f}s〜 (再マッチ 一致 {s['conf']:.2f})"
            else:
                kind = f"🎤 MV {s['o_start']:.1f}s〜 (一致 {s['conf']:.2f})"
            print(f"     {s['r_start']:6.1f}〜{s['r_end']:6.1f}秒 : {kind}")
    # remix_to_orig 写像も返す（歌詞位置同期の配置で使う）
    return merged, remix_to_orig


# ------------------------------------------------------------
# 本体: リップシンクMVを組み立てる
# ------------------------------------------------------------
def _motion_pool(mv_path, mv_dur, n=14, verbose=False):
    """原曲MVから“動きが大きい=映える”時刻を抽出してVJ素材プールを返す。
    cv2が無い/失敗時は中盤を均等割り。戻り値: 開始秒のリスト（時刻順）。"""
    ts = []
    try:
        import cv2
        cap = cv2.VideoCapture(str(mv_path))
        step = 0.6
        prev = None; scores = []
        t = 0.3
        limit = min(mv_dur, 600)
        while t < limit:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2GRAY).astype("float32")
            if prev is not None:
                scores.append((t, float(np.mean(np.abs(g - prev)))))
            prev = g; t += step
        cap.release()
        if scores:
            # 動きの大きい順に、最低3秒間隔を空けて選ぶ → 時刻順に整列
            scores.sort(key=lambda x: x[1], reverse=True)
            picked = []
            for tt, sc in scores:
                if all(abs(tt - p) >= 3.0 for p in picked):
                    picked.append(tt)
                if len(picked) >= n:
                    break
            ts = sorted(picked)
    except Exception as e:
        if verbose:
            print(f"     ⚠️ 動き解析に失敗（{e}）→ 中盤を均等割りで使用")
        ts = []
    if len(ts) < 4:
        a, b = mv_dur * 0.10, mv_dur * 0.85
        if b - a < 2:
            return [0.0]
        ts = [a + (b - a) * i / (n - 1) for i in range(n)]
    return ts

def _build_vj_drop(mv_path, mv_dur, dur, out, tmp_dir, pool, state, cut=1.8):
    """ドロップ/VJ区間を、MVの“映える”箇所(pool)から短く切り替えて構成する。
    poolを順に回して使うので、同じ映像の繰り返しにならない。成功でTrue。"""
    tmp_dir = Path(tmp_dir)
    pieces = []
    remain = dur
    idx = 0
    while remain > 0.05:
        d = min(cut, remain)
        if d < 0.15:   # ごく僅かな端数のみ切り捨て（尺は最終調整で補完）
            break
        src_t = pool[state[0] % len(pool)] if pool else 0.0
        state[0] += 1
        # MV末尾を超えないように調整
        if src_t + d > mv_dur - 0.05:
            src_t = max(0.0, mv_dur - d - 0.05)
        pe = tmp_dir / f"{out.stem}_p{idx:02d}.mp4"
        idx += 1
        _run(["ffmpeg", "-y", "-ss", f"{src_t:.3f}", "-t", f"{d:.3f}",
              "-i", str(mv_path), "-vf", VF_NORM, *ENC_ARGS, "-an", str(pe)])
        if pe.exists() and pe.stat().st_size > 0:
            pieces.append(pe)
        remain -= d
    if not pieces:
        return False
    if len(pieces) == 1:
        import shutil
        shutil.copy(pieces[0], out)
        return True
    lst = tmp_dir / f"{out.stem}_list.txt"
    with open(lst, "w") as f:
        for p in pieces:
            f.write(f"file '{Path(p).resolve()}'\n")
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
          "-c", "copy", str(out)])
    return out.exists() and out.stat().st_size > 0


def make_vocal_lipsync_remix(music_path, mv_path, output_path, tmp_dir, music_dur,
                             filler_cb=None, verbose=True):
    """
    Demucsでボーカルを分離し、原曲MVを remix にリップシンク同期した mp4 を作る。
    成功で True。素材不足/一致不足なら False（→従来法へフォールバック）。
    Demucsが使えない場合はHPSS近似で続行する（完全には止めない）。
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    music_path = Path(music_path)
    mv_path = Path(mv_path)
    output_path = Path(output_path)

    mv_dur = _get_duration(mv_path)
    if mv_dur < 3.0:
        if verbose:
            print("  ⚠️ MVが短すぎ → リップシンク不可")
        return False

    demucs_ready = ensure_demucs(verbose)
    if not demucs_ready and verbose:
        print("  ℹ️ Demucs無しのためHPSS近似で続行します（精度は落ちます）")

    if verbose:
        print("\n  🎤 ボーカルstem分離中（remix / 原曲MV）...")
    remix_voc, m1 = separate_vocals(music_path, tmp_dir / "sep_remix",
                                    max_sec=min(music_dur, 420), verbose=verbose,
                                    demucs_ready=demucs_ready)
    orig_voc, m2 = separate_vocals(mv_path, tmp_dir / "sep_orig",
                                   max_sec=min(mv_dur, 420), verbose=verbose,
                                   demucs_ready=demucs_ready)
    if remix_voc.size < SR or orig_voc.size < SR:
        return False

    remix_mix = _load_mono(music_path, SR, duration=min(music_dur, 420))
    orig_mix = _load_mono(mv_path, SR, duration=min(mv_dur, 420))
    ratio = _tempo_ratio(remix_mix, orig_mix)
    if verbose:
        print(f"     テンポ比 remix/orig = x{ratio:.3f}（分離: remix={m1} / orig={m2}）")

    if verbose:
        print("  🔬 ボーカル包絡でリップシンク区間を解析中...")
    segs, remix_to_orig = find_vocal_lipsync_segments(remix_voc, orig_voc, ratio=ratio, verbose=verbose)
    if not segs:
        return False

    n_lip = sum(1 for s in segs if s["o_start"] is not None)
    lip_sec = sum(s["r_end"] - s["r_start"] for s in segs if s["o_start"] is not None)
    if verbose:
        print(f"     リップシンク採用 {n_lip}/{len(segs)} 区間（{lip_sec:.0f}秒 / 全{music_dur:.0f}秒）")

    if lip_sec < music_dur * 0.15:
        if verbose:
            print("     ⚠️ ボーカル一致が少なすぎ → 従来法へフォールバック")
        return False

    # --- 原曲MVの顔アップ時間帯を検出（歌唱シーン優先のため）---
    # FACE_PRIORITY_WINDOW=0 なら顔優先オフ（DTWの対応をそのまま信頼）
    face_times, face_scores = None, None
    if FACE_PRIORITY_WINDOW > 0:
        if verbose:
            print("  🙂 原曲MVの歌唱シーン（顔アップ）を検出中...")
        face_times, face_scores = detect_face_score_timeline(
            mv_path, sample_fps=1.0, max_sec=min(mv_dur, 420), verbose=verbose)
        if verbose and face_times is not None:
            n_face = int(np.sum(face_scores > 0.01))
            print(f"     顔検出: {n_face}/{len(face_times)}秒で顔を検出")
    else:
        if verbose:
            print("  🎯 DTW対応をそのまま使用（顔優先オフ）")

    if verbose:
        print(f"  🎬 原曲MVを等速で配置（See You Again式・歌詞位置に合わせて細切れ）")

    # --- セグメント生成（等速・See You Again式 / doc12方式） ---
    #   各サブカットを「原曲MVの対応位置から SUBSEG 秒、等速で」切り出す（setptsで伸縮しない）。
    SUBSEG = 4.0   # 細切れの長さ（短いほど局所テンポ変化に強いが、カットが増える）
    seg_files = []
    seg_idx = 0
    for s in segs:
        total_dur = s["r_end"] - s["r_start"]
        if total_dur < 0.3:
            continue

        if s["o_start"] is not None:
            # 歌区間: SUBSEG秒ごとに原曲MVの対応位置から“等速”で切り出す
            anchor = s.get("anchor_o")   # 後半の再アンカー区間のみ非None（前半=Noneで素のDTW）
            seg_r0 = s["r_start"]
            done = 0.0
            while done < total_dur - 0.05:
                sub_dur = min(SUBSEG, total_dur - done)
                r0 = seg_r0 + done
                if anchor is None:
                    o_pos = remix_to_orig(r0)          # DTW対応位置（前半はここ＝doc12と同じ）
                else:
                    o_pos = anchor + (r0 - seg_r0)      # 再アンカー位置から等速で進める（後半）
                o_pos = max(0.0, o_pos)
                if o_pos + sub_dur > mv_dur - 0.05:
                    o_pos = max(0.0, mv_dur - sub_dur - 0.05)
                out = tmp_dir / f"vlip_{seg_idx:03d}.mp4"
                seg_idx += 1
                _run(["ffmpeg", "-y", "-ss", f"{o_pos:.3f}", "-t", f"{sub_dur:.3f}",
                      "-i", str(mv_path), "-vf", VF_NORM, *ENC_ARGS, "-an", str(out)])
                if out.exists() and out.stat().st_size > 0:
                    seg_files.append(out)
                done += sub_dur
        else:
            # ドロップ/VJ区間（歌詞なし）→ filler_cb（原曲MVループ等）。doc12はVJカット無し。
            out = tmp_dir / f"vlip_{seg_idx:03d}.mp4"
            seg_idx += 1
            dur = total_dur
            made = False
            if filler_cb is not None:
                try:
                    filler_cb(dur, out)
                    made = out.exists() and out.stat().st_size > 0
                except Exception:
                    made = False
            # フォールバック: 原曲MVの適当なシーンを単発切り出し
            if not made:
                out = tmp_dir / f"vlip_{seg_idx:03d}.mp4"
                seg_idx += 1
                import random
                ss = random.uniform(0, max(0.0, mv_dur - dur - 0.5)) if mv_dur > dur + 1 else 0.0
                _run(["ffmpeg", "-y", "-ss", f"{ss:.3f}", "-i", str(mv_path),
                      "-t", f"{dur:.3f}", "-vf", VF_NORM, *ENC_ARGS, "-an", str(out)])
            if out and out.exists() and out.stat().st_size > 0:
                seg_files.append(out)

    if not seg_files:
        return False

    # --- 結合（concat demuxerは相対パスをlistの場所基準で解決するので絶対パスで書く） ---
    listf = tmp_dir / "vlip_concat.txt"
    with open(listf, "w") as f:
        for s in seg_files:
            f.write(f"file '{Path(s).resolve()}'\n")
    combined = tmp_dir / "vlip_combined.mp4"
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
          "-c", "copy", str(combined)])
    if not combined.exists() or combined.stat().st_size == 0:
        return False

    # --- 長さ合わせ（音声より短ければ延長） ---
    cdur = _get_duration(combined)
    if cdur < music_dur - 0.1:
        pad = music_dur - cdur + 0.5
        padf = tmp_dir / "vlip_pad.mp4"
        made = False
        if filler_cb is not None:
            try:
                filler_cb(pad, padf)
                made = padf.exists() and padf.stat().st_size > 0
            except Exception:
                made = False
        if not made:
            _run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(combined),
                  "-t", f"{pad:.3f}", "-vf", VF_NORM, *ENC_ARGS, "-an", str(padf)])
            made = padf.exists() and padf.stat().st_size > 0
        if made:
            l2 = tmp_dir / "vlip_ext.txt"
            with open(l2, "w") as f:
                f.write(f"file '{Path(combined).resolve()}'\n"
                        f"file '{Path(padf).resolve()}'\n")
            ext = tmp_dir / "vlip_combined_ext.mp4"
            _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(l2),
                  "-c", "copy", str(ext)])
            if ext.exists() and ext.stat().st_size > 0:
                combined = ext

    # --- 音楽と合成 ---
    _run(["ffmpeg", "-y", "-i", str(combined), "-i", str(music_path),
          "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
          "-b:a", "320k", "-t", f"{music_dur:.3f}", "-movflags", "+faststart",
          str(output_path)])
    if output_path.exists() and output_path.stat().st_size > 0:
        if verbose:
            mb = output_path.stat().st_size / 1024 / 1024
            tag = "（Demucs）" if m1 == "demucs" else "（HPSS近似）"
            print(f"  ✅ リップシンク完成{tag}: {output_path.name} ({mb:.1f} MB)")
        return True
    return False


# ------------------------------------------------------------
# 単体診断: `python3 vocal_sync.py` で Demucs が使えるか確認できる
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🔎 Demucs 実行可否を診断します...")
    ok, detail = demucs_runtime_status()
    if ok:
        print("✅ そのまま使えます:", detail)
    else:
        print("❌ 現状は使えません。導入を試みます...")
        if ensure_demucs(verbose=True):
            print("✅ 導入成功。次回からDemucsでリップシンクします。")
        else:
            print("⚠️ 導入できませんでした。上のエラーを開発者に共有してください。")
