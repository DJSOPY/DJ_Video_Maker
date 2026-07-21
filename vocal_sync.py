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
#                             music_dur, filler_cb=None,
#                             strict_fail_closed=False) -> bool
# ============================================================

import os, subprocess, shutil, sys, importlib, tempfile
from fractions import Fraction
from pathlib import Path
import numpy as np

try:
    _numba_cache = Path.home() / ".dj_video_maker" / "numba_cache"
    _numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba_cache))
except OSError:
    try:
        _numba_cache = Path(tempfile.gettempdir()) / f"djvm_numba_{os.getuid()}"
        _numba_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba_cache))
    except OSError:
        pass

SR = 22050

# core 側の規格と必ず一致させること（結合時のズレ防止）
VF_NORM = ("scale=1280:720:force_original_aspect_ratio=decrease,"
           "pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1")
ENC_ARGS = ["-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-video_track_timescale", "15360"]
OUTPUT_FPS = Fraction(30, 1)
# 旧方式は声の包絡/クロマ中心で、反復サビの別歌詞を
# phoneme内容で区別できない。ProのHuBERT局所証明を迂回しないよう、
# strict呼び出しでは旧方式の人物表示を無効化する。
ALLOW_LEGACY_VISIBLE_FACES_IN_STRICT_MODE = False

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


def _load_stereo_f32(path, sr):
    """ffmpegで stereo float32 (2, N) [-1,1] にデコード（Demucs API入力用）"""
    cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", str(path),
           "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.float32)
    n = (a.size // 2) * 2
    return a[:n].reshape(-1, 2).T.copy()          # (2, N)


def _resample(y, orig_sr, target_sr):
    """mono配列のサンプリングレート変換（librosa→scipyの順に試す）"""
    if orig_sr == target_sr or y.size == 0:
        return y
    try:
        import librosa
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    except Exception:
        pass
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(orig_sr), int(target_sr))
        return resample_poly(y, int(target_sr) // g, int(orig_sr) // g).astype(np.float32)
    except Exception:
        return y


def _get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, errors="replace")
    try:
        return float(r.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def _has_av_streams(path):
    """最終出力に映像と音声の両方があるか確認。"""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, errors="replace")
    kinds = set((r.stdout or "").split())
    return r.returncode == 0 and {"video", "audio"}.issubset(kinds)


def _valid_video(path, expected_dur=None, tolerance=1.0):
    """途中で打ち切られたmp4を成功と誤認しないための検査。"""
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    dur = _get_duration(path)
    if dur <= 0.05:
        return False
    return (expected_dur is None
            or abs(dur - float(expected_dur)) <= float(tolerance))


def _decoded_video_frame_count(path):
    """実際に最後までデコードできた映像フレーム数。取得不能はNone。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames",
             "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            return None
        values = (r.stdout or "").strip().splitlines()
        if not values or values[0] in ("", "N/A"):
            return None
        count = int(values[0])
        return count if count >= 0 else None
    except (OSError, TypeError, ValueError, IndexError):
        return None


def _video_has_exact_frames(path, expected_frames):
    try:
        expected_frames = int(expected_frames)
    except (TypeError, ValueError, OverflowError):
        return False
    p = Path(path)
    return bool(expected_frames > 0 and p.exists() and p.stat().st_size > 0
                and _decoded_video_frame_count(p) == expected_frames)


def _concat_video_exact(list_path, output_path, expected_frames):
    """結合結果が途中欠けなら再エンコードし、それでも違えば不採用。"""
    output_path = Path(output_path)
    output_path.unlink(missing_ok=True)
    r = _run(["ffmpeg", "-v", "error", "-y", "-fflags", "+genpts",
              "-f", "concat", "-safe", "0", "-i", str(list_path),
              "-c", "copy", str(output_path)])
    if r.returncode == 0 and _video_has_exact_frames(output_path, expected_frames):
        return True, r
    output_path.unlink(missing_ok=True)
    vf = _exact_frame_filter(VF_NORM, expected_frames)
    r = _run(["ffmpeg", "-v", "error", "-y", "-fflags", "+genpts",
              "-f", "concat", "-safe", "0", "-i", str(list_path),
              "-vf", vf, *ENC_ARGS, "-an", "-frames:v", str(int(expected_frames)),
              str(output_path)])
    return bool(r.returncode == 0
                and _video_has_exact_frames(output_path, expected_frames)), r


def _mux_video_audio_exact(video_path, audio_path, output_path,
                           expected_frames, music_dur):
    output_path = Path(output_path)
    output_path.unlink(missing_ok=True)
    r = _run(["ffmpeg", "-v", "error", "-y", "-i", str(video_path),
              "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0",
              "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
              "-frames:v", str(int(expected_frames)), "-t", f"{music_dur:.3f}",
              "-movflags", "+faststart", str(output_path)])
    ok = bool(r.returncode == 0
              and _video_has_exact_frames(output_path, expected_frames)
              and _has_av_streams(output_path))
    return ok, r


def _tail(text, n=15):
    lines = (text or "").strip().splitlines()
    return lines[-n:] if lines else []


def _to_fraction(value):
    """秒/FPSを、長尺でも浮動小数誤差が蓄積しない有理数へ変換する。"""
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def _round_frame_boundary(seconds, fps=OUTPUT_FPS):
    """時刻を最寄りのフレーム境界へ丸める（0.5は未来側）。"""
    seconds = _to_fraction(seconds)
    fps = _to_fraction(fps)
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    frames = seconds * fps
    return (2 * frames.numerator + frames.denominator) // (2 * frames.denominator)


class _CumulativeFrameClock:
    """各区間を独立丸めせず、累積時刻の境界差から必要フレーム数を返す。"""

    def __init__(self, fps=OUTPUT_FPS):
        self.fps = _to_fraction(fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        self.elapsed = Fraction(0, 1)
        self.frame_boundary = 0

    def take(self, duration):
        duration = _to_fraction(duration)
        if duration < 0:
            raise ValueError("duration must be non-negative")
        self.elapsed += duration
        next_boundary = _round_frame_boundary(self.elapsed, self.fps)
        count = next_boundary - self.frame_boundary
        self.frame_boundary = next_boundary
        return count


def _cumulative_frame_counts(durations, fps=OUTPUT_FPS):
    """テストや事前計画用。合計誤差を半フレーム以内に保つ区間別枚数。"""
    clock = _CumulativeFrameClock(fps)
    return [clock.take(duration) for duration in durations]


def _exact_frame_filter(base_filter, frame_count):
    """不足時は末尾を補い、指定枚数で必ず打ち切る映像フィルター。"""
    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    return (f"{base_filter},tpad=stop_mode=clone:stop_duration=1,"
            f"trim=end_frame={frame_count},setpts=PTS-STARTPTS")


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


def _separate_vocals_api(src, sr, dev, verbose=True):
    """
    Demucsを「ファイル保存を挟まず」Python API(apply_model)で実行し、
    ボーカルstemを mono np.array (sr) で返す。失敗時は (None, 理由)。

    なぜCLIでなくAPIか:
      demucs CLI は分離後の stem を torchaudio.save() で書き出す。
      torchaudio 2.9以降 save() は torchcodec に処理を委譲するため、
      torchcodec が無い環境では ImportError で必ず失敗する
      （＝分離自体は成功しているのに保存だけで落ちてHPSSに落ちる）。
      APIならテンソルを直接受け取れるので、この保存経路を完全に回避できる。
    """
    # demucs.api は 4.1 以降にしか無い。4.0系にも必ずある低レベルAPIを直接使う。
    try:
        import torch as th
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except Exception as e:
        return None, f"demucsのimportに失敗: {e!r}"

    try:
        model = get_model("htdemucs")
        model.eval()
        model_sr = int(getattr(model, "samplerate", 44100))       # htdemucs は 44100
        ch = int(getattr(model, "audio_channels", 2))

        wav = _load_stereo_f32(src, model_sr)                     # (2, N)
        if wav.size == 0:
            return None, "音声のデコードに失敗（0サンプル）"
        x = th.from_numpy(np.ascontiguousarray(wav))
        if ch == 1:
            x = x.mean(dim=0, keepdim=True)

        # demucsの標準前処理（api/CLIと同じ正規化）
        ref = x.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with th.no_grad():
            out = apply_model(model, ((x - mean) / std)[None],
                              shifts=0, split=True, overlap=0.25,
                              device=dev, progress=False)
        out = out * std + mean                                    # (1, stems, ch, N)

        sources = list(getattr(model, "sources", []))
        if "vocals" not in sources:
            return None, f"vocals stem がありません（sources={sources}）"
        voc = out[0, sources.index("vocals")]                     # (ch, N)

        arr = voc.detach().to("cpu").float().mean(dim=0).numpy()  # → mono
        arr = _resample(arr, model_sr, sr).astype(np.float32)
        if arr.size == 0 or float(np.max(np.abs(arr))) <= 1e-4:
            return None, "分離結果が無音でした"
        return arr, "ok"
    except Exception as e:
        return None, repr(e)


def _ensure_torchcodec(verbose=True):
    """
    torchaudio 2.9+ の save() が要求する torchcodec を導入する。
    demucs CLI 経路の ImportError 自己修復用。戻り値: True=導入済み/成功
    """
    try:
        importlib.import_module("torchcodec")
        return True
    except Exception:
        pass
    if verbose:
        print("     📦 torchaudio が torchcodec を要求しています → 自動導入します")
    for cmd in ([sys.executable, "-m", "pip", "install", "-q",
                 "--break-system-packages", "torchcodec"],
                [sys.executable, "-m", "pip", "install", "-q", "torchcodec"]):
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if r.returncode == 0:
            break
    try:
        importlib.invalidate_caches()
        importlib.import_module("torchcodec")
        return True
    except Exception:
        if verbose:
            print("     ⚠️ torchcodec を導入できませんでした")
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

        def _cache_and_return(arr, dev, how):
            # 次回から同じ結果になるよう、解析配列そのものをキャッシュ保存
            if cache_npy:
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    np.save(cache_npy, arr.astype(np.float32))
                except Exception:
                    pass
            if verbose:
                tag = f"（{dev.upper()}）" if dev != "cpu" else ""
                print(f"     🎤 Demucsでボーカル分離 成功{tag}{how}")
            return arr.astype(np.float32), "demucs"

        # --- 経路1: Python API（推奨）------------------------------------
        # stemをテンソルで直接受け取るので torchaudio.save / torchcodec に一切依存しない。
        reasons = []
        for dev in devs:
            arr, why = _separate_vocals_api(src, sr, dev, verbose=verbose)
            if arr is not None:
                return _cache_and_return(arr, dev, "")
            reasons.append(f"{dev}: {why}")
            if verbose and dev != devs[-1]:
                print(f"     ↪︎ Demucs APIの{dev.upper()}経路が使えず → APIのCPU経路を試行")

        # --- 経路2: CLI（APIが使えない古いdemucs等の保険）-------------------
        # --two-stems=vocals → vocals.wav と no_vocals.wav の2本だけ（高速）/ --shifts 0 → 決定性
        r = None
        tried_torchcodec = False
        for dev in devs:
            for _attempt in (1, 2):
                r = _run([sys.executable, "-m", "demucs", "--two-stems=vocals", "--shifts", "0",
                          "-d", dev, "-n", "htdemucs", "--out", str(outdir), str(src)])
                voc = list(outdir.glob("**/vocals.wav"))
                if voc:
                    arr = _load_mono(voc[0], sr)
                    if arr.size and float(np.max(np.abs(arr))) > 1e-4:
                        return _cache_and_return(arr, dev, "（CLI）")
                # 保存段の torchcodec 不足なら一度だけ自己修復して再試行
                out_all = ((r.stderr or "") + (r.stdout or "")) if r else ""
                if (not tried_torchcodec) and "torchcodec" in out_all.lower():
                    tried_torchcodec = True
                    if _ensure_torchcodec(verbose):
                        continue
                break

        # 失敗 → 本当のエラーを表示してから HPSS へ
        if verbose:
            print("     ⚠️ Demucs分離に失敗 → HPSSにフォールバック。原因:")
            for ln in reasons:
                print("        ", ln)
            for ln in _tail(((r.stderr if r else "") or "") + "\n" + ((r.stdout if r else "") or ""), 8):
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


def _vocal_active_mask_from_envelope(env, fps, max_active_island=0.20):
    """silence分割と全frame視覚証明で共有するclean-vocal mask。"""
    try:
        env = np.asarray(env, dtype=float).reshape(-1)
        fps = float(fps)
        max_active_island = max(0.0, float(max_active_island))
    except (TypeError, ValueError, OverflowError):
        return None
    if len(env) == 0 or not np.isfinite(fps) or fps <= 0.0:
        return None
    finite = np.where(np.isfinite(env), env, 0.0)
    peak = float(np.max(finite)) if len(finite) else 0.0
    if peak <= 1e-9:
        return np.zeros(len(finite), dtype=bool)
    threshold = max(0.06, peak * 0.12)
    active = finite >= threshold
    # 無声に挟まれた短いactive島はstem分離ノイズとして閉じる。
    max_island_frames = int(np.ceil(max_active_island * fps))
    if max_island_frames > 0:
        i = 0
        while i < len(active):
            if not active[i]:
                i += 1; continue
            j = i + 1
            while j < len(active) and active[j]:
                j += 1
            if i > 0 and j < len(active) and j - i <= max_island_frames:
                active[i:j] = False
            i = j
    return active


def _vocal_silence_ranges_from_envelope(env, fps, duration=None,
                                         min_silence=0.02,
                                         max_active_island=0.20,
                                         guard=0.12):
    """clean vocal stemの全無声区間を、安全側に抽出する。

    分離ノイズによる短いactive島（既定0.20秒以下）は無声へ戻す。
    1解析frameの無声から隠し、短い歌抜きでも原曲MVの口を表示しない。
    境界はguard秒広げて子音端の漏れも防ぐ。
    """
    try:
        env = np.asarray(env, dtype=float).reshape(-1)
        fps = float(fps); min_silence = max(0.0, float(min_silence))
        max_active_island = max(0.0, float(max_active_island))
        guard = max(0.0, float(guard))
        duration = (len(env) / fps if duration is None else float(duration))
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return []
    if len(env) == 0 or not np.isfinite(fps) or fps <= 0.0:
        return []
    active = _vocal_active_mask_from_envelope(
        env, fps, max_active_island=max_active_island)
    if active is None:
        return []
    if not np.any(active):
        return [(0.0, max(0.0, duration))] if duration > 0.0 else []

    ranges = []
    min_frames = max(1, int(np.ceil(min_silence * fps)))
    i = 0
    while i < len(active):
        if active[i]:
            i += 1; continue
        j = i + 1
        while j < len(active) and not active[j]:
            j += 1
        if j - i >= min_frames:
            ranges.append((max(0.0, i / fps - guard),
                           min(max(0.0, duration), j / fps + guard)))
        i = j

    merged = []
    for start, end in ranges:
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _build_all_frame_visual_profile(mv_path):
    """検出失敗をNoneで返す、旧方式用の全source frame profile。"""
    try:
        import mouth_sync
        profile = mouth_sync.build_mouth_profile(
            str(mv_path), fps=1000.0, use_cache=False)
        if not isinstance(profile, dict):
            return None
        profile = dict(profile)
        profile["_all_source_frames"] = True
        return profile
    except Exception:
        return None


def _strict_vocal_onset_envelope(voc, sr=SR):
    try:
        import librosa
        y = np.asarray(voc, dtype=np.float32).reshape(-1)
        if len(y) < int(sr):
            return None, None
        hop = 256
        env = librosa.onset.onset_strength(y=y, sr=int(sr), hop_length=hop)
        if len(env) < 4 or not np.all(np.isfinite(env)):
            return None, None
        peak = float(np.max(env))
        if peak <= 1e-9:
            return None, None
        return (np.arange(len(env), dtype=float) * hop / float(sr),
                np.asarray(env / peak, dtype=np.float32))
    except Exception:
        return None, None


def _strict_onset_peak_count(times, env, lo, hi):
    try:
        times = np.asarray(times, dtype=float); env = np.asarray(env, dtype=float)
        values = env[(times >= float(lo)) & (times < float(hi))]
        if len(values) < 5 or not np.all(np.isfinite(values)):
            return 0
        threshold = max(0.08, float(np.percentile(values, 70)) * 0.65)
        return int(np.sum((values[1:-1] >= values[:-2])
                          & (values[1:-1] > values[2:])
                          & (values[1:-1] >= threshold)))
    except Exception:
        return 0


def _mapped_segment_has_visual_proof(profile, remix_start, duration, nframes,
                                     source_start, source_duration,
                                     vocal_active, vocal_fps,
                                     output_start_frame=None,
                                     onset_times=None, onset_envelope=None):
    """旧方式の実際の各出力frameを、Proと同じ厳格基準で証明。"""
    try:
        nframes = int(nframes); duration = float(duration)
        source_start = float(source_start); source_duration = float(source_duration)
        vocal_fps = float(vocal_fps)
        raw_active = np.asarray(vocal_active).reshape(-1)
        if (np.issubdtype(raw_active.dtype, np.number)
                and not np.all(np.isfinite(raw_active.astype(float)))):
            return False
        active = raw_active.astype(bool)
        if (profile is None or nframes <= 0 or duration <= 0.0
                or source_duration <= 0.0 or len(active) == 0
                or not np.isfinite(vocal_fps) or vocal_fps <= 0.0):
            return False
        local = np.arange(nframes, dtype=float) / float(OUTPUT_FPS)
        local = np.clip(local, 0.0, max(0.0, duration - 1e-9))
        if output_start_frame is None:
            output_start_frame = int(round(float(remix_start) * float(OUTPUT_FPS)))
        remix_times = ((int(output_start_frame) + np.arange(nframes, dtype=float))
                       / float(OUTPUT_FPS))
        source_times = source_start + local * (source_duration / duration)
        frame_span = 1.0 / float(OUTPUT_FPS)
        frame_active = np.zeros(nframes, dtype=bool)
        for j, t in enumerate(remix_times):
            i0 = int(np.floor((float(t) + 1e-12) * vocal_fps))
            i1 = int(np.ceil((float(t) + frame_span - 1e-12) * vocal_fps))
            if i0 < 0 or i1 <= i0 or i1 > len(active):
                return False
            frame_active[j] = bool(np.all(active[i0:i1]))
        import mouth_sync
        verifier = getattr(
            mouth_sync, "mapped_frames_have_verified_lipsync_visual", None)
        if (not callable(verifier)
                or not verifier(profile, source_times, frame_active)):
            return False
        if (_strict_onset_peak_count(
                onset_times, onset_envelope, remix_times[0],
                remix_times[-1] + frame_span) < 3):
            return False
        phase_measure = getattr(mouth_sync, "measure_micro_mouth_lag", None)
        if not callable(phase_measure):
            return False
        lag, corr, fcov, prom = phase_measure(
            profile, remix_times, source_times,
            onset_times, onset_envelope, max_lag=0.60, step=0.02,
            min_face=0.45)
        return bool(
            lag is not None and np.isfinite(lag) and abs(float(lag)) <= 0.12
            and float(corr) >= 0.32 and float(fcov) >= 0.45
            and float(prom) >= 2.0)
    except Exception:
        return False


def clean_vocal_silence_ranges(audio_path, work_dir, duration,
                               min_silence=0.02, verbose=False):
    """Demucs clean vocalで無声区間を返す。証明不能ならNone。"""
    try:
        ready = ensure_demucs(verbose=verbose)
        if not ready:
            return None
        vocals, method = separate_vocals(
            audio_path, Path(work_dir), sr=SR, max_sec=duration,
            verbose=verbose, demucs_ready=True)
        if method != "demucs" or vocals is None or len(vocals) < SR:
            return None
        env, fps = _vocal_envelope(vocals, SR, fps=50)
        return _vocal_silence_ranges_from_envelope(
            env, fps, duration=duration, min_silence=min_silence)
    except Exception:
        return None


def _split_segments_on_vocal_silence(segs, remix_to_orig, silence_ranges):
    """歌唱対応segmentを持続無声境界で分割し、無声片を安全側へ送る。"""
    out = []
    ranges = list(silence_ranges or [])
    for original in segs or []:
        try:
            s0 = float(original["r_start"]); e0 = float(original["r_end"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if e0 <= s0:
            continue
        cuts = {s0, e0}
        for lo, hi in ranges:
            if lo < e0 and hi > s0:
                cuts.update((max(s0, float(lo)), min(e0, float(hi))))
        cuts = sorted(cuts)
        for a, b in zip(cuts, cuts[1:]):
            if b <= a + 1e-9:
                continue
            mid = (a + b) * 0.5
            silent = any(float(lo) <= mid < float(hi) for lo, hi in ranges)
            piece = dict(original)
            piece["r_start"] = a; piece["r_end"] = b
            if silent:
                piece.update(o_start=None, o_end=None, anchor_o=None,
                             voiced_sec=0.0, conf=0.0)
            elif original.get("o_start") is not None:
                anchor = original.get("anchor_o")
                if anchor is not None:
                    new_anchor = float(anchor) + (a - s0)
                    piece["anchor_o"] = new_anchor
                    piece["o_start"] = new_anchor
                    piece["o_end"] = new_anchor + (b - a)
                else:
                    piece["o_start"] = float(remix_to_orig(a))
                    piece["o_end"] = float(remix_to_orig(b))
                piece["voiced_sec"] = b - a
            else:
                # 声はあるが一致が弱い片。品質分母には残し、映像は安全側。
                piece["voiced_sec"] = b - a
            out.append(piece)
    # 境界に残った短い人物SHOW島は、理想動画と同じく安全側へ吸収する。
    for piece in out:
        if (piece.get("o_start") is not None
                and float(piece["r_end"] - piece["r_start"]) < 0.75):
            piece.update(o_start=None, o_end=None, anchor_o=None)
    return out


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
            segs.append({"r_start": rs, "r_end": re_, "o_start": None, "conf": 0.0,
                         "voiced_sec": 0.0})
            t = re_; continue

        # チャンク中央のクロマ一致度を信頼度に
        mid = (rs + re_) / 2
        sim = local_chroma_sim(mid)
        o_start = remix_to_orig(rs)

        if sim >= conf_th:
            segs.append({"r_start": rs, "r_end": re_, "o_start": float(o_start),
                         "o_end": float(remix_to_orig(re_)), "conf": sim,
                         "voiced_sec": re_ - rs})
        else:
            segs.append({"r_start": rs, "r_end": re_, "o_start": None,
                         "o_end": None, "conf": sim, "voiced_sec": re_ - rs})
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
                    p["voiced_sec"] = p.get("voiced_sec", 0.0) + s.get("voiced_sec", 0.0)
                    p["conf"] = min(float(p.get("conf", 0.0)), float(s.get("conf", 0.0)))
                    continue
            if p["o_start"] is None and s["o_start"] is None:
                p["r_end"] = s["r_end"]
                p["voiced_sec"] = p.get("voiced_sec", 0.0) + s.get("voiced_sec", 0.0)
                p["conf"] = max(float(p.get("conf", 0.0)), float(s.get("conf", 0.0)))
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


def _legacy_alignment_quality(segs, remix_method, orig_method,
                              strict_fail_closed=False):
    """旧エンジンの採用品質を「全曲尺」ではなく歌唱区間基準で評価。"""
    n_lip = sum(1 for s in segs if s.get("o_start") is not None)
    lip_sec = sum(float(s["r_end"] - s["r_start"])
                  for s in segs if s.get("o_start") is not None)
    voiced_sec = sum(float(s.get("voiced_sec", 0.0)) for s in segs)
    voiced_coverage = lip_sec / max(voiced_sec, 1e-6)
    matched_conf = [float(s.get("conf", 0.0)) for s in segs if s.get("o_start") is not None]
    median_conf = float(np.median(matched_conf)) if matched_conf else 0.0
    weakest_conf = float(np.min(matched_conf)) if matched_conf else 0.0
    clean_stems = (remix_method == "demucs" and orig_method == "demucs")
    min_coverage = 0.55 if clean_stems else 0.70
    min_conf = 0.48 if clean_stems else 0.56
    max_end = max((float(s.get("r_end", 0.0)) for s in segs), default=0.0)
    tail_start = max_end * 0.60
    tail_voiced = 0.0; tail_matched = 0.0
    for s in segs:
        rs = float(s.get("r_start", 0.0)); re = float(s.get("r_end", rs))
        dur = max(0.0, re - rs)
        overlap = max(0.0, re - max(rs, tail_start))
        if dur <= 0.0 or overlap <= 0.0:
            continue
        voiced_fraction = min(1.0, max(0.0, float(s.get("voiced_sec", 0.0)) / dur))
        v = overlap * voiced_fraction
        tail_voiced += v
        if s.get("o_start") is not None:
            tail_matched += v
    tail_coverage = tail_matched / max(tail_voiced, 1e-6)
    tail_ok = (tail_voiced < 1.0 or
               (tail_matched >= 2.0 and tail_coverage >= 0.55))
    accepted = bool(voiced_sec > 0.0 and lip_sec >= 4.0
                    and voiced_coverage >= min_coverage and median_conf >= min_conf
                    and tail_ok)
    if strict_fail_closed:
        # HPSS/raw近似や前半だけの成功を、Remix全編の成功と見なさない。
        accepted = bool(accepted and clean_stems
                        and voiced_coverage >= 0.68 and median_conf >= 0.55
                        and weakest_conf >= 0.62
                        and (tail_voiced < 1.0 or tail_coverage >= 0.60))
    return {"accepted": accepted, "n_lip": n_lip, "lip_sec": lip_sec,
            "voiced_sec": voiced_sec, "voiced_coverage": voiced_coverage,
            "median_conf": median_conf, "clean_stems": clean_stems,
            "weakest_conf": weakest_conf,
            "tail_voiced_sec": tail_voiced, "tail_matched_sec": tail_matched,
            "tail_coverage": tail_coverage}


def _call_filler_exact(filler_cb, duration, out, frame_count):
    """新3引数callbackを優先し、旧2引数callbackとも互換を保つ。"""
    if filler_cb is None:
        return False
    try:
        import inspect
        sig = inspect.signature(filler_cb)
        params = list(sig.parameters.values())
        accepts_three = (any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
                         or len([p for p in params
                                 if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                               inspect.Parameter.POSITIONAL_OR_KEYWORD)]) >= 3)
        if accepts_three:
            filler_cb(duration, out, frame_count)
        else:
            filler_cb(duration, out)
    except Exception:
        return False
    return bool(Path(out).exists() and Path(out).stat().st_size > 0)


def _make_black_no_mouth(out, frame_count):
    """callback失敗時も人物MVへ戻らず、正確な枚数の純黒にする。"""
    frames = max(1, int(frame_count))
    out = Path(out)
    out.unlink(missing_ok=True)
    r = _run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
              "color=c=black:s=1280x720:r=30", "-frames:v", str(frames),
              *ENC_ARGS, "-an", str(out)])
    return bool(r.returncode == 0 and _video_has_exact_frames(out, frames))


def make_vocal_lipsync_remix(music_path, mv_path, output_path, tmp_dir, music_dur,
                             filler_cb=None, verbose=True,
                             strict_fail_closed=False):
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
    if not np.isfinite(mv_dur) or mv_dur < 3.0:
        if verbose:
            print("  ⚠️ MVが短すぎ → リップシンク不可")
        return False

    demucs_ready = ensure_demucs(verbose)
    if strict_fail_closed and not demucs_ready:
        if verbose:
            print("  ⚠️ 厳格モードではHPSS近似を採用しません → 安全背景へ切替")
        return False
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
    # 厳格モードでは0.45程度の和音一致を人物表示へ通さない。
    # 個々の6秒区間が0.62以上のときだけ歌唱MVとして採用し、弱い区間は
    # o_start=Noneのまま安全フィラーへ送る。
    segs, remix_to_orig = find_vocal_lipsync_segments(
        remix_voc, orig_voc, ratio=ratio,
        conf_th=(0.62 if strict_fail_closed else 0.45), verbose=verbose)
    if not segs:
        return False

    strict_vocal_active = None
    strict_vocal_fps = None
    if strict_fail_closed:
        env, env_fps = _vocal_envelope(remix_voc, SR, fps=50)
        strict_vocal_active = _vocal_active_mask_from_envelope(
            env, env_fps, max_active_island=0.20)
        strict_vocal_fps = env_fps
        silence_ranges = _vocal_silence_ranges_from_envelope(
            env, env_fps, duration=music_dur, min_silence=0.02)
        if silence_ranges:
            segs = _split_segments_on_vocal_silence(
                segs, remix_to_orig, silence_ranges)
            if verbose:
                hidden = sum(max(0.0, b - a) for a, b in silence_ranges)
                print(f"     🛡️ 持続無声{len(silence_ranges)}区間・{hidden:.1f}秒を"
                      "人物映像から分離")

    q = _legacy_alignment_quality(
        segs, m1, m2, strict_fail_closed=strict_fail_closed)
    if verbose:
        print(f"     リップシンク採用 {q['n_lip']}/{len(segs)} 区間（{q['lip_sec']:.0f}秒 / "
              f"歌唱{q['voiced_sec']:.0f}秒 = {q['voiced_coverage']*100:.0f}% / "
              f"一致中央{q['median_conf']:.2f}・最低{q['weakest_conf']:.2f} / "
              f"後半{q['tail_coverage']*100:.0f}%）")

    if not q["accepted"]:
        if verbose:
            method_note = "Demucs" if q["clean_stems"] else "HPSS/raw（厳格判定）"
            print(f"     ⚠️ ボーカル一致品質が不足（{method_note}） → この方式は採用しない")
        return False

    # Pro不採用後の旧方式が、視覚検出失敗を人物表示の
    # 許可としないよう同じ全frame profileを必須化する。Noneなら
    # 以下の全matched片が安全背景になる。
    strict_visual_profile = None
    strict_onset_times = None
    strict_onset_envelope = None
    if strict_fail_closed:
        if verbose:
            print("  🛡️ 旧方式も口元を全source frameで安全確認中...")
        strict_visual_profile = _build_all_frame_visual_profile(mv_path)
        strict_onset_times, strict_onset_envelope = (
            _strict_vocal_onset_envelope(remix_voc, SR))
        if strict_visual_profile is None and verbose:
            print("     ⚠️ 視覚証明不能 → 人物区間は全て安全背景")
        if strict_onset_times is None and verbose:
            print("     ⚠️ 発音onset証明不能 → 人物区間は全て安全背景")

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
        print(f"  🎬 原曲MVを歌詞位置に配置（局所テンポ差も追従）")

    # --- セグメント生成（See You Again式 / doc12改良） ---
    #   対応位置は維持し、連続区間だけDTWの局所進行量に追従する。
    SUBSEG = 2.0   # 局所テンポ差の蓄積を最大約2秒窓に抑える
    seg_files = []
    seg_idx = 0
    seg_fail = 0
    filler_fail = 0
    visual_hidden = 0
    # 各2秒片を個別に丸めると、setptsの倍率次第で60枚/61枚が混在し、
    # 結合後の後半ほど映像が遅れる。全区間共通の時刻境界で枚数を割り当てる。
    frame_clock = _CumulativeFrameClock(OUTPUT_FPS)
    for s in segs:
        total_dur = s["r_end"] - s["r_start"]
        if total_dur <= 0.0:
            continue

        if s["o_start"] is not None:
            # 歌区間: SUBSEG秒ごとに原曲MVの対応位置から切り出す
            anchor = s.get("anchor_o")   # 後半の再アンカー区間のみ非None（前半=Noneで素のDTW）
            seg_r0 = s["r_start"]
            done = 0.0
            while done < total_dur - 1e-9:
                sub_dur = min(SUBSEG, total_dur - done)
                target_frames = frame_clock.take(sub_dur)
                if target_frames <= 0:
                    done += sub_dur
                    continue
                output_start_frame = frame_clock.frame_boundary - target_frames
                r0 = seg_r0 + done
                if anchor is None:
                    raw_o_pos = remix_to_orig(r0)      # DTW対応位置
                    raw_o_end = remix_to_orig(r0 + sub_dur)
                else:
                    raw_o_pos = anchor + (r0 - seg_r0) # 再アンカー位置から等速
                    raw_o_end = raw_o_pos + sub_dur
                try:
                    raw_o_pos = float(raw_o_pos); raw_o_end = float(raw_o_end)
                    mapping_finite = bool(
                        np.isfinite(raw_o_pos) and np.isfinite(raw_o_end))
                except (TypeError, ValueError, OverflowError):
                    raw_o_pos = raw_o_end = 0.0
                    mapping_finite = False
                o_pos = max(0.0, float(raw_o_pos)) if mapping_finite else 0.0
                o_end = float(raw_o_end) if mapping_finite else 0.0
                local_src = float(o_end - o_pos)
                ratio_local = local_src / max(0.05, sub_dur)
                use_rate = 0.8 <= ratio_local <= 1.25
                src_dur = local_src if use_rate else sub_dur
                if o_pos + src_dur > mv_dur - 0.05:
                    o_pos = max(0.0, mv_dur - src_dur - 0.05)
                src_dur = min(src_dur, max(0.05, mv_dur - o_pos - 0.02))
                out = tmp_dir / f"vlip_{seg_idx:03d}.mp4"
                seg_idx += 1
                out.unlink(missing_ok=True)
                visually_proven = (mapping_finite and (
                                    not strict_fail_closed
                                    or (ALLOW_LEGACY_VISIBLE_FACES_IN_STRICT_MODE
                                        and _mapped_segment_has_visual_proof(
                                        strict_visual_profile, r0, sub_dur,
                                        target_frames, o_pos, src_dur,
                                        strict_vocal_active,
                                        strict_vocal_fps,
                                        output_start_frame=output_start_frame,
                                        onset_times=strict_onset_times,
                                        onset_envelope=strict_onset_envelope))))
                if not visually_proven:
                    visual_hidden += 1
                    made = _call_filler_exact(
                        filler_cb, sub_dur, out, target_frames)
                    made = bool(made and _video_has_exact_frames(
                        out, target_frames))
                    if not made:
                        out.unlink(missing_ok=True)
                        made = _make_black_no_mouth(out, target_frames)
                    if made and _video_has_exact_frames(out, target_frames):
                        seg_files.append(out)
                    else:
                        filler_fail += 1
                    done += sub_dur
                    continue
                if use_rate:
                    # DTWが示す局所的なMV進行量をそのままsub_durに収める。
                    # 従来は常に等速だったため、10%テンポ差なら4秒窓の
                    # 末尾で400msずれ、次カットで急に戻る鋸歯状の口ずれが出ていた。
                    factor = sub_dur / max(0.05, src_dur)
                    vf = f"setpts={factor:.6f}*PTS,{VF_NORM}"
                else:
                    vf = VF_NORM
                # fpsフィルター任せの区間別丸めを禁止し、累積境界で決めた枚数に固定。
                # tpadは入力末尾の丸めで1枚不足する場合だけ最終フレームを補う。
                vf = _exact_frame_filter(vf, target_frames)
                rr = _run(["ffmpeg", "-y", "-ss", f"{o_pos:.3f}", "-t", f"{src_dur:.3f}",
                           "-i", str(mv_path), "-vf", vf, *ENC_ARGS, "-an",
                           "-frames:v", str(target_frames), str(out)])
                if (rr.returncode == 0
                        and _video_has_exact_frames(out, target_frames)):
                    seg_files.append(out)
                else:
                    seg_fail += 1
                    if verbose and seg_fail == 1:
                        print("     ⚠️ 区間の切り出しに失敗（ffmpeg出力・末尾）:")
                        for ln in _tail(rr.stderr or rr.stdout, 6):
                            print("        ", ln)
                done += sub_dur
        else:
            # ドロップ/VJ区間（歌詞なし）→ filler_cb（原曲MVループ等）。doc12はVJカット無し。
            # 歌唱区間のフレーム境界を全体タイムライン上で維持するため進めておく。
            target_frames = frame_clock.take(total_dur)
            if target_frames <= 0:
                continue
            out = tmp_dir / f"vlip_{seg_idx:03d}.mp4"
            seg_idx += 1
            dur = total_dur
            made = _call_filler_exact(filler_cb, dur, out, target_frames)
            made = bool(made and _video_has_exact_frames(out, target_frames))
            # 認証callbackが使えない場合も、未検証の原曲MVへは戻さない。
            if not made:
                out.unlink(missing_ok=True)
                out = tmp_dir / f"vlip_{seg_idx:03d}.mp4"
                seg_idx += 1
                made = _make_black_no_mouth(out, target_frames)
            if made and _video_has_exact_frames(out, target_frames):
                seg_files.append(out)
            else:
                filler_fail += 1

    if verbose and visual_hidden:
        print(f"     🛡️ 全frame視覚証明の不合格 {visual_hidden}片を"
              "安全背景へ退避")

    if not seg_files:
        if verbose:
            print(f"  ❌ リップシンク映像を1本も作れませんでした（切り出し失敗{seg_fail}件）")
        return False
    if seg_fail or filler_fail:
        if verbose:
            print(f"  ⚠️ 映像区間が欠落（歌唱{seg_fail}/安全背景{filler_fail}）。"
                  "後半を前詰めせず全体を不採用")
        return False

    # --- 結合（concat demuxerは相対パスをlistの場所基準で解決するので絶対パスで書く） ---
    listf = tmp_dir / "vlip_concat.txt"
    with open(listf, "w") as f:
        for s in seg_files:
            f.write(f"file '{Path(s).resolve()}'\n")
    combined = tmp_dir / "vlip_combined.mp4"
    expected_video_frames = frame_clock.frame_boundary
    expected_video_dur = expected_video_frames / OUTPUT_FPS
    concat_tol = (0.20 if strict_fail_closed else
                  max(1.0, min(3.0, expected_video_dur * 0.01)))
    concat_ok, rr = _concat_video_exact(
        listf, combined, expected_video_frames)
    if not concat_ok or not _valid_video(combined, expected_video_dur, concat_tol):
        if verbose:
            print("  ❌ リップシンク映像の結合に失敗（ffmpeg出力・末尾）:")
            for ln in _tail(rr.stderr or rr.stdout, 8):
                print("        ", ln)
        return False

    # --- 長さ合わせ（音声より短ければ延長） ---
    current_frames = _decoded_video_frame_count(combined) or 0
    target_music_frames = max(1, int(round(float(music_dur) * OUTPUT_FPS)))
    if current_frames < target_music_frames:
        pad_frames = target_music_frames - current_frames
        pad = pad_frames / OUTPUT_FPS
        padf = tmp_dir / "vlip_pad.mp4"
        made = _call_filler_exact(filler_cb, pad, padf, pad_frames)
        made = bool(made and _video_has_exact_frames(padf, pad_frames))
        if not made:
            padf.unlink(missing_ok=True)
            made = _make_black_no_mouth(padf, pad_frames)
        if strict_fail_closed and not made:
            if verbose:
                print("  ⚠️ 末尾の安全背景を生成できないため全体を不採用")
            return False
        if made:
            l2 = tmp_dir / "vlip_ext.txt"
            with open(l2, "w") as f:
                f.write(f"file '{Path(combined).resolve()}'\n"
                        f"file '{Path(padf).resolve()}'\n")
            ext = tmp_dir / "vlip_combined_ext.mp4"
            ext_expected_frames = current_frames + pad_frames
            ext_expected = ext_expected_frames / OUTPUT_FPS
            ext_tol = (0.20 if strict_fail_closed else
                       max(1.0, min(3.0, ext_expected * 0.01)))
            ext_ok, er = _concat_video_exact(
                l2, ext, ext_expected_frames)
            if ext_ok and _valid_video(ext, ext_expected, ext_tol):
                combined = ext
            else:
                if verbose:
                    print("     ⚠️ 延長映像の結合に失敗。元の長さで続行します")
                    for ln in _tail(er.stderr or er.stdout, 5):
                        print("        ", ln)
                if strict_fail_closed:
                    return False

    # --- 音楽と合成 ---
    output_path = Path(output_path)
    mux_ok, rr = _mux_video_audio_exact(
        combined, music_path, output_path, target_music_frames, music_dur)
    final_tol = 0.20 if strict_fail_closed else 1.0
    if (mux_ok and _valid_video(output_path, music_dur, tolerance=final_tol)
            and _has_av_streams(output_path)):
        if verbose:
            mb = output_path.stat().st_size / 1024 / 1024
            tag = "（Demucs）" if m1 == "demucs" else "（HPSS近似）"
            print(f"  ✅ リップシンク完成{tag}: {output_path.name} ({mb:.1f} MB)")
        return True
    output_path.unlink(missing_ok=True)
    if verbose:
        print("  ❌ 音楽との合成に失敗（ffmpeg出力・末尾）:")
        for ln in _tail(rr.stderr or rr.stdout, 8):
            print("        ", ln)
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
