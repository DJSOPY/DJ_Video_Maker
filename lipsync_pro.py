#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
#  DJ LipSync Pro  —  実験用・高精度リップシンク（別系統）
#  created by DJ SOPY / @sousouagain
#
#  ※既存の DJ_Video_Maker（dj_maker_core.py / 安定版doc12）には
#    一切触りません。これは完全に独立した「実験コマンド」です。
#
#  パイプライン（MIR提案の最終アーキテクチャ）:
#    Demucs(vocal stem)              ← vocal_sync.separate_vocals を再利用
#      → 特徴量 HuBERT or MFCC
#      → subsequence DTW（窓ごとtop-k）
#      → Viterbiで全体経路推定（サビ反復/並べ替えに強い／境界ジャンプは固定コストで許可）
#      → 区分的(piecewise)平滑化：後退ジャンプで区切り各区間内のみ単調化＝hook-first保持
#      → 局所DTW微調整
#      → 子音オンセット ピーク合わせ（±0.5s）
#      → 映像warp（setptsでMVを remix時間軸へ伸縮）＋ remix音をmux
#
#  重いステージ（HuBERT等）は無ければ自動で軽い方に落ちます。
# ============================================================

import os
import sys
import subprocess
import tempfile
import shutil
import hashlib
import json
import bisect
from pathlib import Path

import numpy as np

# numba/librosaが読み取り専用site-packages内へcacheを書こうとして無効化
# される環境でも、口マイクロ補正を黙って飛ばさない。
_NUMBA_CACHE_DIR = Path.home() / ".dj_video_maker" / "numba_cache"
try:
    _NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE_DIR))
except OSError:
    try:
        _NUMBA_CACHE_DIR = Path(tempfile.gettempdir()) / f"djvm_numba_{os.getuid()}"
        _NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE_DIR))
    except OSError:
        pass

# ---- 安全な再利用（vocal_sync は __main__ ガードあり=importしてOK）----
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
try:
    from vocal_sync import separate_vocals, SR, _load_mono
except Exception as e:
    print("❌ vocal_sync.py が読み込めません。DJ_LipSync_Pro.command と同じフォルダに")
    print("   vocal_sync.py（と dj_maker_core.py）を置いてください。")
    print("   詳細:", e)
    sys.exit(1)

# ---- 出力設定 ----
OUT_W, OUT_H, FPS = 1280, 720, 30
WIN_SEC = 5.0       # subsequence DTW の窓
HOP_SEC = 2.5       # 窓のホップ
TOPK = 8            # 窓ごとの候補数（昼/夜など別テイク・MV後半も拾うため多め）
JUMP_PENALTY = 0.08 # Viterbi: 等速からのズレ1秒あたりの罰則（大きいほど単調寄り＝順番厳守）
# --- 並べ替え（hook-first / サビ頭出し）対応：区分的単調マッチング ---
# 等速ズレの罰則は SWITCH_COST で頭打ちにする＝区間境界で「一度だけ固定コストを払えば
# どこへでも飛べる」。これより小さいズレ（=区間内の自然なドリフト）では従来と完全に同一挙動。
SWITCH_COST  = 0.50 # 区間境界ジャンプの固定上限コスト（小さいほど飛びやすい／大きいほど単調寄り）
BACK_JUMP_MIN = 6.0 # smooth_anchors が「本物の区間境界」とみなす後退量(秒)。これ以上の後退で区切る
INTRO_VOICED_FRAC = 0.18  # 冒頭がこの割合以上歌っていれば hook-first とみなしイントロ捨てをしない
HUBERT_LAYER = 9    # HuBERT中間層（音素/歌詞情報が最も濃い。最終層は抽象的すぎてリップシンクに不利）
HUBERT_CHUNK_SEC = 20.0  # 長尺を一括推論せず、attentionのメモリ爆発を防ぐ
HUBERT_OVERLAP_SEC = 1.0 # チャンク端の特徴劣化を捨てるための前後文脈
DTW_DS = 2          # subsequence DTW を内部で時間1/DS にダウンサンプルして高速化（候補の粗い位置決め用。
                    #   後段の local_refine / 子音合わせはフル解像度のままなので最終精度は無傷）
# --- 多様化（MV全体をまんべんなく使い切る）---
DIV_CONT = 0.02     # 連続性（前と滑らかに繋ぐ）。小さめ＝多様化を優先
DIV_REUSE = 1.0     # 同じMV区間を再利用するほど罰則（大きいほど別テイク/未使用へ）
DIV_RECENT = 1.4    # 直近で使った区間を続けて使う強い罰則（連続重複の禁止）
DIV_UNUSED = 0.6    # まだ一度も使ってないMV区間を優先するボーナス（全体カバー）
DIV_BIN = 4.0       # MV区間を何秒単位で「使用済み」管理するか
DIV_RECENT_N = 4    # 「直近」とみなす窓数
CONSONANT_SEARCH = 0.5  # 子音合わせの探索 ±秒
CONSONANT_MIN_SIM = 0.24   # 子音オンセット補正を信じる最低コサイン類似度
CONSONANT_MIN_GAIN = 0.035 # 現在位置(0ms)からこれだけ改善した時だけ動かす
CONSONANT_MIN_PROM = 1.0   # 探索ピークの突出度

# --- 自動ズレ補正の信頼度ゲート（波形オンセット相関が曖昧な区間で誤爆させない）---
OFFSET_MIN_R   = 0.25   # 相関係数がこれ未満の区間は「合っていない」→補正しない
OFFSET_BIG_SEC = 0.20   # これより大きい補正は
OFFSET_BIG_R   = 0.40   #   さらに高い相関を要求（大ジャンプ誤爆=±300ms級を防ぐ）
OFFSET_BIG_PROM = 2.5   #   ＋ピークの突出度も要求

# --- 粗ズレ補正（区間まるごと±数秒ズレてる時に、強い確信がある場合だけ引き戻す）---
COARSE_MAX_SEC   = 8.0   # 区間単位で探す最大ズレ（±秒）
COARSE_MIN_SHIFT = 0.45  # これ未満は微調整に任せる（粗補正は大きいズレ専用）
COARSE_MIN_R     = 0.42  # 粗補正を適用する相関係数の下限（微調整より厳しめ）
COARSE_MIN_PROM  = 4.0   # ＋ピーク突出度（広い探索なので高め＝誤爆防止）

# --- 発音内容(HuBERT特徴)ベースの粗ズレ探索（リズムは同じだが歌詞位置が数秒ズレてる時用）---
#   オンセット(リズム)相関では「繰り返すサビの別位置」を見抜けないため、発音内容で探す。
FEAT_COARSE_MIN_SIM  = 0.40  # 採用する平均コサイン類似度の下限（厳しめ＝強い一致時のみ動く）
FEAT_COARSE_MIN_PROM = 4.0   # ＋ピーク突出度（δスイープ内でどれだけ突出してるか）

# --- 無声ゲート（Remixが歌ってない区間で「MVで歌ってる口パクカット」を避ける）---
VOC_GATE = True      # ON: Remix無声区間は、MVも歌ってないカット（Bロール等）を優先
SILENCE_PEN = 3.0    # Remix無声なのにMVが歌ってる候補への罰則（強め）
VOC_HOP = 0.05       # 声エネルギー包絡の時間解像度（秒）
# 検出済み顔以外の見逃しを実写MVで完全否定できないため、絶対安全モード
# では別MVカットをBロール認証せず、危険区間を必ず非人物背景へ送る。
ALLOW_REAL_MV_SAFE_BROLL = False
SYNC_FIRST = True    # 歌ってる区間は同期最優先（その区間だけ多様化を弱める）
FORCE_BROLL = True   # 無声区間で歌唱カットしか候補に無い時、非歌唱カットを強制的に当てる

# --- 口マイクロ補正（口の開きフラックス × 歌声オンセットで100ms級の残ズレを詰める）---
#   apply_mouth_lag（±4s粗補正・後半のみ・0.6s以上）が扱わない「0.6s未満の残ズレ」を
#   全区間対象・高分解能で仕上げる。Remixは音のオンセット相関が伴奏/マスタリング差で
#   誤爆しやすいため、映像側の口という“物理の真値”で微調整する。
#   このあとの Forced Alignment最終再固定(±0.55s)と補完関係：単語が一致する所は
#   再固定が勝ち、Whisperが効かない所（歌詞違い/加工強め）ではこの補正が残る。
MOUTH_MICRO      = True   # ON/OFF（mediapipeが無ければ自動でスキップ＝従来通り）
MOUTH_MICRO_MAX  = 0.6    # 探索する最大残ズレ（±秒）。粗補正の min_shift と接続
MOUTH_MICRO_MIN  = 0.06   # これ未満のズレは触らない（既に十分合っている）
MOUTH_MICRO_STEP = 0.02   # ラグ探索の刻み（放物線補間でさらに細かくなる）
MOUTH_MICRO_FACE = 0.45   # 区間の顔カバー率の下限
MOUTH_MICRO_CORR = 0.32   # 口フラックス×声オンセット相関の下限
MOUTH_MICRO_PROM = 2.0    # ピーク突出度の下限（曖昧なら補正しない）
CACHE_DIR = Path.home() / ".dj_video_maker" / "lipsync_pro_cache"
FEATURE_CACHE_SCHEMA = 2


def run(cmd, capture=True):
    return subprocess.run(cmd, capture_output=capture, text=True, errors="replace")


def ffprobe_dur(path):
    try:
        r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
        return float((r.stdout or "0").strip())
    except Exception:
        return 0.0


def _decoded_video_frame_count(path):
    """実際にデコードできる映像フレーム数を返す。取得不能はNone。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames",
             "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            return None
        value = (r.stdout or "").strip().splitlines()
        if not value or value[0] in ("", "N/A"):
            return None
        count = int(value[0])
        return count if count >= 0 else None
    except (OSError, TypeError, ValueError, IndexError):
        return None


def _video_has_exact_frames(path, expected_frames):
    """存在だけで成功扱いせず、全デコード後の枚数まで一致させる。"""
    try:
        expected_frames = int(expected_frames)
    except (TypeError, ValueError, OverflowError):
        return False
    p = Path(path)
    return bool(expected_frames > 0 and p.exists() and p.stat().st_size > 0
                and _decoded_video_frame_count(p) == expected_frames)


def _has_av_streams(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, errors="replace")
        kinds = set((r.stdout or "").split())
        return r.returncode == 0 and {"video", "audio"}.issubset(kinds)
    except OSError:
        return False


def _exact_frame_filter(base_filter, nframes):
    nframes = int(nframes)
    if nframes <= 0:
        raise ValueError("nframes must be positive")
    return (f"{base_filter},tpad=stop_mode=clone:stop_duration=1,"
            f"trim=end_frame={nframes},setpts=PTS-STARTPTS")


def _concat_segments_exact(list_path, output_path, expected_frames):
    """copy結合と互換再エンコードの両方を、指定枚数で検証する。"""
    output_path = Path(output_path)
    output_path.unlink(missing_ok=True)
    r = run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", str(list_path), "-c", "copy", str(output_path)])
    if (getattr(r, "returncode", 1) == 0
            and _video_has_exact_frames(output_path, expected_frames)):
        return True
    output_path.unlink(missing_ok=True)
    vf = _exact_frame_filter(f"fps={FPS}", expected_frames)
    r = run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", str(list_path), "-an", "-vf", vf,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-frames:v", str(int(expected_frames)), str(output_path)])
    return bool(getattr(r, "returncode", 1) == 0
                and _video_has_exact_frames(output_path, expected_frames))


def _mux_exact_video_audio(video_path, music_path, output_path, expected_frames):
    """予定枚数の映像と音声が両方ある最終ファイルだけを成功扱いする。"""
    output_path = Path(output_path)
    output_path.unlink(missing_ok=True)
    r = run(["ffmpeg", "-v", "error", "-y", "-i", str(video_path),
             "-i", str(music_path), "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
             # -shortestはAAC終端の丸めで末尾1〜2frameを落とすため使わない。
             # 映像は既に曲長の整数frameへ固定済みで、音声は自然終端する。
             "-frames:v", str(int(expected_frames)),
             "-movflags", "+faststart", str(output_path)])
    return bool(getattr(r, "returncode", 1) == 0
                and _video_has_exact_frames(output_path, expected_frames)
                and _has_av_streams(output_path))


def extract_audio_wav(src, dst, sr=44100):
    run(["ffmpeg", "-v", "quiet", "-y", "-i", str(src), "-vn",
         "-ac", "2", "-ar", str(sr), str(dst)])
    return dst


def is_static_video(video_path, samples=6):
    """ジャケ画像/音声のみアップロードの判定（動かない＝静止）。"""
    try:
        dur = ffprobe_dur(video_path)
        if dur <= 1.0:
            return False
        W, H = 64, 36
        frames = []
        for i in range(samples):
            t = dur * (i + 1) / (samples + 1)
            r = subprocess.run(
                ["ffmpeg", "-v", "quiet", "-ss", f"{t:.2f}", "-i", str(video_path),
                 "-frames:v", "1", "-vf", f"scale={W}:{H},format=gray",
                 "-f", "rawvideo", "-"], capture_output=True)
            buf = r.stdout or b""
            if len(buf) >= W * H:
                frames.append(np.frombuffer(buf[:W * H], dtype=np.uint8).astype(np.float32))
        if len(frames) < 2:
            return False
        diffs = [float(np.mean(np.abs(frames[i] - frames[i - 1]))) for i in range(1, len(frames))]
        return float(np.mean(diffs)) < 2.0
    except Exception:
        return False


def download_youtube(url, out_dir):
    """yt-dlp で1本ダウンロード（Chromeクッキー利用）。失敗時 None。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mv.mp4"
    base = ["yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
            "-o", str(out), url]
    # まず Chrome クッキーで、ダメならクッキー無しで
    for extra in (["--cookies-from-browser", "chrome"], []):
        r = run(base + extra)
        cands = list(out_dir.glob("mv.*"))
        vid = [c for c in cands if c.suffix.lower() in (".mp4", ".mkv", ".webm")]
        if vid:
            return vid[0]
    return None


# ============================================================
#  特徴量
# ============================================================
_HUBERT = {"fe": None, "model": None, "ok": None}


_DEVICE = {"torch": None}

def _torch_device():
    """配布先PCで一番速いデバイスを選ぶ: MPS(Apple GPU) → CUDA → CPU。一度だけ判定してキャッシュ。
    どの段階で失敗しても最終的に 'cpu' に落ちるので、配布先で安全。"""
    if _DEVICE["torch"] is not None:
        return _DEVICE["torch"]
    dev = "cpu"
    try:
        import torch
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
    except Exception:
        dev = "cpu"
    _DEVICE["torch"] = dev
    return dev


def _whisper_device():
    """faster-whisper / WhisperX(CTranslate2) は MPS 非対応。CUDAがあれば使い、無ければCPU(int8)。"""
    return ("cuda", "float16") if _torch_device() == "cuda" else ("cpu", "int8")


def _free_device_mem():
    """MPS/CUDAのキャッシュを解放（曲を重ねてもGPUメモリが積み上がらないように）。
    HuBERTの中間データ(13層×フレーム×768)は大きく、解放しないと連続処理でMPSが枯渇する。"""
    try:
        import torch, gc
        gc.collect()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


def _try_load_hubert():
    if _HUBERT["ok"] is not None:
        return _HUBERT["ok"]
    try:
        import torch  # noqa
        from transformers import Wav2Vec2FeatureExtractor, HubertModel
        _HUBERT["fe"] = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        _HUBERT["model"] = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        _HUBERT["model"].eval()
        # 一番速いデバイスへ載せる（失敗時はCPUのまま）
        dev = _torch_device()
        try:
            _HUBERT["model"].to(dev); _HUBERT["dev"] = dev
        except Exception:
            _HUBERT["dev"] = "cpu"
        if _HUBERT.get("dev", "cpu") != "cpu":
            print(f"     ⚡ HuBERT: {_HUBERT['dev'].upper()} を使用")
        _HUBERT["ok"] = True
    except Exception as e:
        _HUBERT["ok"] = False
        _HUBERT["err"] = str(e)
    return _HUBERT["ok"]


def _hubert_features(wav_mono, sr):
    import torch
    import librosa
    y16 = librosa.resample(wav_mono, orig_sr=sr, target_sr=16000) if sr != 16000 else wav_mono
    y16 = np.asarray(y16, dtype=np.float32)
    fe, model = _HUBERT["fe"], _HUBERT["model"]
    dev = _HUBERT.get("dev", "cpu")

    def _run(samples, on_dev):
        with torch.no_grad():
            inp = fe(samples, sampling_rate=16000, return_tensors="pt").input_values
            if on_dev != "cpu":
                inp = inp.to(on_dev)
            out = model(inp, output_hidden_states=True)
            hs = out.hidden_states          # tuple: 長さ13（embeddings + 12層）
            layer = min(HUBERT_LAYER, len(hs) - 1)
            return hs[layer][0].float().cpu().numpy()   # (T,768) ~20ms hop

    # 数分の楽曲をHuBERTへ一括入力すると、Transformerのattentionで
    # メモリが急増し、本来のHuBERTからMFCCへ降格しやすかった。
    # 20秒の本体に前後1秒の文脈を付け、本体のフレームだけを繋ぐ。
    chunk_n = max(16000, int(round(HUBERT_CHUNK_SEC * 16000)))
    overlap_n = max(0, int(round(HUBERT_OVERLAP_SEC * 16000)))
    pieces, piece_times = [], []
    n_chunks = max(1, int(np.ceil(len(y16) / chunk_n)))
    for ci, core_start in enumerate(range(0, max(1, len(y16)), chunk_n)):
        core_end = min(len(y16), core_start + chunk_n)
        if core_end - core_start < 400:
            break
        ext_start = max(0, core_start - overlap_n)
        ext_end = min(len(y16), core_end + overlap_n)
        samples = y16[ext_start:ext_end]
        try:
            part = _run(samples, dev)
        except Exception as e:
            # MPS等のop非対応・メモリ不足なら、その場でCPUに
            # 切り替え。特徴方式自体はHuBERTのまま維持する。
            if dev == "cpu":
                raise
            print(f"     ⚠️ HuBERT {dev.upper()} で失敗→CPUにフォールバック: {str(e)[:80]}")
            try:
                model.to("cpu")
            except Exception:
                pass
            dev = "cpu"
            _HUBERT["dev"] = "cpu"
            part = _run(samples, dev)

        # 実際の出力フレーム数からこのチャンク内時刻を求める。
        # 概ね20ms hopだが、切り上げ誤差をチャンク間に蓄積させない。
        ext_dur = (ext_end - ext_start) / 16000.0
        pt = ext_start / 16000.0 + (np.arange(len(part)) + 0.5) * (ext_dur / max(1, len(part)))
        core_lo = core_start / 16000.0
        core_hi = core_end / 16000.0
        keep = (pt >= core_lo) & ((pt < core_hi) if ci < n_chunks - 1 else (pt <= core_hi))
        if np.any(keep):
            pieces.append(part[keep])
            piece_times.append(pt[keep])

    if not pieces:
        raise RuntimeError("HuBERT特徴フレームが取得できませんでした")
    feat = np.concatenate(pieces, axis=0)
    times = np.concatenate(piece_times).astype(np.float64)
    if n_chunks > 1:
        print(f"     🧩 HuBERT長尺入力: {n_chunks}チャンクに分割（境界オーバーラップ付き）")
    _free_device_mem()   # HuBERTの中間データ(大)を曲ごとに解放してMPS枯渇を防ぐ
    return feat.astype(np.float32), times


def _mfcc_features(wav_mono, sr):
    import librosa
    hop = 512
    m = librosa.feature.mfcc(y=wav_mono, sr=sr, n_mfcc=20, hop_length=hop)
    d = librosa.feature.delta(m)
    feat = np.vstack([m, d]).T  # (T,40)
    times = librosa.frames_to_time(np.arange(feat.shape[0]), sr=sr, hop_length=hop)
    return feat.astype(np.float32), times


def _zscore(feat):
    mu = feat.mean(axis=0, keepdims=True)
    sd = feat.std(axis=0, keepdims=True) + 1e-8
    return (feat - mu) / sd


def extract_features(wav_mono, sr, tag, use_hubert):
    """戻り値: (feat[T,D] zスコア化, times[T], 名前)"""
    if use_hubert and _try_load_hubert():
        try:
            f, t = _hubert_features(wav_mono, sr)
            print(f"     🧠 特徴量[{tag}]: HuBERT 層{HUBERT_LAYER} ({f.shape[1]}次元 / {len(t)}フレーム)")
            return _zscore(f), t, "hubert"
        except Exception as e:
            print(f"     ⚠️ HuBERT失敗[{tag}]→MFCCへ: {e}")
    f, t = _mfcc_features(wav_mono, sr)
    print(f"     🎚 特徴量[{tag}]: MFCC ({f.shape[1]}次元 / {len(t)}フレーム)")
    return _zscore(f), t, "mfcc"


# ============================================================
#  subsequence DTW（窓ごと top-k）
# ============================================================
def _topk_local_minima(cost, k, min_gap):
    order = np.argsort(cost)
    picks = []
    for idx in order:
        if all(abs(idx - p) >= min_gap for p in picks):
            picks.append(int(idx))
        if len(picks) >= k:
            break
    return [(p, float(cost[p])) for p in picks]


def _dtw_backtrack_start(D, end_col):
    """librosaの既定DTWステップで、指定終点から開始列を復元。

    subsequence DTWの累積コスト最終行は「どこで終わったか」しか
    直接は示さない。テンポ差があるのに開始を `終了-窓長`
    と仮定すると、10秒窓で数秒ずれ得る。top-kの終点だけを
    バックトラックすることで、大きな追加メモリ無しで真の開始を得る。"""
    D = np.asarray(D)
    if D.ndim != 2 or D.size == 0:
        return 0
    i = D.shape[0] - 1
    j = int(np.clip(end_col, 0, D.shape[1] - 1))
    guard = D.shape[0] + D.shape[1] + 4
    while i > 0 and guard > 0:
        guard -= 1
        if j <= 0:
            i -= 1
            continue
        # 既定の許可ステップ: 斜め / 横 / 縦。同点なら斜めを優先。
        prev = np.asarray((D[i - 1, j - 1], D[i, j - 1], D[i - 1, j]), dtype=float)
        prev[~np.isfinite(prev)] = np.inf
        move = int(np.argmin(prev))
        if move == 0:
            i -= 1; j -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
    return max(0, j)


def _subseq_dtw_accumulate_python(C):
    """subsequence DTWの累積コスト。Numbaからも呼べる単純実装。"""
    rows, cols = C.shape
    D = np.empty((rows, cols), dtype=np.float64)
    D[0, :] = C[0, :]                 # 参照のどこからでも開始可
    for i in range(1, rows):
        D[i, 0] = C[i, 0] + D[i - 1, 0]
        for j in range(1, cols):
            prev = D[i - 1, j - 1]
            if D[i, j - 1] < prev:
                prev = D[i, j - 1]
            if D[i - 1, j] < prev:
                prev = D[i - 1, j]
            D[i, j] = C[i, j] + prev
    return D


_SUBSEQ_DTW_COMPILED = None


def _subseq_dtw(X, Y):
    """Librosa/Numbaの設置状態に依存しないsubsequence DTW。"""
    from scipy.spatial.distance import cdist
    C = cdist(np.asarray(X, dtype=np.float64).T,
              np.asarray(Y, dtype=np.float64).T, metric="cosine")
    # 無音/ゼロ特徴のcosineはNaNになるため「不一致」扱い。
    C = np.nan_to_num(C, nan=1.0, posinf=1.0, neginf=1.0)
    global _SUBSEQ_DTW_COMPILED
    if _SUBSEQ_DTW_COMPILED is None:
        try:
            from numba import njit
            _SUBSEQ_DTW_COMPILED = njit(cache=False)(_subseq_dtw_accumulate_python)
        except Exception:
            _SUBSEQ_DTW_COMPILED = _subseq_dtw_accumulate_python
    try:
        return _SUBSEQ_DTW_COMPILED(C)
    except Exception:
        # Python/Numbaの組み合わせが合わないMacでも最終的に続行。
        _SUBSEQ_DTW_COMPILED = _subseq_dtw_accumulate_python
        return _SUBSEQ_DTW_COMPILED(C)


def windowed_topk(rfeat, rt, ofeat, ot, win_sec=WIN_SEC, hop_sec=HOP_SEC, topk=TOPK, ds=DTW_DS):
    if len(rt) < 2 or len(ot) < 2:
        return []
    fr = 1.0 / max(np.median(np.diff(rt)), 1e-6)
    ds = max(1, int(ds))
    # int()の切り捨てと奇数hopにより、ds=2では窓開始が
    # phase 0/1と交互になっていた。参照側は常にphase 0 (ofeat[::2])なので、
    # 同一音声ですら隔窓で完全一致を失う。窓長・hopをdsの倍数へ丸め、
    # 全窓のダウンサンプル位相を固定する。
    wlen = max(2 * ds, int(round((win_sec * fr) / ds)) * ds)
    whop = max(ds, int(round((hop_sec * fr) / ds)) * ds)
    ofps = 1.0 / max(np.median(np.diff(ot)), 1e-6)
    min_gap = max(2, int(2.0 * ofps / ds))
    Yd = np.ascontiguousarray(ofeat[::ds].T)  # (D, Torig/ds) ★DTWを軽くするため時間1/ds
    windows = []
    i = 0
    while i < len(rfeat):
        win = rfeat[i:i + wlen]
        if len(win) < wlen * 0.6:
            break
        try:
            wd = np.ascontiguousarray(win[::ds].T)
            # backtrack=False（経路_wpは未使用だった）＋ 1/ds で大幅高速化。Dの最終行だけ使う
            D = _subseq_dtw(wd, Yd)
            lastrow = D[-1]
            cand = _topk_local_minima(lastrow, topk, min_gap)
            rt0 = float(rt[i])
            rt_end = float(rt[min(i + wlen - 1, len(rt) - 1)])
            wdur = max(0.1, rt_end - rt0)
            nlen = max(1, len(win[::ds]))
            out = []
            for col, cost in cand:
                start_col = _dtw_backtrack_start(D, col)
                osc = min(start_col * ds, len(ot) - 1)
                start_ot = max(0.0, float(ot[osc]))
                # コストはフレーム数で正規化（平均コサイン距離＝従来と同スケール。Viterbi罰則は不変でOK）
                out.append((start_ot, cost / nlen))
            # 異なる終了列が同じDTW開始位置へ戻ることがある。同一候補で
            # top-kを消費しないよう、開始時刻ごとに最低コストだけを残す。
            unique = {}
            for pos, cc in out:
                key = round(float(pos), 2)
                if key not in unique or cc < unique[key][1]:
                    unique[key] = (float(pos), float(cc))
            out = sorted(unique.values(), key=lambda x: x[1])[:topk]
            windows.append((rt0, wdur, out))
        except Exception as e:
            print(f"     ⚠️ DTW窓スキップ @ {rt[i]:.1f}s: {e}")
        i += whop
    return windows


def merge_windows_by_time(windows, topk=TOPK, eps=1e-3):
    """粗/細スケールの同一remix時刻窓を1つにまとめる。
    同じrtが連続するとViterbiがd_rt=0の遷移を踏み、境界ジャンプ判定が不安定になる。"""
    if not windows:
        return []
    out = []
    for rt0, wdur, cands in sorted(windows, key=lambda w: w[0]):
        if out and abs(out[-1][0] - rt0) <= eps:
            prt, pwdur, pcands = out[-1]
            by_pos = {}
            for ot0, cost in pcands + list(cands):
                key = round(float(ot0), 2)
                prev = by_pos.get(key)
                if prev is None or cost < prev[1]:
                    by_pos[key] = (float(ot0), float(cost))
            merged = sorted(by_pos.values(), key=lambda x: x[1])[:topk]
            out[-1] = (prt, min(pwdur, wdur), merged)
        else:
            out.append((rt0, wdur, list(cands)[:topk]))
    return out


# ============================================================
#  Viterbi で全体経路（サビ反復・並べ替えに強い）
# ============================================================
def viterbi_path(windows, jump_penalty=JUMP_PENALTY, switch_cost=SWITCH_COST,
                 strict_monotonic=False):
    """窓候補から最小コスト経路を求める。

    ``strict_monotonic`` は後方互換のため既定では無効。True の場合は
    MV時刻が後戻りする遷移を「高い罰則」ではなく到達不能として扱う。
    switch_cost=inf だけでは大きな後退も有限コストのため選べてしまう。
    """
    if not windows:
        return []
    if any(not cs for (_rt, _wd, cs) in windows):
        return []
    # emissionコスト正規化
    allc = [c for (_, _, cs) in windows for (_, c) in cs]
    cmin, cmax = (min(allc), max(allc)) if allc else (0.0, 1.0)
    rng = max(1e-6, cmax - cmin)

    def emit(c):
        return (c - cmin) / rng  # 0..1（小さいほど良い）

    N = len(windows)
    dp = []      # 各窓: 各候補までの最小累積コスト
    bk = []      # backpointer
    # 初期化
    rt0, _wd0, cs0 = windows[0]
    dp.append([emit(c) for (_, c) in cs0])
    bk.append([-1] * len(cs0))
    for j in range(1, N):
        rtj, _wdj, csj = windows[j]
        rti, _wdi, csi = windows[j - 1]
        d_rt = rtj - rti
        row, brow = [], []
        for (otj, cj) in csj:
            best, bidx = 1e18, -1
            for ki, (oti, _ci) in enumerate(csi):
                d_ot = otj - oti
                if strict_monotonic and d_ot < -1e-9:
                    continue
                trans = jump_penalty * abs(d_ot - d_rt)  # 等速からのズレ罰則
                if trans > switch_cost:                  # 区間境界の大ジャンプは固定コストで頭打ち
                    trans = switch_cost                  # →hook-first/サビ頭出し等の「後退」を許可
                v = dp[j - 1][ki] + trans
                if v < best:
                    best, bidx = v, ki
            row.append(best + emit(cj))
            brow.append(bidx)
        dp.append(row)
        bk.append(brow)
    # バックトラック
    finite_last = [i for i, v in enumerate(dp[-1]) if np.isfinite(v) and v < 1e17]
    if not finite_last:
        return []
    last = min(finite_last, key=lambda i: dp[-1][i])
    path_idx = [last]
    for j in range(N - 1, 0, -1):
        last = bk[j][last]
        if last < 0:
            return []
        path_idx.append(last)
    path_idx.reverse()
    anchors = []
    for j, ki in enumerate(path_idx):
        rt0, _wd, cs = windows[j]
        ot0, c = cs[ki]
        anchors.append([rt0, ot0, c])
    return anchors


def choose_viterbi_path(windows, rfeat, rt, ofeat, ot, rvoc=None, ovoc=None, sr=None,
                        rmx_act=None, feature_kind="hubert", verbose=True):
    """並べ替え許可版と『単調（順番を崩さない）版』の両方の経路を作り、
    実際の音の一致品質で勝った方を採用する。

    なぜ必要か:
      viterbi_path の遷移コストは switch_cost で頭打ちになるため、
      「どれだけ遠くへ飛んでも罰は一定」＝サビが繰り返される曲では
      別のサビへ飛んだ方が得になり、経路がバラバラに壊れることがある。
      hook-first等の本物の並べ替えリミックスには必要な性質だが、
      素直なリミックス（順番そのまま・テンポ違い）では誤爆になる。
      → 両方作って、局所の一致（block p20）で決める。単調側を既定で優遇する。
    """
    reord = viterbi_path(windows)
    # 飛びの頭打ちを外すだけでは後退は有限コストのままなので、
    # 単調候補では後戻り遷移そのものを禁止する。
    mono = viterbi_path(windows, switch_cost=float("inf"), strict_monotonic=True)

    def _score(anchors):
        if len(anchors) < 2:
            return None
        q = alignment_quality_report(anchors, windows, rfeat, rt, ofeat, ot,
                                     rvoc=rvoc, ovoc=ovoc, sr=sr,
                                     rmx_act=rmx_act, feature_kind=feature_kind)
        return q

    qr, qm = _score(reord), _score(mono)
    if qm is None:
        return reord
    if qr is None:
        return mono

    # 単調側を既定にし、並べ替え側が「はっきり良い」ときだけ乗り換える。
    # （反復サビへの誤マッチは局所p20に出るので、そこを主指標にする）
    better = (qr["block_similarity_p20"] >= qm["block_similarity_p20"] + 0.02
              and qr["feature_similarity"] >= qm["feature_similarity"])
    if verbose:
        print(f"     🧭 経路比較: 単調 局所p20={qm['block_similarity_p20']:.2f}"
              f"/類似{qm['feature_similarity']:.2f}"
              f" ｜ 並べ替え 局所p20={qr['block_similarity_p20']:.2f}"
              f"/類似{qr['feature_similarity']:.2f}"
              f" → {'並べ替え' if better else '単調'}を採用")
    return reord if better else mono


def _voc_envelope(voc, sr, hop=VOC_HOP):
    """声のRMS包絡から『歌ってる/無声』を判定 → (times, active_bool)。失敗時None。"""
    try:
        h = max(1, int(sr * hop))
        n = len(voc) // h
        if n < 2:
            return None
        v = np.asarray(voc, dtype=np.float32)
        rms = np.sqrt(np.maximum(0.0, np.array(
            [np.mean(v[i*h:(i+1)*h] ** 2) for i in range(n)], dtype=np.float64)))
        peak = float(np.percentile(rms, 95))
        if not np.isfinite(peak) or peak <= 1e-7:
            # 完全無音では全フレームFalse。従来は sqrt(+1e-9)の
            # 床値から閾値を作ったため、無音が全て歌唱扱いだった。
            return (hop, np.zeros(n, dtype=bool))
        floor = float(np.percentile(rms, 20))
        # 分離stemの残響/ノイズ床と実ボーカルの間に閾値を置く。
        thr = floor + 0.15 * max(0.0, peak - floor)
        active = rms > max(thr, peak * 0.035)
        # 50msの単発ノイズを歌唱としない。前後いずれかに声がある
        # フレームだけを残し、短い子音は消しすぎない。
        if len(active) >= 3:
            active = active & (np.r_[False, active[:-1]] | np.r_[active[1:], False])
        return (hop, active)
    except Exception:
        return None


def _act_at(act, t):
    """時刻tで声が出てるか。actが無ければ True 扱い（＝従来通り）。"""
    if not act:
        return True
    hop, active = act
    i = int(min(max(round(t / hop), 0), len(active) - 1))
    return bool(active[i])


def _active_fraction(act, start, end):
    """時間窓内のボーカルactive率。actが無ければ1.0。"""
    if not act:
        return 1.0
    hop, active = act
    if len(active) == 0:
        return 0.0
    i0 = int(np.clip(np.floor(float(start) / hop), 0, len(active) - 1))
    i1 = int(np.clip(np.ceil(float(end) / hop), i0 + 1, len(active)))
    return float(np.mean(active[i0:i1])) if i1 > i0 else float(active[i0])


def _first_voiced_time(rmx_act, default=0.0):
    if not rmx_act:
        return default
    hop, active = rmx_act
    idx = np.where(active)[0]
    return float(idx[0] * hop) if len(idx) else default


def _last_voiced_time(rmx_act, music_dur):
    if not rmx_act:
        return music_dur
    hop, active = rmx_act
    idx = np.where(active)[0]
    return float(idx[-1] * hop) if len(idx) else music_dur


def intro_outro_rough(anchors, music_dur, mv_dur, bpm, rmx_act):
    """MVよりremixが長い時、実際の無声イントロ/アウトロだけをラフ配置にする。

    以前はほぼ同じ曲尺でも固定の8小節+16小節を精密同期から
    外していた。末尾まで歌う曲では正しい口パクを大きく壊すため、
    ボーカルstemで確認できた無声部だけに不足分を逃がす。
    戻り: (新anchors, (intro_end,outro_start,intro_bars,outro_bars) or None)"""
    if not anchors or not rmx_act:
        return anchors, None
    bpm = max(60.0, min(200.0, bpm if bpm and bpm > 0 else 125.0))
    sec_per_bar = 4.0 * 60.0 / bpm            # 1小節＝4拍
    deficit = music_dur - mv_dur              # MV不足分
    # 1〜2秒程度のコンテナ/VBR誤差で構成を変えない。
    if deficit <= max(3.0, music_dur * 0.015):
        return anchors, None
    intro_end = max(0.0, _first_voiced_time(rmx_act, 0.0))
    last_voice = min(music_dur, _last_voiced_time(rmx_act, music_dur))
    # 最後のactiveフレーム自体をラフにしないよう、1hop後から。
    outro_start = min(music_dur, last_voice + float(rmx_act[0]))
    quiet_capacity = intro_end + max(0.0, music_dur - outro_start)
    if quiet_capacity + 0.5 < deficit:
        # 無声部だけで不足を吸収できない時は、歌唱部の口パクを
        # 捨てるより、精密アンカーの再利用を優先する。
        return anchors, None
    if outro_start <= intro_end + 5.0:
        return anchors, None
    precise_intro_o = _interp_orig(intro_end, anchors)
    mid = [list(a) for a in anchors if intro_end <= a[0] <= outro_start]
    if len(mid) < 3:
        return anchors, None
    if intro_end <= 0.5:
        intro = []
        print("     🎯 hook-first 検出：冒頭が歌→イントロ捨てを無効化（先頭の一致を保持）")
    else:
        # 無声イントロはMV先頭を流すが、歌い出し時刻ちょうどで
        # 元の精密経路へ戻す。o=intro_endのラフ位置を置くと、
        # 次の2.5秒アンカーまで歌唱映像が遅れるため、元経路を補間。
        intro = [[0.0, 0.0, 1.0], [intro_end, precise_intro_o, 1.0]]
    last_mid_mv = mid[-1][1]
    out_src0 = max(mv_dur - (music_dur - outro_start), last_mid_mv)
    out_src0 = min(out_src0, mv_dur - 1.0)
    outro = ([[outro_start, out_src0, 1.0], [music_dur, mv_dur, 1.0]]
             if music_dur - outro_start > 0.5 else [])
    merged = sorted(intro + mid + outro, key=lambda a: a[0])
    intro_bars = intro_end / sec_per_bar
    outro_bars = max(0.0, music_dur - outro_start) / sec_per_bar
    return merged, (intro_end, outro_start, intro_bars, outro_bars)


def diversified_path(windows, cont=DIV_CONT, reuse=DIV_REUSE, recent=DIV_RECENT,
                     unused=DIV_UNUSED, bin_sec=DIV_BIN, recent_n=DIV_RECENT_N,
                     rmx_act=None, mv_act=None, gate=VOC_GATE, silence_pen=SILENCE_PEN,
                     sync_first=SYNC_FIRST, force_broll=FORCE_BROLL):
    """top-k候補から、MV全体をまんべんなく使い切るように選ぶ。
    ・繰り返しの歌詞には別テイク（昼→夜など）を回す
    ・まだ使ってないMV区間を優先（全体カバー）
    ・直近で使った区間は続けて使わない（連続重複の禁止）"""
    if not windows:
        return []
    allc = [c for (_, _, cs) in windows for (_, c) in cs]
    cmin, cmax = (min(allc), max(allc)) if allc else (0.0, 1.0)
    rng = max(1e-6, cmax - cmin)

    def ne(c):
        return (c - cmin) / rng

    # MV全体のbin数（カバー率ログ用）
    all_bins = set()
    for (_, _, cs) in windows:
        for (ot, _c) in cs:
            all_bins.add(int(ot // bin_sec))

    usage = {}          # bin -> 累計使用回数
    recent_bins = []    # 直近で使ったbin
    anchors = []
    prev_ot = None
    prev_rt = None

    # MVの「静かなbin（主に非歌唱）」を集める＝無声区間のBロール差し替え用
    quiet_bin_times = []
    if mv_act:
        hop_a, active_a = mv_act
        agg = {}
        for i, a in enumerate(active_a):
            bb = int((i * hop_a) // bin_sec)
            q, tot = agg.get(bb, (0, 0))
            agg[bb] = (q + (0 if a else 1), tot + 1)
        quiet_bin_times = sorted(bb * bin_sec + bin_sec * 0.5
                                 for bb, (q, tot) in agg.items() if tot and q / tot >= 0.5)
    quiet_used = {}

    for (rt0, wdur, cands) in windows:
        r_voiced = _act_at(rmx_act, rt0)     # Remixはこの窓で歌ってるか
        # 同期最優先：歌ってる窓は多様化を弱め、ベスト一致カットが勝ちやすくする
        # A=同期の正確さ最優先：歌ってる区間は多様化をほぼ切り、一番口が合うMV位置を選ぶ
        _reuse = reuse * (0.08 if (sync_first and r_voiced) else 1.0)
        _unused = unused * (0.08 if (sync_first and r_voiced) else 1.0)
        # 歌ってる窓は「前に進む連続性」を強める＝順番をキープ（繰り返し歌詞での飛びを抑制）
        _cont = cont * (6.0 if (sync_first and r_voiced) else 1.0)
        best, best_sc = None, 1e18
        for (ot, cost) in cands:
            b = int(ot // bin_sec)
            u = usage.get(b, 0)
            sc = ne(cost)
            sc += _reuse * u                     # 使うほど嫌う＝別テイク/未使用へ
            if u == 0:
                sc -= _unused                    # まだ使ってない区間を優先（全体カバー）
            if b in recent_bins:
                sc += recent                     # 直近の区間は続けて使わない
            if prev_ot is not None:              # 連続性（暴れ防止・上限付き）
                expected = prev_ot + (rt0 - prev_rt)
                sc += _cont * min(abs(ot - expected), 60.0)
            if gate and (not r_voiced) and _act_at(mv_act, ot):
                sc += silence_pen                # Remix無声なのにMVが歌ってる→口パク回避
            if sc < best_sc:
                best_sc, best = sc, (ot, cost, b)
        ot, cost, b = best
        # 無声なのに選ばれたのが歌唱カット → 非歌唱カットを強制的に当てる（網羅優先で回す）
        if force_broll and (not r_voiced) and quiet_bin_times and _act_at(mv_act, ot):
            qt = min(quiet_bin_times, key=lambda q: quiet_used.get(int(q // bin_sec), 0))
            ot = qt; b = int(qt // bin_sec)
            quiet_used[b] = quiet_used.get(b, 0) + 1
        usage[b] = usage.get(b, 0) + 1
        recent_bins.append(b)
        recent_bins = recent_bins[-recent_n:]
        anchors.append([rt0, ot, cost])
        prev_ot, prev_rt = ot, rt0
    # 使ったMV区間の広がり＋カバー率をログ
    used = sorted(usage.keys())
    span = (max(used) - min(used) + 1) * bin_sec if used else 0
    cover = (len(used) / max(1, len(all_bins))) * 100
    print(f"     🎨 多様化選択: MV {len(used)}区間({span:.0f}秒幅)を使用 / "
          f"候補全体の{cover:.0f}%をカバー / 連続重複を回避")
    if gate and rmx_act:
        n_silent = sum(1 for (rt0, _, _) in windows if not _act_at(rmx_act, rt0))
        if n_silent:
            print(f"     🔇 無声区間 {n_silent}個 → 歌ってないカットを優先（口パク回避）")
    return anchors


# ============================================================
#  局所DTW微調整（各アンカー周辺で精密オフセット）
# ============================================================
def local_refine(anchors, rfeat, rt, ofeat, ot, span=WIN_SEC):
    import librosa
    fr_r = 1.0 / max(np.median(np.diff(rt)), 1e-6)
    fr_o = 1.0 / max(np.median(np.diff(ot)), 1e-6)
    refined = []
    for (r0, o0, c) in anchors:
        try:
            ri = int(np.searchsorted(rt, r0))
            oi = int(np.searchsorted(ot, o0))
            rw = rfeat[ri:ri + int(span * fr_r)]
            ow = ofeat[max(0, oi - int(0.5 * span * fr_o)):oi + int(1.5 * span * fr_o)]
            if len(rw) < 4 or len(ow) < 4:
                refined.append([r0, o0, c]); continue
            D, wp = librosa.sequence.dtw(X=rw.T, Y=ow.T, subseq=True,
                                         metric="cosine", backtrack=True)
            # wp[-1] が開始対応（librosaは終点→始点の順）
            ow_base_i = max(0, oi - int(0.5 * span * fr_o))
            o_start_frame = wp[-1][1]
            o0_ref = float(ot[min(ow_base_i, len(ot) - 1)]) + o_start_frame / fr_o
            if abs(o0_ref - o0) > span:
                o0_ref = o0
            refined.append([r0, o0_ref, c])
        except Exception:
            refined.append([r0, o0, c])
    return refined


# ============================================================
#  子音オンセット ピーク合わせ（±0.5s）
# ============================================================
def consonant_align(anchors, rvoc, ovoc, sr, search=CONSONANT_SEARCH):
    try:
        import librosa
        hop = 256
        os_r = librosa.onset.onset_strength(y=rvoc, sr=sr, hop_length=hop)
        os_o = librosa.onset.onset_strength(y=ovoc, sr=sr, hop_length=hop)
        t_r = librosa.frames_to_time(np.arange(len(os_r)), sr=sr, hop_length=hop)
        t_o = librosa.frames_to_time(np.arange(len(os_o)), sr=sr, hop_length=hop)
        win = 1.0  # 相関を取る窓（秒）
        out = []
        n_applied = 0; n_gated = 0
        for (r0, o0, c) in anchors:
            ri0 = np.searchsorted(t_r, r0); ri1 = np.searchsorted(t_r, r0 + win)
            seg_r = os_r[ri0:ri1]
            nr = float(np.linalg.norm(seg_r))
            if len(seg_r) < 4 or nr < 1e-8:
                out.append([r0, o0, c]); continue
            step = 0.02
            offsets = np.arange(-search, search + 1e-9, step)
            vals = np.full(len(offsets), np.nan, dtype=float)
            for kk, off in enumerate(offsets):
                oi0 = np.searchsorted(t_o, o0 + off)
                oi1 = oi0 + len(seg_r)
                seg_o = os_o[oi0:oi1]
                if len(seg_o) == len(seg_r) and len(seg_o) > 0:
                    no = float(np.linalg.norm(seg_o))
                    if no >= 1e-8:
                        vals[kk] = float(np.dot(seg_r, seg_o) / (nr * no + 1e-8))
            finite = np.isfinite(vals)
            if not np.any(finite):
                out.append([r0, o0, c]); continue
            best_i = int(np.nanargmax(vals))
            best_off = float(offsets[best_i]); best_val = float(vals[best_i])
            zero_i = int(np.argmin(np.abs(offsets)))
            zero_val = float(vals[zero_i]) if np.isfinite(vals[zero_i]) else 0.0
            fv = vals[finite]
            prom = (best_val - float(np.median(fv))) / (float(np.std(fv)) + 1e-8)
            # 現在位置から本当に改善し、かつピークが明確な時だけ
            # 補正。従来は無音/平坦でも常に-search端を選び、合って
            # いたアンカーを最大500ms壊すことがあった。
            passed = (best_val >= CONSONANT_MIN_SIM
                      and (best_val - zero_val) >= CONSONANT_MIN_GAIN
                      and prom >= CONSONANT_MIN_PROM)
            if abs(best_off) <= step * 0.51:
                passed = True  # 0msが最良なら「動かさない」ので安全
            if passed:
                out.append([r0, max(0.0, o0 + best_off), c])
                if abs(best_off) > step * 0.51:
                    n_applied += 1
            else:
                out.append([r0, o0, c]); n_gated += 1
        if n_applied:
            print(f"     🗣 子音ピーク補正: {n_applied}アンカー")
        if n_gated:
            print(f"     🛡 子音補正ゲート: {n_gated}アンカーは曖昧なため維持")
        return out
    except Exception as e:
        print(f"     ⚠️ 子音合わせスキップ: {e}")
        return anchors


# ============================================================
#  映像warp（setptsでMVを remix時間軸へ伸縮）＋ remix音をmux
# ============================================================
def despike_anchors(anchors, thresh=18.0, near=2.5, passes=2):
    """単発の飛び（前後どちらからも遠いが、前後同士は近い＝一瞬だけ別シーンへ飛んで戻る）を、
    前後の中点に均す。本物の並べ替え（hook first等＝前後ごと動く）は残す。
    """
    if len(anchors) < 3:
        return anchors
    a = [list(x) for x in anchors]
    fixed = 0
    for _ in range(passes):
        for i in range(1, len(a) - 1):
            op, oc, on = a[i - 1][1], a[i][1], a[i + 1][1]
            if abs(oc - op) > thresh and abs(oc - on) > thresh and abs(op - on) <= thresh * near:
                a[i][1] = 0.5 * (op + on)   # 前後の中点へ
                fixed += 1
    if fixed:
        print(f"     🩹 単発の飛びを{fixed}個ならしました（並べ替えは保持）")
    return [tuple(x) for x in a]


def _isotonic(y):
    """PAVA: yを非減少に矯正（隣接違反プールの平均化）。"""
    y = list(map(float, y))
    n = len(y)
    vals = y[:]          # 各ブロックの値
    wts = [1.0] * n
    idx = list(range(n))  # ブロック境界（簡易実装）
    out = y[:]
    # 単純実装：左から見て違反したら前ブロックと平均統合
    blocks = [[v, 1.0] for v in y]
    i = 0
    merged = []
    for v in y:
        merged.append([v, 1.0])
        while len(merged) >= 2 and merged[-2][0] > merged[-1][0]:
            v2, w2 = merged.pop()
            v1, w1 = merged.pop()
            merged.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    res = []
    for v, w in merged:
        res.extend([v] * int(round(w)))
    return res[:n] if len(res) >= n else res + [res[-1]] * (n - len(res))


def _smooth_run(run, mv_dur, lo=0.8, hi=1.3):
    """1区間（内部は前進方向）の列を despike＋単調化だけ。
    ★測定位置を尊重する：以前あった『MVを広く使い切る再スケール』は、正しく合っている
      アンカーまで引き伸ばして口パクをズラすため廃止した（再スケールは口ズレの主因だった）。
    lo/hi は後方互換のため残すが未使用。"""
    if len(run) < 4:
        return [list(x) for x in run]
    a = [list(x) for x in run]
    ot = np.array([x[1] for x in a], dtype=float)
    # 1) despike（局所中点から大きく外れる単発点。両隣が近い時のみ＝境界は触らない）
    for _ in range(2):
        for i in range(1, len(ot) - 1):
            mid = 0.5 * (ot[i - 1] + ot[i + 1])
            if abs(ot[i] - mid) > 15.0 and abs(ot[i - 1] - ot[i + 1]) < 30.0:
                ot[i] = mid
    # 2) 単調化（非減少）※測定位置に最も近い単調列。既に単調な点は動かさない
    ot = np.clip(np.array(_isotonic(ot), dtype=float), 0.0, mv_dur)
    for i in range(len(a)):
        a[i][1] = float(ot[i])
    return a


def slope_run_cleanup(anchors, slope_tol=0.30, min_run=4):
    """音内容アライン(content_align)の発想をリップシンクに応用した整形。
    slope≈1(1:1で前進)の連続ランを『信頼できる本筋』とみなし、
    どのランにも属さない短い飛びのうち、前後の本筋がMV的に連続している箇所
    （＝反復で別位置へ飛んで戻った blip）だけを、前後から補間して引き戻す。
    前後が不連続な飛び（＝本物の並べ替え境界、hook-first等）は触らない。
    戻り: 整形後アンカー（信頼ランが無ければ元のまま＝壊さない）。"""
    n = len(anchors)
    if n < min_run + 2:
        return anchors
    a = [list(x) for x in anchors]
    rt = [x[0] for x in a]; ot = [x[1] for x in a]
    # slope≈1 の連続ランに区切る
    runs = []; cur = [0]
    for i in range(1, n):
        dr = rt[i] - rt[i - 1]; do = ot[i] - ot[i - 1]
        if dr > 1e-3 and abs(do / dr - 1.0) <= slope_tol:
            cur.append(i)
        else:
            runs.append(cur); cur = [i]
    runs.append(cur)
    trusted = [r for r in runs if len(r) >= min_run]
    if not trusted:
        return anchors                      # 信頼できる本筋が無い→何もしない（壊さない）
    tset = set(i for r in trusted for i in r)
    ti = sorted(tset)
    n_fix = 0
    for k in range(len(ti) - 1):
        j0, j1 = ti[k], ti[k + 1]
        if j1 == j0 + 1:
            continue                        # 隙間なし
        dr = rt[j1] - rt[j0]; do = ot[j1] - ot[j0]
        # 前後の信頼アンカーが slope≈1 で連続している＝間の飛びは blip → 本筋に引き戻す
        if dr > 1e-3 and abs(do / dr - 1.0) <= slope_tol:
            for m in range(j0 + 1, j1):
                f = (rt[m] - rt[j0]) / dr
                ot[m] = ot[j0] + f * do
                n_fix += 1
        # 不連続（do/dr が 1 から大きく外れる）＝本物の並べ替え境界 → 触らない
    for m in range(n):
        a[m][1] = max(0.0, ot[m])
    if n_fix:
        print(f"     🧹 連続ラン整形：{n_fix}個の飛びを本筋に引き戻し（音内容アライン式）")
    return [tuple(x) for x in a]


def smooth_anchors(anchors, mv_dur, lo=0.8, hi=1.3):
    """『区分的単調』平滑化：後退ジャンプ（=区間境界）で区切り、各区間内だけを
       despike＋単調化＋MVを広く使い切る再スケール。区間をまたいで単調化しない
       ＝hook-first/サビ頭出しなどの本物の並べ替えを保持する。"""
    if len(anchors) < 4:
        return anchors
    a = [list(x) for x in anchors]
    ot = [x[1] for x in a]
    # 後退ジャンプ位置で区間を分割（down→up の単発スパイクは境界にしない）
    bounds = [0]
    for i in range(1, len(a)):
        if ot[i] - ot[i - 1] < -BACK_JUMP_MIN:
            nxt = ot[i + 1] if i + 1 < len(a) else ot[i]
            if nxt < ot[i - 1] - BACK_JUMP_MIN * 0.5:   # 低い側に留まる＝本物の境界
                bounds.append(i)
    bounds.append(len(a))
    n_seg = len(bounds) - 1
    out = []
    for s in range(n_seg):
        out.extend(_smooth_run(a[bounds[s]:bounds[s + 1]], mv_dur, lo, hi))
    if n_seg > 1:
        print(f"     📐 区分的平滑化：{n_seg}区間（並べ替え保持・各区間内のみ単調化）")
    else:
        print(f"     📐 単調＆MV広く使い切る矯正（1区間）")
    return [tuple(x) for x in out]


_WHISPER = {"model": None, "ok": None}        # faster-whisper（フォールバック）
_WHISPERX = {"model": None, "align": {}, "ok": None}   # WhisperX（高精度・優先）


def _try_load_whisper():
    if _WHISPER["ok"] is not None:
        return _WHISPER["ok"]
    try:
        from faster_whisper import WhisperModel
        wdev, wct = _whisper_device()
        try:
            _WHISPER["model"] = WhisperModel("base", device=wdev, compute_type=wct)
        except Exception:
            _WHISPER["model"] = WhisperModel("base", device="cpu", compute_type="int8")
        _WHISPER["ok"] = True
    except Exception as e:
        _WHISPER["ok"] = False
        _WHISPER["err"] = str(e)
    return _WHISPER["ok"]


def _try_load_whisperx():
    if _WHISPERX["ok"] is not None:
        return _WHISPERX["ok"]
    try:
        import whisperx
        wdev, wct = _whisper_device()
        try:
            _WHISPERX["model"] = whisperx.load_model("base", wdev, compute_type=wct)
            _WHISPERX["dev"] = wdev
        except Exception:
            _WHISPERX["model"] = whisperx.load_model("base", "cpu", compute_type="int8")
            _WHISPERX["dev"] = "cpu"
        _WHISPERX["ok"] = True
    except Exception as e:
        _WHISPERX["ok"] = False
        _WHISPERX["err"] = str(e)
    return _WHISPERX["ok"]


def _norm_word(w):
    return "".join(ch for ch in (w or "").lower() if ch.isalnum())


def _whisperx_words(y16):
    """WhisperX：Whisper文字起こし＋wav2vec2強制アラインで単語境界を高精度化（±20ms級）。"""
    import whisperx
    res = _WHISPERX["model"].transcribe(np.asarray(y16, dtype=np.float32), batch_size=16)
    lang = res.get("language", "en")
    adev = _WHISPERX.get("dev", "cpu")
    if lang not in _WHISPERX["align"]:
        try:
            am, meta = whisperx.load_align_model(language_code=lang, device=adev)
        except Exception:
            adev = "cpu"
            am, meta = whisperx.load_align_model(language_code=lang, device="cpu")
        _WHISPERX["align"][lang] = (am, meta)
    am, meta = _WHISPERX["align"][lang]
    aligned = whisperx.align(res["segments"], am, meta,
                             np.asarray(y16, dtype=np.float32), adev,
                             return_char_alignments=False)
    words = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and w.get("word"):
                t = _norm_word(w["word"])
                if t:
                    words.append([t, float(w["start"]), float(w.get("end", w["start"]))])
    return words


def _faster_whisper_words(y16):
    segs, _info = _WHISPER["model"].transcribe(
        np.asarray(y16, dtype=np.float32), word_timestamps=True, vad_filter=True)
    words = []
    for s in segs:
        for w in (s.words or []):
            t = _norm_word(w.word)
            if t:
                words.append([t, float(w.start), float(w.end)])
    return words


def _whisper_words(wav_mono, sr, tag):
    """ボーカルstemを単語タイムスタンプ化 → [(単語, 開始秒, 終了秒)]。
    WhisperX（高精度）優先 → faster-whisper（フォールバック）。音源ハッシュでキャッシュ。"""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        h = hashlib.md5(np.asarray(wav_mono, dtype=np.float32).tobytes()).hexdigest()[:16]
        fp = CACHE_DIR / f"{h}_words2.json"     # v2: WhisperX対応で旧キャッシュと区別
        if fp.exists():
            print(f"     ♻️ 単語タイムスタンプ[{tag}]キャッシュ再利用")
            return json.loads(fp.read_text())
        import librosa
        y16 = librosa.resample(wav_mono, orig_sr=sr, target_sr=16000) if sr != 16000 else wav_mono
        words, engine = None, None
        if _try_load_whisperx():
            try:
                words = _whisperx_words(y16); engine = "WhisperX"
            except Exception as e:
                print(f"     ⚠️ WhisperX失敗→faster-whisperへ: {e}"); words = None
        if not words and _try_load_whisper():
            words = _faster_whisper_words(y16); engine = "faster-whisper"
        if not words:
            return None
        fp.write_text(json.dumps(words))
        print(f"     🗣 単語タイムスタンプ[{tag}]: {len(words)}語（{engine}）")
        return words
    except Exception as e:
        print(f"     ⚠️ 単語タイムスタンプ[{tag}]失敗: {e}")
        return None


def whisper_word_align(anchors, rvoc, ovoc, sr, search=2.5):
    """Forced Alignment：Remixと原曲の歌詞を単語タイムスタンプ化し、
    既存アンカー(粗ガイド)が示すMV位置の近くで“同じ単語”にスナップ＝単語レベルの精密同期。
    歌詞が同じ(原曲アカペラ流用)前提で効く。使えない時は従来アンカーを返す。
    searchは後方互換の既定2.5秒。最終再固定では小さくして、大きな確信済み補正を守れる。"""
    if not anchors:
        return anchors
    guide = sorted(anchors, key=lambda a: float(a[0]))
    rw = _whisper_words(rvoc, sr, "remix")
    ow = _whisper_words(ovoc, sr, "MV")
    if not rw or not ow:
        print("     ℹ️ Forced Alignment: Whisper使用不可 → 従来アンカー維持")
        return anchors
    # MV単語: text -> 昇順の出現時刻
    mv_idx = {}
    for (w, s, _e) in ow:
        mv_idx.setdefault(w, []).append(s)
    for k in mv_idx:
        mv_idx[k].sort()
    SEARCH = max(0.05, float(search))  # 粗ガイドが示すMV位置の±この秒で同じ単語を探す
    # 並べ替え境界ごとに単語列を分け、同一区間内ではMV単語時刻の
    # 後戻りを許さない。反復サビの同じ短語を独立に最近傍選択すると、
    # 一語だけ前のサビへ戻ることがあるため。
    bounds = _segment_bounds_by_jump(guide)
    section_starts = [float(guide[i][0]) for i in bounds[:-1]]
    last_mv_by_section = {}
    snapped = []
    for (w, rs, _re) in sorted(rw, key=lambda x: float(x[1])):
        times = mv_idx.get(w)
        if not times:
            continue
        exp = _interp_orig(rs, guide)           # 粗ガイドのMV予想位置
        sec = max(0, bisect.bisect_right(section_starts, float(rs)) - 1)
        min_mv = last_mv_by_section.get(sec, -float("inf")) + 0.02
        lo = bisect.bisect_left(times, exp - SEARCH)
        hi = bisect.bisect_right(times, exp + SEARCH)
        eligible = [t for t in times[lo:hi] if t >= min_mv]
        best = min(eligible, key=lambda t: abs(t - exp)) if eligible else None
        bestd = abs(best - exp) if best is not None else 1e9
        if best is not None and bestd <= SEARCH:
            snapped.append([rs, best, 0.0])
            last_mv_by_section[sec] = best
    if len(snapped) < 8:
        print(f"     ℹ️ Forced Alignment: 一致語{len(snapped)}個（少）→ 従来アンカー維持")
        return anchors
    # 歌のない区間（単語アンカーが疎な所）は粗アンカーで補完
    snapped.sort(key=lambda a: a[0])
    # 同じ短語が反復するサビで、単1語だけ隣の出現時刻へ
    # ±2.5秒スナップすると、後段の単調化でも1秒以上残る。
    # 粗ガイドからの残差を近傍語と比べ、450ms以上の孤立外れ値だけを除外。
    residuals = np.array([float(o - _interp_orig(r, guide)) for r, o, _c in snapped])
    keep = np.ones(len(snapped), dtype=bool)
    for i, (r, _o, _c) in enumerate(snapped):
        neigh = [residuals[j] for j, (rr, _oo, _cc) in enumerate(snapped)
                 if j != i and abs(float(rr) - float(r)) <= 4.0]
        if len(neigh) >= 2 and abs(residuals[i] - float(np.median(neigh))) > 0.45:
            keep[i] = False
    n_word_outliers = int(np.sum(~keep))
    if n_word_outliers:
        snapped = [a for a, k in zip(snapped, keep) if k]
        print(f"     🛡 Forced Alignment: 孤立した単語ズレ{n_word_outliers}個を除外")
    if len(snapped) < 8:
        print(f"     ℹ️ Forced Alignment: 外れ値除外後の一致語{len(snapped)}個（少）→ 従来アンカー維持")
        return anchors
    wr = [a[0] for a in snapped]
    merged = list(snapped)
    for (r0, o0, c) in guide:
        k = bisect.bisect_left(wr, r0)
        near = min((abs(wr[i] - r0) for i in (k - 1, k) if 0 <= i < len(wr)), default=1e9)
        if near > 4.0:
            merged.append([r0, o0, c])
    merged.sort(key=lambda a: a[0])
    print(f"     🗣 Forced Alignment: {len(snapped)}語を単語境界にスナップ（口パク精度↑）")
    return [tuple(x) for x in merged]


def _mapping_interval_is_jump(a0, a1, rate_lo=0.8, rate_hi=1.25):
    """隣接アンカーが、伸縮ではなく別MV位置へのカットか。"""
    dr = float(a1[0] - a0[0]); do = float(a1[1] - a0[1])
    if dr <= 1e-9:
        return True
    rate = do / dr
    # Whisper単語境界は数百ms間隔のため、100ms程度の時刻揺れで
    # 見かけのrateが0.8〜1.25を外れる。それを全てシーンジャンプに
    # すると極短カットが量産されるので、速度域外に加え、絶対的に
    # 十分大きな予測差がある場合だけをカットとみなす。
    mismatch = abs(do - dr)
    material = mismatch >= max(0.75, 0.35 * dr)
    return material and (do <= 0.0 or rate < rate_lo or rate > rate_hi)


def _interp_orig(r, anchors):
    """アンカー列(rt,ot)から remix時刻r に対応する 原曲時刻o を補間。

    連続区間だけを線形補間し、別セクションへのジャンプは
    境界直前まで旧位置を等速で進める。境界時刻ちょうどは必ず
    新アンカーを返す（右連続）。"""
    if not anchors:
        return 0.0
    try:
        r = float(r)
        rtimes = [float(a[0]) for a in anchors]
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    if not np.isfinite(r) or not all(np.isfinite(x) for x in rtimes):
        return float("nan")
    def safe_nonnegative(value):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return float("nan")
        return max(0.0, value) if np.isfinite(value) else float("nan")
    # bisect_rightにより exact anchor はそのアンカー側を選ぶ。
    # 同一remix時刻が複数ある時も右端（最新の補正）を採用。
    k = bisect.bisect_right(rtimes, r) - 1
    if k < 0:
        return safe_nonnegative(float(anchors[0][1]) + (r - rtimes[0]))
    if k >= len(anchors) - 1:
        return safe_nonnegative(float(anchors[-1][1]) + (r - rtimes[-1]))
    r0, o0 = rtimes[k], float(anchors[k][1])
    r1, o1 = rtimes[k + 1], float(anchors[k + 1][1])
    if not np.isfinite(o0) or not np.isfinite(o1):
        return float("nan")
    if r1 <= r0:
        return safe_nonnegative(o0)
    if _mapping_interval_is_jump(anchors[k], anchors[k + 1]):
        return safe_nonnegative(o0 + (r - r0))
    f = (r - r0) / (r1 - r0)
    return safe_nonnegative(o0 + f * (o1 - o0))


def estimate_global_offset(anchors, rvoc, ovoc, sr, max_lag=0.4):
    """並べた経路に沿ってMVのボーカルonsetをremix時間に写像し、
    remixのボーカルonsetと相互相関 → 残った一定ズレ量(秒)を自動推定。
    戻り値: 補正秒（正=MVを後ろから取る=映像を前倒し）。"""
    try:
        import librosa
        hop = 256
        os_r = librosa.onset.onset_strength(y=rvoc, sr=sr, hop_length=hop)
        os_o = librosa.onset.onset_strength(y=ovoc, sr=sr, hop_length=hop)
        fr = sr / hop
        t_r = np.arange(len(os_r)) / fr
        mv_t = np.array([_interp_orig(float(t), anchors) for t in t_r])
        idx = np.clip((mv_t * fr).astype(int), 0, len(os_o) - 1)
        os_map = os_o[idx]
        a = (os_r - os_r.mean()) / (os_r.std() + 1e-8)
        b = (os_map - os_map.mean()) / (os_map.std() + 1e-8)
        L = int(max_lag * fr)
        best_lag, best = 0, -1e18
        for lag in range(-L, L + 1):
            if lag >= 0:
                v = float(np.dot(a[lag:], b[:len(b) - lag])) if len(b) - lag > 0 else -1e18
            else:
                v = float(np.dot(a[:lag], b[-lag:]))
            if v > best:
                best, best_lag = v, lag
        # b(=映像) が a(=音) より遅れている(best_lag<0)なら映像を前倒し(+)
        return -best_lag / fr
    except Exception as e:
        print(f"     ⚠️ 自動ズレ推定スキップ: {e}")
        return 0.0


def _lag_and_conf(a, b, L):
    """正規化相互相関の最良ラグ＋信頼度を返す。
      prom = (ピーク - 中央値)/std … ピークがノイズからどれだけ突出してるか
      r    = ピーク / 重なり長        … その位置の相関係数(≈Pearson, -1..1)
    _best_lag_xcorr と同じ規約（np.roll(+s)に対し -s を返す）。"""
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    lags = list(range(-L, L + 1))
    vals = np.full(len(lags), -1e18)
    ns = np.ones(len(lags))
    for k, lag in enumerate(lags):
        if lag >= 0:
            n = len(b) - lag
            if n > 0:
                vals[k] = float(np.dot(a[lag:], b[:n])); ns[k] = n
        else:
            n = len(a) + lag
            if n > 0:
                vals[k] = float(np.dot(a[:lag], b[-lag:])); ns[k] = n
    ki = int(np.argmax(vals))
    fin = vals[vals > -1e17]
    med = float(np.median(fin)) if fin.size else 0.0
    sd = float(fin.std()) + 1e-8 if fin.size else 1.0
    prom = (vals[ki] - med) / sd
    r = vals[ki] / max(1.0, ns[ki])
    return lags[ki], float(prom), float(r)


def _best_lag_xcorr(a, b, L):
    """正規化相互相関で a に対する b の最良ラグ(frame)を返す。純numpy・テスト可能。"""
    return _lag_and_conf(a, b, L)[0]


def _segment_bounds_by_backjump(anchors, back_min=BACK_JUMP_MIN):
    """後退ジャンプ位置で区間境界indexを返す（smooth_anchorsと同じ基準）。"""
    ot = [a[1] for a in anchors]
    bounds = [0]
    for i in range(1, len(anchors)):
        if ot[i] - ot[i - 1] < -back_min:
            nxt = ot[i + 1] if i + 1 < len(anchors) else ot[i]
            if nxt < ot[i - 1] - back_min * 0.5:
                bounds.append(i)
    bounds.append(len(anchors))
    return bounds


def _segment_bounds_by_jump(anchors, jump_min=BACK_JUMP_MIN):
    """MVが不連続に飛んだ位置(前進・後退どちらも)で区間境界を返す。
    ズレ補正専用：1区間に複数のMV位置が混ざる(=定数δが相殺して効かない)のを防ぐ。
    連続前進(slope≈1)の自然なテンポ差では割らず、MVが曲の進みから大きく外れた所だけ割る。"""
    rt = [a[0] for a in anchors]; ot = [a[1] for a in anchors]
    bounds = [0]
    for i in range(1, len(anchors)):
        do = ot[i] - ot[i - 1]
        dr = rt[i] - rt[i - 1]
        # 後退ジャンプ、または前進の飛び（MV進みが曲進みより jump_min 秒以上多い）
        if do < -jump_min or (do - dr) > jump_min:
            bounds.append(i)
    if bounds[-1] != len(anchors):
        bounds.append(len(anchors))
    return bounds


def _block_offset(seg, os_r, os_o, t_r, fr, L, r_lo, r_hi):
    """区間の一部[r_lo,r_hi]の残ズレ(秒)と信頼度を返す。測れなければ (None,0,0)。"""
    m = (t_r >= r_lo) & (t_r < r_hi)
    if int(m.sum()) < 8:
        return None, 0.0, 0.0
    mv_t = np.array([_interp_orig(float(t), seg) for t in t_r[m]])
    idx = np.clip((mv_t * fr).astype(int), 0, len(os_o) - 1)
    best_lag, prom, r = _lag_and_conf(os_r[m], os_o[idx], L)
    off = -best_lag / fr
    # 信頼度ゲート（曖昧な相関は不採用）
    if r < OFFSET_MIN_R:
        return None, prom, r
    if abs(off) > OFFSET_BIG_SEC and (r < OFFSET_BIG_R or prom < OFFSET_BIG_PROM):
        return None, prom, r
    return off, prom, r


def _feat_coarse_offset(seg, rfeat, rt, ofeat, ot, max_sec=COARSE_MAX_SEC, step=0.1):
    """区間まるごとの粗ズレを【発音内容(HuBERT特徴)】で探す。
    各候補シフトδについて、remix各フレームを「現アンカー写像+δ」のMV位置の特徴と
    コサイン類似度で突き合わせ、平均類似度が最大のδを返す。
    リズム(オンセット)では見抜けない『繰り返すサビの別位置(数秒ズレ)』を当てられる。
    戻り: (δ秒, prom突出度, peak平均類似度) / 測れなければ (None,0,0)。"""
    if len(seg) < 2 or rfeat is None or ofeat is None or len(rt) < 2 or len(ot) < 2:
        return None, 0.0, 0.0
    try:
        fr_r = 1.0 / max(np.median(np.diff(rt)), 1e-6)
        fr_o = 1.0 / max(np.median(np.diff(ot)), 1e-6)
        r_lo, r_hi = seg[0][0], seg[-1][0]
        ri0 = int(np.searchsorted(rt, r_lo)); ri1 = int(np.searchsorted(rt, r_hi))
        if ri1 - ri0 < 8:
            return None, 0.0, 0.0
        sub = max(1, int(round(0.1 * fr_r)))           # ~0.1sごとに間引いて高速化
        fidx = np.arange(ri0, ri1, sub)
        R = rfeat[fidx]
        Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
        m_t = np.array([_interp_orig(float(rt[f]), seg) for f in fidx])
        On = ofeat / (np.linalg.norm(ofeat, axis=1, keepdims=True) + 1e-8)
        deltas = np.arange(-max_sec, max_sec + 1e-9, step)
        sims = np.full(len(deltas), -1e18)
        for k, d in enumerate(deltas):
            oi = np.clip(((m_t + d) * fr_o).astype(int), 0, len(ofeat) - 1)
            sims[k] = float(np.mean(np.sum(Rn * On[oi], axis=1)))
        ki = int(np.argmax(sims))
        peak = sims[ki]; med = float(np.median(sims)); sd = float(sims.std()) + 1e-8
        return float(deltas[ki]), (peak - med) / sd, peak
    except Exception:
        return None, 0.0, 0.0


def apply_segment_offsets(anchors, rvoc, ovoc, sr, max_lag=0.4, block_sec=8.0,
                          rfeat=None, rt=None, ofeat=None, ot=None):
    """波形(オンセット包絡)の相互相関で残ズレを推定し【区間ごとに個別】補正する。
    並べ替え(hook-first等)は各セクションの残ズレが違うため、全体一律ではなく
    区間単位で合わせる。さらに各区間を ~block_sec ごとに測って区分線形で補正する
    ＝区間内で“じわじわズレていく”テンポ差(傾き)も吸収する。
    戻り: (補正後anchors, 各区間の補正量[ms])。"""
    if len(anchors) < 2:
        return anchors, []
    out = [list(a) for a in anchors]
    bounds = _segment_bounds_by_jump(anchors)
    offs_ms = []
    n_gated = 0
    n_drift = 0
    n_coarse = 0
    try:
        import librosa
        hop = 256
        os_r = librosa.onset.onset_strength(y=rvoc, sr=sr, hop_length=hop)
        os_o = librosa.onset.onset_strength(y=ovoc, sr=sr, hop_length=hop)
        fr = sr / hop
        t_r = np.arange(len(os_r)) / fr
        L = int(max_lag * fr)
        Lc = int(COARSE_MAX_SEC * fr)
        for s in range(len(bounds) - 1):
            i0, i1 = bounds[s], bounds[s + 1]
            seg = anchors[i0:i1]
            if len(seg) < 2:
                offs_ms.append(0); continue
            r_lo, r_hi = seg[0][0], seg[-1][0]
            span = r_hi - r_lo
            if span < 1e-3:
                offs_ms.append(0); continue

            # ---- 段階1: 粗ズレ補正（区間まるごと±数秒のズレを、強い確信時のみ引き戻す）----
            coarse = 0.0
            # まず発音内容(HuBERT特徴)で探す＝『リズムは同じだが歌詞位置が数秒ズレ』を当てる
            f_off, f_prom, f_sim = _feat_coarse_offset(seg, rfeat, rt, ofeat, ot, max_sec=COARSE_MAX_SEC)
            if (f_off is not None and abs(f_off) >= COARSE_MIN_SHIFT
                    and f_sim >= FEAT_COARSE_MIN_SIM and f_prom >= FEAT_COARSE_MIN_PROM):
                coarse = f_off
            else:
                # 特徴が無い/確信不足なら、従来のオンセット(リズム)相関で粗探索
                m_all = (t_r >= r_lo) & (t_r <= r_hi)
                if int(m_all.sum()) >= 8:
                    mv_t = np.array([_interp_orig(float(t), seg) for t in t_r[m_all]])
                    idx = np.clip((mv_t * fr).astype(int), 0, len(os_o) - 1)
                    c_lag, c_prom, c_r = _lag_and_conf(os_r[m_all], os_o[idx], Lc)
                    c_off = -c_lag / fr
                    if abs(c_off) >= COARSE_MIN_SHIFT and c_r >= COARSE_MIN_R and c_prom >= COARSE_MIN_PROM:
                        coarse = c_off
            if coarse:
                for i in range(i0, i1):
                    out[i][1] = max(0.0, out[i][1] + coarse)
                n_coarse += 1
            # 微調整は粗補正後の位置を基準にする
            seg2 = [(rt, max(0.0, ot + coarse), c) for (rt, ot, c) in seg] if coarse else seg

            # ---- 段階2: 区間内を ~block_sec ごとに測って残ズレ＋傾きを微調整 ----
            nblk = max(1, int(round(span / block_sec)))
            bt, bo = [], []
            for b in range(nblk):
                b_lo = r_lo + span * b / nblk
                b_hi = r_lo + span * (b + 1) / nblk
                off, prom, r = _block_offset(seg2, os_r, os_o, t_r, fr, L, b_lo, b_hi)
                if off is not None:
                    bt.append(0.5 * (b_lo + b_hi)); bo.append(off)
            if not bt:
                if not coarse:
                    n_gated += 1
                offs_ms.append(int(round(coarse * 1000))); continue
            for i in range(i0, i1):
                o_off = float(np.interp(anchors[i][0], bt, bo)) if len(bt) > 1 else bo[0]
                out[i][1] = max(0.0, out[i][1] + o_off)
            if len(bt) > 1 and (max(bo) - min(bo)) > 0.08:
                n_drift += 1
            offs_ms.append(int(round((coarse + float(np.mean(bo))) * 1000)))
    except Exception as e:
        print(f"     ⚠️ 区間ズレ推定スキップ: {e}")
        return anchors, []
    if n_coarse:
        print(f"     🎯 粗ズレ補正：{n_coarse}区間を±数秒の大きなズレから引き戻し")
    if n_gated:
        print(f"     🛡 信頼度ゲート：{n_gated}区間は相関が曖昧なため補正せず（誤爆防止）")
    if n_drift:
        print(f"     📈 区間内ドリフト補正：{n_drift}区間で傾き(じわじわズレ)を補正")
    return [tuple(x) for x in out], offs_ms


def _lyric_best_matches(tokens_r, owords, span_pad=6, min_sep=8.0,
                        want_index=False):
    """remix単語列をMV単語列へスライド照合し、時間的に離れた上位3候補を返す。

    単語単体の最近傍では区別できない反復サビを、前後を含む並び順で識別する。
    scoreはremix窓の被覆率で、末尾の余白量には左右されない。
    """
    import difflib
    n = len(tokens_r)
    if n < 6 or len(owords) < 6:
        return []
    mv_tok = [w[0] for w in owords]
    cands = []
    for i in range(0, max(1, len(mv_tok) - n + 1)):
        seg = mv_tok[i:i + n + span_pad]
        sm = difflib.SequenceMatcher(None, tokens_r, seg, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
        matched = sum(b.size for b in blocks)
        score = matched / float(n)
        j0 = i
        if blocks:
            a, b, _size = blocks[0]
            j0 = max(0, i + b - a)
        j0 = min(j0, len(owords) - 1)
        cands.append((float(owords[j0][1]), score, i))
    cands.sort(key=lambda x: -x[1])
    picked = []
    for t, score, i in cands:
        if all(abs(t - pt) >= min_sep for pt, *_rest in picked):
            picked.append((t, score, i) if want_index else (t, score))
        if len(picked) >= 3:
            break
    return picked


def _lyric_window_pins(wlist, owords, i_start, span_pad=6, min_block=3):
    """文脈一致した窓から、3語以上連続一致する単語時刻ピンを取り出す。"""
    import difflib
    tokens_r = [w[0] for w in wlist]
    mv_tok = [w[0] for w in owords]
    seg = mv_tok[i_start:i_start + len(tokens_r) + span_pad]
    sm = difflib.SequenceMatcher(None, tokens_r, seg, autojunk=False)
    pins = []
    for a, b, size in sm.get_matching_blocks():
        if size >= min_block:
            for k in range(size):
                ri = a + k
                oi = i_start + b + k
                if ri < len(wlist) and oi < len(owords):
                    pins.append((float(wlist[ri][1]), float(owords[oi][1])))
    return pins


def apply_lyrics_align(anchors, rwords, owords, music_dur=None, late_frac=0.45,
                       min_words=8, min_score=0.50, min_gap=0.15,
                       min_shift=1.0, mv_dur=None, win=20.0, hop=10.0):
    """歌詞文脈で、後半の「別の反復サビ」への誤接続を区間単位で直す。

    20秒窓を単語列として照合し、1位が十分強く2位から明確に離れる窓だけを
    採用する。数十秒の大移動にはさらに強い一致を要求し、曖昧なら何もしない。
    文脈検証済みの連続一致語はピンとして残し、区間内ドリフトも抑える。
    戻り: (補正後anchors, 補正した区間数)。
    """
    if not anchors or not rwords or not owords:
        return anchors, 0
    if music_dur is None:
        music_dur = max(a[0] for a in anchors)
    late_t = float(music_dur) * float(late_frac)
    out = [list(a) for a in anchors]
    bounds = _segment_bounds_by_jump(anchors)
    n_fix = 0
    all_pins = []
    for k in range(len(bounds) - 1):
        i0, i1 = bounds[k], bounds[k + 1]
        seg = anchors[i0:i1]
        if len(seg) < 3:
            continue
        r_lo, r_hi = float(seg[0][0]), float(seg[-1][0])
        if 0.5 * (r_lo + r_hi) < late_t:
            continue

        shifts = []
        seg_pins = []
        t = r_lo
        while t < r_hi:
            t1 = min(t + win, r_hi)
            wlist = [w for w in rwords if t <= float(w[1]) < t1]
            toks = [w[0] for w in wlist]
            if len(toks) >= min_words:
                picked = _lyric_best_matches(toks, owords, want_index=True)
                if picked:
                    best_t, best_s, best_i = picked[0]
                    gap = best_s - (picked[1][1] if len(picked) > 1 else 0.0)
                    if best_s >= min_score and gap >= min_gap:
                        r_ref = float(wlist[0][1])
                        cur_mv = _interp_orig(r_ref, seg)
                        shifts.append((t, best_t - cur_mv, best_s, gap))
                        seg_pins.extend(_lyric_window_pins(wlist, owords, best_i))
            t += hop
        if not shifts:
            print(f"     🎼 [見送り] remix{r_lo:.0f}-{r_hi:.0f}s: "
                  "一意な歌詞文脈なし（歌詞不足/反復が曖昧）")
            continue

        med = float(np.median([s for _t, s, _sc, _g in shifts]))
        spread = (float(np.std([s for _t, s, _sc, _g in shifts]))
                  if len(shifts) > 1 else 0.0)
        med_score = float(np.median([sc for _t, _s, sc, _g in shifts]))
        med_gap = float(np.median([g for _t, _s, _sc, g in shifts]))
        detail = " ".join(
            f"[{wt:.0f}s:{shift:+.1f}s({score:.2f})]"
            for wt, shift, score, _gap in shifts[:4])
        large_shift_weak = (abs(med) >= 8.0
                            and (med_score < 0.72 or med_gap < 0.20))
        if spread > (1.5 if abs(med) >= 8.0 else 2.0) or large_shift_weak:
            why = ("大移動の文脈確信不足" if large_shift_weak
                   else f"窓ごとの答えが不一致(ばらつき{spread:.1f}s)")
            print(f"     🎼 [見送り] remix{r_lo:.0f}-{r_hi:.0f}s: {why} {detail}")
            continue

        if abs(med) >= min_shift:
            for i in range(i0, i1):
                nv = float(out[i][1]) + med
                if mv_dur is not None:
                    nv = min(nv, max(0.0, float(mv_dur) - 0.05))
                out[i][1] = max(0.0, nv)
            n_fix += 1
            print(f"     🎼 歌詞文脈補正：remix{r_lo:.0f}-{r_hi:.0f}s を {med:+.1f}s {detail}")
        else:
            print(f"     🎼 [確認] remix{r_lo:.0f}-{r_hi:.0f}s: "
                  f"歌詞文脈は現在位置を支持（中央値{med:+.1f}s）")

        if seg_pins:
            seg_pins.sort(key=lambda p: p[0])
            clean = []
            for pr, po in seg_pins:
                if clean and pr - clean[-1][0] < 0.15:
                    continue
                if clean and po < clean[-1][1] - 0.5:
                    continue
                clean.append((pr, po))
            if len(clean) >= 3:
                all_pins.append((clean[0][0], clean[-1][0], clean))

    if all_pins:
        n_pin = 0
        kept = list(out)
        for p_lo, p_hi, clean in all_pins:
            kept = [a for a in kept if not (p_lo - 0.3 <= a[0] <= p_hi + 0.3)]
            kept.extend([[pr, po, 0.0] for pr, po in clean])
            n_pin += len(clean)
        kept.sort(key=lambda a: a[0])
        out = kept
        print(f"     🎼 歌詞文脈ピン：{n_pin}語を固定（反復誤認・区間内ドリフトを抑制）")
    return [tuple(x) for x in out], n_fix


def apply_mouth_lag(anchors, mouth_profile, rvoc, sr, max_lag=4.0, min_shift=0.6,
                    music_dur=None, late_frac=0.45):
    """口の動き(MV) × 歌声(remix) を区間ごとに相互相関し、数秒のズレを補正する。
    同じ歌詞の繰り返しでも『口の開閉 vs 発声』は物理的に同期するので、
    HuBERT特徴では測れなかった『2〜3秒ズレた別の繰り返しサビ』を測って引き戻せる。
    ★Remixは後半ほど崩れやすく、前半は概ね合っているため、曲の後半(late_frac以降)
      の区間だけを補正対象にする（前半の合っている所を壊さない）。
    顔が十分取れて相関が強い区間だけ補正（誤爆防止のゲートは measure 側）。
    戻り: (補正後アンカー, 補正区間数)。"""
    if not mouth_profile or not anchors:
        return anchors, 0
    try:
        import numpy as np
        import mouth_sync as _ms
    except Exception:
        return anchors, 0
    # 後半だけ補正する境界（曲長が分からなければアンカーのremix範囲から推定）
    if music_dur is None:
        music_dur = max(a[0] for a in anchors)
    late_t = music_dur * late_frac
    hop = max(1, int(sr * 0.05))
    n = len(rvoc) // hop
    if n < 4:
        return anchors, 0
    env = np.array([np.sqrt(np.mean(rvoc[i*hop:(i+1)*hop] ** 2) + 1e-12) for i in range(n)])
    env = env / (env.max() + 1e-9)
    vt = np.arange(n) * (hop / sr)
    out = [list(a) for a in anchors]
    bounds = _segment_bounds_by_jump(anchors)
    n_fix = 0
    for k in range(len(bounds) - 1):
        i0, i1 = bounds[k], bounds[k + 1]
        seg = anchors[i0:i1]
        if len(seg) < 2:
            continue
        # アンカー自体は通常2.5秒間隔で、0.1秒刻みのラグ探索には
        # 疎すぎる。区間内を10Hzに展開し、現アンカー経路上のMV時刻を
        # 毎点で求めてから口×声相関を測る。
        eval_lo = max(float(seg[0][0]), float(late_t))
        eval_hi = float(seg[-1][0])
        if eval_hi - eval_lo < 1.0:
            continue
        srt = np.arange(eval_lo, eval_hi + 0.050001, 0.1, dtype=float)
        srt = srt[srt <= eval_hi + 1e-6]
        if len(srt) < 10:
            continue
        smv = np.array([_interp_orig(float(t), seg) for t in srt], dtype=float)
        lag, corr, fcov = _ms.measure_segment_mouth_lag(mouth_profile, srt, smv, vt, env,
                                                        max_lag=max_lag, ret_always=True)
        # ゲートをここで明示判定（顔40%以上・相関0.35以上・ズレ0.6s以上のときだけ補正）
        passed = (fcov >= 0.40 and corr is not None and corr >= 0.35
                  and lag is not None and abs(lag) >= min_shift)
        if passed:
            for i in range(i0, i1):
                # late_tで実際に分け、前半アンカーまで一括シフト
                # しない。以前は全曲が1runだと平均時刻だけで後半と
                # 判定され、曲頭も動いていた。
                if anchors[i][0] >= late_t - 1e-9:
                    out[i][1] = max(0.0, out[i][1] + lag)
            n_fix += 1
            print(f"     👄 口×歌声ズレ補正：remix{srt[0]:.0f}-{srt[-1]:.0f}s を {lag:+.1f}s "
                  f"(相関{corr:.2f}/顔{fcov*100:.0f}%)")
        else:
            why = ("顔不足" if fcov < 0.40
                   else ("相関弱め" if (corr is None or corr < 0.35)
                         else ("ズレ小" if (lag is not None and abs(lag) < min_shift) else "—")))
            lv = f"{lag:+.1f}s" if lag is not None else "NA"
            print(f"     👄 [見送り] remix{srt[0]:.0f}-{srt[-1]:.0f}s: "
                  f"顔{fcov*100:.0f}% 相関{(corr if corr else 0):.2f} ズレ{lv} → {why}")
    return [tuple(x) for x in out], n_fix


def apply_mouth_micro_lag(anchors, mouth_profile, rvoc, sr,
                          max_lag=MOUTH_MICRO_MAX, min_shift=MOUTH_MICRO_MIN,
                          step=MOUTH_MICRO_STEP):
    """『口の開きフラックス × 歌声オンセット』で区間ごとの残ズレ(<0.6s)を微補正する。
    apply_mouth_lag は ±数秒の大ズレ専用（min_shift=0.6・後半のみ）なので、
    体感で一番目立つ 100〜500ms の残ズレは今まで音オンセット相関しか直せなかった。
    Remixは伴奏が原曲と違い音オンセットが誤爆/ゲート落ちしやすい → 口の開閉という
    物理の真値で全区間を仕上げる。ゲート（顔率・相関・突出度）未達の区間は触らない。
    ※この後の Forced Alignment最終再固定（±0.55s）が、単語一致の取れる区間を
      さらに精密化する。取れない区間ではこの補正がそのまま生きる。
    戻り: (補正後アンカー, 補正区間数)。"""
    if not MOUTH_MICRO or not mouth_profile or not anchors or len(anchors) < 2:
        return anchors, 0
    try:
        import librosa
        import mouth_sync as _ms
    except Exception:
        return anchors, 0
    if not hasattr(_ms, "measure_micro_mouth_lag"):
        return anchors, 0     # 旧mouth_sync.pyと同居しても安全に何もしない
    if mouth_profile.get("face_rate", 0.0) < 0.20:
        return anchors, 0     # 顔がほぼ無いMVでは測れない
    try:
        hop = 256
        os_r = librosa.onset.onset_strength(y=rvoc, sr=sr, hop_length=hop)
        fr = sr / hop
        onset_t = np.arange(len(os_r)) / fr
        os_r = os_r / (float(np.max(os_r)) + 1e-9)
    except Exception:
        return anchors, 0
    out = [list(a) for a in anchors]
    bounds = _segment_bounds_by_jump(anchors)
    n_fix = 0
    for k in range(len(bounds) - 1):
        i0, i1 = bounds[k], bounds[k + 1]
        seg = anchors[i0:i1]
        if len(seg) < 2:
            continue
        r_lo, r_hi = float(seg[0][0]), float(seg[-1][0])
        if r_hi - r_lo < 3.0:
            continue           # 短すぎる区間はイベント数不足＝信頼できない
        srt = np.arange(r_lo, r_hi + 1e-6, 0.05, dtype=float)
        smv = np.array([_interp_orig(float(t), seg) for t in srt], dtype=float)
        lag, corr, fcov, prom = _ms.measure_micro_mouth_lag(
            mouth_profile, srt, smv, onset_t, os_r,
            max_lag=max_lag, step=step, min_face=MOUTH_MICRO_FACE)
        passed = (lag is not None and fcov >= MOUTH_MICRO_FACE
                  and corr >= MOUTH_MICRO_CORR and prom >= MOUTH_MICRO_PROM
                  and min_shift <= abs(lag) <= max_lag)
        if passed:
            for i in range(i0, i1):
                out[i][1] = max(0.0, out[i][1] + lag)
            n_fix += 1
            print(f"     👄 口マイクロ補正：remix{r_lo:.0f}-{r_hi:.0f}s を {lag*1000:+.0f}ms "
                  f"(相関{corr:.2f}/顔{fcov*100:.0f}%/突出{prom:.1f})")
        elif lag is not None and abs(lag) >= min_shift:
            why = ("顔不足" if fcov < MOUTH_MICRO_FACE
                   else ("相関弱め" if corr < MOUTH_MICRO_CORR
                         else ("突出不足" if prom < MOUTH_MICRO_PROM else "—")))
            print(f"     👄 [微調整見送り] remix{r_lo:.0f}-{r_hi:.0f}s: "
                  f"ズレ{lag*1000:+.0f}ms 相関{corr:.2f} 顔{fcov*100:.0f}% 突出{prom:.1f} → {why}")
    return [tuple(x) for x in out], n_fix


def _merge_time_ranges(ranges, lo=0.0, hi=None, join_gap=0.05):
    """時刻範囲を正規化して結合する（診断・描画で共有する純粋関数）。"""
    clean = []
    for item in ranges or []:
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        start = max(float(lo), start)
        if hi is not None:
            end = min(float(hi), end)
        if end > start + 1e-9:
            clean.append((start, end))
    clean.sort()
    merged = []
    for start, end in clean:
        if merged and start <= merged[-1][1] + max(0.0, float(join_gap)):
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _unsafe_mask_ranges(times, unsafe, sample_sec=None):
    """サンプル単位の危険マスクを、端の半サンプルを含む時刻範囲へ変換。"""
    times = np.asarray(times, dtype=float)
    unsafe = np.asarray(unsafe, dtype=bool)
    if len(times) == 0 or len(times) != len(unsafe):
        return []
    if sample_sec is None:
        diffs = np.diff(times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
        sample_sec = float(np.median(diffs)) if len(diffs) else 0.1
    half = max(0.001, float(sample_sec) * 0.5)
    out = []
    i = 0
    while i < len(times):
        if not unsafe[i]:
            i += 1
            continue
        j = i + 1
        while j < len(times) and unsafe[j]:
            j += 1
        out.append((float(times[i]) - half, float(times[j - 1]) + half))
        i = j
    return out


def _prepare_unsafe_ranges(ranges, lo, hi, pad=0.25, min_safe=0.75):
    """危険区間を広げ、0.75秒未満の口元表示候補を危険側へ吸収する。"""
    try:
        lo = float(lo); hi = float(hi)
        pad = max(0.0, float(pad)); min_safe = max(0.0, float(min_safe))
    except (TypeError, ValueError, OverflowError):
        return []
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []
    padded = []
    for item in ranges or []:
        try:
            padded.append((float(item[0]) - pad, float(item[1]) + pad))
        except (TypeError, ValueError, IndexError, OverflowError):
            continue
    expanded = _merge_time_ranges(padded, lo=lo, hi=hi)
    if not expanded:
        return []
    closed = []
    for start, end in expanded:
        if not closed:
            if start - lo < min_safe - 1e-9:
                start = lo
            closed.append((start, end))
        elif start - closed[-1][1] < min_safe - 1e-9:
            closed[-1] = (closed[-1][0], max(closed[-1][1], end))
        else:
            closed.append((start, end))
    if hi - closed[-1][1] < min_safe - 1e-9:
        closed[-1] = (closed[-1][0], hi)
    return closed


def _sustained_inactive_ranges(rmx_act, lo, hi, min_silence=0.02,
                               max_active_island=0.20):
    """歌声が無い全解析frame区間を返す。

    ボーカルが無い時間は発音同期を証明できないため、原曲MV側の口が
    偶然動いていても見せない。短い無声も人物表示へ戻さない。
    """
    if not rmx_act:
        return []
    try:
        hop, active = rmx_act
        hop = float(hop); active = np.asarray(active, dtype=bool)
        lo = max(0.0, float(lo)); hi = max(lo, float(hi))
        min_silence = max(0.0, float(min_silence))
        max_active_island = max(0.0, float(max_active_island))
    except (TypeError, ValueError, OverflowError):
        return []
    if not np.isfinite(hop) or hop <= 0.0 or len(active) == 0 or hi <= lo:
        return []
    # Demucsの残留ノイズで100ms程度だけactiveになる島は、持続無声を
    # 分断する証拠にしない。無声に挟まれた短い島だけを橋渡しする。
    active = active.copy()
    max_island_frames = int(np.ceil(max_active_island / hop))
    if max_island_frames > 0:
        p = 0
        while p < len(active):
            if not active[p]:
                p += 1; continue
            q = p + 1
            while q < len(active) and active[q]:
                q += 1
            if p > 0 and q < len(active) and q - p <= max_island_frames:
                active[p:q] = False
            p = q

    i0 = max(0, int(np.floor(lo / hop)))
    i1 = min(len(active), int(np.ceil(hi / hop)))
    out = []
    i = i0
    while i < i1:
        if active[i]:
            i += 1
            continue
        j = i + 1
        while j < i1 and not active[j]:
            j += 1
        start = max(lo, i * hop)
        end = min(hi, j * hop)
        if end - start >= min_silence - 1e-9:
            out.append((start, end))
        i = j
    return out


def alignment_quality_report(anchors, windows, rfeat, rt, ofeat, ot,
                             rvoc=None, ovoc=None, sr=SR, rmx_act=None,
                             feature_kind="hubert", mouth_profile=None):
    """最終アンカーが「強い一致」を持つかを絶対値で検査。

    Viterbi内部の0..1正規化は、全候補が悪い別曲でも必ず
    「最良」を作ってしまう。ここでは、実コサイン類似度、無関係な
    時刻に対する上積み、ボーカルonset相関、カバー率、異常ジャンプ率を
    まとめ、明らかに弱い場合は旧方式へフォールバックさせる。"""
    report = {
        "accepted": False, "coverage": 0.0, "feature_similarity": -1.0,
        "tail_coverage": 0.0, "longest_invalid_seconds": float("inf"),
        "ending_invalid_seconds": float("inf"),
        "block_similarity_p20": -1.0, "longest_bad_seconds": float("inf"),
        "feature_lift": 0.0, "onset_correlation": 0.0,
        "median_dtw_cost": float("inf"), "p75_dtw_cost": float("inf"),
        "jump_fraction": 1.0, "unsafe_ranges": [],
    }
    if (not anchors or rfeat is None or ofeat is None or len(rt) < 2 or len(ot) < 2
            or np.ndim(rfeat) != 2 or np.ndim(ofeat) != 2
            or rfeat.shape[1] != ofeat.shape[1]):
        return report
    try:
        rt = np.asarray(rt, dtype=float); ot = np.asarray(ot, dtype=float)
        rfeat = np.asarray(rfeat, dtype=np.float32); ofeat = np.asarray(ofeat, dtype=np.float32)
        # 約0.1秒間隔に間引き。ボーカルがある点を優先する。
        feat_hop = max(float(np.median(np.diff(rt))), 1e-4)
        stride = max(1, int(round(0.1 / feat_hop)))
        all_ridx = np.arange(0, min(len(rt), len(rfeat)), stride, dtype=int)
        # 類似度を測れる点だけを後で残すのではなく、まず全時間軸で
        # 写像がMV範囲内かを検査する。これにより「前半70%が完全一致、
        # 末尾30%はMV終端の外」という経路を成功扱いしない。
        all_mapped = np.array(
            [_interp_orig(float(rt[i]), anchors) for i in all_ridx], dtype=float)
        range_tol = max(0.10, 2.0 * float(np.median(np.diff(ot))))
        all_valid = ((all_mapped >= float(ot[0]) - range_tol)
                     & (all_mapped <= float(ot[-1]) + range_tol))
        report["coverage"] = float(np.mean(all_valid)) if len(all_valid) else 0.0

        if len(all_ridx):
            all_times = rt[all_ridx]
            span = max(0.0, float(all_times[-1] - all_times[0]))
            tail_start = float(all_times[0]) + 0.70 * span
            tail_mask = all_times >= tail_start - 1e-9
            report["tail_coverage"] = (
                float(np.mean(all_valid[tail_mask])) if np.any(tail_mask) else 0.0)

            sample_sec = (float(np.median(np.diff(all_times)))
                          if len(all_times) >= 2 else feat_hop * stride)
            sample_sec = max(sample_sec, 1e-4)
            longest = current = 0.0
            for ok in all_valid:
                current = 0.0 if ok else current + sample_sec
                longest = max(longest, current)
            ending = 0.0
            for ok in all_valid[::-1]:
                if ok:
                    break
                ending += sample_sec
            report["longest_invalid_seconds"] = float(longest)
            report["ending_invalid_seconds"] = float(ending)
            report["unsafe_ranges"].extend(
                _unsafe_mask_ranges(all_times, ~all_valid, sample_sec))

        ridx = all_ridx
        if rmx_act is not None:
            voiced = np.array([_act_at(rmx_act, float(rt[i])) for i in ridx], dtype=bool)
            if int(np.sum(voiced)) >= 20:
                ridx = ridx[voiced]
        mapped = np.array([_interp_orig(float(rt[i]), anchors) for i in ridx], dtype=float)
        valid = ((mapped >= float(ot[0]) - range_tol)
                 & (mapped <= float(ot[-1]) + range_tol))
        ridx = ridx[valid]; mapped = mapped[valid]
        if len(ridx) >= 20:
            oi = np.searchsorted(ot, mapped, side="left")
            oi = np.clip(oi, 0, len(ot) - 1)
            R = rfeat[ridx]; O = ofeat[oi]
            Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
            On = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-8)
            sims = np.sum(Rn * On, axis=1)
            report["feature_similarity"] = float(np.median(sims))
            sim_times = rt[ridx]
            # 全曲中央値だけだと、前半60%が完璧で後半40%が
            # 別位置でも通ってしまう。2秒ブロックごとの平均と
            # 下位分位を持ち、局所的な崩れを検出する。
            block_sec = 2.0
            block_ids = np.floor(sim_times / block_sec).astype(int)
            block_pairs = []
            for bid in np.unique(block_ids):
                bm = block_ids == bid
                if int(np.sum(bm)) >= 4:
                    block_pairs.append((int(bid), float(np.mean(sims[bm]))))
            if block_pairs:
                report["block_similarity_p20"] = float(
                    np.percentile([v for _b, v in block_pairs], 20))
                report["_block_pairs"] = block_pairs
            # 同じ元特徴の「原曲の別時刻」と比べて上積みがあるか。
            shifts = [max(1, len(ofeat) // 3), max(1, len(ofeat) // 2)]
            bases = []
            for sh in shifts:
                Ob = ofeat[(oi + sh) % len(ofeat)]
                Ob = Ob / (np.linalg.norm(Ob, axis=1, keepdims=True) + 1e-8)
                bases.append(float(np.median(np.sum(Rn * Ob, axis=1))))
            report["feature_lift"] = report["feature_similarity"] - float(np.median(bases))

        costs = []
        for wr, wd, cands in (windows or []):
            finite_cands = [(float(p), float(c)) for p, c in cands if np.isfinite(c)]
            if (finite_cands and
                    (rmx_act is None
                     or _active_fraction(rmx_act, wr, float(wr) + float(wd)) >= 0.15)):
                # 各窓の最良値ではなく、最終アンカーが実際に選んだ
                # 位置に最も近い候補のコストを使う。
                mapped_pos = _interp_orig(float(wr), anchors)
                _p, chosen_cost = min(finite_cands, key=lambda pc: abs(pc[0] - mapped_pos))
                costs.append(chosen_cost)
        if not costs:
            costs = [float(a[2]) for a in anchors if len(a) > 2 and np.isfinite(a[2])]
        if costs:
            report["median_dtw_cost"] = float(np.median(costs))
            report["p75_dtw_cost"] = float(np.percentile(costs, 75))
        if len(anchors) >= 2:
            jumps = sum(_mapping_interval_is_jump(anchors[i - 1], anchors[i])
                        for i in range(1, len(anchors)))
            report["jump_fraction"] = float(jumps / max(1, len(anchors) - 1))
        else:
            report["jump_fraction"] = 0.0

        if rvoc is not None and ovoc is not None:
            try:
                import librosa
                hop = 256
                er = librosa.onset.onset_strength(y=np.asarray(rvoc), sr=sr, hop_length=hop)
                eo = librosa.onset.onset_strength(y=np.asarray(ovoc), sr=sr, hop_length=hop)
                tr = np.arange(len(er), dtype=float) * hop / sr
                mask = (tr >= anchors[0][0]) & (tr <= anchors[-1][0])
                if rmx_act is not None:
                    mask &= np.array([_act_at(rmx_act, float(x)) for x in tr], dtype=bool)
                if int(np.sum(mask)) >= 20 and len(eo) > 1:
                    mt = np.array([_interp_orig(float(x), anchors) for x in tr[mask]])
                    good = (mt >= 0.0) & (mt <= (len(eo) - 1) * hop / sr)
                    aa = er[mask][good]
                    bb = eo[np.clip(np.rint(mt[good] * sr / hop).astype(int), 0, len(eo) - 1)]
                    if len(aa) >= 20 and np.std(aa) > 1e-8 and np.std(bb) > 1e-8:
                        aa = (aa - aa.mean()) / (aa.std() + 1e-8)
                        bb = (bb - bb.mean()) / (bb.std() + 1e-8)
                        report["onset_correlation"] = float(np.mean(aa * bb))
            except Exception:
                pass

        is_hubert = "hubert" in str(feature_kind).lower()
        min_sim = 0.08 if is_hubert else 0.04
        min_block = 0.04 if is_hubert else 0.015
        max_cost = 1.10 if is_hubert else 1.25
        # 連続する弱一致ブロックの最長尺。無声でblock idが飛ぶと
        # 連続はリセットされる。
        longest = 0; cur = 0; prev_bid = None
        block_pairs = report.pop("_block_pairs", [])
        for bid, val in block_pairs:
            if val < min_block:
                cur = (cur + 1) if (prev_bid is not None and bid == prev_bid + 1) else 1
                longest = max(longest, cur)
                report["unsafe_ranges"].append(
                    (float(bid) * 2.0, (float(bid) + 1.0) * 2.0))
            else:
                cur = 0
            prev_bid = bid
        report["longest_bad_seconds"] = float(longest * 2.0)

        # 大きなMV位置ジャンプの境界を含む描画カットは保守的に隠す。
        for i in range(1, len(anchors)):
            if _mapping_interval_is_jump(anchors[i - 1], anchors[i]):
                boundary = float(anchors[i][0])
                report["unsafe_ranges"].append((boundary - 0.10, boundary + 0.10))

        # Remix側に歌声が無い持続区間は、音響写像が正しくても口形同期を
        # 証明できない。MVの歌う口を偶然表示しないよう常に非表示へ送る。
        if len(all_ridx):
            silence_lo = max(0.0, float(rt[0]))
            silence_hi = float(rt[min(len(rt), len(rfeat)) - 1])
            if rmx_act is None:
                # ボーカル有無の解析自体が失敗した区間は人物表示へ進めない。
                report["unsafe_ranges"].append((silence_lo, silence_hi))
            else:
                report["unsafe_ranges"].extend(_sustained_inactive_ranges(
                    rmx_act, silence_lo, silence_hi,
                    min_silence=0.02, max_active_island=0.20))

        # 音響一致が通っていても、歌声と口の動きが3点中2点で矛盾する
        # 2秒窓は口元を見せない。
        if mouth_profile is not None and rmx_act is not None and len(all_ridx):
            try:
                import mouth_sync as _ms_quality
                t0 = max(0.0, float(rt[0]))
                t1 = float(rt[min(len(rt), len(rfeat)) - 1])
                block_start = 2.0 * np.floor(t0 / 2.0)
                while block_start < t1 + 1e-9:
                    conflicts = 0
                    for remix_t in (block_start + 0.5, block_start + 1.0,
                                     block_start + 1.5):
                        if remix_t < t0 or remix_t > t1:
                            continue
                        mv_t = _interp_orig(remix_t, anchors)
                        if _act_at(rmx_act, remix_t):
                            conflicts += int(_ms_quality.mv_face_but_silent(
                                mouth_profile, mv_t))
                        else:
                            conflicts += int(_ms_quality.is_mv_singing(
                                mouth_profile, mv_t))
                    if conflicts >= 2:
                        report["unsafe_ranges"].append(
                            (float(block_start), float(block_start + 2.0)))
                    block_start += 2.0
            except Exception:
                pass

        unsafe_lo = max(0.0, float(rt[0]))
        unsafe_hi = float(rt[min(len(rt), len(rfeat)) - 1]) + max(feat_hop, 0.1)
        report["unsafe_ranges"] = _prepare_unsafe_ranges(
            report["unsafe_ranges"], unsafe_lo, unsafe_hi,
            pad=0.25, min_safe=0.75)
        content_ok = (
            (report["feature_similarity"] >= min_sim and report["feature_lift"] >= 0.015)
            or report["onset_correlation"] >= 0.12
            or (report["feature_similarity"] >= min_sim + 0.08
                and report["median_dtw_cost"] <= 0.70)
        )
        local_ok = (report["block_similarity_p20"] >= min_block
                    and report["longest_bad_seconds"] <= 6.0)
        strong_reordered = (report["block_similarity_p20"] >= min_block + 0.08
                            and report["feature_lift"] >= 0.05
                            and report["median_dtw_cost"] <= 0.80)
        jump_ok = report["jump_fraction"] <= 0.65 or strong_reordered
        # 末尾はレンダラーが映像を供給できることが必須。全体平均だけで
        # 隠さず、末尾30%と連続範囲外区間を hard gate にする。
        range_ok = (report["coverage"] >= 0.98
                    and report["tail_coverage"] >= 0.98
                    and report["longest_invalid_seconds"] <= 0.75
                    and report["ending_invalid_seconds"] <= 0.25)
        report["accepted"] = bool(
            range_ok
            and report["median_dtw_cost"] <= max_cost
            and report["p75_dtw_cost"] <= max_cost + 0.20
            and jump_ok
            and local_ok
            and content_ok
        )
    except Exception:
        pass
    return report


def _validate_rendered_output(path, expected_dur, tolerance=0.20):
    """出力ファイルの存在だけでなく、映像/音声ストリームと尺を検査。"""
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return False
    dur = ffprobe_dur(p)
    if dur <= 0 or abs(float(dur) - float(expected_dur)) > float(tolerance):
        return False
    expected_frames = int(round(float(expected_dur) * FPS))
    return (_video_has_exact_frames(p, expected_frames)
            and _has_av_streams(p))


def _publish_rendered_output(source_path, destination_path, expected_dur):
    """出力先と同じvolumeに完全コピーして検証後、atomicに公開する。"""
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.unlink(missing_ok=True)
    stage = destination_path.with_name(
        f".{destination_path.name}.djvm-{os.getpid()}.partial")
    stage.unlink(missing_ok=True)
    try:
        shutil.copy2(str(source_path), str(stage))
        if not _validate_rendered_output(stage, expected_dur):
            return False
        os.replace(str(stage), str(destination_path))
        return True
    except (OSError, shutil.Error):
        return False
    finally:
        stage.unlink(missing_ok=True)


def _equal_cut_times(anchors, music_dur, subseg=2.0):
    """固定間隔、映像ジャンプ、意味のある単語補正を統合したカット時刻。"""
    anchors = sorted(list(anchors), key=lambda a: a[0])
    music_dur = max(0.0, float(music_dur))
    grid = max(0.10, float(subseg))
    base_cuts = [0.0, music_dur]
    t = grid
    while t < music_dur - 1e-6:
        base_cuts.append(t); t += grid
    mandatory = []
    for i in range(1, len(anchors)):
        if _mapping_interval_is_jump(anchors[i - 1], anchors[i]):
            at = float(anchors[i][0])
            if 0.0 < at < music_dur:
                mandatory.append(at)
    base_cuts = sorted(set(round(x, 6) for x in base_cuts + mandatory))

    # 2秒窓内のWhisper単語アンカーを、窓両端の単一rateだけで
    # 描画すると、100〜200msの単語補正が消える。固定窓の線形予測から
    # 100ms以上外れるアンカーも境界候補にする。過分割防止のため
    # 最短間隔は250msとし、残差の大きい点から採用。
    precise = []
    for a in anchors:
        at = float(a[0])
        if at <= 0.0 or at >= music_dur or any(abs(at - x) < 1e-6 for x in base_cuts):
            continue
        j = bisect.bisect_right(base_cuts, at) - 1
        if j < 0 or j >= len(base_cuts) - 1:
            continue
        l, r = base_cuts[j], base_cuts[j + 1]
        if r - l <= 1e-6:
            continue
        ol = _interp_orig(l, anchors); or_ = _interp_orig(r, anchors)
        pred = ol + (at - l) / (r - l) * (or_ - ol)
        err = abs(float(a[1]) - pred)
        if err >= 0.10:
            precise.append((err, at))
    selected = list(base_cuts)
    for _err, at in sorted(precise, reverse=True):
        if min(abs(at - x) for x in selected) >= 0.25:
            selected.append(at)
    # 境界を1つ追加すると、その新しい区間内で別のアンカーの
    # 残差が顕在化する。最大残差点を逐次追加し、全区間が100ms未満
    # になるまで繰り返す（最短250msの制約は維持）。
    for _ in range(len(anchors)):
        selected.sort()
        best = None
        for a in anchors:
            at = float(a[0])
            if at <= 0.0 or at >= music_dur or min(abs(at - x) for x in selected) < 0.25:
                continue
            j = bisect.bisect_right(selected, at) - 1
            if j < 0 or j >= len(selected) - 1:
                continue
            l, r = selected[j], selected[j + 1]
            ol = _interp_orig(l, anchors); or_ = _interp_orig(r, anchors)
            pred = ol + (at - l) / max(1e-9, r - l) * (or_ - ol)
            err = abs(float(a[1]) - pred)
            if best is None or err > best[0]:
                best = (err, at)
        if best is None or best[0] < 0.10:
            break
        selected.append(best[1])
    return sorted(set(round(x, 6) for x in selected))


def _unsafe_overlap_seconds(start, duration, unsafe_ranges):
    """描画区間と危険範囲の重なり秒数（重複範囲は二重加算しない）。"""
    try:
        start = float(start); end = start + max(0.0, float(duration))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return float(sum(max(0.0, min(end, hi) - max(start, lo))
                     for lo, hi in _merge_time_ranges(
                         unsafe_ranges, lo=start, hi=end)))


def _segment_requires_safe_visual(start, duration, unsafe_ranges,
                                  min_overlap_sec=0.04):
    """危険範囲と実質的に重なるカットだけ人物非表示へ切り替える。"""
    try:
        duration = max(0.0, float(duration))
    except (TypeError, ValueError, OverflowError):
        return False
    threshold = min(max(0.001, float(min_overlap_sec)),
                    max(0.001, duration * 0.10))
    return _unsafe_overlap_seconds(start, duration, unsafe_ranges) + 1e-9 >= threshold


def _profile_interval_has_no_visible_mouth(profile, start, duration, mv_dur=None,
                                           safety_margin=0.20):
    """全source frameで ``mouth_absent=True`` の区間だけを安全認証。

    face=0、mouth_visible=False、小さい口、旧profile、間引き解析は
    「不明」であり安全ではない。いずれもFalseにして黒背景へ倒す。
    """
    if not profile or profile.get("_all_source_frames") is not True:
        return False
    try:
        times = np.asarray(profile.get("times", []), dtype=float)
        absent = np.asarray(profile.get("mouth_absent"), dtype=float)
        start = float(start); duration = float(duration)
        analyzed_fps = float(profile.get("fps", 0.0))
    except (TypeError, ValueError, OverflowError):
        return False
    if (times.ndim != 1 or absent.ndim != 1 or len(times) < 2
            or len(times) != len(absent) or duration <= 0.0
            or not np.isfinite(analyzed_fps) or analyzed_fps < 15.0
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0)):
        return False
    # 不明値はabsentではない。全顔の集約結果が1のframeだけ許可する。
    absent = np.where(np.isfinite(absent), absent >= 0.5, False)
    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0.0:
        return False
    margin = max(0.0, float(safety_margin))
    query_lo = max(0.0, start - margin)
    query_hi = start + duration + margin
    if mv_dur is not None:
        try:
            query_hi = min(query_hi, float(mv_dur))
        except (TypeError, ValueError, OverflowError):
            return False
    if query_hi <= query_lo:
        return False
    # fps=1000指定でstep=1（全source frame）になったprofileのみを想定。
    # 2.5 frame超の時刻空白があれば、その間は未認証として拒否する。
    max_gap = max(2.5 / analyzed_fps, 2.5 * dt)
    idx = np.flatnonzero((times >= query_lo - 1e-9)
                         & (times <= query_hi + 1e-9))
    if len(idx) == 0:
        return False
    sampled = times[idx]
    if (sampled[0] - query_lo > max_gap or query_hi - sampled[-1] > max_gap
            or (len(sampled) > 1 and np.max(np.diff(sampled)) > max_gap)):
        return False
    return bool(np.all(absent[idx]))


def _pick_verified_no_mouth_time(mouth_module, profile, want_dur, mv_dur,
                                 avoid=None):
    """厳格API候補を全source frame profileで独立再検証する。"""
    if mouth_module is None or profile is None:
        return None
    picker = getattr(mouth_module, "pick_no_mouth_mv_time", None)
    if not callable(picker):
        return None
    try:
        value = picker(profile, want_dur, mv_dur, avoid=avoid)
        if value is None:
            return None
        value = float(value)
    except Exception:
        return None
    if (not np.isfinite(value) or value < 0.0
            or not _profile_interval_has_no_visible_mouth(
                profile, value, want_dur, mv_dur=mv_dur)):
        return None
    return value


def _build_all_frame_mouth_profile(mouth_module, mv_path):
    """fps=1000（step=1）で全source frame用の安全profileを構築。"""
    if mouth_module is None or not hasattr(mouth_module, "build_mouth_profile"):
        return None
    try:
        profile = mouth_module.build_mouth_profile(
            str(mv_path), fps=1000.0, use_cache=False)
    except Exception:
        return None
    if not isinstance(profile, dict):
        return None
    profile = dict(profile)
    profile["_all_source_frames"] = True
    return profile


def _mapped_frames_have_verified_lipsync_visual(profile, remix_times,
                                                source_times, rmx_act):
    """全出力frameがclean-vocal発声中で、対応MV frameの口が
    明瞭かつ動いている場合だけTrue。検出失敗はFalse。
    """
    if profile is None or rmx_act is None:
        return False
    try:
        remix_times = np.asarray(remix_times, dtype=float).reshape(-1)
        source_times = np.asarray(source_times, dtype=float).reshape(-1)
        hop, raw_active = rmx_act
        hop = float(hop)
        raw_active = np.asarray(raw_active).reshape(-1)
        if (np.issubdtype(raw_active.dtype, np.number)
                and not np.all(np.isfinite(raw_active.astype(float)))):
            return False
        active = raw_active.astype(bool)
        if (len(remix_times) == 0 or len(remix_times) != len(source_times)
                or len(active) == 0 or not np.isfinite(hop) or hop <= 0.0
                or not np.all(np.isfinite(remix_times))):
            return False
        # RMS binは [i*hop,(i+1)*hop) の区間。roundで最寄り1点だけ
        # 見ると、例:t=.033sの出力frameが実際に跨ぐ0.00-0.05s
        # 無声binを見落とす。各出力frameが跨ぐ全binを必須にする。
        frame_span = 1.0 / float(FPS)
        vocal_active = np.zeros(len(remix_times), dtype=bool)
        for j, t in enumerate(remix_times):
            i0 = int(np.floor((float(t) + 1e-12) / hop))
            i1 = int(np.ceil((float(t) + frame_span - 1e-12) / hop))
            if i0 < 0 or i1 <= i0 or i1 > len(active):
                return False
            vocal_active[j] = bool(np.all(active[i0:i1]))
        import mouth_sync as _ms_verify
        verifier = getattr(
            _ms_verify, "mapped_frames_have_verified_lipsync_visual", None)
        if not callable(verifier):
            return False
        return bool(verifier(profile, source_times, vocal_active))
    except Exception:
        return False


def _rendered_frame_mapping(remix_start, duration, nframes,
                            source_start, source_duration, fps=FPS):
    """実際の出力frame時刻と、伸縮後に対応するMV時刻。"""
    try:
        remix_start = float(remix_start); duration = float(duration)
        source_start = float(source_start); source_duration = float(source_duration)
        nframes = int(nframes); fps = float(fps)
        if (nframes <= 0 or duration <= 0.0 or source_duration <= 0.0
                or fps <= 0.0 or not all(np.isfinite(v) for v in
                                         (remix_start, duration, source_start,
                                          source_duration, fps))):
            return np.array([], dtype=float), np.array([], dtype=float)
        first_frame = int(round(remix_start * fps))
        remix_times = (first_frame + np.arange(nframes, dtype=float)) / fps
        # ffmpegの入力先頭はグローバルremix frame gridの丸めと
        # 無関係に source_start。source側は必ずlocal frame 0から始める。
        source_local = np.arange(nframes, dtype=float) / fps
        source_local = np.clip(source_local, 0.0,
                               max(0.0, duration - 1e-9))
        source_times = source_start + source_local * (source_duration / duration)
        return remix_times, source_times
    except (TypeError, ValueError, OverflowError):
        return np.array([], dtype=float), np.array([], dtype=float)


def _equal_source_plan(o_pos, o_end, duration, mv_dur):
    """equal rendererが実際に読むMV開始、尺、進行率を確定。"""
    try:
        o_pos = float(o_pos); o_end = float(o_end)
        duration = float(duration); mv_dur = float(mv_dur)
    except (TypeError, ValueError, OverflowError):
        return 0.0, 0.0, None
    if not all(np.isfinite(v) for v in (o_pos, o_end, duration, mv_dur)):
        return 0.0, 0.0, None
    o_pos = max(0.0, o_pos); o_end = max(0.0, o_end)
    duration = max(0.0, duration); mv_dur = max(0.0, mv_dur)
    if duration <= 0.0:
        return o_pos, 0.0, None
    local_src = o_end - o_pos
    rate = None
    if local_src > 0.0:
        ratio = local_src / duration
        rate_lo, rate_hi = ((0.60, 1.60) if duration < 1.0 else (0.8, 1.25))
        if rate_lo <= ratio <= rate_hi:
            rate = ratio
    if rate is not None:
        src = rate * duration
        if o_pos + src > mv_dur - 0.02:
            src = max(0.05, mv_dur - 0.02 - o_pos)
        return o_pos, src, rate
    if o_pos + duration > mv_dur - 0.05:
        o_pos = max(0.0, mv_dur - duration - 0.05)
    return o_pos, duration, None


def _vocal_onset_envelope(voc, sr):
    """clean vocalの発音onset包絡。位相証明にだけ使う。"""
    try:
        import librosa
        y = np.asarray(voc, dtype=np.float32).reshape(-1)
        sr = int(sr)
        if len(y) < sr or sr <= 0:
            return None, None
        hop = 256
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        if len(env) < 4 or not np.all(np.isfinite(env)):
            return None, None
        peak = float(np.max(env))
        if peak <= 1e-9:
            return None, None
        env = np.asarray(env / peak, dtype=np.float32)
        times = np.arange(len(env), dtype=float) * hop / sr
        return times, env
    except Exception:
        return None, None


def _onset_peak_count(times, envelope, lo, hi):
    try:
        times = np.asarray(times, dtype=float)
        env = np.asarray(envelope, dtype=float)
        mask = (times >= float(lo)) & (times < float(hi))
        values = env[mask]
        if len(values) < 5 or not np.all(np.isfinite(values)):
            return 0
        threshold = max(0.08, float(np.percentile(values, 70)) * 0.65)
        peaks = ((values[1:-1] >= values[:-2])
                 & (values[1:-1] > values[2:])
                 & (values[1:-1] >= threshold))
        return int(np.sum(peaks))
    except Exception:
        return 0


def _mask_to_minimum_ranges(mask, fps=FPS, min_show=0.75):
    """True runをframe境界の証明済み範囲に変換。"""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    min_frames = max(1, int(np.ceil(float(min_show) * float(fps))))
    out = []; i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1; continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        if j - i >= min_frames:
            out.append((i / float(fps), j / float(fps)))
        i = j
    return out


def _visual_phase_proven_ranges_from_mapping(
        remix_times, source_times, all_frame_profile, rvoc, sr,
        window_sec=4.0, hop_sec=1.0, context_sec=1.0,
        max_lag=0.60, max_residual_lag=0.12, min_corr=0.32,
        min_face=0.45, min_prom=2.0, min_onset_peaks=3):
    """最終renderer写像で口opening flux×clean-vocal onset位相を証明。

    4秒contextに成功した窓の中央2秒だけを候補とし、
    そのframeを覆う全窓が成功した時だけwhitelistに入れる。
    """
    try:
        remix_times = np.asarray(remix_times, dtype=float).reshape(-1)
        source_times = np.asarray(source_times, dtype=float).reshape(-1)
        if (not isinstance(all_frame_profile, dict)
                or all_frame_profile.get("_all_source_frames") is not True
                or len(remix_times) == 0 or len(remix_times) != len(source_times)
                or not np.all(np.isfinite(remix_times))):
            return []
        onset_t, onset_env = _vocal_onset_envelope(rvoc, sr)
        if onset_t is None:
            return []
        import mouth_sync as _ms_phase
        measure = getattr(_ms_phase, "measure_micro_mouth_lag", None)
        if not callable(measure):
            return []
        n = len(remix_times)
        covered = np.zeros(n, dtype=np.int16)
        passed = np.zeros(n, dtype=np.int16)
        finite_map = np.isfinite(source_times)
        track_end = ((n / float(FPS))
                     if n else 0.0)
        last_start = track_end - float(window_sec)
        if last_start < -1e-9:
            return []
        starts = np.arange(0.0, last_start + 1e-9, float(hop_sec))
        for start in starts:
            end = start + float(window_sec)
            central = ((remix_times >= start + float(context_sec) - 1e-9)
                       & (remix_times < end - float(context_sec) - 1e-9))
            covered[central] += 1
            window = ((remix_times >= start - 1e-9)
                      & (remix_times < end - 1e-9) & finite_map)
            idx = np.flatnonzero(window)
            ok = False
            if (len(idx) >= int(0.75 * window_sec * float(FPS))
                    and _onset_peak_count(onset_t, onset_env, start, end)
                    >= int(min_onset_peaks)):
                lag, corr, fcov, prom = measure(
                    all_frame_profile, remix_times[idx], source_times[idx],
                    onset_t, onset_env, max_lag=max_lag, step=0.02,
                    min_face=min_face)
                ok = bool(
                    lag is not None and np.isfinite(lag)
                    and abs(float(lag)) <= float(max_residual_lag)
                    and float(corr) >= float(min_corr)
                    and float(fcov) >= float(min_face)
                    and float(prom) >= float(min_prom))
            if ok:
                passed[central] += 1
        proof = (covered > 0) & (passed == covered) & finite_map
        # fail境界の両側2frameも非表示。
        bad = ~proof
        expanded_bad = bad.copy()
        for shift in (1, 2):
            expanded_bad[shift:] |= bad[:-shift]
            expanded_bad[:-shift] |= bad[shift:]
        proof &= ~expanded_bad
        return _mask_to_minimum_ranges(proof, fps=FPS, min_show=0.75)
    except Exception:
        return []


def _visual_ranges_cover_segment(start, duration, proven_ranges):
    """カット全体が連続whitelist内にある時だけTrue。"""
    try:
        start = float(start); end = start + float(duration)
    except (TypeError, ValueError, OverflowError):
        return False
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return False
    cursor = start
    for lo, hi in _merge_time_ranges(proven_ranges, lo=start, hi=end):
        if lo > cursor + 1e-7:
            return False
        cursor = max(cursor, hi)
        if cursor >= end - 1e-7:
            return True
    return False


def _equal_timeline_mapping(anchors, cut_times, music_dur, mv_dur):
    """equal rendererと同じplanで全出力frameのMV対応時刻を作る。"""
    total = int(round(float(music_dur) * FPS))
    remix_times = np.arange(total, dtype=float) / float(FPS)
    source_times = np.full(total, np.nan, dtype=float)
    for a, b in zip(cut_times, cut_times[1:]):
        nframes = int(round(b * FPS)) - int(round(a * FPS))
        first = int(round(a * FPS))
        if nframes <= 0 or first < 0 or first + nframes > total:
            continue
        raw0 = _interp_orig(a, anchors); raw1 = _interp_orig(b, anchors)
        try:
            raw0 = float(raw0); raw1 = float(raw1)
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(raw0) or not np.isfinite(raw1):
            continue
        o0, src_dur, _rate = _equal_source_plan(raw0, raw1, b - a, mv_dur)
        _rt, st = _rendered_frame_mapping(
            a, b - a, nframes, o0, src_dur)
        if len(st) == nframes:
            source_times[first:first + nframes] = st
    return remix_times, source_times


def _safe_visual_plan(requires_safe_visual, certified_mv_time=None):
    """危険時に人物MVへ戻る経路を作らない純粋な描画元選択。"""
    if not requires_safe_visual:
        return "aligned_mv", None
    if certified_mv_time is not None:
        try:
            value = float(certified_mv_time)
            if np.isfinite(value) and value >= 0.0:
                return "no_mouth_mv", value
        except (TypeError, ValueError, OverflowError):
            pass
    return "safe_background", None


_SAFE_BG_PALETTE = "c0=0x0b1e4a:c1=0x3a0ca3:c2=0x7209b7"   # 深青→紫→マゼンタ(クラブ調)
_safe_bg_gradient_ok = None


def _gradient_background_supported():
    """このffmpegで gradients ソースが使えるか(初回だけ実測して以後キャッシュ)。"""
    global _safe_bg_gradient_ok
    if _safe_bg_gradient_ok is None:
        try:
            r = subprocess.run(
                ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                 f"gradients=s=64x36:n=3:{_SAFE_BG_PALETTE}:r=30",
                 "-frames:v", "1", "-f", "null", "-"],
                capture_output=True, text=True, errors="replace", timeout=20)
            _safe_bg_gradient_ok = (r.returncode == 0)
        except Exception:
            _safe_bg_gradient_ok = False
    return _safe_bg_gradient_ok


def _safe_background_ffmpeg_command(output_path, nframes, width=OUT_W,
                                    height=OUT_H, fps=FPS):
    """口元が存在しない安全背景を指定フレーム数ぴったり作る。
    真っ黒だと“壊れた動画”に見えるため、ゆっくり動く抽象グラデーション
    (人物・顔・口に見える図形なし)を既定にする。gradients非対応のffmpegでは
    従来の黒に自動退避する(枚数保証はどちらも同じ)。"""
    nframes = int(nframes); width = int(width); height = int(height); fps = float(fps)
    if nframes <= 0:
        raise ValueError("nframes must be positive")
    if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("invalid background video geometry")
    if _gradient_background_supported():
        src = (f"gradients=s={width}x{height}:n=3:{_SAFE_BG_PALETTE}"
               f":speed=0.04:r={fps:g}")
        vf = "hue=H=0.12*t,vignette=PI/6,noise=alls=3:allf=t,format=yuv420p,setsar=1"
    else:
        src = f"color=c=black:s={width}x{height}:r={fps:g}"
        vf = "format=yuv420p,setsar=1"
    return [
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src, "-an",
        "-vf", vf, "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-frames:v", str(nframes),
        str(output_path),
    ]


def equal_and_mux(anchors, mv_path, music_path, music_dur, mv_dur, out_path, tmp_dir,
                  subseg=2.0, rmx_act=None, mouth_profile=None,
                  safe_mouth_profile=None, unsafe_ranges=None,
                  rvoc=None, vocal_sr=SR, visual_phase_proof_ranges=None):
    """See You Again式＝基本は等速配置。Proの賢いアンカーで“位置”を決める。
    滑らかに前進してる区間は、ローカルな進行速度に合わせて軽く伸縮（±25%まで）し、
    テンポ差による区間内の口ズレを打ち消す。ジャンプ箇所は等速のカット。
    低信頼区間は全source frameで口不在を認証した素材だけを使い、
    認証不能なら黒背景へfail-closedする。高信頼区間も、
    全出力frameで「発声中の明瞭に動く口」を証明できた場合だけ表示。"""
    tmp_dir = Path(tmp_dir)
    anchors = sorted([a for a in anchors], key=lambda a: a[0])
    if not anchors:
        print("  ❌ アンカーが空"); return False
    seg_files = []
    listf = tmp_dir / "concat.txt"
    idx = 0
    scale_pad = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                 f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}")
    # --- 口パク差し替え（remix無声 & MV口パク中 → 口が動いてないMV箇所へ）---
    _ms = None
    if mouth_profile is not None or safe_mouth_profile is not None:
        try:
            import mouth_sync as _ms
        except Exception:
            _ms = None
    _swap_avoid = []          # 直前に差し替えた先（チカチカ回避）
    _n_swap = 0
    _n_safe_background = 0
    # 通常の固定間隔に加え、アンカーが別のMV位置へ飛ぶ時刻を
    # 必ずカット境界にする。これが無いと、例えば53.7sのサビ戻りを
    # 52-54sの旧映像で流し、最大2秒遅れて切り替えていた。
    cut_times = _equal_cut_times(anchors, music_dur, subseg)
    expected_total_frames = int(round(float(music_dur) * FPS))
    planned_frames = 0
    if visual_phase_proof_ranges is None:
        phase_rt, phase_st = _equal_timeline_mapping(
            anchors, cut_times, music_dur, mv_dur)
        visual_phase_proof_ranges = _visual_phase_proven_ranges_from_mapping(
            phase_rt, phase_st, safe_mouth_profile, rvoc, vocal_sr)
    else:
        visual_phase_proof_ranges = _merge_time_ranges(
            visual_phase_proof_ranges, lo=0.0, hi=music_dur)
    phase_seconds = sum(max(0.0, b - a)
                        for a, b in visual_phase_proof_ranges)
    if phase_seconds > 0.0:
        print(f"     🛡️ 口×発音位相証明: {phase_seconds:.1f}秒を表示候補")
    else:
        print("     🛡️ 口×発音位相の証明なし → 人物は安全背景")

    for ci in range(len(cut_times) - 1):
        r = cut_times[ci]
        dur = cut_times[ci + 1] - r
        # フレーム数を「累積の丸め差」で決める＝各カットの端数が積み上がらない。
        # -t 秒指定だけだと ffmpeg が1フレーム多く出すことがあり、
        # 連結後の尺が音源より数十msはみ出していた（DJ用途では尺は一致必須）。
        nframes = int(round(cut_times[ci + 1] * FPS)) - int(round(r * FPS))
        if nframes <= 0:
            continue
        planned_frames += nframes
        raw_o_pos = _interp_orig(r, anchors)
        raw_o_end = _interp_orig(r + dur, anchors)
        try:
            raw_o_pos = float(raw_o_pos); raw_o_end = float(raw_o_end)
            mapping_finite = bool(
                np.isfinite(raw_o_pos) and np.isfinite(raw_o_end))
        except (TypeError, ValueError, OverflowError):
            raw_o_pos = raw_o_end = 0.0
            mapping_finite = False
        o_pos = max(0.0, raw_o_pos) if mapping_finite else 0.0
        mapped_o_end = max(0.0, raw_o_end) if mapping_finite else 0.0
        aligned_o_pos, aligned_src_dur, aligned_rate = _equal_source_plan(
            o_pos, mapped_o_end, dur, mv_dur)
        remix_frame_times, source_frame_times = _rendered_frame_mapping(
            r, dur, nframes, aligned_o_pos, aligned_src_dur)
        # --- 低信頼/口パク矛盾は、口不存在を認証した映像だけへ逃がす ---
        #   ①remix無声 × MV口パク中            → 歌ってないのに口パク映像 → 逃がす
        #   ②remix歌ってる × 顔あり×口が止まってる → リップシンク破綻の確証 → 逃がす
        #   ③音響品質レポートの低信頼区間 → 解析不能でも必ず人物を隠す
        _swapped = False
        _safe_background = False
        need_swap = (not mapping_finite
                     or not _visual_ranges_cover_segment(
                         r, dur, visual_phase_proof_ranges)
                     or _segment_requires_safe_visual(r, dur, unsafe_ranges))
        if _ms is not None and mouth_profile is not None and rmx_act is not None:
            # Remixは区間中央、MVは区間先頭という異なる時刻を比べて
            # いたため、2秒カットで約1秒分の誤判定があった。対応する
            # 3時点をペアで比べ、多数決でのみ差し替える。
            r_samples = [r + dur * f for f in (0.25, 0.50, 0.75)]
            mv_samples = [max(0.0, _interp_orig(x, anchors)) for x in r_samples]
            silent_conflicts = 0; singing_conflicts = 0
            for rr_t, mv_t in zip(r_samples, mv_samples):
                if _act_at(rmx_act, rr_t):
                    singing_conflicts += int(_ms.mv_face_but_silent(mouth_profile, mv_t))
                else:
                    silent_conflicts += int(_ms.is_mv_singing(mouth_profile, mv_t))
            need_swap = need_swap or max(silent_conflicts, singing_conflicts) >= 2
        # 音響ゲートが高信頼でも、実際に表示する全frameの
        # 口運動が認証できなければ人物を表示しない。
        need_swap = need_swap or not _mapped_frames_have_verified_lipsync_visual(
            safe_mouth_profile, remix_frame_times, source_frame_times, rmx_act)
        if need_swap:
            alt = None
            if ALLOW_REAL_MV_SAFE_BROLL:
                alt = _pick_verified_no_mouth_time(
                    _ms, safe_mouth_profile, dur, mv_dur, avoid=_swap_avoid)
            visual_plan, safe_mv_time = _safe_visual_plan(True, alt)
            if visual_plan == "no_mouth_mv":
                o_pos = safe_mv_time
                _swap_avoid.append(safe_mv_time)
                if len(_swap_avoid) > 8:
                    _swap_avoid.pop(0)
                _n_swap += 1
                _swapped = True
            else:
                _safe_background = True
                _n_safe_background += 1

        seg = tmp_dir / f"seg_{idx:04d}.mp4"
        idx += 1
        seg.unlink(missing_ok=True)
        if _safe_background:
            rr = run(_safe_background_ffmpeg_command(seg, nframes))
            if (getattr(rr, "returncode", 1) == 0
                    and _video_has_exact_frames(seg, nframes)):
                seg_files.append(seg)
            else:
                print(f"     ⚠️ 安全背景セグメント失敗 r={r:.1f}: "
                      f"{(rr.stderr or '')[:120]}")
                # 区間を省略して続行すると、後続の人物映像が音声より前詰め
                # される。安全背景すら作れない出力は全体を不採用にする。
                return False
            continue
        # 差し替えた区間は等速。通常区間は上で安全確認した
        # まさに同じsource範囲を読む。
        if _swapped:
            o_pos, src, rate = _equal_source_plan(
                o_pos, o_pos + dur, dur, mv_dur)
        else:
            o_pos, src, rate = aligned_o_pos, aligned_src_dur, aligned_rate
        if rate is not None:
            # MVの src 秒を dur 秒へ伸縮（setpts）。区間内ドリフトを打ち消す
            factor = dur / max(0.05, src)
            vf = _exact_frame_filter(
                f"setpts={factor:.5f}*PTS,{scale_pad}", nframes)
            rr = run(["ffmpeg", "-v", "error", "-y", "-ss", f"{o_pos:.3f}", "-t", f"{src:.3f}",
                      "-i", str(mv_path), "-an", "-vf", vf,
                      "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                      "-frames:v", str(nframes), str(seg)])
        else:
            # ジャンプ／範囲外：等速のカット
            vf = _exact_frame_filter(scale_pad, nframes)
            rr = run(["ffmpeg", "-v", "error", "-y", "-ss", f"{o_pos:.3f}", "-t", f"{dur:.3f}",
                      "-i", str(mv_path), "-an", "-vf", vf,
                      "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                      "-frames:v", str(nframes), str(seg)])
        if (getattr(rr, "returncode", 1) == 0
                and _video_has_exact_frames(seg, nframes)):
            seg_files.append(seg)
        else:
            print(f"     ⚠️ セグメント失敗 r={r:.1f}: {(rr.stderr or '')[:120]}")
            # 0.2秒未満でも欠落を許可しない。同期境界を保つため全体不採用。
            return False
    if not seg_files:
        print("  ❌ 有効なセグメントが無い"); return False
    if planned_frames != expected_total_frames:
        print(f"  ❌ 映像フレーム予算が不一致: {planned_frames}/{expected_total_frames}")
        return False
    if _n_swap > 0 or _n_safe_background > 0:
        print(f"     👄 口元非表示：全frame認証MV {_n_swap}区間 / "
              f"安全背景 {_n_safe_background}区間")
    with open(listf, "w") as f:
        for s in seg_files:
            f.write(f"file '{s.as_posix()}'\n")
    silent = tmp_dir / "video_silent.mp4"
    if not _concat_segments_exact(listf, silent, expected_total_frames):
        print("  ❌ セグメント結合後のフレーム数が不一致")
        return False
    return _mux_exact_video_audio(
        silent, music_path, out_path, expected_total_frames)


def warp_and_mux(anchors, mv_path, music_path, music_dur, mv_dur, out_path, tmp_dir,
                 rmx_act=None, safe_mouth_profile=None, unsafe_ranges=None,
                 visual_phase_proof_ranges=None):
    """warp配置もequalと同じ全frame視覚証明を必須にする。"""
    tmp_dir = Path(tmp_dir)
    # アンカーを時刻順に整理、両端を補完
    anchors = sorted([a for a in anchors], key=lambda a: a[0])
    if not anchors:
        print("  ❌ アンカーが空。warp不可")
        return False
    if anchors[0][0] > 1e-6:
        anchors.insert(0, [0.0, max(0.0, anchors[0][1] - anchors[0][0]), 1.0])
    if anchors[-1][0] < music_dur - 1e-6:
        anchors.append([music_dur, min(mv_dur, anchors[-1][1] +
                                       (music_dur - anchors[-1][0])), 1.0])

    seg_files = []
    listf = tmp_dir / "concat.txt"
    expected_total_frames = int(round(float(music_dur) * FPS))
    planned_frames = 0
    safe_background_count = 0
    for k in range(len(anchors) - 1):
        try:
            r0, o0, _ = anchors[k]
            r1, o1, _ = anchors[k + 1]
            r0 = float(r0); r1 = float(r1)
            raw_o0 = float(o0); raw_o1 = float(o1)
        except (TypeError, ValueError, OverflowError):
            return False
        if not np.isfinite(r0) or not np.isfinite(r1):
            return False
        source_mapping_finite = bool(
            np.isfinite(raw_o0) and np.isfinite(raw_o1))
        o0 = raw_o0 if source_mapping_finite else 0.0
        o1 = raw_o1 if source_mapping_finite else 0.0
        dur = r1 - r0
        nframes = int(round(r1 * FPS)) - int(round(r0 * FPS))
        if nframes <= 0:
            continue
        planned_frames += nframes
        # 並べ替えで後退/異常なら等速で前進
        src_dur = o1 - o0
        if src_dur <= 0.05 or src_dur > dur * 4 or src_dur < dur * 0.25:
            src_dur = dur
            o1 = min(mv_dur, o0 + dur)
        src_dur = max(0.05, min(src_dur, mv_dur - o0))
        if src_dur <= 0.05:
            o0 = max(0.0, mv_dur - dur); src_dur = min(dur, mv_dur - o0)
        pts = dur / src_dur  # >1で遅く（伸ばす）, <1で速く
        seg = tmp_dir / f"seg_{k:04d}.mp4"
        seg.unlink(missing_ok=True)
        remix_frame_times, source_frame_times = _rendered_frame_mapping(
            r0, dur, nframes, o0, src_dur)
        need_safe = (not source_mapping_finite
                     or not _visual_ranges_cover_segment(
                         r0, dur, visual_phase_proof_ranges)
                     or _segment_requires_safe_visual(r0, dur, unsafe_ranges))
        need_safe = need_safe or not _mapped_frames_have_verified_lipsync_visual(
            safe_mouth_profile, remix_frame_times, source_frame_times, rmx_act)
        if need_safe:
            safe_background_count += 1
            r = run(_safe_background_ffmpeg_command(seg, nframes))
        else:
            vf = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                  f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                  f"setpts={pts:.6f}*PTS,fps={FPS},"
                  f"tpad=stop_mode=clone:stop_duration=1,"
                  f"trim=end_frame={nframes},setpts=PTS-STARTPTS")
            r = run(["ffmpeg", "-v", "error", "-y", "-ss", f"{o0:.3f}",
                     "-t", f"{src_dur:.3f}", "-i", str(mv_path), "-an",
                     "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
                     "-crf", "20", "-frames:v", str(nframes), str(seg)])
        if (getattr(r, "returncode", 1) == 0
                and _video_has_exact_frames(seg, nframes)):
            seg_files.append(seg)
        else:
            print(f"     ⚠️ セグメント失敗 k={k}: {(r.stderr or '')[:160]}")
            return False

    if not seg_files:
        print("  ❌ 有効なセグメントが無い")
        return False
    if planned_frames != expected_total_frames:
        print(f"  ❌ 映像フレーム予算が不一致: {planned_frames}/{expected_total_frames}")
        return False
    if safe_background_count:
        print(f"     🛡️ warp口元非表示: 安全背景 "
              f"{safe_background_count}区間")
    with open(listf, "w") as f:
        for s in seg_files:
            f.write(f"file '{s.as_posix()}'\n")

    silent = tmp_dir / "video_silent.mp4"
    if not _concat_segments_exact(listf, silent, expected_total_frames):
        print("  ❌ warp結合後のフレーム数が不一致")
        return False
    return _mux_exact_video_audio(
        silent, music_path, out_path, expected_total_frames)


# ============================================================
#  メイン
# ============================================================
def cache_feat(tag, wav, sr, use_hubert):
    """特徴量を音源ハッシュでキャッシュ（再走を高速化）。

    「HuBERTの読み込みに成功したか」ではなく、実際に抽出できた
    特徴方式の名前で保存する。以前はHuBERT推論が途中で失敗し
    MFCCへ落ちても hubert-L9 としてキャッシュされ、次回に
    40次元と768次元が混在することがあった。"""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        wav = np.asarray(wav, dtype=np.float32)
        h = hashlib.md5(wav.tobytes()).hexdigest()[:16]
        desired = (f"hubert-L{HUBERT_LAYER}" if (use_hubert and _try_load_hubert()) else "mfcc")
        fp = CACHE_DIR / f"{h}_feat-v{FEATURE_CACHE_SCHEMA}_{desired}.npz"
        if fp.exists():
            d = np.load(fp, allow_pickle=False)
            feat = d["feat"]; times = d["times"]
            saved_kind = str(np.asarray(d["kind"]).item()) if "kind" in d else ""
            schema = int(np.asarray(d["schema"]).item()) if "schema" in d else 0
            if (schema == FEATURE_CACHE_SCHEMA and saved_kind == desired
                    and feat.ndim == 2 and len(feat) == len(times) and feat.shape[1] > 0):
                print(f"     ♻️ 特徴量[{tag}]キャッシュ再利用（{saved_kind}）")
                return feat, times, saved_kind
        feat, times, name = extract_features(wav, sr, tag, use_hubert)
        actual = f"hubert-L{HUBERT_LAYER}" if name == "hubert" else str(name)
        actual_fp = CACHE_DIR / f"{h}_feat-v{FEATURE_CACHE_SCHEMA}_{actual}.npz"
        np.savez(actual_fp, feat=feat, times=times, kind=np.asarray(actual),
                 schema=np.asarray(FEATURE_CACHE_SCHEMA, dtype=np.int16))
        return feat, times, actual
    except Exception:
        return extract_features(wav, sr, tag, use_hubert)


def process(music_path, mv_source, out_path, use_hubert=True, placement="equal",
            sync_offset_ms=0):
    final_out = Path(out_path)
    final_out.parent.mkdir(parents=True, exist_ok=True)
    # 失敗時に以前の人物映像を今回の完成品と誤認しない。
    final_out.unlink(missing_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="lipsync_pro_"))
    try:
        # --- MV取得（URL or ローカル）---
        if str(mv_source).startswith("http"):
            print("🎬 MVをダウンロード中...")
            mv = download_youtube(mv_source, tmp / "dl")
            if mv is None:
                print("  ❌ ダウンロード失敗"); return False
            if is_static_video(mv):
                print("  ⚠️ このMVは静止画（ジャケ/音声のみ）です。動くMVのURLか動画ファイルを指定してください。")
        else:
            mv = Path(mv_source)
            if not mv.exists():
                print("  ❌ 動画ファイルが見つかりません:", mv); return False
        mv_dur = ffprobe_dur(mv)
        music_dur = ffprobe_dur(music_path)
        print(f"  音楽: {music_dur:.1f}秒 / MV: {mv_dur:.1f}秒")

        # --- Demucs ボーカル分離（再利用・キャッシュ付き）---
        print("🎤 ボーカル分離中（remix / MV）...")
        mv_wav = extract_audio_wav(mv, tmp / "mv_audio.wav")
        rvoc, m1 = separate_vocals(music_path, tmp / "sep_remix")
        ovoc, m2 = separate_vocals(mv_wav, tmp / "sep_mv")
        print(f"     分離方式: remix={m1} / MV={m2}")
        if (len(rvoc) < SR or len(ovoc) < SR
                or float(np.sqrt(np.mean(np.asarray(rvoc, dtype=np.float64) ** 2))) < 1e-6
                or float(np.sqrt(np.mean(np.asarray(ovoc, dtype=np.float64) ** 2))) < 1e-6):
            print("     ⚠️ 有効なボーカルstemが取れないためPro同期をスキップ")
            return False
        decoded_music_dur = len(rvoc) / float(SR)
        if decoded_music_dur > 1.0 and abs(decoded_music_dur - music_dur) > 0.20:
            print(f"     ℹ️ VBR尺補正: ffprobe {music_dur:.2f}s → デコード実測 {decoded_music_dur:.2f}s")
            music_dur = decoded_music_dur
        rmx_act = _voc_envelope(rvoc, SR)

        # --- 特徴量 ---
        print("🧠 特徴量を抽出中...")
        rfeat, rt, fn = cache_feat("remix", rvoc, SR, use_hubert)
        ofeat, ot, ofn = cache_feat("MV", ovoc, SR, use_hubert)
        if rfeat.ndim != 2 or ofeat.ndim != 2 or rfeat.shape[1] != ofeat.shape[1]:
            # 一方のHuBERTだけが失敗した時は、特徴空間を混ぜない。
            # 両方をMFCCに揃えれば従来精度で安全に続行できる。
            print(f"     ⚠️ 特徴次元が不一致（remix={rfeat.shape} / MV={ofeat.shape}）"
                  " → 両方をMFCCに統一")
            rfeat, rt, fn = cache_feat("remix", rvoc, SR, False)
            ofeat, ot, ofn = cache_feat("MV", ovoc, SR, False)

        # --- subsequence DTW（multi-scale：粗10s＋細5sの2スケールで候補抽出）---
        print("🔬 subsequence DTW（multi-scale 粗10s＋細5s で候補抽出）...")
        win_fine = windowed_topk(rfeat, rt, ofeat, ot, win_sec=WIN_SEC, hop_sec=HOP_SEC)
        win_coarse = windowed_topk(rfeat, rt, ofeat, ot, win_sec=WIN_SEC * 2, hop_sec=WIN_SEC)
        windows_raw = sorted(win_fine + win_coarse, key=lambda w: w[0])
        windows = merge_windows_by_time(windows_raw)
        print(f"     窓数: 細{len(win_fine)} + 粗{len(win_coarse)} = {len(windows)}（重複時刻を統合）")
        voiced_windows = [w for w in windows
                          if _active_fraction(rmx_act, w[0], w[0] + w[1]) >= 0.15]
        if len(voiced_windows) >= 3:
            removed = len(windows) - len(voiced_windows)
            windows = voiced_windows
            if removed:
                print(f"     🎤 ボーカルが薄い{removed}窓をDTW経路から除外（ドロップ誤マッチ防止）")

        # --- 全体経路推定（Viterbi＝等速基準で時間順を保つ。図の通り）---
        print("🧭 全体経路推定（Viterbi＝等速を基準に順番を保つ）...")
        anchors = choose_viterbi_path(windows, rfeat, rt, ofeat, ot,
                                      rvoc=rvoc, ovoc=ovoc, sr=SR,
                                      rmx_act=rmx_act, feature_kind=fn)
        print(f"     アンカー数: {len(anchors)}")
        if len(anchors) < 2:
            print("     ⚠️ ボーカル対応アンカーが不足 → 旧方式へ")
            return False

        # --- 局所DTW微調整 ---
        print("🔧 局所DTWで微調整...")
        anchors = local_refine(anchors, rfeat, rt, ofeat, ot)

        # --- 子音オンセット合わせ ---
        print("🗣 子音オンセットで微合わせ...")
        anchors = consonant_align(anchors, rvoc, ovoc, SR)

        # --- 単発の飛びを均す（一瞬だけ別シーンに飛んで戻るのを抑制）---
        anchors = despike_anchors(anchors)

        # --- 連続ラン整形（音内容アライン式：本筋から外れた blip を引き戻す。並べ替えは保持）---
        anchors = slope_run_cleanup(anchors)

        # --- 中盤アンカーを単調＆滑らかに矯正（後戻り・テンポ暴れを抑える）---
        anchors = smooth_anchors(anchors, mv_dur)

        # --- Forced Alignment初期ピン（広い±2.5秒探索。後段補正後に狭域で再固定）---
        print("🗣 Forced Alignment（Whisper単語タイムスタンプで初期単語ピン）...")
        anchors = whisper_word_align(anchors, rvoc, ovoc, SR)
        # 単語スナップは精密だが、反復サビの同じ短語や認識揺れで
        # 局所的な逆走/飛びが再混入し得る。Whisper前だけでなく、後にも
        # 同じ区分的整形をかけ、本物のセクション並べ替えは保ったままノイズを落とす。
        anchors = despike_anchors(anchors)
        anchors = slope_run_cleanup(anchors)
        anchors = smooth_anchors(anchors, mv_dur)

        # --- 3分割（MVが足りない曲：イントロ/アウトロをラフにして中盤にMV優先配分）---
        bpm = 125.0
        try:
            import librosa
            ymus, _sr = librosa.load(str(music_path), sr=22050, mono=True)
            _tempo = librosa.beat.beat_track(y=ymus, sr=22050)[0]
            bpm = float(np.atleast_1d(_tempo)[0]) or 125.0
        except Exception:
            bpm = 125.0
        anchors, split = intro_outro_rough(anchors, music_dur, mv_dur, bpm, rmx_act)
        if split:
            ie, os_, ib, ob = split
            print(f"🎬 MV不足→3分割: イントロ0〜{ie:.0f}s({ib}小節) / 中盤(歌→精密) / "
                  f"アウトロ{os_:.0f}s〜末(≈{ob:.0f}小節)（BPM{bpm:.0f}・前後はMV自前を使用）")

        # ログ: 何秒→MV何秒
        for (r0, o0, c) in anchors[:: max(1, len(anchors) // 12)]:
            print(f"        remix {r0:6.1f}s → MV {o0:6.1f}s")

        # --- 映像warp + mux ---
        print("🎞 映像をwarpして remix音にmux...")
        # --- グローバルのズレ補正（nudge）---
        # --- 残ズレ補正（波形オンセットで合わせる）---
        manual_offset_ms = 0
        if sync_offset_ms == "auto":
            anchors, seg_offs = apply_segment_offsets(anchors, rvoc, ovoc, SR,
                                                      rfeat=rfeat, rt=rt, ofeat=ofeat, ot=ot)
            if len(seg_offs) <= 1:
                v = seg_offs[0] if seg_offs else 0
                print(f"     🎯 自動ズレ推定（波形オンセット・1区間）: {v:+d}ms")
            else:
                print(f"     🎯 自動ズレ推定（波形オンセット・{len(seg_offs)}区間を個別補正）: "
                      + " / ".join(f"{m:+d}ms" for m in seg_offs))
        elif sync_offset_ms:
            # 手動指定はユーザーが意図した最終nudge。
            # Forced Alignmentの再固定で消さないよう、この後に適用する。
            manual_offset_ms = int(sync_offset_ms)

        # --- 歌詞文脈整列（同じ短語ではなく前後の並びで、別サビを識別）---
        # 単語近傍スナップでは数十秒離れた反復へ到達できないため、
        # 後段の音響補正後・口の動き補正前に、一意な文脈だけで区間を修正する。
        try:
            lyric_rw = _whisper_words(rvoc, SR, "remix")
            lyric_ow = _whisper_words(ovoc, SR, "MV")
            if lyric_rw and lyric_ow:
                print("🎼 歌詞文脈整列（後半の繰り返しサビを前後の単語列で判定）...")
                anchors, n_lyric = apply_lyrics_align(
                    anchors, lyric_rw, lyric_ow,
                    music_dur=music_dur, mv_dur=mv_dur)
                if n_lyric == 0:
                    print("     🎼 歌詞文脈補正：大きな位置修正なし（現在位置支持/曖昧）")
        except Exception as lyric_err:
            print(f"     ⚠️ 歌詞文脈整列スキップ: {str(lyric_err)[:80]}")

        # --- 口パク差し替え用プロファイル（mouth_sync があれば。無ければNoneで従来通り）---
        mouth_profile = None
        safe_mouth_profile = None
        try:
            import mouth_sync as _msmod
            print("👄 口の動きを解析中（mouth_sync：歌ってない区間の口パク映像を回避）...")
            mouth_profile = _msmod.build_mouth_profile(str(mv), fps=10.0)
            if mouth_profile is not None:
                fr = mouth_profile.get("face_rate", 0.0)
                print(f"     👄 口プロファイル取得：顔検出率{fr*100:.0f}%（API:{mouth_profile.get('backend')}）"
                      + ("" if fr >= 0.25 else " ※低めのため効果は限定的"))
                # 口×歌声で『数秒ズレた繰り返しサビ』を測って引き戻す（強いゲート付き）
                anchors, n_mlag = apply_mouth_lag(anchors, mouth_profile, rvoc, SR,
                                                  music_dur=music_dur)
                if n_mlag == 0:
                    print("     👄 口×歌声ズレ補正：該当区間なし（顔不足/相関弱め/ズレ小）")
                # 仕上げ：口の開きフラックス×声オンセットで100ms級の残ズレを全区間微調整。
                # この直後のForced Alignment最終再固定が、単語一致の取れる区間を
                # さらに精密化する（取れない区間ではこの補正が生きる＝補完関係）。
                anchors, n_micro = apply_mouth_micro_lag(anchors, mouth_profile, rvoc, SR)
                if n_micro:
                    print(f"     👄 口マイクロ補正：{n_micro}区間の残ズレ(<{MOUTH_MICRO_MAX:.1f}s)を仕上げ")
                else:
                    print("     👄 口マイクロ補正：補正不要または確信不足（そのまま）")
            # B-rollの安全認証は10fps推測ではなく全source frameを解析する。
            # fps=1000指定はmouth_sync内部のstepを1にする。失敗、旧API、
            # mouth_absentキー欠落はレンダラー側で必ず黒へ倒す。
            print("🛡️ 口元非表示用に全フレームを安全確認中...")
            safe_mouth_profile = _build_all_frame_mouth_profile(_msmod, mv)
            if safe_mouth_profile is not None:
                if "mouth_absent" not in safe_mouth_profile:
                    print("     ⚠️ 明示的mouth_absent判定なし → 低信頼区間は安全背景")
        except Exception as _e:
            # 同期補正用10fps profileは取得済みなら残す。安全profileは
            # 失敗時にNoneのまま＝低信頼区間では必ず黒。
            safe_mouth_profile = None

        # 自動ズレ/口の相関補正の後に、単語境界を最終再固定する。
        # 探索は±0.55秒に限定し、大きな粗補正や口×歌声補正を
        # 逆に巻き戻さず、後段の微調整で崩れた近傍ピンだけを戻す。
        print("🗣 Forced Alignment最終確定（後段補正後の単語境界を再固定）...")
        anchors = whisper_word_align(anchors, rvoc, ovoc, SR, search=0.55)

        if manual_offset_ms:
            off = manual_offset_ms / 1000.0
            anchors = [[r, max(0.0, o + off), c] for (r, o, c) in anchors]
            print(f"     🎯 ズレ補正 {manual_offset_ms:+d}ms を最終的に全体適用")

        # --- 最終採用ゲート：「全候補が悪い中の相対ベスト」を成功扱いしない ---
        q = alignment_quality_report(
            anchors, windows, rfeat, rt, ofeat, ot, rvoc=rvoc, ovoc=ovoc, sr=SR,
            rmx_act=rmx_act, feature_kind=fn, mouth_profile=mouth_profile)
        print("     📊 Pro同期品質: "
              f"カバー{q['coverage']*100:.0f}% (末尾{q['tail_coverage']*100:.0f}%/"
              f"連続範囲外{q['longest_invalid_seconds']:.1f}s) / "
              f"発音類似{q['feature_similarity']:.2f} "
              f"(局所p20={q['block_similarity_p20']:.2f}/上積み{q['feature_lift']:+.2f}) / "
              f"onset{q['onset_correlation']:.2f} / DTW{q['median_dtw_cost']:.2f} "
              f"(p75={q['p75_dtw_cost']:.2f}) / ジャンプ{q['jump_fraction']*100:.0f}%")
        if q.get("unsafe_ranges"):
            unsafe_seconds = sum(max(0.0, b - a) for a, b in q["unsafe_ranges"])
            print(f"     🛡️ 口元非表示対象: {len(q['unsafe_ranges'])}区間 / "
                  f"計{unsafe_seconds:.1f}s（認証不能時は安全背景）")
        if not q["accepted"]:
            print("     ⚠️ Pro同期の絶対信頼度が不足 → この映像を採用せず旧方式へ")
            return False

        # --- 配置（既定=等速 See You Again式／warpは任意）---
        render_out = tmp / "pro_rendered.mp4"
        # warpはアンカー間全体を別の速度で変形し、位相証明時の
        # 実表示写像とずれる余地がある。絶対安全モードでは最終
        # rendererと同じpiecewise写像を使えるequalに統一する。
        if placement == "warp":
            print("     🛡️ warp指定も全frame位相証明の可能な安全equal配置へ切替")
        print("🎞 等速配置（単語アンカーに追従）で remix音にmux...")
        ok = equal_and_mux(
            anchors, mv, music_path, music_dur, mv_dur, render_out, tmp,
            rmx_act=rmx_act, mouth_profile=mouth_profile,
            safe_mouth_profile=safe_mouth_profile,
            unsafe_ranges=q.get("unsafe_ranges", []), rvoc=rvoc,
            vocal_sr=SR)
        if ok and _validate_rendered_output(render_out, music_dur):
            if _publish_rendered_output(render_out, final_out, music_dur):
                if "_pro_context" in Path(out_path).name:
                    # ハイブリッド局所Proの作業用クリップ。Web UIはログの
                    # 「✅ 完成: *.mp4」を完成品としてダウンロード一覧に載せるため、
                    # 中間ファイルはその書式で印字しない（一覧混入バグの修正）。
                    print(f"     ✔ Pro区間クリップ生成: {Path(out_path).name}")
                else:
                    print(f"✅ 完成: {out_path}")
                return True
            print("     ⚠️ 検証済み出力の安全な公開に失敗")
            final_out.unlink(missing_ok=True)
            return False
        if ok:
            got = ffprobe_dur(render_out)
            print(f"     ⚠️ Pro出力検査で不合格（期待{music_dur:.2f}s / 実測{got:.2f}s）")
        return False
    finally:
        _free_device_mem()   # 曲の終わりにGPUキャッシュを解放（次の曲のために空ける）
        shutil.rmtree(tmp, ignore_errors=True)


def _clean_path(s):
    """Terminalドラッグのパスを正規化（\\ エスケープ・クォートを解除）。"""
    s = (s or "").strip()
    if not s:
        return ""
    try:
        import shlex
        parts = shlex.split(s)
        if parts:
            return parts[0]
    except Exception:
        pass
    return s.strip().strip("'\"").replace("\\ ", " ").replace("\\", "")


def main():
    print("=" * 56)
    print("  🎧 DJ LipSync Pro  —  実験用 高精度リップシンク")
    print("        created by DJ SOPY / @sousouagain")
    print("=" * 56)
    print()
    music = ""
    while True:
        music = _clean_path(input("🎵 Remix音源（mp3/m4a/wav/mp4等）をドラッグ&ドロップ（qで終了）:\n  > "))
        if music.lower() in ("q", "quit", "exit"):
            return
        if music and Path(music).exists():
            break
        print("  ❌ 見つかりません。ファイルをこの行にドラッグ&ドロップしてEnter（qで終了）\n")
    print()
    mv = ""
    while True:
        raw = input("🎬 MVを指定（YouTube URL を貼る／または動画ファイルをドラッグ）（qで終了）:\n  > ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            return
        if raw.startswith("http"):
            mv = raw; break
        mv = _clean_path(raw)
        if mv and Path(mv).exists():
            break
        print("  ❌ URLを貼るか、動画ファイルをドラッグしてEnter（qで終了）\n")
    print()
    uh = input("🧠 HuBERT特徴量を使いますか？（重い/高精度。Enter=使う, n=MFCCで軽く）:\n  > ").strip().lower()
    use_hubert = (uh != "n")
    print()
    pm = input("🎞 配置方法（Enter=等速/See You Again式・おすすめ, w=warp/テンポ伸縮）:\n  > ").strip().lower()
    placement = "warp" if pm == "w" else "equal"
    print()
    so = input("🎯 ズレ補正（Enter=自動おすすめ / 手動ms 例150・-150 / 0=補正なし）:\n  > ").strip()
    if so == "":
        sync_offset_ms = "auto"
    else:
        try:
            sync_offset_ms = int(so)
        except Exception:
            sync_offset_ms = "auto"

    stem = Path(music).stem
    desktop = Path.home() / "Desktop"
    out = (desktop if desktop.exists() else Path.cwd()) / f"{stem}_LIPSYNC_PRO.mp4"

    print()
    ok = process(music, mv, str(out), use_hubert=use_hubert, placement=placement,
                 sync_offset_ms=sync_offset_ms)
    print()
    if ok:
        print(f"🎉 完了！ → {out}")
    else:
        print("⚠️ 失敗しました。上のログ（どのステージで止まったか）を教えてください。")
    input("Enterで閉じる...")


if __name__ == "__main__":
    main()
