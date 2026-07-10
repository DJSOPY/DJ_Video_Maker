#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
#  DJ LipSync Pro  —  高精度リップシンクエンジン（DJ Video Maker 本体の中核）
#  created by DJ SOPY / @sousouagain
#
#  dj_maker_core.py から呼ばれる。波形/クロマで合わない別アレンジRemixを、
#  ボーカル照合（HuBERT）・歌詞整列・口の動き解析で公式MVに同期させる。
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
SYNC_FIRST = True    # 歌ってる区間は同期最優先（その区間だけ多様化を弱める）
FORCE_BROLL = True   # 無声区間で歌唱カットしか候補に無い時、非歌唱カットを強制的に当てる
CACHE_DIR = Path.home() / ".dj_video_maker" / "lipsync_pro_cache"


def run(cmd, capture=True):
    return subprocess.run(cmd, capture_output=capture, text=True, errors="replace")


def ffprobe_dur(path):
    try:
        r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
        return float((r.stdout or "0").strip())
    except Exception:
        return 0.0


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
    fe, model = _HUBERT["fe"], _HUBERT["model"]
    dev = _HUBERT.get("dev", "cpu")

    def _run(on_dev):
        with torch.no_grad():
            inp = fe(y16, sampling_rate=16000, return_tensors="pt").input_values
            if on_dev != "cpu":
                inp = inp.to(on_dev)
            out = model(inp, output_hidden_states=True)
            hs = out.hidden_states          # tuple: 長さ13（embeddings + 12層）
            layer = min(HUBERT_LAYER, len(hs) - 1)
            return hs[layer][0].float().cpu().numpy()   # (T,768) ~20ms hop

    try:
        feat = _run(dev)
    except Exception as e:
        # MPS等のop非対応で落ちたら、CPUに切り替えて再実行（以後CPU固定）
        if dev != "cpu":
            print(f"     ⚠️ HuBERT {dev.upper()} で失敗→CPUにフォールバック: {str(e)[:80]}")
            try:
                model.to("cpu")
            except Exception:
                pass
            _HUBERT["dev"] = "cpu"
            feat = _run("cpu")
        else:
            raise
    _free_device_mem()   # HuBERTの中間データ(大)を曲ごとに解放してMPS枯渇を防ぐ
    times = np.arange(feat.shape[0]) * 0.02
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


def windowed_topk(rfeat, rt, ofeat, ot, win_sec=WIN_SEC, hop_sec=HOP_SEC, topk=TOPK, ds=DTW_DS):
    import librosa
    if len(rt) < 2 or len(ot) < 2:
        return []
    fr = 1.0 / max(np.median(np.diff(rt)), 1e-6)
    wlen = max(4, int(win_sec * fr))
    whop = max(2, int(hop_sec * fr))
    ofps = 1.0 / max(np.median(np.diff(ot)), 1e-6)
    ds = max(1, int(ds))
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
            D = librosa.sequence.dtw(X=wd, Y=Yd, subseq=True,
                                     metric="cosine", backtrack=False)
            lastrow = D[-1]
            cand = _topk_local_minima(lastrow, topk, min_gap)
            rt0 = float(rt[i])
            rt_end = float(rt[min(i + wlen - 1, len(rt) - 1)])
            wdur = max(0.1, rt_end - rt0)
            nlen = max(1, len(win[::ds]))
            out = []
            for col, cost in cand:
                oc = min(col * ds, len(ot) - 1)          # ダウンサンプル列→元の時間軸へ戻す
                end_ot = float(ot[oc])
                start_ot = max(0.0, end_ot - wdur)
                # コストはフレーム数で正規化（平均コサイン距離＝従来と同スケール。Viterbi罰則は不変でOK）
                out.append((start_ot, cost / nlen))
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
def viterbi_path(windows, jump_penalty=JUMP_PENALTY, switch_cost=SWITCH_COST):
    if not windows:
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
    last = int(np.argmin(dp[-1]))
    path_idx = [last]
    for j in range(N - 1, 0, -1):
        last = bk[j][last]
        path_idx.append(last)
    path_idx.reverse()
    anchors = []
    for j, ki in enumerate(path_idx):
        rt0, _wd, cs = windows[j]
        ot0, c = cs[ki]
        anchors.append([rt0, ot0, c])
    return anchors


def _voc_envelope(voc, sr, hop=VOC_HOP):
    """声のRMS包絡から『歌ってる/無声』を判定 → (times, active_bool)。失敗時None。"""
    try:
        h = max(1, int(sr * hop))
        n = len(voc) // h
        if n < 2:
            return None
        v = np.asarray(voc, dtype=np.float32)
        rms = np.sqrt(np.array([np.mean(v[i*h:(i+1)*h] ** 2) for i in range(n)]) + 1e-9)
        thr = 0.15 * np.percentile(rms, 90)   # 90%値の15%を無声境界に
        return (hop, rms > thr)
    except Exception:
        return None


def _act_at(act, t):
    """時刻tで声が出てるか。actが無ければ True 扱い（＝従来通り）。"""
    if not act:
        return True
    hop, active = act
    i = int(min(max(round(t / hop), 0), len(active) - 1))
    return bool(active[i])


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
    """MVが足りない曲(曲尺>MV尺)を3分割する：
      ・イントロ = 最初の8小節 ∪ 歌い出し前 → ラフ（MV自前の先頭を流す。使い回しOK）
      ・中盤     = 歌ってる区間 → リップシンクのアンカーをそのまま（MVを中盤に優先配分）
      ・アウトロ = 最後の16小節 ∪ 歌い終わり後 → ラフ（MV自前の末尾を流す）
    これで「後半MV尽きて逆走」を防ぐ（中盤に絞ればMVが足りる）。
    戻り: (新anchors, (intro_end,outro_start,intro_bars,outro_bars) or None)"""
    if music_dur <= mv_dur + 1.0 or not anchors:
        return anchors, None
    bpm = max(60.0, min(200.0, bpm if bpm and bpm > 0 else 125.0))
    sec_per_bar = 4.0 * 60.0 / bpm            # 1小節＝4拍
    intro_bars = 8
    deficit = music_dur - mv_dur              # MV不足分
    intro_len = min(intro_bars * sec_per_bar, music_dur * 0.30)   # 安全上限
    # アウトロ＝「16小節」と「不足分＋マージン」の広い方（中盤をMVに収める逆算）
    outro_len = max(16 * sec_per_bar, deficit + 8.0)
    outro_len = min(outro_len, music_dur * 0.45)                  # 安全上限
    outro_bars = outro_len / sec_per_bar
    # hook-first 判定：冒頭(イントロ窓)がしっかり歌っている＝先頭を「イントロ捨て」しない
    intro_voiced = False
    if rmx_act:
        _hop, _active = rmx_act
        _k = int(min(len(_active), max(1, round(intro_len / _hop))))
        intro_voiced = bool(_active[:_k].mean() >= INTRO_VOICED_FRAC) if _k > 0 else False

    if intro_voiced:
        intro_end = max(0.0, _first_voiced_time(rmx_act, 0.0))   # ≈0：先頭から歌
    else:
        intro_end = max(intro_len, _first_voiced_time(rmx_act, 0.0))
    outro_start = min(music_dur - outro_len, _last_voiced_time(rmx_act, music_dur))
    if outro_start <= intro_end + 5.0:
        return anchors, None                  # 中盤が取れない→従来通り
    mid = [list(a) for a in anchors if intro_end <= a[0] <= outro_start]
    if len(mid) < 3:
        return anchors, None
    if intro_voiced:
        intro = []   # hook-first：先頭の一致アンカーをそのまま使う（イントロ捨てしない）
        print("     🎯 hook-first 検出：冒頭が歌→イントロ捨てを無効化（先頭の一致を保持）")
    else:
        # イントロ：MV自前の先頭を等速で（歌が薄いので合わなくてOK）
        intro = [[0.0, 0.0, 1.0], [intro_end, min(intro_end, mv_dur - 1.0), 1.0]]
    # アウトロ：中盤の続き（最後のMV位置）から末尾へ。逆戻りさせない
    last_mid_mv = mid[-1][1]
    out_src0 = max(mv_dur - (music_dur - outro_start), last_mid_mv)
    out_src0 = min(out_src0, mv_dur - 1.0)
    outro = [[outro_start, out_src0, 1.0], [music_dur, mv_dur, 1.0]]
    merged = sorted(intro + mid + outro, key=lambda a: a[0])
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
        for (r0, o0, c) in anchors:
            ri0 = np.searchsorted(t_r, r0); ri1 = np.searchsorted(t_r, r0 + win)
            seg_r = os_r[ri0:ri1]
            if len(seg_r) < 4:
                out.append([r0, o0, c]); continue
            best_off, best_val = 0.0, -1e18
            step = 0.02
            off = -search
            while off <= search:
                oi0 = np.searchsorted(t_o, o0 + off)
                oi1 = oi0 + len(seg_r)
                seg_o = os_o[oi0:oi1]
                if len(seg_o) == len(seg_r) and len(seg_o) > 0:
                    v = float(np.dot(seg_r, seg_o) /
                              (np.linalg.norm(seg_r) * np.linalg.norm(seg_o) + 1e-8))
                    if v > best_val:
                        best_val, best_off = v, off
                off += step
            out.append([r0, max(0.0, o0 + best_off), c])
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


def whisper_word_align(anchors, rvoc, ovoc, sr):
    """Forced Alignment：Remixと原曲の歌詞を単語タイムスタンプ化し、
    既存アンカー(粗ガイド)が示すMV位置の近くで“同じ単語”にスナップ＝単語レベルの精密同期。
    歌詞が同じ(原曲アカペラ流用)前提で効く。使えない時は従来アンカーを返す。"""
    if not anchors:
        return anchors
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
    SEARCH = 2.5  # 粗ガイドが示すMV位置の±この秒で同じ単語を探す
    snapped = []
    for (w, rs, _re) in rw:
        times = mv_idx.get(w)
        if not times:
            continue
        exp = _interp_orig(rs, anchors)         # 粗ガイドのMV予想位置
        j = bisect.bisect_left(times, exp)
        best, bestd = None, 1e9
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < len(times):
                d = abs(times[jj] - exp)
                if d < bestd:
                    bestd, best = d, times[jj]
        if best is not None and bestd <= SEARCH:
            snapped.append([rs, best, 0.0])
    if len(snapped) < 8:
        print(f"     ℹ️ Forced Alignment: 一致語{len(snapped)}個（少）→ 従来アンカー維持")
        return anchors
    # 歌のない区間（単語アンカーが疎な所）は粗アンカーで補完
    snapped.sort(key=lambda a: a[0])
    wr = [a[0] for a in snapped]
    merged = list(snapped)
    for (r0, o0, c) in anchors:
        k = bisect.bisect_left(wr, r0)
        near = min((abs(wr[i] - r0) for i in (k - 1, k) if 0 <= i < len(wr)), default=1e9)
        if near > 4.0:
            merged.append([r0, o0, c])
    merged.sort(key=lambda a: a[0])
    print(f"     🗣 Forced Alignment: {len(snapped)}語を単語境界にスナップ（口パク精度↑）")
    return [tuple(x) for x in merged]


def _interp_orig(r, anchors):
    """アンカー列(rt,ot)から remix時刻r に対応する 原曲時刻o を区分線形で補間。"""
    if r <= anchors[0][0]:
        return max(0.0, anchors[0][1] + (r - anchors[0][0]))
    if r >= anchors[-1][0]:
        return anchors[-1][1] + (r - anchors[-1][0])
    for k in range(len(anchors) - 1):
        r0, o0 = anchors[k][0], anchors[k][1]
        r1, o1 = anchors[k + 1][0], anchors[k + 1][1]
        if r0 <= r <= r1 and r1 > r0:
            f = (r - r0) / (r1 - r0)
            # 並べ替えで後退する区間は“その区間の先頭から等速”にして破綻回避
            if o1 < o0:
                return max(0.0, o0 + (r - r0))
            return o0 + f * (o1 - o0)
    return max(0.0, anchors[-1][1])


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


def _lyric_best_matches(tokens_r, owords, span_pad=6, min_sep=8.0, want_index=False):
    """remixの単語列をMV単語列上でスライド照合し、上位3候補の(MV時刻, 一致率[, 窓開始index])を返す。
    一致率＝remix単語の被覆率（何割の単語が並び順どおり一致したか）。
    ratio()だと余白や末尾の切れた窓でスコアが歪むため、被覆率で公平に測る。
    MV時刻は整列ブロックから『remix先頭トークンに対応する位置』を精密に逆算する。"""
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
        score = matched / float(n)          # remix単語の被覆率（0〜1）
        j0 = i
        if blocks:
            a, b, _sz = blocks[0]
            j0 = max(0, i + b - a)
        j0 = min(j0, len(owords) - 1)
        cands.append((float(owords[j0][1]), score, i))
    cands.sort(key=lambda x: -x[1])
    picked = []
    for t, s, i in cands:
        if all(abs(t - pt) >= min_sep for pt, *_ in picked):
            picked.append((t, s, i) if want_index else (t, s))
        if len(picked) >= 3:
            break
    return picked


def _lyric_window_pins(wlist, owords, i_start, span_pad=6, min_block=3):
    """◎一意と判定した窓について、整列ブロック（min_block語以上の連続一致）から
    『remix単語時刻 ↔ MV単語時刻』の高精度ピンを取り出す。
    20秒の文脈が一致した窓の中の対応なので、行内の同語反復での取り違えが起きない。"""
    import difflib
    tokens_r = [w[0] for w in wlist]
    mv_tok = [w[0] for w in owords]
    seg = mv_tok[i_start:i_start + len(tokens_r) + span_pad]
    sm = difflib.SequenceMatcher(None, tokens_r, seg, autojunk=False)
    pins = []
    for a, b, size in sm.get_matching_blocks():
        if size >= min_block:
            for k in range(size):
                ri = a + k; oi = i_start + b + k
                if ri < len(wlist) and oi < len(owords):
                    pins.append((float(wlist[ri][1]), float(owords[oi][1])))
    return pins


def apply_lyrics_align(anchors, rwords, owords, music_dur=None, late_frac=0.45,
                       min_words=8, min_score=0.50, min_gap=0.15,
                       min_shift=1.0, mv_dur=None, win=20.0, hop=10.0):
    """歌詞（Whisper単語列）の“文脈”整列で、繰り返しサビの取り違えを直す。
    音(HuBERT)や口では区別できない『同じサビの別の繰り返し』も、
    前後の歌詞が違うため単語列の整列なら一意に特定できる。
    ★区間まるごとだと長すぎて一致率が薄まる（実測0.38）ため、
      区間内を win 秒窓（hop 刻み）でスライドし、◎一意（一致率min_score以上かつ
      2位との差min_gap以上）の窓だけからシフトを測る。
    後半(late_frac以降)のみ・窓のシフト中央値が min_shift 以上のときだけ区間を補正。
    戻り: (補正後アンカー, 補正区間数)。"""
    if not anchors or not rwords or not owords:
        return anchors, 0
    import numpy as np
    if music_dur is None:
        music_dur = max(a[0] for a in anchors)
    late_t = music_dur * late_frac
    out = [list(a) for a in anchors]
    bounds = _segment_bounds_by_jump(anchors)
    n_fix = 0
    all_pins = []          # 文脈検証済みの単語ピン [(r_lo, r_hi, [(r,mv),...]), ...]
    for k in range(len(bounds) - 1):
        i0, i1 = bounds[k], bounds[k + 1]
        seg = anchors[i0:i1]
        if len(seg) < 3:
            continue
        r_lo, r_hi = seg[0][0], seg[-1][0]
        if (r_lo + r_hi) * 0.5 < late_t:
            continue                      # 前半は触らない（合っている所を守る）
        # --- 区間内を窓でスライドし、◎一意の窓ごとにシフトとピンを測る ---
        shifts = []
        seg_pins = []
        t = r_lo
        while t < r_hi:
            t1 = min(t + win, r_hi)
            wlist = [w for w in rwords if t <= w[1] < t1]
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
            print(f"     🎼 [見送り] remix{r_lo:.0f}-{r_hi:.0f}s: ◎一意の窓なし（歌詞不足/曖昧）")
            continue
        med = float(np.median([s for _t, s, _sc, _g in shifts]))
        spread = float(np.std([s for _t, s, _sc, _g in shifts])) if len(shifts) > 1 else 0.0
        detail = " ".join(f"[{wt:.0f}s:{sh:+.1f}s({sc:.2f})]" for wt, sh, sc, _g in shifts[:4])
        if spread > 2.0:
            print(f"     🎼 [見送り] remix{r_lo:.0f}-{r_hi:.0f}s: 窓ごとの答えが不一致 "
                  f"(ばらつき{spread:.1f}s) {detail}")
            continue
        if abs(med) >= min_shift:
            for i in range(i0, i1):
                nv = out[i][1] + med
                if mv_dur is not None:
                    nv = min(nv, mv_dur - 0.05)
                out[i][1] = max(0.0, nv)
            n_fix += 1
            print(f"     🎼 歌詞整列補正：remix{r_lo:.0f}-{r_hi:.0f}s を {med:+.1f}s {detail}")
        else:
            print(f"     🎼 [確認] remix{r_lo:.0f}-{r_hi:.0f}s: 歌詞は現在位置を支持 "
                  f"(中央値{med:+.1f}s) {detail}")
        # --- 文脈検証済みピンを打ち込む（ピン間のDTWドリフトを殺し、レートも歌詞に固定）---
        if seg_pins:
            seg_pins.sort(key=lambda p: p[0])
            clean = []
            for pr, po in seg_pins:
                if clean and pr - clean[-1][0] < 0.15:
                    continue                              # remix側の重複除去
                if clean and po < clean[-1][1] - 0.5:
                    continue                              # MV側の後退（窓またぎの矛盾）を捨てる
                clean.append((pr, po))
            if len(clean) >= 3:
                p_lo, p_hi = clean[0][0], clean[-1][0]
                all_pins.append((p_lo, p_hi, clean))
    # --- ピン統合：ピンが張られた範囲は、既存アンカーを外してピンで置き換える ---
    #     ピンは「文脈が一致した歌詞の実タイミング」なので、DTW由来のジワジワ遅れを殺し、
    #     ピン間の再生レートも歌詞に沿って固定される（＝サビの中でズレが積もらない）。
    if all_pins:
        n_pin = 0
        kept = list(out)
        for p_lo, p_hi, clean in all_pins:
            kept = [a for a in kept if not (p_lo - 0.3 <= a[0] <= p_hi + 0.3)]
            kept.extend([[pr, po, 0.0] for pr, po in clean])
            n_pin += len(clean)
        kept.sort(key=lambda a: a[0])
        out = kept
        print(f"     🎼 歌詞ピン：{n_pin}語を文脈検証つきで固定（ピン間のドリフトを抑制）")
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
        if len(seg) < 4:
            continue
        srt = np.array([a[0] for a in seg]); smv = np.array([a[1] for a in seg])
        # 区間の中心が曲の後半に入っていなければスキップ（前半は触らない）
        if float(np.mean(srt)) < late_t:
            continue
        lag, corr, fcov = _ms.measure_segment_mouth_lag(mouth_profile, srt, smv, vt, env,
                                                        max_lag=max_lag, ret_always=True)
        # ゲートをここで明示判定（顔40%以上・相関0.35以上・ズレ0.6s以上のときだけ補正）
        passed = (fcov >= 0.40 and corr is not None and corr >= 0.35
                  and lag is not None and abs(lag) >= min_shift)
        if passed:
            for i in range(i0, i1):
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


def equal_and_mux(anchors, mv_path, music_path, music_dur, mv_dur, out_path, tmp_dir,
                  subseg=2.0, rmx_act=None, mouth_profile=None):
    """See You Again式＝基本は等速配置。Proの賢いアンカーで“位置”を決める。
    滑らかに前進してる区間は、ローカルな進行速度に合わせて軽く伸縮（±25%まで）し、
    テンポ差による区間内の口ズレを打ち消す。ジャンプ箇所は等速のカット。
    rmx_act/mouth_profile があれば、remix無声区間にMVの口パク映像が来た時だけ
    口が動いていないMV箇所へ差し替える（最終段フィルタ・経路には不干渉）。"""
    tmp_dir = Path(tmp_dir)
    anchors = sorted([a for a in anchors], key=lambda a: a[0])
    if not anchors:
        print("  ❌ アンカーが空"); return False
    seg_files = []
    listf = tmp_dir / "concat.txt"
    r = 0.0
    idx = 0
    scale_pad = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                 f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}")
    # --- 口パク差し替え（remix無声 & MV口パク中 → 口が動いてないMV箇所へ）---
    _ms = None
    if rmx_act is not None and mouth_profile is not None:
        try:
            import mouth_sync as _ms
        except Exception:
            _ms = None
    _swap_avoid = []          # 直前に差し替えた先（チカチカ回避）
    _n_swap = 0
    while r < music_dur - 0.05:
        dur = min(subseg, music_dur - r)
        o_pos = max(0.0, _interp_orig(r, anchors))
        # --- 口パク判断で差し替える ---
        #   ①remix無声 × MV口パク中            → 歌ってないのに口パク映像 → 逃がす
        #   ②remix歌ってる × 顔あり×口が止まってる → リップシンク破綻の確証 → 逃がす
        #   （顔が取れてない区間は判断不能なので触らない＝正しい映像を誤って捨てない）
        _swapped = False
        if _ms is not None:
            r_singing = _act_at(rmx_act, r + dur * 0.5)
            need_swap = False
            if not r_singing:
                if _ms.is_mv_singing(mouth_profile, o_pos):
                    need_swap = True
            else:
                if _ms.mv_face_but_silent(mouth_profile, o_pos):
                    need_swap = True
            if need_swap:
                alt = _ms.pick_quiet_mv_time(mouth_profile, dur, mv_dur, avoid=_swap_avoid)
                if alt is not None:
                    o_pos = max(0.0, alt)
                    _swap_avoid.append(alt)
                    if len(_swap_avoid) > 8:
                        _swap_avoid.pop(0)
                    _n_swap += 1
                    _swapped = True
        # 差し替えた区間は静かなMV箇所を等速で流す（元アンカーの伸縮には乗せない）
        o_end = (o_pos + dur) if _swapped else max(0.0, _interp_orig(r + dur, anchors))
        local_src = o_end - o_pos          # この区間でMV側が進む量
        # 滑らかな前進（ジャンプでない）なら、その速度に追従して区間内も同期維持
        rate = None
        if local_src > 0:
            ratio = local_src / dur        # >1: MVが速い（縮める）/ <1: MVが遅い（伸ばす）
            if 0.8 <= ratio <= 1.25:
                rate = ratio
        seg = tmp_dir / f"seg_{idx:04d}.mp4"
        idx += 1
        if rate is not None:
            src = rate * dur
            if o_pos + src > mv_dur - 0.02:
                src = max(0.05, mv_dur - 0.02 - o_pos)
            # MVの src 秒を dur 秒へ伸縮（setpts）。区間内ドリフトを打ち消す
            factor = dur / max(0.05, src)
            vf = f"setpts={factor:.5f}*PTS,{scale_pad}"
            rr = run(["ffmpeg", "-v", "error", "-y", "-ss", f"{o_pos:.3f}", "-t", f"{src:.3f}",
                      "-i", str(mv_path), "-an", "-vf", vf,
                      "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(seg)])
        else:
            # ジャンプ／範囲外：等速のカット
            if o_pos + dur > mv_dur - 0.05:
                o_pos = max(0.0, mv_dur - dur - 0.05)
            vf = scale_pad
            rr = run(["ffmpeg", "-v", "error", "-y", "-ss", f"{o_pos:.3f}", "-t", f"{dur:.3f}",
                      "-i", str(mv_path), "-an", "-vf", vf,
                      "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(seg)])
        if seg.exists() and seg.stat().st_size > 0:
            seg_files.append(seg)
        else:
            print(f"     ⚠️ セグメント失敗 r={r:.1f}: {(rr.stderr or '')[:120]}")
        r += dur
    if not seg_files:
        print("  ❌ 有効なセグメントが無い"); return False
    if _ms is not None and _n_swap > 0:
        print(f"     👄 口パク差し替え：{_n_swap}区間（remix無声×MV口パク → 静かなMV映像へ）")
    with open(listf, "w") as f:
        for s in seg_files:
            f.write(f"file '{s.as_posix()}'\n")
    silent = tmp_dir / "video_silent.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(listf), "-c", "copy", str(silent)])
    if not silent.exists():
        run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", str(listf), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(silent)])
    run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(music_path),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
         "-shortest", str(out_path)])
    return Path(out_path).exists() and Path(out_path).stat().st_size > 0


def warp_and_mux(anchors, mv_path, music_path, music_dur, mv_dur, out_path, tmp_dir):
    tmp_dir = Path(tmp_dir)
    # アンカーを時刻順に整理、両端を補完
    anchors = sorted([a for a in anchors], key=lambda a: a[0])
    if not anchors:
        print("  ❌ アンカーが空。warp不可")
        return False
    if anchors[0][0] > 0.5:
        anchors.insert(0, [0.0, max(0.0, anchors[0][1] - anchors[0][0]), 1.0])
    anchors.append([music_dur, min(mv_dur, anchors[-1][1] + (music_dur - anchors[-1][0])), 1.0])

    seg_files = []
    listf = tmp_dir / "concat.txt"
    for k in range(len(anchors) - 1):
        r0, o0, _ = anchors[k]
        r1, o1, _ = anchors[k + 1]
        dur = r1 - r0
        if dur <= 0.05:
            continue
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
        vf = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
              f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
              f"setpts={pts:.6f}*PTS,fps={FPS}")
        r = run(["ffmpeg", "-v", "error", "-y", "-ss", f"{o0:.3f}", "-t", f"{src_dur:.3f}",
                 "-i", str(mv_path), "-an", "-vf", vf,
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 "-t", f"{dur:.3f}", str(seg)])
        if seg.exists() and seg.stat().st_size > 0:
            seg_files.append(seg)
        else:
            print(f"     ⚠️ セグメント失敗 k={k}: {(r.stderr or '')[:160]}")

    if not seg_files:
        print("  ❌ 有効なセグメントが無い")
        return False
    with open(listf, "w") as f:
        for s in seg_files:
            f.write(f"file '{s.as_posix()}'\n")

    silent = tmp_dir / "video_silent.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(listf), "-c", "copy", str(silent)])
    if not silent.exists():
        run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", str(listf), "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", str(silent)])

    # remix音をmux（映像長を音に合わせる）
    run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(music_path),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
         "-shortest", str(out_path)])
    return Path(out_path).exists() and Path(out_path).stat().st_size > 0


# ============================================================
#  メイン
# ============================================================
def cache_feat(tag, wav, sr, use_hubert):
    """特徴量を音源ハッシュでキャッシュ（再走を高速化）。"""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        h = hashlib.md5(wav.tobytes()).hexdigest()[:16]
        kind = (f"hubert-L{HUBERT_LAYER}" if (use_hubert and _try_load_hubert()) else "mfcc")
        fp = CACHE_DIR / f"{h}_{kind}.npz"
        if fp.exists():
            d = np.load(fp)
            print(f"     ♻️ 特徴量[{tag}]キャッシュ再利用（{kind}）")
            return d["feat"], d["times"], kind
        feat, times, name = extract_features(wav, sr, tag, use_hubert)
        np.savez(fp, feat=feat, times=times)
        return feat, times, name
    except Exception:
        return extract_features(wav, sr, tag, use_hubert)


def process(music_path, mv_source, out_path, use_hubert=True, placement="equal",
            sync_offset_ms=0):
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

        # --- 特徴量 ---
        print("🧠 特徴量を抽出中...")
        rfeat, rt, fn = cache_feat("remix", rvoc, SR, use_hubert)
        ofeat, ot, _ = cache_feat("MV", ovoc, SR, use_hubert)

        # --- subsequence DTW（multi-scale：粗10s＋細5sの2スケールで候補抽出）---
        print("🔬 subsequence DTW（multi-scale 粗10s＋細5s で候補抽出）...")
        win_fine = windowed_topk(rfeat, rt, ofeat, ot, win_sec=WIN_SEC, hop_sec=HOP_SEC)
        win_coarse = windowed_topk(rfeat, rt, ofeat, ot, win_sec=WIN_SEC * 2, hop_sec=WIN_SEC)
        windows_raw = sorted(win_fine + win_coarse, key=lambda w: w[0])
        windows = merge_windows_by_time(windows_raw)
        print(f"     窓数: 細{len(win_fine)} + 粗{len(win_coarse)} = {len(windows)}（重複時刻を統合）")

        # --- 全体経路推定（Viterbi＝等速基準で時間順を保つ。図の通り）---
        print("🧭 全体経路推定（Viterbi＝等速を基準に順番を保つ）...")
        anchors = viterbi_path(windows)
        print(f"     アンカー数: {len(anchors)}")

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

        # --- Forced Alignment（Whisper単語スナップ＝口パク精度の最終段）---
        print("🗣 Forced Alignment（Whisper単語タイムスタンプで単語境界に合わせ）...")
        anchors = whisper_word_align(anchors, rvoc, ovoc, SR)

        # --- 3分割（MVが足りない曲：イントロ/アウトロをラフにして中盤にMV優先配分）---
        rmx_act = _voc_envelope(rvoc, SR)
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
            off = sync_offset_ms / 1000.0
            anchors = [[r, max(0.0, o + off), c] for (r, o, c) in anchors]
            print(f"     🎯 ズレ補正 {sync_offset_ms:+d}ms を全体に適用")

        # --- 歌詞整列補正（繰り返しサビの取り違えを“前後の歌詞の文脈”で正す）---
        try:
            _rw = _whisper_words(rvoc, SR, "remix")
            _ow = _whisper_words(ovoc, SR, "MV")
            if _rw and _ow:
                print("🎼 歌詞整列（Whisper単語列の文脈で後半の繰り返しを判定）...")
                anchors, n_lyr = apply_lyrics_align(anchors, _rw, _ow,
                                                    music_dur=music_dur, mv_dur=mv_dur)
                if n_lyr == 0:
                    print("     🎼 歌詞整列補正：該当区間なし（全区間◎一意に届かず/ズレ小）")
        except Exception as _e:
            print(f"     ⚠️ 歌詞整列スキップ: {str(_e)[:60]}")

        # --- 口パク差し替え用プロファイル（mouth_sync があれば。無ければNoneで従来通り）---
        mouth_profile = None
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
        except Exception as _e:
            mouth_profile = None   # mediapipe等が無ければスキップ（従来通り動く）

        # --- 配置（既定=等速 See You Again式／warpは任意）---
        if placement == "warp":
            print("🎞 映像をwarp（テンポ伸縮）して remix音にmux...")
            ok = warp_and_mux(anchors, mv, music_path, music_dur, mv_dur, out_path, tmp)
        else:
            print("🎞 等速配置（単語アンカーに追従）で remix音にmux...")
            ok = equal_and_mux(anchors, mv, music_path, music_dur, mv_dur, out_path, tmp,
                               rmx_act=rmx_act, mouth_profile=mouth_profile)
        if ok:
            print(f"✅ 完成: {out_path}")
        return ok
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
