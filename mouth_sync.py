#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mouth_sync.py  ―― 段階1：MV映像から「口の動き（＝歌ってるか）」が取れるか検証する。

使い方（単体）:
    python3 mouth_sync.py /path/to/MV.mp4
    python3 mouth_sync.py /path/to/MV.mp4 --fps 10 --csv out.csv --max 60

何をする?:
  - 動画を一定fpsでサンプリングし、各フレームで顔の口の開き(MAR)を測る
  - 「口が動いている＝歌ってる」度合いを、MARの時間的なゆらぎ(局所標準偏差)で出す
  - 区間ごとに「口パク中／静か・顔なし」を表示
  - その動画が“口の動きを使える素材か”（顔検出率）を判定して教える

設計メモ:
  - 歌ってる判定に使うのは「口の開きの絶対値」ではなく「動き(ゆらぎ)」。
    歌う＝口を開け閉めする＝MARが時間的に変動する。静止した口や無表情は変動が小さい。
  - 顔が映ってない/横向き等で取れないフレームは face=0（不明）として扱い、
    歌ってる判定には使わない（音側のフォールバックに任せる前提）。
  - mediapipe は環境により API が違うので、solutions（旧・モデル同梱）を優先し、
    無ければ Tasks（新・モデル自動DL）にフォールバックする両対応。
  - ここでは判定するだけで、まだ既存の同期処理には一切手を出さない（独立スクリプト）。
"""

import sys
import os
import math
import urllib.request

# --- 口(内唇)と口角のFaceMeshランドマーク番号（468点モデル共通）---
UP_INNER = 13    # 上唇の内側中央
LO_INNER = 14    # 下唇の内側中央
L_CORNER = 61    # 口角(左)
R_CORNER = 291   # 口角(右)

# Tasks API 用モデル（solutionsが無い環境のときだけ自動DL）
_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
_TASK_PATH = os.path.expanduser("~/.dj_video_maker/face_landmarker.task")


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _make_extractor():
    """利用可能な mediapipe API を見つけ、
       extract(rgb, w, h) -> [(x,y), ...] or None と backend名 とハンドルを返す。"""
    import mediapipe as mp

    # --- 旧 solutions API（モデル同梱・最も一般的）---
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        def extract(rgb, w, h, timestamp_ms=None):
            res = fm.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark
            return [(lm[i].x * w, lm[i].y * h) for i in range(len(lm))]
        return extract, "solutions", fm

    # --- 新 Tasks API（モデルを自動DL）---
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    if not os.path.exists(_TASK_PATH):
        os.makedirs(os.path.dirname(_TASK_PATH), exist_ok=True)
        print("   ⬇️ 顔モデルを初回ダウンロード中（face_landmarker.task）...")
        urllib.request.urlretrieve(_TASK_URL, _TASK_PATH)
    base = mp_python.BaseOptions(model_asset_path=_TASK_PATH)
    opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=1,
                                        running_mode=vision.RunningMode.VIDEO)
    lmk = vision.FaceLandmarker.create_from_options(opts)

    def extract(rgb, w, h, timestamp_ms=None, _last_ts=[-1]):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEOモードの時刻は、モデルの想定30fpsではなく実際に
        # サンプリングしたフレームのPTSを渡す。これがずれると顔追跡の
        # 速度想定が狂い、口の動き量も不安定になる。
        ts = int(round(float(timestamp_ms or 0)))
        ts = max(ts, _last_ts[0] + 1)
        _last_ts[0] = ts
        res = lmk.detect_for_video(mp_img, ts)
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        return [(p.x * w, p.y * h) for p in lm]
    return extract, "tasks", lmk


def analyze_mouth(video_path, fps=10.0, max_seconds=None):
    """動画を解析して (times, mar, face_mask, activity, dur, backend) を返す。"""
    import numpy as np
    import cv2

    extract, backend, _h = _make_extractor()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = (n_frames / src_fps) if (n_frames and src_fps) else None
    step = max(1, int(round(src_fps / float(fps))))
    actual_fps = src_fps / step

    times, mar, face_mask = [], [], []
    fi = 0; last_t = -1.0
    try:
        while True:
            if not cap.grab():
                break
            if fi % step == 0:
                ok2, frame = cap.retrieve()
                if not ok2:
                    break
                # CFRならframe/fps、VFRならOpenCVが返す実PTSを優先。
                # 一部backendはPOS_MSECを0のまま返すため、単調増加しない
                # 場合はframe/fpsへ安全に戻す。
                t_nominal = fi / src_fps
                t_pts = None
                try:
                    pos_prop = getattr(cv2, "CAP_PROP_POS_MSEC", None)
                    pos_ms = float(cap.get(pos_prop)) if pos_prop is not None else -1.0
                    if np.isfinite(pos_ms) and pos_ms >= 0.0:
                        cand = pos_ms / 1000.0
                        if last_t < 0.0 or cand > last_t + 1e-6:
                            t_pts = cand
                except Exception:
                    pass
                t = t_nominal if t_pts is None else t_pts
                if t <= last_t:
                    t = t_nominal
                last_t = t
                if max_seconds is not None and t > max_seconds:
                    break
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pts = extract(rgb, w, h, int(round(t * 1000.0)))
                if pts and len(pts) > max(UP_INNER, LO_INNER, L_CORNER, R_CORNER):
                    vert = _dist(pts[UP_INNER], pts[LO_INNER])
                    horiz = _dist(pts[L_CORNER], pts[R_CORNER]) + 1e-6
                    times.append(t); mar.append(vert / horiz); face_mask.append(1)
                else:
                    times.append(t); mar.append(float("nan")); face_mask.append(0)
            fi += 1
    finally:
        cap.release()

    times = np.array(times); mar = np.array(mar); face_mask = np.array(face_mask)
    if len(times) > 1:
        diffs = np.diff(times)
        diffs = diffs[diffs > 1e-6]
        if len(diffs):
            actual_fps = 1.0 / float(np.median(diffs))
    activity = _activity_from_mar(mar, face_mask, actual_fps)
    return times, mar, face_mask, activity, dur, backend


def _activity_from_mar(mar, face_mask, fps):
    """口の動き度 = MARの局所標準偏差(±0.5秒窓)。顔なし区間は0。"""
    import numpy as np
    activity = np.zeros_like(mar)
    if len(mar) < 3:
        return activity
    win = max(1, int(round(0.5 * fps)))
    valid = (np.asarray(face_mask) > 0) & np.isfinite(mar)
    p = 0
    while p < len(mar):
        if not valid[p]:
            p += 1; continue
        run_lo = p
        while p < len(mar) and valid[p]:
            p += 1
        run_hi = p
        # 顔が消えたカットを跨いで前のMARを引き継がない。
        # シーン切替の前後の口形差が「歌唱」と誤検出されるのを防ぐ。
        for i in range(run_lo, run_hi):
            lo = max(run_lo, i - win); hi = min(run_hi, i + win + 1)
            vals = np.asarray(mar[lo:hi], dtype=float)
            if len(vals) >= 2:
                activity[i] = float(np.std(vals))
    return activity


def summarize(times, mar, face_mask, activity, dur, backend, seg_sec=10.0, act_thresh=0.012):
    import numpy as np
    n = len(times)
    if n == 0:
        print("  ⚠️ フレームが取れませんでした"); return
    face_rate = float(np.mean(face_mask))
    print(f"   使用API: {backend}")
    print(f"🙂 顔検出: 全{n}サンプル中 {int(face_mask.sum())} ({face_rate*100:.0f}%) で顔を検出")
    print(f"👄 区間ごとの「口の動き(歌ってる)度」: (しきい値 act≥{act_thresh})")
    t_end = times[-1]; s = 0.0
    while s < t_end:
        m = (times >= s) & (times < s + seg_sec)
        if int(m.sum()) > 0:
            a = float(np.mean(activity[m])); fr = float(np.mean(face_mask[m]))
            tag = "口パク中" if a >= act_thresh else "静か/顔なし"
            print(f"   {s:5.0f}-{s+seg_sec:4.0f}s : 動き={a:.3f}  顔{fr*100:3.0f}%  → {tag}")
        s += seg_sec
    print("")
    if face_rate >= 0.5:
        print(f"✅ 顔検出率 {face_rate*100:.0f}% → 口の動きを“使える”素材。次の段階(歌区間との照合)に進めます。")
    elif face_rate >= 0.25:
        print(f"△ 顔検出率 {face_rate*100:.0f}% → 部分的に使える。顔アップ区間だけ補助に使う形なら有効。")
    else:
        print(f"❌ 顔検出率 {face_rate*100:.0f}% → 顔アップが少なく、このMVには不向き。音側の判定に頼るべき。")


def build_mouth_profile(video_path, fps=10.0, act_thresh=0.012, max_seconds=None):
    """MV動画を解析して「各時刻の口パク度」プロファイルを返す。
    戻り: dict { 'times','activity','face','fps','thresh','face_rate','backend' } または None。
    mediapipe/cv2 が無い・解析失敗時は None（＝この機能をスキップして従来通り）。"""
    try:
        import numpy as np
        times, mar, face_mask, activity, dur, backend = analyze_mouth(
            video_path, fps=fps, max_seconds=max_seconds)
        if len(times) == 0:
            return None
        sample_fps = (1.0 / float(np.median(np.diff(times)))) if len(times) > 1 else float(fps)
        return {"times": times, "mar": mar, "activity": activity, "face": face_mask,
                "fps": sample_fps, "thresh": act_thresh,
                "face_rate": float(np.mean(face_mask)), "backend": backend}
    except Exception as e:
        print(f"     ⚠️ 口解析スキップ（mouth_sync）: {str(e)[:80]}")
        return None


def _nearest_profile_index(profile, t):
    """口プロファイルで時刻tに最も近いサンプルindex。"""
    import numpy as np
    times = profile.get("times", []) if profile else []
    if len(times) == 0:
        return None
    right = int(np.clip(np.searchsorted(times, t, side="left"), 0, len(times) - 1))
    left = max(0, right - 1)
    return left if abs(float(t) - times[left]) <= abs(times[right] - float(t)) else right


def is_mv_singing(profile, t):
    """MVの時刻tで口パク中(歌ってる)か。顔が取れてない/不明なら False（＝確証なし）。"""
    if not profile:
        return False
    times = profile["times"]
    if len(times) == 0:
        return False
    i = _nearest_profile_index(profile, t)
    if profile["face"][i] == 0:        # 顔が取れていない点は歌ってると断定しない
        return False
    return bool(profile["activity"][i] >= profile["thresh"])


def mv_face_but_silent(profile, t):
    """MVの時刻tで『顔はちゃんと取れているのに口が止まっている』か。
    これは remixが歌っている時に来ると“ズレ”の確証になる（顔なしは判定不能→False）。"""
    if not profile:
        return False
    times = profile["times"]
    if len(times) == 0:
        return False
    i = _nearest_profile_index(profile, t)
    if profile["face"][i] == 0:        # 顔が取れていない→判断不能（誤爆させない）
        return False
    return bool(profile["activity"][i] < profile["thresh"])


def pick_quiet_mv_time(profile, want_dur, mv_dur, avoid=None, avoid_win=4.0):
    """MVの中で『口が動いていない（歌っていない/顔なし）』区間の先頭時刻を返す。
    remixの無声区間に貼るBロール用。want_dur 秒ぶん連続で静かな箇所を優先。
    avoid: 直前に使った時刻のリスト（近接を避けてチカチカを防ぐ）。見つからねば None。"""
    if not profile:
        return None
    import numpy as np
    times = profile["times"]; act = profile["activity"]
    thresh = profile["thresh"]; fps = profile["fps"]
    if len(times) < 2:
        return None
    need = max(1, int(round(want_dur * fps)))
    quiet = (act < thresh)
    face = profile.get("face")
    avoid = avoid or []
    best_t = None; best_score = -1.0
    i = 0; n = len(times)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) >= need:
                t0 = float(times[i])
                if t0 + want_dur <= mv_dur - 0.05:
                    far = min([abs(t0 - a) for a in avoid], default=99.0)
                    # 顔が映っていない区間（=真のBロール）を優先（歌区間の逃がし先に最適）
                    noface_bonus = 0.0
                    if face is not None:
                        import numpy as np
                        noface_bonus = 8.0 * float(np.mean(face[i:j] == 0))
                    score = (j - i) + min(far, 10.0) + noface_bonus
                    if score > best_score:
                        best_score = score; best_t = t0
            i = j
        else:
            i += 1
    return best_t


def voice_envelope(audio_path, hop=0.05, sr=22050):
    """remix音声から声の大小(RMS包絡)を出す → (times, env)。口の開閉と照合する用。"""
    import numpy as np
    import librosa
    y, _sr = librosa.load(str(audio_path), sr=sr, mono=True)
    h = max(1, int(sr * hop))
    rms = librosa.feature.rms(y=y, frame_length=h * 2, hop_length=h)[0]
    t = np.arange(len(rms)) * (h / sr)
    rms = rms / (np.max(rms) + 1e-9)
    return t, rms.astype(np.float32)


def measure_segment_mouth_lag(profile, seg_remix_t, seg_mv_t, venv_times, venv,
                              max_lag=4.0, step=0.1, min_face=0.40, min_corr=0.35,
                              ret_always=False):
    """区間の『口の動き vs 歌声』のズレ秒数を相互相関で測る。
      seg_remix_t : 区間内のremix時刻（配列）
      seg_mv_t    : 各remix時刻に対応する“今当ててるMV時刻”（アンカー線）
      venv_times,venv: remixの声の包絡
    口の開閉(MAR動き)と歌声の包絡が一番揃うδを探す。
    戻り: (lag秒, corr, face_cov)。ゲート未達なら (None, corr, face_cov)。
      ret_always=True のときは lag を None にせず実測値を返す（デバッグ用）。
    lag>0 は「MVのもっと後ろを当てるべき(映像が手前すぎ)」を意味する。"""
    if not profile or len(seg_remix_t) < 4:
        return None, 0.0, 0.0
    import numpy as np
    times = profile["times"]; act = profile["activity"]; face = profile["face"]
    if len(times) < 4:
        return None, 0.0, 0.0
    seg_remix_t = np.asarray(seg_remix_t, float)
    seg_mv_t = np.asarray(seg_mv_t, float)
    V = np.interp(seg_remix_t, venv_times, venv)
    if np.std(V) < 1e-6:
        return None, 0.0, 0.0
    V = (V - V.mean()) / (V.std() + 1e-9)
    def nearest_values(query, values):
        """顔マスクは0/1の離散値なので線形補間しない。"""
        q = np.asarray(query, dtype=float)
        right = np.searchsorted(times, q, side="left")
        right = np.clip(right, 0, len(times) - 1)
        left = np.clip(right - 1, 0, len(times) - 1)
        use_left = np.abs(q - times[left]) <= np.abs(times[right] - q)
        idx = np.where(use_left, left, right)
        return np.asarray(values)[idx]

    # 探索中のラグごとに「その移動先で顔が取れているか」も
    # 再計算する。従来は移動前の顔率だけを使っていたため、顔の無い
    # カットへのシフトが偶然高相関に見えることがあった。
    deltas = np.arange(-max_lag, max_lag + 1e-9, step)
    best_corr = -2.0; best_lag = 0.0; best_fcov = 0.0; best_score = -2.0
    for d in deltas:
        shifted = seg_mv_t + d
        in_range = (shifted >= times[0]) & (shifted <= times[-1])
        valid = in_range & (nearest_values(shifted, face.astype(float)) >= 0.5)
        fc = float(np.mean(valid))
        if int(np.sum(valid)) < 4:
            continue
        A = np.interp(shifted[valid], times, act)
        VV = V[valid]
        if np.std(A) < 1e-6 or np.std(VV) < 1e-6:
            continue
        An = (A - A.mean()) / (A.std() + 1e-9)
        Vn = (VV - VV.mean()) / (VV.std() + 1e-9)
        c = float(np.mean(Vn * An))
        # 顔カバー率が少ない候補が、数点の偶然一致だけで
        # 勝たないよう軽い罰則を与える。生の相関値はログ用に保つ。
        score = c - 0.15 * max(0.0, min_face - fc)
        if score > best_score:
            best_score = score; best_corr = c; best_lag = float(d); best_fcov = fc
    if best_score <= -2.0:
        return None, 0.0, 0.0
    # ゲート判定
    passed = (best_fcov >= min_face) and (best_corr >= min_corr)
    if not passed and not ret_always:
        return None, best_corr, best_fcov
    if not passed and ret_always:
        return best_lag, best_corr, best_fcov   # 実測値は返すが呼び出し側で min_shift 等で弾く
    return best_lag, best_corr, best_fcov


def main():
    args = sys.argv[1:]
    if not args:
        print("使い方: python3 mouth_sync.py <動画ファイル> [--fps 10] [--csv out.csv] [--max 秒]")
        sys.exit(1)
    video = args[0]
    fps = 10.0; csv_path = None; max_sec = None
    if "--fps" in args: fps = float(args[args.index("--fps") + 1])
    if "--csv" in args: csv_path = args[args.index("--csv") + 1]
    if "--max" in args: max_sec = float(args[args.index("--max") + 1])
    if not os.path.exists(video):
        print(f"❌ ファイルが見つかりません: {video}"); sys.exit(1)

    print(f"🎬 動画: {os.path.basename(video)} / 解析fps={fps}")
    times, mar, face_mask, activity, dur, backend = analyze_mouth(video, fps=fps, max_seconds=max_sec)
    if dur: print(f"   長さ≈{dur:.0f}秒 / サンプル数={len(times)}")
    summarize(times, mar, face_mask, activity, dur, backend)

    if csv_path:
        with open(csv_path, "w") as f:
            f.write("time,mar,face,activity\n")
            for i in range(len(times)):
                f.write(f"{times[i]:.3f},{mar[i]:.4f},{int(face_mask[i])},{activity[i]:.4f}\n")
        print(f"\n📄 CSV書き出し: {csv_path}")


if __name__ == "__main__":
    main()
