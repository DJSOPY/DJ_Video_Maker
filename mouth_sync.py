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

# 口元の三値判定。UNCERTAINを「口なし」と扱わないことが重要。
MOUTH_ABSENT = -1
MOUTH_UNCERTAIN = 0
MOUTH_CLEAR = 1
MIN_VISIBLE_MOUTH_WIDTH = 0.020
MOUTH_FRAME_MARGIN = 0.005

_AUX_CASCADES = None

# Tasks API 用モデル（solutionsが無い環境のときだけ自動DL）
_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
_TASK_PATH = os.path.expanduser("~/.dj_video_maker/face_landmarker.task")


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _landmark_visibility(pts, width, height,
                         min_mouth_width=MIN_VISIBLE_MOUTH_WIDTH,
                         frame_margin=MOUTH_FRAME_MARGIN):
    """単一顔のランドマークを正規化し、口元の可視性を返す。

    閉口や小さい口、部分的な画面外は UNCERTAIN。ABSENTは口の
    主要ランドマークのbbox全体が画面外と明示できる場合だけ。
    """
    nan2 = (float("nan"), float("nan"))
    nan4 = (float("nan"),) * 4
    result = {
        "face_visible": False, "face_visibility": 0.0,
        "face_bbox": nan4, "mouth_state": MOUTH_UNCERTAIN,
        "mouth_visible": False, "mouth_absent": False,
        "mouth_center": nan2, "mouth_size": nan2, "mouth_bbox": nan4,
    }
    try:
        width = float(width); height = float(height)
        if not pts or width <= 0.0 or height <= 0.0:
            return result
        finite = []
        in_frame = 0
        for p in pts:
            x = float(p[0]) / width; y = float(p[1]) / height
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            finite.append((x, y))
            in_frame += int(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
        if not finite:
            return result
        xs = [p[0] for p in finite]; ys = [p[1] for p in finite]
        result["face_visible"] = True
        result["face_visibility"] = float(in_frame) / float(len(pts))
        result["face_bbox"] = (min(xs), min(ys), max(xs), max(ys))

        required = (UP_INNER, LO_INNER, L_CORNER, R_CORNER)
        if len(pts) <= max(required):
            return result
        mouth = []
        for idx in required:
            x = float(pts[idx][0]) / width; y = float(pts[idx][1]) / height
            if not (math.isfinite(x) and math.isfinite(y)):
                return result
            mouth.append((x, y))
        mx = [p[0] for p in mouth]; my = [p[1] for p in mouth]
        left, right, upper, lower = mouth[2], mouth[3], mouth[0], mouth[1]
        mouth_width = math.hypot(right[0] - left[0], right[1] - left[1])
        mouth_height = math.hypot(lower[0] - upper[0], lower[1] - upper[1])
        bbox = (min(mx), min(my), max(mx), max(my))
        result["mouth_center"] = ((left[0] + right[0]) * 0.5,
                                  (upper[1] + lower[1]) * 0.5)
        result["mouth_size"] = (mouth_width, mouth_height)
        result["mouth_bbox"] = bbox
        wholly_outside = (bbox[2] < 0.0 or bbox[0] > 1.0 or
                           bbox[3] < 0.0 or bbox[1] > 1.0)
        safely_inside = all(
            frame_margin <= x <= 1.0 - frame_margin and
            frame_margin <= y <= 1.0 - frame_margin for x, y in mouth)
        if wholly_outside:
            result["mouth_state"] = MOUTH_ABSENT
            result["mouth_absent"] = True
        elif safely_inside and mouth_width >= float(min_mouth_width):
            result["mouth_state"] = MOUTH_CLEAR
            result["mouth_visible"] = True
        return result
    except (TypeError, ValueError, IndexError, OverflowError):
        return result


def _aggregate_mouth_observation(face_measurements, auxiliary_presence=None):
    """複数顔と補助検出の結果を、安全側の三値にまとめる。"""
    face_measurements = list(face_measurements or [])
    # 「検出器を正常実行したが0件」と「本当に口元が存在しない」は同義ではない。
    # 小顔・暗所・横顔・モーションブラーでは複数検出器が同時に見逃し得るため、
    # 0件をABSENTへ昇格させない。これは安全Bロールを減らしてでも守る不変条件。
    if not face_measurements:
        return MOUTH_UNCERTAIN
    if any(m.get("mouth_state") == MOUTH_CLEAR for m in face_measurements):
        return MOUTH_CLEAR
    if any(m.get("mouth_state") != MOUTH_ABSENT for m in face_measurements):
        return MOUTH_UNCERTAIN
    # ここへ来るのはランドマークで実在を確認できた全顔について、口bboxが
    # 完全に画面外だった場合だけ。別の見逃し顔がいないことも補助確認する。
    return MOUTH_ABSENT if auxiliary_presence is False else MOUTH_UNCERTAIN


def _load_aux_cascades(cv2):
    """OpenCV同梱の検出器を一度だけ読み込む。失敗時はNone。"""
    global _AUX_CASCADES
    if _AUX_CASCADES is not None:
        return _AUX_CASCADES or None
    try:
        base = cv2.data.haarcascades
        names = {
            "frontal": "haarcascade_frontalface_alt2.xml",
            "profile": "haarcascade_profileface.xml",
            "smile": "haarcascade_smile.xml",
        }
        loaded = {}
        for key, name in names.items():
            cascade = cv2.CascadeClassifier(os.path.join(base, name))
            if cascade.empty():
                _AUX_CASCADES = {}
                return None
            loaded[key] = cascade
        _AUX_CASCADES = loaded
        return loaded
    except Exception:
        _AUX_CASCADES = {}
        return None


def _auxiliary_presence(frame, cv2):
    """縮小画像をHaarで多重確認。True=人物候補あり、False=全器正常で候補なし。"""
    cascades = _load_aux_cascades(cv2)
    if not cascades:
        return None
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > 480:
            scale = 480.0 / float(gray.shape[1])
            gray = cv2.resize(gray, (480, max(1, int(round(gray.shape[0] * scale)))))
        gray = cv2.equalizeHist(gray)
        common = dict(scaleFactor=1.15, minNeighbors=3, minSize=(24, 24))
        if len(cascades["frontal"].detectMultiScale(gray, **common)):
            return True
        if len(cascades["profile"].detectMultiScale(gray, **common)):
            return True
        flipped = cv2.flip(gray, 1)
        if len(cascades["profile"].detectMultiScale(flipped, **common)):
            return True
        if len(cascades["smile"].detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=8, minSize=(12, 8))):
            return True
        return False
    except Exception:
        return None


def _make_extractor():
    """利用可能な mediapipe API を見つけ、
       extract(rgb, w, h) -> [[(x,y), ...], ...] と backend名 とハンドルを返す。"""
    import mediapipe as mp

    # --- 旧 solutions API（モデル同梱・最も一般的）---
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=4, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        def extract(rgb, w, h, timestamp_ms=None):
            res = fm.process(rgb)
            if not res.multi_face_landmarks:
                return []
            return [[(p.x * w, p.y * h) for p in face.landmark]
                    for face in res.multi_face_landmarks]
        return extract, "solutions", fm

    # --- 新 Tasks API（モデルを自動DL）---
    # MediaPipe 0.10.35/macOS のFaceLandmarkerは、CPU delegateを指定しても
    # 内部のDrishtiMetalHelperがMetal初期化に失敗するとSIGABRTする環境がある。
    # Pythonのtry/exceptでは捕捉不能でDJ Maker全体を落とすため、legacy solutions
    # が無いmacOSではTasksを起動せず、呼び出し側をUNCERTAIN→安全背景へ倒す。
    if sys.platform == "darwin":
        raise RuntimeError(
            "macOSでは不安定なMediaPipe Tasks顔解析を無効化しました"
            "（legacy solutions未搭載）")
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    if not os.path.exists(_TASK_PATH):
        os.makedirs(os.path.dirname(_TASK_PATH), exist_ok=True)
        print("   ⬇️ 顔モデルを初回ダウンロード中（face_landmarker.task）...")
        urllib.request.urlretrieve(_TASK_URL, _TASK_PATH)
    # macOS の Tasks API は delegate 未指定だと Metal(GPU) を選ぶ環境があり、
    # DrishtiMetalHelper の初期化失敗が Python 例外ではなく SIGABRT になる。
    # 例外処理すら通らずアプリ全体が落ちるため、安全判定は明示的に CPU 固定する。
    # CPU は少し遅いが、全フレーム検査を途中で失うより確実性を優先する。
    base = mp_python.BaseOptions(
        model_asset_path=_TASK_PATH,
        delegate=mp_python.BaseOptions.Delegate.CPU)
    opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=4,
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
            return []
        return [[(p.x * w, p.y * h) for p in face]
                for face in res.face_landmarks]
    return extract, "tasks", lmk


def analyze_mouth(video_path, fps=10.0, max_seconds=None,
                  include_visibility=False):
    """動画を解析して (times, mar, face_mask, activity, dur, backend) を返す。

    include_visibility=Trueのときだけ第7要素に三値可視性dictを追加。
    既存の6要素APIは保持する。
    """
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
    visibility = {
        "mouth_state": [], "mouth_visible": [], "mouth_absent": [],
        "primary_mouth_state": [], "primary_mouth_visible": [],
        "primary_mouth_absent": [],
        "face_visibility": [], "face_bbox": [], "mouth_center": [],
        "mouth_size": [], "mouth_bbox": [], "face_count": [],
        "aux_presence": [],
    }
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
                faces = extract(rgb, w, h, int(round(t * 1000.0))) or []
                measurements = [_landmark_visibility(pts, w, h) for pts in faces]
                if any(m["mouth_state"] == MOUTH_CLEAR for m in measurements):
                    aux_presence = None
                elif (not measurements or all(
                        m["mouth_state"] == MOUTH_ABSENT for m in measurements)):
                    aux_presence = _auxiliary_presence(frame, cv2)
                else:
                    aux_presence = None
                state = _aggregate_mouth_observation(measurements, aux_presence)

                # 従来MARは最大の顔を1人選んで継続。可視性判定は全員を見る。
                primary_i = None
                primary_score = -1.0
                for face_i, m in enumerate(measurements):
                    box = m["face_bbox"]
                    score = ((box[2] - box[0]) * (box[3] - box[1])
                             if all(math.isfinite(v) for v in box) else 0.0)
                    if score > primary_score:
                        primary_score = score; primary_i = face_i
                primary = (measurements[primary_i] if primary_i is not None else
                           _landmark_visibility(None, w, h))
                pts = faces[primary_i] if primary_i is not None else None
                if pts and len(pts) > max(UP_INNER, LO_INNER, L_CORNER, R_CORNER):
                    vert = _dist(pts[UP_INNER], pts[LO_INNER])
                    horiz = _dist(pts[L_CORNER], pts[R_CORNER]) + 1e-6
                    times.append(t); mar.append(vert / horiz); face_mask.append(1)
                else:
                    times.append(t); mar.append(float("nan")); face_mask.append(0)
                row = {
                    "mouth_state": state,
                    "mouth_visible": int(state == MOUTH_CLEAR),
                    "mouth_absent": int(state == MOUTH_ABSENT),
                    "primary_mouth_state": primary["mouth_state"],
                    "primary_mouth_visible": int(
                        primary["mouth_state"] == MOUTH_CLEAR),
                    "primary_mouth_absent": int(
                        primary["mouth_state"] == MOUTH_ABSENT),
                    "face_visibility": primary["face_visibility"],
                    "face_bbox": primary["face_bbox"],
                    "mouth_center": primary["mouth_center"],
                    "mouth_size": primary["mouth_size"],
                    "mouth_bbox": primary["mouth_bbox"],
                    "face_count": len(faces),
                    "aux_presence": (-1 if aux_presence is None else
                                     int(bool(aux_presence))),
                }
                for key in visibility:
                    visibility[key].append(row[key])
            fi += 1
    finally:
        cap.release()

    times = np.array(times); mar = np.array(mar); face_mask = np.array(face_mask)
    if len(times) > 1:
        diffs = np.diff(times)
        diffs = diffs[diffs > 1e-6]
        if len(diffs):
            actual_fps = 1.0 / float(np.median(diffs))
    activity = _activity_from_mar(
        mar, face_mask, actual_fps,
        mouth_center=np.asarray(visibility["mouth_center"], dtype=float),
        mouth_size=np.asarray(visibility["mouth_size"], dtype=float),
        face_bbox=np.asarray(visibility["face_bbox"], dtype=float))
    if include_visibility:
        int_keys = {"mouth_state", "mouth_visible", "mouth_absent",
                    "primary_mouth_state", "primary_mouth_visible",
                    "primary_mouth_absent",
                    "face_count", "aux_presence"}
        visibility = {
            key: np.asarray(values, dtype=(np.int8 if key in int_keys else float))
            for key, values in visibility.items()
        }
        return times, mar, face_mask, activity, dur, backend, visibility
    return times, mar, face_mask, activity, dur, backend


def _identity_break_mask(face_mask, mouth_center=None, mouth_size=None,
                         face_bbox=None, center_jump=0.12,
                         size_ratio=1.60, min_bbox_iou=0.10):
    """連続frameが別人/別ショットへ切り替わった可能性を返す。"""
    import numpy as np
    face = np.asarray(face_mask).reshape(-1)
    n = len(face); breaks = np.zeros(n, dtype=bool)
    try:
        center = np.asarray(mouth_center, dtype=float)
        size = np.asarray(mouth_size, dtype=float)
        bbox = np.asarray(face_bbox, dtype=float)
    except (TypeError, ValueError):
        return breaks
    if center.shape != (n, 2) or size.shape != (n, 2) or bbox.shape != (n, 4):
        return breaks

    def iou(a, b):
        ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
        ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(1e-9, aa + bb - inter)

    for i in range(1, n):
        if face[i] <= 0 or face[i - 1] <= 0:
            breaks[i] = True; continue
        vals = np.r_[center[i - 1], center[i], size[i - 1], size[i],
                     bbox[i - 1], bbox[i]]
        if not np.all(np.isfinite(vals)):
            breaks[i] = True; continue
        if float(np.linalg.norm(center[i] - center[i - 1])) > center_jump:
            breaks[i] = True; continue
        w0 = max(1e-9, float(size[i - 1, 0]))
        w1 = max(1e-9, float(size[i, 0]))
        if max(w0 / w1, w1 / w0) > size_ratio:
            breaks[i] = True; continue
        if iou(bbox[i - 1], bbox[i]) < min_bbox_iou:
            breaks[i] = True
    return breaks


def _activity_from_mar(mar, face_mask, fps, mouth_center=None,
                       mouth_size=None, face_bbox=None):
    """口の動き度 = MARの局所標準偏差(±0.5秒窓)。顔なし区間は0。"""
    import numpy as np
    activity = np.zeros_like(mar)
    if len(mar) < 3:
        return activity
    win = max(1, int(round(0.5 * fps)))
    valid = (np.asarray(face_mask) > 0) & np.isfinite(mar)
    identity_breaks = _identity_break_mask(
        face_mask, mouth_center=mouth_center, mouth_size=mouth_size,
        face_bbox=face_bbox)
    p = 0
    while p < len(mar):
        if not valid[p]:
            p += 1; continue
        run_lo = p
        while (p < len(mar) and valid[p]
               and (p == run_lo or not identity_breaks[p])):
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


_MOUTH_CACHE_DIR = os.path.expanduser("~/.dj_video_maker/mouth_profile_cache")
_MOUTH_CACHE_SCHEMA = 3


def _mouth_cache_key(video_path, fps, act_thresh, max_seconds):
    """動画の同一性キー。全読みは重いので サイズ+先頭/末尾1MB のハッシュで代用。
    （同一ファイルの再エンコードは別キーになるが、それは解析し直すのが正しい）"""
    import hashlib
    try:
        size = os.path.getsize(video_path)
        h = hashlib.md5()
        h.update(str(size).encode())
        with open(video_path, "rb") as f:
            h.update(f.read(1024 * 1024))
            if size > 2 * 1024 * 1024:
                f.seek(-1024 * 1024, 2)
                h.update(f.read(1024 * 1024))
        h.update(f"|v{_MOUTH_CACHE_SCHEMA}|fps{fps}|th{act_thresh}|max{max_seconds}".encode())
        return h.hexdigest()[:16]
    except Exception:
        return None


def _mouth_cache_load(key):
    import numpy as np
    try:
        fp = os.path.join(_MOUTH_CACHE_DIR, f"{key}.npz")
        if not os.path.exists(fp):
            return None
        z = np.load(fp, allow_pickle=False)
        times = z["times"]
        n = len(times)
        def optional(name, default, dtype=float, shape=()):
            if name in z:
                value = np.asarray(z[name], dtype=dtype)
                if value.shape == (n,) + shape:
                    return value
            return np.full((n,) + shape, default, dtype=dtype)

        prof = {"times": times, "mar": z["mar"], "activity": z["activity"],
                "face": z["face"], "fps": float(z["fps"]), "thresh": float(z["thresh"]),
                "face_rate": float(z["face_rate"]),
                "backend": str(z["backend"]) if "backend" in z else "cache"}
        # schema 1のキャッシュには不在の明示証拠が無い。
        # すべてUNCERTAIN / mouth_absent=0で読み、選択はfail-closedにする。
        prof.update({
            "mouth_state": optional("mouth_state", MOUTH_UNCERTAIN, np.int8),
            "mouth_visible": optional("mouth_visible", 0, np.int8),
            "mouth_absent": optional("mouth_absent", 0, np.int8),
            "face_visibility": optional("face_visibility", 0.0),
            "face_bbox": optional("face_bbox", float("nan"), shape=(4,)),
            "mouth_center": optional("mouth_center", float("nan"), shape=(2,)),
            "mouth_size": optional("mouth_size", float("nan"), shape=(2,)),
            "mouth_bbox": optional("mouth_bbox", float("nan"), shape=(4,)),
            "face_count": optional("face_count", 0, np.int8),
            "primary_mouth_state": optional(
                "primary_mouth_state", MOUTH_UNCERTAIN, np.int8),
            "primary_mouth_visible": optional(
                "primary_mouth_visible", 0, np.int8),
            "primary_mouth_absent": optional(
                "primary_mouth_absent", 0, np.int8),
            "aux_presence": optional("aux_presence", -1, np.int8),
        })
        prof["mouth_visible_rate"] = float(np.mean(prof["mouth_visible"]))
        prof["mouth_absent_rate"] = float(np.mean(prof["mouth_absent"]))
        return prof
    except Exception:
        return None            # 壊れたキャッシュは無視して再解析


def _mouth_cache_save(key, prof):
    import numpy as np
    try:
        os.makedirs(_MOUTH_CACHE_DIR, exist_ok=True)
        fp = os.path.join(_MOUTH_CACHE_DIR, f"{key}.npz")
        np.savez_compressed(fp, times=prof["times"], mar=prof["mar"],
                            activity=prof["activity"], face=prof["face"],
                            fps=prof["fps"], thresh=prof["thresh"],
                            face_rate=prof["face_rate"], backend=str(prof["backend"]),
                            schema=_MOUTH_CACHE_SCHEMA,
                            mouth_state=prof["mouth_state"],
                            mouth_visible=prof["mouth_visible"],
                            mouth_absent=prof["mouth_absent"],
                            face_visibility=prof["face_visibility"],
                            face_bbox=prof["face_bbox"],
                            mouth_center=prof["mouth_center"],
                            mouth_size=prof["mouth_size"],
                            mouth_bbox=prof["mouth_bbox"],
                            face_count=prof["face_count"],
                            primary_mouth_state=prof["primary_mouth_state"],
                            primary_mouth_visible=prof["primary_mouth_visible"],
                            primary_mouth_absent=prof["primary_mouth_absent"],
                            aux_presence=prof["aux_presence"])
    except Exception:
        pass                   # 保存失敗は無害（次回また解析するだけ）


def build_mouth_profile(video_path, fps=10.0, act_thresh=0.012,
                        max_seconds=None, use_cache=True):
    """MV動画を解析して「各時刻の口パク度」プロファイルを返す。
    戻り: dict { 'times','mar','activity','face','fps','thresh','face_rate','backend' } または None。
    mediapipe/cv2 が無い・解析失敗時は None（＝この機能をスキップして従来通り）。
    ★同じ動画・同じ設定なら結果をキャッシュ再利用する（mediapipe全編解析は重く、
      ハイブリッド局所Proでは同じMVに対して複数回呼ばれるため）。"""
    key = (_mouth_cache_key(str(video_path), fps, act_thresh, max_seconds)
           if use_cache else None)
    if key:
        cached = _mouth_cache_load(key)
        if cached is not None:
            print("     ♻️ 口プロファイルをキャッシュ再利用")
            return cached
    try:
        import numpy as np
        times, mar, face_mask, activity, dur, backend, visibility = analyze_mouth(
            video_path, fps=fps, max_seconds=max_seconds,
            include_visibility=True)
        if len(times) == 0:
            return None
        sample_fps = (1.0 / float(np.median(np.diff(times)))) if len(times) > 1 else float(fps)
        prof = {"times": times, "mar": mar, "activity": activity,
                "face": face_mask, "fps": sample_fps, "thresh": act_thresh,
                "face_rate": float(np.mean(face_mask)), "backend": backend}
        prof.update(visibility)
        prof["mouth_visible_rate"] = float(np.mean(prof["mouth_visible"]))
        prof["mouth_absent_rate"] = float(np.mean(prof["mouth_absent"]))
        if key:
            _mouth_cache_save(key, prof)
        return prof
    except Exception as e:
        print(f"     ⚠️ 口解析スキップ（mouth_sync）: {str(e)[:80]}")
        return None


def mapped_frames_have_verified_lipsync_visual(profile, source_times,
                                               vocal_active,
                                               min_profile_fps=15.0,
                                               max_gap_frames=1.5):
    """表示する全出力frameで「発声中の明瞭に動く口」を認証する。

    ``source_times`` は各出力frameに対応するMV時刻、
    ``vocal_active`` は同じframeのclean-vocal発声判定。次のいずれか
    1つでも欠けたらFalseとし、呼び出し側を非人物背景へ退避させる。

    - step=1で作った全source frame profile
    - 欠損の無い時刻列と対応source frame
    - clean vocal発声中
    - MOUTH_CLEAR / face / mouth_visible
    - 局所口運動 activity >= thresh

    閉口、小さい口、画面端、検出失敗、旧profile、NaNは
    すべて「不明」であり、成功扱いしない。
    """
    try:
        import numpy as np
        if not isinstance(profile, dict) or profile.get("_all_source_frames") is not True:
            return False
        src = np.asarray(source_times, dtype=float).reshape(-1)
        raw_active = np.asarray(vocal_active).reshape(-1)
        if len(src) == 0 or len(src) != len(raw_active):
            return False
        if not np.all(np.isfinite(src)):
            return False
        # NaNをbool変換するとTrueになるため、先に有限性を見る。
        if np.issubdtype(raw_active.dtype, np.number):
            if not np.all(np.isfinite(raw_active.astype(float))):
                return False
        active = raw_active.astype(bool)
        if not bool(np.all(active)):
            return False

        times = np.asarray(profile.get("times", []), dtype=float).reshape(-1)
        state = np.asarray(profile.get("mouth_state", []), dtype=float).reshape(-1)
        primary_state = np.asarray(
            profile.get("primary_mouth_state", []), dtype=float).reshape(-1)
        face = np.asarray(profile.get("face", []), dtype=float).reshape(-1)
        visible = np.asarray(profile.get("mouth_visible", []), dtype=float).reshape(-1)
        primary_visible = np.asarray(
            profile.get("primary_mouth_visible", []), dtype=float).reshape(-1)
        absent = np.asarray(profile.get("mouth_absent", []), dtype=float).reshape(-1)
        activity = np.asarray(profile.get("activity", []), dtype=float).reshape(-1)
        face_count = np.asarray(profile.get("face_count", []), dtype=float).reshape(-1)
        mouth_center = np.asarray(profile.get("mouth_center", []), dtype=float)
        mouth_size = np.asarray(profile.get("mouth_size", []), dtype=float)
        face_bbox = np.asarray(profile.get("face_bbox", []), dtype=float)
        fps = float(profile.get("fps", 0.0))
        thresh = float(profile.get("thresh", float("nan")))
        n = len(times)
        if (n < 2 or any(len(a) != n for a in
                         (state, primary_state, face, visible,
                          primary_visible, absent, activity, face_count))
                or mouth_center.shape != (n, 2)
                or mouth_size.shape != (n, 2)
                or face_bbox.shape != (n, 4)
                or not np.isfinite(fps) or fps < float(min_profile_fps)
                or not np.isfinite(thresh) or thresh <= 0.0
                or not np.all(np.isfinite(times))
                or np.any(np.diff(times) <= 0.0)):
            return False

        dt = float(np.median(np.diff(times)))
        if not np.isfinite(dt) or dt <= 0.0:
            return False
        max_gap = max(float(max_gap_frames) / fps,
                      float(max_gap_frames) * dt)
        # 1枚欠落は両隣が2frame間隔になる。1.5frame未満に
        # 限定し、全frame profileの名前だけで通さない。
        nearest_tol = max(0.75 / fps, 0.75 * dt)
        if (float(np.min(src)) < times[0] - nearest_tol
                or float(np.max(src)) > times[-1] + nearest_tol):
            return False

        right = np.searchsorted(times, src, side="left")
        right = np.clip(right, 0, n - 1)
        left = np.clip(right - 1, 0, n - 1)
        use_left = np.abs(src - times[left]) <= np.abs(times[right] - src)
        idx = np.where(use_left, left, right)
        if np.any(np.abs(src - times[idx]) > nearest_tol + 1e-9):
            return False
        # 対応時刻を挟むsource frame間に欠損があれば不明。
        internal = (right > 0) & (right < n)
        if np.any((times[right[internal]] - times[left[internal]])
                  > max_gap + 1e-9):
            return False

        # ffmpegのfps間引き/丸めで隣接source frameが選ばれても
        # fail-openしないよう、対応時刻の間にある全source frameと
        # 両端の0.75frame guardをまるごと認証する。
        # -ss/fpsのceil-snapは次source frameを選ぶ場合がある。
        # 欠損判定用nearest_tolと分け、表示範囲は両端を1.25frame守る。
        interval_guard = max(1.25 / fps, 1.25 * dt)
        interval_idx = np.flatnonzero(
            (times >= float(np.min(src)) - interval_guard - 1e-9)
            & (times <= float(np.max(src)) + interval_guard + 1e-9))
        if len(interval_idx) == 0:
            return False
        if (len(interval_idx) > 1
                and np.max(np.diff(times[interval_idx])) > max_gap + 1e-9):
            return False
        arrays = (state[interval_idx], primary_state[interval_idx],
                  face[interval_idx], visible[interval_idx],
                  primary_visible[interval_idx], absent[interval_idx],
                  activity[interval_idx], face_count[interval_idx],
                  mouth_center[interval_idx], mouth_size[interval_idx],
                  face_bbox[interval_idx])
        if not all(np.all(np.isfinite(a)) for a in arrays):
            return False
        identity_breaks = _identity_break_mask(
            face, mouth_center=mouth_center, mouth_size=mouth_size,
            face_bbox=face_bbox)
        if np.any(identity_breaks[interval_idx]):
            return False
        return bool(
            np.all(state[interval_idx] == MOUTH_CLEAR)
            and np.all(primary_state[interval_idx] == MOUTH_CLEAR)
            and np.all(face[interval_idx] >= 0.5)
            and np.all(visible[interval_idx] >= 0.5)
            and np.all(primary_visible[interval_idx] >= 0.5)
            and np.all(absent[interval_idx] < 0.5)
            and np.all(activity[interval_idx] >= thresh)
            and np.all(face_count[interval_idx] == 1.0)
        )
    except (TypeError, ValueError, IndexError, OverflowError):
        return False


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


def _profile_mouth_absent(profile, sample_count):
    """明示的ABSENTだけのmask。旧/不正profileは必ずfail-closed。"""
    if not profile or "mouth_absent" not in profile or sample_count <= 0:
        return None
    import numpy as np
    try:
        raw = np.asarray(profile["mouth_absent"], dtype=float)
    except (TypeError, ValueError):
        return None
    if raw.ndim != 1 or len(raw) != sample_count:
        return None
    return np.isfinite(raw) & (raw >= 0.5)


def pick_no_mouth_mv_time(profile, want_dur, mv_dur, avoid=None,
                          avoid_win=4.0, hysteresis_sec=0.25):
    """口元が『明示的に不在』のBロール先頭時刻を返す。

    want_dur全サンプルで mouth_absent=1 が必要。MediaPipe検出失敗、
    顔あり閉口、小さい口、旧キャッシュはUNCERTAINなので選ばない。
    安全区間がなければNoneを返し、呼び出し側が黒画面に退避できる。
    """
    if not profile:
        return None
    import numpy as np
    try:
        want_dur = float(want_dur); mv_dur = float(mv_dur)
        avoid_win = max(0.0, float(avoid_win))
        hysteresis_sec = max(0.0, float(hysteresis_sec))
    except (TypeError, ValueError, OverflowError):
        return None
    if (not math.isfinite(want_dur) or not math.isfinite(mv_dur) or
            want_dur <= 0.0 or mv_dur <= 0.0 or want_dur > mv_dur):
        return None
    try:
        times = np.asarray(profile.get("times", []), dtype=float)
    except (TypeError, ValueError):
        return None
    if (times.ndim != 1 or len(times) < 2 or
            not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0)):
        return None
    absent = _profile_mouth_absent(profile, len(times))
    if absent is None:
        return None
    dt = float(np.median(np.diff(times)))
    try:
        fps = float(profile.get("fps", 1.0 / dt))
    except (TypeError, ValueError, ZeroDivisionError):
        fps = 1.0 / dt
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 1.0 / dt

    # UNCERTAIN/CLEARの周辺はカット境界を含め危険なので膨張。
    unsafe = ~absent
    guard = min(int(math.ceil(hysteresis_sec * fps)), len(times))
    if guard > 0 and np.any(unsafe):
        hits = np.flatnonzero(unsafe)
        starts = np.maximum(0, hits - guard)
        ends = np.minimum(len(unsafe), hits + guard + 1)
        changes = np.zeros(len(unsafe) + 1, dtype=int)
        np.add.at(changes, starts, 1); np.add.at(changes, ends, -1)
        unsafe = np.cumsum(changes[:-1]) > 0
    eligible = ~unsafe

    clean_avoid = []
    values = [] if avoid is None else avoid
    try:
        values = iter(values)
    except TypeError:
        values = iter(())
    for value in values:
        try:
            value = float(value)
            if math.isfinite(value):
                clean_avoid.append(value)
        except (TypeError, ValueError, OverflowError):
            pass

    max_gap = max(0.5, 2.5 * dt)
    best = None; best_score = None
    i = 0; n = len(times)
    while i < n:
        if not eligible[i]:
            i += 1; continue
        run_lo = i; i += 1
        while (i < n and eligible[i] and
               float(times[i] - times[i - 1]) <= max_gap):
            i += 1
        run_hi = i
        run_end = min(mv_dur, float(times[run_hi - 1]) + dt)
        for k in range(run_lo, run_hi):
            t0 = max(0.0, float(times[k])); t1 = t0 + want_dur
            if t1 > run_end + 1e-9 or t1 > mv_dur - 0.05:
                break
            far = min((abs(t0 - a) for a in clean_avoid), default=float("inf"))
            if far < avoid_win:
                continue
            safety = min(t0 - float(times[run_lo]), run_end - t1)
            distance = min(far, 60.0) if math.isfinite(far) else 60.0
            score = (safety, distance, -t0)
            if best_score is None or score > best_score:
                best_score = score; best = t0
    return best


def mouth_open_flux(profile):
    """MARの『正の微分』＝口を開く瞬間の強さ（開きフラックス）を返す → (times, flux)。
    歌う時の口は「開く瞬間」が発声オンセットと物理的に揃うため、
    0.5秒平滑のactivity（動き量）より時間分解能が高く、微小ラグ計測に向く。
    顔が途切れたラン（カット跨ぎ）では微分を計算しない（シーン切替の誤検出防止）。
    取れなければ (None, None)。"""
    import numpy as np
    if not profile:
        return None, None
    times = np.asarray(profile.get("times", []), dtype=float)
    mar = np.asarray(profile.get("mar", []), dtype=float)
    face = np.asarray(profile.get("face", []), dtype=float)
    if len(times) < 4 or len(mar) != len(times):
        return None, None
    flux = np.zeros_like(mar)
    valid = (face > 0) & np.isfinite(mar)
    identity_breaks = _identity_break_mask(
        face, mouth_center=profile.get("mouth_center"),
        mouth_size=profile.get("mouth_size"),
        face_bbox=profile.get("face_bbox"))
    p = 0
    n = len(times)
    while p < n:
        if not valid[p]:
            p += 1; continue
        lo = p
        while (p < n and valid[p]
               and (p == lo or not identity_breaks[p])):
            p += 1
        hi = p
        if hi - lo < 3:
            continue
        seg_t = times[lo:hi]
        seg_m = mar[lo:hi]
        # ラン内の中央値をベースラインとして除去（顔の距離/角度による絶対値差を吸収）
        seg_m = seg_m - float(np.median(seg_m))
        dt = np.diff(seg_t)
        dt[dt <= 1e-6] = 1e-6
        d = np.diff(seg_m) / dt
        flux[lo + 1:hi] = np.maximum(0.0, d)   # 開く方向のみ
    mx = float(np.max(flux))
    if mx > 1e-9:
        flux = flux / mx
    return times, flux


def _parabolic_refine(deltas, scores, ki):
    """離散スコア列のピークを放物線補間してサブステップのラグを返す。
    端や退化時は補間せず離散値をそのまま返す。"""
    import numpy as np
    if ki <= 0 or ki >= len(deltas) - 1:
        return float(deltas[ki])
    y0, y1, y2 = float(scores[ki - 1]), float(scores[ki]), float(scores[ki + 1])
    denom = (y0 - 2.0 * y1 + y2)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return float(deltas[ki])
    off = 0.5 * (y0 - y2) / denom
    off = float(np.clip(off, -1.0, 1.0))       # 1ステップ以内に制限（誤爆防止）
    step = float(deltas[1] - deltas[0]) if len(deltas) > 1 else 0.0
    return float(deltas[ki]) + off * step


def measure_micro_mouth_lag(profile, seg_remix_t, seg_mv_t, onset_times, onset_env,
                            max_lag=0.6, step=0.02, min_face=0.45):
    """『口の開きフラックス vs 歌声オンセット』で区間の残ズレを微計測する。
      seg_remix_t : 区間内のremix時刻（配列, 例0.05s刻み）
      seg_mv_t    : 各remix時刻に対応する現アンカー写像のMV時刻
      onset_times/onset_env : remixボーカルのオンセット強度包絡
    activity(0.5s平滑)×RMSの粗計測と違い、鋭いイベント同士を突き合わせるので
    ±0.6秒内の100ms級の残ズレを高分解能（放物線補間つき）で測れる。
    戻り: (lag秒 or None, corr, face_cov, prom)。lag>0 = MVのもっと後ろを当てるべき。"""
    import numpy as np
    ft, flux = mouth_open_flux(profile)
    if ft is None:
        return None, 0.0, 0.0, 0.0
    seg_remix_t = np.asarray(seg_remix_t, float)
    seg_mv_t = np.asarray(seg_mv_t, float)
    if len(seg_remix_t) < 20:
        return None, 0.0, 0.0, 0.0
    V = np.interp(seg_remix_t, np.asarray(onset_times, float), np.asarray(onset_env, float))
    if float(np.std(V)) < 1e-8:
        return None, 0.0, 0.0, 0.0
    V = (V - V.mean()) / (V.std() + 1e-9)
    face = np.asarray(profile["face"], dtype=float)
    times = np.asarray(profile["times"], dtype=float)

    def face_at(query):
        q = np.asarray(query, dtype=float)
        right = np.clip(np.searchsorted(times, q, side="left"), 0, len(times) - 1)
        left = np.clip(right - 1, 0, len(times) - 1)
        use_left = np.abs(q - times[left]) <= np.abs(times[right] - q)
        return face[np.where(use_left, left, right)]

    deltas = np.arange(-max_lag, max_lag + 1e-9, step)
    scores = np.full(len(deltas), -1e18)
    fcovs = np.zeros(len(deltas))
    for k, d in enumerate(deltas):
        shifted = seg_mv_t + d
        in_range = (shifted >= ft[0]) & (shifted <= ft[-1])
        valid = in_range & (face_at(shifted) >= 0.5)
        fc = float(np.mean(valid))
        fcovs[k] = fc
        if int(np.sum(valid)) < 16:
            continue
        A = np.interp(shifted[valid], ft, flux)
        VV = V[valid]
        if float(np.std(A)) < 1e-8 or float(np.std(VV)) < 1e-8:
            continue
        An = (A - A.mean()) / (A.std() + 1e-9)
        Vn = (VV - VV.mean()) / (VV.std() + 1e-9)
        c = float(np.mean(An * Vn))
        scores[k] = c - 0.15 * max(0.0, min_face - fc)
    fin = scores > -1e17
    if int(np.sum(fin)) < 5:
        return None, 0.0, 0.0, 0.0
    ki = int(np.argmax(scores))
    corr = float(scores[ki] + 0.15 * max(0.0, min_face - fcovs[ki]))  # 生の相関に戻す
    med = float(np.median(scores[fin])); sd = float(np.std(scores[fin])) + 1e-9
    prom = (float(scores[ki]) - med) / sd
    lag = _parabolic_refine(deltas, np.where(fin, scores, med), ki)
    return float(lag), corr, float(fcovs[ki]), float(prom)


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
    all_scores = np.full(len(deltas), np.nan)
    best_k = -1
    for k, d in enumerate(deltas):
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
        all_scores[k] = score
        if score > best_score:
            best_score = score; best_corr = c; best_lag = float(d); best_fcov = fc
            best_k = k
    if best_score <= -2.0:
        return None, 0.0, 0.0
    # 放物線ピーク補間でサブステップ（<0.1s）精度に引き上げる。
    # 隣接スコアが取れていない場合は従来の離散値のまま（挙動互換）。
    if (0 < best_k < len(deltas) - 1
            and np.isfinite(all_scores[best_k - 1]) and np.isfinite(all_scores[best_k + 1])):
        best_lag = _parabolic_refine(deltas, np.nan_to_num(all_scores, nan=-1e18), best_k)
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
    times, mar, face_mask, activity, dur, backend, visibility = analyze_mouth(
        video, fps=fps, max_seconds=max_sec, include_visibility=True)
    if dur: print(f"   長さ≈{dur:.0f}秒 / サンプル数={len(times)}")
    summarize(times, mar, face_mask, activity, dur, backend)

    if csv_path:
        with open(csv_path, "w") as f:
            f.write("time,mar,face,activity,mouth_state,mouth_visible,mouth_absent,"
                    "face_count,face_visibility,face_x0,face_y0,face_x1,face_y1,"
                    "mouth_x,mouth_y,mouth_w,mouth_h\n")
            for i in range(len(times)):
                fb = visibility["face_bbox"][i]
                mc = visibility["mouth_center"][i]
                ms = visibility["mouth_size"][i]
                f.write(
                    f"{times[i]:.3f},{mar[i]:.4f},{int(face_mask[i])},"
                    f"{activity[i]:.4f},{int(visibility['mouth_state'][i])},"
                    f"{int(visibility['mouth_visible'][i])},"
                    f"{int(visibility['mouth_absent'][i])},"
                    f"{int(visibility['face_count'][i])},"
                    f"{visibility['face_visibility'][i]:.4f},"
                    f"{fb[0]:.4f},{fb[1]:.4f},{fb[2]:.4f},{fb[3]:.4f},"
                    f"{mc[0]:.4f},{mc[1]:.4f},{ms[0]:.4f},{ms[1]:.4f}\n")
        print(f"\n📄 CSV書き出し: {csv_path}")


if __name__ == "__main__":
    main()
