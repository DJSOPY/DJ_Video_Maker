#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""口マイクロ補正の合成テスト。
既知のズレ(±ms)を仕込んだ「口プロファイル＋歌声オンセット」を作り、
measure_micro_mouth_lag / apply_mouth_micro_lag が正しく検出・補正するか検証する。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mouth_sync as ms

rng = np.random.default_rng(7)

def make_synthetic(true_lag=0.25, dur=40.0, fps=10.0, face_rate=1.0, noise=0.15):
    """MVの口の開閉(MAR)と、それに true_lag 秒だけズレて対応する remix 歌声オンセットを作る。
    仕様: lag>0 = 「MVのもっと後ろを当てるべき」= 現マッピングだと映像が手前すぎ
        = MV時刻 t の口イベントは remix時刻 (t - true_lag) の声イベントに対応する。"""
    t = np.arange(0, dur, 1.0 / fps)
    # 音節イベント（歌の発声タイミング）を remix 時間軸で生成
    ev = np.cumsum(rng.uniform(0.25, 0.55, size=200))
    ev = ev[ev < dur - 1.0]
    # remix 声オンセット包絡（20ms解像度）
    ot = np.arange(0, dur, 0.02)
    env = np.zeros_like(ot)
    for e in ev:
        env += np.exp(-0.5 * ((ot - e) / 0.03) ** 2)
    env += noise * rng.random(len(ot))
    env /= env.max()
    # MVのMAR: 各声イベントに対応して口が開く。MV時間軸では (event + true_lag) の位置。
    mar = 0.15 + np.zeros_like(t)
    for e in ev:
        mar += 0.25 * np.exp(-0.5 * ((t - (e + true_lag)) / 0.06) ** 2)
    mar += 0.01 * rng.standard_normal(len(t))
    face = (rng.random(len(t)) < face_rate).astype(int)
    mar_out = mar.copy(); mar_out[face == 0] = np.nan
    prof = {"times": t, "mar": mar_out, "face": face,
            "activity": np.zeros_like(t), "fps": fps, "thresh": 0.012,
            "face_rate": float(face.mean()), "backend": "synthetic"}
    return prof, ot, env

def run_case(true_lag, face_rate=1.0, noise=0.15, label=""):
    prof, ot, env = make_synthetic(true_lag=true_lag, face_rate=face_rate, noise=noise)
    dur = 40.0
    srt = np.arange(2.0, dur - 2.0, 0.05)
    smv = srt.copy()   # 現マッピング＝恒等（ズレはMV側に仕込み済み）
    lag, corr, fcov, prom = ms.measure_micro_mouth_lag(prof, srt, smv, ot, env,
                                                        max_lag=0.6, step=0.02)
    err = (lag - true_lag) * 1000 if lag is not None else float("nan")
    status = "OK " if (lag is not None and abs(lag - true_lag) < 0.035) else "NG "
    print(f"  {status}{label:28s} 真値{true_lag*1000:+5.0f}ms → 検出"
          f"{'None' if lag is None else f'{lag*1000:+5.0f}ms'}"
          f" (誤差{err:+.0f}ms, 相関{corr:.2f}, 顔{fcov*100:.0f}%, 突出{prom:.1f})")
    return lag, corr, prom

print("=== measure_micro_mouth_lag 検出精度 ===")
run_case(+0.25, label="+250ms・顔100%")
run_case(-0.18, label="-180ms・顔100%")
run_case(+0.10, label="+100ms・顔100%")
run_case(+0.30, face_rate=0.6, label="+300ms・顔60%")
run_case(0.00, label="ズレなし(誤爆しないか)")
run_case(+0.40, noise=0.5, label="+400ms・強ノイズ")

print()
print("=== 顔が無いMVでゲートされるか ===")
prof, ot, env = make_synthetic(true_lag=0.3, face_rate=0.05)
srt = np.arange(2.0, 38.0, 0.05)
lag, corr, fcov, prom = ms.measure_micro_mouth_lag(prof, srt, srt, ot, env)
print(f"  顔5%: lag={lag} corr={corr:.2f} fcov={fcov:.2f} → "
      + ("OK（低顔率が数値に反映）" if (lag is None or fcov < 0.45) else "NG"))

print()
print("=== 無関係な信号(口と声が無相関)で誤爆しないか ===")
prof, ot, env = make_synthetic(true_lag=0.0)
env2 = rng.random(len(ot))   # デタラメな声
lag, corr, fcov, prom = ms.measure_micro_mouth_lag(prof, srt, srt, ot, env2)
print(f"  無相関: lag={'None' if lag is None else f'{lag*1000:+.0f}ms'} "
      f"corr={corr:.2f} prom={prom:.1f} → "
      + ("OK（相関/突出が低い＝ゲートで弾ける）" if corr < 0.32 or prom < 2.0 else "NG"))

print()
print("=== 粗計測(measure_segment_mouth_lag): フレーズ構造で真値+1230ms ===")
# activity(±0.5s平滑)はフレーズ単位の歌/休み構造を測る設計 → 歌4s/休み2sブロックで検証
dur2, fps, lagc = 60.0, 10.0, 1.23
t2 = np.arange(0, dur2, 1 / fps)
vt = np.arange(0, dur2, 0.05)
venv = np.zeros_like(vt)
s = 1.0; blocks = []
while s < dur2 - 5:
    e = s + rng.uniform(3.0, 5.0)
    blocks.append((s, min(e, dur2 - 1))); s = e + rng.uniform(1.5, 2.5)
for (a, b) in blocks:
    m = (vt >= a) & (vt < b)
    venv[m] = 0.7 + 0.3 * rng.random(int(m.sum()))
mar2 = 0.15 + 0.005 * rng.standard_normal(len(t2))
for (a, b) in blocks:
    m = (t2 >= a + lagc) & (t2 < b + lagc)
    mar2[m] += 0.12 * np.abs(np.sin(2 * np.pi * 3.0 * t2[m])) + 0.02 * rng.standard_normal(int(m.sum()))
face2 = np.ones(len(t2), dtype=int)
prof2 = {"times": t2, "mar": mar2, "face": face2, "fps": fps, "thresh": 0.012,
         "face_rate": 1.0, "backend": "synthetic"}
prof2["activity"] = ms._activity_from_mar(mar2, face2, fps)
srt2 = np.arange(2.0, dur2 - 3.0, 0.1)
got, corr, fcov = ms.measure_segment_mouth_lag(prof2, srt2, srt2, vt, venv,
                                               max_lag=4.0, ret_always=True)
# 粗計測は±0.5s平滑のフレーズ級ステージ。役割は「数秒ズレの引き戻し」で、
# 精度目標は ~±120ms（残りはマイクロ補正が仕上げる）。
print(f"  検出{got*1000:+.0f}ms (誤差{(got-lagc)*1000:+.0f}ms, 相関{corr:.2f})"
      + ("  OK（±120ms＝粗ステージの役割どおり）" if abs(got - lagc) < 0.12 else "  NG"))

print()
print("=== apply_mouth_micro_lag 統合テスト(lipsync_pro側・本番librosa経路) ===")
# 検出の意味論：口を「開き始める瞬間」(フラックスのピーク=MAR山の中心-σ)と
# 発声アタック(オンセット強度ピーク)を揃える。よってMAR山の中心を e+lag+σ に置く。
import lipsync_pro as lp
sr = 22050
true_lag = 0.22; dur = 40.0; sigma = 0.06
tt = np.arange(0, dur, 1.0 / sr)
ev = np.cumsum(rng.uniform(0.25, 0.55, size=200)); ev = ev[ev < dur - 1.0]
rvoc = np.zeros_like(tt)
for e in ev:
    m = (tt >= e) & (tt < e + 0.15)
    n = int(m.sum())
    if n:
        rvoc[m] += np.sin(2 * np.pi * 220 * (tt[m] - e)) * np.hanning(n)
rvoc += 0.01 * rng.standard_normal(len(tt))
mt = np.arange(0, dur, 0.1)
mar = 0.15 + np.zeros_like(mt)
for e in ev:
    mar += 0.25 * np.exp(-0.5 * ((mt - (e + true_lag + sigma)) / sigma) ** 2)
prof = {"times": mt, "mar": mar, "face": np.ones(len(mt), int),
        "activity": np.zeros_like(mt), "fps": 10.0, "thresh": 0.012,
        "face_rate": 1.0, "backend": "synthetic"}
anchors = [(float(r), float(r), 0.1) for r in np.arange(1.0, dur - 1.0, 2.5)]
fixed, nfix = lp.apply_mouth_micro_lag(anchors, prof, rvoc.astype(np.float32), sr)
shift = np.mean([f[1] - a[1] for f, a in zip(fixed, anchors)])
print(f"  補正区間={nfix}, 平均シフト={shift*1000:+.0f}ms（期待≈+{true_lag*1000:.0f}ms）"
      + ("  OK" if nfix >= 1 and abs(shift - true_lag) < 0.05 else "  NG"))
