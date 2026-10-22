# -*- coding: utf-8 -*-
"""meeting59/实施/探针_交付指标.py —— 交付音频的双口径指标（LTA 与 M-B 帧相对）、F0、句末比、回放。

口径说明见 00_任务简报.md §4：
  · LTA（冻结主口径）：活动帧平均功率 → 7 带 log → 减均值 → 余弦；
  · M-B（诊断口径）：逐帧 7 带相对 dB（每帧减帧均值）→ 帧间平均 → 余弦；
  · F0：本实现自相关（与拟合/qa 各自实现并列登记）；
  · 句末比：句内 voiced run 时长，末段 / 中位；
  · 回放：参考与输出的归一化互相关峰值（应远小）。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEET59 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import 声学工具 as T  # noqa: E402


def 帧相对包络(x: np.ndarray, 门倍: float = 0.20) -> np.ndarray:
    n = max(1, (len(x) - 512) // 160 + 1) if len(x) >= 512 else 0
    env = np.array([T.均方根(x[i * 160:i * 160 + 512]) for i in range(n)])
    门 = max(1e-4, 门倍 * float(np.percentile(env, 85)))
    频 = np.fft.rfftfreq(2048, 1.0 / T.FS)
    掩 = [(频 >= lo) & (频 < hi) for lo, hi in zip(T.带边[:-1], T.带边[1:])]
    累 = np.zeros(len(掩))
    计 = 0
    for i, r in enumerate(env):
        if r < 门:
            continue
        seg = x[i * 160:i * 160 + 512]
        p = np.abs(np.fft.rfft(seg * np.hanning(512), 2048)) ** 2
        v = np.array([np.log(float(np.mean(p[m])) + 1e-12) if np.any(m) else -30.0 for m in 掩])
        累 += v - float(np.mean(v))
        计 += 1
    return 累 / max(计, 1)


def 句末比(x: np.ndarray) -> float | None:
    n = len(x) // 160
    env = np.array([T.均方根(x[i * 160:(i + 1) * 160]) for i in range(n)])
    门 = 0.12 * float(np.percentile(env, 90))
    活 = env > 门
    runs, 起 = [], None
    for i, v in enumerate(活):
        if v and 起 is None:
            起 = i
        elif not v and 起 is not None:
            runs.append(i - 起)
            起 = None
    if 起 is not None:
        runs.append(n - 起)
    runs = [r for r in runs if r >= 6]
    if len(runs) < 4:
        return None
    return float(runs[-1] / float(np.median(runs)))


def 互相关峰值(a: np.ndarray, b: np.ndarray) -> float:
    长 = min(len(a), len(b), 4 * T.FS)
    a = a[:长] - np.mean(a[:长])
    b = b[:长] - np.mean(b[:长])
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    n = 1 << int(np.ceil(np.log2(2 * 长)))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    return float(np.max(np.abs(c)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def 参考音频(名: str) -> np.ndarray:
    目录 = os.path.join(MEET59, "素材", 名)
    ps = sorted(glob.glob(os.path.join(目录, "ref_*.wav")))
    if 名 == "洛天依":
        ps = [p for p in ps if "清唱" in os.path.basename(p)] or ps
    xs = [T.读wav16(p) for p in ps]
    x = np.concatenate([np.concatenate([v, np.zeros(int(0.15 * T.FS))]) for v in xs])
    return T.活动片段(x)


def main() -> int:
    出 = os.path.join(MEET59, "产物", "测试音频")
    print("%-4s | %-8s %-8s | %-8s %-8s | %-7s %-7s | %s"
          % ("角色", "LTA", "M-B", "F0交付", "F0参考", "Δst", "句末比", "回放峰"))
    记录 = {}
    R, O = {}, {}
    for 名 in ("EmoCompanion", "SpeakerA", "洛天依"):
        xr = 参考音频(名)
        w = T.读wav16(os.path.join(出, "角色_%s.wav" % 名))
        R[名] = dict(lta=T.带包络(xr), m=帧相对包络(xr))
        f0r, _, _ = T.基频轨迹(xr)
        f0w, _, _ = T.基频轨迹(w)
        vr, vw = f0r[np.isfinite(f0r)], f0w[np.isfinite(f0w)]
        O[名] = dict(lta=T.带包络(w), m=帧相对包络(w), w=w)
        记录[名] = dict(
            LTA余弦=round(T.带余弦(O[名]["lta"], R[名]["lta"]), 4),
            MB帧相对余弦=round(T.带余弦(O[名]["m"], R[名]["m"]), 4),
            F0交付=round(float(np.median(vw)), 1), F0参考=round(float(np.median(vr)), 1),
            Δst=round(T.半音距(float(np.median(vw)), float(np.median(vr))), 2),
            句末比=(None if 句末比(w) is None else round(句末比(w), 3)),
            回放峰=round(互相关峰值(xr, w), 4),
        )
        d = 记录[名]
        print("%-4s | %-8.4f %-8.4f | %-8.1f %-8.1f | %-7.2f %-7s | %.4f"
              % (名, d["LTA余弦"], d["MB帧相对余弦"], d["F0交付"], d["F0参考"],
                 d["Δst"], d["句末比"], d["回放峰"]))
    ks = list(O)
    print("--- 两两（LTA / M-B）---")
    两两 = {}
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ks[i], ks[j]
            v = dict(交付LTA=round(T.带余弦(O[a]["lta"], O[b]["lta"]), 4),
                     交付MB=round(T.带余弦(O[a]["m"], O[b]["m"]), 4),
                     参考LTA=round(T.带余弦(R[a]["lta"], R[b]["lta"]), 4),
                     参考MB=round(T.带余弦(R[a]["m"], R[b]["m"]), 4))
            两两["%s|%s" % (a, b)] = v
            print("%-12s 交付LTA=%.4f 交付MB=%.4f | 参考LTA=%.4f 参考MB=%.4f"
                  % ("%s|%s" % (a, b), v["交付LTA"], v["交付MB"], v["参考LTA"], v["参考MB"]))
    print("--- 错参考（交付 vs 别角色参考，LTA）---")
    错 = {}
    for 名 in ks:
        for 他 in ks:
            if 他 == 名:
                continue
            c = T.带余弦(O[名]["lta"], R[他]["lta"])
            错["%s→%s" % (名, 他)] = round(c, 4)
            print("%s 的交付 vs %s 参考: LTA=%.4f（自己=%.4f）"
                  % (名, 他, c, 记录[名]["LTA余弦"]))
    记录["两两"] = 两两
    记录["错参考"] = 错
    路 = os.path.join(MEET59, "产物", "指标", "交付指标.json")
    import json
    with open(路, "w", encoding="utf-8") as f:
        json.dump(记录, f, ensure_ascii=False, indent=1)
    print("落盘:", 路)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
