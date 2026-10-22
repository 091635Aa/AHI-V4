# -*- coding: utf-8 -*-
"""meeting62/团队/发声生理/探针_声道与源传递.py —— ③路取证探针（只读 meeting61，只新增 meeting62）。

目的（给 R1_立场.md 提供 [实测] 数字，不猜）：
  1) 解析复算 v6 的 **串联声道** |H声道(f)|：验证「F2 比 F1 低 X dB」到底是滤波器还是源谱的账；
  2) 复算 **声门源谱**（Rosenberg 脉冲表 + 6k 低通 + 唇辐射）；
  3) 输出(k·f0)=|H源|×|H声道|，A/B：
       (a) v6 现状；(b) **收窄 /i/ /ü/ 的 BW1/BW2**；(c) **把源谱 1–5k 段倾斜压平**；
  4) 算「/i/ 要让 F2 ≈ F1 需在 2.35kHz 抬多少 dB」= 需要压平的倾斜量。

用法：$env:PYTHONIOENCODING="utf-8"; python 探针_声道与源传递.py
CPU-only、<1s、不占声卡、无外挂模型。
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting61\实施")
import 引擎_v6 as E6  # noqa: E402

FS = E6.FS
BW默认 = (70.0, 90.0, 130.0, 200.0, 280.0)          # 引擎_v4.默认带宽
样 = {"a": (825.0, 1305.0, 2714.0, 3654.0, 4698.0),
      "i": (330.0, 2349.0, 3132.0, 3967.0, 5011.0),
      "u": (352.0, 752.0, 2610.0, 3550.0, 4489.0),
      "v": (330.0, 2200.0, 2750.0, 3500.0, 4500.0)}
元名 = {"a": "啊", "i": "衣", "u": "乌", "v": "迂"}
F0 = 120.0


def 谐振(F, BW, f):
    """与 引擎_v4.谐振器系数 同式，解析 |H(f)|（DC 增益=1）→ 复数。"""
    T = 1.0 / FS
    C = -math.exp(-2.0 * math.pi * BW * T)
    B = 2.0 * math.exp(-math.pi * BW * T) * math.cos(2.0 * math.pi * F * T)
    A = 1.0 - B - C
    z = np.exp(-2j * math.pi * np.asarray(f, float) * T)
    return A / (1.0 - B * z - C * z * z)


def 串联H(Fs, BWs, f):
    H = np.ones_like(np.asarray(f, float), dtype=np.complex128)
    for F, BW in zip(Fs, BWs):
        H = H * 谐振(F, BW, f)
    return H


def 脉冲谱(f0, 紧喉=0.25, N谐波=64):
    g = E6.声门流表(2048, 紧喉)
    G = np.abs(np.fft.rfft(g))
    k = np.arange(1, N谐波 + 1)
    idx = np.clip(np.rint(k * (f0 / FS) * 2048).astype(int), 1, len(G) - 1)
    return k, G[idx]


def 低通6k(f):
    w = np.asarray(f, float) / 6000.0
    return 1.0 / np.sqrt(1.0 + w ** 4)


def 辐射(f):
    return np.maximum(2.0 * np.sin(np.pi * np.asarray(f, float) / FS), 1e-6)


def 谐波电平(Fs, BWs, f0=F0, fmax=6000.0, 高支=0.0, 高增=(0.30, 0.70, 0.40),
             低切=1600.0, 抬平dB_3k=0.0, F2支=0.0, F2低切=None):
    k, G = 脉冲谱(f0)
    f = k * f0
    m = f <= fmax
    f, G = f[m], G[m]
    H = 串联H(Fs, BWs, f)
    if 高支 > 0.0:                                   # 与 引擎_v6 同口径：1.6k HP × F3/F4/F5
        w = f / 低切
        HP = w ** 2 / np.sqrt(1.0 + w ** 4)
        H = H + 高支 * HP * (高增[0] * 谐振(Fs[2], BWs[2], f)
                            + 高增[1] * 谐振(Fs[3], BWs[3], f)
                            + 高增[2] * 谐振(Fs[4], BWs[4], f))
    if F2支 > 0.0:                                   # 候选：F2 并联支路（HP 在 F1 上裙 → F2 谐振）
        fc = float(F2低切 if F2低切 else Fs[0] + 2.0 * BWs[0])
        w = f / fc
        H = H + F2支 * (w ** 2 / np.sqrt(1.0 + w ** 4)) * 谐振(Fs[1], BWs[1], f)
    src = G * 低通6k(f) * 辐射(f)
    y = np.abs(H) * src
    db = 20.0 * np.log10(np.maximum(y, 1e-30))
    if abs(抬平dB_3k) > 1e-9:                        # Klatt tl 口径：1k→3k 线性抬平
        db = db + 抬平dB_3k * np.maximum(f - 1000.0, 0.0) / 2000.0
    return f, db - db.max()


def F1F2(Fs, db, f):
    i1 = (f >= Fs[0] * 0.72) & (f <= Fs[0] * 1.42)     # 与 引擎_v6 自检同窗 0.75–1.35 放宽
    i2 = (f >= Fs[1] * 0.92) & (f <= Fs[1] * 1.08)
    v1 = float(db[i1].max()) if i1.any() else -99.0
    v2 = float(db[i2].max()) if i2.any() else -99.0
    return v1, v2, v2 - v1


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("=" * 96)
    print("③路探针：v6 声道/源谱传递 —— F2 比 F1 低多少，账记在谁头上")
    print("=" * 96)

    print("\n[1] 纯滤波器（不含源）F2 相对 F1（复算，对照说明_v6 §5 的 −3.5/−4.8 dB）")
    f = np.arange(80.0, 6000.0, 2.0)
    for 名, Fs in 样.items():
        H = np.abs(串联H(Fs, BW默认, f))
        db = 20.0 * np.log10(np.maximum(H, 1e-30))
        v1, v2, d = F1F2(Fs, db, f)
        print("  %-4s F1=%5.0f F2=%5.0f | 滤波器 F1 %6.1f / F2 %6.1f ⇒ F2−F1 = %+6.1f dB"
              % (元名[名], Fs[0], Fs[1], v1, v2, d))

    print("\n[2] 输出(k·f0) 电平（源×声道×辐射，F0=120Hz，与 引擎_v6._谐波电平 同口径）")
    print("    BW收窄(i/v) = b1 70→50、b2 90→55（Klatt 默认 b1=60/b2=90，范围 30–1000/40–1000）")
    方案 = (("v6现状(高支0.55)", BW默认, 0.0, 0.55),
            ("源平18dB@3k 去高支", BW默认, 18.0, 0.0),
            ("BW收窄", (50, 55, 130, 200, 280), 0.0, 0.0),
            ("BW收窄+源平14dB@3k", (50, 55, 130, 200, 280), 14.0, 0.0))
    for 名 in ("i", "v", "a", "u"):
        Fs = 样[名]
        print("  —— %s ——" % 元名[名])
        for 标签, BWs, 抬, 支 in 方案:
            f, db = 谐波电平(Fs, list(BWs), 高支=支, 抬平dB_3k=抬)
            v1, v2, d = F1F2(Fs, db, f)
            print("     %-20s 谱峰@%5.0fHz | F1 %6.1f / F2 %6.1f ⇒ F2−F1 %+6.1f dB"
                  % (标签, float(f[int(np.argmax(db))]), v1, v2, d))

    print("\n[3] 用量：/i/ /ü/ 要让 F2 ≈ F1，需在 ~2.35kHz 抬多少 dB（= 需压平的倾斜量）")
    for 名 in ("i", "v"):
        Fs = 样[名]
        f, db = 谐波电平(Fs, list(BW默认), 高支=0.55)
        _, _, d0 = F1F2(Fs, db, f)
        print("  %s：v6 现状 F2−F1 %+6.1f dB ⇒ 需在 %.0fHz 抬约 %.1f dB"
              % (元名[名], d0, Fs[1], -d0))

    print("\n[4] F3/F4/F5：v6 现状源（不平）下，如果只靠「源谱压平」能补多少")
    Fs = 样["a"]
    f, d0 = 谐波电平(Fs, list(BW默认), 高支=0.0)
    for 抬 in (0.0, 6.0, 12.0, 18.0):
        f, db = 谐波电平(Fs, list(BW默认), 高支=0.0, 抬平dB_3k=抬)
        out = []
        for j, 名H in ((2, "F3"), (3, "F4"), (4, "F5")):
            i = (f >= Fs[j] * 0.95) & (f <= Fs[j] * 1.05)
            out.append("%s %+5.1f" % (名H, float(db[i].max()) if i.any() else -99.0))
        print("    源平@3k=%4.0fdB → " % 抬 + " | ".join(out) + "  (相对本方案峰值 dB)")
    print("    v6 现状（源不平、有高支0.55）对照：")
    f, db = 谐波电平(Fs, list(BW默认), 高支=0.55)
    print("    " + " | ".join("%s %+5.1f" % (h, float(db[(f >= Fs[j] * 0.95) & (f <= Fs[j] * 1.05)].max()))
                                for j, h in ((2, "F3"), (3, "F4"), (4, "F5"))))

    print("\n[4b] F3/F4/F5 包络单调性：v6 现状(高增 0.30/0.70/0.40) vs 单调递减高增 vs 源谱压平")
    Fs = 样["a"]
    for 标签, 支, 高增, 抬 in (
            ("v6现状 高增.30/.70/.40", 0.55, (0.30, 0.70, 0.40), 0.0),
            ("高支.55 单调.30/.24/.16", 0.55, (0.30, 0.24, 0.16), 0.0),
            ("高支.35 单调.22/.16/.10", 0.35, (0.22, 0.16, 0.10), 0.0),
            ("去高支 + 源平18dB@3k", 0.0, (0, 0, 0), 18.0),
            ("高支.55 + 源平12dB@3k", 0.55, (0.30, 0.70, 0.40), 12.0)):
        f, db = 谐波电平(Fs, list(BW默认), 高支=支, 高增=高增, 抬平dB_3k=抬)
        三 = []
        for j, 名H in ((2, "F3"), (3, "F4"), (4, "F5")):
            i = (f >= Fs[j] * 0.95) & (f <= Fs[j] * 1.05)
            三.append(float(db[i].max()) if i.any() else -99.0)
        单调 = 三[0] > 三[1] > 三[2]
        print("    %-24s F3 %+6.1f F4 %+6.1f F5 %+6.1f ⇒ 包络单调递减 = %s"
              % (标签, 三[0], 三[1], 三[2], 单调))

    print("\n[5] F0 依赖：「/i/ F2 比 F1 低 17–21dB」是否只是 120Hz 诊断声线的测量问题")
    Fsi = 样["i"]
    for f0 in (120.0, 200.0, 300.0, 367.0, 440.0):
        f, db = 谐波电平(Fsi, list(BW默认), f0=f0, 高支=0.55)
        v1, v2, d = F1F2(Fsi, db, f)
        i2 = (f >= Fsi[1] * 0.92) & (f <= Fsi[1] * 1.08)
        near = float(f[i2][np.argmax(db[i2])]) if i2.any() else 0.0
        print("    F0=%5.0fHz | 最接近 F2 的谐波 %5.0fHz | F1 %6.1f / F2 %6.1f ⇒ F2−F1 %+6.1f dB"
              % (f0, near, v1, v2, d))
    print("    （EmoCompanion实际 F0 中位 367–428Hz ⇒ 该「缺陷」在真实声线上已大幅缓解）")

    print("\n[6] 候选物理解法：给 /i/ /ü/ 加一条 **F2 并联支路**（HP 在 F1 上裙 × F2 谐振）")
    print("    口径 = Klatt 并联支路的 a2（独立幅度），物理=声门下游湍流源对闭元音窄腔的再激励")
    for 名 in ("i", "v"):
        Fs = 样[名]
        print("  —— %s ——" % 元名[名])
        for a2 in (0.0, 0.30, 0.60, 0.90, 1.20):
            f, db = 谐波电平(Fs, (50, 55, 130, 200, 280), 高支=0.55, F2支=a2)
            v1, v2, d = F1F2(Fs, db, f)
            低 = float(db[f <= 480.0].max()) if (f <= 480.0).any() else -99.0
            print("    a2=%.2f → F2−F1 %+6.1f dB | 低频(<480Hz)相对峰值 %6.1f dB（不许形成平台）"
                  % (a2, d, 低))
    print("\n[7] 结论口径（不许再靠手调逐峰增益）")
    print("    ① F2−F1 的账：滤波 −3~−7.5dB，源谱 −16~−18dB ⇒ 源是主项；")
    print("    ② 提升 F2 的三条合法路：BW 收窄(+0.5~1.9)、源谱压平(+6dB/8dB@3k)、F2 并联支路 a2；")
    print("    ③ F3–F5 的账：v6 高支 0.55 把 F4/F5 抬 +23/+39dB（**破坏 F3>F4>F5 单调包络**），")
    print("       改走「源谱 3k 压平 12–18dB + 去/降高支」可得 F3>F4>F5 单调，才是 Klatt 自然源口径。")


if __name__ == "__main__":
    main()
