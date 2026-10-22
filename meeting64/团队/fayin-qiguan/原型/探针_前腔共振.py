# -*- coding: utf-8 -*-
"""meeting64/团队/fayin-qiguan/原型/探针_前腔共振.py —— 路线 C 前提的最小探针（纯 DSP、零外挂、C13）。

只回答一个问题（不跑 ASR、不写生成路径、不改既有件）：
    「若把擦音噪声谱从『音韵_v5 手写频带表』换成『部位 → 前腔长度 L → 闭-开管 ABCD 传递 +
      唇辐射 + 文献量级湍流源谱』，这个纯几何模型能不能重现 meeting64 声学诊断的
      谱重心（s 6620 / x 4914 / sh 3338 Hz）？」

模型（lossless 平面波管，可解释、无拟合；ABCD 级联 = 标准声传输线）：
    前腔 = 收紧点→唇的一串 5mm 圆柱段，截面由收紧处 0.15cm² 张到唇 0.8cm²；
    噪声体速度源注入收紧点，唇端活塞辐射阻抗 Z_rad；
    H(f) = p_lip/U_src = Z_rad / (C·Z_rad + D)（C,D = 前腔链 ABCD 的 1,0 与 1,1 元素）。
    源谱固定为文献量级常数（不是拟合出来的）：压力谱 −6dB/oct above 3 kHz（tilt=1, fc=3000），
    只用它把「源」这一半补齐；若把它变成自由参数，本探针就不能再说「几何给了排序」。

判据（先定，不许事后改）：
    P1 排序：s > x > sh（实测排序）。不过 = 单管几何不足以替代部位频带表。
    P2 量级：不拟合相对误差中位 ≤ 40%。
    P3 单标量标定：只许 1 个前腔有效长度倍 β，标定后中位误差 ≤ 20% 且 β∈[0.5,1.5]。
    f（唇齿）预注册为「模型缺口」不参与判据：它的湍流源在口腔外（齿唇间），闭-开管前腔模型不适用。

运行：& "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe" 探针_前腔共振.py
"""
from __future__ import annotations

import math

import numpy as np

FS = 44100.0
C = 343.0                 # 声速 m/s（20°C）
RHO = 1.2                 # 空气密度 kg/m³
段长 = 0.005              # 5mm/段

# 部位 → 唇到收紧点的距离（m）。解剖常识常数（非对本引擎拟合）。
部位_前腔 = {"唇": 0.000, "龈": 0.020, "卷舌": 0.040, "硬腭": 0.055, "软腭": 0.070}
实测 = {"s": 6620.0, "sh": 3338.0, "x": 4914.0, "f": 3708.0}
部位 = {"s": "龈", "sh": "卷舌", "x": "硬腭", "f": "唇"}
# 源谱：文献量级常数（湍流噪声压力谱在 ~3kHz 以上约 −6dB/oct）
源谱_fc, 源谱_tilt = 3000.0, 1.0


def _面积剖面(前腔长: float, 收紧面积: float = 0.15, 唇面积: float = 0.8):
    n段 = max(int(round(max(前腔长, 段长) / 段长)), 1)
    x = np.linspace(0.0, 1.0, n段 + 1)                     # 0=收紧点，1=唇
    A = 收紧面积 + (唇面积 - 收紧面积) * x ** 0.8          # cm²
    return n段, A * 1e-4


def _辐射阻抗(f, S_lip_m2: float):
    """活塞辐射阻抗 Z_rad = ρc/S·[(ka)²/2 + j(8/3π)ka]。"""
    a = math.sqrt(S_lip_m2 / math.pi)
    k = 2.0 * math.pi * np.asarray(f, dtype=np.float64) / C
    ka = k * a
    return (RHO * C / S_lip_m2) * (0.5 * ka ** 2 + 1j * ka * 8.0 / (3.0 * math.pi))


def 传递(f, 前腔长: float, 收紧面积: float = 0.15):
    """返回 H(f) = p_lip / U_src（收紧点体速度源 → 唇辐射声压）。"""
    n段, A = _面积剖面(前腔长, 收紧面积)
    f = np.asarray(f, dtype=np.float64)
    th = 2.0 * np.pi * f / C * 段长
    Zr = _辐射阻抗(f, A[-1])
    M = np.zeros((len(f), 2, 2), dtype=np.complex128)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    for i in range(1, n段 + 1):                            # 逐段左乘，得 [p_src;u_src] = M [p_lip;u_lip]
        Z = RHO * C / A[i]
        Mi = np.zeros((len(f), 2, 2), dtype=np.complex128)
        Mi[:, 0, 0] = np.cos(th)
        Mi[:, 0, 1] = 1j * Z * np.sin(th)
        Mi[:, 1, 0] = 1j * np.sin(th) / Z
        Mi[:, 1, 1] = np.cos(th)
        M = Mi @ M
    C_el, D_el = M[:, 1, 0], M[:, 1, 1]
    return Zr / (C_el * Zr + D_el)


def 谱重心(前腔长: float, 源谱: bool = True, 上限=22050.0):
    f = np.linspace(20.0, FS / 2.0 - 1.0, 8192)
    H = 传递(f, 前腔长)
    P = np.abs(H) ** 2
    if 源谱:
        P = P / (1.0 + (f / 源谱_fc) ** 2) ** 源谱_tilt
    m = f <= 上限
    f, P = f[m], P[m]
    return float(np.sum(f * P) / max(np.sum(P), 1e-30))


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("=" * 78)
    print("探针_前腔共振 ｜ 部位→前腔→ABCD×唇辐射×文献源谱 ｜ 谱重心 Hz")
    print("=" * 78)
    for src in (False, True):
        预 = {p: 谱重心(部位_前腔[部位[p]], 源谱=src) for p in ("s", "sh", "x", "f")}
        print("\n[源谱=%s] " % ("文献量级 fc=3k,−6dB/oct" if src else "理想白源（无滚降）"))
        for p in ("s", "sh", "x", "f"):
            print("  %-3s 前腔=%4.1fcm 预测=%6.0f 实测=%6.0f 比=%5.2f"
                  % (p, 部位_前腔[部位[p]] * 100.0, 预[p], 实测[p], 预[p] / 实测[p]))
        if src:
            预s = 预
    s, sh, x = 预s["s"], 预s["sh"], 预s["x"]
    排序 = s > x > sh
    print("\n[P1 排序] 预测 s=%.0f > x=%.0f > sh=%.0f ⇒ %s ｜ 实测 6620>4914>3338"
          % (s, x, sh, 排序))
    错 = [abs(预s[p] / 实测[p] - 1.0) for p in ("s", "sh", "x")]
    med = float(np.median(错))
    print("[P2 量级] 不拟合相对误差 s/sh/x = %s ｜ 中位 %.2f ⇒ ≤0.40 = %s"
          % ("/".join("%.2f" % e for e in 错), med, med <= 0.40))
    best = None
    for beta in np.linspace(0.5, 1.5, 201):
        e = [math.log(谱重心(部位_前腔[部位[p]] * beta) / 实测[p]) for p in ("s", "sh", "x")]
        rms = float(np.sqrt(np.mean(np.square(e))))
        if best is None or rms < best[1]:
            best = (float(beta), rms)
    beta = best[0]
    残 = [abs(谱重心(部位_前腔[部位[p]] * beta) / 实测[p] - 1.0) for p in ("s", "sh", "x")]
    med3 = float(np.median(残))
    print("[P3 单标量标定] β=%.2f ｜ 标定后 s/sh/x 相对误差 = %s ｜ 中位 %.2f ⇒ 通过=%s（≤0.20 且 β∈[0.5,1.5]）"
          % (beta, "/".join("%.2f" % e for e in 残), med3, bool(med3 <= 0.20 and 0.5 <= beta <= 1.5)))
    print("\n[缺口] f（唇齿）预测 %.0f vs 实测 %.0f：源在口腔外，闭-开管前腔模型不适用（预注册为模型缺口）"
          % (预s.get("f", 0.0), 实测["f"]))
    print("[交叉核对] c/(4L) 解析 1/4 波："
          + "  ".join("%s=%.0f" % (p, C / (4.0 * max(部位_前腔[部位[p]], 1e-4)))
                      for p in ("s", "sh", "x")))
    print("=" * 78)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
