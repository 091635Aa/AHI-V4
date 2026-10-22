# -*- coding: utf-8 -*-
"""meeting62/实施/引擎_v7.py —— v6 基础上落地「发声生理三项修」的新引擎（纯新增，C14）。

只读 import `meeting61/实施/引擎_v6.py`、`引擎_v4.py`；**不改 v6 一字**。
本件相对 引擎_v6 的改动（全部落在**浊音道**；清辅音/噪声道逐位同 v6 ⇒ C14 护栏）：

  修① 幅度颤音 → 去掉：`_微颤深` v6=0.012（乘在源周期波形上＝幅度颤音）→ v7=0.0。
       颤音改由 `韵律_v7` 放在 **f0 序列**上（见 韵律_v7.颤音半音 等字段）⇒ 得到真正的 F0 颤音。
  修② `_高频支路增` 由 v6 的 (0.30, 0.70, 0.40)（**F4>F3 非单调**）改为单调递减
       (0.30, 0.24, 0.16) ⇒ 串联高次峰 F3>F4>F5 单调。
  修③ Klatt **源谱平坦化 `tl`**（可选）：`级联入` 的源改用 `源滤 + a·源高`（a 由 `_源平坦dB3k`
       决定，1.6k 以上抬升 tl dB@3k），并按解析增益补偿使浊音 RMS 不漂移。
       默认 `_源平坦dB3k = 0.0`（关闭=逐位同 v6）；是否采纳由实测决定（破 F1 突出度 ⇒ 不采纳）。
  可选④ 鼻零极点：帧可带 `鼻极/鼻零/鼻增益`（`_鼻增益默认=0` ⇒ 关闭，逐位同 v6）。

接口与 v6 相同：``合成(帧, 音色=None, 种子=300) -> (波形, 元信息)``。
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy.signal import butter, freqz, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting61", "实施"))

import 引擎_v6 as E6  # noqa: E402  只读
import 引擎_v4 as E4  # noqa: E402  只读（常量/表/工具）

FS = E4.FS
帧率 = E4.帧率
每帧样本 = E4.每帧样本
产物临时 = E4.产物临时

# —— 标定（v7 沿用 v6 值；修①②③各自相对 v6 的改动在下方注明）——
_源标定 = 0.3065                      # 修②配套补偿：高支单调化(0.30/0.24/0.16)使浊音电平低 0.07dB
                                      # ⇒ 由 v6 的 0.304 提到 0.3065，使 E2 由 −3.96 回 −4.04（与 v6 −4.03 持平）
_气声源权重 = 0.012
_气声直通权重 = 0.010
_噪标定 = 0.020
_高频支路 = 0.55                      # 与 v6 同（不动高支总平衡 ⇒ /e/ 谱包络峰仍落 F1 窗内）
_高频低切Hz = 1600.0
_高频支路增 = (0.30, 0.24, 0.16)        # 修②：单调（v6=(0.30,0.70,0.40)）
_声源低通Hz = 6000.0
_声源低通阶 = 2
_微颤Hz = 4.7
_微颤深 = 0.0                            # 修①：幅度颤音置 0（搬到 F0）
_并联下限 = 400.0
_默认共振峰增益 = (1.0, 0.55, 0.20, 0.09, 0.04)

# 修③ Klatt 源谱平坦化 tl：1.6kHz 以上按 a=10^(dB/20)-1 抬升；补偿用解析增益
_源平坦dB3k = 0.0                        # 0=关闭（逐位同 v6）；采纳值由实测决定
_源平坦高切Hz = 1600.0
_鼻极Hz默认 = 0.0                        # 可选④：0=关闭
_鼻零Hz默认 = 0.0
_鼻增益默认 = 0.0

_表栈 = E4._表栈
_表层数 = E4.表层数
_表长 = E4.表长
_谐振器系数 = E4.谐振器系数
_滤波块 = E4._滤波块
_帧数组 = E4._帧数组
_平滑 = E4._平滑
声门流表 = E6.声门流表
_流栈 = E6._流栈
_辐射增益 = E6._辐射增益


def 平坦系数(dB3k: float = None):
    """源谱平坦化参数：返回 (a, 增益补偿)。a=10^(dB/20)-1；补偿=1/√(1+2a·m1+a²·m2)。"""
    dB = _源平坦dB3k if dB3k is None else float(dB3k)
    a = 10.0 ** (float(dB) / 20.0) - 1.0
    if a <= 1e-9:
        return 0.0, 1.0
    _bh, _ah = butter(2, _源平坦高切Hz / (FS * 0.5), btype="high")
    _w, H = freqz(_bh, _ah, worN=2048, fs=FS)
    Hm = np.abs(H)
    m1 = float(np.mean(Hm))
    m2 = float(np.mean(Hm * Hm))
    补 = 1.0 / math.sqrt(max(1.0 + 2.0 * a * m1 + a * a * m2, 1e-9))
    return float(a), float(补)


# 预计算（模块加载时一次，确定性）
_平坦a, _平坦补 = 平坦系数()


def 设源平坦(dB3k: float):
    """运行时设定 Klatt tl（dB@3k）；返回 (a, 补偿)。用于标定/对照，确定性。"""
    global _源平坦dB3k, _平坦a, _平坦补
    _源平坦dB3k = float(dB3k)
    _平坦a, _平坦补 = 平坦系数()
    return _平坦a, _平坦补


def 设高频支路增(三值):
    """运行时设定 F3/F4/F5 高支相对幅度（用于对照），确定性。"""
    global _高频支路增
    _高频支路增 = tuple(float(x) for x in 三值)
    return _高频支路增


def _反谐振系数(fz: float, bw: float):
    """二阶反谐振（近纯零点）分子 1−2r cosθ z⁻¹+r²z⁻²，DC 归一（r→1 得近纯零）。"""
    fz = max(float(fz), 40.0)
    bw = max(float(bw), 20.0)
    r = math.exp(-math.pi * bw / FS)
    th = 2.0 * math.pi * fz / FS
    b2 = 1.0
    b1 = -2.0 * r * math.cos(th)
    b0 = r * r
    s = b0 + b1 + b2
    if abs(s) < 1e-12:
        s = 1.0
    return b0 / s, b1 / s, b2 / s


def 合成(帧: dict, 音色: dict = None, 种子: int = 300):
    """与 引擎_v6.合成 同接口；改动只在浊音道（见模块 docstring）。"""
    t0 = time.perf_counter()
    音 = E4._规整音色(音色)
    if not 音色 or "共振峰增益" not in 音色:
        音["共振峰增益"] = _默认共振峰增益
    N = len(帧.get("f0") or [])
    if N == 0:
        return np.zeros(0, dtype=np.float64), dict(
            采样率=FS, 帧率=帧率, 每帧样本=每帧样本, 样本数=0, 帧数=0,
            时长秒=0.0, 峰值=0.0, 峰值缩放=False, 音色指纹=音["_指纹"],
            高频支路dB=音["高频支路dB"], 合成耗时秒=0.0, 实时率=0.0)

    总样本 = N * 每帧样本
    帧轴 = np.arange(N, dtype=np.float64)
    样_idx = np.arange(总样本, dtype=np.float64) / 每帧样本

    def 插(a):
        return np.interp(样_idx, 帧轴, a)

    f0帧 = np.clip(_帧数组(帧, "f0", N, 0.0), 0.0, FS * 0.45)
    幅帧 = np.clip(_帧数组(帧, "幅", N, 0.5), 0.0, 1.5)
    气帧 = np.clip(_帧数组(帧, "气声", N, 0.0), 0.0, 1.0)
    送帧 = np.clip(_帧数组(帧, "送气", N, 0.0), 0.0, 1.0)
    紧帧 = np.clip(_帧数组(帧, "紧喉", N, 0.3), 0.0, 1.0)
    粗帧 = np.clip(_帧数组(帧, "粗糙", N, 0.0), 0.0, 1.0)
    源帧 = np.rint(_帧数组(帧, "声源", N, 1.0)).astype(np.int64)
    源帧 = np.where(源帧 > 0, 1, np.where(源帧 < 0, -1, 0))
    噪形帧 = np.rint(_帧数组(帧, "噪形", N, 0.0)).astype(np.int64)
    噪形帧 = np.where(噪形帧 > 0, 1, 0)
    # 可选④：鼻零极点（默认 0 ⇒ 关闭）
    鼻极帧 = np.clip(_帧数组(帧, "鼻极", N, _鼻极Hz默认), 0.0, 0.45 * FS)
    鼻零帧 = np.clip(_帧数组(帧, "鼻零", N, _鼻零Hz默认), 0.0, 0.45 * FS)
    鼻g帧 = np.clip(_帧数组(帧, "鼻增益", N, _鼻增益默认), 0.0, 1.0)

    缺省f = (800.0, 1200.0, 2500.0, 3500.0, 4500.0)
    F帧 = np.empty((5, N), dtype=np.float64)
    for j, 键 in enumerate(("f1", "f2", "f3", "f4", "f5")):
        F帧[j] = _帧数组(帧, 键, N, 缺省f[j])
    BW帧 = np.empty((5, N), dtype=np.float64)
    for j in range(5):
        BW帧[j] = E4.默认带宽[j]
    sF = np.asarray(音["共振峰缩放"], dtype=np.float64)[:, None]
    sB = np.asarray(音["带宽缩放"], dtype=np.float64)[:, None]
    F帧 = np.clip(F帧 * sF, 40.0, FS * 0.5 - 250.0)
    BW帧 = np.clip(BW帧 * sB, 20.0, FS * 0.5 - 250.0)
    if N > 1:
        for j in range(5):
            F帧[j] = _平滑(F帧[j])
    F帧 = np.clip(F帧, 40.0, FS * 0.5 - 250.0)

    噪带s = 帧.get("噪带")
    if 噪带s is None:
        噪带列表 = [None] * N
    else:
        噪带列表 = list(噪带s)
        if len(噪带列表) == 1:
            噪带列表 = 噪带列表 * N
        elif len(噪带列表) != N:
            raise ValueError("噪带长度 %d ≠ f0 长度 %d" % (len(噪带列表), N))

    rng = np.random.default_rng(int(种子))
    白气 = rng.standard_normal(总样本)
    白帧 = rng.standard_normal(总样本)
    _bln, _aln = butter(2, min(7500.0, FS * 0.45) / (FS * 0.5), btype="low")
    帧带 = lfilter(_bln, _aln, 白帧)
    _rn = float(np.std(帧带)) or 1.0
    帧带 = 帧带 / _rn
    _bl, _al = butter(2, min(6000.0, FS * 0.45) / (FS * 0.5), btype="low")
    气带 = lfilter(_bl, _al, 白气)
    _bh, _ah = butter(2, 180.0 / (FS * 0.5), btype="high")
    气带 = lfilter(_bh, _ah, 气带)
    _r = float(np.std(气带)) or 1.0
    气带 = 气带 / _r

    气声样 = 插(气帧) * 音["气声增益"]
    送气样 = 插(送帧) * 音["送气增益"]
    紧喉样 = 插(紧帧)
    粗糙样 = 插(粗帧) * 音["抖动增益"]
    幅样 = 插(幅帧)

    # —— 声门源（与 v5/v6 逐位同：查表脉冲 + 抖动/次谐波/幅度颤音）——
    抖动 = 1.0 + 粗糙样 * 0.025 * (
        np.sin(2.0 * math.pi * 13.7 * np.arange(总样本) / FS + 0.5)
        + 0.6 * np.sin(2.0 * math.pi * 31.3 * np.arange(总样本) / FS + 1.9))
    f0样 = 插(f0帧) * 抖动
    相位 = np.mod(np.cumsum(f0样 / FS), 1.0)
    相index = np.minimum((相位 * _表长).astype(np.int64), _表长 - 1)
    层index = np.clip(np.rint(紧喉样 * (_表层数 - 1)).astype(np.int64), 0, _表层数 - 1)
    源周期 = _流栈[层index, 相index]
    if 音["次谐波增益"] > 0.0 and float(np.max(粗糙样)) > 1e-6:
        相位_sub = np.mod(np.cumsum(f0样 * 0.5 / FS), 1.0)
        相index_sub = np.minimum((相位_sub * _表长).astype(np.int64), _表长 - 1)
        源周期 = 源周期 + (0.32 * 粗糙样 * 音["次谐波增益"]) * _流栈[层index, 相index_sub]
    源周期 = 源周期 * (1.0 + 0.15 * 粗糙样 * np.sin(2.0 * math.pi * 33.0
                                                      * np.arange(总样本) / FS + 0.7))
    if _微颤深 > 0.0:      # 修①：v7 默认 0 ⇒ 此行不执行
        源周期 = 源周期 * (1.0 + _微颤深 * np.sin(2.0 * math.pi * _微颤Hz
                                                  * np.arange(总样本) / FS + 1.3))
    _b, _a = butter(int(_声源低通阶), min(_声源低通Hz, FS * 0.45) / (FS * 0.5), btype="low")
    源滤 = lfilter(_b, _a, 源周期)
    # F4/F5 高通并联支路的激励（直流处增益=0 ⇒ 不会造成低频平台；与 v6 逐位同）
    _bh2, _ah2 = butter(2, _高频低切Hz / (FS * 0.5), btype="high")
    源高 = lfilter(_bh2, _ah2, 源滤)
    # 修③：级联源谱平坦化（tl）——只作用于**串联级联入**；高支入保持 v6 口径
    if _平坦a > 1e-9:
        源平坦 = (源滤 + _平坦a * 源高) * _平坦补
    else:
        源平坦 = 源滤
    噪混 = 气带 * (气声样 * _气声源权重 + 送气样 * _气声源权重)
    级联入 = (源平坦 + 噪混) * _源标定 * 幅样
    高支入 = 源高 * _源标定 * _高频支路 * 幅样
    g_气直通 = (气声样 * _气声直通权重 + 送气样 * _气声直通权重) * 音["噪声增益"]

    预 = np.zeros(总样本, dtype=np.float64)
    噪预 = np.zeros(总样本, dtype=np.float64)
    噪增益帧 = np.ones(N, dtype=np.float64)
    串zi = [np.zeros(2) for _ in range(5)]
    高3zi = np.zeros(2)
    高4zi = np.zeros(2)
    高5zi = np.zeros(2)
    鼻极zi = np.zeros(2)
    鼻零zi = [np.zeros(2), np.zeros(2)]
    带zi1 = np.zeros(2)
    带zi2 = np.zeros(2)
    形zi = [np.zeros(2) for _ in range(3)]

    for fi in range(N):
        a, b = fi * 每帧样本, min((fi + 1) * 每帧样本, 总样本)
        if a >= b:
            break
        typ = int(源帧[fi])
        if typ == 1:
            y = 级联入[a:b]
            for j in range(5):
                A, B, C = _谐振器系数(F帧[j, fi], BW帧[j, fi])
                y, 串zi[j] = _滤波块(A, B, C, y, 串zi[j])
            x4 = 高支入[a:b]
            y3, 高3zi = _滤波块(*_谐振器系数(F帧[2, fi], BW帧[2, fi]), x4, 高3zi)
            y4, 高4zi = _滤波块(*_谐振器系数(F帧[3, fi], BW帧[3, fi]), x4, 高4zi)
            y5, 高5zi = _滤波块(*_谐振器系数(F帧[4, fi], BW帧[4, fi]), x4, 高5zi)
            出 = (y + _高频支路增[0] * y3 + _高频支路增[1] * y4
                  + _高频支路增[2] * y5 + g_气直通[a:b] * 气带[a:b])
            # 可选④：并联鼻支路 声门源 × 谐振(鼻极) × 反谐振(鼻零) × g（默认 g=0 ⇒ 无贡献）
            if 鼻g帧[fi] > 1e-9:
                xn = 级联入[a:b] / max(_源标定, 1e-9)
                yn, 鼻极zi = _滤波块(*_谐振器系数(鼻极帧[fi], 120.0), xn, 鼻极zi)
                b0, b1, b2 = _反谐振系数(鼻零帧[fi], 200.0)
                zn = np.zeros_like(yn)
                zn[0] = b0 * yn[0]
                if len(yn) > 1:
                    zn[1] = b0 * yn[1] + b1 * yn[0]
                if len(yn) > 2:
                    zn[2:] = b0 * yn[2:] + b1 * yn[1:-1] + b2 * yn[:-2]
                出 = 出 + 鼻g帧[fi] * 幅样[a:b] * _源标定 * zn
            预[a:b] = 出
        elif typ == 0:
            blk = 帧带[a:b]
            if fi == 0 or int(源帧[fi - 1]) != 0:
                带zi1 = np.zeros(2)
                带zi2 = np.zeros(2)
                形zi = [np.zeros(2) for _ in range(3)]
            if int(噪形帧[fi]) == 1:
                y = np.zeros_like(blk)
                for j, g in ((0, 1.00), (1, 0.62), (2, 0.38)):
                    A, B, C = _谐振器系数(max(F帧[j, fi], _并联下限), 260.0 + 90.0 * j)
                    yj, 形zi[j] = _滤波块(A, B, C, blk, 形zi[j])
                    y += g * yj
                x = y + 0.18 * blk
            else:
                噪带 = 噪带列表[fi]
                if isinstance(噪带, (tuple, list)) and len(噪带) >= 2:
                    Fc = min(max(float(噪带[0]), 40.0), FS * 0.5 - 250.0)
                    BWc = min(max(float(噪带[1]), 30.0), FS * 0.5 - 250.0)
                elif 噪带 is None:
                    Fc, BWc = 3000.0, 2000.0
                else:
                    Fc, BWc = min(max(float(噪带), 40.0), FS * 0.5 - 250.0), 1400.0
                A, B, C = _谐振器系数(Fc, BWc)
                y, 带zi1 = _滤波块(A, B, C, blk, 带zi1)
                y, 带zi2 = _滤波块(A, B, C, y, 带zi2)
                x = y + 0.22 * blk
            噪预[a:b] = x
            增益 = _辐射增益(x)
            目标 = float(幅帧[fi]) * _噪标定 * 音["噪声增益"] * (0.85 + 0.30 * 送帧[fi])
            rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            噪增益帧[fi] = (目标 / (rms * 增益)) if (rms > 1e-9 and 目标 > 0.0) else 0.0
            z = np.zeros(b - a, dtype=np.float64)
            for j in range(5):
                A, B, C = _谐振器系数(F帧[j, fi], BW帧[j, fi])
                _, 串zi[j] = _滤波块(A, B, C, z, 串zi[j])
            _, 高3zi = _滤波块(*_谐振器系数(F帧[2, fi], BW帧[2, fi]), z, 高3zi)
            _, 高4zi = _滤波块(*_谐振器系数(F帧[3, fi], BW帧[3, fi]), z, 高4zi)
            _, 高5zi = _滤波块(*_谐振器系数(F帧[4, fi], BW帧[4, fi]), z, 高5zi)
            if 鼻g帧[fi] > 1e-9:
                b0, b1, b2 = _反谐振系数(鼻零帧[fi], 200.0)
                zn = np.zeros_like(z)
                if len(z):
                    zn[0] = b0 * z[0]
                if len(z) > 1:
                    zn[1] = b0 * z[1] + b1 * z[0]
                if len(z) > 2:
                    zn[2:] = b0 * z[2:] + b1 * z[1:-1] + b2 * z[:-2]
                zn, 鼻极zi = _滤波块(*_谐振器系数(鼻极帧[fi], 120.0), zn, 鼻极zi)
        else:
            预[a:b] = 0.0
            z = np.zeros(b - a, dtype=np.float64)
            for j in range(5):
                A, B, C = _谐振器系数(F帧[j, fi], BW帧[j, fi])
                _, 串zi[j] = _滤波块(A, B, C, z, 串zi[j])
            _, 高3zi = _滤波块(*_谐振器系数(F帧[2, fi], BW帧[2, fi]), z, 高3zi)
            _, 高4zi = _滤波块(*_谐振器系数(F帧[3, fi], BW帧[3, fi]), z, 高4zi)
            _, 高5zi = _滤波块(*_谐振器系数(F帧[4, fi], BW帧[4, fi]), z, 高5zi)

    平滑噪增益 = 噪增益帧.copy()
    for fi in range(N):
        if int(源帧[fi]) != 0:
            continue
        lo, hi = max(0, fi - 1), min(N, fi + 2)
        邻 = [噪增益帧[k] for k in range(lo, hi) if int(源帧[k]) == 0]
        if 邻:
            平滑噪增益[fi] = float(sum(邻)) / len(邻)
    噪idx = np.where(源帧 == 0)[0]
    if len(噪idx):
        for fi in range(N):
            k = int(np.searchsorted(噪idx, fi))
            候 = []
            if k < len(噪idx):
                候.append(int(噪idx[k]))
            if k > 0:
                候.append(int(噪idx[k - 1]))
            最近 = min(候, key=lambda q: abs(q - fi))
            平滑噪增益[fi] = 平滑噪增益[最近]
    预 += 噪预 * 插(平滑噪增益)

    段包络 = np.ones(总样本, dtype=np.float64)
    渐变量 = min(88, 每帧样本 // 4)
    变点 = np.where(np.diff(源帧) != 0)[0] + 1
    for bf in 变点:
        bf = int(bf)
        if not (int(源帧[bf]) == 1 or int(源帧[bf - 1]) == 1):
            continue
        b = bf * 每帧样本
        a0 = max(0, b - 渐变量)
        if b > 0:
            段包络[a0:b] *= np.linspace(1.0, 0.0, b - a0)
        a1 = min(总样本, b + 渐变量)
        段包络[b:a1] = np.minimum(段包络[b:a1], np.linspace(0.0, 1.0, a1 - b))
    预 *= 段包络

    辐 = np.empty_like(预)
    if 总样本:
        辐[0] = 预[0]
        if 总样本 > 1:
            辐[1:] = 预[1:] - 预[:-1]
    边 = min(int(0.008 * FS), 总样本 // 2)
    if 边 > 1:
        辐[:边] *= np.linspace(0.0, 1.0, 边, endpoint=False)
        辐[-边:] *= np.linspace(1.0, 0.0, 边, endpoint=False)
    out = 辐 * 音["增益"]

    if abs(音["谱倾斜"]) > 1e-9:
        t = 音["谱倾斜"]
        sh = np.empty_like(out)
        sh[0] = out[0]
        if len(out) > 1:
            sh[1:] = out[1:] - t * out[:-1]
        out = sh / (1.0 + 0.5 * abs(t))

    峰0 = float(np.max(np.abs(out))) if 总样本 else 0.0
    if not np.isfinite(峰0):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        峰0 = float(np.max(np.abs(out)))
    缩放 = False
    if 峰0 > 0.95:
        out = out * (0.95 / 峰0)
        缩放 = True
    峰值 = float(np.max(np.abs(out))) if 总样本 else 0.0
    耗时 = time.perf_counter() - t0
    元信息 = dict(
        采样率=FS, 帧率=帧率, 每帧样本=每帧样本, 样本数=总样本, 帧数=N,
        时长秒=总样本 / FS, 峰值=峰值, 峰值缩放=缩放, 音色指纹=音["_指纹"],
        高频支路dB=音["高频支路dB"],
        共振峰均值=tuple(round(float(x), 1) for x in np.mean(F帧, axis=1)),
        声道="Klatt串联F1-F5 + 单调高支(0.30/0.24/0.16)" + (" + 源平坦%.1fdB@3k" % _源平坦dB3k if _平坦a > 1e-9 else ""),
        合成耗时秒=round(耗时, 4), 实时率=round(耗时 / max(总样本 / FS, 1e-9), 4),
        f0样=f0样)
    return out, 元信息


__all__ = ["合成", "FS", "帧率", "每帧样本", "平坦系数"]
