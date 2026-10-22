# -*- coding: utf-8 -*-
"""meeting55/实施/说_音色.py —— 情感音色输出：原生声带 + 音色档案 + 音色模仿 + 外放。

复用 `AHI4/meeting32/声带.py`（只读 import，不改）：参数化发声（F0/气声/紧喉/声区/语速…）。
本文件只加：
  · 音色档案（定制）：F0/气声/紧喉/声区/语速/幅度 的一组参数，可存 JSON；
  · 情绪韵律：高兴/低落/急促/平静 → 参数偏移；
  · 音色模仿：从参考录音估 F0/气声/亮度 → 生成同参数声音；
  · 外放：sounddevice 播放 wav（真扬声器）。
"""
from __future__ import annotations

import json
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
_声带目录 = os.path.join(_ROOT, "meeting32")
sys.path.insert(0, _声带目录)
import 声带 as 声带模块  # noqa: E402

FS = 16000
产物 = os.path.join(HERE, "产物")
音色目录 = os.path.join(HERE, "音色档案")
os.makedirs(产物, exist_ok=True)
os.makedirs(音色目录, exist_ok=True)


# ─────────────────────── 音色档案（定制） ───────────────────────
音色档案 = {
    "女声-明亮": dict(基频=215.0, 语速=3.6, 幅度=0.55, 气声度=0.15, 紧喉度=0.28, 粗糙度=0.05, 送气量=0.20, 声区=0.05),
    "女声-温柔": dict(基频=190.0, 语速=3.1, 幅度=0.45, 气声度=0.35, 紧喉度=0.15, 粗糙度=0.04, 送气量=0.30, 声区=0.03),
    "男声-低沉": dict(基频=112.0, 语速=2.9, 幅度=0.55, 气声度=0.20, 紧喉度=0.40, 粗糙度=0.08, 送气量=0.18, 声区=0.02),
    "男声-清亮": dict(基频=145.0, 语速=3.5, 幅度=0.55, 气声度=0.12, 紧喉度=0.30, 粗糙度=0.05, 送气量=0.15, 声区=0.08),
}

情绪偏移 = {
    "高兴": dict(基频倍=1.15, 语速倍=1.18, 幅度倍=1.10, 气声加=-0.05, 紧喉加=0.10),
    "低落": dict(基频倍=0.88, 语速倍=0.82, 幅度倍=0.85, 气声加=0.15, 紧喉加=-0.10),
    "急促": dict(基频倍=1.10, 语速倍=1.35, 幅度倍=1.05, 气声加=-0.02, 紧喉加=0.12),
    "平静": dict(基频倍=1.00, 语速倍=1.00, 幅度倍=1.00, 气声加=0.00, 紧喉加=0.00),
}


def 载档案(名="女声-明亮"):
    if 名 in 音色档案:
        return dict(音色档案[名])
    p = os.path.join(音色目录, 名 + ".json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return dict(音色档案["女声-明亮"])


def 存档案(名, 参):
    p = os.path.join(音色目录, 名 + ".json")
    json.dump(参, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return p


def 情绪律(档案: dict, 情绪: str) -> dict:
    m = 情绪偏移.get(情绪, 情绪偏移["平静"])
    基 = float(档案.get("基频", 190.0)) * m["基频倍"]
    幅 = min(max(float(档案.get("幅度", 0.5)) * m["幅度倍"], 0.1), 1.0)
    气 = min(max(float(档案.get("气声度", 0.2)) + m["气声加"], 0.0), 0.8)
    紧 = min(max(float(档案.get("紧喉度", 0.3)) + m["紧喉加"], 0.0), 1.0)
    return dict(基频=float(min(max(基, 70.0), 400.0)),
                语速=float(min(max(float(档案.get("语速", 3.2)) * m["语速倍"], 1.5), 7.0)),
                幅度=幅, 气声度=气, 紧喉度=紧,
                粗糙度=float(档案.get("粗糙度", 0.05)),
                送气量=float(档案.get("送气量", 0.2)),
                声区=float(档案.get("声区", 0.05)), _ñ=0.5)


class 声带发声(声带模块.声带器官):
    """子类：把 律 固定成「音色档案×情绪」，绕开遥测映射。"""

    def __init__(self, seed: int = 300, D: int = 1024):
        super().__init__(seed=seed, D=D, 目录=os.path.join(产物, "发声"),
                         台账路径=os.path.join(产物, "发声台账.jsonl"))
        os.makedirs(self.目录, exist_ok=True)
        self.当前律 = None

    def 韵律(self, 遥测: dict) -> dict:  # noqa: N802
        if self.当前律 is not None:
            return dict(self.当前律)
        return super().韵律(遥测)


_器 = None


def 取器():
    global _器
    if _器 is None:
        _器 = 声带发声()
    return _器


_说序号 = 0


def 说(文本: str, 音色="女声-明亮", 情绪="平静", 向量=None, 语速倍=1.0, 音量倍=1.0) -> dict:
    """兼容旧口径的适配层：内部走 声带.py 的**旧路径**（当前律+合成波形），
    因此 meeting55 的 H2/H3/V1 等既有判据逐位语义不变；
    新的深度优化路径（音色档案/紧喉/粗糙/共振峰/谱倾斜/后置增益/分块 IIR）在
    meeting58 的 协议_音色渲染.py 里，不经过本函数。
    并发说明：每调用新建一个 声带发声（构造约 1.2ms），发声号用模块级序号保证 wav 不撞名。"""
    global _说序号
    器 = 声带发声()
    _说序号 += 1
    器.发声号 = _说序号 - 1
    参 = 载档案(音色)
    器.当前律 = 情绪律(参, 情绪)
    器.当前律["语速"] = float(min(max(器.当前律["语速"] * 语速倍, 1.5), 7.0))
    器.当前律["幅度"] = float(min(max(器.当前律["幅度"] * 音量倍, 0.1), 1.0))
    D = 器.D
    向量 = np.asarray(向量, dtype=np.float32) if 向量 is not None else \
        np.sign(np.random.default_rng(55).standard_normal(D)).astype(np.float32)
    遥测 = dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5)
    器.当前律["_ñ"] = 0.5
    rec = 器.发声(向量[:D], 文本, 遥测)
    rec["音色"] = 音色
    rec["情绪"] = 情绪
    rec["参数_档案"] = 参
    return rec


def 说_快(文本: str, 音色="女声-明亮", 情绪="平静", 向量=None,
         语速倍: float = 1.0, 音量倍: float = 1.0) -> dict:
    """meeting58 深度优化路径：原生音色接口（紧喉/粗糙/幅度/共振峰/带宽/谱倾斜 +
    分块 IIR）。与 说() 的差别：说() 保持旧路径逐位语义（既有判据用），
    说_快() 是新输出层；默认共振峰=(1,1,1)、谱倾斜=0，只有 8 参数生效。"""
    器 = 取器()
    参 = 载档案(音色)
    律 = 情绪律(参, 情绪)
    律["语速"] = float(min(max(律["语速"] * 语速倍, 1.5), 7.0))
    律["幅度"] = float(min(max(律["幅度"] * 音量倍, 0.1), 1.0))
    D = 器.D
    向量 = np.asarray(向量, dtype=np.float32) if 向量 is not None else \
        np.sign(np.random.default_rng(55).standard_normal(D)).astype(np.float32)
    音色参数 = {k: float(律[k]) for k in tuple(声带模块.参数表) + ("语速",)}
    音色参数.update(共振峰缩放=(1.0, 1.0, 1.0), 带宽缩放=1.0, 谱倾斜=0.0,
                    送气增益=0.0, 增益=1.0)
    rec = 器.发声(向量[:D], 文本, dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5),
                  音色=音色参数)
    rec["音色"] = 音色
    rec["情绪"] = 情绪
    rec["参数_档案"] = 参
    return rec


def 播放(wav路径: str) -> bool:
    """真扬声器播放（sounddevice）。"""
    try:
        import sounddevice as sd
        from scipy.io import wavfile
        sr, x = wavfile.read(wav路径)
        x = x.astype(np.float32) / 32768.0
        sd.play(x, sr)
        sd.wait()
        return True
    except Exception:
        return False


def 读wav(路径: str):
    from scipy.io import wavfile
    sr, x = wavfile.read(路径)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    if np.max(np.abs(x)) > 1.0:
        x = x / 32768.0
    return sr, x


def 音色估计(参考wav: str) -> dict:
    """从参考录音估音色参数（F0 中位/能量/谱亮度/平坦度）→ 声带参数。"""
    sr, x = 读wav(参考wav)
    if sr != FS:
        # 线性重采样到 16k
        n = int(len(x) * FS / sr)
        x = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)
        sr = FS
    F0 = _F0轨迹(x, sr)
    F0v = F0[np.isfinite(F0) & (F0 > 60)]
    f0 = float(np.median(F0v)) if len(F0v) else 190.0
    # 谱亮度/平坦度
    mag = np.abs(np.fft.rfft(x[:min(len(x), sr * 2)] * np.hanning(min(len(x), sr * 2))))
    频 = np.fft.rfftfreq(min(len(x), sr * 2), 1 / sr)
    重心 = float(np.sum(频 * mag) / (np.sum(mag) + 1e-9))
    平坦 = float(np.exp(np.mean(np.log(mag + 1e-9))) / (np.mean(mag) + 1e-9))
    能量 = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    参 = dict(基频=float(min(max(f0, 70), 400)),
              语速=3.2, 幅度=float(min(max(能量 * 4.0, 0.2), 0.9)),
              气声度=float(min(max(0.10 + 0.6 * 平坦, 0.05), 0.7)),
              紧喉度=float(min(max(0.5 - 重心 / 6000.0, 0.05), 0.8)),
              粗糙度=0.05, 送气量=0.2, 声区=0.05)
    return dict(参数=参, F0中位=round(f0, 1), 谱重心=round(重心, 1),
                谱平坦=round(平坦, 4), 能量=round(能量, 4), 设备采样率=sr)


def _F0轨迹(x, sr=FS, 帧长=0.04, 帧移=0.01):
    n = max(1, int((len(x) - int(帧长 * sr)) // int(帧移 * sr) + 1))
    F0 = np.full(n, np.nan)
    for i in range(n):
        seg = x[i * int(帧移 * sr): i * int(帧移 * sr) + int(帧长 * sr)]
        if len(seg) < 100:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        if hi >= len(ac):
            continue
        段 = ac[lo:hi]
        if np.max(段) <= 0:
            continue
        lag = lo + int(np.argmax(段))
        if ac[lag] / (ac[0] + 1e-9) < 0.3:
            continue
        F0[i] = sr / lag
    return F0


def 频谱特征(wav路径: str) -> dict:
    sr, x = 读wav(wav路径)
    F0 = _F0轨迹(x, sr)
    F0v = F0[np.isfinite(F0) & (F0 > 60)]
    return dict(F0均值=round(float(np.mean(F0v)), 1) if len(F0v) else 0.0,
                F0标准差=round(float(np.std(F0v)), 1) if len(F0v) else 0.0,
                能量=round(float(np.sqrt(np.mean(x ** 2))), 4),
                时长=round(float(len(x) / sr), 3))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    for 音 in ("女声-明亮", "男声-低沉"):
        for 情 in ("高兴", "低落", "急促", "平静"):
            rec = 说("你好，我在这里。今天过得怎么样？", 音色=音, 情绪=情)
            f = 频谱特征(os.path.join(产物, "发声", rec["wav"]))
            print(音, 情, "→", rec["wav"], f, "实时率", rec["实时率"])
