# -*- coding: utf-8 -*-
"""meeting60 · 44.1k 客观指标与多组测试矩阵（负责人 matrix）

只读输入 wav；写 ``产物/指标/组指标.json``，并可生成 ``产物/测试组/测试组清单.md``。
纯 numpy/scipy（C1/C10），无外部模型；同输入逐位同结果（C13）。

用法::

    $env:PYTHONIOENCODING="utf-8"
    & $py 实施\\矩阵_指标.py                      # 全量扫描 产物/测试组/组*/ 下所有 wav
    & $py 实施\\矩阵_指标.py --单文件 x.wav        # 单文件读数（打印）
    & $py 实施\\矩阵_指标.py --输出json a.json --输出md b.md --无md

口径（与 ``测试组清单.md`` 必须一致）：
- 时域指标（RMS/峰值/削波/时长）在**原采样率**上计算；
- 谱指标（谱重心/滚降/能量占比/谱平坦度）与 F0 统一 **resample_poly 到 44.1kHz** 后再分析；
  16k 源属于上采样，不会新增高频内容；
- 能量占比分母 = **0…Nyquist 全带功率**（44.1k 分析即 0–22.05kHz）。所以 16k 旧版源
  ``8–16k`` 恒≈0 是**采样带宽上限**，不是修复失败；
- 削波比例 = ``|x| > 0.99`` 的采样点占比；峰值因数 = 峰值/RMS（dB），偏低=被压限/饱和，
  作为**失真代理之一**；
- F0 中位 = 40ms 帧 / 20ms 跳、归一化自相关、搜索 70–500Hz，取 ``rmax≥0.35`` 且帧能量
  ≥ 最大帧 1% 的浊音帧中位数（带次谐波修正：取 r≥0.85·rmax 的最短延迟）；
- 谱平坦度主值在 **0–8kHz 共带**上算（跨采样率可比），另给 0–16k 全带值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

FS分析 = 44100
角色们 = ("EmoCompanion", "SpeakerA", "洛天依")
分析带宽 = FS分析 / 2.0  # 22050

# 简报 §4 的组定义（用于清单文字；实际以磁盘上出现的目录为准）
组说明 = {
    "A": dict(
        做法="meeting59 旧版 v2 交付（16kHz 输出、F1–F3 三谐振器、±8dB 静态 EQ、tanh 饱和）只读复制",
        口径="旧版对照基线；16k 采样上限 ⇒ 8kHz 以上无内容",
        听点="闷/模糊/金属感是否明显；与 B–G 对照听高频是否回来、有无失真毛刺",
        边界="旧版本身是问题样本，不是可接受目标；8–16k≈0 由 16k 采样上限决定，不能当作修复失败的反证",
    ),
    "B": dict(
        做法="新高保真引擎（44.1k、F1–F5+唇辐射）+ 参考拟合声线，**无人类韵律、无 EQ**",
        口径="裸声对照：只看引擎/声门源本身，排除韵律与 EQ 的影响",
        听点="裸声是否仍有金属/机器人味；高频是否回来、齿音/气声是否自然；是否有削波毛刺",
        边界="无韵律 ⇒ 听感上必然不如 C/D 自然，本组只用于隔离引擎问题",
    ),
    "C": dict(
        做法="组 B + 人类化韵律（停顿/换气/句调/变调/重音/句末/微颤）",
        口径="纯韵律增益；与 B 对比看韵律是否伤害谱质",
        听点="停顿/换气/句调是否像人；是否出现「唱腔」或拖腔不自然；元音处是否发闷",
        边界="韵律参数为规则化设计，不是从参考直接学习；自然度仍受声线拟合上界限制",
    ),
    "D": dict(
        做法="组 C + ≤±3dB 静态谱校正（轻 EQ）",
        口径="交付候选；EQ 幅度严格限制，禁止 v2 的 ±8dB 强校正",
        听点="整体音色是否平衡、是否过亮/齿音过重；与 C 对比判断轻 EQ 是否值得保留",
        边界="EQ 只是静态包络补偿，不能修复引擎缺失的谐波；≠ 参考原声",
    ),
    "E": dict(
        做法="组 D 声线音高 ×0.85（约 -2.8 半音），声明「降为说话音高」",
        口径="自然音高变体；F0 预期下降、谱包络不变",
        听点="是否更像日常说话而不是歌唱；降调后是否显浑浊、共振峰是否失配",
        边界="只改 F0 层，不动共振峰 ⇒ 降调后音色会偏离参考（声道长度未同步）",
    ),
    "F": dict(
        做法="组 D 的同一句文本三声线（EmoCompanion/SpeakerA/洛天依）",
        口径="同句可比：文本、韵律骨架、句长一致，只换声线",
        听点="三条声线是否可分辨；是否仍有「同一人变调」感；F0 间距是否可听",
        边界="三个参考源本身声学接近（旧版实测 F0 差 0.19 半音），可分辨性有数据上界",
    ),
    "G": dict(
        做法="44.1k 参考片段 ↔ 组 D 合成交替（含参考原声，标注）",
        口径="A/B 参照；参考片段仅本机估参用，不再分发",
        听点="直接 A/B：高频/齿音/气声/共振峰位置差多少；哪一项差距最大",
        边界="参考原声不是交付物、不能当合成结果分发；A/B 只在本机对比，未做真人 MOS",
    ),
}

# 参考目标兜底（简报 §0 对EmoCompanion 24k 参考源的实测；有组 G 参考 wav 时以实测为准）
兜底参考 = {
    "EmoCompanion": {"谱重心Hz": 974.0, "滚降95Hz": 3539.0, "占比_4_8k": 0.029, "占比_8_16k": 0.005},
}

指标键 = [
    "采样率", "声道", "时长s", "RMS_dBFS", "峰值_dBFS", "峰值线性", "峰值因数_dB", "削波比例",
    "谱重心Hz", "滚降50Hz", "滚降85Hz", "滚降95Hz",
    "占比_0_1k", "占比_1_4k", "占比_4_8k", "占比_8_16k", "占比_16_22k",
    "谱平坦度_0_8k", "谱平坦度_0_16k", "F0中位Hz", "F0_p5Hz", "F0_p95Hz", "浊音占比",
]


# ---------------------------------------------------------------- 基础 IO
def 读wav(路径: Path):
    """读任意 wav → (fs, 单声道 float64 [-1,1])。"""
    fs, 原始 = wavfile.read(str(路径))
    x = 原始
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) / 128.0
    elif x.dtype in (np.float32, np.float64):
        x = x.astype(np.float64)
    else:
        raise TypeError(f"不支持的 wav dtype: {x.dtype} @ {路径}")
    声道 = 1 if x.ndim == 1 else int(x.shape[1])
    if x.ndim > 1:
        x = x.mean(axis=1)
    return int(fs), x, 声道


def 重采样(x: np.ndarray, fs: int, 目标: int = FS分析) -> np.ndarray:
    if fs == 目标:
        return np.asarray(x, dtype=np.float64)
    g = math.gcd(int(fs), int(目标))
    return resample_poly(np.asarray(x, dtype=np.float64), int(目标) // g, int(fs) // g)


def 指纹(路径: Path) -> str:
    h = hashlib.sha1()
    with open(路径, "rb") as f:
        for 块 in iter(lambda: f.read(1 << 20), b""):
            h.update(块)
    return h.hexdigest()[:16]


def _db(v: float) -> float:
    return 20.0 * math.log10(v) if v > 0 else -200.0


def _帧(x: np.ndarray, 帧长: int, 跳: int) -> np.ndarray:
    n = len(x)
    if n <= 帧长:
        y = np.zeros(帧长, dtype=np.float64)
        y[:n] = x
        return y[None, :]
    个数 = 1 + (n - 帧长) // 跳
    idx = np.arange(帧长, dtype=np.int64)[None, :] + 跳 * np.arange(个数, dtype=np.int64)[:, None]
    return np.asarray(x, dtype=np.float64)[idx]


# ---------------------------------------------------------------- 谱与 F0
def 谱统计(x: np.ndarray, fs: int) -> dict:
    """44.1k 下的谱统计（谱重心/滚降/带占比/平坦度）。"""
    x = np.asarray(x, dtype=np.float64)
    帧长 = 4096
    frames = _帧(x, 帧长, 帧长 // 2)
    frames = frames - frames.mean(axis=1, keepdims=True)
    窗 = np.hanning(帧长)
    F = np.fft.rfft(frames * 窗[None, :], axis=1)
    P = np.mean(np.abs(F) ** 2, axis=0)
    P[0] = 0.0
    f = np.fft.rfftfreq(帧长, 1.0 / fs)
    总 = float(P.sum())
    if 总 <= 0 or not np.isfinite(总):
        return dict(谱重心Hz=0.0, 滚降50Hz=0.0, 滚降85Hz=0.0, 滚降95Hz=0.0,
                    占比_0_1k=0.0, 占比_1_4k=0.0, 占比_4_8k=0.0, 占比_8_16k=0.0,
                    占比_16_22k=0.0, 谱平坦度_0_8k=0.0, 谱平坦度_0_16k=0.0)
    重心 = float((f * P).sum() / 总)
    cs = np.cumsum(P) / 总
    滚降 = {}
    for p in (50, 85, 95):
        i = int(np.argmax(cs >= p / 100.0))
        滚降[f"滚降{p}Hz"] = float(f[i])
    带 = {}
    for 名, lo, hi in (
        ("占比_0_1k", 0.0, 1000.0),
        ("占比_1_4k", 1000.0, 4000.0),
        ("占比_4_8k", 4000.0, 8000.0),
        ("占比_8_16k", 8000.0, 16000.0),
        ("占比_16_22k", 16000.0, 22000.0),
    ):
        m = (f >= lo) & (f < hi)
        带[名] = float(P[m].sum() / 总) if m.any() else 0.0

    def 平坦(hi: float) -> float:
        m = (f > 0.0) & (f <= hi)
        p = P[m]
        if p.size == 0 or p.mean() <= 0:
            return 0.0
        g = math.exp(float(np.mean(np.log(p + 1e-30))))
        return float(g / p.mean())

    return dict(谱重心Hz=重心, **滚降, **带,
                谱平坦度_0_8k=平坦(8000.0), 谱平坦度_0_16k=平坦(16000.0))


def F0统计(x: np.ndarray, fs: int, 帧长s: float = 0.04, 跳s: float = 0.02,
          fmin: float = 70.0, fmax: float = 500.0, 阈: float = 0.35) -> dict:
    """帧级归一化自相关 → F0 中位/分位/浊音占比。"""
    x = np.asarray(x, dtype=np.float64)
    帧长 = max(64, int(round(帧长s * fs)))
    跳 = max(32, int(round(跳s * fs)))
    if len(x) < 帧长 // 2:
        return dict(F0中位Hz=None, F0_p5Hz=None, F0_p95Hz=None, 浊音占比=0.0)
    frames = _帧(x, 帧长, 跳)
    能量 = np.mean(frames ** 2, axis=1)
    门 = 0.01 * float(能量.max()) if 能量.max() > 0 else 0.0
    frames = frames - frames.mean(axis=1, keepdims=True)
    frames = frames * np.hanning(帧长)[None, :]
    nfft = 1 << int(math.ceil(math.log2(2 * 帧长)))
    F = np.fft.rfft(frames, n=nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)
    r0 = ac[:, 0].copy()
    r0[r0 <= 0] = 1e-20
    acn = ac / r0[:, None]
    最短 = max(1, int(math.floor(fs / fmax)))
    最长 = min(nfft - 1, int(math.ceil(fs / fmin)))
    seg = acn[:, 最短:最长 + 1]
    if seg.shape[1] < 3:
        return dict(F0中位Hz=None, F0_p5Hz=None, F0_p95Hz=None, 浊音占比=0.0)
    imax = np.argmax(seg, axis=1)
    rmax = seg[np.arange(len(imax)), imax]
    # 次谐波修正：在 r ≥ 0.85·rmax 的局部极大里取最短延迟（最高 F0）
    阈候选 = rmax[:, None] * 0.85
    局部 = np.zeros_like(seg, dtype=bool)
    局部[:, 1:-1] = (seg[:, 1:-1] >= seg[:, :-2]) & (seg[:, 1:-1] >= seg[:, 2:])
    候选 = (seg >= 阈候选) & 局部
    有候选 = 候选.sum(axis=1) > 0
    首位 = np.argmax(候选, axis=1)
    首位[~有候选] = imax[~有候选]
    f0 = fs / (首位 + 最短)
    有效 = (rmax >= 阈) & (能量 > 门)
    f0s = f0[有效]
    if f0s.size == 0:
        return dict(F0中位Hz=None, F0_p5Hz=None, F0_p95Hz=None, 浊音占比=0.0)
    return dict(
        F0中位Hz=float(np.median(f0s)),
        F0_p5Hz=float(np.percentile(f0s, 5)),
        F0_p95Hz=float(np.percentile(f0s, 95)),
        浊音占比=float(有效.mean()),
    )


# ---------------------------------------------------------------- 主指标
def 计算指标(路径: Path, 组名: str = "", 角色: str = "", 来源: str = "合成") -> dict:
    路径 = Path(路径)
    fs, x, 声道 = 读wav(路径)
    时长 = len(x) / fs if fs else 0.0
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    峰值 = float(np.max(np.abs(x))) if len(x) else 0.0
    削波 = float(np.mean(np.abs(x) > 0.99)) if len(x) else 0.0
    y = 重采样(x, fs)
    谱 = 谱统计(y, FS分析)
    f0 = F0统计(y, FS分析)
    记录 = dict(
        文件=路径.name,
        组=组名,
        角色=角色,
        来源=来源,
        采样率=int(fs),
        声道=int(声道),
        时长s=round(float(时长), 4),
        RMS_dBFS=round(_db(rms), 3),
        峰值_dBFS=round(_db(峰值), 3),
        峰值线性=round(峰值, 6),
        峰值因数_dB=round(_db(峰值 / rms), 3) if rms > 0 else 0.0,
        削波比例=round(削波, 8),
        指纹sha1=指纹(路径),
    )
    记录.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in 谱.items()})
    记录.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in f0.items()})
    return 记录


def 角色识别(相对路径: str) -> str:
    for r in 角色们:
        if r in 相对路径:
            return r
    return Path(相对路径).stem


# ---------------------------------------------------------------- 扫描与对比
def 扫描(根: Path) -> dict:
    组目 = 根 / "产物" / "测试组"
    结果 = {"根": str(根), "组": {}}
    if not 组目.is_dir():
        return 结果
    for d in sorted([p for p in 组目.iterdir() if p.is_dir()], key=lambda p: p.name):
        if not d.name.startswith("组"):
            continue
        wavs = sorted(d.rglob("*.wav"))
        音频 = {}
        for w in wavs:
            rel = w.relative_to(d).as_posix()
            role = 角色识别(rel)
            来源 = "参考" if "参考" in rel else "合成"
            key = f"参考_{role}" if 来源 == "参考" else role
            if key in 音频:
                n = 2
                while f"{key}#{n}" in 音频:
                    n += 1
                key = f"{key}#{n}"
            音频[key] = 计算指标(w, 组名=d.name, 角色=role, 来源=来源)
        结果["组"][d.name] = {"组码": d.name[1] if len(d.name) > 1 else d.name,
                             "文件数": len(wavs), "音频": 音频}
    return 结果


def 找基线(结果: dict) -> str | None:
    for 名 in 结果["组"]:
        if 名.startswith("组A"):
            return 名
    return None


def 对比表(结果: dict) -> dict:
    基线 = 找基线(结果)
    if 基线 is None:
        return {}
    基 = 结果["组"][基线]["音频"]
    方向 = {"占比_4_8k": 1, "占比_8_16k": 1, "占比_1_4k": 1, "滚降95Hz": 1,
            "谱重心Hz": 1, "削波比例": -1, "峰值因数_dB": 1, "谱平坦度_0_8k": 1,
            "F0中位Hz": 1, "时长s": 1}
    输出 = {"基线": 基线, "方向": 方向, "组": {}}
    for 名, 数据 in 结果["组"].items():
        if 名 == 基线:
            continue
        逐 = {}
        for key, m in 数据["音频"].items():
            if key not in 基:
                continue
            d = {}
            for k in 方向:
                a = 基[key].get(k)
                b = m.get(k)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    d[k] = b - a
                else:
                    d[k] = None
            if d:
                逐[key] = d
        均值 = {}
        if 逐:
            for k in 方向:
                vs = [d[k] for d in 逐.values() if isinstance(d.get(k), (int, float))]
                if vs:
                    均值[k] = float(np.mean(vs))
        输出["组"][名] = {"逐角色": 逐, "均值Δ": 均值}
    return 输出


def 参考目标(结果: dict) -> dict:
    汇总 = {}
    for 数据 in 结果["组"].values():
        for key, m in 数据["音频"].items():
            if m.get("来源") != "参考":
                continue
            role = key.replace("参考_", "").split("#")[0]
            汇总.setdefault(role, []).append(m)
    目标 = {}
    for role, ms in 汇总.items():
        t = {}
        for k in ("谱重心Hz", "滚降95Hz", "占比_4_8k", "占比_8_16k", "削波比例",
                  "峰值因数_dB", "谱平坦度_0_8k"):
            vs = [m[k] for m in ms if isinstance(m.get(k), (int, float))]
            if vs:
                t[k] = float(np.mean(vs))
        目标[role] = {"读数": t, "文件数": len(ms), "来源": "组G 参考 wav 实测（44.1k 分析）"}
    if not 目标:
        目标 = {r: {"读数": dict(v), "文件数": 0, "来源": "简报 §0 实测（24k 参考源）兜底"}
                for r, v in 兜底参考.items()}
    return 目标


def 参考接近度(结果: dict, 对比: dict, 目标: dict) -> dict:
    """与组 A 相比，向参考目标靠近的幅度（距离减少量；正=更好）。"""
    基线 = 对比.get("基线")
    if not 基线:
        return {}
    基 = 结果["组"][基线]["音频"]
    距离键 = ["谱重心Hz", "滚降95Hz", "占比_4_8k", "占比_8_16k", "削波比例", "峰值因数_dB"]
    尺度 = {"谱重心Hz": 1000.0, "滚降95Hz": 1000.0, "占比_4_8k": 0.02,
            "占比_8_16k": 0.005, "削波比例": 0.001, "峰值因数_dB": 3.0}
    输出 = {}
    for 名, 数据 in 结果["组"].items():
        if 名 == 基线:
            continue
        逐 = {}
        for key, m in 数据["音频"].items():
            if m.get("来源") == "参考" or key not in 基:
                continue
            role = m.get("角色", key)
            t = (目标.get(role) or {}).get("读数")
            if not t:
                continue
            基m = 基[key]
            ds = {}
            for k in 距离键:
                if k not in t or not isinstance(基m.get(k), (int, float)) or not isinstance(m.get(k), (int, float)):
                    continue
                d0 = abs(基m[k] - t[k]) / 尺度[k]
                d1 = abs(m[k] - t[k]) / 尺度[k]
                ds[k] = d0 - d1  # 正 = 距离缩小
            if ds:
                逐[role] = ds
        if 逐:
            均值 = {}
            for k in 距离键:
                vs = [d[k] for d in 逐.values() if k in d]
                if vs:
                    均值[k] = float(np.mean(vs))
            输出[名] = {"逐角色": 逐, "均值改善": 均值}
    return 输出


# ---------------------------------------------------------------- JSON / MD
def _可json(o):
    if isinstance(o, dict):
        return {str(k): _可json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_可json(v) for v in o]
    if isinstance(o, np.floating):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def 文本(值, 位=3, 后缀="") -> str:
    if 值 is None:
        return "—"
    return f"{值:.{位}f}{后缀}"


def 生成清单md(结果: dict, 对比: dict, 排名: dict, 接近: dict, 目标: dict) -> str:
    L = []
    L.append("# 测试组清单（meeting60 · 多组测试矩阵）")
    L.append("")
    L.append("> 自动表由 `实施/矩阵_指标.py` 生成；**读数口径见 §0**。口径始终是「参数化模仿，不是声克隆/角色本人」。")
    L.append("")
    L.append("## 0. 口径与诚实边界（先读）")
    L.append("")
    L.append("- 时域指标（RMS/峰值/削波/时长）在**原采样率**上算；谱指标与 F0 统一 **resample_poly→44.1kHz** 后分析（16k 源上采样不新增高频）。")
    L.append("- 能量占比分母 = 0…Nyquist 全带功率（44.1k 分析即 0–22.05kHz）⇒ 16k 旧版源 `8–16k` 恒≈0 是**采样带宽上限**，不是修复失败。")
    L.append("- 削波比例 = `|x|>0.99` 采样点占比；峰值因数 = 峰值/RMS(dB)，偏低=被压限/饱和（失真代理之一）。")
    L.append("- F0 中位：40ms 帧/20ms 跳、归一化自相关、70–500Hz、`rmax≥0.35` 且帧能量≥最大帧 1% 的浊音帧中位；含次谐波修正。")
    L.append("- 谱平坦度主值在 **0–8kHz 共带**（跨采样率可比），另给 0–16k 全带值。")
    L.append("- 客观指标 ≠ 听感；**未做真人盲评/MOS**。组 G 含参考原声，仅供本机 A/B，参考原声不是交付物、不再分发。")
    L.append("- 组 E 只改 F0、共振峰未同步 ⇒ 降调后音色必然偏离参考；组 F 三个参考源本身声学接近，可分辨性有数据上界。")
    L.append("")
    L.append("## 1. 组总览")
    L.append("")
    L.append("| 组 | 做法 | 口径 | 文件数 | 状态 |")
    L.append("|---|---|---|---|---|")
    for 名, 数据 in 结果["组"].items():
        code = 数据["组码"]
        info = 组说明.get(code, {})
        L.append(f"| {名} | {info.get('做法','—')} | {info.get('口径','—')} | {数据['文件数']} | 已测 |")
    for code in "ABCDEFG":
        if not any(数据["组码"] == code for 数据 in 结果["组"].values()):
            info = 组说明.get(code, {})
            L.append(f"| 组{code} | {info.get('做法','—')} | {info.get('口径','—')} | — | 等 lead 渲染 |")
    L.append("")
    L.append("## 2. 客观读数（按 角色×组）")
    L.append("")
    列 = [("采样率", "采样率", 0), ("时长s", "时长s", 2), ("RMS_dBFS", "RMS", 1),
          ("峰值_dBFS", "峰值", 1), ("削波比例", "削波%", 3), ("峰值因数_dB", "峰因", 1),
          ("谱重心Hz", "谱重心", 0), ("滚降50Hz", "滚降50", 0), ("滚降85Hz", "滚降85", 0),
          ("滚降95Hz", "滚降95", 0), ("占比_0_1k", "0–1k%", 2), ("占比_1_4k", "1–4k%", 2),
          ("占比_4_8k", "4–8k%", 2), ("占比_8_16k", "8–16k%", 2), ("占比_16_22k", "16–22k%", 2),
          ("谱平坦度_0_8k", "平坦度", 4), ("F0中位Hz", "F0中位", 1), ("浊音占比", "浊音%", 1)]
    for 名, 数据 in 结果["组"].items():
        L.append(f"### {名}（{数据['文件数']} 个 wav）")
        L.append("")
        L.append("| 角色 | " + " | ".join(c[1] for c in 列) + " |")
        L.append("|---" * (len(列) + 1) + "|")
        for key, m in 数据["音频"].items():
            行 = []
            for k, 标签, 位 in 列:
                v = m.get(k)
                if k == "削波比例":
                    行.append(文本(None if v is None else v * 100, 3, ""))
                elif k in ("占比_0_1k", "占比_1_4k", "占比_4_8k", "占比_8_16k", "占比_16_22k", "浊音占比"):
                    行.append(文本(None if v is None else v * 100, 2, ""))
                else:
                    行.append(文本(v, 位, ""))
            L.append(f"| {key} | " + " | ".join(行) + " |")
        L.append("")
    L.append("## 3. 相对组 A 的差异（Δ；组 A 为 0）“+”= 更高）")
    L.append("")
    if 对比 and 对比.get("组"):
        L.append("| 组 | 角色 | Δ4–8k(pp) | Δ8–16k(pp) | Δ滚降95(Hz) | Δ谱重心(Hz) | Δ削波(pp) | Δ峰因(dB) | Δ平坦度0–8k | ΔF0(Hz) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for 名, 数据 in 对比["组"].items():
            for key, d in 数据["逐角色"].items():
                值 = lambda k, 倍=1.0, 位=2: 文本(None if d.get(k) is None else d[k] * 倍, 位, "")
                L.append(f"| {名} | {key} | {值('占比_4_8k',100)} | {值('占比_8_16k',100)} | "
                         f"{值('滚降95Hz',1,0)} | {值('谱重心Hz',1,0)} | {值('削波比例',100,3)} | "
                         f"{值('峰值因数_dB',1,1)} | {值('谱平坦度_0_8k',1,4)} | {值('F0中位Hz',1,1)} |")
        L.append("")
    L.append("## 4. 与组 A 相比改善最大的项")
    L.append("")
    if 排名:
        L.append("方向性指标（不代表一定更好，需配合组 G 参考读数判断）：")
        L.append("")
        for 名, 内容 in 排名.items():
            L.append(f"- **{内容['指标']}**：{内容['最优组']}（均值 Δ {内容['Δ']:+.3f}）" if isinstance(内容['Δ'], (int, float))
                     else f"- **{内容['指标']}**：—")
        L.append("")
    if 接近:
        L.append("### 向参考目标靠近（距离减少量，已按量级归一；正=更接近参考）")
        L.append("")
        L.append("| 组 | 谱重心 | 滚降95 | 4–8k | 8–16k | 削波 | 峰因 | 均值 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for 名, 数据 in 接近.items():
            m = 数据["均值改善"]
            均值 = float(np.mean(list(m.values()))) if m else None
            L.append(f"| {名} | " + " | ".join(文本(m.get(k), 3) for k in
                     ["谱重心Hz", "滚降95Hz", "占比_4_8k", "占比_8_16k", "削波比例", "峰值因数_dB"]) +
                     f" | {文本(均值, 3)} |")
        L.append("")
    L.append("## 5. 听点（人工，每组）")
    L.append("")
    for code in "ABCDEFG":
        名 = next((n for n, d in 结果["组"].items() if d["组码"] == code), f"组{code}")
        if not any(d["组码"] == code for d in 结果["组"].values()):
            L.append(f"- **组{code}**：等 lead 渲染。")
            continue
        info = 组说明.get(code, {})
        L.append(f"- **{名}**：{info.get('听点','—')}")
    L.append("")
    L.append("## 6. 各组诚实边界")
    L.append("")
    for code in "ABCDEFG":
        名 = next((n for n, d in 结果["组"].items() if d["组码"] == code), f"组{code}")
        if not any(d["组码"] == code for d in 结果["组"].values()):
            continue
        info = 组说明.get(code, {})
        L.append(f"- **{名}**：{info.get('边界','—')}")
    L.append("")
    L.append("## 7. 参考目标（组 G 参考实测；无则用简报 §0 兜底）")
    L.append("")
    L.append("| 角色 | 来源 | 谱重心 | 滚降95 | 4–8k% | 8–16k% | 削波% | 峰因 | 平坦度 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for role, t in 目标.items():
        r = t["读数"]
        L.append(f"| {role} | {t['来源']} | {文本(r.get('谱重心Hz'),0)} | {文本(r.get('滚降95Hz'),0)} | "
                 f"{文本(None if r.get('占比_4_8k') is None else r['占比_4_8k']*100,2)} | "
                 f"{文本(None if r.get('占比_8_16k') is None else r['占比_8_16k']*100,2)} | "
                 f"{文本(None if r.get('削波比例') is None else r['削波比例']*100,3)} | "
                 f"{文本(r.get('峰值因数_dB'),1)} | {文本(r.get('谱平坦度_0_8k'),4)} |")
    L.append("")
    return "\n".join(L) + "\n"


def 计算排名(对比: dict) -> dict:
    if not 对比 or not 对比.get("组"):
        return {}
    项 = [("占比_4_8k", "高频 4–8k 占比 ↑"), ("占比_8_16k", "高频 8–16k 占比 ↑"),
          ("滚降95Hz", "95% 滚降 ↑"), ("谱重心Hz", "谱重心 ↑"),
          ("削波比例", "削波比例 ↓"), ("峰值因数_dB", "峰值因数 ↑")]
    输出 = {}
    for k, 标签 in 项:
        最好 = None
        for 名, 数据 in 对比["组"].items():
            v = 数据["均值Δ"].get(k)
            if not isinstance(v, (int, float)):
                continue
            方向 = 对比["方向"].get(k, 1)
            if 最好 is None or 方向 * v > 方向 * 最好[1]:
                最好 = (名, v)
        if 最好:
            输出[k] = {"指标": 标签, "最优组": 最好[0], "Δ": 最好[1]}
    return 输出


def main(argv=None) -> int:
    本目 = Path(__file__).resolve().parent
    根 = 本目.parent  # meeting60
    ap = argparse.ArgumentParser(description="meeting60 44.1k 客观指标")
    ap.add_argument("--根", default=str(根), help="meeting60 根目录")
    ap.add_argument("--输出json", default=str(根 / "产物" / "指标" / "组指标.json"))
    ap.add_argument("--输出md", default=str(根 / "产物" / "测试组" / "测试组清单.md"))
    ap.add_argument("--无md", action="store_true")
    ap.add_argument("--单文件", default=None)
    args = ap.parse_args(argv)

    if args.单文件:
        m = 计算指标(Path(args.单文件))
        for k, v in m.items():
            print(f"{k}: {v}")
        return 0

    结果 = 扫描(Path(args.根))
    if not 结果["组"]:
        print(f"[提示] 未找到任何 组*/ 目录：{Path(args.根) / '产物' / '测试组'}")
    对比 = 对比表(结果)
    目标 = 参考目标(结果)
    接近 = 参考接近度(结果, 对比, 目标)
    排名 = 计算排名(对比)
    载荷 = _可json({
        "工具": "meeting60/实施/矩阵_指标.py",
        "口径": {
            "时域": "原采样率：RMS/峰值/削波/时长",
            "谱与F0": "resample_poly→44.1kHz 分析（16k 源上采样不新增高频）",
            "带占比分母": "0…Nyquist 全带功率",
            "削波": "|x|>0.99 采样点占比",
            "峰值因数": "峰值/RMS(dB)，偏低=压限/饱和（失真代理）",
            "F0": "40ms/20ms 帧，归一化自相关 70–500Hz，rmax≥0.35，能量门=1%最大帧；中位",
            "谱平坦度": "0–8k 共带主值；另给 0–16k",
            "诚实边界": "客观指标≠听感；未做真人盲评/MOS；参考原声仅本机 A/B，不再分发",
        },
        "组": 结果["组"],
        "对比组A": 对比,
        "改善排行": 排名,
        "参考目标": 目标,
        "参考接近度": 接近,
    })
    输出json = Path(args.输出json)
    输出json.parent.mkdir(parents=True, exist_ok=True)
    输出json.write_text(json.dumps(载荷, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {输出json}（{len(结果['组'])} 组 / {sum(d['文件数'] for d in 结果['组'].values())} 文件）")

    if not args.无md:
        输出md = Path(args.输出md)
        输出md.parent.mkdir(parents=True, exist_ok=True)
        输出md.write_text(生成清单md(结果, 对比, 排名, 接近, 目标), encoding="utf-8")
        print(f"[OK] {输出md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
